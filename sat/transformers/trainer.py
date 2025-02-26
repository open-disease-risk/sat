"""Training argument class that supports MPS devices."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch

from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader

from accelerate import Accelerator
from transformers import TrainingArguments


# Simple hack to work with MPS devices from: https://github.com/huggingface/transformers/issues/17971#issuecomment-1171579884
class TrainingArgumentsWithMPSSupport(TrainingArguments):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @property
    def device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model,
        args,
        train_dataset,
        eval_dataset,
        data_collator,
        metrics,
        callbacks,
    ):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.metrics = metrics
        self.callbacks = callbacks

        if self.args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

    def train(self):
        # Enable pinned memory for faster CPU->GPU transfer
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
            pin_memory=torch.cuda.is_available(),  # Use pinned memory if CUDA available
            num_workers=4,  # Parallel data loading
            prefetch_factor=2,  # Prefetch batches
        )
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),  # Use pinned memory if CUDA available
            num_workers=4,  # Parallel data loading
        )

        # Setup mixed precision with AMP
        from torch.cuda.amp import autocast, GradScaler

        scaler = GradScaler(enabled=self.args.fp16)

        adam_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }
        mixed_precision = "fp16" if self.args.fp16 else "no"

        accelerator = Accelerator(mixed_precision=mixed_precision)
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            self.model,
            AdamW(self.model.parameters(), lr=self.args.learning_rate, **adam_kwargs),
            train_dataloader,
            eval_dataloader,
        )

        # Add learning rate scheduler
        from transformers import get_linear_schedule_with_warmup

        num_training_steps = len(train_dataloader) * self.args.num_train_epochs
        num_warmup_steps = int(0.1 * num_training_steps)  # 10% warmup
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        model.train()
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader, start=1):
                # Use AMP autocast for mixed precision training
                with autocast(enabled=self.args.fp16):
                    loss = model(**batch).loss

                # More efficient gradient accumulation
                if step % self.args.gradient_accumulation_steps != 0:
                    # Use no_sync for more efficient multi-GPU training
                    # Only synchronize gradients when we're going to update
                    if hasattr(model, "no_sync") and accelerator.num_processes > 1:
                        with model.no_sync():
                            scaler.scale(
                                loss / self.args.gradient_accumulation_steps
                            ).backward()
                    else:
                        scaler.scale(
                            loss / self.args.gradient_accumulation_steps
                        ).backward()
                else:
                    scaler.scale(
                        loss / self.args.gradient_accumulation_steps
                    ).backward()
                    # Unscale before optimizer step to check for infs/NaNs
                    scaler.unscale_(optimizer)
                    # Apply gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    # Update with scaler for mixed precision
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()  # Update learning rate
                    optimizer.zero_grad(
                        set_to_none=True
                    )  # More efficient than zero_grad()

                # Evaluate periodically
                if step % self.args.eval_steps == 0:
                    self.evaluate(eval_dataloader)

    def evaluate(self, data_loader):
        self.model.eval()  # Set the model to evaluation mode

        with torch.no_grad():
            eval_loss = 0.0
            for step, batch in enumerate(data_loader, start=1):
                outputs = self.model(**batch)
                eval_loss += outputs.loss

        self.model.train()

    def predict(self, dataset):
        pass

    def save_model(self):
        pass
