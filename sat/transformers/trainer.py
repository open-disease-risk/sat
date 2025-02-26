"""Training argument class that supports MPS devices."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data.dataloader import DataLoader

from accelerate import Accelerator
from accelerate.utils import find_batch_size, set_seed
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

        # Initialize accelerator
        mixed_precision = "fp16" if self.args.fp16 else "no"
        self.accelerator = Accelerator(
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            project_dir=self.args.output_dir,
            log_with="tensorboard" if self.args.logging_dir else None,
        )

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
        # Set seed for reproducibility
        set_seed(self.args.seed)

        # Prepare model, optimizer, and dataloaders with accelerate
        model, optimizer, train_dataloader, eval_dataloader = self.accelerator.prepare(
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
            # Use accelerator's built-in progress tracking
            progress_bar = self.accelerator.get_progress_bar(train_dataloader)
            for step, batch in enumerate(progress_bar):
                # Handle accumulation with accelerator
                with self.accelerator.accumulate(model):
                    # Forward pass
                    outputs = model(**batch)
                    loss = outputs.loss

                    # Backward pass with accelerator handling gradient scaling
                    self.accelerator.backward(loss)

                    # Apply gradient clipping if accumulation step is complete
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(
                            model.parameters(), max_norm=1.0
                        )

                        # Step optimizer and scheduler
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad(set_to_none=True)

                    # Update progress bar with loss info
                    progress_bar.set_postfix(loss=loss.item())

                # Evaluate periodically
                if step % self.args.eval_steps == 0:
                    self.evaluate(eval_dataloader)

    def evaluate(self, data_loader):
        self.model.eval()  # Set the model to evaluation mode

        # Track eval metrics
        total_loss = 0
        num_samples = 0

        # Use accelerator's progress tracking
        eval_progress_bar = self.accelerator.get_progress_bar(data_loader)

        with torch.no_grad():
            for step, batch in enumerate(eval_progress_bar):
                # Get batch size for proper averaging
                batch_size = find_batch_size(batch)
                num_samples += batch_size

                # Forward pass
                outputs = self.model(**batch)
                loss = outputs.loss

                # Gather loss from all processes if distributed
                loss = self.accelerator.gather_for_metrics(loss).mean().item()
                total_loss += loss * batch_size

                # Update progress bar
                eval_progress_bar.set_postfix(eval_loss=loss)

        # Calculate average loss
        avg_loss = total_loss / num_samples if num_samples > 0 else 0

        # Log metrics
        self.accelerator.log({"eval/loss": avg_loss})

        self.model.train()

        return avg_loss

    def predict(self, dataset):
        """Run inference on a dataset.

        Args:
            dataset: The dataset to predict on

        Returns:
            List of model predictions
        """
        predict_dataloader = DataLoader(
            dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
        )

        # Prepare dataloader with accelerator
        predict_dataloader = self.accelerator.prepare(predict_dataloader)

        # Set model to evaluation mode
        self.model.eval()

        all_preds = []

        # Create progress bar
        predict_progress_bar = self.accelerator.get_progress_bar(predict_dataloader)

        with torch.no_grad():
            for batch in predict_progress_bar:
                # Get model outputs
                outputs = self.model(**batch)

                # Get predictions based on model output type
                if hasattr(outputs, "logits"):
                    preds = outputs.logits
                else:
                    preds = outputs

                # Gather predictions from all processes if distributed
                preds = self.accelerator.gather_for_metrics(preds)
                all_preds.append(preds.cpu().numpy())

        # Concatenate all predictions
        all_preds = np.concatenate(all_preds, axis=0)

        return all_preds

    def save_model(self):
        """Save model checkpoint using accelerator."""
        unwrapped_model = self.accelerator.unwrap_model(self.model)

        # Save unwrapped model
        self.accelerator.save(
            unwrapped_model.state_dict(), f"{self.args.output_dir}/model.pt"
        )

        # Save the full training state including optimizer and scheduler
        self.accelerator.save_state(self.args.output_dir)

        # Save model config and tokenizer configuration if available
        if hasattr(unwrapped_model, "config"):
            unwrapped_model.config.save_pretrained(self.args.output_dir)

        return self.args.output_dir
