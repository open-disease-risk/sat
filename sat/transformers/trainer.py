""" Training argument class that supports MPS devices.
"""

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
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            collate_fn=self.data_collator,
            shuffle=True,
        )
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            shuffle=False,
        )

        adam_kwargs = {
            "betas": (self.args.adam_beta1, self.args.adam_beta2),
            "eps": self.args.adam_epsilon,
        }
        mixed_precision = "fp16" if self.args.fp16 else "no"

        accelerator = Accelerator(mixed_precision=mixed_precision)
        model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
            self.model,
            AdamW(self.model.parameters(), lr=self.args.learning_rate),
            train_dataloader,
            eval_dataloader,
        )

        model.train()
        for epoch in range(self.args.num_train_epochs):
            for step, batch in enumerate(train_dataloader, start=1):
                loss = model(**batch).loss
                loss = loss / self.args.gradient_accumulation_steps
                accelerator.backward(loss)

                if step % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

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
