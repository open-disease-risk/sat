"""Tests for the Trainer class."""

import os
import shutil
import sys
import unittest
from unittest.mock import MagicMock, patch, ANY

import numpy as np
import torch
from torch.utils.data import TensorDataset

# Import from the sat package
from sat.transformers.trainer import Trainer, TrainingArgumentsWithMPSSupport


class TestTrainer(unittest.TestCase):
    def setUp(self):
        # Create a test output directory
        self.test_output_dir = "./test-trainer-output"
        os.makedirs(self.test_output_dir, exist_ok=True)

        # Create a simple mock model
        self.mock_model = MagicMock()
        self.mock_model.train.return_value = None
        self.mock_model.eval.return_value = None

        # Mock model parameters for optimizer
        mock_param = torch.nn.Parameter(torch.randn(5, 5))
        self.mock_model.parameters.return_value = [mock_param]

        # Mock loss and output
        mock_output = MagicMock()
        mock_output.loss = torch.tensor(0.5)
        mock_output.logits = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
        self.mock_model.return_value = mock_output

        # Create simple datasets
        self.train_dataset = TensorDataset(torch.randn(10, 5), torch.randn(10, 2))
        self.eval_dataset = TensorDataset(torch.randn(5, 5), torch.randn(5, 2))
        self.predict_dataset = TensorDataset(torch.randn(3, 5), torch.randn(3, 2))

        # Mock collator
        self.mock_collator = lambda x: {
            "input_ids": torch.stack([i[0] for i in x]),
            "labels": torch.stack([i[1] for i in x]),
        }

        # Training arguments
        self.args = TrainingArgumentsWithMPSSupport(
            output_dir=self.test_output_dir,
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            num_train_epochs=1,
            learning_rate=1e-4,
            fp16=False,
            gradient_accumulation_steps=1,
            eval_steps=5,
            seed=42,
            logging_dir=None,
        )

        # Create trainer
        self.trainer = Trainer(
            model=self.mock_model,
            args=self.args,
            train_dataset=self.train_dataset,
            eval_dataset=self.eval_dataset,
            data_collator=self.mock_collator,
            metrics={},
            callbacks=[],
        )

    def tearDown(self):
        # Clean up test output directory
        if os.path.exists(self.test_output_dir):
            shutil.rmtree(self.test_output_dir)

    @patch("sat.transformers.trainer.transformers.get_linear_schedule_with_warmup")
    @patch("sat.transformers.trainer.set_seed")
    def test_train(self, mock_set_seed, mock_scheduler_fn):
        """Test the train method."""
        # Replace the accelerator instance with a mock
        original_accelerator = self.trainer.accelerator
        self.trainer.accelerator = MagicMock()

        # Mock scheduler
        mock_scheduler = MagicMock()
        mock_scheduler_fn.return_value = mock_scheduler

        # Setup accumulate context manager
        mock_context = MagicMock()
        self.trainer.accelerator.accumulate.return_value = mock_context
        mock_context.__enter__.return_value = None
        mock_context.__exit__.return_value = None

        # Setup sync_gradients property
        self.trainer.accelerator.sync_gradients = True

        # Configure the prepare method
        mock_prepared_model = self.mock_model
        mock_optimizer = MagicMock()
        mock_train_loader = [
            {"input_ids": torch.randn(2, 5), "labels": torch.randn(2, 2)}
            for _ in range(2)
        ]
        mock_eval_loader = [
            {"input_ids": torch.randn(2, 5), "labels": torch.randn(2, 2)}
            for _ in range(1)
        ]

        self.trainer.accelerator.prepare.return_value = (
            mock_prepared_model,
            mock_optimizer,
            mock_train_loader,
            mock_eval_loader,
        )

        # Setup progress bar
        mock_progress_bar = MagicMock()
        self.trainer.accelerator.get_progress_bar.return_value = mock_progress_bar
        mock_progress_bar.__iter__.return_value = mock_train_loader

        try:
            # Run training
            self.trainer.train()

            # Assertions
            self.assertTrue(
                self.mock_model.train.call_count >= 1, "train() was not called"
            )
            self.trainer.accelerator.prepare.assert_called_once()
            mock_scheduler_fn.assert_called_once()
            self.trainer.accelerator.backward.assert_called()
            mock_optimizer.step.assert_called()
            mock_scheduler.step.assert_called()
            mock_optimizer.zero_grad.assert_called_with(set_to_none=True)
        finally:
            # Restore the original accelerator
            self.trainer.accelerator = original_accelerator

    def test_evaluate(self):
        """Test the evaluate method."""
        # Replace the accelerator instance with a mock
        original_accelerator = self.trainer.accelerator
        self.trainer.accelerator = MagicMock()

        # Setup gathered tensor
        mock_gathered_tensor = torch.tensor([0.5])
        self.trainer.accelerator.gather_for_metrics.return_value = mock_gathered_tensor

        # Setup progress bar
        mock_eval_dataloader = [
            {"input_ids": torch.randn(2, 5), "labels": torch.randn(2, 2)}
            for _ in range(2)
        ]
        mock_progress_bar = MagicMock()
        self.trainer.accelerator.get_progress_bar.return_value = mock_progress_bar
        mock_progress_bar.__iter__.return_value = mock_eval_dataloader

        # Mock the find_batch_size function
        try:
            with patch("sat.transformers.trainer.find_batch_size", return_value=2):
                # Run evaluation
                result = self.trainer.evaluate(mock_eval_dataloader)

                # Assertions
                self.mock_model.eval.assert_called_once()
                self.mock_model.train.assert_called_once()
                self.trainer.accelerator.gather_for_metrics.assert_called()
                self.trainer.accelerator.log.assert_called_once()
                self.assertIsInstance(result, float)

                # Reset model mocks for the next test
                self.mock_model.eval.reset_mock()
                self.mock_model.train.reset_mock()
        finally:
            # Restore the original accelerator
            self.trainer.accelerator = original_accelerator

    def test_predict(self):
        """Test the predict method."""
        # Replace the accelerator instance with a mock
        original_accelerator = self.trainer.accelerator
        self.trainer.accelerator = MagicMock()

        # Setup progress bar
        mock_predict_dataloader = MagicMock()
        mock_progress_bar = MagicMock()
        self.trainer.accelerator.get_progress_bar.return_value = mock_progress_bar
        mock_batches = [
            {"input_ids": torch.randn(2, 5), "labels": torch.randn(2, 2)}
            for _ in range(2)
        ]
        mock_progress_bar.__iter__.return_value = mock_batches

        # Mock the prepare method
        self.trainer.accelerator.prepare.return_value = mock_predict_dataloader

        # Mock logits for prediction
        mock_logits = torch.randn(2, 2)
        mock_gathered_tensor = mock_logits  # Simulate the gather operation
        self.trainer.accelerator.gather_for_metrics.return_value = mock_gathered_tensor

        # Run prediction
        try:
            with patch(
                "numpy.concatenate", return_value=np.array([[0.1, 0.2], [0.3, 0.4]])
            ):
                predictions = self.trainer.predict(self.predict_dataset)

                # Assertions
                self.mock_model.eval.assert_called_once()
                self.trainer.accelerator.gather_for_metrics.assert_called()
                self.assertIsInstance(predictions, np.ndarray)
                self.assertEqual(
                    predictions.shape, (2, 2)
                )  # Should match our mocked return
        finally:
            # Restore the original accelerator
            self.trainer.accelerator = original_accelerator

    def test_save_model(self):
        """Test the save_model method."""
        # Replace the accelerator instance with a mock
        original_accelerator = self.trainer.accelerator
        self.trainer.accelerator = MagicMock()

        # Mock unwrap_model
        mock_unwrapped = MagicMock()
        self.trainer.accelerator.unwrap_model.return_value = mock_unwrapped
        mock_unwrapped.state_dict.return_value = {"layer1.weight": torch.randn(5, 5)}

        # Add config to unwrapped model
        mock_unwrapped.config = MagicMock()
        mock_unwrapped.config.save_pretrained = MagicMock()

        # Run save_model
        try:
            output_dir = self.trainer.save_model()

            # Assertions
            self.trainer.accelerator.unwrap_model.assert_called_once_with(
                self.mock_model
            )
            self.trainer.accelerator.save.assert_called_once_with(
                mock_unwrapped.state_dict(), f"{self.args.output_dir}/model.pt"
            )
            self.trainer.accelerator.save_state.assert_called_once_with(
                self.args.output_dir
            )
            mock_unwrapped.config.save_pretrained.assert_called_once_with(
                self.args.output_dir
            )
            self.assertEqual(output_dir, self.args.output_dir)
        finally:
            # Restore the original accelerator
            self.trainer.accelerator = original_accelerator

    def test_training_arguments_mps_support(self):
        """Test the TrainingArgumentsWithMPSSupport class."""
        args = TrainingArgumentsWithMPSSupport(
            output_dir="./test-args",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
        )

        with patch("torch.cuda.is_available", return_value=True):
            self.assertEqual(args.device, torch.device("cuda"))

        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=True):
                self.assertEqual(args.device, torch.device("mps"))

            with patch("torch.backends.mps.is_available", return_value=False):
                self.assertEqual(args.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
