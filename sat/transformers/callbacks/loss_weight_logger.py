"""Callback for logging loss weights during training."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

from transformers.integrations import TensorBoardCallback

from sat.utils import logging

logger = logging.get_default_logger()


class LossWeightLoggerCallback(TensorBoardCallback):
    """
    Callback for logging loss weights during training.

    This callback checks if the model's loss function has a method get_loss_weights()
    and logs the weights at regular intervals and during evaluation.
    It's useful for tracking how adaptive loss weighting schemes change during training.
    """

    def __init__(self):
        super().__init__()

    def on_train_begin(self, args, state, control, **kwargs):
        # Call the parent implementation to ensure proper initialization
        super().on_train_begin(args, state, control, **kwargs)
        # Now self.tb_writer is available for use

    def _log_weights(self, model, prefix, state=None, args=None):
        """Log weights if the model's loss function has get_loss_weights()."""
        # Handle different model structures
        if hasattr(model, "loss_fn") and hasattr(model.loss_fn, "get_loss_weights"):
            # Standard model with loss_fn directly attached
            loss_obj = model.loss_fn
            logger.debug(f"Found loss_fn on model: {loss_obj}")
        elif hasattr(model, "loss") and hasattr(model.loss, "get_loss_weights"):
            # Model with loss directly attached (like SurvivalTaskHead)
            loss_obj = model.loss
            logger.debug(f"Found loss on model: {loss_obj}")
        elif hasattr(model, "heads") and len(getattr(model, "heads", [])) > 0:
            # MTLForSurvival model with multiple task heads
            logger.debug(f"Model has {len(model.heads)} task heads")
            # Check if any head has a loss with get_loss_weights
            metrics = {}
            for i, head in enumerate(model.heads):
                if hasattr(head, "loss") and hasattr(head.loss, "get_loss_weights"):
                    try:
                        weights = head.loss.get_loss_weights()
                        if weights is not None:
                            for j, w in enumerate(weights):
                                metrics[f"{prefix}/head_{i}_weight_{j}"] = (
                                    w.item() if hasattr(w, "item") else w
                                )
                            logger.debug(f"Logged weights for head {i}: {weights}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to log loss weights for head {i}: {str(e)}"
                        )
                else:
                    logger.debug(f"Head {i} has no suitable loss function")

            if metrics:
                if self.tb_writer is not None:
                    for k, v in metrics.items():
                        self.tb_writer.add_scalar(k, v, state.global_step)
            return
        else:
            logger.debug("No suitable loss function found on model")
            return

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        """Log weights during evaluation."""
        # Call parent implementation to log standard metrics
        super().on_evaluate(args, state, control, **kwargs)

        # Add your custom evaluation metrics
        if self.tb_writer is not None:
            self._log_weights(model, "eval", state, args)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        """Log weights at regular intervals during training."""
        # Only log on specified frequency and on main process
        if state.global_step % args.logging_steps == 0:
            self._log_weights(model, "train", state, args)
