"""Utilities for SAT models."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
from torch import nn

from sat.utils import logging

logger = logging.get_default_logger()


# Cache device detection result for efficiency
_CACHED_DEVICE = None


def get_device():
    global _CACHED_DEVICE
    if _CACHED_DEVICE is not None:
        return _CACHED_DEVICE

    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"

    device = torch.device(device_str)
    _CACHED_DEVICE = (device_str, device)

    if logger.isEnabledFor(logging.DEBUG):
        logger.debug(f"Using device: {device_str}")

    return _CACHED_DEVICE


def load_model(resolved_archive_file, model):
    # Get target device for loading
    _, device = get_device()

    try:
        # Load directly to the target device for efficiency
        state_dict = torch.load(resolved_archive_file, map_location=device)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Loaded model weights to {device}")
    except Exception:
        raise OSError(
            "Unable to load weights from pytorch checkpoint file. "
            "If you tried to load a PyTorch model from a TF 2.0 checkpoint, please set from_tf=True. "
        )

    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # Convert old format to new format if needed from a PyTorch state_dict
    old_keys = []
    new_keys = []
    for key in state_dict.keys():
        new_key = None
        if "gamma" in key:
            new_key = key.replace("gamma", "weight")
        if "beta" in key:
            new_key = key.replace("beta", "bias")
        if new_key:
            old_keys.append(key)
            new_keys.append(new_key)
    for old_key, new_key in zip(old_keys, new_keys, strict=False):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, prefix=""):
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"load state for module {module} with prefix {prefix}")

        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            True,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + ".")

    # Make sure we are able to load base models as well as derived models (with heads)
    start_prefix = ""
    model_to_load = model

    load(model_to_load, prefix=start_prefix)

    if model.__class__.__name__ != model_to_load.__class__.__name__:
        base_model_state_dict = model_to_load.state_dict().keys()
        logging.debug(f"base model dicts {base_model_state_dict}")
        head_model_state_dict_without_base_prefix = [
            # key.split(cls.base_model_prefix + ".")[-1]
            # for key in model.state_dict().keys()
        ]

        missing_keys.extend(
            head_model_state_dict_without_base_prefix - base_model_state_dict
        )

    if len(missing_keys) > 0:
        logger.info(
            "Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys
            )
        )
    if len(unexpected_keys) > 0:
        logger.info(
            "Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys
            )
        )
    if len(error_msgs) > 0:
        raise RuntimeError(
            "Error(s) in loading state_dict for {}:\n\t{}".format(
                model.__class__.__name__, "\n\t".join(error_msgs)
            )
        )
    model.tie_weights()  # make sure token embedding weights are still tied if needed

    # Set model in evaluation mode to deactivate DropOut modules by default
    model.eval()

    return model
