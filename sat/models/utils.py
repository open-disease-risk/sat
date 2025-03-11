"""Utilities for SAT models."""

__authors__ = ["Dominik Dahlem"]
__status__ = "Development"

import torch
from torch import nn

from sat.utils import logging

logger = logging.get_default_logger()


def _detect_problematic_modules(model):
    """Detect modules that are known to cause issues with torch.compile.
    
    Args:
        model: PyTorch model to inspect
        
    Returns:
        bool: True if problematic modules found, False otherwise
        list: List of problematic module names if any
    """
    problematic_modules = []
    
    # Check for specific module types or operations known to be problematic
    for name, module in model.named_modules():
        # Dropouts are often problematic on MPS
        if isinstance(module, torch.nn.Dropout):
            problematic_modules.append(f"{name} (Dropout)")
            
        # Check module string representation for known problematic operations
        module_str = str(type(module)).lower()
        for issue in ['softplus', 'index_put']:
            if issue in module_str:
                problematic_modules.append(f"{name} ({issue})")
                
    return len(problematic_modules) > 0, problematic_modules


def compile_model(model, use_compile=True, mode=None, fullgraph=False):
    """Apply torch.compile to a model with platform-specific configurations.
    
    Args:
        model: The PyTorch model to compile
        use_compile: Whether to compile the model (default: True)
        mode: Compilation mode, one of 'default', 'reduce-overhead', or 'max-autotune'
            If None, will use 'default' on CPU/CUDA and 'reduce-overhead' on MPS (Apple Silicon)
        fullgraph: Whether to use fullgraph mode for compilation
            Set to False for development (better debugging)
            Set to True for production (potentially better optimization)
    
    Returns:
        The compiled model if compilation is successful, otherwise the original model
    """
    # Ensure torch is imported within function scope to avoid UnboundLocalError
    import torch
    
    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not use_compile or not hasattr(torch, 'compile'):
        logger.info("Torch compile not used: feature not available or disabled")
        return model
    
    try:
        # Set appropriate backend and mode based on device
        is_mps = torch.backends.mps.is_available()
        
        if is_mps:  # Apple Silicon
            # Apple Silicon requires more careful handling
            has_issues, issue_modules = _detect_problematic_modules(model)
            
            if has_issues:
                logger.warning(
                    f"Detected modules that may be incompatible with MPS compilation: {issue_modules}. "
                    f"Will apply selective compilation strategy."
                )
                
                # Configure compilation for better MPS compatibility
                # For PyTorch 2.6.0, we use the official torch.compiler.disable API
                # to exclude problematic modules
                
                # Import the problematic module to disable it
                try:
                    # Try to import the module to get its object
                    import importlib.util
                    import sys
                    
                    # Specifically disable the log_softplus function in pycox utils
                    from sat.pycox.models.utils import log_softplus
                    torch.compiler.disable(log_softplus)
                    logger.info("Disabled compilation for log_softplus function")
                    
                except Exception as e:
                    logger.warning(f"Could not disable specific functions: {e}")
                    
                # Set general compilation options for better compatibility
                # These are generally safe across PyTorch versions
                if hasattr(torch, '_dynamo'):
                    # These may exist in some versions
                    if hasattr(torch._dynamo, 'config'):
                        torch._dynamo.config.suppress_errors = True
                        if hasattr(torch._dynamo.config, 'cache_size_limit'):
                            torch._dynamo.config.cache_size_limit = 64
            
            # Use most compatible settings for MPS
            backend = "aot_eager"
            if mode is None:
                mode = "reduce-overhead"
        else:  # CUDA or CPU
            backend = "inductor"
            if mode is None:
                mode = "default"
        
        logger.info(f"Compiling model with backend={backend}, mode={mode}, fullgraph={fullgraph}")
        
        # Apply compilation with specified settings
        compiled_model = torch.compile(
            model, 
            backend=backend,
            mode=mode,
            fullgraph=fullgraph,
            # Disable graph breaking on error for better reliability
            disable_on_error=False
        )
        
        logger.info("Model compilation successful")
        return compiled_model
    except Exception as e:
        logger.warning(f"Model compilation failed, using eager execution instead: {e}")
        return model


def get_device():
    device_str = "cpu"
    if torch.cuda.is_available():
        device_str = "cuda"
    elif torch.backends.mps.is_available():
        device_str = "mps"

    device = torch.device(device_str)
    return (device_str, device)


def load_model(resolved_archive_file, model):
    try:
        state_dict = torch.load(resolved_archive_file, map_location="cpu")
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
    for old_key, new_key in zip(old_keys, new_keys):
        state_dict[new_key] = state_dict.pop(old_key)

    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, prefix=""):
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
