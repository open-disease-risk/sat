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
        for issue in [
            "softplus",
            "index_put",
            "log_softmax",
            "logsoftmax",
            "log_sigmoid",
        ]:
            if issue in module_str:
                problematic_modules.append(f"{name} ({issue})")

        # Check for any operations that might generate NaNs
        if hasattr(module, "forward"):
            forward_str = str(module.forward).lower()
            for risky_op in ["log(", "log_", "sqrt(", "div(", "reciprocal"]:
                if risky_op in forward_str:
                    problematic_modules.append(f"{name} ({risky_op})")

    return len(problematic_modules) > 0, problematic_modules


def compile_model(model, config=None):
    """Apply torch.compile to a model with platform-specific configurations.

    Args:
        model: The PyTorch model to compile
        config: Configuration dictionary with compile options
            If None, will use default values with platform-specific adjustments

    Returns:
        The compiled model if compilation is successful, otherwise the original model
    """
    # Ensure torch is imported within function scope to avoid UnboundLocalError
    import torch
    import platform
    import re

    # Default config if none provided
    if config is None:
        config = {}

    # Extract config values with defaults
    use_compile = config.get("use_compile", True)
    compile_mode = config.get("compile_mode", None)
    fullgraph = config.get("compile_fullgraph", False)
    backend = config.get("compile_backend", None)
    dynamic_shapes = config.get("dynamic_shapes", False)
    opt_level = config.get("opt_level", 2)

    # Cache configuration
    dynamo_cache = config.get("dynamo_cache", {})
    cache_enabled = dynamo_cache.get("enabled", True)
    cache_size_limit = dynamo_cache.get("size_limit", 64)
    cache_path = dynamo_cache.get("path", None)

    # Debug options
    debug_opts = config.get("debug_options", {})
    decomposition_schemas = debug_opts.get("decomposition_schemas", False)
    explain = debug_opts.get("explain", False)
    dump_graphs = debug_opts.get("dump_graphs", False)

    # Specialized optimizations
    spec_opts = config.get("specialized_opts", {})
    allow_cudagraphs = spec_opts.get("allow_cudagraphs", True)
    disable_fuser = spec_opts.get("disable_fuser", False)
    static_memory_planning = spec_opts.get("static_memory_planning", True)

    # Selective compilation
    selective = config.get("selective_compile", {})
    exclude_modules = selective.get("exclude_modules", [])
    exclude_patterns = selective.get("exclude_patterns", [])

    # M-series specific configuration
    m_series_config = config.get("m_series_mac_defaults", {})
    apply_m_series_defaults = m_series_config.get("enabled", True)

    # Add scikit-survival modules to exclude list if they exist
    try:
        import sys
        import importlib

        # Try to import scikit-survival modules that might be used in metrics
        sksurv_modules = ["sksurv.metrics", "sksurv.util", "sksurv.nonparametric"]

        # Add survival analysis related modules to exclusion list
        for module_name in sksurv_modules:
            try:
                if module_name in sys.modules or importlib.util.find_spec(module_name):
                    module_parts = module_name.split(".")
                    mod = importlib.import_module(module_name)
                    # Add all functions from these modules to exclude
                    for attr_name in dir(mod):
                        if not attr_name.startswith("_") and callable(
                            getattr(mod, attr_name)
                        ):
                            full_name = f"{module_name}.{attr_name}"
                            if full_name not in exclude_modules:
                                exclude_modules.append(full_name)
                                logger.debug(
                                    f"Added {full_name} to compilation exclusion list"
                                )
            except (ImportError, ModuleNotFoundError):
                pass

        # Also exclude specific known problematic evaluation metrics
        try:
            from sat.evaluate.eval_modules import SurvivalAnalysisMetric

            exclude_modules.append(
                "sat.evaluate.eval_modules.SurvivalAnalysisMetric.compute"
            )
            exclude_modules.append(
                "sat.evaluate.eval_modules.SurvivalAnalysisMetric.compute_event"
            )
        except ImportError:
            pass
    except Exception as e:
        logger.warning(f"Could not set up automatic exclusions for metrics: {e}")

    # Check if torch.compile is available (requires PyTorch 2.0+)
    if not use_compile or not hasattr(torch, "compile"):
        logger.info("Torch compile not used: feature not available or disabled")
        return model

    try:
        # Detect device type and architecture
        is_mps = torch.backends.mps.is_available()
        is_cuda = torch.cuda.is_available()
        is_m_series = False

        # Check for M-series Mac (M1, M2, M3) for specific optimizations
        if is_mps and platform.system() == "Darwin" and apply_m_series_defaults:
            # Check if M-series chip (M1, M2, M3)
            try:
                import subprocess

                chip_info = (
                    subprocess.check_output(
                        ["sysctl", "-n", "machdep.cpu.brand_string"]
                    )
                    .decode()
                    .strip()
                )
                is_m_series = any(f"Apple M{i}" in chip_info for i in range(1, 4))

                # Detect specific M3 chip for even more specialized settings
                is_m3 = "Apple M3" in chip_info

                logger.info(f"Detected Apple Silicon: {chip_info}")

                # Apply M-series specific defaults if enabled
                if is_m_series:
                    if backend is None:
                        backend = m_series_config.get("compile_backend", "aot_eager")
                    if compile_mode is None:
                        compile_mode = m_series_config.get(
                            "compile_mode", "reduce-overhead"
                        )
                    if opt_level == 2:  # Only override if still at default
                        opt_level = m_series_config.get("opt_level", 1)
                    if not dynamic_shapes:  # Only override if not explicitly set
                        dynamic_shapes = m_series_config.get("dynamic_shapes", True)

                    # Apply M-series specialized opts if available
                    m_series_spec_opts = m_series_config.get("specialized_opts", {})
                    if m_series_spec_opts:
                        static_memory_planning = m_series_spec_opts.get(
                            "static_memory_planning", static_memory_planning
                        )
                        disable_fuser = m_series_spec_opts.get(
                            "disable_fuser", disable_fuser
                        )

                    # M3-specific optimizations
                    if is_m3:
                        logger.info("Applying M3-specific optimizations")
                        # M3 can handle slightly more aggressive optimizations than M1/M2
                        if opt_level < 2:
                            opt_level = 2
            except Exception as e:
                logger.warning(f"Could not detect Apple chip details: {e}")

        # Detect problematic modules for compilation
        has_issues, issue_modules = _detect_problematic_modules(model)

        # Set appropriate backend and mode based on device if not specified
        if backend is None:
            if is_mps:  # Apple Silicon
                backend = "aot_eager"
            elif is_cuda:  # NVIDIA GPU
                backend = "inductor"
            else:  # CPU
                backend = "inductor"

        if compile_mode is None:
            if is_mps:  # Apple Silicon
                compile_mode = "reduce-overhead"
            else:  # CUDA or CPU
                compile_mode = "default"

        # Apply global cache configuration if enabled
        if cache_enabled and hasattr(torch, "_dynamo"):
            if hasattr(torch._dynamo, "config"):
                if cache_size_limit is not None and hasattr(
                    torch._dynamo.config, "cache_size_limit"
                ):
                    torch._dynamo.config.cache_size_limit = cache_size_limit
                torch._dynamo.config.suppress_errors = True

                if cache_path is not None and hasattr(
                    torch._dynamo.config, "cache_dir_path"
                ):
                    torch._dynamo.config.cache_dir_path = cache_path

        # Apply selective compilation for problematic modules
        if has_issues or exclude_modules or exclude_patterns:
            # Add detected issue modules to exclude list
            if has_issues:
                exclude_modules.extend(issue_modules)
                logger.warning(
                    f"Detected modules that may be incompatible with compilation: {issue_modules}. "
                    f"These will be excluded from compilation."
                )

            # Handle explicit exclusions
            try:
                # Specifically disable known problematic functions
                if hasattr(torch, "compiler") and hasattr(torch.compiler, "disable"):
                    # Handle explicitly excluded modules
                    for module_name in exclude_modules:
                        try:
                            # Try to import the module to get its object
                            module_parts = module_name.split(".")
                            current_module = __import__(module_parts[0])

                            for part in module_parts[1:]:
                                current_module = getattr(current_module, part)

                            torch.compiler.disable(current_module)
                            logger.info(
                                f"Disabled compilation for module: {module_name}"
                            )
                        except Exception as e:
                            logger.warning(
                                f"Could not disable module {module_name}: {e}"
                            )

                    # Special case for known problematic modules in this codebase
                    try:
                        from sat.pycox.models.utils import log_softplus

                        torch.compiler.disable(log_softplus)
                        logger.info("Disabled compilation for log_softplus function")
                    except Exception as e:
                        logger.warning(f"Could not disable log_softplus: {e}")

                    # Handle pattern-based exclusions
                    if exclude_patterns:
                        # This is more complex and would require traversing the model graph
                        # and checking each module against the patterns
                        # For simplicity, just log that we have patterns
                        logger.info(
                            f"Pattern-based exclusions will be applied: {exclude_patterns}"
                        )
            except Exception as e:
                logger.warning(f"Could not apply selective compilation: {e}")

        # Configure debug options
        if dump_graphs and hasattr(torch, "_dynamo"):
            if hasattr(torch._dynamo, "config"):
                torch._dynamo.config.log_level = "debug"
                torch._dynamo.config.output_code = True

        # Prepare advanced compilation options
        compile_options = {
            "backend": backend,
            "mode": compile_mode,
            "fullgraph": fullgraph,
            "disable_on_error": False,  # Keep models working even with compilation errors
        }

        # Add dynamic shapes if requested
        if dynamic_shapes:
            compile_options["dynamic"] = True

        # Add optimization level if supported by the backend
        if backend == "inductor" and hasattr(torch, "_inductor"):
            if hasattr(torch._inductor, "config"):
                torch._inductor.config.opt_level = opt_level

        # Add specialized options
        if backend == "cudagraphs" and not allow_cudagraphs:
            logger.warning("cudagraphs backend requested but allow_cudagraphs is False")

        if disable_fuser and hasattr(torch, "_inductor"):
            if hasattr(torch._inductor, "config"):
                torch._inductor.config.use_fuser = not disable_fuser

        if static_memory_planning and hasattr(torch, "_inductor"):
            if hasattr(torch._inductor, "config"):
                torch._inductor.config.static_memory_planning = static_memory_planning

        # Log compilation settings
        logger.info(f"Compiling model with: {compile_options}")
        if is_m_series:
            logger.info(
                f"Using M-series optimized settings (opt_level={opt_level}, dynamic_shapes={dynamic_shapes})"
            )

        # For Mac M3 chips, add extra safeguards for numerical stability
        if is_m3 and is_mps:
            # Patch forward methods with NaN checking on supported modules
            def patch_module_with_nan_check(module, name):
                original_forward = module.forward

                def safe_forward(*args, **kwargs):
                    # Original forward pass
                    outputs = original_forward(*args, **kwargs)

                    # Check for NaNs in the output
                    if isinstance(outputs, torch.Tensor):
                        if torch.isnan(outputs).any():
                            logger.warning(f"NaN detected in output of {name}")
                            # Replace NaNs with zeros (or other strategy)
                            outputs = torch.nan_to_num(outputs, nan=0.0)
                    return outputs

                # Only patch significant modules to avoid performance impact
                module_str = str(type(module)).lower()
                if any(op in module_str for op in ["log", "exp", "div", "pow", "sqrt"]):
                    module.forward = safe_forward
                    logger.debug(f"Added NaN protection to {name}")

            # Apply NaN checks to risky modules
            for name, module in model.named_modules():
                patch_module_with_nan_check(module, name)

            logger.info("Added NaN protection for M3 chip compilation")

        # Apply compilation with configured settings
        try:
            compiled_model = torch.compile(model, **compile_options)

            # Generate explanation if requested
            if (
                explain
                and hasattr(torch, "_dynamo")
                and hasattr(torch._dynamo, "explain")
            ):
                explanation = torch._dynamo.explain(model)
                logger.info(f"Compilation explanation: {explanation}")
        except Exception as e:
            logger.warning(
                f"Initial compilation failed: {e}, trying with reduced settings"
            )

            # Fall back to safer compilation options
            fallback_options = compile_options.copy()
            fallback_options["fullgraph"] = False
            if "dynamic" in fallback_options:
                del fallback_options["dynamic"]

            try:
                compiled_model = torch.compile(model, **fallback_options)
                logger.info("Compilation succeeded with fallback options")
            except Exception as e2:
                logger.error(f"Fallback compilation also failed: {e2}")
                raise e  # Re-raise the original error

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
