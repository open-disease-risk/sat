#!/usr/bin/env python
"""
Helper script to fix the "cannot import name 'trapz' from 'scipy.integrate'" error in lifelines.

This script monkey-patches the lifelines package to use scipy's trapezoid function instead of trapz,
which has been moved in newer scipy versions.
"""

import importlib
import inspect
import sys
import warnings


def apply_lifelines_patch():
    """
    Apply a monkey patch to lifelines to fix the scipy.integrate.trapz import error.
    """
    try:
        # Try to import lifelines
        import lifelines
        from scipy import integrate  # Need to check if trapz exists

        # Check if trapz is directly available in scipy.integrate
        if not hasattr(integrate, "trapz"):
            print(
                "Applying patch for lifelines: scipy.integrate.trapz is not available"
            )

            # Import trapezoid function from scipy
            from scipy import trapezoid

            # Monkey patch the trapz function in lifelines modules that use it
            # Get the path to lifelines fitters for reference (not used but kept for debugging)
            # fitters_init_path = Path(inspect.getfile(lifelines.fitters))

            # List of lifelines modules that might use trapz
            modules_to_patch = [
                "lifelines.fitters",
                "lifelines.fitters.kaplan_meier_fitter",
                "lifelines.fitters.nelson_aalen_fitter",
                "lifelines.utils",
            ]

            for module_name in modules_to_patch:
                try:
                    module = importlib.import_module(module_name)
                    if hasattr(module, "trapz"):
                        # Module already has trapz defined, no need to patch
                        continue

                    # Add trapezoid function as trapz
                    module.trapz = trapezoid

                    # If the module imports trapz directly
                    module_code = inspect.getsource(module)
                    if "from scipy.integrate import trapz" in module_code:
                        print(f"Patched {module_name}")
                except (ImportError, AttributeError) as e:
                    print(f"Could not patch {module_name}: {e}")

            # Specifically patch scipy.integrate
            if not hasattr(integrate, "trapz"):
                integrate.trapz = trapezoid
                print("Patched scipy.integrate.trapz")

            print("Lifelines patch applied successfully")
            return True
    except ImportError as e:
        warnings.warn(f"Could not patch lifelines: {e}", stacklevel=2)
        return False


if __name__ == "__main__":
    success = apply_lifelines_patch()
    sys.exit(0 if success else 1)
