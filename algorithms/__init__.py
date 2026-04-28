# Circle fitting algorithms
#
# Import individual modules as needed:
#   from algorithms.CCC_FBI import ccc_fbi
#   from algorithms.CIBICA import CIBICA
#   from algorithms.preprocessing import preprocess_image
#
# Or import all at once (requires all dependencies installed):
#   from algorithms import *

__all__ = [
    'ccc_fbi', 'CIBICA', 'cibica', 'vectorized_XYR', 'median_3d', 'LS_circle',
    'greco_2022', 'guo_2019', 'HOUGH', 'nurunnabi', 'qi_2024', 'rcd', 'rfca', 'rht',
    'preprocess_image', 'get_preprocessing_configs',
    'preprocess_green_level', 'preprocess_median_filter',
]


def __getattr__(name):
    """Lazy import to avoid loading all modules on package import."""
    if name == 'ccc_fbi':
        from .CCC_FBI import ccc_fbi
        return ccc_fbi
    elif name in ('CIBICA', 'cibica', 'vectorized_XYR', 'median_3d', 'LS_circle'):
        import importlib
        mod = importlib.import_module('.CIBICA', package=__name__)
        return getattr(mod, name)
    elif name == 'greco_2022':
        from .GRECO import greco_2022
        return greco_2022
    elif name == 'guo_2019':
        from .GUO import guo_2019
        return guo_2019
    elif name == 'HOUGH':
        from .HOUGH import HOUGH
        return HOUGH
    elif name == 'nurunnabi':
        from .NURUNNABI import nurunnabi
        return nurunnabi
    elif name == 'qi_2024':
        from .QI import qi_2024
        return qi_2024
    elif name == 'rcd':
        from .RCD import rcd
        return rcd
    elif name == 'rfca':
        from .RFCA import rfca
        return rfca
    elif name == 'rht':
        from .RHT import rht
        return rht
    elif name in ('preprocess_image', 'get_preprocessing_configs',
                  'preprocess_green_level', 'preprocess_median_filter'):
        from . import preprocessing as mod
        return getattr(mod, name)
    raise AttributeError(f"module 'algorithms' has no attribute '{name}'")
