# auxiliar_analysis/__init__.py
# Keep imports lightweight to avoid optional dependencies during analysis notebooks.
from .config import SEED, TARGET_NAME, DATA_PATH, DATASET_CONFIGS

# Optional utilities (may not exist in this package)
try:
    from .utils import ClassificationMetricsF1, pycol_complexity
except ModuleNotFoundError:
    ClassificationMetricsF1 = None
    pycol_complexity = None

# Optional pipeline (exists, but keep import safe in case of environment issues)
# try:
#     from . import experiments_pipeline as exp
# except ModuleNotFoundError:
#     exp = None
