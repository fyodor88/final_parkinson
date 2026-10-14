# ParkinSense — training package
try:
    from .train import ModelTrainer
    __all__ = ["ModelTrainer"]
except ImportError:
    __all__ = []

