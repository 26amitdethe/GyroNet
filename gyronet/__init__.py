"""GyroNet: stellar age prediction via gyrochronology."""

from importlib.metadata import version
__version__ = version("gyronet")

from gyronet.predict import predict, predict_csv
from gyronet.posterior import Posterior

__all__ = ["predict", "predict_csv", "Posterior"]
