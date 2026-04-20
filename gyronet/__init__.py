"""GyroNet: stellar age prediction via gyrochronology."""

__version__ = "0.1.0"

from gyronet.predict import predict, predict_csv
from gyronet.posterior import Posterior

__all__ = ["predict", "predict_csv", "Posterior"]
