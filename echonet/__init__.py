"""
The echonet package contains code for loading echocardiogram videos, and
functions for training and testing segmentation and ejection fraction
prediction models.
"""

from .__version__ import __version__
from .config import CONFIG as config
from . import datasets
from . import utils