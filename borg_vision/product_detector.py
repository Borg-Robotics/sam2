"""Backwards-compatible shim.

The package detector and result moved to `borg_vision.detectors.package` and
`borg_vision.results` during the multi-mode refactor. This module re-exports
them under their original names so existing imports
(`from borg_vision.product_detector import ProductDetector`) keep working.
"""

from .detectors.package import PackageDetector as ProductDetector
from .results import PackageResult as ProductDetectionResult

__all__ = ["ProductDetector", "ProductDetectionResult"]
