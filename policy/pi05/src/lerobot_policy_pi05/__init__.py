"""Custom PI05 policy package for LeRobot (local fork)."""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    ) from e

from .configuration_pi05 import PI05Config
from .modeling_pi05 import PI05Policy
from .processor_pi05 import make_my_pi05_pre_post_processors

__all__ = [
    "PI05Config",
    "PI05Policy",
    "make_my_pi05_pre_post_processors",
]
