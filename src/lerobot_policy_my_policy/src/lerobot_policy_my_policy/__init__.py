"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    ) from e

from .configuration_my_policy import MyPolicyConfig
from .modeling_my_policy import MyPolicyPolicy
from .processor_my_policy import make_my_policy_pre_post_processors

__all__ = [
    "MyPolicyConfig",
    "MyPolicyPolicy",
    "make_my_policy_pre_post_processors",
]
