"""Custom policy package for LeRobot."""

try:
    import lerobot  # noqa: F401
except ImportError as e:
    raise ImportError(
        "lerobot is not installed. Please install lerobot to use this policy package."
    ) from e

from .configuration_demo import DemoConfig
from .modeling_demo import DemoPolicy
from .processor_demo import make_demo_policy_pre_post_processors


#它主要影响： from lerobot_policy_my_policy import *   # 只导出 __all__ 里列出的名字
__all__ = [
    "DemoConfig",
    "DemoPolicy",
    "make_demo_policy_pre_post_processors",
]
