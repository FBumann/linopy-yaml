"""linopy_yaml — YAML-based math definition layer for linopy."""

from linopy_yaml.accessor import _install
from linopy_yaml.helpers import register
from linopy_yaml.schema import MathSchema

# Monkey-patch linopy.Model with .from_yaml() and .yaml accessor
_install()

__all__ = ["MathSchema", "register"]
__version__ = "0.0.1"
