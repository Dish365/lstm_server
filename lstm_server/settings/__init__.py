from .base import *

# Import environment-specific settings
try:
    from .local import *
except ImportError:
    from .production import * 