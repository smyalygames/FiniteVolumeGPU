from os import environ

# TODO this is temporary, remove
from .cuda import *

__env_name = 'GPU_LANG'

if __env_name in environ and environ.get(__env_name).lower() == "cuda":
    from .cuda import *
else:
    from .hip import *