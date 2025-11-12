"""
nothing spacial to say~~~
"""
from . import Popup
from . import FileOperate
from . import Logger
from . import Evaluate

__all__ = ['Popup', 'FileOperate', 'Logger', 'Evaluate']

import os
import sys

package_root = os.path.dirname(os.path.abspath(__file__))  # EyeEvaluate
project_root = os.path.abspath(os.path.join(package_root, ".."))  # LocalGaze
if project_root not in sys.path:
    sys.path.insert(0, project_root)
