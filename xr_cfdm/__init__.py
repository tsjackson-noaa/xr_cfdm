__date__ = "2021-05-23"
__version__ = "0.1"

_requires = (
    "numpy",
    "netCDF4",
    "cftime",
    "cfunits",
    "cfdm",
)

x = ", ".join(_requires)
_error0 = f"cf v{ __version__} requires the modules {x}. "

try:
    import cfdm
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

__cf_version__ = cfdm.core.__cf_version__

from distutils.version import LooseVersion
import importlib.util
import platform

# Check the version of Python
_minimum_vn = "3.6.0"
if LooseVersion(platform.python_version()) < LooseVersion(_minimum_vn):
    raise ValueError(
        f"Bad python version: cf requires python version {_minimum_vn} "
        f"or later. Got {platform.python_version()}"
    )

if LooseVersion(platform.python_version()) < LooseVersion("3.7.0"):
    print(
        "\nDeprecation Warning: Python 3.6 support will be removed at "
        "the next version of cf\n"
    )

_found_ESMF = bool(importlib.util.find_spec("ESMF"))

try:
    import netCDF4
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

try:
    import numpy
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

try:
    import cftime
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

try:
    import cfunits
except ImportError as error1:
    raise ImportError(_error0 + str(error1))

# Check the version of netCDF4
_minimum_vn = "1.5.4"
if LooseVersion(netCDF4.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        f"Bad netCDF4 version: cf requires netCDF4>={_minimum_vn}. "
        f"Got {netCDF4.__version__} at {netCDF4.__file__}"
    )

# Check the version of cftime
_minimum_vn = "1.5.0"
if LooseVersion(cftime.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        f"Bad cftime version: cf requires cftime>={_minimum_vn}. "
        f"Got {cftime.__version__} at {cftime.__file__}"
    )

# Check the version of numpy
_minimum_vn = "1.15"
if LooseVersion(numpy.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        f"Bad numpy version: cf requires numpy>={_minimum_vn}. "
        f"Got {numpy.__version__} at {numpy.__file__}"
    )

# Check the version of cfunits
_minimum_vn = "3.3.2"
if LooseVersion(cfunits.__version__) < LooseVersion(_minimum_vn):
    raise RuntimeError(
        f"Bad cfunits version: cf requires cfunits>={_minimum_vn}. "
        f"Got {cfunits.__version__} at {cfunits.__file__}"
    )

# Check the version of cfdm
_minimum_vn = "1.8.9.0"
_maximum_vn = "1.8.10.0"
_cfdm_version = LooseVersion(cfdm.__version__)
if not LooseVersion(_minimum_vn) <= _cfdm_version < LooseVersion(_maximum_vn):
    raise RuntimeError(
        f"Bad cfdm version: cf requires {_minimum_vn}<=cfdm<{_maximum_vn}. "
        f"Got {_cfdm_version} at {cfdm.__file__}"
    )


from .cf_python import *
# from .data import * # our .data gets imported by .cf_python.data's __init__
from .units import Units
from .field import XRField
from .aggregate import aggregate
from .implementation import CFImplementation, implementation
from .read_write import read, write

# Set up basic logging for the full project with a root logger
import logging
import sys

# Configure the root logger which all module loggers inherit from:
logging.basicConfig(
    stream=sys.stdout,
    style="{",  # default is old style ('%') string formatting
    format="{message}",  # no module names or datetimes etc. for basic case
    level=logging.WARNING,  # default but change level via log_level()
)

# And create custom level inbetween 'INFO' & 'DEBUG', to understand value see:
# https://docs.python.org/3.8/howto/logging.html#logging-levels
logging.DETAIL = 15  # set value as an attribute as done for built-in levels
logging.addLevelName(logging.DETAIL, "DETAIL")


def detail(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.DETAIL):
        self._log(logging.DETAIL, message, args, **kwargs)


logging.Logger.detail = detail


# Also create special, secret level below even 'DEBUG'. It will not be
# advertised to users. The user-facing cf.log_level() can set all but this
# one level; we deliberately have not set up:
#     cf.log_level('PARTITIONING')
# to work to change the level to logging.PARTITIONING. Instead, to set this
# manipulate the cf root logger directly via a built-in method, i.e call:
#     cf.logging.getLogger().setLevel('PARTITIONING')
logging.PARTITIONING = 5
logging.addLevelName(logging.PARTITIONING, "PARTITIONING")


def partitioning(self, message, *args, **kwargs):
    if self.isEnabledFor(logging.PARTITIONING):
        self._log(logging.PARTITIONING, message, args, **kwargs)


logging.Logger.partitioning = partitioning
