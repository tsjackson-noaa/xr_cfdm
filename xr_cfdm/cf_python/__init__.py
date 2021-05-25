__Conventions__ = "CF-1.8"
__date__ = "2021-05-23"

# # our modifications
# from ..aggregate import aggregate
# from ..mixin import Coordinate
# from ..units import Units
# from ..field import Field
# from ..data import Data
# from ..implementation import CFImplementation, implementation
# from ..read_write import read, write

# imports as done in cf-python

# disable incomplete MPI code
mpi_on = False
mpi_size = 1

from .data import (
    # our additions
    Data,
    XRArray,
    # unchanged from cf-python
    FilledArray,
    GatheredArray,
    RaggedContiguousArray,
    RaggedIndexedArray,
    RaggedIndexedContiguousArray,
)

from .constructs import Constructs

from .count import Count
from .index import Index
from .list import List
from .nodecountproperties import NodeCountProperties
from .partnodecountproperties import PartNodeCountProperties
from .interiorring import InteriorRing

from .bounds import Bounds
from .domain import Domain
from .datum import Datum
from .coordinateconversion import CoordinateConversion

from .cfdatetime import dt, dt_vector
from .flags import Flags
from .timeduration import TimeDuration, Y, M, D, h, m, s

from .constructlist import ConstructList
from .fieldlist import FieldList

from .dimensioncoordinate import DimensionCoordinate
from .auxiliarycoordinate import AuxiliaryCoordinate
from .coordinatereference import CoordinateReference
from .cellmethod import CellMethod
from .cellmeasure import CellMeasure
from .domainancillary import DomainAncillary
from .domainaxis import DomainAxis
from .fieldancillary import FieldAncillary

from .query import (
    Query,
    lt,
    le,
    gt,
    ge,
    eq,
    ne,
    contain,
    contains,
    wi,
    wo,
    set,
    year,
    month,
    day,
    hour,
    minute,
    second,
    dtlt,
    dtle,
    dtgt,
    dtge,
    dteq,
    dtne,
    cellsize,
    cellge,
    cellgt,
    cellle,
    celllt,
    cellwi,
    cellwo,
    djf,
    mam,
    jja,
    son,
    seasons,
)
from .constants import *  # noqa: F403
from .functions import *  # noqa: F403
# from .maths import relative_vorticity, histogram
# from .examplefield import example_field, example_fields, example_domain

