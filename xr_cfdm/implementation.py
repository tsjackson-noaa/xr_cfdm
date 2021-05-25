"""Collect component classes from this project, cfdm and cf-python into a
compatible "implementation" of the abstract CF data model.

This is cfdm's main supported extensbility mechanism; see
`<https://ncas-cms.github.io/cfdm/extensions.html>`__.
"""
import cfdm

from . import data
from .field import XRField
from .cf_python import (
    AuxiliaryCoordinate,
    CellMethod,
    CellMeasure,
    CoordinateReference,
    DimensionCoordinate,
    DomainAncillary,
    DomainAxis,
    FieldAncillary,
    Bounds,
    InteriorRing,
    CoordinateConversion,
    Datum,
    Count,
    List,
    Index,
    NodeCountProperties,
    PartNodeCountProperties,
    GatheredArray,
    RaggedContiguousArray,
    RaggedIndexedArray,
    RaggedIndexedContiguousArray,
)
from .cf_python.functions import CF

import logging
logger = logging.getLogger(__name__)

class CFImplementation(cfdm.CFDMImplementation):
    pass

_implementation = CFImplementation(
    # our modifications
    Field=XRField,
    Data=data.Data,
    NetCDFArray=data.XRArray,

    # reminder carried over from cf-python
    cf_version=CF(),

    # parts of Field (via composition)
    AuxiliaryCoordinate=AuxiliaryCoordinate,
    CellMeasure=CellMeasure,
    CellMethod=CellMethod,
    CoordinateReference=CoordinateReference,
    DimensionCoordinate=DimensionCoordinate,
    DomainAncillary=DomainAncillary,
    DomainAxis=DomainAxis,
    FieldAncillary=FieldAncillary,

    # geometry
    Bounds=Bounds,
    InteriorRing=InteriorRing,
    CoordinateConversion=CoordinateConversion,
    Datum=Datum,

    # discrete sampling geometry
    List=List,
    Index=Index,
    Count=Count,
    NodeCountProperties=NodeCountProperties,
    PartNodeCountProperties=PartNodeCountProperties,

    # other data containers
    GatheredArray=GatheredArray,
    RaggedContiguousArray=RaggedContiguousArray,
    RaggedIndexedArray=RaggedIndexedArray,
    RaggedIndexedContiguousArray=RaggedIndexedContiguousArray,
)

def implementation():
    """Return a container for the CF data model implementation.
    """
    return _implementation.copy()
