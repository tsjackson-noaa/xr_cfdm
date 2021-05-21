"""Collect component classes from this project, cfdm and cf-python into a
compatible "implementation" of the abstract CF data model.

This is cfdm's main supported extensbility mechanism; see
`<https://ncas-cms.github.io/cfdm/extensions.html>`__.
"""
import cfdm
import cf

from .field import XRField
from . import data

class CFImplementation(cfdm.CFDMImplementation):
    pass

_implementation = CFImplementation(
    # our modifications
    Field=XRField,
    Data=data.XRData,
    NetCDFArray=data.XRArray,

    # reminder carried over from cf-python
    cf_version=cf.CF(),

    # parts of Field (via composition)
    AuxiliaryCoordinate=cf.AuxiliaryCoordinate,
    CellMeasure=cf.CellMeasure,
    CellMethod=cf.CellMethod,
    CoordinateReference=cf.CoordinateReference,
    DimensionCoordinate=cf.DimensionCoordinate,
    DomainAncillary=cf.DomainAncillary,
    DomainAxis=cf.DomainAxis,
    FieldAncillary=cf.FieldAncillary,

    # geometry
    Bounds=cf.Bounds,
    InteriorRing=cf.InteriorRing,
    CoordinateConversion=cf.CoordinateConversion,
    Datum=cf.Datum,

    # discrete sampling geometry
    List=cf.List,
    Index=cf.Index,
    Count=cf.Count,
    NodeCountProperties=cf.NodeCountProperties,
    PartNodeCountProperties=cf.PartNodeCountProperties,

    # other data containers
    GatheredArray=cf.GatheredArray,
    RaggedContiguousArray=cf.RaggedContiguousArray,
    RaggedIndexedArray=cf.RaggedIndexedArray,
    RaggedIndexedContiguousArray=cf.RaggedIndexedContiguousArray,
)

def implementation():
    """Return a container for the CF data model implementation.
    """
    return _implementation.copy()
