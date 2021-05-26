from collections import namedtuple
from functools import reduce
from operator import mul as operator_mul
from operator import itemgetter

from .propertiesdata import PropertiesData
from .data.data import Data
from .units import Units

import logging

try:
    from matplotlib.path import Path
except ImportError:
    pass

from numpy import arange as numpy_arange
from numpy import array as numpy_array
from numpy import array_equal as numpy_array_equal

from numpy import asanyarray as numpy_asanyarray
from numpy import can_cast as numpy_can_cast
from numpy import diff as numpy_diff
from numpy import delete as numpy_delete
from numpy import empty as numpy_empty
from numpy import finfo as numpy_finfo
from numpy import full as numpy_full
from numpy import nan as numpy_nan
from numpy import ndarray as numpy_ndarray
from numpy import pi as numpy_pi
from numpy import prod as numpy_prod
from numpy import reshape as numpy_reshape
from numpy import shape as numpy_shape
from numpy import size as numpy_size
from numpy import squeeze as numpy_squeeze
from numpy import tile as numpy_tile
from numpy import unique as numpy_unique
from numpy import where as numpy_where

from numpy.ma import is_masked as numpy_ma_is_masked
from numpy.ma import isMA as numpy_ma_isMA

from numpy.ma import MaskedArray as numpy_ma_MaskedArray
from numpy.ma import where as numpy_ma_where

import cfdm

from .cf_python import (
    AuxiliaryCoordinate,
    Bounds,
    CellMethod,
    DimensionCoordinate,
    Domain,
    DomainAncillary,
    DomainAxis,
    Flags,
    Constructs,
    FieldList,
    Count,
    Index,
    List,
)

from .cf_python.constants import masked as cf_masked

from .cf_python.functions import parse_indices, chunksize, _section
from .cf_python.functions import relaxed_identities as cf_relaxed_identities
from .cf_python.query import Query, ge, gt, le, lt, eq
from .cf_python.regrid import Regrid
from .cf_python.timeduration import TimeDuration

from .cf_python.subspacefield import SubspaceField

from .cf_python.data import RaggedContiguousArray
from .cf_python.data import RaggedIndexedArray
from .cf_python.data import RaggedIndexedContiguousArray
from .cf_python.data import GatheredArray

from .cf_python import mixin

from .cf_python.functions import (
    _DEPRECATION_ERROR,
    _DEPRECATION_ERROR_ARG,
    _DEPRECATION_ERROR_KWARGS,
    _DEPRECATION_ERROR_METHOD,
    _DEPRECATION_ERROR_KWARG_VALUE,
    DeprecationError,
)

from .cf_python.formula_terms import FormulaTerms

from .cf_python.decorators import (
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _deprecated_kwarg_check,
    _manage_log_level_via_verbosity,
)


logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Commonly used units
# --------------------------------------------------------------------
# _units_degrees = Units("degrees")
_units_radians = Units("radians")
_units_metres = Units("m")
_units_1 = Units("1")

_earth_radius = Data(6371229.0, "m")

_empty_set = set()


# don't *think* we need to mess with cfdm.Field, or any of its parents

class Field(mixin.FieldDomain, PropertiesData, cfdm.Field):
    """A field construct of the CF data model.

    The field construct is central to the CF data model, and includes
    all the other constructs. A field corresponds to a CF-netCDF data
    variable with all of its metadata. All CF-netCDF elements are
    mapped to a field construct or some element of the CF field
    construct. The field construct contains all the data and metadata
    which can be extracted from the file using the CF conventions.

    The field construct consists of a data array and the definition of
    its domain (that describes the locations of each cell of the data
    array), field ancillary constructs containing metadata defined
    over the same domain, and cell method constructs to describe how
    the cell values represent the variation of the physical quantity
    within the cells of the domain. The domain is defined collectively
    by the following constructs of the CF data model: domain axis,
    dimension coordinate, auxiliary coordinate, cell measure,
    coordinate reference and domain ancillary constructs.

    The field construct also has optional properties to describe
    aspects of the data that are independent of the domain. These
    correspond to some netCDF attributes of variables (e.g. units,
    long_name and standard_name), and some netCDF global file
    attributes (e.g. history and institution).

    **NetCDF interface**

    {{netCDF variable}}

    {{netCDF global attributes}}

    {{netCDF group attributes}}

    {{netCDF geometry group}}

    Some components exist within multiple constructs, but when written
    to a netCDF dataset the netCDF names associated with such
    components will be arbitrarily taken from one of them. The netCDF
    variable, dimension and sample dimension names and group
    structures for such components may be set or removed consistently
    across all such components with the `nc_del_component_variable`,
    `nc_set_component_variable`, `nc_set_component_variable_groups`,
    `nc_clear_component_variable_groups`,
    `nc_del_component_dimension`, `nc_set_component_dimension`,
    `nc_set_component_dimension_groups`,
    `nc_clear_component_dimension_groups`,
    `nc_del_component_sample_dimension`,
    `nc_set_component_sample_dimension`,
    `nc_set_component_sample_dimension_groups`,
    `nc_clear_component_sample_dimension_groups` methods.

    CF-compliance issues for field constructs read from a netCDF
    dataset may be accessed with the `dataset_compliance` method.

    """

    def __new__(cls, *args, **kwargs):
        """TODO."""
        instance = super().__new__(cls)
        instance._AuxiliaryCoordinate = AuxiliaryCoordinate
        instance._Bounds = Bounds
        instance._Constructs = Constructs
        instance._Domain = Domain
        instance._DomainAncillary = DomainAncillary
        instance._DomainAxis = DomainAxis
        #        instance._Data = Data
        instance._RaggedContiguousArray = RaggedContiguousArray
        instance._RaggedIndexedArray = RaggedIndexedArray
        instance._RaggedIndexedContiguousArray = RaggedIndexedContiguousArray
        instance._GatheredArray = GatheredArray
        instance._Count = Count
        instance._Index = Index
        instance._List = List
        return instance

    _special_properties = mixin.PropertiesData._special_properties
    _special_properties += ("flag_values", "flag_masks", "flag_meanings")

    def __init__(
        self, properties=None, source=None, copy=True, _use_data=True
    ):
        """**Initialization**

        :Parameters:

            properties: `dict`, optional
                Set descriptive properties. The dictionary keys are
                property names, with corresponding values. Ignored if the
                *source* parameter is set.

                *Parameter example:*
                  ``properties={'standard_name': 'air_temperature'}``

                Properties may also be set after initialisation with the
                `set_properties` and `set_property` methods.

            source: optional
                Initialize the properties, data and metadata constructs
                from those of *source*.

            copy: `bool`, optional
                If False then do not deep copy input parameters prior to
                initialization. By default arguments are deep copied.

        """
        super().__init__(
            properties=properties,
            source=source,
            copy=copy,
            _use_data=_use_data,
        )

        if source:
            flags = getattr(source, "Flags", None)
            if flags is not None:
                self.Flags = flags.copy()

    def __getitem__(self, indices):
        """Return a subspace of the field construct defined by indices.

        f.__getitem__(indices) <==> f[indices]

        Subspacing by indexing uses rules that are very similar to the
        numpy indexing rules, the only differences being:

        * An integer index i specified for a dimension reduces the size of
          this dimension to unity, taking just the i-th element, but keeps
          the dimension itself, so that the rank of the array is not
          reduced.

        * When two or more dimensions’ indices are sequences of integers
          then these indices work independently along each dimension
          (similar to the way vector subscripts work in Fortran). This is
          the same indexing behaviour as on a Variable object of the
          netCDF4 package.

        * For a dimension that is cyclic, a range of indices specified by
          a `slice` that spans the edges of the data (such as ``-2:3`` or
          ``3:-2:-1``) is assumed to "wrap" around, rather then producing
          a null result.

        .. seealso:: `indices`, `squeeze`, `subspace`, `where`

        **Examples:**

        >>> f.shape
        (12, 73, 96)
        >>> f[0].shape
        (1, 73, 96)
        >>> f[3, slice(10, 0, -2), 95:93:-1].shape
        (1, 5, 2)

        >>> f.shape
        (12, 73, 96)
        >>> f[:, [0, 72], [5, 4, 3]].shape
        (12, 2, 3)

        >>> f.shape
        (12, 73, 96)
        >>> f[...].shape
        (12, 73, 96)
        >>> f[slice(0, 12), :, 10:0:-2].shape
        (12, 73, 5)
        >>> f[[True, True, False, True, True, False, False, True, True, True,
        ...    True, True]].shape
        (9, 64, 128)
        >>> f[..., :6, 9:1:-2, [1, 3, 4]].shape
        (6, 4, 3)

        """
        logger.debug(
            self.__class__.__name__ + ".__getitem__"
        )  # pragma: no cover
        logger.debug(f"    input indices = {indices}")  # pragma: no cover

        if indices is Ellipsis:
            return self.copy()

        data = self.data
        shape = data.shape

        # Parse the index
        if not isinstance(indices, tuple):
            indices = (indices,)

        if isinstance(indices[0], str) and indices[0] == "mask":
            auxiliary_mask = indices[:2]
            indices2 = indices[2:]
        else:
            auxiliary_mask = None
            indices2 = indices

        indices, roll = parse_indices(shape, indices2, cyclic=True)

        if roll:
            new = self
            axes = data._axes
            cyclic_axes = data._cyclic
            for iaxis, shift in roll.items():
                axis = axes[iaxis]
                if axis not in cyclic_axes:
                    _ = self.get_data_axes()[iaxis]
                    raise IndexError(
                        "Can't take a cyclic slice from non-cyclic "
                        f"{self.constructs.domain_axis_identity(_)!r} axis"
                    )

                logger.debug(
                    f"    roll, iaxis, shift = {roll} {iaxis} {shift}"
                )  # pragma: no cover

                new = new.roll(iaxis, shift)
        else:
            new = self.copy()

        # ------------------------------------------------------------
        # Subspace the field construct's data
        # ------------------------------------------------------------
        if auxiliary_mask:
            auxiliary_mask = list(auxiliary_mask)
            findices = auxiliary_mask + indices
        else:
            findices = indices

        logger.debug("    shape    = {}".format(shape))  # pragma: no cover
        logger.debug("    indices  = {}".format(indices))  # pragma: no cover
        logger.debug("    indices2 = {}".format(indices2))  # pragma: no cover
        logger.debug("    findices = {}".format(findices))  # pragma: no cover

        new_data = new.data[tuple(findices)]

        # Set sizes of domain axes
        data_axes = new.get_data_axes()
        domain_axes = new.domain_axes(todict=True)
        for axis, size in zip(data_axes, new_data.shape):
            domain_axes[axis].set_size(size)

        # ------------------------------------------------------------
        # Subspace constructs with data
        # ------------------------------------------------------------
        if data_axes:
            construct_data_axes = new.constructs.data_axes()

            for key, construct in new.constructs.filter_by_axis(
                *data_axes, axis_mode="or", todict=True
            ).items():
                construct_axes = construct_data_axes[key]
                dice = []
                needs_slicing = False
                for axis in construct_axes:
                    if axis in data_axes:
                        needs_slicing = True
                        dice.append(indices[data_axes.index(axis)])
                    else:
                        dice.append(slice(None))

                # Generally we do not apply an auxiliary mask to the
                # metadata items, but for DSGs we do.
                if auxiliary_mask and new.DSG:
                    item_mask = []
                    for mask in auxiliary_mask[1]:
                        iaxes = [
                            data_axes.index(axis)
                            for axis in construct_axes
                            if axis in data_axes
                        ]
                        for i, (axis, size) in enumerate(
                            zip(data_axes, mask.shape)
                        ):
                            if axis not in construct_axes:
                                if size > 1:
                                    iaxes = None
                                    break

                                mask = mask.squeeze(i)

                        if iaxes is None:
                            item_mask = None
                            break
                        else:
                            mask1 = mask.transpose(iaxes)
                            for i, axis in enumerate(construct_axes):
                                if axis not in data_axes:
                                    mask1.inset_dimension(i)

                            item_mask.append(mask1)

                    if item_mask:
                        needs_slicing = True
                        dice = [auxiliary_mask[0], item_mask] + dice

                # Replace existing construct with its subspace
                if needs_slicing:
                    new.set_construct(
                        construct[tuple(dice)],
                        key=key,
                        axes=construct_axes,
                        copy=False,
                    )

        new.set_data(new_data, axes=data_axes, copy=False)

        return new

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    def __setitem__(self, indices, value):
        """Called to implement assignment to x[indices]=value.

        x.__setitem__(indices, value) <==> x[indices]=value

        .. versionadded:: 2.0

        """
        if isinstance(value, self.__class__):
            value = self._conform_for_assignment(value)

        try:
            data = value.get_data(None, _fill_value=False)
        except AttributeError:
            pass
        else:
            if data is None:
                raise ValueError(
                    f"Can't assign to a {self.__class__.__name__} from a "
                    f"{value.__class__.__name__} with no data"
                )

            value = data

        data = self.get_data(_fill_value=False)
        data[indices] = value

    @property
    def _cyclic(self):
        """Storage for axis cyclicity.

        Do not change the value in-place.

        """
        return self._custom.get("_cyclic", _empty_set)

    @_cyclic.setter
    def _cyclic(self, value):
        """value must be a set.

        Do not change the value in-place.

        """
        self._custom["_cyclic"] = value

    @_cyclic.deleter
    def _cyclic(self):
        self._custom["_cyclic"] = _empty_set

    def analyse_items(self, relaxed_identities=None):
        """Analyse a domain.

        :Returns:

            `dict`
                A description of the domain.

        **Examples:**

        >>> print(f)
        Axes           : time(3) = [1979-05-01 12:00:00, ..., 1979-05-03 12:00:00] gregorian
                       : air_pressure(5) = [850.000061035, ..., 50.0000038147] hPa
                       : grid_longitude(106) = [-20.5400109887, ..., 25.6599887609] degrees
                       : grid_latitude(110) = [23.3200002313, ..., -24.6399995089] degrees
        Aux coords     : latitude(grid_latitude(110), grid_longitude(106)) = [[67.1246607722, ..., 22.8886948065]] degrees_N
                       : longitude(grid_latitude(110), grid_longitude(106)) = [[-45.98136251, ..., 35.2925499052]] degrees_E
        Coord refs     : <CF CoordinateReference: rotated_latitude_longitude>

        >>> f.analyse_items()
        {
         'dim_coords': {'dim0': <CF Dim ....>,
         'aux_coords': {'N-d': {'aux0': <CF AuxiliaryCoordinate: latitude(110, 106) degrees_N>,
                                'aux1': <CF AuxiliaryCoordinate: longitude(110, 106) degrees_E>},
                        'dim0': {'1-d': {},
                                 'N-d': {},},
                        'dim1': {'1-d': {},
                                 'N-d': {},},
                        'dim2': {'1-d': {},
                                 'N-d': {'aux0': <CF AuxiliaryCoordinate: latitude(110, 106) degrees_N>,
                                         'aux1': <CF AuxiliaryCoordinate: longitude(110, 106) degrees_E>},},
                        'dim3': {'1-d': {},
                                 'N-d': {'aux0': <CF AuxiliaryCoordinate: latitude(110, 106) degrees_N>,
                                         'aux1': <CF AuxiliaryCoordinate: longitude(110, 106) degrees_E>},},},
         'axis_to_coord': {'dim0': <CF DimensionCoordinate: time(3) gregorian>,
                           'dim1': <CF DimensionCoordinate: air_pressure(5) hPa>,
                           'dim2': <CF DimensionCoordinate: grid_latitude(110) degrees>,
                           'dim3': <CF DimensionCoordinate: grid_longitude(106) degrees>},
         'axis_to_id': {'dim0': 'time',
                        'dim1': 'air_pressure',
                        'dim2': 'grid_latitude',
                        'dim3': 'grid_longitude'},
         'cell_measures': {'N-d': {},
                           'dim0': {'1-d': {},
                                    'N-d': {},},
                           'dim1': {'1-d': {},
                                    'N-d': {},},
                           'dim2': {'1-d': {},
                                    'N-d': {},},
                           'dim3': {'1-d': {},
                                    'N-d': {},},
            },
         'id_to_aux': {},
         'id_to_axis': {'air_pressure': 'dim1',
                        'grid_latitude': 'dim2',
                        'grid_longitude': 'dim3',
                        'time': 'dim0'},
         'id_to_coord': {'air_pressure': <CF DimensionCoordinate: air_pressure(5) hPa>,
                         'grid_latitude': <CF DimensionCoordinate: grid_latitude(110) degrees>,
                         'grid_longitude': <CF DimensionCoordinate: grid_longitude(106) degrees>,
                         'time': <CF DimensionCoordinate: time(3) gregorian>},
         'id_to_key': {'air_pressure': 'dim1',
                       'grid_latitude': 'dim2',
                       'grid_longitude': 'dim3',
                       'time': 'dim0'},
         'undefined_axes': [],
         'warnings': [],
        }

        """
        # ------------------------------------------------------------
        # Map each axis identity to its identifier, if such a mapping
        # exists.
        #
        # For example:
        # >>> id_to_axis
        # {'time': 'dim0', 'height': dim1'}
        # ------------------------------------------------------------
        id_to_axis = {}

        # ------------------------------------------------------------
        # For each dimension that is identified by a 1-d auxiliary
        # coordinate, map its dimension's its identifier.
        #
        # For example:
        # >>> id_to_aux
        # {'region': 'aux0'}
        # ------------------------------------------------------------
        id_to_aux = {}

        # ------------------------------------------------------------
        # The keys of the coordinate items which provide axis
        # identities
        #
        # For example:
        # >>> id_to_key
        # {'region': 'aux0'}
        # ------------------------------------------------------------
        #        id_to_key = {}

        axis_to_id = {}

        # ------------------------------------------------------------
        # Map each dimension's identity to the coordinate which
        # provides that identity.
        #
        # For example:
        # >>> id_to_coord
        # {'time': <CF Coordinate: time(12)>}
        # ------------------------------------------------------------
        id_to_coord = {}

        axis_to_coord = {}

        # ------------------------------------------------------------
        # List the dimensions which are undefined, in that no unique
        # identity can be assigned to them.
        #
        # For example:
        # >>> undefined_axes
        # ['dim2']
        # ------------------------------------------------------------
        undefined_axes = []

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        warnings = []
        id_to_dim = {}
        axis_to_aux = {}
        axis_to_dim = {}

        if relaxed_identities is None:
            relaxed_identities = cf_relaxed_identities()

        #        dimension_coordinates = self.dimension_coordinates(view=True)
        #        auxiliary_coordinates = self.auxiliary_coordinates(view=True)

        for axis in self.domain_axes(todict=True):

            #            dims = self.constructs.chain(
            #                "filter_by_type",
            #                ("dimension_coordinate",), "filter_by_axis", (axis,)
            #                mode="and", todict=True
            #            )
            key, dim = self.dimension_coordinate(
                item=True, default=(None, None), filter_by_axis=(axis,)
            )

            if dim is not None:
                # This axis of the domain has a dimension coordinate
                identity = dim.identity(strict=True, default=None)
                if identity is None:
                    # Dimension coordinate has no identity, but it may
                    # have a recognised axis.
                    for ctype in ("T", "X", "Y", "Z"):
                        if getattr(dim, ctype, False):
                            identity = ctype
                            break

                if identity is None and relaxed_identities:
                    identity = dim.identity(relaxed=True, default=None)

                if identity:
                    if identity in id_to_axis:
                        warnings.append("Field has multiple {identity!r} axes")

                    axis_to_id[axis] = identity
                    id_to_axis[identity] = axis
                    axis_to_coord[axis] = key
                    id_to_coord[identity] = key
                    axis_to_dim[axis] = key
                    id_to_dim[identity] = key
                    continue

            else:
                key, aux = self.auxiliary_coordinate(
                    filter_by_axis=(axis,),
                    axis_mode="and",  # TODO check this "and"
                    item=True,
                    default=(None, None),
                )
                if aux is not None:
                    # This axis of the domain does not have a
                    # dimension coordinate but it does have exactly
                    # one 1-d auxiliary coordinate, so that will do.
                    identity = aux.identity(strict=True, default=None)

                    if identity is None and relaxed_identities:
                        identity = aux.identity(relaxed=True, default=None)

                    if identity and aux.has_data():
                        if identity in id_to_axis:
                            warnings.append(
                                f"Field has multiple {identity!r} axes"
                            )

                        axis_to_id[axis] = identity
                        id_to_axis[identity] = axis
                        axis_to_coord[axis] = key
                        id_to_coord[identity] = key
                        axis_to_aux[axis] = key
                        id_to_aux[identity] = key
                        continue

            # Still here? Then this axis is undefined
            undefined_axes.append(axis)

        return {
            "axis_to_id": axis_to_id,
            "id_to_axis": id_to_axis,
            "axis_to_coord": axis_to_coord,
            "axis_to_dim": axis_to_dim,
            "axis_to_aux": axis_to_aux,
            "id_to_coord": id_to_coord,
            "id_to_dim": id_to_dim,
            "id_to_aux": id_to_aux,
            "undefined_axes": undefined_axes,
            "warnings": warnings,
        }

    def _is_broadcastable(self, shape):
        """Checks the field's data array is broadcastable to a shape.

        :Parameters:

            shape: sequence of `int`

        :Returns:

            `bool`

        """
        shape0 = getattr(self, "shape", None)
        if shape is None:
            return False

        shape1 = shape

        if tuple(shape1) == tuple(shape0):
            # Same shape
            return True

        ndim0 = len(shape0)
        ndim1 = len(shape1)
        if not ndim0 or not ndim1:
            # Either or both is scalar
            return True

        for setN in set(shape0), set(shape1):
            if setN == {1}:
                return True

        if ndim1 > ndim0:
            return False

        for n, m in zip(shape1[::-1], shape0[::-1]):
            if n != m and n != 1:
                return False

        return True

    def _axis_positions(self, axes, parse=True, return_axes=False):
        """Convert the given axes to their positions in the data.

        Any domain axes that are not spanned by the data are ignored.

        If there is no data then an empty list is returned.

        .. versionadded:: 3.9.0

        :Parameters:
            axes: (sequence of) `str` or `int`
                The axes to be converted. TODO domain axis selection

            parse: `bool`, optional

                If False then do not parse the *axes*. Parsing should
                always occur unless the given *axes* are the output of
                a previous call to `parse_axes`. By default *axes* is
                parsed by `_parse_axes`.

            return_axes: `bool`, optional

                If True then also return the domain axis identifiers
                corresponding to the positions.

        :Returns:

            `list` [, `list`]
                The domain axis identifiers. If *return_axes* is True
                then also return the corresponding domain axis
                identifiers.

        """
        data_axes = self.get_data_axes(default=None)
        if data_axes is None:
            return []

        if parse:
            axes = self._parse_axes(axes)

        axes = [a for a in axes if a in data_axes]
        positions = [data_axes.index(a) for a in axes]

        if return_axes:
            return positions, axes

        return positions

    def _conform_cell_methods(self):
        """Changes the axes of the field's cell methods so they conform.

        :Returns:

            `None`

        """
        axis_map = {}

        for cm in self.cell_methods(todict=True).values():
            for axis in cm.get_axes(()):
                if axis in axis_map:
                    continue

                if axis == "area":
                    axis_map[axis] = axis
                    continue

                axis_map[axis] = self.domain_axis(axis, key=True, default=axis)

            cm.change_axes(axis_map, inplace=True)

    def _conform_for_assignment(self, other, check_coordinates=False):
        """Conform *other* so that it is ready for metadata-unaware
        assignment broadcasting across *self*.

        Note that *other* is not changed.

        :Parameters:

            other: `Field`
                The field to conform.

        :Returns:

            `Field`
                The conformed version of *other*.

        **Examples:**

        >>> h = f._conform_for_assignment(g)

        """
        # Analyse each domain
        s = self.analyse_items()
        v = other.analyse_items()

        if s["warnings"] or v["warnings"]:
            raise ValueError(
                "Can't setitem: {0}".format(s["warnings"] or v["warnings"])
            )

        # Find the set of matching axes
        matching_ids = set(s["id_to_axis"]).intersection(v["id_to_axis"])
        if not matching_ids:
            raise ValueError("Can't assign: No matching axes")

        # ------------------------------------------------------------
        # Check that any matching axes defined by auxiliary
        # coordinates are done so in both fields.
        # ------------------------------------------------------------
        for identity in matching_ids:
            if (identity in s["id_to_aux"]) + (
                identity in v["id_to_aux"]
            ) == 1:
                raise ValueError(
                    "Can't assign: {0!r} axis defined by auxiliary in only "
                    "1 field".format(identity)
                )

        copied = False

        # ------------------------------------------------------------
        # Check that 1) all undefined axes in other have size 1 and 2)
        # that all of other's unmatched but defined axes have size 1
        # and squeeze any such axes out of its data array.
        #
        # For example, if   self.data is        P T     Z Y   X   A
        #              and  other.data is     1     B C   Y 1 X T
        #              then other.data becomes            Y   X T
        # ------------------------------------------------------------
        squeeze_axes1 = []
        other_domain_axes = other.domain_axes(todict=True)

        for axis1 in v["undefined_axes"]:
            axis_size = other_domain_axes[axis1].get_size()
            if axis_size != 1:
                raise ValueError(
                    "Can't assign: Can't broadcast undefined axis with "
                    f"size {axis_size}"
                )

            squeeze_axes1.append(axis1)

        for identity in set(v["id_to_axis"]).difference(matching_ids):
            axis1 = v["id_to_axis"][identity]
            axis_size = other_domain_axes[axis1].get_size()
            if axis_size != 1:
                raise ValueError(
                    "Can't assign: Can't broadcast size "
                    f"{axis_size} {identity!r} axis"
                )

            squeeze_axes1.append(axis1)

        if squeeze_axes1:
            if not copied:
                other = other.copy()
                copied = True

            other.squeeze(squeeze_axes1, inplace=True)

        # ------------------------------------------------------------
        # Permute the axes of other.data so that they are in the same
        # order as their matching counterparts in self.data
        #
        # For example, if   self.data is       P T Z Y X   A
        #              and  other.data is            Y X T
        #              then other.data becomes   T   Y X
        # ------------------------------------------------------------
        data_axes0 = self.get_data_axes()
        data_axes1 = other.get_data_axes()

        transpose_axes1 = []
        for axis0 in data_axes0:
            identity = s["axis_to_id"][axis0]
            if identity in matching_ids:
                axis1 = v["id_to_axis"][identity]
                if axis1 in data_axes1:
                    transpose_axes1.append(axis1)

        if transpose_axes1 != data_axes1:
            if not copied:
                other = other.copy()
                copied = True

            other.transpose(transpose_axes1, inplace=True)

        # ------------------------------------------------------------
        # Insert size 1 axes into other.data to match axes in
        # self.data which other.data doesn't have.
        #
        # For example, if   self.data is       P T Z Y X A
        #              and  other.data is        T   Y X
        #              then other.data becomes 1 T 1 Y X 1
        # ------------------------------------------------------------
        expand_positions1 = []
        for i, axis0 in enumerate(data_axes0):
            identity = s["axis_to_id"][axis0]
            if identity in matching_ids:
                axis1 = v["id_to_axis"][identity]
                if axis1 not in data_axes1:
                    expand_positions1.append(i)
            else:
                expand_positions1.append(i)

        if expand_positions1:
            if not copied:
                other = other.copy()
                copied = True

            for i in expand_positions1:
                new_axis = other.set_construct(other._DomainAxis(1))
                other.insert_dimension(new_axis, position=i, inplace=True)

        # ----------------------------------------------------------------
        # Make sure that each pair of matching axes has the same
        # direction
        # ----------------------------------------------------------------
        flip_axes1 = []
        for identity in matching_ids:
            axis1 = v["id_to_axis"][identity]
            axis0 = s["id_to_axis"][identity]
            if other.direction(axis1) != self.direction(axis0):
                flip_axes1.append(axis1)

        if flip_axes1:
            if not copied:
                other = other.copy()
                copied = True

            other.flip(flip_axes1, inplace=True)

        # Find the axis names which are present in both fields
        if not check_coordinates:
            return other

        # Still here?
        matching_ids = set(s["id_to_axis"]).intersection(v["id_to_axis"])

        for identity in matching_ids:
            key0 = s["id_to_coord"][identity]
            key1 = v["id_to_coord"][identity]

            coord0 = self.constructs[key0]
            coord1 = other.constructs[key1]

            # Check the sizes of the defining coordinates
            size0 = coord0.size
            size1 = coord1.size
            if size0 != size1:
                if size0 == 1 or size1 == 1:
                    continue

                raise ValueError(
                    "Can't broadcast {!r} axes with sizes {} and {}".format(
                        identity, size0, size1
                    )
                )

            # Check that equally sized defining coordinate data arrays
            # are compatible
            if not coord0._equivalent_data(coord1):
                raise ValueError(
                    f"Matching {identity!r} coordinate constructs have "
                    "different data"
                )

            # If the defining coordinates are attached to
            # coordinate references then check that those
            # coordinate references are equivalent

            # For each field, find the coordinate references which
            # contain the defining coordinate.
            refs0 = [
                key
                for key, ref in self.coordinate_references(todict=True).items()
                if key0 in ref.coordinates()
            ]
            refs1 = [
                key
                for key, ref in other.coordinate_references(
                    todict=True
                ).items()
                if key1 in ref.coordinates()
            ]

            nrefs = len(refs0)
            if nrefs > 1 or nrefs != len(refs1):
                raise ValueError("TODO")

            if nrefs and not self._equivalent_coordinate_references(
                other, key0=refs0[0], key1=refs1[0], s=s, t=v
            ):
                raise ValueError("TODO")

        return other

    def _conform_for_data_broadcasting(self, other):
        """Conforms the field with another, ready for data broadcasting.

        Note that the other field, *other*, is not changed in-place.

        :Parameters:

            other: `Field`
                The field to conform.

        :Returns:

            `Field`
                The conformed version of *other*.

        **Examples:**

        >>> h = f._conform_for_data_broadcasting(g)

        """

        other = self._conform_for_assignment(other, check_coordinates=True)

        # Remove leading size one dimensions
        ndiff = other.ndim - self.ndim
        if ndiff > 0 and set(other.shape[:ndiff]) == set((1,)):
            for i in range(ndiff):
                other = other.squeeze(0)

        return other

    @_manage_log_level_via_verbosity
    def _equivalent_construct_data(
        self,
        field1,
        key0=None,
        key1=None,
        s=None,
        t=None,
        atol=None,
        rtol=None,
        verbose=None,
        axis_map=None,
    ):
        """True if the field has equivalent construct data to another.

        Two real numbers ``x`` and ``y`` are considered equal if
        ``|x-y|<=atol+rtol|y|``, where ``atol`` (the tolerance on absolute
        differences) and ``rtol`` (the tolerance on relative differences)
        are positive, typically very small numbers. See the *atol* and
        *rtol* parameters.

        :Parameters:

            key0: `str`

            key1: `str`

            field1: `Field`

            s: `dict`, optional

            t: `dict`, optional

            atol: `float`, optional
                The tolerance on absolute differences between real
                numbers. The default value is set by the `atol` function.

            rtol: `float`, optional
                The tolerance on relative differences between real
                numbers. The default value is set by the `rtol` function.

            {{verbose: `int` or `str` or `None`, optional}}

        """
        item0 = self.constructs[key0]
        item1 = field1.constructs[key1]

        if item0.has_data() != item1.has_data():
            logger.info(
                "{0}: Only one item has data".format(self.__class__.__name__)
            )  # pragma: no cover
            return False

        if not item0.has_data():
            # Neither field has a data array
            return True

        if item0.size != item1.size:
            logger.info(
                "{}: Different metadata construct data array size: "
                "{} != {}".format(
                    self.__class__.__name__, item0.size, item1.size
                )
            )  # pragma: no cover
            return False

        if item0.ndim != item1.ndim:
            logger.info(
                "{0}: Different data array ranks ({1}, {2})".format(
                    self.__class__.__name__, item0.ndim, item1.ndim
                )
            )  # pragma: no cover
            return False

        axes0 = self.get_data_axes(key0, default=())
        axes1 = field1.get_data_axes(key1, default=())

        if s is None:
            s = self.analyse_items()
        if t is None:
            t = field1.analyse_items()

        transpose_axes = []
        if axis_map is None:
            for axis0 in axes0:
                axis1 = t["id_to_axis"].get(s["axis_to_id"][axis0], None)
                if axis1 is None:
                    # TODO: improve message here (make user friendly):
                    logger.info(
                        "t['id_to_axis'] does not have a key "
                        "s['axis_to_id'][axis0] for {}".format(
                            self.__class__.__name__
                        )
                    )  # pragma: no cover
                    return False

                transpose_axes.append(axes1.index(axis1))
        else:
            for axis0 in axes0:
                axis1 = axis_map.get(axis0)
                if axis1 is None:
                    # TODO: improve message here (make user friendly):
                    logger.info(
                        "axis_map[axis0] is None for {}".format(
                            self.__class__.__name__
                        )
                    )  # pragma: no cover
                    return False

                transpose_axes.append(axes1.index(axis1))

        copy1 = True

        if transpose_axes != list(range(item1.ndim)):
            if copy1:
                item1 = item1.copy()
                copy1 = False

            item1.transpose(transpose_axes, inplace=True)

        if item0.shape != item1.shape:
            # add traceback TODO
            return False

        flip_axes = [
            i
            for i, (axis1, axis0) in enumerate(zip(axes1, axes0))
            if field1.direction(axis1) != self.direction(axis0)
        ]

        if flip_axes:
            if copy1:
                item1 = item1.copy()
                copy1 = False

            item1.flip(flip_axes, inplace=True)

        if not item0._equivalent_data(
            item1, rtol=rtol, atol=atol, verbose=verbose
        ):
            # add traceback TODO
            return False

        return True


    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def DSG(self):
        """True if the field contains a collection of discrete sampling
        geometries.

        .. versionadded:: 2.0

        .. seealso:: `featureType`

        **Examples:**

        >>> f.featureType
        'timeSeries'
        >>> f.DSG
        True

        >>> f.get_property('featureType', 'NOT SET')
        NOT SET
        >>> f.DSG
        False

        """
        return self.has_property("featureType")

    @property
    def Flags(self):
        """A `Flags` object containing self-describing CF flag values.

        Stores the `flag_values`, `flag_meanings` and `flag_masks` CF
        properties in an internally consistent manner.

        **Examples:**

        >>> f.Flags
        <CF Flags: flag_values=[0 1 2], flag_masks=[0 2 2], flag_meanings=['low' 'medium' 'high']>

        """
        try:
            return self._custom["Flags"]
        except KeyError:
            raise AttributeError(
                "{!r} object has no attribute 'Flags'".format(
                    self.__class__.__name__
                )
            )

    @Flags.setter
    def Flags(self, value):
        self._custom["Flags"] = value

    @Flags.deleter
    def Flags(self):
        try:
            return self._custom.pop("Flags")
        except KeyError:
            raise AttributeError(
                "{!r} object has no attribute 'Flags'".format(
                    self.__class__.__name__
                )
            )

    @property
    def rank(self):
        """The number of axes in the domain.

        Note that this may be greater the number of data array axes.

        .. seealso:: `ndim`, `unsqueeze`

        **Examples:**

        >>> print(f)
        air_temperature field summary
        -----------------------------
        Data           : air_temperature(time(12), latitude(64), longitude(128)) K
        Cell methods   : time: mean
        Axes           : time(12) = [ 450-11-16 00:00:00, ...,  451-10-16 12:00:00] noleap
                       : latitude(64) = [-87.8638000488, ..., 87.8638000488] degrees_north
                       : longitude(128) = [0.0, ..., 357.1875] degrees_east
                       : height(1) = [2.0] m
        >>> f.rank
        4
        >>> f.ndim
        3
        >>> f
        <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
        >>> f.unsqueeze(inplace=True)
        <CF Field: air_temperature(height(1), time(12), latitude(64), longitude(128)) K>
        >>> f.rank
        4
        >>> f.ndim
        4

        """
        return len(self.domain_axes(todict=True))

    @property
    def varray(self):
        """A numpy array view of the data array.

        Changing the elements of the returned view changes the data array.

        .. seealso:: `array`, `data`, `datetime_array`

        **Examples:**

        >>> f.data
        <CF Data(5): [0, ... 4] kg m-1 s-2>
        >>> a = f.array
        >>> type(a)
        <type 'numpy.ndarray'>
        >>> print(a)
        [0 1 2 3 4]
        >>> a[0] = 999
        >>> print(a)
        [999 1 2 3 4]
        >>> print(f.array)
        [999 1 2 3 4]
        >>> f.data
        <CF Data(5): [999, ... 4] kg m-1 s-2>

        """
        self.uncompress(inplace=True)
        return super().varray

    # ----------------------------------------------------------------
    # CF properties
    # ----------------------------------------------------------------
    @property
    def flag_values(self):
        """The flag_values CF property.

        Provides a list of the flag values. Use in conjunction with
        `flag_meanings`. See http://cfconventions.org/latest.html for
        details.

        Stored as a 1-d numpy array but may be set as any array-like
        object.

        **Examples:**

        >>> f.flag_values = ['a', 'b', 'c']
        >>> f.flag_values
        array(['a', 'b', 'c'], dtype='|S1')
        >>> f.flag_values = numpy.arange(4)
        >>> f.flag_values
        array([1, 2, 3, 4])
        >>> del f.flag_values

        >>> f.set_property('flag_values', 1)
        >>> f.get_property('flag_values')
        array([1])
        >>> f.del_property('flag_values')

        """
        try:
            return self.Flags.flag_values
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_values'"
            )

    @flag_values.setter
    def flag_values(self, value):
        try:
            flags = self.Flags
        except AttributeError:
            self.Flags = Flags(flag_values=value)
        else:
            flags.flag_values = value

    @flag_values.deleter
    def flag_values(self):
        try:
            del self.Flags.flag_values
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_values'"
            )
        else:
            if not self.Flags:
                del self.Flags

    @property
    def flag_masks(self):
        """The flag_masks CF property.

        Provides a list of bit fields expressing Boolean or enumerated
        flags. See http://cfconventions.org/latest.html for details.

        Stored as a 1-d numpy array but may be set as array-like object.

        **Examples:**

        >>> f.flag_masks = numpy.array([1, 2, 4], dtype='int8')
        >>> f.flag_masks
        array([1, 2, 4], dtype=int8)
        >>> f.flag_masks = (1, 2, 4, 8)
        >>> f.flag_masks
        array([1, 2, 4, 8], dtype=int8)
        >>> del f.flag_masks

        >>> f.set_property('flag_masks', 1)
        >>> f.get_property('flag_masks')
        array([1])
        >>> f.del_property('flag_masks')

        """
        try:
            return self.Flags.flag_masks
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_masks'"
            )

    @flag_masks.setter
    def flag_masks(self, value):
        try:
            flags = self.Flags
        except AttributeError:
            self.Flags = Flags(flag_masks=value)
        else:
            flags.flag_masks = value

    @flag_masks.deleter
    def flag_masks(self):
        try:
            del self.Flags.flag_masks
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_masks'"
            )
        else:
            if not self.Flags:
                del self.Flags

    @property
    def flag_meanings(self):
        """The flag_meanings CF property.

        Use in conjunction with `flag_values` to provide descriptive words
        or phrases for each flag value. If multi-word phrases are used to
        describe the flag values, then the words within a phrase should be
        connected with underscores. See
        http://cfconventions.org/latest.html for details.

        Stored as a 1-d numpy string array but may be set as a space
        delimited string or any array-like object.

        **Examples:**

        >>> f.flag_meanings = 'low medium      high'
        >>> f.flag_meanings
        array(['low', 'medium', 'high'],
              dtype='|S6')
        >>> del flag_meanings

        >>> f.flag_meanings = ['left', 'right']
        >>> f.flag_meanings
        array(['left', 'right'],
              dtype='|S5')

        >>> f.flag_meanings = 'ok'
        >>> f.flag_meanings
        array(['ok'],
              dtype='|S2')

        >>> f.set_property('flag_meanings', numpy.array(['a', 'b']))
        >>> f.get_property('flag_meanings')
        array(['a', 'b'],
              dtype='|S1')
        >>> f.del_property('flag_meanings')

        """
        try:
            return " ".join(self.Flags.flag_meanings)
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_meanings'"
            )

    @flag_meanings.setter
    def flag_meanings(self, value):
        try:  # TODO deal with space-delimited strings
            flags = self.Flags
        except AttributeError:
            self.Flags = Flags(flag_meanings=value)
        else:
            flags.flag_meanings = value

    @flag_meanings.deleter
    def flag_meanings(self):
        try:
            del self.Flags.flag_meanings
        except AttributeError:
            raise AttributeError(
                f"{self.__class__.__name__!r} doesn't have CF property "
                "'flag_meanings'"
            )
        else:
            if not self.Flags:
                del self.Flags

    @property
    def Conventions(self):
        """The Conventions CF property.

        The name of the conventions followed by the field. See
        http://cfconventions.org/latest.html for details.

        **Examples:**

        >>> f.Conventions = 'CF-1.6'
        >>> f.Conventions
        'CF-1.6'
        >>> del f.Conventions

        >>> f.set_property('Conventions', 'CF-1.6')
        >>> f.get_property('Conventions')
        'CF-1.6'
        >>> f.del_property('Conventions')

        """
        return self.get_property("Conventions")

    @Conventions.setter
    def Conventions(self, value):
        self.set_property("Conventions", value, copy=False)

    @Conventions.deleter
    def Conventions(self):
        self.del_property("Conventions")

    @property
    def featureType(self):
        """The featureType CF property.

        The type of discrete sampling geometry, such as ``point`` or
        ``timeSeriesProfile``. See http://cfconventions.org/latest.html
        for details.

        .. versionadded:: 2.0

        **Examples:**

        >>> f.featureType = 'trajectoryProfile'
        >>> f.featureType
        'trajectoryProfile'
        >>> del f.featureType

        >>> f.set_property('featureType', 'profile')
        >>> f.get_property('featureType')
        'profile'
        >>> f.del_property('featureType')

        """
        return self.get_property("featureType")

    @featureType.setter
    def featureType(self, value):
        self.set_property("featureType", value, copy=False)

    @featureType.deleter
    def featureType(self):
        self.del_property("featureType")

    @property
    def institution(self):
        """The institution CF property.

        Specifies where the original data was produced. See
        http://cfconventions.org/latest.html for details.

        **Examples:**

        >>> f.institution = 'University of Reading'
        >>> f.institution
        'University of Reading'
        >>> del f.institution

        >>> f.set_property('institution', 'University of Reading')
        >>> f.get_property('institution')
        'University of Reading'
        >>> f.del_property('institution')

        """
        return self.get_property("institution")

    @institution.setter
    def institution(self, value):
        self.set_property("institution", value, copy=False)

    @institution.deleter
    def institution(self):
        self.del_property("institution")

    @property
    def references(self):
        """The references CF property.

        Published or web-based references that describe the data or
        methods used to produce it. See
        http://cfconventions.org/latest.html for details.

        **Examples:**

        >>> f.references = 'some references'
        >>> f.references
        'some references'
        >>> del f.references

        >>> f.set_property('references', 'some references')
        >>> f.get_property('references')
        'some references'
        >>> f.del_property('references')

        """
        return self.get_property("references")

    @references.setter
    def references(self, value):
        self.set_property("references", value, copy=False)

    @references.deleter
    def references(self):
        self.del_property("references")

    @property
    def standard_error_multiplier(self):
        """The standard_error_multiplier CF property.

        If a data variable with a `standard_name` modifier of
        ``'standard_error'`` has this attribute, it indicates that the
        values are the stated multiple of one standard error. See
        http://cfconventions.org/latest.html for details.

        **Examples:**

        >>> f.standard_error_multiplier = 2.0
        >>> f.standard_error_multiplier
        2.0
        >>> del f.standard_error_multiplier

        >>> f.set_property('standard_error_multiplier', 2.0)
        >>> f.get_property('standard_error_multiplier')
        2.0
        >>> f.del_property('standard_error_multiplier')

        """
        return self.get_property("standard_error_multiplier")

    @standard_error_multiplier.setter
    def standard_error_multiplier(self, value):
        self.set_property("standard_error_multiplier", value)

    @standard_error_multiplier.deleter
    def standard_error_multiplier(self):
        self.del_property("standard_error_multiplier")

    @property
    def source(self):
        """The source CF property.

        The method of production of the original data. If it was
        model-generated, `source` should name the model and its version,
        as specifically as could be useful. If it is observational,
        `source` should characterize it (for example, ``'surface
        observation'`` or ``'radiosonde'``). See
        http://cfconventions.org/latest.html for details.

        **Examples:**

        >>> f.source = 'radiosonde'
        >>> f.source
        'radiosonde'
        >>> del f.source

        >>> f.set_property('source', 'surface observation')
        >>> f.get_property('source')
        'surface observation'
        >>> f.del_property('source')

        """
        return self.get_property("source")

    @source.setter
    def source(self, value):
        self.set_property("source", value, copy=False)

    @source.deleter
    def source(self):
        self.del_property("source")

    @property
    def title(self):
        """The title CF property.

        A short description of the file contents from which this field was
        read, or is to be written to. See
        http://cfconventions.org/latest.html for details.

        **Examples:**

        >>> f.title = 'model data'
        >>> f.title
        'model data'
        >>> del f.title

        >>> f.set_property('title', 'model data')
        >>> f.get_property('title')
        'model data'
        >>> f.del_property('title')

        """
        return self.get_property("title")

    @title.setter
    def title(self, value):
        self.set_property("title", value, copy=False)

    @title.deleter
    def title(self):
        self.del_property("title")

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def cell_area(
        self,
        radius="earth",
        great_circle=False,
        set=False,
        insert=False,
        force=False,
    ):
        """Return a field containing horizontal cell areas.

        .. versionadded:: 1.0

        .. seealso:: `bin`, `collapse`, `radius`, `weights`

        :Parameters:

            radius: optional
                Specify the radius used for calculating the areas of
                cells defined in spherical polar coordinates. The
                radius is that which would be returned by this call of
                the field construct's `~cf.Field.radius` method:
                ``f.radius(radius)``. See the `cf.Field.radius` for
                details.

                By default *radius* is ``'earth'`` which means that if
                and only if the radius can not found from the datums
                of any coordinate reference constructs, then the
                default radius taken as 6371229 metres.

            great_circle: `bool`, optional
                If True then allow, if required, the derivation of i)
                area weights from polygon geometry cells by assuming
                that each cell part is a spherical polygon composed of
                great circle segments; and ii) and the derivation of
                line-length weights from line geometry cells by
                assuming that each line part is composed of great
                circle segments.

                .. versionadded:: 3.2.0

            insert: deprecated at version 3.0.0

            force: deprecated at version 3.0.0

        :Returns:

            `Field`
                A field construct containing the horizontal cell
                areas.

        **Examples:**

        >>> a = f.cell_area()
        >>> a = f.cell_area(radius=cf.Data(3389.5, 'km'))
        >>> a = f.cell_area(insert=True)

        """
        if insert:
            _DEPRECATION_ERROR_KWARGS(
                self, "cell_area", {"insert": insert}, version="3.0.0"
            )  # pragma: no cover

        if force:
            _DEPRECATION_ERROR_KWARGS(
                self, "cell_area", {"force": force}, version="3.0.0"
            )  # pragma: no cover

        w = self.weights(
            "area",
            radius=radius,
            measure=True,
            scale=None,
            great_circle=great_circle,
        )

        w.set_property("standard_name", "cell_area", copy=False)

        return w

    def radius(self, default=None):
        """Return the radius used for calculating cell areas in
        spherical polar coordinates.

        The radius is taken from the datums of any coordinate
        reference constructs, but if and only if this is not possible
        then a default value may be used instead.

        .. versionadded:: 3.0.2

        .. seealso:: `bin`, `cell_area`, `collapse`, `weights`

        :Parameters:

            default: optional
                The radius is taken from the datums of any coordinate
                reference constructs, but if and only if this is not
                possible then the value set by the *default* parameter
                is used. May be set to any numeric scalar object,
                including `numpy` and `Data` objects. The units of the
                radius are assumed to be metres, unless specified by a
                `Data` object. If the special value ``'earth'`` is
                given then the default radius taken as 6371229
                metres. If *default* is `None` an exception will be
                raised if no unique datum can be found in the
                coordinate reference constructs.

                *Parameter example:*
                  Five equivalent ways to set a default radius of 6371200
                  metres: ``default=6371200``,
                  ``default=numpy.array(6371200)``,
                  ``default=cf.Data(6371200)``, ``default=cf.Data(6371200,
                  'm')``, ``default=cf.Data(6371.2, 'km')``.

        :Returns:

            `Data`
                The radius of the sphere, in units of metres.

        **Examples:**

        >>> f.radius()
        <CF Data(): 6371178.98 m>

        >>> g.radius()
        ValueError: No radius found in coordinate reference constructs and no default provided
        >>> g.radius('earth')
        <CF Data(): 6371229.0 m>
        >>> g.radius(1234)
        <CF Data(): 1234.0 m>

        """
        radii = []
        for cr in self.coordinate_references(todict=True).values():
            r = cr.datum.get_parameter("earth_radius", None)
            if r is not None:
                r = Data.asdata(r)
                if not r.Units:
                    r.override_units("m", inplace=True)

                if r.size != 1:
                    radii.append(r)
                    continue

                got = False
                for _ in radii:
                    if r == _:
                        got = True
                        break

                if not got:
                    radii.append(r)

        if len(radii) > 1:
            raise ValueError(
                "Multiple radii found in coordinate reference "
                f"constructs: {radii!r}"
            )

        if not radii:
            if default is None:
                raise ValueError(
                    "No radius found in coordinate reference constructs "
                    "and no default provided"
                )

            if isinstance(default, str):
                if default != "earth":
                    raise ValueError(
                        "The default parameter must be numeric or the "
                        "string 'earth'"
                    )

                return _earth_radius.copy()

            r = Data.asdata(default).squeeze()
        else:
            r = Data.asdata(radii[0]).squeeze()

        if r.size != 1:
            raise ValueError(f"Multiple radii: {r!r}")

        r.Units = Units("m")
        r.dtype = float
        return r

    def map_axes(self, other):
        """Map the axis identifiers of the field to their equivalent
        axis identifiers of another.

        :Parameters:

            other: `Field`

        :Returns:

            `dict`
                A dictionary whose keys are the axis identifiers of the
                field with corresponding values of axis identifiers of the
                of other field.

        **Examples:**

        >>> f.map_axes(g)
        {'dim0': 'dim1',
         'dim1': 'dim0',
         'dim2': 'dim2'}

        """
        s = self.analyse_items()
        t = other.analyse_items()
        id_to_axis1 = t["id_to_axis"]

        out = {}
        for axis, identity in s["axis_to_id"].items():
            if identity in id_to_axis1:
                out[axis] = id_to_axis1[identity]

        return out

    def close(self):
        """Close all files referenced by the field construct.

        Note that a closed file will be automatically reopened if its
        contents are subsequently required.

        :Returns:

            `None`

        **Examples:**

        >>> f.close()

        """
        super().close()

        for construct in self.constructs.filter_by_data(todict=True).values():
            construct.close()

    def iscyclic(self, *identity, **filter_kwargs):
        """Returns True if the specified axis is cyclic.

        .. versionadded:: 1.0

        .. seealso:: `axis`, `cyclic`, `period`, `domain_axis`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique domain axis construct returned by
                ``f.domain_axis(*identity, **filter_kwargs)``. See
                `domain_axis` for details.

        :Returns:

            `bool`
                True if the selected axis is cyclic, otherwise False.

        **Examples:**

        >>> f.iscyclic('X')
        True
        >>> f.iscyclic('latitude')
        False

        >>> x = f.iscyclic('long_name=Latitude')
        >>> x = f.iscyclic('dimensioncoordinate1')
        >>> x = f.iscyclic('domainaxis2')
        >>> x = f.iscyclic('key%domainaxis2')
        >>> x = f.iscyclic('ncdim%y')
        >>> x = f.iscyclic(2)

        """
        axis = self.domain_axis(
            *identity, key=True, default=None, **filter_kwargs
        )
        if axis is None:
            raise ValueError("Can't identify unique domain axis")

        return axis in self.cyclic()

    @classmethod
    def concatenate(cls, fields, axis=0, _preserve=True):
        """Join a sequence of fields together.

        This is different to `cf.aggregate` because it does not account
        for all metadata. For example, it assumes that the axis order is
        the same in each field.

        .. versionadded:: 1.0

        .. seealso:: `cf.aggregate`, `Data.concatenate`

        :Parameters:

            fields: `FieldList`
                The sequence of fields to concatenate.

            axis: `int`, optional
                The axis along which the arrays will be joined. The
                default is 0. Note that scalar arrays are treated as if
                they were one dimensional.

        :Returns:

            `Field`
                The field generated from the concatenation of input fields.

        """
        if isinstance(fields, cls):
            return fields.copy()

        field0 = fields[0]
        out = field0.copy()

        if len(fields) == 1:
            return out

        new_data = Data.concatenate(
            [f.get_data(_fill_value=False) for f in fields],
            axis=axis,
            _preserve=_preserve,
        )

        # Change the domain axis size
        dim = out.get_data_axes()[axis]
        out.set_construct(DomainAxis(size=new_data.shape[axis]), key=dim)

        # Insert the concatenated data
        out.set_data(new_data, set_axes=False, copy=False)

        # ------------------------------------------------------------
        # Concatenate constructs with data
        # ------------------------------------------------------------
        for key, construct in field0.constructs.filter_by_data(
            todict=True
        ).items():
            construct_axes = field0.get_data_axes(key)

            if dim not in construct_axes:
                # This construct does not span the concatenating axis
                # in the first field
                continue

            constructs = [construct]
            for f in fields[1:]:
                c = f.constructs.get(key)
                if c is None:
                    # This field does not have this construct
                    constructs = None
                    break

                constructs.append(c)

            if not constructs:
                # Not every field has this construct, so remove it
                # from the output field.
                out.del_construct(key)
                continue

            # Still here? Then try concatenating the constructs from
            # each field.
            try:
                construct = construct.concatenate(
                    constructs,
                    axis=construct_axes.index(dim),
                    _preserve=_preserve,
                )
            except ValueError:
                # Couldn't concatenate this construct, so remove it from
                # the output field.
                out.del_construct(key)
            else:
                # Successfully concatenated this construct, so insert
                # it into the output field.
                out.set_construct(
                    construct, key=key, axes=construct_axes, copy=False
                )

        return out

    def cyclic(
        self, *identity, iscyclic=True, period=None, config={}, **filter_kwargs
    ):
        """Set the cyclicity of an axis.

        .. versionadded:: 1.0

        .. seealso:: `autocyclic`, `domain_axis`, `iscyclic`,
                     `period`, `domain_axis`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique domain axis construct returned by
                ``f.domain_axis(*identity, **filter_kwargs)``. See
                `domain_axis` for details.

            iscyclic: `bool`, optional
                If False then the axis is set to be non-cyclic. By
                default the selected axis is set to be cyclic.

            period: optional
                The period for a dimension coordinate construct which
                spans the selected axis. May be any numeric scalar
                object that can be converted to a `Data` object (which
                includes numpy array and `Data` objects). The absolute
                value of *period* is used. If *period* has units then
                they must be compatible with those of the dimension
                coordinates, otherwise it is assumed to have the same
                units as the dimension coordinates.

            config: `dict`
                Additional parameters for optimizing the
                operation. See the code for details.

                .. versionadded:: 3.9.0

            axes: deprecated at version 3.0.0
                Use the *identity* and **filter_kwargs* parameters
                instead.

        :Returns:

            `set`
                The construct keys of the domain axes which were
                cyclic prior to the new setting, or the current cyclic
                domain axes if no axis was specified.

        **Examples:**

        >>> f.cyclic()
        set()
        >>> f.cyclic('X', period=360)
        set()
        >>> f.cyclic()
        {'domainaxis2'}
        >>> f.cyclic('X', iscyclic=False)
        {'domainaxis2'}
        >>> f.cyclic()
        set()

        """
        if not iscyclic and config.get("no-op"):
            return self._cyclic.copy()

        old = None
        cyclic = self._cyclic

        if not identity and not filter_kwargs:
            return cyclic.copy()

        axis = config.get("axis")
        if axis is None:
            axis = self.domain_axis(*identity, key=True, **filter_kwargs)

        data = self.get_data(None, _fill_value=False)
        if data is not None:
            try:
                data_axes = self.get_data_axes()
                data.cyclic(data_axes.index(axis), iscyclic)
            except ValueError:
                pass

        if iscyclic:
            dim = config.get("coord")
            if dim is None:
                dim = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )

            if dim is not None:
                if config.get("period") is not None:
                    dim.period(**config)
                elif period is not None:
                    dim.period(period, **config)
                elif dim.period() is None:
                    raise ValueError(
                        "A cyclic dimension coordinate must have a period"
                    )

            if axis not in cyclic:
                # Never change _cyclic in-place
                old = cyclic.copy()
                cyclic = cyclic.copy()
                cyclic.add(axis)
                self._cyclic = cyclic

        elif axis in cyclic:
            # Never change _cyclic in-place
            old = cyclic.copy()
            cyclic = cyclic.copy()
            cyclic.discard(axis)
            self._cyclic = cyclic

        if old is None:
            old = cyclic.copy()

        return old

    def has_construct(self, *identity, **filter_kwargs):
        """Whether a metadata construct exists.

        .. versionadded:: 3.4.0

        .. seealso:: `construct`, `del_construct`, `get_construct`,
                     `set_construct`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique construct returned by
                ``f.construct(*identity, **filter_kwargs)``. See
                `construct` for details.

        :Returns:

            `bool`
                `True` if the construct exists, otherwise `False`.

        **Examples:**

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        >>> f.has_construct('T')
        True
        >>> f.has_construct('longitude')
        True
        >>> f.has_construct('Z')
        False

        """
        return (
            self.construct(*identity, default=None, **filter_kwargs)
            is not None
        )

    def histogram(self, digitized):
        """Return a multi-dimensional histogram of the data.

        **This has moved to** `cf.histogram`

        """
        raise RuntimeError("Use cf.histogram instead.")

    @_inplace_enabled(default=False)
    def insert_dimension(self, axis, position=0, inplace=False):
        """Insert a size 1 axis into the data array.

        .. versionadded:: 3.0.0

        .. seealso:: `domain_axis`, `flatten`, `flip`, `squeeze`,
                     `transpose`, `unsqueeze`

        :Parameters:

            axis:
                Select the domain axis to insert, generally defined by that
                which would be selected by passing the given axis description
                to a call of the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected.

                If *axis* is `None` then a new domain axis construct will
                created for the inserted dimension.

            position: `int`, optional
                Specify the position that the new axis will have in the
                data array. By default the new axis has position 0, the
                slowest varying position.

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`
                The field construct with expanded data, or `None` if the
                operation was in-place.

        **Examples:**

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> g = f.insert_dimension('T', 0)
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(time(1), latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        A previously non-existent size 1 axis must be created prior to
        insertion:

        >>> f.insert_dimension(None, 1, inplace=True)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(time(1), key%domainaxis3(1), latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]

        """
        f = _inplace_enabled_define_and_cleanup(self)

        if axis is None:
            axis = f.set_construct(self._DomainAxis(1))
        else:
            axis = f.domain_axis(
                axis,
                key=True,
                default=ValueError("Can't identify a unique axis to insert"),
            )

        # Expand the dims in the field construct's data array
        super(Field, f).insert_dimension(
            axis=axis, position=position, inplace=True
        )

        return f

    def indices(self, *mode, **kwargs):
        """Create indices that define a subspace of the field construct.

        The subspace is defined by identifying indices based on the
        metadata constructs.

        Metadata constructs are selected conditions are specified on their
        data. Indices for subspacing are then automatically inferred from
        where the conditions are met.

        The returned tuple of indices may be used to created a subspace by
        indexing the original field construct with them.

        Metadata constructs and the conditions on their data are defined
        by keyword parameters.

        * Any domain axes that have not been identified remain unchanged.

        * Multiple domain axes may be subspaced simultaneously, and it
          doesn't matter which order they are specified in.

        * Subspace criteria may be provided for size 1 domain axes that
          are not spanned by the field construct's data.

        * Explicit indices may also be assigned to a domain axis
          identified by a metadata construct, with either a Python `slice`
          object, or a sequence of integers or booleans.

        * For a dimension that is cyclic, a subspace defined by a slice or
          by a `Query` instance is assumed to "wrap" around the edges of
          the data.

        * Conditions may also be applied to multi-dimensional metadata
          constructs. The "compress" mode is still the default mode (see
          the positional arguments), but because the indices may not be
          acting along orthogonal dimensions, some missing data may still
          need to be inserted into the field construct's data.

        **Auxiliary masks**

        When creating an actual subspace with the indices, if the first
        element of the tuple of indices is ``'mask'`` then the extent of
        the subspace is defined only by the values of elements three and
        onwards. In this case the second element contains an "auxiliary"
        data mask that is applied to the subspace after its initial
        creation, in order to set unselected locations to missing data.

        .. seealso:: `subspace`, `where`, `__getitem__`, `__setitem__`

        :Parameters:

            mode: `str`, *optional*
                There are three modes of operation, each of which provides
                indices for a different type of subspace:

                ==============  ==========================================
                *mode*          Description
                ==============  ==========================================
                ``'compress'``  This is the default mode. Unselected
                                locations are removed to create the
                                returned subspace. Note that if a
                                multi-dimensional metadata construct is
                                being used to define the indices then some
                                missing data may still be inserted at
                                unselected locations.

                ``'envelope'``  The returned subspace is the smallest that
                                contains all of the selected
                                indices. Missing data is inserted at
                                unselected locations within the envelope.

                ``'full'``      The returned subspace has the same domain
                                as the original field construct. Missing
                                data is inserted at unselected locations.
                ==============  ==========================================

            kwargs: *optional*
                A keyword name is an identity of a metadata construct, and
                the keyword value provides a condition for inferring
                indices that apply to the dimension (or dimensions)
                spanned by the metadata construct's data. Indices are
                created that select every location for which the metadata
                construct's data satisfies the condition.

        :Returns:

            `tuple`
                The indices meeting the conditions.

        **Examples:**

        >>> q = cf.example_field(0)
        >>> print(q)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> indices = q.indices(X=112.5)
        >>> print(indices)
        (slice(0, 5, 1), slice(2, 3, 1))
        >>> q[indices]
        <CF Field: specific_humidity(latitude(5), longitude(1)) 1>
        >>> q.indices(X=112.5, latitude=cf.gt(-60))
        (slice(1, 5, 1), slice(2, 3, 1))
        >>> q.indices(latitude=cf.eq(-45) | cf.ge(20))
        (array([1, 3, 4]), slice(0, 8, 1))
        >>> q.indices(X=[1, 2, 4], Y=slice(None, None, -1))
        (slice(4, None, -1), array([1, 2, 4]))
        >>> q.indices(X=cf.wi(-100, 200))
        (slice(0, 5, 1), slice(-2, 4, 1))
        >>> q.indices(X=slice(-2, 4))
        (slice(0, 5, 1), slice(-2, 4, 1))
        >>> q.indices('compress', X=[1, 2, 4, 6])
        (slice(0, 5, 1), array([1, 2, 4, 6]))
        >>> q.indices(Y=[True, False, True, True, False])
        (array([0, 2, 3]), slice(0, 8, 1))
        >>> q.indices('envelope', X=[1, 2, 4, 6])
        ('mask', [<CF Data(1, 6): [[False, ..., False]]>], slice(0, 5, 1), slice(1, 7, 1))
        >>> indices = q.indices('full', X=[1, 2, 4, 6])
        ('mask', [<CF Data(1, 8): [[True, ..., True]]>], slice(0, 5, 1), slice(0, 8, 1))
        >>> print(indices)
        >>> print(q)
        <CF Field: specific_humidity(latitude(5), longitude(8)) 1>

        >>> print(a)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(120), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(120) = [1959-12-16 12:00:00, ..., 1969-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> a.indices(T=410.5)
        (slice(2, 3, 1), slice(0, 5, 1), slice(0, 8, 1))
        >>> a.indices(T=cf.dt('1960-04-16'))
        (slice(4, 5, 1), slice(0, 5, 1), slice(0, 8, 1))
        >>> indices = a.indices(T=cf.wi(cf.dt('1962-11-01'),
        ...                             cf.dt('1967-03-17 07:30')))
        >>> print(indices)
        (slice(35, 88, 1), slice(0, 5, 1), slice(0, 8, 1))
        >>> a[indices]
        <CF Field: air_potential_temperature(time(53), latitude(5), longitude(8)) K>

        >>> print(t)
        Field: air_temperature (ncvar%ta)
        ---------------------------------
        Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K
        Cell methods    : grid_latitude(10): grid_longitude(9): mean where land (interval: 0.1 degrees) time(1): maximum
        Field ancils    : air_temperature standard_error(grid_latitude(10), grid_longitude(9)) = [[0.76, ..., 0.32]] K
        Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                        : grid_latitude(10) = [2.2, ..., -1.76] degrees
                        : grid_longitude(9) = [-4.7, ..., -1.18] degrees
                        : time(1) = [2019-01-01 00:00:00]
        Auxiliary coords: latitude(grid_latitude(10), grid_longitude(9)) = [[53.941, ..., 50.225]] degrees_N
                        : longitude(grid_longitude(9), grid_latitude(10)) = [[2.004, ..., 8.156]] degrees_E
                        : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., b'kappa']
        Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
        >>> indices = t.indices(latitude=cf.wi(51, 53))
        >>> print(indices)
        ('mask', [<CF Data(1, 5, 9): [[[False, ..., False]]]>], slice(0, 1, 1), slice(3, 8, 1), slice(0, 9, 1))
        >>> t[indices]
        <CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(5), grid_longitude(9)) K>

        """
        if "exact" in mode:
            _DEPRECATION_ERROR_ARG(
                self,
                "indices",
                "exact",
                "Keywords are now never interpreted as regular expressions.",
            )  # pragma: no cover

        if len(mode) > 1:
            raise ValueError(
                "Can't provide more than one positional argument. "
                f"Got: {', '.join(repr(x) for x in mode)}"
            )

        if not mode or "compress" in mode:
            mode = "compress"
        elif "envelope" in mode:
            mode = "envelope"
        elif "full" in mode:
            mode = "full"
        else:
            raise ValueError(f"Invalid value for 'mode' argument: {mode[0]!r}")

        data_axes = self.get_data_axes()

        # ------------------------------------------------------------
        # Get the indices for every domain axis in the domain,
        # including any auxiliary masks
        # ------------------------------------------------------------
        domain_indices = self._indices(mode, data_axes, True, **kwargs)

        # Initialise the output indices with any auxiliary masks
        auxiliary_mask = domain_indices["mask"]
        if auxiliary_mask:
            # Ensure that each auxiliary mask is broadcastable to the
            # data
            masks = []
            for axes, mask in auxiliary_mask.items():
                axes = list(axes)
                for i, axis in enumerate(data_axes):
                    if axis not in axes:
                        axes.insert(0, axis)
                        mask.insert_dimension(0, inplace=True)

                new_order = [axes.index(axis) for axis in data_axes]
                mask.transpose(new_order, inplace=True)
                masks.append(mask)

            indices = ["mask", tuple(masks)]
        else:
            indices = []

        # Add the indices that apply to the field's data dimensions
        indices.extend([domain_indices["indices"][axis] for axis in data_axes])

        return tuple(indices)

        # iiiiiiiiiiiiiiiiiiii

        if "exact" in mode:
            _DEPRECATION_ERROR_ARG(
                self,
                "indices",
                "exact",
                "Keywords are now never interpreted as regular expressions.",
            )  # pragma: no cover

        if len(mode) > 1:
            raise ValueError(
                "Can't provide more than one positional argument."
            )

        envelope = "envelope" in mode
        full = "full" in mode
        compress = "compress" in mode or not (envelope or full)

        logger.debug("Field.indices:")  # pragma: no cover
        logger.debug(
            "    envelope, full, compress = {} {} {}".format(
                envelope, full, compress
            )
        )  # pragma: no cover

        auxiliary_mask = []

        data_axes = self.get_data_axes()

        # Initialize indices
        indices = [slice(None)] * self.ndim

        domain_axes = self.domain_axes(todict=True)

        parsed = {}
        unique_axes = set()
        n_axes = 0
        for identity, value in kwargs.items():
            if identity in domain_axes:
                axes = (identity,)
                key = None
                construct = None
            else:
                key, construct = self.construct(
                    identity,
                    filter_by_data=True,
                    item=True,
                    default=(None, None),
                )
                if construct is None:
                    raise ValueError(
                        "Can't find indices: Ambiguous axis or axes: "
                        f"{identity!r}"
                    )

                axes = self.get_data_axes(key)

            sorted_axes = tuple(sorted(axes))
            if sorted_axes not in parsed:
                n_axes += len(sorted_axes)

            parsed.setdefault(sorted_axes, []).append(
                (axes, key, construct, value)
            )

            unique_axes.update(sorted_axes)

        if len(unique_axes) < n_axes:
            raise ValueError(
                "Can't find indices: Multiple constructs with incompatible "
                "domain axes"
            )

        for sorted_axes, axes_key_construct_value in parsed.items():
            axes, keys, constructs, points = list(
                zip(*axes_key_construct_value)
            )
            n_items = len(constructs)
            n_axes = len(sorted_axes)

            if n_items > n_axes:
                if n_axes == 1:
                    a = "axis"
                else:
                    a = "axes"

                raise ValueError(
                    "Error: Can't specify {} conditions for {} {}: {}".format(
                        n_items, n_axes, a, points
                    )
                )

            create_mask = False

            item_axes = axes[0]

            logger.debug(
                "    item_axes = {!r}".format(item_axes)
            )  # pragma: no cover
            logger.debug(
                "    keys      = {!r}".format(keys)
            )  # pragma: no cover

            if n_axes == 1:
                # ----------------------------------------------------
                # 1-d construct
                # ----------------------------------------------------
                ind = None

                logger.debug(
                    "    {} 1-d constructs: {!r}".format(n_items, constructs)
                )  # pragma: no cover

                axis = item_axes[0]
                item = constructs[0]
                value = points[0]

                logger.debug(
                    "    axis      = {!r}".format(axis)
                )  # pragma: no cover
                logger.debug(
                    "    value     = {!r}".format(value)
                )  # pragma: no cover

                if isinstance(value, (list, slice, tuple, numpy_ndarray)):
                    # ------------------------------------------------
                    # 1-dimensional CASE 1: Value is already an index,
                    #                       e.g. [0], (0,3),
                    #                       slice(0,4,2),
                    #                       numpy.array([2,4,7]),
                    #                       [True, False, True]
                    # -------------------------------------------------
                    logger.debug("    1-d CASE 1: ")  # pragma: no cover

                    index = value

                    if envelope or full:
                        size = self.constructs[axis].get_size()
                        d = Data(list(range(size)))
                        ind = (d[value].array,)
                        index = slice(None)

                elif (
                    item is not None
                    and isinstance(value, Query)
                    and value.operator in ("wi", "wo")
                    and item.construct_type == "dimension_coordinate"
                    and self.iscyclic(axis)
                ):
                    # self.iscyclic(sorted_axes)):
                    # ------------------------------------------------
                    # 1-dimensional CASE 2: Axis is cyclic and
                    #                       subspace criterion is a
                    #                       'within' or 'without'
                    #                       Query instance
                    # -------------------------------------------------
                    logger.debug("    1-d CASE 2: ")  # pragma: no cover

                    if item.increasing:
                        anchor0 = value.value[0]
                        anchor1 = value.value[1]
                    else:
                        anchor0 = value.value[1]
                        anchor1 = value.value[0]

                    a = self.anchor(axis, anchor0, dry_run=True)["roll"]
                    b = self.flip(axis).anchor(axis, anchor1, dry_run=True)[
                        "roll"
                    ]

                    size = item.size
                    if abs(anchor1 - anchor0) >= item.period():
                        if value.operator == "wo":
                            set_start_stop = 0
                        else:
                            set_start_stop = -a

                        start = set_start_stop
                        stop = set_start_stop
                    elif a + b == size:
                        b = self.anchor(axis, anchor1, dry_run=True)["roll"]
                        if (b == a and value.operator == "wo") or not (
                            b == a or value.operator == "wo"
                        ):
                            set_start_stop = -a
                        else:
                            set_start_stop = 0

                        start = set_start_stop
                        stop = set_start_stop
                    else:
                        if value.operator == "wo":
                            start = b - size
                            stop = -a + size
                        else:
                            start = -a
                            stop = b - size

                    index = slice(start, stop, 1)

                    if full:
                        # index = slice(start, start+size, 1)
                        d = Data(list(range(size)))
                        d.cyclic(0)
                        ind = (d[index].array,)

                        index = slice(None)

                elif item is not None:
                    # -------------------------------------------------
                    # 1-dimensional CASE 3: All other 1-d cases
                    # -------------------------------------------------
                    logger.debug("    1-d CASE 3:")  # pragma: no cover

                    item_match = value == item

                    if not item_match.any():
                        raise ValueError(
                            "No {!r} axis indices found from: {}".format(
                                identity, value
                            )
                        )

                    index = numpy_asanyarray(item_match)

                    if envelope or full:
                        if numpy_ma_isMA(index):
                            ind = numpy_ma_where(index)
                        else:
                            ind = numpy_where(index)

                        index = slice(None)

                else:
                    raise ValueError(
                        "Must specify a domain axis construct or a construct "
                        "with data for which to create indices"
                    )

                logger.debug(
                    "    index = {}".format(index)
                )  # pragma: no cover

                # Put the index into the correct place in the list of
                # indices.
                #
                # Note that we might overwrite it later if there's an
                # auxiliary mask for this axis.
                if axis in data_axes:
                    indices[data_axes.index(axis)] = index

            else:
                # -----------------------------------------------------
                # N-dimensional constructs
                # -----------------------------------------------------
                logger.debug(
                    "    {} N-d constructs: {!r}".format(n_items, constructs)
                )  # pragma: no cover
                logger.debug(
                    "    {} points        : {!r}".format(len(points), points)
                )  # pragma: no cover
                logger.debug(
                    "    field.shape     : {}".format(self.shape)
                )  # pragma: no cover

                # Make sure that each N-d item has the same relative
                # axis order as the field's data array.
                #
                # For example, if the data array of the field is
                # ordered T Z Y X and the item is ordered Y T then the
                # item is transposed so that it is ordered T Y. For
                # example, if the field's data array is ordered Z Y X
                # and the item is ordered X Y T (T is size 1) then
                # transpose the item so that it is ordered Y X T.
                g = self.transpose(data_axes, constructs=True)

                #                g = self
                #                data_axes = .get_data_axes(default=None)
                #                for item_axes2 in axes:
                #                    if item_axes2 != data_axes:
                #                        g = self.transpose(data_axes, constructs=True)
                #                        break

                item_axes = g.get_data_axes(keys[0])

                constructs = [g.constructs[key] for key in keys]
                logger.debug(
                    "    transposed N-d constructs: {!r}".format(constructs)
                )  # pragma: no cover

                item_matches = [
                    (value == construct).data
                    for value, construct in zip(points, constructs)
                ]

                item_match = item_matches.pop()

                for m in item_matches:
                    item_match &= m

                item_match = item_match.array  # LAMA alert

                if numpy_ma_isMA:
                    ind = numpy_ma_where(item_match)
                else:
                    ind = numpy_where(item_match)

                logger.debug(
                    "    item_match  = {}".format(item_match)
                )  # pragma: no cover
                logger.debug(
                    "    ind         = {}".format(ind)
                )  # pragma: no cover

                bounds = [
                    item.bounds.array[ind]
                    for item in constructs
                    if item.has_bounds()
                ]

                contains = False
                if bounds:
                    points2 = []
                    for v, construct in zip(points, constructs):
                        if isinstance(v, Query):
                            if v.operator == "contains":
                                contains = True
                                v = v.value
                            elif v.operator == "eq":
                                v = v.value
                            else:
                                contains = False
                                break

                        v = Data.asdata(v)
                        if v.Units:
                            v.Units = construct.Units

                        points2.append(v.datum())

                if contains:
                    # The coordinates have bounds and the condition is
                    # a 'contains' Query object. Check each
                    # potentially matching cell for actually including
                    # the point.
                    try:
                        Path
                    except NameError:
                        raise ImportError(
                            "Must install matplotlib to create indices based "
                            "on {}-d constructs and a 'contains' Query "
                            "object".format(constructs[0].ndim)
                        )

                    if n_items != 2:
                        raise ValueError(
                            "Can't index for cell from {}-d coordinate "
                            "objects".format(n_axes)
                        )

                    if 0 < len(bounds) < n_items:
                        raise ValueError("bounds alskdaskds TODO")

                    # Remove grid cells if, upon closer inspection,
                    # they do actually contain the point.
                    delete = [
                        n
                        for n, vertices in enumerate(zip(*zip(*bounds)))
                        if not Path(zip(*vertices)).contains_point(points2)
                    ]

                    if delete:
                        ind = [numpy_delete(ind_1d, delete) for ind_1d in ind]

            if ind is not None:
                mask_shape = [None] * self.ndim
                masked_subspace_size = 1
                ind = numpy_array(ind)
                logger.debug("    ind = {}".format(ind))  # pragma: no cover

                for i, (axis, start, stop) in enumerate(
                    zip(item_axes, ind.min(axis=1), ind.max(axis=1))
                ):
                    if axis not in data_axes:
                        continue

                    position = data_axes.index(axis)

                    if indices[position] == slice(None):
                        if compress:
                            # Create a compressed index for this axis
                            size = stop - start + 1
                            index = sorted(set(ind[i]))
                        elif envelope:
                            # Create an envelope index for this axis
                            stop += 1
                            size = stop - start
                            index = slice(start, stop)
                        elif full:
                            # Create a full index for this axis
                            start = 0
                            #                            stop = self.axis_size(axis)
                            stop = domain_axes[axis].get_size()
                            size = stop - start
                            index = slice(start, stop)
                        else:
                            raise ValueError(
                                "Must have full, envelope or compress"
                            )  # pragma: no cover

                        indices[position] = index

                    mask_shape[position] = size
                    masked_subspace_size *= size
                    ind[i] -= start

                create_mask = ind.shape[1] < masked_subspace_size
            else:
                create_mask = False

            # --------------------------------------------------------
            # Create an auxiliary mask for these axes
            # --------------------------------------------------------
            logger.debug(
                "    create_mask = {}".format(create_mask)
            )  # pragma: no cover

            if create_mask:
                logger.debug(
                    "    mask_shape  = {}".format(mask_shape)
                )  # pragma: no cover

                mask = self.data._create_auxiliary_mask_component(
                    mask_shape, ind, compress
                )
                auxiliary_mask.append(mask)
                logger.debug(
                    "    mask_shape  = {}".format(mask_shape)
                )  # pragma: no cover
                logger.debug(
                    "    mask.shape  = {}".format(mask.shape)
                )  # pragma: no cover

        indices = tuple(parse_indices(self.shape, tuple(indices)))

        if auxiliary_mask:
            indices = ("mask", auxiliary_mask) + indices

            logger.debug(
                "    Final indices = {}".format(indices)
            )  # pragma: no cover

        # Return the tuple of indices and the auxiliary mask (which
        # may be None)
        return indices

    @_inplace_enabled(default=True)
    def set_data(
        self, data, axes=None, set_axes=True, copy=True, inplace=True
    ):
        """Set the field construct data.

        .. versionadded:: 3.0.0

        .. seealso:: `data`, `del_data`, `get_data`, `has_data`,
                     `set_construct`

        :Parameters:

            data: `Data`
                The data to be inserted.

                {{data_like}}

            axes: (sequence of) `str` or `int`, optional
                Set the domain axes constructs that are spanned by the
                data. If unset, and the *set_axes* parameter is True, then
                an attempt will be made to assign existing domain axis
                constructs to the data.

                The contents of the *axes* parameter is mapped to domain
                axis constructs by translating each element into a domain
                axis construct key via the `domain_axis` method.

                *Parameter example:*
                  ``axes='domainaxis1'``

                *Parameter example:*
                  ``axes='X'``

                *Parameter example:*
                  ``axes=['latitude']``

                *Parameter example:*
                  ``axes=['X', 'longitude']``

                *Parameter example:*
                  ``axes=[1, 0]``

            set_axes: `bool`, optional
                If False then do not set the domain axes constructs that
                are spanned by the data, even if the *axes* parameter has
                been set. By default the axes are set either according to
                the *axes* parameter, or if any domain axis constructs
                exist then an attempt will be made to assign existing
                domain axis constructs to the data.

                If the *axes* parameter is `None` and no domain axis
                constructs exist then no attempt is made to assign domain
                axes constructs to the data, regardless of the value of
                *set_axes*.

            copy: `bool`, optional
                If True then set a copy of the data. By default the data
                are copied.

            {{inplace: `bool`, optional (default True)}}

                .. versionadded:: 3.7.0

        :Returns:

            `None` or `Field`
                If the operation was in-place then `None` is returned,
                otherwise return a new `Field` instance containing the new
                data.

        **Examples:**

        >>> f = cf.Field()
        >>> f.set_data([1, 2, 3])
        >>> f.has_data()
        True
        >>> f.get_data()
        <CF Data(3): [1, 2, 3]>
        >>> f.data
        <CF Data(3): [1, 2, 3]>
        >>> f.del_data()
        <CF Data(3): [1, 2, 3]>
        >>> g = f.set_data([4, 5, 6], inplace=False)
        >>> g.data
        <CF Data(3): [4, 5, 6]>
        >>> f.has_data()
        False
        >>> print(f.get_data(None))
        None
        >>> print(f.del_data(None))
        None

        """
        data = self._Data(data, copy=False)

        # Construct new field
        f = _inplace_enabled_define_and_cleanup(self)

        domain_axes = f.domain_axes(todict=True)
        if axes is None and not domain_axes:
            set_axes = False

        if not set_axes:
            if not data.Units:
                units = getattr(f, "Units", None)
                if units is not None:
                    if copy:
                        copy = False
                        data = data.override_units(units, inplace=False)
                    else:
                        data.override_units(units, inplace=True)

            super(cfdm.Field, f).set_data(
                data, axes=None, copy=copy, inplace=True
            )

            return f

        if data.isscalar:
            # --------------------------------------------------------
            # The data array is scalar
            # --------------------------------------------------------
            if axes or axes == 0:
                raise ValueError(
                    "Can't set data: Wrong number of axes for scalar data "
                    f"array: axes={axes}"
                )

            axes = []

        elif axes is not None:
            # --------------------------------------------------------
            # Axes have been set
            # --------------------------------------------------------
            if isinstance(axes, (str, int, slice)):
                axes = (axes,)

            axes = [f.domain_axis(axis, key=True) for axis in axes]

            if len(axes) != data.ndim:
                raise ValueError(
                    "Can't set data: {} axes provided, but {} needed".format(
                        len(axes), data.ndim
                    )
                )

            for axis, size in zip(axes, data.shape):
                axis_size = domain_axes[axis].get_size(None)
                if size != axis_size:
                    axes_shape = tuple(
                        domain_axes[axis].get_size(None) for axis in axes
                    )
                    raise ValueError(
                        f"Can't set data: Data shape {data.shape} differs "
                        f"from shape implied by axes {axes}: {axes_shape}"
                    )

        elif f.get_data_axes(default=None) is None:
            # --------------------------------------------------------
            # The data is not scalar and axes have not been set and
            # the domain does not have data axes defined
            #
            # => infer the axes
            # --------------------------------------------------------
            data_shape = data.shape
            if len(data_shape) != len(set(data_shape)):
                raise ValueError(
                    f"Can't insert data: Ambiguous data shape: {data_shape}. "
                    "Consider setting the axes parameter."
                )

            if not domain_axes:
                raise ValueError("Can't set data: No domain axes exist")

            axes = []
            for n in data_shape:
                da_key = f.domain_axis(
                    filter_by_size=(n,), key=True, default=None
                )
                if da_key is None:
                    raise ValueError(
                        "Can't insert data: Ambiguous data shape: "
                        f"{data_shape}. Consider setting the axes parameter."
                    )

                axes.append(da_key)

        else:
            # --------------------------------------------------------
            # The data is not scalar and axes have not been set, but
            # there are data axes defined on the field.
            # --------------------------------------------------------
            axes = f.get_data_axes()
            if len(axes) != data.ndim:
                raise ValueError(
                    f"Wrong number of axes for data array: {axes!r}"
                )

            for axis, size in zip(axes, data.shape):
                if domain_axes[axis].get_size(None) != size:
                    raise ValueError(
                        "Can't insert data: Incompatible size for axis "
                        f"{axis!r}: {size}"
                    )

        if not data.Units:
            units = getattr(f, "Units", None)
            if units is not None:
                if copy:
                    copy = False
                    data = data.override_units(units, inplace=False)
                else:
                    data.override_units(units, inplace=True)

        super(cfdm.Field, f).set_data(data, axes=axes, copy=copy, inplace=True)

        # Apply cyclic axes
        if axes:
            cyclic = self._cyclic
            if cyclic:
                cyclic_axes = [
                    axes.index(axis) for axis in cyclic if axis in axes
                ]
                if cyclic_axes:
                    data.cyclic(cyclic_axes, True)

        return f

    def domain_mask(self, **kwargs):
        """Return a boolean field that is True where criteria are met.

        .. versionadded:: 1.1

        .. seealso:: `indices`, `mask`, `subspace`

        :Parameters:

            kwargs: optional
                A dictionary of keyword arguments to pass to the `indices`
                method to define the criteria to meet for a element to be
                set as `True`.

        :Returns:

            `Field`
                The domain mask.

        **Examples:**

        Create a domain mask which is masked at all between between -30
        and 30 degrees of latitude:

        >>> m = f.domain_mask(latitude=cf.wi(-30, 30))

        """
        mask = self.copy()

        mask.clear_properties()
        mask.nc_del_variable(None)

        for key in self.constructs.filter_by_type(
            "cell_method", "field_ancillary", todict=True
        ):
            mask.del_construct(key)

        false_everywhere = Data.zeros(self.shape, dtype=bool)

        mask.set_data(false_everywhere, axes=self.get_data_axes(), copy=False)

        mask.subspace[mask.indices(**kwargs)] = True

        mask.long_name = "domain mask"

        return mask

    @_inplace_enabled(default=False)
    @_manage_log_level_via_verbosity
    def compute_vertical_coordinates(
        self, default_to_zero=True, strict=True, inplace=False, verbose=None
    ):
        """Compute non-parametric vertical coordinates.

        When vertical coordinates are a function of horizontal location as
        well as parameters which depend on vertical location, they cannot
        be stored in a vertical dimension coordinate construct. In such
        cases a parametric vertical dimension coordinate construct is
        stored and a coordinate reference construct contains the formula
        for computing the required non-parametric vertical coordinates.

        {{formula terms links}}

        For example, multi-dimensional non-parametric parametric ocean
        altitude coordinates can be computed from one-dimensional
        parametric ocean sigma coordinates.

        Coordinate reference systems based on parametric vertical
        coordinates are identified from the coordinate reference
        constructs and, if possible, the corresponding non-parametric
        vertical coordinates are computed and stored in a new auxiliary
        coordinate construct.

        If there are no appropriate coordinate reference constructs then
        the field construct is unchanged.

        .. versionadded:: 3.8.0

        .. seealso:: `CoordinateReference`

        :Parameters:

            {{default_to_zero: `bool`, optional}}

            strict: `bool`
                If False then allow the computation to occur when

                * A domain ancillary construct has no standard name, but
                  the corresponding term has a standard name that is
                  prescribed

                * When the computed standard name can not be found by
                  inference from the standard names of the domain
                  ancillary constructs, nor from the
                  ``computed_standard_name`` parameter of the relevant
                  coordinate reference construct.

                By default an exception is raised in these cases.

                If a domain ancillary construct does have a standard name,
                but one that is inconsistent with any prescribed standard
                names, then an exception is raised regardless of the value
                of *strict*.

            {{inplace: `bool`, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

        :Returns:

            `Field` or `None`
                The field construct with the new non-parametric vertical
                coordinates, or `None` if the operation was in-place.

        **Examples**

        >>> f = cf.example_field(1)
        >>> print(f)
        Field: air_temperature (ncvar%ta)
        ---------------------------------
        Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K
        Cell methods    : grid_latitude(10): grid_longitude(9): mean where land (interval: 0.1 degrees) time(1): maximum
        Field ancils    : air_temperature standard_error(grid_latitude(10), grid_longitude(9)) = [[0.76, ..., 0.32]] K
        Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                        : grid_latitude(10) = [2.2, ..., -1.76] degrees
                        : grid_longitude(9) = [-4.7, ..., -1.18] degrees
                        : time(1) = [2019-01-01 00:00:00]
        Auxiliary coords: latitude(grid_latitude(10), grid_longitude(9)) = [[53.941, ..., 50.225]] degrees_N
                        : longitude(grid_longitude(9), grid_latitude(10)) = [[2.004, ..., 8.156]] degrees_E
                        : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., b'kappa']
        Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
        >>> print(f.auxiliary_coordinate('altitude', default=None))
        None
        >>> g = f.compute_vertical_coordinates()
        >>> print(g.auxiliary_coordinates)
        Constructs:
        {'auxiliarycoordinate0': <CF AuxiliaryCoordinate: latitude(10, 9) degrees_N>,
         'auxiliarycoordinate1': <CF AuxiliaryCoordinate: longitude(9, 10) degrees_E>,
         'auxiliarycoordinate2': <CF AuxiliaryCoordinate: long_name=Grid latitude name(10) >,
         'auxiliarycoordinate3': <CF AuxiliaryCoordinate: altitude(1, 10, 9) m>}
        >>> g.auxiliary_coordinate('altitude').dump()
        Auxiliary coordinate: altitude
            long_name = 'Computed from parametric atmosphere_hybrid_height_coordinate
                         vertical coordinates'
            standard_name = 'altitude'
            units = 'm'
            Data(1, 10, 9) = [[[10.0, ..., 5410.0]]] m
            Bounds:units = 'm'
            Bounds:Data(1, 10, 9, 2) = [[[[5.0, ..., 5415.0]]]] m

        """
        f = _inplace_enabled_define_and_cleanup(self)

        for cr in f.coordinate_references(todict=True).values():
            # --------------------------------------------------------
            # Compute the non-parametric vertical coordinates, if
            # possible.
            # --------------------------------------------------------
            (
                standard_name,
                computed_standard_name,
                computed,
                computed_axes,
                k_axis,
            ) = FormulaTerms.formula(f, cr, default_to_zero, strict)

            if computed is None:
                # No non-parametric vertical coordinates were
                # computed
                continue

            # --------------------------------------------------------
            # Convert the computed domain ancillary construct to an
            # auxiliary coordinate construct, and insert it into the
            # field construct.
            # --------------------------------------------------------
            c = f._AuxiliaryCoordinate(source=computed, copy=False)
            c.clear_properties()
            c.long_name = (
                "Computed from parametric {} "
                "vertical coordinates".format(standard_name)
            )
            if computed_standard_name:
                c.standard_name = computed_standard_name

            logger.detail(
                "Non-parametric coordinates:\n{}".format(
                    c.dump(display=False, _level=1)
                )
            )  # pragma: no cover

            key = f.set_construct(c, axes=computed_axes, copy=False)

            # Reference the new coordinates from the coordinate
            # reference construct
            cr.set_coordinate(key)

            logger.debug(
                "Non-parametric coordinates construct key: {key!r}\n"
                "Updated coordinate reference construct:\n"
                f"{cr.dump(display=False, _level=1)}"
            )  # pragma: no cover

        return f

    def match_by_construct(self, *identities, OR=False, **conditions):
        """Whether or not there are particular metadata constructs.

        .. note:: The API changed at version 3.1.0

        .. versionadded:: 3.0.0

        .. seealso:: `match`, `match_by_property`, `match_by_rank`,
                     `match_by_identity`, `match_by_ncvar`,
                     `match_by_units`, `construct`

        :Parameters:

            identities: optional
                Select the unique construct returned by
                ``f.construct(*identities)``. See `construct` for
                details.

            conditions: optional
                Identify the metadata constructs that have any of the
                given identities or construct keys, and whose data satisfy
                conditions.

                A construct identity or construct key (as defined by the
                *identities* parameter) is given as a keyword name and a
                condition on its data is given as the keyword value.

                The condition is satisfied if any of its data values
                equals the value provided.

                *Parameter example:*
                  ``longitude=180.0``

                *Parameter example:*
                  ``time=cf.dt('1959-12-16')``

                *Parameter example:*
                  ``latitude=cf.ge(0)``

                *Parameter example:*
                  ``latitude=cf.ge(0), air_pressure=500``

                *Parameter example:*
                  ``**{'latitude': cf.ge(0), 'long_name=soil_level': 4}``

            OR: `bool`, optional
                If True then return `True` if at least one metadata
                construct matches at least one of the criteria given by
                the *identities* or *conditions* arguments. By default
                `True` is only returned if the field constructs matches
                each of the given criteria.

            mode: deprecated at version 3.1.0
                Use the *OR* parameter instead.

            constructs: deprecated at version 3.1.0

        :Returns:

            `bool`
                Whether or not the field construct contains the specified
                metadata constructs.

        **Examples:**

            TODO

        """
        if identities:
            if identities[0] == "or":
                _DEPRECATION_ERROR_ARG(
                    self,
                    "match_by_construct",
                    "or",
                    message="Use 'OR=True' instead.",
                    version="3.1.0",
                )  # pragma: no cover

            if identities[0] == "and":
                _DEPRECATION_ERROR_ARG(
                    self,
                    "match_by_construct",
                    "and",
                    message="Use 'OR=False' instead.",
                    version="3.1.0",
                )  # pragma: no cover

        if not identities and not conditions:
            return True

        constructs = self.constructs

        if not constructs:
            return False

        n = 0

        # TODO - replace ().ordered() with (todict=True) when Python
        #        3.6 is deprecated
        self_cell_methods = self.cell_methods().ordered()

        for identity in identities:
            cms = False
            try:
                cms = ": " in identity
            except TypeError:
                cms = False

            if cms:
                cms = CellMethod.create(identity)
                for cm in cms:
                    axes = [
                        self.domain_axis(axis, key=True, default=axis)
                        for axis in cm.get_axes(())
                    ]
                    if axes:
                        cm.set_axes(axes)

            if not cms:
                filtered = constructs(identity)
                if filtered:
                    # Check for cell methods
                    if set(filtered.construct_types().values()) == {
                        "cell_method"
                    }:
                        key = tuple(self_cell_methods)[-1]
                        filtered = self.cell_method(
                            identity, filter_by_key=(key,), default=None
                        )
                        if filtered is None:
                            if not OR:
                                return False

                            n -= 1

                    n += 1
                elif not OR:
                    return False
            else:
                cell_methods = tuple(self_cell_methods.values())[-len(cms) :]
                for cm0, cm1 in zip(cms, cell_methods):
                    if cm0.has_axes() and set(cm0.get_axes()) != set(
                        cm1.get_axes(())
                    ):
                        if not OR:
                            return False

                        n -= 1
                        break

                    if cm0.has_method() and (
                        cm0.get_method() != cm1.get_method(None)
                    ):
                        if not OR:
                            return False

                        n -= 1
                        break

                    ok = True
                    for key, value in cm0.qualifiers():
                        if value != cm1.get_qualifier(key, None):
                            if not OR:
                                return False

                            ok = False
                            break

                    if not ok:
                        n -= 1
                        break

                n += 1

        if conditions:
            for identity, value in conditions.items():
                if self.subspace("test", **{identity: value}):
                    n += 1
                elif not OR:
                    return False

        if OR:
            return bool(n)

        return True

    @_inplace_enabled(default=False)
    def moving_window(
        self,
        method,
        window_size=None,
        axis=None,
        weights=None,
        mode=None,
        cval=None,
        origin=0,
        scale=None,
        radius="earth",
        great_circle=False,
        inplace=False,
    ):
        """Perform moving window calculations along an axis.

        Moving mean, sum, and integral calculations are possible.

        By default moving means are unweighted, but weights based on
        the axis cell sizes (or custom weights) may applied to the
        calculation via the *weights* parameter.

        By default moving integrals must be weighted.

        When appropriate, a new cell method construct is created to
        describe the calculation.

        .. note:: The `moving_window` method can not, in general, be
                  emulated by the `convolution_filter` method, as the
                  latter i) can not change the window weights as the
                  filter passes through the axis; and ii) does not
                  update the cell method constructs.

        .. versionadded:: 3.3.0

        .. seealso:: `bin`, `collapse`, `convolution_filter`, `radius`,
                     `weights`

        :Parameters:

            method: `str`
                Define the moving window method. The method is given
                by one of the following strings (see
                https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
                for precise definitions):

                ==================  ============================  ========
                *method*            Description                   Weighted
                ==================  ============================  ========
                ``'sum'``           The sum of the values.        Never

                ``'mean'``          The weighted or unweighted    May be
                                    mean of the values.

                ``'integral'``      The integral of values.       Always
                ==================  ============================  ========

                * Methods that are "Never" weighted ignore the
                  *weights* parameter, even if it is set.

                * Methods that "May be" weighted will only be weighted
                  if the *weights* parameter is set.

                * Methods that are "Always" weighted require the
                  *weights* parameter to be set.

            window_size: `int`
                Specify the size of the window used to calculate the
                moving window.

                *Parameter example:*
                  A 5-point moving window is set with
                  ``window_size=5``.

            axis: `str` or `int`
                Select the domain axis over which the filter is to be
                applied, defined by that which would be selected by
                passing the given axis description to a call of the
                field construct's `domain_axis` method. For example,
                for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected.

            weights: optional
                Specify the weights for the moving window. The weights
                are, those that would be returned by this call of the
                field construct's `weights` method:
                ``f.weights(weights, axes=axis, radius=radius,
                great_circle=great_circle, data=True)``. See the
                *axis*, *radius* and *great_circle* parameters and
                `cf.Field.weights` for details.

                .. note:: By default *weights* is `None`, resulting in
                          **unweighted calculations**.

                .. note:: Setting *weights* to `True` is generally a
                          good way to ensure that the moving window
                          calculations are appropriately weighted
                          according to the field construct's
                          metadata. In this case, if it is not
                          possible to create weights for the selected
                          *axis* then an exception will be raised.

                *Parameter example:*
                  To specify weights on the cell sizes of the selected
                  axis: ``weights=True``.

            mode: `str`, optional
                The *mode* parameter determines how the input array is
                extended when the filter overlaps an array border. The
                default value is ``'constant'`` or, if the dimension
                being convolved is cyclic (as ascertained by the
                `iscyclic` method), ``'wrap'``. The valid values and
                their behaviours are as follows:

                ==============  ==========================  ===========================
                *mode*          Description                 Behaviour
                ==============  ==========================  ===========================
                ``'reflect'``   The input is extended by    ``(c b a | a b c | c b a)``
                                reflecting about the edge

                ``'constant'``  The input is extended by    ``(k k k | a b c | k k k)``
                                filling all values beyond
                                the edge with the same
                                constant value (``k``),
                                defined by the *cval*
                                parameter.

                ``'nearest'``   The input is extended by    ``(a a a | a b c | c c c)``
                                replicating the last point

                ``'mirror'``    The input is extended by    ``(c b | a b c | b a)``
                                reflecting about the
                                centre of the last point.

                ``'wrap'``      The input is extended by    ``(a b c | a b c | a b c)``
                                wrapping around to the
                                opposite edge.
                ==============  ==========================  ===========================

                The position of the window relative to each value can
                be changed by using the *origin* parameter.

            cval: scalar, optional
                Value to fill past the edges of the array if *mode* is
                ``'constant'``. Ignored for other modes. Defaults to
                `None`, in which case the edges of the array will be
                filled with missing data. The only other valid value
                is ``0``.

                *Parameter example:*
                   To extend the input by filling all values beyond
                   the edge with zero: ``cval=0``

            origin: `int`, optional
                Controls the placement of the filter. Defaults to 0,
                which is the centre of the window. If the window size,
                defined by the *window_size* parameter, is even then
                then a value of 0 defines the index defined by
                ``window_size/2 -1``.

                *Parameter example:*
                  For a window size of 5, if ``origin=0`` then the
                  window is centred on each point. If ``origin=-2``
                  then the window is shifted to include the previous
                  four points. If ``origin=1`` then the window is
                  shifted to include the previous point and the and
                  the next three points.

            radius: optional
                Specify the radius used for calculating the areas of
                cells defined in spherical polar coordinates. The
                radius is that which would be returned by this call of
                the field construct's `~cf.Field.radius` method:
                ``f.radius(radius)``. See the `cf.Field.radius` for
                details.

                By default *radius* is ``'earth'`` which means that if
                and only if the radius can not found from the datums
                of any coordinate reference constructs, then the
                default radius taken as 6371229 metres.

            great_circle: `bool`, optional
                If True then allow, if required, the derivation of i)
                area weights from polygon geometry cells by assuming
                that each cell part is a spherical polygon composed of
                great circle segments; and ii) and the derivation of
                line-length weights from line geometry cells by
                assuming that each line part is composed of great
                circle segments.

            scale: number, optional
                If set to a positive number then scale the weights so
                that they are less than or equal to that number. By
                default the weights are scaled to lie between 0 and 1
                (i.e.  *scale* is 1).

                Ignored if the moving window method is not
                weighted. The *scale* parameter can not be set for
                moving integrals.

                *Parameter example:*
                  To scale all weights so that they lie between 0 and
                  0.5: ``scale=0.5``.

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`
                The field construct of moving window values, or `None`
                if the operation was in-place.

        **Examples:**

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(f.array)
        [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
         [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
         [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
        >>> print(f.coordinate('X').bounds.array)
        [[  0.  45.]
         [ 45.  90.]
         [ 90. 135.]
         [135. 180.]
         [180. 225.]
         [225. 270.]
         [270. 315.]
         [315. 360.]]
        >>> f.iscyclic('X')
        True
        >>> f.iscyclic('Y')
        False

        Create a weighted 3-point running mean for the cyclic 'X'
        axis:

        >>> g = f.moving_window('mean', 3, axis='X', weights=True)
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean longitude(8): mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(g.array)
        [[0.02333 0.01467 0.017   0.01167 0.023   0.02633 0.03    0.02   ]
         [0.04167 0.03467 0.04767 0.051   0.06033 0.04167 0.04833 0.03167]
         [0.084   0.12167 0.13367 0.119   0.112   0.08233 0.057   0.05933]
         [0.035   0.04233 0.056   0.05567 0.06667 0.04633 0.03267 0.01833]
         [0.01833 0.02033 0.03    0.024   0.03    0.02967 0.028   0.01767]]
        >>> print(g.coordinate('X').bounds.array)
        [[-45.  90.]
         [  0. 135.]
         [ 45. 180.]
         [ 90. 225.]
         [135. 270.]
         [180. 315.]
         [225. 360.]
         [270. 405.]]

        Create an unweighted 3-point running mean for the cyclic 'X'
        axis:

        >>> g = f.moving_window('mean', 3, axis='X')

        Create an weighted 4-point running integral for the non-cyclic
        'Y' axis:

        >>> g = f.moving_window('integral', 4, axis='Y', weights=True)
        >>> g.Units
        <Units: 0.0174532925199433 rad>
        >>> print(g.array)
        [[   --    --    --    --   --    --   --   --]
         [ 8.37 11.73 10.05 13.14 8.88 11.64 4.59 4.02]
         [ 8.34 11.79 10.53 13.77 8.88 11.64 4.89 3.54]
         [   --    --    --    --   --    --   --   --]
         [   --    --    --    --   --    --   --   --]]
        >>> print(g.coordinate('Y').bounds.array)
        [[-90.  30.]
         [-90.  60.]
         [-60.  90.]
         [-30.  90.]
         [ 30.  90.]]
        >>> g = f.moving_window('integral', 4, axis='Y', weights=True, cval=0)
        >>> print(g.array)
        [[ 7.5   9.96  8.88 11.04  7.14  9.48  4.32  3.51]
         [ 8.37 11.73 10.05 13.14  8.88 11.64  4.59  4.02]
         [ 8.34 11.79 10.53 13.77  8.88 11.64  4.89  3.54]
         [ 7.65 10.71  9.18 11.91  7.5   9.45  4.71  1.56]
         [ 1.05  2.85  1.74  3.15  2.28  3.27  1.29  0.9 ]]

        """
        method_values = ("mean", "sum", "integral")
        if method not in method_values:
            raise ValueError(
                f"Non-valid 'method' parameter value: {method!r}. "
                f"Expected one of {method_values!r}"
            )

        if cval is not None and cval != 0:
            raise ValueError("The cval parameter must be None or 0")

        window_size = int(window_size)

        # Construct new field
        f = _inplace_enabled_define_and_cleanup(self)

        # Find the axis for the moving window
        axis = f.domain_axis(axis, key=True)
        iaxis = self.get_data_axes().index(axis)

        if method == "sum" or weights is False:
            weights = None

        if method == "integral":
            measure = True
            if weights is None:
                raise ValueError(
                    "Must set weights parameter for 'integral' method"
                )

            if scale is not None:
                raise ValueError(
                    "Can't set the 'scale' parameter for moving integrals"
                )
        else:
            if scale is None:
                scale = 1.0

            measure = False

        if weights is not None:
            if isinstance(weights, Data):
                if weights.ndim > 1:
                    raise ValueError(
                        f"The input weights (shape {weights.shape}) do not "
                        f"match the selected axis (size {f.shape[iaxis]})"
                    )

                if weights.ndim == 1:
                    if weights.shape[0] != f.shape[iaxis]:
                        raise ValueError(
                            f"The input weights (size {weights.size}) do not "
                            f"match the selected axis (size {f.shape[iaxis]})"
                        )

            # Get the data weights
            w = f.weights(
                weights,
                axes=axis,
                measure=measure,
                scale=scale,
                radius=radius,
                great_circle=great_circle,
                data=True,
            )

            # Multiply the field by the (possibly adjusted) weights
            if numpy_can_cast(w.dtype, f.dtype):
                f *= w
            else:
                f = f * w

        # Create the window weights
        window = numpy_full((window_size,), 1.0)
        if weights is None and method == "mean":
            # If there is no data weighting, make sure that the sum of
            # the window weights is 1.
            window /= window.size

        f.convolution_filter(
            window,
            axis=axis,
            mode=mode,
            cval=cval,
            origin=origin,
            update_bounds=True,
            inplace=True,
        )

        if weights is not None and method == "mean":
            # Divide the field by the running sum of the adjusted data
            # weights
            w.convolution_filter(
                window=window,
                axis=iaxis,
                mode=mode,
                cval=0,
                origin=origin,
                inplace=True,
            )
            if numpy_can_cast(w.dtype, f.dtype):
                f /= w
            else:
                f = f / w

        # Add a cell method
        if f.domain_axis(axis).get_size() > 1 or method == "integral":
            f._update_cell_methods(
                method=method, domain_axes=f.domain_axes(axis, todict=True)
            )

        return f

    @_deprecated_kwarg_check("i")
    @_inplace_enabled(default=False)
    def convolution_filter(
        self,
        window=None,
        axis=None,
        mode=None,
        cval=None,
        origin=0,
        update_bounds=True,
        inplace=False,
        weights=None,
        i=False,
    ):
        """Convolve the field construct along the given axis with the
        specified filter.

        The magnitude of the integral of the filter (i.e. the sum of
        the window weights defined by the *window* parameter) affects
        the convolved values. For example, window weights of ``[0.2,
        0.2 0.2, 0.2, 0.2]`` will produce a non-weighted 5-point
        running mean; and window weights of ``[1, 1, 1, 1, 1]`` will
        produce a 5-point running sum. Note that the window weights
        returned by functions of the `scipy.signal.windows` package do
        not necessarily sum to 1 (see the examples for details).

        .. note:: The `moving_window` method can not, in general, be
                  emulated by the `convolution_filter` method, as the
                  latter i) can not change the window weights as the
                  filter passes through the axis; and ii) does not
                  update the cell method constructs.

        .. seealso:: `collapse`, `derivative`, `moving_window`,
                     `cf.relative_vorticity`

        :Parameters:

            window: sequence of numbers
                Specify the window weights to use for the filter.

                *Parameter example:*
                  An unweighted 5-point moving average can be computed
                  with ``window=[0.2, 0.2, 0.2, 0.2, 0.2]``

                Note that the `scipy.signal.windows` package has suite
                of window functions for creating window weights for
                filtering (see the examples for details).

                .. versionadded:: 3.3.0 (replaces the old weights
                                  parameter)

            axis:
                Select the domain axis over which the filter is to be
                applied, defined by that which would be selected by
                passing the given axis description to a call of the field
                construct's `domain_axis` method. For example, for a value
                of ``'X'``, the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

            mode: `str`, optional
                The *mode* parameter determines how the input array is
                extended when the filter overlaps an array border. The
                default value is ``'constant'`` or, if the dimension being
                convolved is cyclic (as ascertained by the `iscyclic`
                method), ``'wrap'``. The valid values and their behaviours
                are as follows:

                ==============  ==========================  ===========================
                *mode*          Description                 Behaviour
                ==============  ==========================  ===========================
                ``'reflect'``   The input is extended by    ``(c b a | a b c | c b a)``
                                reflecting about the edge

                ``'constant'``  The input is extended by    ``(k k k | a b c | k k k)``
                                filling all values beyond
                                the edge with the same
                                constant value (``k``),
                                defined by the *cval*
                                parameter.

                ``'nearest'``   The input is extended by    ``(a a a | a b c | d d d)``
                                replicating the last point

                ``'mirror'``    The input is extended by    ``(c b | a b c | b a)``
                                reflecting about the
                                centre of the last point.

                ``'wrap'``      The input is extended by    ``(a b c | a b c | a b c)``
                                wrapping around to the
                                opposite edge.
                ==============  ==========================  ===========================

                The position of the window relative to each value can be
                changed by using the *origin* parameter.

            cval: scalar, optional
                Value to fill past the edges of the array if *mode* is
                ``'constant'``. Ignored for other modes. Defaults to
                `None`, in which case the edges of the array will be
                filled with missing data.

                *Parameter example:*
                   To extend the input by filling all values beyond the
                   edge with zero: ``cval=0``

            origin: `int`, optional
                Controls the placement of the filter. Defaults to 0, which
                is the centre of the window. If the window has an even
                number of weights then then a value of 0 defines the index
                defined by ``width/2 -1``.

                *Parameter example:*
                  For a weighted moving average computed with a weights
                  window of ``[0.1, 0.15, 0.5, 0.15, 0.1]``, if
                  ``origin=0`` then the average is centred on each
                  point. If ``origin=-2`` then the average is shifted to
                  include the previous four points. If ``origin=1`` then
                  the average is shifted to include the previous point and
                  the and the next three points.

            update_bounds: `bool`, optional
                If False then the bounds of a dimension coordinate
                construct that spans the convolved axis are not
                altered. By default, the bounds of a dimension coordinate
                construct that spans the convolved axis are updated to
                reflect the width and origin of the window.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            weights: deprecated at version 3.3.0
                Use the *window* parameter instead.

        :Returns:

            `Field` or `None`
                The convolved field construct, or `None` if the operation
                was in-place.

        **Examples:**

        >>> f = cf.example_field(2)
        >>> print(f)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(f.array[:, 0, 0])
        [210.7 305.3 249.4 288.9 231.1 200.  234.4 289.2 204.3 203.6 261.8 256.2
         212.3 231.7 255.1 213.9 255.8 301.2 213.3 200.1 204.6 203.2 244.6 238.4
         304.5 269.8 267.9 282.4 215.  288.7 217.3 307.1 299.3 215.9 290.2 239.9]
        >>> print(f.coordinate('T').bounds.dtarray[0])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-01-01 00:00:00)]
        >>> print(f.coordinate('T').bounds.dtarray[2])
        [cftime.DatetimeGregorian(1960-02-01 00:00:00)
         cftime.DatetimeGregorian(1960-03-01 00:00:00)]

        Create a 5-point (non-weighted) running mean:

        >>> g = f.convolution_filter([0.2, 0.2, 0.2, 0.2, 0.2], 'T')
        >>> print(g)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(g.array[:, 0, 0])
        [ -- -- 257.08 254.94 240.76 248.72 231.8 226.3 238.66 243.02 227.64
         233.12 243.42 233.84 233.76 251.54 247.86 236.86 235.0 224.48 213.16
         218.18 239.06 252.1 265.04 272.6 267.92 264.76 254.26 262.1 265.48
         265.66 265.96 270.48 -- --]
        >>> print(g.coordinate('T').bounds.dtarray[0])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-03-01 00:00:00)]
        >>> print(g.coordinate('T').bounds.dtarray[2])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-05-01 00:00:00)]

        Create a 5-point running sum:

        >>> g = f.convolution_filter([1, 1, 1, 1, 1], 'T')
        >>> print(g)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(g.array[:, 0, 0])
        [ -- -- 1285.4 1274.7 1203.8 1243.6 1159.0 1131.5 1193.3 1215.1
         1138.2 1165.6 1217.1 1169.2 1168.8 1257.7 1239.3 1184.3 1175.0
         1122.4 1065.8 1090.9 1195.3 1260.5 1325.2 1363.0 1339.6 1323.8
         1271.3 1310.5 1327.4 1328.3 1329.8 1352.4 -- --]
        >>> print(g.coordinate('T').bounds.dtarray[0])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-03-01 00:00:00)]
        >>> print(g.coordinate('T').bounds.dtarray[2])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-05-01 00:00:00)]

        Calculate a convolution along the time axis with Gaussian window
        weights, using the "nearest" mode at the border of the edges of
        the time dimension (note that the window weights returned by
        `scipy.signal.windows` functions do not necessarily sum to 1):

        >>> import scipy.signal.windows
        >>> gaussian_window = scipy.signal.windows.gaussian(3, std=0.4)
        >>> print(gaussian_window)
        [0.04393693 1.         0.04393693]
        >>> g = f.convolution_filter(gaussian_window, 'T', mode='nearest')
        >>> print(g.array[:, 0, 0])
        [233.37145775 325.51538316 275.50732596 310.01169661 252.58076685
         220.4526426  255.89394793 308.47513278 225.95212089 224.07900476
         282.00220208 277.03050023 233.73682991 252.23612278 274.67829762
         236.34737939 278.43191451 321.81081556 235.32558483 218.46124456
         222.31976533 222.93647058 264.00254989 262.52577025 326.82874967
         294.94950081 292.16197475 303.61714525 240.09238279 307.69393641
         243.47762505 329.79781991 322.27901629 241.80082237 310.22645435
         263.19096851]
        >>> print(g.coordinate('T').bounds.dtarray[0])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-02-01 00:00:00)]
        >>> print(g.coordinate('T').bounds.dtarray[1])
        [cftime.DatetimeGregorian(1959-12-01 00:00:00)
         cftime.DatetimeGregorian(1960-03-01 00:00:00)]

        """
        if weights is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "convolution_filter",
                {"weights": weights},
                message="Use keyword 'window' instead.",
                version="3.3.0",
            )  # pragma: no cover

        if isinstance(window, str):
            _DEPRECATION_ERROR(
                "A string-valued 'window' parameter has been deprecated "
                "at version 3.0.0 and is no longer available. Provide a "
                "sequence of numerical window weights instead. "
                "scipy.signal.windows may be used to generate particular "
                "window functions."
            )  # pragma: no cover

        if isinstance(window[0], str):
            _DEPRECATION_ERROR(
                "A string-valued 'window' parameter element has been "
                "deprecated at version 3.0.0 and is no longer available. "
                "Provide a sequence of numerical window weights instead. "
                "scipy.signal.windows may be used to generate particular "
                "window functions."
            )  # pragma: no cover

        # Retrieve the axis
        axis_key = self.domain_axis(axis, key=True)
        iaxis = self.get_data_axes().index(axis_key)

        # Default mode to 'wrap' if the axis is cyclic
        if mode is None:
            if self.iscyclic(axis_key):
                mode = "wrap"
            else:
                mode = "constant"

        # Construct new field
        f = _inplace_enabled_define_and_cleanup(self)

        f.data.convolution_filter(
            window=window,
            axis=iaxis,
            mode=mode,
            cval=cval,
            origin=origin,
            inplace=True,
        )

        # Update the bounds of the convolution axis if necessary
        if update_bounds:
            coord = f.dimension_coordinate(
                filter_by_axis=(axis_key,), default=None
            )
            if coord is not None and coord.has_bounds():
                old_bounds = coord.bounds.array
                length = old_bounds.shape[0]
                new_bounds = numpy_empty((length, 2))
                len_weights = len(window)
                lower_offset = len_weights // 2 + origin
                upper_offset = len_weights - 1 - lower_offset
                if mode == "wrap":
                    if coord.direction():
                        new_bounds[:, 0] = coord.roll(
                            0, upper_offset
                        ).bounds.array[:, 0]
                        new_bounds[:, 1] = (
                            coord.roll(0, -lower_offset).bounds.array[:, 1]
                            + coord.period()
                        )
                    else:
                        new_bounds[:, 0] = (
                            coord.roll(0, upper_offset).bounds.array[:, 0]
                            + 2 * coord.period()
                        )
                        new_bounds[:, 1] = (
                            coord.roll(0, -lower_offset).bounds.array[:, 1]
                            + coord.period()
                        )
                else:
                    new_bounds[upper_offset:length, 0] = old_bounds[
                        0 : length - upper_offset, 0
                    ]
                    new_bounds[0:upper_offset, 0] = old_bounds[0, 0]
                    new_bounds[0 : length - lower_offset, 1] = old_bounds[
                        lower_offset:length, 1
                    ]
                    new_bounds[length - lower_offset : length, 1] = old_bounds[
                        length - 1, 1
                    ]

                coord.set_bounds(
                    self._Bounds(data=Data(new_bounds, units=coord.Units))
                )

        return f

    def convert(
        self, *identity, full_domain=True, cellsize=False, **filter_kwargs
    ):
        """Convert a metadata construct into a new field construct.

        The new field construct has the properties and data of the
        metadata construct, and domain axis constructs corresponding to
        the data. By default it also contains other metadata constructs
        (such as dimension coordinate and coordinate reference constructs)
        that define its domain.

        The `cf.read` function allows a field construct to be derived
        directly from a netCDF variable that corresponds to a metadata
        construct. In this case, the new field construct will have a
        domain limited to that which can be inferred from the
        corresponding netCDF variable - typically only domain axis and
        dimension coordinate constructs. This will usually result in a
        different field construct to that created with the convert method.

        .. versionadded:: 3.0.0

        .. seealso:: `cf.read`, `construct`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique construct returned by
                ``f.construct(*identity, **filter_kwargs)``. See
                `construct` for details.

            full_domain: `bool`, optional
                If False then do not create a domain, other than domain
                axis constructs, for the new field construct. By default
                as much of the domain as possible is copied to the new
                field construct.

            cellsize: `bool`, optional
                If True then create a field construct from the selected
                metadata construct's cell sizes.

        :Returns:

            `Field`
                The new field construct.

        **Examples:**

        TODO

        """
        key, construct = self.construct(
            *identity, item=True, default=(None, None), **filter_kwargs
        )
        if key is None:
            raise ValueError(
                f"Can't find metadata construct with identity {identity!r}"
            )

        f = super().convert(key, full_domain=full_domain)

        if cellsize:
            # Change the new field's data to cell sizes
            try:
                cs = construct.cellsize
            except AttributeError as error:
                raise ValueError(error)

            f.set_data(cs.data, set_axes=False, copy=False)

        return f

    @_inplace_enabled(default=False)
    def cumsum(
        self, axis, masked_as_zero=False, coordinate=None, inplace=False
    ):
        """Return the field cumulatively summed along the given axis.

        The cell bounds of the axis are updated to describe the range
        over which the sums apply, and a new "sum" cell method
        construct is added to the resulting field construct.

        .. versionadded:: 3.0.0

        .. seealso:: `collapse`, `convolution_filter`, `moving_window`,
                     `sum`

        :Parameters:

            axis:
                Select the domain axis over which the cumulative sums are
                to be calculated, defined by that which would be selected
                by passing the given axis description to a call of the
                field construct's `domain_axis` method. For example, for a
                value of ``'X'``, the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

            masked_as_zero: `bool`, optional
                If True then set missing data values to zero before
                calculating the cumulative sum. By default the output data
                will be masked at the same locations as the original data.
                .. note:: Sums produced entirely from masked elements will
                          always result in masked values in the output
                          data, regardless of the setting of
                          *masked_as_zero*.

            coordinate: `str`, optional
                Set how the cell coordinate values for the summed axis are
                defined, relative to the new cell bounds. By default they
                are unchanged from the original field construct. The
                *coordinate* parameter may be one of:

                ===============  =========================================
                *coordinate*     Description
                ===============  =========================================
                `None`           This is the default.
                                 Output coordinates are unchanged.
                ``'mid_range'``  An output coordinate is the average of
                                 its output coordinate bounds.
                ``'minimum'``    An output coordinate is the minimum of
                                 its output coordinate bounds.
                ``'maximum'``    An output coordinate is the maximum of
                                 its output coordinate bounds.
                ===============  =========================================

                *Parameter Example:*
                  ``coordinate='maximum'``
            {{inplace: `bool`, optional}}

        :Returns:
            `Field` or `None`
                The field construct with the cumulatively summed axis, or
                `None` if the operation was in-place.

        **Examples:**
        >>> f = cf.example_field(2)
        >>> print(f)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(f.dimension_coordinate('T').bounds[[0, -1]].datetime_array)
        [[cftime.DatetimeGregorian(1959-12-01 00:00:00)
          cftime.DatetimeGregorian(1960-01-01 00:00:00)]
         [cftime.DatetimeGregorian(1962-11-01 00:00:00)
          cftime.DatetimeGregorian(1962-12-01 00:00:00)]]
        >>> print(f.array[:, 0, 0])
        [210.7 305.3 249.4 288.9 231.1 200.  234.4 289.2 204.3 203.6 261.8 256.2
         212.3 231.7 255.1 213.9 255.8 301.2 213.3 200.1 204.6 203.2 244.6 238.4
         304.5 269.8 267.9 282.4 215.  288.7 217.3 307.1 299.3 215.9 290.2 239.9]
        >>> g = f.cumsum('T')
        >>> print(g)
        Field: air_potential_temperature (ncvar%air_potential_temperature)
        ------------------------------------------------------------------
        Data            : air_potential_temperature(time(36), latitude(5), longitude(8)) K
        Cell methods    : area: mean time(36): sum
        Dimension coords: time(36) = [1959-12-16 12:00:00, ..., 1962-11-16 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : air_pressure(1) = [850.0] hPa
        >>> print(g.dimension_coordinate('T').bounds[[0, -1]].datetime_array)
        [[cftime.DatetimeGregorian(1959-12-01 00:00:00)
          cftime.DatetimeGregorian(1960-01-01 00:00:00)]
         [cftime.DatetimeGregorian(1959-12-01 00:00:00)
          cftime.DatetimeGregorian(1962-12-01 00:00:00)]]
        >>> print(g.array[:, 0, 0])
        [ 210.7  516.   765.4 1054.3 1285.4 1485.4 1719.8 2009.  2213.3 2416.9
         2678.7 2934.9 3147.2 3378.9 3634.  3847.9 4103.7 4404.9 4618.2 4818.3
         5022.9 5226.1 5470.7 5709.1 6013.6 6283.4 6551.3 6833.7 7048.7 7337.4
         7554.7 7861.8 8161.1 8377.  8667.2 8907.1]
        >>> g = f.cumsum('latitude', masked_as_zero=True)
        >>> g = f.cumsum('latitude', coordinate='mid_range')
        >>> f.cumsum('latitude', inplace=True)

        """
        # Retrieve the axis
        axis_key = self.domain_axis(axis, key=True)
        if axis_key is None:
            raise ValueError(f"Invalid axis specifier: {axis!r}")

        # Construct new field
        f = _inplace_enabled_define_and_cleanup(self)

        # Get the axis index
        axis_index = f.get_data_axes().index(axis_key)

        f.data.cumsum(axis_index, masked_as_zero=masked_as_zero, inplace=True)

        if self.domain_axis(axis_key).get_size() > 1:
            # Update the bounds of the summed axis if necessary
            coord = f.dimension_coordinate(
                filter_by_axis=(axis_key,), default=None
            )
            if coord is not None and coord.has_bounds():
                bounds = coord.get_bounds()
                bounds[:, 0] = bounds[0, 0]

                data = coord.get_data(None, _fill_value=False)

                if coordinate is not None and data is not None:
                    if coordinate == "mid_range":
                        data[...] = (
                            (bounds[:, 0] + bounds[:, 1]) * 0.5
                        ).squeeze()
                    elif coordinate == "minimum":
                        data[...] = coord.lower_bounds
                    elif coordinate == "maximum":
                        data[...] = coord.upper_bounds
                    else:
                        raise ValueError(
                            "'coordinate' parameter must be one of "
                            "(None, 'mid_range', 'minimum', 'maximum'). "
                            f"Got {coordinate!r}"
                        )

            # Add a cell method
            f._update_cell_methods(
                method="sum", domain_axes=f.domain_axes(axis_key, todict=True)
            )

        return f

    @_inplace_enabled(default=False)
    def flip(self, axes=None, inplace=False, i=False, **kwargs):
        """Flip (reverse the direction of) axes of the field.

        .. seealso:: `domain_axis`, `flatten`, `insert_dimension`,
                     `squeeze`, `transpose`, `unsqueeze`

        :Parameters:

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes to flip, defined by the domain
                axes that would be selected by passing each given axis
                description to a call of the field construct's
                `domain_axis` method. For example, for a value of
                ``'X'``, the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

                If no axes are provided then all axes are flipped.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                The field construct with flipped axes, or `None` if
                the operation was in-place.

        **Examples:**

        >>> g = f.flip()
        >>> g = f.flip('time')
        >>> g = f.flip(1)
        >>> g = f.flip(['time', 1, 'dim2'])
        >>> f.flip(['dim2'], inplace=True)

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(self, "flip", kwargs)  # pragma: no cover

        if axes is None and not kwargs:
            # Flip all the axes
            axes = set(self.get_data_axes(default=()))
            iaxes = list(range(self.ndim))
        else:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes = set([self.domain_axis(axis, key=True) for axis in axes])

            data_axes = self.get_data_axes(default=())
            iaxes = [
                data_axes.index(axis)
                for axis in axes.intersection(self.get_data_axes())
            ]

        # Flip the requested axes in the field's data array
        f = _inplace_enabled_define_and_cleanup(self)
        super(Field, f).flip(iaxes, inplace=True)

        # Flip any constructs which span the flipped axes
        for key, construct in f.constructs.filter_by_data(todict=True).items():
            construct_axes = f.get_data_axes(key)
            construct_flip_axes = axes.intersection(construct_axes)
            if construct_flip_axes:
                iaxes = [
                    construct_axes.index(axis) for axis in construct_flip_axes
                ]
                construct.flip(iaxes, inplace=True)

        return f

    def argmax(self, axis=None):
        """Return the indices of the maximum values along an axis.

        If no axis is specified then the returned index locates the
        maximum of the whole data.

        .. seealso:: `argmin`, `where`

        :Parameters:

        :Returns:

        **Examples:**

        >>> g = f.argmax('T')

        """
        print("This method is not ready for use.")
        return

    # Keep these commented lines for using with the future dask version
    #
    #        standard_name = None
    #
    #        if axis is not None:
    #            axis_key = self.domain_axis(
    #                axis, key=True, default=ValueError("TODO")
    #            )
    #            axis = self.get_data_axes.index(axis_key)
    #            standard_name = self.domain_axis_identity(
    #                axis_key, strict=True, default=None
    #            )
    #
    #        indices = self.data.argmax(axis, unravel=True)
    #
    #        if axis is None:
    #            return self[indices]
    #
    #        # What if axis_key does not span array?
    #        out = self.subspace(**{axis_key: [0]})
    #        out.squeeze(axis_key, inplace=True)
    #
    #        for i in indices.ndindex():
    #            out.data[i] = org.data[indices[i].datum()]
    #
    #        for key, c in tuple(
    #            out.constructs.filter_by_type(
    #                "dimension_coordinate",
    #                "auxiliary_coordinate",
    #                "cell_measure",
    #                "domain_ancillary",
    #                "field_ancillary",
    #            )
    #            .filter_by_axis("and", axis_key)
    #            .items()
    #        ):
    #
    #            out.del_construct(key)
    #
    #            if c.construct_type == (
    #                "cell_measure",
    #                "domain_ancillary",
    #                "field_ancillary",
    #            ):
    #                continue
    #
    #            aux = self._AuxiliaryCoordinate()
    #            aux.set_properties(c.properties())
    #
    #            c_data = c.get_data(None)
    #            if c_data is not None:
    #                data = Data.empty(indices.shape, dtype=c.dtype)
    #                for x in indices.ndindex():
    #                    data[x] = c_data[indices[x]]
    #
    #                aux.set_data(data, copy=False)
    #
    #            c_bounds_data = c.get_bounds_data(None)
    #            if c_bounds_data is not None:
    #                bounds = Data.empty(
    #                    indices.shape + (c_bounds_data.shape[-1],),
    #                    dtype=c_bounds_data.dtype,
    #                )
    #                for x in indices.ndindex():
    #                    bounds[x] = c_bounds_data[indices[x]]
    #
    #                aux.set_bounds(
    #                    self._Bounds(data=bounds, copy=False), copy=False
    #                )
    #
    #            out.set_construct(aux, axes=out.get_data_axes(), copy=False)
    #
    #        if standard_name:
    #            cm = CellMethod()
    #            cm.create(standard_name + ": maximum")
    #
    #        return out

    @_deprecated_kwarg_check("i")
    def squeeze(self, axes=None, inplace=False, i=False, **kwargs):
        """Remove size 1 axes from the data.

        By default all size 1 axes are removed, but particular size 1 axes
        may be selected for removal.

        Squeezed domain axis constructs are not removed from the metadata
        constructs, nor from the domain of the field construct.

        .. seealso:: `domain_axis`, `flatten`, `insert_dimension`, `flip`,
                     `remove_axes`, `transpose`, `unsqueeze`

        :Parameters:

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes to squeeze, defined by the domain
                axes that would be selected by passing each given axis
                description to a call of the field construct's
                `domain_axis` method. For example, for a value of ``'X'``,
                the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

                If no axes are provided then all size 1 axes are squeezed.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                The field construct with squeezed data, or `None` if the
                operation was in-place.

                **Examples:**

        >>> g = f.squeeze()
        >>> g = f.squeeze('time')
        >>> g = f.squeeze(1)
        >>> g = f.squeeze(['time', 1, 'dim2'])
        >>> f.squeeze(['dim2'], inplace=True)

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "squeeze", kwargs
            )  # pragma: no cover

        data_axes = self.get_data_axes()

        if axes is None:
            domain_axes = self.domain_axes(todict=True)
            axes = [
                axis
                for axis in data_axes
                if domain_axes[axis].get_size(None) == 1
            ]
        else:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes = [self.domain_axis(x, key=True) for x in axes]
            axes = set(axes).intersection(data_axes)

        iaxes = [data_axes.index(axis) for axis in axes]

        # Squeeze the field's data array
        return super().squeeze(iaxes, inplace=inplace)

    @_inplace_enabled(default=False)
    def swapaxes(self, axis0, axis1, inplace=False, i=False):
        """Interchange two axes of the data.

        .. seealso:: `flatten`, `flip`, `insert_dimension`, `squeeze`,
                     `transpose`

        :Parameters:

            axis0, axis1: TODO
                Select the axes to swap. Each axis is identified by its
                original integer position.

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`
                The field construct with data with swapped axis
                positions. If the operation was in-place then `None` is
                returned.

        **Examples:**

        >>> f.shape
        (1, 2, 3)
        >>> f.swapaxes(1, 0).shape
        (2, 1, 3)
        >>> f.swapaxes(0, -1).shape
        (3, 2, 1)
        >>> f.swapaxes(1, 1).shape
        (1, 2, 3)
        >>> f.swapaxes(-1, -1).shape
        (1, 2, 3)

        """
        data_axes = self.get_data_axes(default=None)

        da_key0 = self.domain_axis(axis0, key=True)
        da_key1 = self.domain_axis(axis1, key=True)

        if da_key0 not in data_axes:
            raise ValueError(
                f"Can't swapaxes: Bad axis specification: {axis0!r}"
            )

        if da_key1 not in data_axes:
            raise ValueError(
                f"Can't swapaxes: Bad axis specification: {axis1!r}"
            )

        axis0 = data_axes.index(da_key0)
        axis1 = data_axes.index(da_key1)

        f = _inplace_enabled_define_and_cleanup(self)
        super(Field, f).swapaxes(axis0, axis1, inplace=True)

        if data_axes is not None:
            data_axes = list(data_axes)
            data_axes[axis1], data_axes[axis0] = (
                data_axes[axis0],
                data_axes[axis1],
            )
            f.set_data_axes(data_axes)

        return f

    @_deprecated_kwarg_check("i")
    def transpose(
        self,
        axes=None,
        constructs=False,
        inplace=False,
        items=True,
        i=False,
        **kwargs,
    ):
        """Permute the axes of the data array.

        By default the order of the axes is reversed, but any ordering may
        be specified by selecting the axes of the output in the required
        order.

        By default metadata constructs are not transposed, but they may be
        if the *constructs* parameter is set.

        .. seealso:: `domain_axis`, `flatten`, `insert_dimension`, `flip`,
                     `squeeze`, `unsqueeze`

        :Parameters:

            axes: (sequence of) `str` or `int`, optional
                Select the domain axis order, defined by the domain axes
                that would be selected by passing each given axis
                description to a call of the field construct's
                `domain_axis` method. For example, for a value of ``'X'``,
                the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

                Each dimension of the field construct's data must be
                provided, or if no axes are specified then the axis order
                is reversed.

            constructs: `bool`
                If True then metadata constructs are also transposed so
                that their axes are in the same relative order as in the
                transposed data array of the field. By default metadata
                constructs are not altered.

            {{inplace: `bool`, optional}}

            items: deprecated at version 3.0.0
                Use the *constructs* parameter instead.

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                The field construct with transposed data, or `None` if the
                operation was in-place.

        **Examples:**

        >>> f.ndim
        3
        >>> g = f.transpose()
        >>> g = f.transpose(['time', 1, 'dim2'])
        >>> f.transpose(['time', -2, 'dim2'], inplace=True)

        """
        if not items:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "transpose",
                {"items": items},
                "Use keyword 'constructs' instead.",
            )  # pragma: no cover

        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "transpose", kwargs
            )  # pragma: no cover

        if axes is None:
            iaxes = list(range(self.ndim - 1, -1, -1))
        else:
            data_axes = self.get_data_axes(default=())
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes2 = [self.domain_axis(x, key=True) for x in axes]

            if sorted(axes2) != sorted(data_axes):
                raise ValueError(
                    f"Can't transpose {self.__class__.__name__}: "
                    f"Bad axis specification: {axes!r}"
                )

            iaxes = [data_axes.index(axis) for axis in axes2]

        # Transpose the field's data array
        return super().transpose(iaxes, constructs=constructs, inplace=inplace)

    @_deprecated_kwarg_check("i")
    @_inplace_enabled(default=False)
    def unsqueeze(self, inplace=False, i=False, axes=None, **kwargs):
        """Insert size 1 axes into the data array.

        All size 1 domain axes which are not spanned by the field
        construct's data are inserted.

        The axes are inserted into the slowest varying data array positions.

        .. seealso:: `flatten`, `flip`, `insert_dimension`, `squeeze`,
                     `transpose`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            axes: deprecated at version 3.0.0

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                The field construct with size-1 axes inserted in its data,
                or `None` if the operation was in-place.

        **Examples:**

        >>> g = f.unsqueeze()
        >>> f.unsqueeze(['dim2'], inplace=True)

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "unsqueeze", kwargs
            )  # pragma: no cover

        if axes is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "unsqueeze",
                {"axes": axes},
                "All size one domain axes missing from the data are "
                "inserted. Use method 'insert_dimension' to insert an "
                "individual size one domain axis.",
            )  # pragma: no cover

        f = _inplace_enabled_define_and_cleanup(self)

        size_1_axes = self.domain_axes(filter_by_size=(1,), todict=True)
        for axis in set(size_1_axes).difference(self.get_data_axes()):
            f.insert_dimension(axis, position=0, inplace=True)

        return f

    def cell_method(
        self,
        *identity,
        default=ValueError(),
        key=False,
        item=False,
        **filter_kwargs,
    ):
        """Select a cell method construct.

        {{unique construct}}

        .. versionadded:: 3.0.0

        .. seealso:: `construct`, `cell_methods`

        :Parameters:

            identity: optional
                Select cell method constructs that have an identity,
                defined by their `!identities` methods, that matches
                any of the given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                Additionally, if for a given value
                ``f.domain_axes(value)`` returns a unique domain axis
                construct then any cell method constructs that span
                exactly that axis are selected. See `domain_axes` for
                details.

                If no values are provided then all cell method
                constructs are selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}

        **Examples:**

        """
        return self._construct(
            "cell_method",
            "cell_methods",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

    def field_ancillary(
        self,
        *identity,
        default=ValueError(),
        key=False,
        item=False,
        **filter_kwargs,
    ):
        """Select a field ancillary construct.

        {{unique construct}}

        .. versionadded:: 3.0.0

        .. seealso:: `construct`, `field_ancillaries`

        :Parameters:

            identity: optional
                Select field ancillary constructs that have an
                identity, defined by their `!identities` methods, that
                matches any of the given values.

                Additionally, the values are matched against construct
                identifiers, with or without the ``'key%'`` prefix.

                If no values are provided then all field ancillary
                constructs are selected.

                {{value match}}

                {{displayed identity}}

            {{key: `bool`, optional}}

            {{item: `bool`, optional}}

                .. versionadded:: (cfdm) 3.9.0

            default: optional
                Return the value of the *default* parameter if there
                is no unique construct.

                {{default Exception}}

            {{filter_kwargs: optional}}

                .. versionadded:: (cfdm) 3.9.0

        :Returns:

                {{Returns construct}}

        **Examples:**

        """
        return self._construct(
            "field_ancillary",
            "field_ancillaries",
            identity,
            key=key,
            item=item,
            default=default,
            **filter_kwargs,
        )

    def domain_axis_position(self, *identity, **filter_kwargs):
        """Return the position in the data of a domain axis construct.

        .. versionadded:: 3.0.0

        .. seealso:: `domain_axis`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique domain axis construct returned by
                ``f.domain_axis(*identity, **filter_kwargs)``. See
                `domain_axis` for details.

        :Returns:

            `int`
                The position in the field construct's data of the
                selected domain axis construct.

        **Examples:**

        >>> f
        <CF Field: air_temperature(time(12), latitude(64), longitude(128)) K>
        >>> f.get_data_axes()
        ('domainaxis0', 'domainaxis1', 'domainaxis2')
        >>> f.domain_axis_position('T')
        0
        >>> f.domain_axis_position('latitude')
        1
        >>> f.domain_axis_position('domainaxis1')
        1
        >>> f.domain_axis_position(2)
        2
        >>> f.domain_axis_position(-2)
        1

        """
        key = self.domain_axis(*identity, key=True)
        return self.get_data_axes().index(key)

    def axes_names(self, *identities, **kwargs):
        """Return canonical identities for each domain axis construct.

        :Parameters:

            kwargs: deprecated at version 3.0.0

        :Returns:

            `dict`
                The canonical name for the domain axis construct.

        **Examples:**

        >>> f.axis_names()
        {'domainaxis0': 'atmosphere_hybrid_height_coordinate',
         'domainaxis1': 'grid_latitude',
         'domainaxis2': 'grid_longitude'}

        """
        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "axes_names", kwargs
            )  # pragma: no cover

        out = self.domain_axes(todict=True).copy()

        for key in tuple(out):
            value = self.constructs.domain_axis_identity(key)
            if value is not None:
                out[key] = value
            else:
                del out[key]

        return out

    def axis_size(
        self, *identity, default=ValueError(), axes=None, **filter_kwargs
    ):
        """Return the size of a domain axis construct.

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique domain axis construct returned by
                ``f.domain_axis(*identity, **filter_kwargs)``. See
                `domain_axis` for details.

            default: optional
                Return the value of the *default* parameter if a domain
                axis construct can not be found. If set to an `Exception`
                instance then it will be raised instead.

            axes: deprecated at version 3.0.0

        :Returns:

            `int`
                The size of the selected domain axis

        **Examples:**

        >>> f
        <CF Field: eastward_wind(time(3), air_pressure(5), latitude(110), longitude(106)) m s-1>
        >>> f.axis_size('longitude')
        106
        >>> f.axis_size('Z')
        5

        """
        if axes:
            _DEPRECATION_ERROR_KWARGS(
                self, "axis_size", "Use keyword 'identity' instead."
            )  # pragma: no cover

        axis = self.domain_axis(*identity, default=None, **filter_kwargs)
        if axis is None:
            return self._default(default)

        return axis.get_size(default=default)

    def get_data_axes(self, *identity, default=ValueError(), **filter_kwargs):
        """Return domain axis constructs spanned by data.

        Specifically, returns the keys of the domain axis constructs
        spanned by the field's data, or the data of a metadata construct.

        .. versionadded:: 3.0.0

        .. seealso:: `del_data_axes`, `has_data_axes`,
                     `set_data_axes`, `construct`

        :Parameters:

            identity, filter_kwargs: optional
                Select the unique construct returned by
                ``f.construct(*identity, **filter_kwargs)``. See
                `construct` for details.

                If neither *identity* nor *filter_kwargs* are set then
                the domain of the field constructs's data are
                returned.

            default: optional
                Return the value of the *default* parameter if the
                data axes have not been set.

                {{default Exception}}

        :Returns:

            `tuple` of `str`
                The keys of the domain axis constructs spanned by the
                data.

        **Examples:**

        >>> f.set_data_axes(['domainaxis0', 'domainaxis1'])
        >>> f.get_data_axes()
        ('domainaxis0', 'domainaxis1')
        >>> f.del_data_axes()
        ('domainaxis0', 'domainaxis1')
        >>> print(f.del_dataxes(None))
        None
        >>> print(f.get_data_axes(default=None))
        None

        """
        if not identity and not filter_kwargs:
            # Get axes of the Field data array
            return super().get_data_axes(default=default)

        key = self.construct(*identity, key=True, **filter_kwargs)

        axes = super().get_data_axes(key, default=None)
        if axes is None:
            return self._default(
                default, "Can't get axes for non-existent construct"
            )

        return axes

    @_inplace_enabled(default=False)
    @_manage_log_level_via_verbosity
    def halo(
        self,
        size,
        axes=None,
        tripolar=None,
        fold_index=-1,
        inplace=False,
        verbose=None,
    ):
        """Expand the field construct by adding a halo to its data.

        The halo may be applied over a subset of the data dimensions and
        each dimension may have a different halo size (including
        zero). The halo region is populated with a copy of the proximate
        values from the original data.

        The metadata constructs are similarly extended where appropriate.

        **Cyclic axes**

        A cyclic axis that is expanded with a halo of at least size 1 is
        no longer considered to be cyclic.

        **Tripolar domains**

        Global tripolar domains are a special case in that a halo added to
        the northern end of the "Y" axis must be filled with values that
        are flipped in "X" direction. Such domains can not be identified
        from the field construct's metadata, so need to be explicitly
        indicated with the *tripolar* parameter.

        .. versionadded:: 3.5.0

        :Parameters:

            size:  `int` or `dict`
                Specify the size of the halo for each axis.

                If *size* is a non-negative `int` then this is the halo
                size that is applied to all of the axes defined by the
                *axes* parameter.

                Alternatively, halo sizes may be assigned to axes
                individually by providing a `dict` for which a key
                specifies an axis (by passing the axis description to a
                call of the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')``) with a corresponding
                value of the halo size for that axis. Axes not specified
                by the dictionary are not expanded, and the *axes*
                parameter must not also be set.

                *Parameter example:*
                  Specify a halo size of 1 for all otherwise selected
                  axes: ``size=1``

                *Parameter example:*
                  Specify a halo size of zero ``size=0``. This results in
                  no change to the data shape.

                *Parameter example:*
                  For data with three dimensions, specify a halo size of 3
                  for the first dimension and 1 for the second dimension:
                  ``size={0: 3, 1: 1}``. This is equivalent to ``size={0:
                  3, 1: 1, 2: 0}``

                *Parameter example:*
                  Specify a halo size of 2 for the "longitude" and
                  "latitude" axes: ``size=2, axes=['latutude',
                  'longitude']``, or equivalently ``size={'latutude': 2,
                  'longitude': 2}``.

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes to be expanded, defined by the
                domain axes that would be selected by passing each given
                axis description to a call of the field construct's
                `domain_axis` method. For example, for a value of ``'X'``,
                the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

                By default, or if *axes* is `None`, all axes that span the
                data are selected. No axes are expanded if *axes* is an
                empty sequence.

                *Parameter example:*
                  ``axes='X'``

                *Parameter example:*
                  ``axes=['Y']``

                *Parameter example:*
                  ``axes=['X', 'Y']``

                *Parameter example:*
                  ``axes='longitude'``

                *Parameter example:*
                  ``axes=2``

                *Parameter example:*
                  ``axes='ncdim%i'``

            tripolar: `dict`, optional
                A dictionary defining the "X" and "Y" axes of a global
                tripolar domain. This is necessary because in the global
                tripolar case the "X" and "Y" axes need special treatment,
                as described above. It must have keys ``'X'`` and ``'Y'``,
                whose values identify the corresponding domain axis
                construct by passing the value to a call of the field
                construct's `domain_axis` method. For example, for a value
                of ``'ncdim%i'``, the domain axis construct returned by
                ``f.domain_axis('ncdim%i')``.

                The "X" and "Y" axes must be a subset of those identified
                by the *size* or *axes* parameter.

                See the *fold_index* parameter.

                *Parameter example:*
                  Define the "X" and Y" axes by their netCDF dimension
                  names: ``tripolar={'X': 'ncdim%i', 'Y': 'ncdim%j'}``

                *Parameter example:*
                  Define the "X" and Y" axes by positions 2 and 1
                  respectively of the data: ``tripolar={'X': 2, 'Y': 1}``

            fold_index: `int`, optional
                Identify which index of the "Y" axis corresponds to the
                fold in "X" axis of a tripolar grid. The only valid values
                are ``-1`` for the last index, and ``0`` for the first
                index. By default it is assumed to be the last
                index. Ignored if *tripolar* is `None`.

            {{inplace: `bool`, optional}}

            {{verbose: `int` or `str` or `None`, optional}}

        :Returns:

            `Field` or `None`
                The expanded field construct, or `None` if the operation
                was in-place.

        **Examples:**

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(f.array)
        [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
         [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
         [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
        >>> print(f.coordinate('X').array)
        [ 22.5  67.5 112.5 157.5 202.5 247.5 292.5 337.5]

        >>> g = f.halo(1)
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(7), longitude(10)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(7) = [-75.0, ..., 75.0] degrees_north
                        : longitude(10) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(g.array)
        [[0.007 0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029 0.029]
         [0.007 0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029 0.029]
         [0.023 0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066 0.066]
         [0.11  0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011 0.011]
         [0.029 0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017 0.017]
         [0.006 0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013 0.013]
         [0.006 0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013 0.013]]
        >>> print(g.coordinate('X').array)
        [ 22.5  22.5  67.5 112.5 157.5 202.5 247.5 292.5 337.5 337.5]

        >>> g = f.halo(1, axes='Y')
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(7), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(7) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(g.array)
        [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
         [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
         [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
        >>> h = f.halo({'Y': 1})
        >>> h.equals(g)
        True

        >>> g = f.halo({'Y': 2, 'X': 1})
        >>> print(g)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(9), longitude(10)) 1
        Cell methods    : area: mean
        Dimension coords: latitude(9) = [-75.0, ..., 75.0] degrees_north
                        : longitude(10) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(g.array)
        [[0.007 0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029 0.029]
         [0.023 0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066 0.066]
         [0.007 0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029 0.029]
         [0.023 0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066 0.066]
         [0.11  0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011 0.011]
         [0.029 0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017 0.017]
         [0.006 0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013 0.013]
         [0.029 0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017 0.017]
         [0.006 0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013 0.013]]

        """
        f = _inplace_enabled_define_and_cleanup(self)

        # Set the halo size for each axis.
        data_axes = f.get_data_axes(default=())
        if isinstance(size, dict):
            if axes is not None:
                raise ValueError("can't set axes when size is a dict TODO")

            axis_halo = {
                self.domain_axis(k, key=True): v for k, v in size.items()
            }

            if not set(data_axes).issuperset(axis_halo):
                raise ValueError(
                    f"Can't apply halo: Bad axis specification: {size!r}"
                )
        else:
            if axes is None:
                axes = data_axes

            if isinstance(axes, (str, int)):
                axes = (axes,)

            axis_halo = {self.domain_axis(k, key=True): size for k in axes}

        if tripolar:
            # Find the X and Y axes of a tripolar grid
            tripolar = tripolar.copy()
            X_axis = tripolar.pop("X", None)
            Y_axis = tripolar.pop("Y", None)

            if X_axis is None:
                raise ValueError("Must provide a tripolar 'X' axis.")

            if Y_axis is None:
                raise ValueError("Must provide a tripolar 'Y' axis.")

            X = self.domain_axis(X_axis, key=True)
            Y = self.domain_axis(Y_axis, key=True)

            try:
                i_X = data_axes.index(X)
            except ValueError:
                raise ValueError(f"Axis {X_axis!r} is not spanned by the data")

            try:
                i_Y = data_axes.index(Y)
            except ValueError:
                raise ValueError(f"Axis {Y_axis!r} is not spanned by the data")

            tripolar["X"] = i_X
            tripolar["Y"] = i_Y

            tripolar_axes = {X: "X", Y: "Y"}

        # Add halos to the field construct's data
        size = {data_axes.index(axis): h for axis, h, in axis_halo.items()}

        f.data.halo(
            size=size,
            tripolar=tripolar,
            fold_index=fold_index,
            inplace=True,
            verbose=verbose,
        )

        # Change domain axis sizes
        for axis, h in axis_halo.items():
            d = f.domain_axis(axis)
            d.set_size(d.get_size() + 2 * h)

        # Add halos to metadata constructs
        for key, c in f.constructs.filter_by_data(todict=True).items():
            construct_axes = f.get_data_axes(key)
            construct_size = {
                construct_axes.index(axis): h
                for axis, h in axis_halo.items()
                if axis in construct_axes
            }

            if not construct_size:
                # This construct does not span an expanded axis
                continue

            construct_tripolar = False
            if tripolar and set(construct_axes).issuperset(tripolar_axes):
                construct_tripolar = {
                    axis_type: construct_axes.index(axis)
                    for axis, axis_type in tripolar_axes.items()
                }

            c.halo(
                size=construct_size,
                tripolar=construct_tripolar,
                fold_index=fold_index,
                inplace=True,
                verbose=verbose,
            )

        return f

    def percentile(
        self, ranks, axes=None, interpolation="linear", squeeze=False, mtol=1
    ):
        """Compute percentiles of the data along the specified axes.

        The default is to compute the percentiles along a flattened
        version of the data.

        If the input data are integers, or floats smaller than float64, or
        the input data contains missing values, then output data type is
        float64. Otherwise, the output data type is the same as that of
        the input.

        If multiple percentile ranks are given then a new, leading data
        dimension is created so that percentiles can be stored for each
        percentile rank.

        The output field construct has a new dimension coordinate
        construct that records the percentile ranks represented by its
        data.

        .. versionadded:: 3.0.4

        .. seealso:: `bin`, `collapse`, `digitize`, `where`

        :Parameters:

            ranks: (sequence of) number
                Percentile ranks, or sequence of percentile ranks, to
                compute, which must be between 0 and 100 inclusive.

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes over which to calculate the
                percentiles, defined by the domain axes that would be
                selected by passing each given axis description to a call
                of the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected.

                By default, or if *axes* is `None`, all axes are selected.

            interpolation: `str`, optional
                Specify the interpolation method to use when the desired
                percentile lies between two data values ``i < j``:

                ===============  =========================================
                *interpolation*  Description
                ===============  =========================================
                ``'linear'``     ``i+(j-i)*fraction``, where ``fraction``
                                 is the fractional part of the index
                                 surrounded by ``i`` and ``j``
                ``'lower'``      ``i``
                ``'higher'``     ``j``
                ``'nearest'``    ``i`` or ``j``, whichever is nearest.
                ``'midpoint'``   ``(i+j)/2``
                ===============  =========================================

                By default ``'linear'`` interpolation is used.

            squeeze: `bool`, optional
                If True then all size 1 axes are removed from the returned
                percentiles data. By default axes over which percentiles
                have been calculated are left in the result as axes with
                size 1, meaning that the result is guaranteed to broadcast
                correctly against the original data.

            mtol: number, optional
                Set the fraction of input data elements which is allowed
                to contain missing data when contributing to an individual
                output data element. Where this fraction exceeds *mtol*,
                missing data is returned. The default is 1, meaning that a
                missing datum in the output array occurs when its
                contributing input array elements are all missing data. A
                value of 0 means that a missing datum in the output array
                occurs whenever any of its contributing input array
                elements are missing data. Any intermediate value is
                permitted.

                *Parameter example:*
                  To ensure that an output array element is a missing
                  datum if more than 25% of its input array elements are
                  missing data: ``mtol=0.25``.

        :Returns:

            `Field`
                The percentiles of the original data.

        **Examples:**

        >>> f = cf.example_field(0)
        >>> print(f)
        Field: specific_humidity
        ------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 1
        Cell methods    : area: mean
        Dimension coords: time(1) = [2019-01-01 00:00:00]
                        : latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
        >>> print(f.array)
        [[0.007 0.034 0.003 0.014 0.018 0.037 0.024 0.029]
         [0.023 0.036 0.045 0.062 0.046 0.073 0.006 0.066]
         [0.11  0.131 0.124 0.146 0.087 0.103 0.057 0.011]
         [0.029 0.059 0.039 0.07  0.058 0.072 0.009 0.017]
         [0.006 0.036 0.019 0.035 0.018 0.037 0.034 0.013]]
        >>> p = f.percentile([20, 40, 50, 60, 80])
        >>> print(p)
        Field: specific_humidity
        ------------------------
        Data            : specific_humidity(long_name=Percentile ranks for latitude, longitude dimensions(5), latitude(1), longitude(1)) 1
        Dimension coords: time(1) = [2019-01-01 00:00:00]
                        : latitude(1) = [0.0] degrees_north
                        : longitude(1) = [180.0] degrees_east
                        : long_name=Percentile ranks for latitude, longitude dimensions(5) = [20, ..., 80]
        >>> print(p.array)
        [[[0.0164]]
         [[0.032 ]]
         [[0.036 ]]
         [[0.0414]]
         [[0.0704]]]

        Find the standard deviation of the values above the 80th percentile:

        >>> p80 = f.percentile(80)
        >>> print(p80)
        Field: specific_humidity
        ------------------------
        Data            : specific_humidity(latitude(1), longitude(1)) 1
        Dimension coords: time(1) = [2019-01-01 00:00:00]
                        : latitude(1) = [0.0] degrees_north
                        : longitude(1) = [180.0] degrees_east
                        : long_name=Percentile ranks for latitude, longitude dimensions(1) = [80]
        >>> g = f.where(f<=p80, cf.masked)
        >>> print(g.array)
        [[  --    --    --    --    --    -- -- --]
         [  --    --    --    --    -- 0.073 -- --]
         [0.11 0.131 0.124 0.146 0.087 0.103 -- --]
         [  --    --    --    --    -- 0.072 -- --]
         [  --    --    --    --    --    -- -- --]]
        >>> g.collapse('standard_deviation', weights=True).data
        <CF Data(1, 1): [[0.024609938742357642]] 1>

        Find the mean of the values above the 45th percentile along the
        X axis:

        >>> p45 = f.percentile(45, axes='X')
        >>> print(p45.array)
        [[0.0189 ]
         [0.04515]
         [0.10405]
         [0.04185]
         [0.02125]]
        >>> g = f.where(f<=p45, cf.masked)
        >>> print(g.array)
        [[  -- 0.034    --    --    -- 0.037 0.024 0.029]
         [  --    --    -- 0.062 0.046 0.073    -- 0.066]
         [0.11 0.131 0.124 0.146    --    --    --    --]
         [  -- 0.059    -- 0.07  0.058 0.072    --    --]
         [  -- 0.036    -- 0.035   --  0.037 0.034    --]]
        >>> print(g.collapse('X: mean', weights=True).array)
        [[0.031  ]
         [0.06175]
         [0.12775]
         [0.06475]
         [0.0355 ]]

        Find the histogram bin boundaries associated with given
        percentiles, and digitize the data based on these bins:

        >>> bins = f.percentile([0, 10, 50, 90, 100], squeeze=True)
        >>> print(bins.array)
        [0.003  0.0088 0.036  0.1037 0.146 ]
        >>> i = f.digitize(bins, closed_ends=True)
        >>> print(i.array)
        [[0 1 0 1 1 2 1 1]
         [1 2 2 2 2 2 0 2]
         [3 3 3 3 2 2 2 1]
         [1 2 2 2 2 2 1 1]
         [0 2 1 1 1 2 1 1]]

        """
        data_axes = self.get_data_axes(default=())

        if axes is None:
            axes = data_axes[:]
            iaxes = list(range(self.ndim))
        else:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes = set([self.domain_axis(axis, key=True) for axis in axes])
            iaxes = [
                data_axes.index(axis)
                for axis in axes.intersection(self.get_data_axes())
            ]

        data = self.data.percentile(
            ranks,
            axes=iaxes,
            interpolation=interpolation,
            squeeze=False,
            mtol=mtol,
        )

        # ------------------------------------------------------------
        # Initialize the output field with the percentile data
        # ------------------------------------------------------------
        out = type(self)()
        out.set_properties(self.properties())

        for axis in [
            axis
            for axis in self.domain_axes(todict=True)
            if axis not in data_axes
        ]:
            out.set_construct(self._DomainAxis(1), key=axis)

        out_data_axes = []
        if data.ndim == self.ndim:
            for n, axis in zip(data.shape, data_axes):
                out_data_axes.append(
                    out.set_construct(self._DomainAxis(n), key=axis)
                )
        elif data.ndim == self.ndim + 1:
            for n, axis in zip(data.shape[1:], data_axes):
                out_data_axes.append(
                    out.set_construct(self._DomainAxis(n), key=axis)
                )

            out_data_axes.insert(
                0, out.set_construct(self._DomainAxis(data.shape[0]))
            )

        out.set_data(data, axes=out_data_axes, copy=False)

        # ------------------------------------------------------------
        # Create dimension coordinate constructs for the percentile
        # axes
        # ------------------------------------------------------------
        if axes:
            for key, c in self.dimension_coordinates(
                filter_by_axis=axes, axis_mode="subset", todict=True
            ).items():
                c_axes = self.get_data_axes(key)

                c = c.copy()

                bounds = c.get_bounds_data(
                    c.get_data(None, _fill_value=False), _fill_value=False
                )
                if bounds is not None and bounds.shape[0] > 1:
                    bounds = Data(
                        [bounds.min().datum(), bounds.max().datum()],
                        units=c.Units,
                    )

                    data = bounds.mean()
                    c.set_data(data, copy=False)

                    bounds.insert_dimension(inplace=True)
                    c.set_bounds(self._Bounds(data=bounds), copy=False)

                out.set_construct(c, axes=c_axes, key=key, copy=False)

        # TODO optimise constructs access?
        other_axes = set(
            [
                axis
                for axis in self.domain_axes(todict=True)
                if axis not in axes or self.domain_axis(axis).size == 1
            ]
        )

        # ------------------------------------------------------------
        # Copy constructs to the output field
        # ------------------------------------------------------------
        if other_axes:
            for key, c in self.constructs.filter_by_axis(
                *other_axes, axis_mode="subset", todict=True
            ).items():
                c_axes = self.get_data_axes(key)
                out.set_construct(c, axes=c_axes, key=key)

        # ------------------------------------------------------------
        # Copy coordinate reference constructs to the output field
        # ------------------------------------------------------------
        out_coordinates = out.coordinates(todict=True)
        out_domain_ancillaries = out.domain_ancillaries(todict=True)

        for cr_key, ref in self.coordinate_references(todict=True).items():
            ref = ref.copy()

            for c_key in ref.coordinates():
                if c_key not in out_coordinates:
                    ref.del_coordinate(c_key)

            for (
                term,
                da_key,
            ) in ref.coordinate_conversion.domain_ancillaries().items():
                if da_key not in out_domain_ancillaries:
                    ref.coordinate_conversion.set_domain_ancillary(term, None)

            out.set_construct(ref, key=cr_key, copy=False)

        # ------------------------------------------------------------
        # Create a dimension coordinate for the percentile ranks
        # ------------------------------------------------------------
        dim = DimensionCoordinate()
        data = Data(ranks).squeeze()
        data.override_units(Units(), inplace=True)
        if not data.shape:
            data.insert_dimension(inplace=True)
        dim.set_data(data, copy=False)

        if out.ndim == self.ndim:
            axis = out.set_construct(self._DomainAxis(1))
        else:
            axis = out_data_axes[0]

        axes = sorted(axes)
        if len(axes) == 1:
            dim.long_name = (
                "Percentile ranks for "
                + self.constructs.domain_axis_identity(axes[0])
                + " dimensions"
            )
        else:
            dim.long_name = (
                "Percentile ranks for "
                + ", ".join(map(self.constructs.domain_axis_identity, axes))
                + " dimensions"
            )

        out.set_construct(dim, axes=axis, copy=False)

        if squeeze:
            out.squeeze(inplace=True)

        return out

    @_inplace_enabled(default=False)
    def flatten(self, axes=None, return_axis=False, inplace=False):
        """Flatten axes of the field.

        Any subset of the domain axes may be flattened.

        The shape of the data may change, but the size will not.

        Metadata constructs whose data spans the flattened axes will
        either themselves be flattened, or else removed.

        Cell method constructs that apply to the flattened axes will
        be removed or, if possible, have their axis specifications
        changed to standard names.

        The flattening is executed in row-major (C-style) order. For
        example, the array ``[[1, 2], [3, 4]]`` would be flattened
        across both dimensions to ``[1 2 3 4]``.

        .. versionadded:: 3.0.2

        .. seealso:: `compress`, `insert_dimension`, `flip`, `swapaxes`,
                     `transpose`

        :Parameters:

            axes: (sequence of) `str` or `int`, optional
                Select the domain axes to be flattened, defined by the
                domain axes that would be selected by passing each
                given axis description to a call of the field
                construct's `domain_axis` method. For example, for a
                value of ``'X'``, the domain axis construct returned
                by ``f.domain_axis('X')`` is selected.

                If no axes are provided then all axes spanned by the
                field construct's data are flattened.

                No axes are flattened if *axes* is an empty sequence.

            return_axis: `bool`, optional
                If True then also return either the key of the
                flattened domain axis construct; or `None` if the axes
                to be flattened do not span the data.

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`, [`str` or `None`]
                The new, flattened field construct, or `None` if the
                operation was in-place.

                If *return_axis* is True then also return either the
                key of the flattened domain axis construct; or `None`
                if the axes to be flattened do not span the data.

        **Examples**

        See `cf.Data.flatten` for more examples of how the data are
        flattened.

        >>> f.shape
        (1, 2, 3, 4)
        >>> f.flatten().shape
        (24,)
        >>> f.flatten([]).shape
        (1, 2, 3, 4)
        >>> f.flatten([1, 3]).shape
        (1, 8, 3)
        >>> f.flatten([0, -1], inplace=True)
        >>> f.shape
        (4, 2, 3)

        >>> print(t)
        Field: air_temperature (ncvar%ta)
        ---------------------------------
        Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K
        Cell methods    : grid_latitude(10): grid_longitude(9): mean where land (interval: 0.1 degrees) time(1): maximum
        Field ancils    : air_temperature standard_error(grid_latitude(10), grid_longitude(9)) = [[0.76, ..., 0.32]] K
        Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                        : grid_latitude(10) = [2.2, ..., -1.76] degrees
                        : grid_longitude(9) = [-4.7, ..., -1.18] degrees
                        : time(1) = [2019-01-01 00:00:00]
        Auxiliary coords: latitude(grid_latitude(10), grid_longitude(9)) = [[53.941, ..., 50.225]] degrees_N
                        : longitude(grid_longitude(9), grid_latitude(10)) = [[2.004, ..., 8.156]] degrees_E
                        : long_name=Grid latitude name(grid_latitude(10)) = [--, ..., b'kappa']
        Cell measures   : measure:area(grid_longitude(9), grid_latitude(10)) = [[2391.9657, ..., 2392.6009]] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(grid_latitude(10), grid_longitude(9)) = [[0.0, ..., 270.0]] m
        >>> print(t.flatten())
        Field: air_temperature (ncvar%ta)
        ---------------------------------
        Data            : air_temperature(key%domainaxis4(90)) K
        Cell methods    : grid_latitude: grid_longitude: mean where land (interval: 0.1 degrees) time(1): maximum
        Field ancils    : air_temperature standard_error(key%domainaxis4(90)) = [0.76, ..., 0.32] K
        Dimension coords: time(1) = [2019-01-01 00:00:00]
        Auxiliary coords: latitude(key%domainaxis4(90)) = [53.941, ..., 50.225] degrees_N
                        : longitude(key%domainaxis4(90)) = [2.004, ..., 8.156] degrees_E
        Cell measures   : measure:area(key%domainaxis4(90)) = [2391.9657, ..., 2392.6009] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : surface_altitude(key%domainaxis4(90)) = [0.0, ..., 270.0] m
        >>> print(t.flatten(['grid_latitude', 'grid_longitude']))
        Field: air_temperature (ncvar%ta)
        ---------------------------------
        Data            : air_temperature(atmosphere_hybrid_height_coordinate(1), key%domainaxis4(90)) K
        Cell methods    : grid_latitude: grid_longitude: mean where land (interval: 0.1 degrees) time(1): maximum
        Field ancils    : air_temperature standard_error(key%domainaxis4(90)) = [0.76, ..., 0.32] K
        Dimension coords: atmosphere_hybrid_height_coordinate(1) = [1.5]
                        : time(1) = [2019-01-01 00:00:00]
        Auxiliary coords: latitude(key%domainaxis4(90)) = [53.941, ..., 50.225] degrees_N
                        : longitude(key%domainaxis4(90)) = [2.004, ..., 8.156] degrees_E
        Cell measures   : measure:area(key%domainaxis4(90)) = [2391.9657, ..., 2392.6009] km2
        Coord references: grid_mapping_name:rotated_latitude_longitude
                        : standard_name:atmosphere_hybrid_height_coordinate
        Domain ancils   : ncvar%a(atmosphere_hybrid_height_coordinate(1)) = [10.0] m
                        : ncvar%b(atmosphere_hybrid_height_coordinate(1)) = [20.0]
                        : surface_altitude(key%domainaxis4(90)) = [0.0, ..., 270.0] m

        >>> t.domain_axes.keys()
        >>> dict_keys(
        ...     ['domainaxis0', 'domainaxis1', 'domainaxis2', 'domainaxis3'])
        >>> t.flatten(return_axis=True)
        (<CF Field: air_temperature(key%domainaxis4(90)) K>,
         'domainaxis4')
        >>> t.flatten('grid_longitude', return_axis=True)
        (<CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>,
         'domainaxis2')
        >>> t.flatten('time', return_axis=True)
        (<CF Field: air_temperature(atmosphere_hybrid_height_coordinate(1), grid_latitude(10), grid_longitude(9)) K>,
         None)

        """
        f = _inplace_enabled_define_and_cleanup(self)

        data_axes = self.get_data_axes()

        if axes is None:
            axes = data_axes
        else:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            axes = [self.domain_axis(x, key=True) for x in axes]
            axes = set(axes).intersection(data_axes)

        # Note that it is important to sort the iaxes, as we rely on
        # the first iaxis in the list being the left-most flattened
        # axis
        iaxes = sorted([data_axes.index(axis) for axis in axes])

        if not len(iaxes):
            if inplace:
                f = None
            if return_axis:
                return f, None
            return f

        if len(iaxes) == 1:
            if inplace:
                f = None
            if return_axis:
                return f, tuple(axes)[0]
            return f

        #        # Make sure that the metadata constructs have the same
        #        # relative axis order as the data (pre-flattening)
        #        f.transpose(f.get_data_axes(), constructs=True, inplace=True)

        # Create the new data axes
        shape = f.shape
        new_data_axes = [
            axis for i, axis in enumerate(data_axes) if i not in iaxes
        ]
        new_axis_size = numpy_prod([shape[i] for i in iaxes])
        new_axis = f.set_construct(self._DomainAxis(new_axis_size))
        new_data_axes.insert(iaxes[0], new_axis)

        # Flatten the field's data
        super(Field, f).flatten(iaxes, inplace=True)

        # Set the new data axes
        f.set_data_axes(new_data_axes)

        # Modify or remove cell methods that span the flatten axes
        for key, cm in f.cell_methods(todict=True).items():
            cm_axes = set(cm.get_axes(()))
            if not cm_axes or cm_axes.isdisjoint(axes):
                continue

            if cm_axes.difference(axes):
                f.del_construct(key)
                continue

            if cm_axes.issubset(axes):
                cm_axes = list(cm_axes)
                set_axes = True
                for i, a in enumerate(cm_axes):
                    sn = None
                    for c in f.coordinates(
                        filter_by_axis=(a,), axis_mode="exact", todict=True
                    ).values():
                        sn = c.get_property("standard_name", None)
                        if sn is not None:
                            break

                    #                    for ctype in (
                    #                        "dimension_coordinate",
                    #                        "auxiliary_coordinate",
                    #                    ):
                    #                        for c in (
                    #                            f.constructs.filter_by_type(ctype, view=True)
                    #                            .filter_by_axis(a, mode="exact", view=True)
                    #                            .values()
                    #                        ):
                    #                            sn = c.get_property("standard_name", None)
                    #                            if sn is not None:
                    #                                break
                    #
                    #                        if sn is not None:
                    #                            break

                    if sn is None:
                        f.del_construct(key)
                        set_axes = False
                        break
                    else:
                        cm_axes[i] = sn

                if set_axes:
                    cm.set_axes(cm_axes)

        # Flatten the constructs that span all of the flattened axes,
        # or all of the flattened axes all bar some which have size 1.
        #        d = dict(f.constructs.filter_by_axis('exact', *axes))
        #        axes2 = [axis for axis in axes
        #                 if f.domain_axes[axis].get_size() > 1]
        #        if axes2 != axes:
        #            d.update(f.constructs.filter_by_axis(
        #                'subset', *axes).filter_by_axis('and', *axes2))

        # Flatten the constructs that span all of the flattened axes,
        # and no others.
        for key, c in f.constructs.filter_by_axis(
            *axes, axis_mode="and", todict=True
        ).items():
            c_axes = f.get_data_axes(key)
            c_iaxes = sorted(
                [c_axes.index(axis) for axis in axes if axis in c_axes]
            )
            c.flatten(c_iaxes, inplace=True)
            new_data_axes = [
                axis for i, axis in enumerate(c_axes) if i not in c_iaxes
            ]
            new_data_axes.insert(c_iaxes[0], new_axis)
            f.set_data_axes(new_data_axes, key=key)

        # Remove constructs that span some, but not all, of the
        # flattened axes
        for key in f.constructs.filter_by_axis(
            *axes, axis_mode="or", todict=True
        ):
            f.del_construct(key)

        # Remove the domain axis constructs for the flattened axes
        for key in axes:
            f.del_construct(key)

        if return_axis:
            return f, new_axis

        return f

    @_deprecated_kwarg_check("i")
    @_inplace_enabled(default=False)
    def roll(self, axis, shift, inplace=False, i=False, **kwargs):
        """Roll the field along a cyclic axis.

        A unique axis is selected with the axes and kwargs parameters.

        .. versionadded:: 1.0

        .. seealso:: `anchor`, `axis`, `cyclic`, `iscyclic`, `period`

        :Parameters:

            axis:
                The cyclic axis to be rolled, defined by that which would
                be selected by passing the given axis description to a
                call of the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected.

            shift: `int`
                The number of places by which the selected cyclic axis is
                to be rolled.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field`
                The rolled field.

        **Examples:**

        Roll the data of the "X" axis one elements to the right:

        >>> f.roll('X', 1)

        Roll the data of the "X" axis three elements to the left:

        >>> f.roll('X', -3)

        """
        # TODODASK - allow multiple roll axes

        axis = self.domain_axis(
            axis,
            key=True,
            default=ValueError(
                f"Can't roll: Bad axis specification: {axis!r}"
            ),
        )

        f = _inplace_enabled_define_and_cleanup(self)

        axis = f._parse_axes(axis)

        # Roll the metadata constructs in-place
        shift = f._roll_constructs(axis, shift)

        iaxes = self._axis_positions(axis, parse=False)
        if iaxes:
            # TODODASK - remove these two lines when multiaxis rolls
            #            are allowed at v4.0.0
            iaxis = iaxes[0]
            shift = shift[0]

            super(Field, f).roll(iaxis, shift, inplace=True)

        return f

    @_deprecated_kwarg_check("i")
    @_manage_log_level_via_verbosity
    def where(
        self,
        condition,
        x=None,
        y=None,
        inplace=False,
        construct=None,
        i=False,
        verbose=None,
        item=None,
        **item_options,
    ):
        """Assign to data elements depending on a condition.

        Data can be changed by assigning to elements that are selected by
        a condition based on the data values of the field construct or on
        its metadata constructs.

        Different values can be assigned to where the conditions are, and
        are not, met.

        **Missing data**

        Data array elements may be set to missing values by assigning them
        to the `cf.masked` constant, or by assignment missing data
        elements of array-valued *x* and *y* parameters.

        By default the data mask is "hard", meaning that masked values can
        not be changed by assigning them to another value. This behaviour
        may be changed by setting the `hardmask` attribute of the field
        construct to `False`, thereby making the data mask "soft" and
        allowing masked elements to be set to non-masked values.

        .. seealso:: `cf.masked`, `hardmask`, `indices`, `mask`,
                     `subspace`, `__setitem__`

        :Parameters:

            condition:
                The condition which determines how to assign values to the
                data.

                In general it may be any scalar or array-like object (such
                as a `numpy`, `Data` or `Field` object) that is
                broadcastable to the shape of the data. Assignment from
                the *x* and *y* parameters will be done where elements of
                the condition evaluate to `True` and `False` respectively.

                *Parameter example:*
                  ``f.where(f.data<0, x=-999)`` will set all data values
                  that are less than zero to -999.

                *Parameter example:*
                  ``f.where(True, x=-999)`` will set all data values to
                  -999. This is equivalent to ``f[...] = -999``.

                *Parameter example:*
                  ``f.where(False, y=-999)`` will set all data values to
                  -999. This is equivalent to ``f[...] = -999``.

                *Parameter example:*
                  If field construct ``f`` has shape ``(5, 3)`` then
                  ``f.where([True, False, True], x=-999, y=cf.masked)``
                  will set data values in columns 0 and 2 to -999, and
                  data values in column 1 to missing data. This works
                  because the condition has shape ``(3,)`` which
                  broadcasts to the field construct's shape.

                If, however, *condition* is a `Query` object then this
                implies a condition defined by applying the query to the
                field construct's data (or a metadata construct's data if
                the *construct* parameter is set).

                *Parameter example:*
                  ``f.where(cf.lt(0), x=-999)`` will set all data values
                  that are less than zero to -999. This is equivalent to
                  ``f.where(f.data<0, x=-999)``.

                If *condition* is another field construct then it is first
                transformed so that it is broadcastable to the data being
                assigned to. This is done by using the metadata constructs
                of the two field constructs to create a mapping of
                physically identical dimensions between the fields, and
                then manipulating the dimensions of other field
                construct's data to ensure that they are broadcastable. If
                either of the field constructs does not have sufficient
                metadata to create such a mapping then an exception will
                be raised. In this case, any manipulation of the
                dimensions must be done manually, and the `Data` instance
                of *construct* (rather than the field construct itself)
                may be used for the condition.

                *Parameter example:*
                  If field construct ``f`` has shape ``(5, 3)`` and ``g =
                  f.transpose() < 0`` then ``f.where(g, x=-999)`` will set
                  all data values that are less than zero to -999,
                  provided there are sufficient metadata for the data
                  dimensions to be mapped. However, ``f.where(g.data,
                  x=-999)`` will always fail in this example, because the
                  shape of the condition is ``(3, 5)``, which does not
                  broadcast to the shape of the ``f``.

            x, y: *optional*
                Specify the assignment values. Where the condition
                evaluates to `True`, assign to the field construct's data
                from *x*, and where the condition evaluates to `False`,
                assign to the field construct's data from *y*. The *x* and
                *y* parameters are each one of:

                * `None`. The appropriate data elements array are
                  unchanged. This the default.

                * Any scalar or array-like object (such as a `numpy`,
                  `Data` or `Field` object) that is broadcastable to the
                  shape of the data.

            ..

                *Parameter example:*
                  ``f.where(condition)``, for any ``condition``, returns a
                  field construct with identical data values.

                *Parameter example:*
                  ``f.where(cf.lt(0), x=-f.data, y=cf.masked)`` will
                  change the sign of all negative data values, and set all
                  other data values to missing data.

                If *x* or *y* is another field construct then it is first
                transformed so that its data is broadcastable to the data
                being assigned to. This is done by using the metadata
                constructs of the two field constructs to create a mapping
                of physically identical dimensions between the fields, and
                then manipulating the dimensions of other field
                construct's data to ensure that they are broadcastable. If
                either of the field constructs does not have sufficient
                metadata to create such a mapping then an exception will
                be raised. In this case, any manipulation of the
                dimensions must be done manually, and the `Data` instance
                of *x* or *y* (rather than the field construct itself) may
                be used for the condition.

                *Parameter example:*
                  If field construct ``f`` has shape ``(5, 3)`` and ``g =
                  f.transpose() * 10`` then ``f.where(cf.lt(0), x=g)``
                  will set all data values that are less than zero to the
                  equivalent elements of field construct ``g``, provided
                  there are sufficient metadata for the data dimensions to
                  be mapped. However, ``f.where(cf.lt(0), x=g.data)`` will
                  always fail in this example, because the shape of the
                  condition is ``(3, 5)``, which does not broadcast to the
                  shape of the ``f``.

            construct: `str`, optional
                Define the condition by applying the *construct* parameter
                to the given metadata construct's data, rather then the
                data of the field construct. Must be

                * The identity or key of a metadata coordinate construct
                  that has data.

            ..

                The *construct* parameter selects the metadata construct
                that is returned by this call of the field construct's
                `construct` method: ``f.construct(construct)``. See
                `cf.Field.construct` for details.

                *Parameter example:*
                  ``f.where(cf.wi(-30, 30), x=cf.masked,
                  construct='latitude')`` will set all data values within
                  30 degrees of the equator to missing data.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            item: deprecated at version 3.0.0
                Use the *construct* parameter instead.

            item_options: deprecated at version 3.0.0

        :Returns:

            `Field` or `None`
                A new field construct with an updated data array, or
                `None` if the operation was in-place.

        **Examples:**

        Set data array values to 15 everywhere:

        >>> f.where(True, 15)

        This example could also be done with subspace assignment:

        >>> f[...] = 15

        Set all negative data array values to zero and leave all other
        elements unchanged:

        >>> g = f.where(f<0, 0)

        Multiply all positive data array elements by -1 and set other data
        array elements to 3.14:

        >>> g = f.where(f>0, -f, 3.14)

        Set all values less than 280 and greater than 290 to missing data:

        >>> g = f.where((f < 280) | (f > 290), cf.masked)

        This example could also be done with a `Query` object:

        >>> g = f.where(cf.wo(280, 290), cf.masked)

        or equivalently:

        >>> g = f.where(f==cf.wo(280, 290), cf.masked)

        Set data array elements in the northern hemisphere to missing data
        in-place:

        >>> condition = f.domain_mask(latitude=cf.ge(0))
        >>> f.where(condition, cf.masked, inplace=True)

        Missing data can only be changed if the mask is "soft":

        >>> f[0] = cf.masked
        >>> g = f.where(True, 99)
        >>> print(g[0],array)
        [--]
        >>> f.hardmask = False
        >>> g = f.where(True, 99)
        >>> print(g[0],array)
        [99]

        This in-place example could also be done with subspace assignment
        by indices:

        >>> northern_hemisphere = f.indices(latitude=cf.ge(0))
        >>> f.subspace[northern_hemisphere] = cf.masked

        Set a polar rows to their zonal-mean values:

        >>> condition = f.domain_mask(latitude=cf.set([-90, 90]))
        >>> g = f.where(condition, f.collapse('longitude: mean'))

        """
        if item is not None:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "where",
                {"item": item},
                "Use keyword 'construct' instead.",
            )  # pragma: no cover

        if item_options:
            _DEPRECATION_ERROR_KWARGS(
                self, "where", {"item_options": item_options}
            )  # pragma: no cover

        if x is None and y is None:
            if inplace:
                return
            return self.copy()

        self_class = self.__class__

        if isinstance(condition, self_class):
            # --------------------------------------------------------
            # Condition is another field construct
            # --------------------------------------------------------
            condition = self._conform_for_assignment(condition)

        elif construct is not None:
            if not isinstance(condition, Query):
                raise ValueError(
                    "A condition on a metadata construct must be a Query "
                    "object"
                )

            # Apply the Query to a metadata construct of the field,
            # making sure that the construct's data is broadcastable
            # to the field's data.
            g = self.transpose(self.get_data_axes(), constructs=True)

            key = g.construct_key(
                construct,
                default=ValueError(
                    f"Can't identify unique {construct!r} construct"
                ),
            )
            construct = g.constructs[key]

            construct_data_axes = g.get_data_axes(key, default=None)
            if construct_data_axes is None:
                raise ValueError("TODO")

            data_axes = g.get_data_axes()

            construct_data = construct.get_data(None, _fill_value=False)
            if construct_data is None:
                raise ValueError(f"{construct!r} has no data")

            if construct_data_axes != data_axes:
                s = [
                    i
                    for i, axis in enumerate(construct_data_axes)
                    if axis not in data_axes
                ]
                if s:
                    construct_data.squeeze(s, inplace=True)
                    construct_data_axes = [
                        axis
                        for axis in construct_data_axes
                        if axis not in data_axes
                    ]

                for i, axis in enumerate(data_axes):
                    if axis not in construct_data_axes:
                        construct_data.insert_dimension(i, inplace=True)

            condition = condition.evaluate(construct_data)

        if x is not None and isinstance(x, self_class):
            x = self._conform_for_assignment(x)

        if y is not None and isinstance(y, self_class):
            y = self._conform_for_assignment(y)

        return super().where(condition, x, y, inplace=inplace, verbose=verbose)

    @property
    def subspace(self):
        """Create a subspace of the field construct.

        Creation of a new field construct which spans a subspace of the
        domain of an existing field construct is achieved either by
        identifying indices based on the metadata constructs (subspacing
        by metadata) or by indexing the field construct directly
        (subspacing by index).

        The subspacing operation, in either case, also subspaces any
        metadata constructs of the field construct (e.g. coordinate
        metadata constructs) which span any of the domain axis constructs
        that are affected. The new field construct is created with the
        same properties as the original field construct.

        **Subspacing by metadata**

        Subspacing by metadata, signified by the use of round brackets,
        selects metadata constructs and specifies conditions on their
        data. Indices for subspacing are then automatically inferred from
        where the conditions are met.

        Metadata constructs and the conditions on their data are defined
        by keyword parameters.

        * Any domain axes that have not been identified remain unchanged.

        * Multiple domain axes may be subspaced simultaneously, and it
          doesn't matter which order they are specified in.

        * Subspace criteria may be provided for size 1 domain axes that
          are not spanned by the field construct's data.

        * Explicit indices may also be assigned to a domain axis
          identified by a metadata construct, with either a Python `slice`
          object, or a sequence of integers or booleans.

        * For a dimension that is cyclic, a subspace defined by a slice or
          by a `Query` instance is assumed to "wrap" around the edges of
          the data.

        * Conditions may also be applied to multi-dimensional metadata
          constructs. The "compress" mode is still the default mode (see
          the positional arguments), but because the indices may not be
          acting along orthogonal dimensions, some missing data may still
          need to be inserted into the field construct's data.

        **Subspacing by index**

        Subspacing by indexing, signified by the use of square brackets,
        uses rules that are very similar to the numpy indexing rules, the
        only differences being:

        * An integer index i specified for a dimension reduces the size of
          this dimension to unity, taking just the i-th element, but keeps
          the dimension itself, so that the rank of the array is not
          reduced.

        * When two or more dimensions’ indices are sequences of integers
          then these indices work independently along each dimension
          (similar to the way vector subscripts work in Fortran). This is
          the same indexing behaviour as on a Variable object of the
          netCDF4 package.

        * For a dimension that is cyclic, a range of indices specified by
          a `slice` that spans the edges of the data (such as ``-2:3`` or
          ``3:-2:-1``) is assumed to "wrap" around, rather then producing
          a null result.


        .. seealso:: `indices`, `squeeze`, `where`, `__getitem__`

        :Parameters:

            positional arguments: *optional*
                There are three modes of operation, each of which provides
                a different type of subspace:

                ==============  ==========================================
                *argument*      Description
                ==============  ==========================================
                ``'compress'``  This is the default mode. Unselected
                                locations are removed to create the
                                returned subspace. Note that if a
                                multi-dimensional metadata construct is
                                being used to define the indices then some
                                missing data may still be inserted at
                                unselected locations.

                ``'envelope'``  The returned subspace is the smallest that
                                contains all of the selected
                                indices. Missing data is inserted at
                                unselected locations within the envelope.

                ``'full'``      The returned subspace has the same domain
                                as the original field construct. Missing
                                data is inserted at unselected locations.

                ``'test'``      May be used on its own or in addition to
                                one of the other positional arguments. Do
                                not create a subspace, but return `True`
                                or `False` depending on whether or not it
                                is possible to create specified the
                                subspace.
                ==============  ==========================================

            keyword parameters: *optional*
                A keyword name is an identity of a metadata construct, and
                the keyword value provides a condition for inferring
                indices that apply to the dimension (or dimensions)
                spanned by the metadata construct's data. Indices are
                created that select every location for which the metadata
                construct's data satisfies the condition.

        :Returns:

            `Field` or `bool`
                An independent field construct containing the subspace of
                the original field. If the ``'test'`` positional argument
                has been set then return `True` or `False` depending on
                whether or not it is possible to create specified
                subspace.

        **Examples:**

        There are further worked examples
        :ref:`in the tutorial <Subspacing-by-metadata>`.

        >>> g = f.subspace(X=112.5)
        >>> g = f.subspace(X=112.5, latitude=cf.gt(-60))
        >>> g = f.subspace(latitude=cf.eq(-45) | cf.ge(20))
        >>> g = f.subspace(X=[1, 2, 4], Y=slice(None, None, -1))
        >>> g = f.subspace(X=cf.wi(-100, 200))
        >>> g = f.subspace(X=slice(-2, 4))
        >>> g = f.subspace(Y=[True, False, True, True, False])
        >>> g = f.subspace(T=410.5)
        >>> g = f.subspace(T=cf.dt('1960-04-16'))
        >>> g = f.subspace(T=cf.wi(cf.dt('1962-11-01'),
        ...                        cf.dt('1967-03-17 07:30')))
        >>> g = f.subspace('compress', X=[1, 2, 4, 6])
        >>> g = f.subspace('envelope', X=[1, 2, 4, 6])
        >>> g = f.subspace('full', X=[1, 2, 4, 6])
        >>> g = f.subspace(latitude=cf.wi(51, 53))

        >>> g = f.subspace[::-1, 0]
        >>> g = f.subspace[:, :, 1]
        >>> g = f.subspace[:, 0]
        >>> g = f.subspace[..., 6:3:-1, 3:6]
        >>> g = f.subspace[0, [2, 3, 9], [4, 8]]
        >>> g = t.subspace[0, :, -2]
        >>> g = f.subspace[0, [2, 3, 9], [4, 8]]
        >>> g = f.subspace[:, -2:3]
        >>> g = f.subspace[:, 3:-2:-1]
        >>> g = f.subspace[..., [True, False, True, True, False]]

        """
        return SubspaceField(self)

    def section(self, axes=None, stop=None, **kwargs):
        """Return a FieldList of m dimensional sections of a Field of n
        dimensions, where M <= N.

        :Parameters:

            axes: optional
                A query for the m axes that define the sections of the
                Field as accepted by the Field object's axes method. The
                keyword arguments are also passed to this method. See
                TODO cf.Field.axes for details. If an axis is returned that is
                not a data axis it is ignored, since it is assumed to be a
                dimension coordinate of size 1.

            stop: `int`, optional
                Stop after taking this number of sections and return. If
                *stop* is `None` all sections are taken.

        :Returns:

            `FieldList`
                The sections of the field construct.

        **Examples:**

        Section a field into 2D longitude/time slices, checking the units:

        >>> f.section({None: 'longitude', units: 'radians'},
        ...           {None: 'time', 'units': 'days since 2006-01-01 00:00:00'})

        Section a field into 2D longitude/latitude slices, requiring
        exact names:

        >>> f.section(['latitude', 'longitude'], exact=True)

        Section a field into 2D longitude/latitude slices, showing
        the results:

        >>> f
        <CF Field: eastward_wind(model_level_number(6), latitude(145), longitude(192)) m s-1>
        >>> f.section(('X', 'Y'))
        [<CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>,
         <CF Field: eastward_wind(model_level_number(1), latitude(145), longitude(192)) m s-1>]

        """
        return FieldList(_section(self, axes, data=False, stop=stop, **kwargs))


    @_deprecated_kwarg_check("i")
    @_inplace_enabled(default=False)
    def derivative(
        self,
        axis,
        wrap=None,
        one_sided_at_boundary=False,
        inplace=False,
        i=False,
        cyclic=None,
    ):
        """Return the derivative along the specified axis.

        The derivative is calculated by centred finite differences along
        the specified axis.

        If the axis is cyclic then the boundary is wrapped around,
        otherwise missing values are used at the boundary unless one-sided
        finite differences are requested.

        :Parameters:

            axis:
                The axis , defined by that which would be selected by
                passing the given axis description to a call of the field
                construct's `domain_axis` method. For example, for a value
                of ``'X'``, the domain axis construct returned by
                ``f.domain_axis('X')`` is selected.

            wrap: `bool`, optional
                If True then the boundary is wrapped around, otherwise the
                value of *one_sided_at_boundary* determines the boundary
                condition. If `None` then the cyclicity of the axis is
                autodetected.

            one_sided_at_boundary: `bool`, optional
                If True then one-sided finite differences are used at the
                boundary, otherwise missing values are used.

            {{inplace: `bool`, optional}}
                If True then do the operation in-place and return `None`.

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Field` or `None`
                TODO , or `None` if the operation was in-place.

        **Examples:**

        TODO

        """
        if cyclic:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "derivative",
                {"cyclic": cyclic},
                "Use the 'wrap' keyword instead",
            )  # pragma: no cover

        #        try:
        #            scipy_convolve1d
        #        except NameError:
        #            raise ImportError(
        #                "Must install scipy to use the derivative method.")

        # Retrieve the axis
        axis = self.domain_axis(axis, key=True, default=None)
        if axis is None:
            raise ValueError("Invalid axis specifier")

        coord = self.dimension_coordinate(filter_by_axis=(axis,), default=None)
        if coord is None:
            raise ValueError("Axis specified is not unique.")

        # Get the axis index
        axis_index = self.get_data_axes().index(axis)

        # Automatically detect the cyclicity of the axis if cyclic is None
        if wrap is None:
            wrap = self.iscyclic(axis)

        # Set the boundary conditions
        if wrap:
            mode = "wrap"
        elif one_sided_at_boundary:
            mode = "nearest"
        else:
            mode = "constant"

        f = _inplace_enabled_define_and_cleanup(self)

        # Find the finite difference of the field
        f.convolution_filter(
            [1, 0, -1], axis=axis, mode=mode, update_bounds=False, inplace=True
        )

        # Find the finite difference of the axis
        d = coord.data.convolution_filter(
            window=[1, 0, -1], axis=0, mode=mode, cval=numpy_nan
        )

        # Reshape the finite difference of the axis for broadcasting
        for _ in range(self.ndim - 1 - axis_index):
            d.insert_dimension(position=1, inplace=True)

        # Find the derivative
        f.data /= d

        # Update the standard name and long name
        standard_name = f.get_property("standard_name", None)
        long_name = f.get_property("long_name", None)
        if standard_name is not None:
            del f.standard_name
            f.long_name = f"derivative of {standard_name}"
        elif long_name is not None:
            f.long_name = f"derivative of {long_name}"

        return f

    # ----------------------------------------------------------------
    # Aliases
    # ----------------------------------------------------------------
    def field_anc(
        self,
        *identity,
        key=False,
        default=ValueError(),
        item=False,
        **filter_kwargs,
    ):
        """Alias for `cf.Field.field_ancillary`."""
        return self.field_ancillary(
            *identity, key=key, default=default, item=item, **filter_kwargs
        )

    def field_ancs(self, *identities, **filter_kwargs):
        """Alias for `field_ancillaries`."""
        return self.field_ancillaries(*identities, **filter_kwargs)
