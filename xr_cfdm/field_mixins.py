from collections import namedtuple
from functools import reduce
from operator import mul as operator_mul
from operator import itemgetter

import logging

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

from .cf_python import (
    DimensionCoordinate,
    CellMethod
)

from .cf_python.constants import masked as cf_masked

from .cf_python.functions import parse_indices, chunksize, _section
from .cf_python.functions import relaxed_identities as cf_relaxed_identities
from .cf_python.query import Query, ge, gt, le, lt, eq
from .cf_python.regrid import Regrid
from .cf_python.timeduration import TimeDuration
from .units import Units
from .cf_python.subspacefield import SubspaceField

from .data.data import Data

from . import mixin

from .cf_python.decorators import (
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _deprecated_kwarg_check,
    _manage_log_level_via_verbosity,
)

from .cf_python.functions import (
    _DEPRECATION_ERROR,
    _DEPRECATION_ERROR_ARG,
    _DEPRECATION_ERROR_KWARGS,
    _DEPRECATION_ERROR_METHOD,
    _DEPRECATION_ERROR_KWARG_VALUE,
    DeprecationError,
)

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# Commonly used units
# --------------------------------------------------------------------
# _units_degrees = Units("degrees")
_units_radians = Units("radians")
_units_metres = Units("m")
_units_1 = Units("1")

# --------------------------------------------------------------------
# Map each allowed input collapse method name to its corresponding
# Data method. Input collapse methods not in this sictionary are
# assumed to have a corresponding Data method with the same name.
# --------------------------------------------------------------------
_collapse_methods = {
    **{
        name: name
        for name in [
            "mean",  # results in 'mean': 'mean' entry, etc.
            "mean_absolute_value",
            "mean_of_upper_decile",
            "max",
            "maximum_absolute_value",
            "min",
            "max",
            "minimum_absolute_value",
            "mid_range",
            "range",
            "median",
            "sd",
            "sum",
            "sum_of_squares",
            "integral",
            "root_mean_square",
            "var",
            "sample_size",
            "sum_of_weights",
            "sum_of_weights2",
        ]
    },
    **{  # non-identical mapped names:
        "avg": "mean",
        "average": "mean",
        "maximum": "max",
        "minimum": "min",
        "standard_deviation": "sd",
        "variance": "var",
    },
}

# --------------------------------------------------------------------
# Map each allowed input collapse method name to its corresponding CF
# cell method.
# --------------------------------------------------------------------
_collapse_cell_methods = {
    **{
        name: name
        for name in [
            "point",
            "mean",
            "mean_absolute_value",
            "mean_of_upper_decile",
            "maximum",
            "maximum_absolute_value",
            "minimum",
            "minimum_absolute_value",
            "mid_range",
            "range",
            "median",
            "standard_deviation",
            "sum",
            "root_mean_square",
            "sum_of_squares",
            "variance",
        ]
    },
    **{  # non-identical mapped names:
        "avg": "mean",
        "average": "mean",
        "max": "maximum",
        "min": "minimum",
        "sd": "standard_deviation",
        "integral": "sum",
        "var": "variance",
        "sample_size": "point",
        "sum_of_weights": "sum",
        "sum_of_weights2": "sum",
    },
}

# --------------------------------------------------------------------
# These Data methods may be weighted
# --------------------------------------------------------------------
_collapse_weighted_methods = set(
    (
        "mean",
        "mean_absolute_value",
        "mean_of_upper_decile",
        "avg",
        "average",
        "sd",
        "standard_deviation",
        "var",
        "variance",
        # 'sum_of_weights',
        # 'sum_of_weights2',
        "integral",
        "root_mean_square",
    )
)

# --------------------------------------------------------------------
# These Data methods may specify a number of degrees of freedom
# --------------------------------------------------------------------
_collapse_ddof_methods = set(
    (
        "sd",
        "var",
    )
)


conservative_regridding_methods = [
    "conservative",
    "conservative_1st",
    "conservative_2nd",
]
regridding_methods = [
    "linear",  # prefer over 'bilinear' as of v3.2.0
    "bilinear",  # only for backward compatibility, use & document 'linear'
    "patch",
    "nearest_stod",
    "nearest_dtos",
] + conservative_regridding_methods



class XRFieldRegridMixin():
    # ----------------------------------------------------------------
    # Worker functions for regridding
    # ----------------------------------------------------------------
    def _regrid_get_latlong(self, name, axes=None):
        """Retrieve the latitude and longitude coordinates of this field
        and associated information. If 1D lat/long coordinates are found
        then these are returned. Otherwise, 2D lat/long coordinates are
        searched for and if found returned.

        :Parameters:

            name: `str`
                A name to identify the field in error messages. Either
                ``'source'`` or ``'destination'``.

            axes: `dict`, optional
                A dictionary specifying the X and Y axes, with keys
                ``'X'`` and ``'Y'``.

                *Parameter example:*
                  ``axes={'X': 'ncdim%x', 'Y': 'ncdim%y'}``

                *Parameter example:*
                  ``axes={'X': 1, 'Y': 0}``

        :Returns:

            axis_keys: `list`
                The keys of the x and y dimension coordinates.

            axis_sizes: `list`
                The sizes of the x and y dimension coordinates.

            coord_keys: `list`
                The keys of the x and y coordinate (1D dimension
                coordinate, or 2D auxilliary coordinates).

            coords: `list`
                The x and y coordinates (1D dimension coordinates or 2D
                auxilliary coordinates).

            coords_2D: `bool`
                True if 2D auxiliary coordinates are returned or if 1D X
                and Y coordinates are returned, which are not long/lat.

        """
        data_axes = self.constructs.data_axes()

        if axes is None:
            # Retrieve the field construct's X and Y dimension
            # coordinates
            x_key, x = self.dimension_coordinate(
                "X",
                item=True,
                default=ValueError(
                    f"No unique X dimension coordinate found for the {name} "
                    "field. If none is present you "
                    "may need to specify the axes keyword."
                ),
            )
            y_key, y = self.dimension_coordinate(
                "Y",
                item=True,
                default=ValueError(
                    f"No unique Y dimension coordinate found for the {name} "
                    "field. If none is present you "
                    "may need to specify the axes keyword."
                ),
            )

            x_axis = data_axes[x_key][0]
            y_axis = data_axes[y_key][0]

            x_size = x.size
            y_size = y.size
        else:
            # --------------------------------------------------------
            # Source axes have been provided
            # --------------------------------------------------------
            for key in ("X", "Y"):
                if key not in axes:
                    raise ValueError(
                        f"Key {key!r} must be specified for axes of {name} "
                        "field."
                    )

            if axes["X"] in (1, 0) and axes["Y"] in (0, 1):
                # Axes specified by integer position in dimensions of
                # lat and lon 2-d auxiliary coordinates
                if axes["X"] == axes["Y"]:
                    raise ValueError("TODO 0")

                lon_key, lon = self.auxiliary_coordinate(
                    "X", item=True, filter_by_naxes=(2,), default=(None, None)
                )
                lat_key, lat = self.auxiliary_coordinate(
                    "Y", item=True, filter_by_naxes=(2,), default=(None, None)
                )
                if lon is None:
                    raise ValueError("TODO x")
                if lat is None:
                    raise ValueError("TODO y")

                if lat.shape != lon.shape:
                    raise ValueError("TODO 222222")

                lon_axes = data_axes[lon_key]
                lat_axes = data_axes[lat_key]
                if lat_axes != lon_axes:
                    raise ValueError("TODO 3333333")

                x_axis = lon_axes[axes["X"]]
                y_axis = lat_axes[axes["Y"]]
            else:
                x_axis = self.domain_axis(
                    axes["X"],
                    key=True,
                    default=ValueError(
                        f"'X' axis specified for {name} field not found."
                    ),
                )

                y_axis = self.domain_axis(
                    axes["Y"],
                    key=True,
                    default=ValueError(
                        f"'Y' axis specified for {name} field not found."
                    ),
                )

            domain_axes = self.domain_axes(todict=True)
            x_size = domain_axes[x_axis].get_size()
            y_size = domain_axes[y_axis].get_size()

        axis_keys = [x_axis, y_axis]
        axis_sizes = [x_size, y_size]

        # If 1-d latitude and longitude coordinates for the field are
        # not found search for 2-d auxiliary coordinates.
        if (
            axes is not None
            or not x.Units.islongitude
            or not y.Units.islatitude
        ):
            lon_found = False
            lat_found = False

            for key, aux in self.auxiliary_coordinates(
                filter_by_naxes=(2,), todict=True
            ).items():
                if aux.Units.islongitude:
                    if lon_found:
                        raise ValueError(
                            "The 2-d auxiliary longitude coordinate "
                            f"of the {name} field is not unique."
                        )
                    else:
                        lon_found = True
                        x = aux
                        x_key = key

                if aux.Units.islatitude:
                    if lat_found:
                        raise ValueError(
                            "The 2-d auxiliary latitude coordinate "
                            f"of the {name} field is not unique."
                        )
                    else:
                        lat_found = True
                        y = aux
                        y_key = key

            if not lon_found or not lat_found:
                raise ValueError(
                    "Both longitude and latitude coordinates "
                    f"were not found for the {name} field."
                )

            if axes is not None:
                if set(axis_keys) != set(data_axes[x_key]):
                    raise ValueError(
                        "Axes of longitude do not match "
                        f"those specified for {name} field."
                    )

                if set(axis_keys) != set(data_axes[y_key]):
                    raise ValueError(
                        "Axes of latitude do not match "
                        f"those specified for {name} field."
                    )

            coords_2D = True
        else:
            coords_2D = False
            # Check for size 1 latitude or longitude dimensions
            if x_size == 1 or y_size == 1:
                raise ValueError(
                    "Neither the longitude nor latitude dimension coordinates "
                    f"of the {name} field can be of size 1."
                )

        coord_keys = [x_key, y_key]
        coords = [x, y]

        return axis_keys, axis_sizes, coord_keys, coords, coords_2D

    def _regrid_get_cartesian_coords(self, name, axes):
        """Retrieve the specified cartesian dimension coordinates of the
        field and their corresponding keys.

        :Parameters:

            name: `str`
                A name to identify the field in error messages.

            axes: sequence of `str`
                Specifiers for the dimension coordinates to be
                retrieved. See cf.Field.axes for details.

        :Returns:

            axis_keys: `list`
                A list of the keys of the dimension coordinates retrieved.

            coords: `list`
                A list of the dimension coordinates retrieved.

        """
        axis_keys = []
        for axis in axes:
            key = self.domain_axis(axis, key=True)
            axis_keys.append(key)

        coords = []
        for key in axis_keys:
            d = self.dimension_coordinate(filter_by_axis=(key,), default=None)
            if d is None:
                raise ValueError(
                    f"No unique {name} dimension coordinate "
                    f"matches key {key!r}."
                )

            coords.append(d.copy())

        return axis_keys, coords

    @_deprecated_kwarg_check("i")
    def _regrid_get_axis_indices(self, axis_keys, i=False):
        """Get axis indices and their orders in rank of this field.

        :Parameters:

            axis_keys: sequence
                A sequence of axis specifiers.

            i: deprecated at version 3.0.0

        :Returns:

            axis_indices: list
                A list of the indices of the specified axes.

            order: ndarray
                A numpy array of the rank order of the axes.

        """
        data_axes = self.get_data_axes()

        # Get the positions of the axes
        axis_indices = []
        for axis_key in axis_keys:
            try:
                axis_index = data_axes.index(axis_key)
            except ValueError:
                self.insert_dimension(axis_key, position=0, inplace=True)
                axis_index = data_axes.index(axis_key)

            axis_indices.append(axis_index)

        # Get the rank order of the positions of the axes
        tmp = numpy_array(axis_indices)
        tmp = tmp.argsort()
        order = numpy_empty((len(tmp),), dtype=int)
        order[tmp] = numpy_arange(len(tmp))

        return axis_indices, order

    def _regrid_get_coord_order(self, axis_keys, coord_keys):
        """Get the ordering of the axes for each N-D auxiliary
        coordinate.

        :Parameters:

            axis_keys: sequence
                A sequence of axis keys.

            coord_keys: sequence
                A sequence of keys for each to the N-D auxiliary
                coordinates.

        :Returns:

            `list`
                A list of lists specifying the ordering of the axes for
                each N-D auxiliary coordinate.

        """
        coord_axes = [
            self.get_data_axes(coord_key) for coord_key in coord_keys
        ]
        coord_order = [
            [coord_axis.index(axis_key) for axis_key in axis_keys]
            for coord_axis in coord_axes
        ]
        return coord_order

    def _regrid_get_section_shape(self, axis_sizes, axis_indices):
        """Get the shape of each regridded section.

        :Parameters:

            axis_sizes: sequence
                A sequence of the sizes of each axis along which the
                section.  will be taken

            axis_indices: sequence
                A sequence of the same length giving the axis index of
                each axis.

        :Returns:

            shape: `list`
                A list of integers defining the shape of each section.

        """

        shape = [1] * self.ndim
        for i, axis_index in enumerate(axis_indices):
            shape[axis_index] = axis_sizes[i]

        return shape

    @classmethod
    def _regrid_check_bounds(
        cls, src_coords, dst_coords, method, ext_coords=None
    ):
        """Check the bounds of the coordinates for regridding and
        reassign the regridding method if auto is selected.

        :Parameters:

            src_coords: sequence
                A sequence of the source coordinates.

            dst_coords: sequence
                A sequence of the destination coordinates.

            method: `str`
                A string indicating the regrid method.

            ext_coords: `None` or sequence
                If a sequence of extension coordinates is present these
                are also checked. Only used for cartesian regridding when
                regridding only 1 (only 1!) dimension of a n>2 dimensional
                field. In this case we need to provided the coordinates of
                the dimensions that aren't being regridded (that are the
                same in both src and dst grids) so that we can create a
                sensible ESMF grid object.

        :Returns:

            `None`

        """
        if method not in conservative_regridding_methods:
            return

        for name, coords in zip(
            ("Source", "Destination"), (src_coords, dst_coords)
        ):
            for coord in coords:
                if not coord.has_bounds():
                    raise ValueError(
                        f"{name} {coord!r} coordinates must have bounds "
                        "for conservative regridding."
                    )

                if not coord.contiguous(overlap=False):
                    raise ValueError(
                        f"{name} {coord!r} coordinates must have "
                        "contiguous, non-overlapping bounds "
                        "for conservative regridding."
                    )

        if ext_coords is not None:
            for coord in ext_coords:
                if not coord.has_bounds():
                    raise ValueError(
                        f"{coord!r} dimension coordinates must have "
                        "bounds for conservative regridding."
                    )
                if not coord.contiguous(overlap=False):
                    raise ValueError(
                        f"{coord!r} dimension coordinates must have "
                        "contiguous, non-overlapping bounds "
                        "for conservative regridding."
                    )

    @classmethod
    def _regrid_check_method(cls, method):
        """Check the regrid method is valid and if not raise an error.

        :Parameters:

            method: `str`
                The regridding method.

        """
        if method is None:
            raise ValueError("Can't regrid: Must select a regridding method")

        elif method not in regridding_methods:
            raise ValueError(f"Can't regrid: Invalid method: {method!r}")

        elif method == "bilinear":  # TODO use logging.info() once have logging
            print(
                "Note the 'bilinear' method argument has been renamed to "
                "'linear' at version 3.2.0. It is still supported for now "
                "but please use 'linear' in future. "
                "'bilinear' will be removed at version 4.0.0"
            )

    @classmethod
    def _regrid_check_use_src_mask(cls, use_src_mask, method):
        """Check that use_src_mask is True for all methods other than
        nearest_stod and if not raise an error.

        :Parameters:

            use_src_mask: `bool`
                Whether to use the source mask in regridding.

            method: `str`
                The regridding method.

        """
        if not use_src_mask and not method == "nearest_stod":
            raise ValueError(
                "use_src_mask can only be False when using the "
                "nearest_stod method."
            )

    def _regrid_get_reordered_sections(
        self, axis_order, regrid_axes, regrid_axis_indices
    ):
        """Get a dictionary of the data sections for regridding and a
        list of its keys reordered if necessary so that they will be
        looped over in the order specified in axis_order.

        :Parameters:

            axis_order: `None` or sequence of axes specifiers.
                If `None` then the sections keys will not be reordered. If
                a particular axis is one of the regridding axes or is not
                found then a ValueError will be raised.

            regrid_axes: sequence
                A sequence of the keys of the regridding axes.

            regrid_axis_indices: sequence
                A sequence of the indices of the regridding axes.

        :Returns:

            section_keys: `list`
                An ordered list of the section keys.

            sections: `dict`
                A dictionary of the data sections for regridding.

        """

        # If we had dynamic masking, we wouldn't need this method, we could
        # sdimply replace it in regrid[sc] with a call to
        # Data.section. However, we don't have it, so this allows us to
        # possibibly reduce the number of trasnistions between different masks
        # - each change is slow.
        data_axes = self.get_data_axes()

        axis_indices = []
        if axis_order is not None:
            for axis in axis_order:
                axis_key = self.dimension_coordinate(
                    filter_by_axis=(axis,),
                    default=None,
                    key=True,
                )
                if axis_key is not None:
                    if axis_key in regrid_axes:
                        raise ValueError("Cannot loop over regridding axes.")

                    try:
                        axis_indices.append(data_axes.index(axis_key))
                    except ValueError:
                        # The axis has been squeezed so do nothing
                        pass

                else:
                    raise ValueError(f"Axis not found: {axis!r}")

        # Section the data
        sections = self.data.section(regrid_axis_indices)

        # Reorder keys correspondingly if required
        if axis_indices:
            section_keys = sorted(
                sections.keys(), key=itemgetter(*axis_indices)
            )
        else:
            section_keys = sections.keys()

        return section_keys, sections

    def _regrid_get_destination_mask(
        self, dst_order, axes=("X", "Y"), cartesian=False, coords_ext=None
    ):
        """Get the mask of the destination field.

        :Parameters:

            dst_order: sequence, optional
                The order of the destination axes.

            axes: optional
                The axes the data is to be sectioned along.

            cartesian: `bool`, optional
                Whether the regridding is Cartesian or spherical.

            coords_ext: sequence, optional
                In the case of Cartesian regridding, extension coordinates
                (see _regrid_check_bounds for details).

        :Returns:

            dst_mask: `numpy.ndarray`
                A numpy array with the mask.

        """
        data_axes = self.get_data_axes()

        indices = {axis: [0] for axis in data_axes if axis not in axes}

        f = self.subspace(**indices)
        f = f.squeeze(tuple(indices)).transpose(dst_order)

        dst_mask = f.mask.array

        if cartesian:
            tmp = []
            for coord in coords_ext:
                tmp.append(coord.size)
                dst_mask = numpy_tile(dst_mask, tmp + [1] * dst_mask.ndim)

        return dst_mask

    def _regrid_fill_fields(self, src_data, srcfield, dstfield):
        """Fill the source field with data and the destination field
        with fill values.

        :Parameters:

            src_data: ndarray
                The data to fill the source field with.

            srcfield: ESMPy Field
                The source field.

            dstfield: ESMPy Field
                The destination field. This get always gets initialised with
                missing values.

        """
        srcfield.data[...] = numpy_ma_MaskedArray(src_data, copy=False).filled(
            self.fill_value(default="netCDF")
        )
        dstfield.data[...] = self.fill_value(default="netCDF")

    def _regrid_compute_field_mass(
        self,
        _compute_field_mass,
        k,
        srcgrid,
        srcfield,
        srcfracfield,
        dstgrid,
        dstfield,
    ):
        """Compute the field mass for conservative regridding. The mass
        should be the same before and after regridding.

        :Parameters:

            _compute_field_mass: `dict`
                A dictionary for the results.

            k: `tuple`
                A key identifying the section of the field being regridded.

            srcgrid: ESMPy grid
                The source grid.

            srcfield: ESMPy grid
                The source field.

            srcfracfield: ESMPy field
                Information about the fraction of each cell of the source
                field used in regridding.

            dstgrid: ESMPy grid
                The destination grid.

            dstfield: ESMPy field
                The destination field.

        """
        if not isinstance(_compute_field_mass, dict):
            raise ValueError(
                "Expected _compute_field_mass to be a dictionary."
            )

        fill_value = self.fill_value(default="netCDF")

        # Calculate the mass of the source field
        srcareafield = Regrid.create_field(srcgrid, "srcareafield")
        srcmass = Regrid.compute_mass_grid(
            srcfield,
            srcareafield,
            dofrac=True,
            fracfield=srcfracfield,
            uninitval=fill_value,
        )

        # Calculate the mass of the destination field
        dstareafield = Regrid.create_field(dstgrid, "dstareafield")
        dstmass = Regrid.compute_mass_grid(
            dstfield, dstareafield, uninitval=fill_value
        )

        # Insert the two masses into the dictionary for comparison
        _compute_field_mass[k] = (srcmass, dstmass)

    def _regrid_get_regridded_data(
        self, method, fracfield, dstfield, dstfracfield
    ):
        """Get the regridded data of frac field as a numpy array from
        the ESMPy fields.

        :Parameters:

            method: `str`
                The regridding method.

            fracfield: `bool`
                Whether to return the frac field or not in the case of
                conservative regridding.

            dstfield: ESMPy field
                The destination field.

            dstfracfield: ESMPy field
                Information about the fraction of each of the destination
                field cells involved in the regridding. For conservative
                regridding this must be taken into account.

        """
        if method in conservative_regridding_methods:
            frac = dstfracfield.data.copy()
            if fracfield:
                regridded_data = frac
            else:
                frac[frac == 0.0] = 1.0
                regridded_data = numpy_ma_MaskedArray(
                    dstfield.data / frac,
                    mask=(dstfield.data == self.fill_value(default="netCDF")),
                )
        else:
            regridded_data = numpy_ma_MaskedArray(
                dstfield.data.copy(),
                mask=(dstfield.data == self.fill_value(default="netCDF")),
            )

        return regridded_data

    def _regrid_update_coordinate_references(
        self,
        dst,
        src_axis_keys,
        dst_axis_sizes,
        method,
        use_dst_mask,
        cartesian=False,
        axes=("X", "Y"),
        n_axes=2,
        src_cyclic=False,
        dst_cyclic=False,
    ):
        """Update the coordinate references of the new field after
        regridding.

        :Parameters:

            dst: `Field` or `dict`
                The object with the destination grid for regridding.

            src_axis_keys: sequence of `str`
                The keys of the source regridding axes.

            dst_axis_sizes: sequence, optional
                The sizes of the destination axes.

            method: `bool`
                The regridding method.

            use_dst_mask: `bool`
                Whether to use the destination mask in regridding.

            i: `bool`
                Whether to do the regridding in place.

            cartesian: `bool`, optional
                Whether to do Cartesian regridding or spherical

            axes: sequence, optional
                Specifiers for the regridding axes.

            n_axes: `int`, optional
                The number of regridding axes.

            src_cyclic: `bool`, optional
                Whether the source longitude is cyclic for spherical
                regridding.

            dst_cyclic: `bool`, optional
                Whether the destination longitude is cyclic for spherical
                regridding.

        """
        domain_ancillaries = self.domain_ancillaries(todict=True)

        # Initialise cached value for domain_axes
        domain_axes = None

        data_axes = self.constructs.data_axes()

        for key, ref in self.coordinate_references(todict=True).items():
            ref_axes = []
            for k in ref.coordinates():
                ref_axes.extend(data_axes[k])

            if set(ref_axes).intersection(src_axis_keys):
                self.del_construct(key)
                continue

            for (
                term,
                value,
            ) in ref.coordinate_conversion.domain_ancillaries().items():
                if value not in domain_ancillaries:
                    continue

                key = value

                # If this domain ancillary spans both X and Y axes
                # then regrid it, otherwise remove it
                x = self.domain_axis("X", key=True)
                y = self.domain_axis("Y", key=True)
                if (
                    self.domain_ancillary(
                        filter_by_axis=(x, y),
                        axis_mode="exact",
                        key=True,
                        default=None,
                    )
                    is not None
                ):
                    # Convert the domain ancillary into an independent
                    # field
                    value = self.convert(key)
                    try:
                        if cartesian:
                            value.regridc(
                                dst,
                                axes=axes,
                                method=method,
                                use_dst_mask=use_dst_mask,
                                inplace=True,
                            )
                        else:
                            value.regrids(
                                dst,
                                src_cyclic=src_cyclic,
                                dst_cyclic=dst_cyclic,
                                method=method,
                                use_dst_mask=use_dst_mask,
                                inplace=True,
                            )
                    except ValueError:
                        ref.coordinate_conversion.set_domain_ancillary(
                            term, None
                        )
                        self.del_construct(key)
                    else:
                        ref.coordinate_conversion.set_domain_ancillary(
                            term, key
                        )
                        d_axes = data_axes[key]

                        domain_axes = self.domain_axes(
                            cached=domain_axes, todict=True
                        )

                        for k_s, new_size in zip(
                            src_axis_keys, dst_axis_sizes
                        ):
                            domain_axes[k_s].set_size(new_size)

                        self.set_construct(
                            self._DomainAncillary(source=value),
                            key=key,
                            axes=d_axes,
                            copy=False,
                        )

    def _regrid_copy_coordinate_references(self, dst, dst_axis_keys):
        """Copy coordinate references from the destination field to the
        new, regridded field.

        :Parameters:

            dst: `Field`
                The destination field.

            dst_axis_keys: sequence of `str`
                The keys of the regridding axes in the destination field.

        :Returns:

            `None`

        """
        dst_data_axes = dst.constructs.data_axes()

        for ref in dst.coordinate_references(todict=True).values():
            axes = set()
            for key in ref.coordinates():
                axes.update(dst_data_axes[key])

            if axes and set(axes).issubset(dst_axis_keys):
                # This coordinate reference's coordinates span the X
                # and/or Y axes
                self.set_coordinate_reference(ref, parent=dst, strict=True)

    @classmethod
    def _regrid_use_bounds(cls, method):
        """Returns whether to use the bounds or not in regridding. This
        is only the case for conservative regridding.

        :Parameters:

            method: `str`
                The regridding method

        :Returns:

            `bool`

        """
        return method in conservative_regridding_methods

    def _regrid_update_coordinates(
        self,
        dst,
        dst_dict,
        dst_coords,
        src_axis_keys,
        dst_axis_keys,
        cartesian=False,
        dst_axis_sizes=None,
        dst_coords_2D=False,
        dst_coord_order=None,
    ):
        """Update the coordinates of the new field.

        :Parameters:

            dst: Field or `dict`
                The object containing the destination grid.

            dst_dict: `bool`
                Whether dst is a dictionary.

            dst_coords: sequence
                The destination coordinates.

            src_axis_keys: sequence
                The keys of the regridding axes in the source field.

            dst_axis_keys: sequence
                The keys of the regridding axes in the destination field.

            cartesian: `bool`, optional
                Whether regridding is Cartesian of spherical, False by
                default.

            dst_axis_sizes: sequence, optional
                The sizes of the destination axes.

            dst_coords_2D: `bool`, optional
                Whether the destination coordinates are 2D, currently only
                applies to spherical regridding.

            dst_coord_order: `list`, optional
                A list of lists specifying the ordering of the axes for
                each 2D destination coordinate.

        """
        # NOTE: May be common ground between cartesian and shperical that
        # could save some lines of code.

        # Remove the source coordinates of new field
        for key in self.coordinates(
            filter_by_axis=src_axis_keys, axis_mode="or", todict=True
        ):
            self.del_construct(key)

        domain_axes = self.domain_axes(todict=True)

        if cartesian:
            # Insert coordinates from dst into new field
            if dst_dict:
                for k_s, coord in zip(src_axis_keys, dst_coords):
                    domain_axes[k_s].set_size(coord.size)
                    self.set_construct(coord, axes=[k_s])
            else:
                axis_map = {
                    key_d: key_s
                    for key_s, key_d in zip(src_axis_keys, dst_axis_keys)
                }

                for key_d in dst_axis_keys:
                    dim = dst.dimension_coordinate(filter_by_axis=(key_d,))
                    key_s = axis_map[key_d]
                    domain_axes[key_s].set_size(dim.size)
                    self.set_construct(dim, axes=[key_s])

                dst_data_axes = dst.constructs.data_axes()

                for aux_key, aux in dst.auxiliary_coordinates(
                    filter_by_axis=dst_axis_keys,
                    axis_mode="subset",
                    todict=True,
                ).items():
                    aux_axes = [
                        axis_map[key_d] for key_d in dst_data_axes[aux_key]
                    ]
                    self.set_construct(aux, axes=aux_axes)
        else:
            # Give destination grid latitude and longitude standard names
            dst_coords[0].standard_name = "longitude"
            dst_coords[1].standard_name = "latitude"

            # Insert 'X' and 'Y' coordinates from dst into new field
            for axis_key, axis_size in zip(src_axis_keys, dst_axis_sizes):
                domain_axes[axis_key].set_size(axis_size)

            if dst_dict:
                if dst_coords_2D:
                    for coord, coord_order in zip(dst_coords, dst_coord_order):
                        axis_keys = [
                            src_axis_keys[index] for index in coord_order
                        ]
                        self.set_construct(coord, axes=axis_keys)
                else:
                    for coord, axis_key in zip(dst_coords, src_axis_keys):
                        self.set_construct(coord, axes=[axis_key])

            else:
                for src_axis_key, dst_axis_key in zip(
                    src_axis_keys, dst_axis_keys
                ):
                    dim_coord = dst.dimension_coordinate(
                        filter_by_axis=(dst_axis_key,), default=None
                    )
                    if dim_coord is not None:
                        self.set_construct(dim_coord, axes=[src_axis_key])

                    for aux in dst.auxiliary_coordinates(
                        filter_by_axis=(dst_axis_key,),
                        axis_mode="exact",
                        todict=True,
                    ).values():
                        self.set_construct(aux, axes=[src_axis_key])

                for aux_key, aux in dst.auxiliary_coordinates(
                    filter_by_axis=dst_axis_keys,
                    axis_mode="subset",
                    todict=True,
                ).items():
                    aux_axes = dst.get_data_axes(aux_key)
                    if aux_axes == tuple(dst_axis_keys):
                        self.set_construct(aux, axes=src_axis_keys)
                    else:
                        self.set_construct(aux, axes=src_axis_keys[::-1])

        # Copy names of dimensions from destination to source field
        if not dst_dict:
            dst_domain_axes = dst.domain_axes(todict=True)
            for src_axis_key, dst_axis_key in zip(
                src_axis_keys, dst_axis_keys
            ):
                ncdim = dst_domain_axes[dst_axis_key].nc_get_dimension(None)
                if ncdim is not None:
                    domain_axes[src_axis_key].nc_set_dimension(ncdim)

    # ----------------------------------------------------------------
    # End of worker functions for regridding
    #
    # TODO move to another file
    # ----------------------------------------------------------------

    @_deprecated_kwarg_check("i")
    @_inplace_enabled(default=False)
    def regrids(
        self,
        dst,
        method,
        src_cyclic=None,
        dst_cyclic=None,
        use_src_mask=True,
        use_dst_mask=False,
        fracfield=False,
        src_axes=None,
        dst_axes=None,
        axis_order=None,
        ignore_degenerate=True,
        inplace=False,
        i=False,
        _compute_field_mass=None,
    ):
        """Return the field regridded onto a new latitude-longitude
        grid.

        Regridding, also called remapping or interpolation, is the
        process of changing the grid underneath field data values
        while preserving the qualities of the original data.

        The regridding method must be specified. First-order
        conservative interpolation conserves the global area integral
        of the field, but may not give approximations to the values as
        good as linear interpolation. Second-order conservative
        interpolation also takes into account the gradient across the
        source cells, so in general gives a smoother, more accurate
        representation of the source field especially when going from
        a coarser to a finer grid. Linear interpolation is
        available. The latter method is particular useful for cases
        when the latitude and longitude coordinate cell boundaries are
        not known nor inferable. Higher order patch recovery is
        available as an alternative to linear interpolation. This
        typically results in better approximations to values and
        derivatives compared to the latter, but the weight matrix can
        be larger than the linear matrix, which can be an issue when
        regridding close to the memory limit on a machine. Nearest
        neighbour interpolation is also available. Nearest source to
        destination is particularly useful for regridding integer
        fields such as land use.


        **Metadata**

        The field construct's domain must have well defined X and Y
        axes with latitude and longitude coordinate values, which may
        be stored as dimension coordinate objects or two dimensional
        auxiliary coordinate objects. If the latitude and longitude
        coordinates are two dimensional then the X and Y axes must be
        defined by dimension coordinates if present or by the netCDF
        dimensions. In the latter case the X and Y axes must be
        specified using the *src_axes* or *dst_axes* keyword. The same
        is true for the destination grid, if it provided as part of
        another field.

        The cyclicity of the X axes of the source field and
        destination grid is taken into account. If an X axis is in
        fact cyclic but is not registered as such by its parent field
        (see `cf.Field.iscyclic`), then the cyclicity may be set with
        the *src_cyclic* or *dst_cyclic* parameters. In the case of
        two dimensional latitude and longitude dimension coordinates
        without bounds it will be necessary to specify *src_cyclic* or
        *dst_cyclic* manually if the field is global.

        The output field construct's coordinate objects which span the
        X and/or Y axes are replaced with those from the destination
        grid. Any fields contained in coordinate reference objects
        will also be regridded, if possible.


        **Mask**

        The data array mask of the field is automatically taken into
        account, such that the regridded data array will be masked in
        regions where the input data array is masked. By default the
        mask of the destination grid is not taken into account. If the
        destination field data has more than two dimensions then the
        mask, if used, is taken from the two dimensional section of
        the data where the indices of all axes other than X and Y are
        zero.


        **Implementation**

        The interpolation is carried out using the `ESMPy` package, a
        Python interface to the Earth System Modeling Framework (ESMF)
        `regridding utility
        <https://www.earthsystemcog.org/projects/esmf/regridding>`_.


        **Logging**

        Whether ESMF logging is enabled or not is determined by
        `cf.regrid_logging`. If it is logging takes place after every
        call. By default logging is disabled.


        **Latitude-Longitude Grid**

        The canonical grid with independent latitude and longitude
        coordinates.


        **Curvilinear Grids**

        Grids in projection coordinate systems can be regridded as
        long as two dimensional latitude and longitude coordinates are
        present.


        **Rotated Pole Grids**

        Rotated pole grids can be regridded as long as two dimensional
        latitude and longitude coordinates are present. It may be
        necessary to explicitly identify the grid latitude and grid
        longitude coordinates as being the X and Y axes and specify
        the *src_cyclic* or *dst_cyclic* keywords.


        **Tripolar Grids**

        Tripolar grids are logically rectangular and so may be able to
        be regridded. If no dimension coordinates are present it will
        be necessary to specify which netCDF dimensions are the X and
        Y axes using the *src_axes* or *dst_axes*
        keywords. Connections across the bipole fold are not currently
        supported, but are not be necessary in some cases, for example
        if the points on either side are together without a gap. It
        will also be necessary to specify *src_cyclic* or *dst_cyclic*
        if the grid is global.

        .. versionadded:: 1.0.4

        .. seealso:: `regridc`

        :Parameters:

            dst: `Field` or `dict`
                The field containing the new grid. If dst is a field
                list the first field in the list is
                used. Alternatively a dictionary can be passed
                containing the keywords 'longitude' and 'latitude'
                with either two 1D dimension coordinates or two 2D
                auxiliary coordinates. In the 2D case both coordinates
                must have their axes in the same order and this must
                be specified by the keyword 'axes' as either of the
                tuples ``('X', 'Y')`` or ``('Y', 'X')``.

            method: `str`
                Specify the regridding method. The *method* parameter must
                be one of the following:

                ======================  ==================================
                Method                  Description
                ======================  ==================================
                ``'linear'``            Bilinear interpolation.

                ``'bilinear'``          Deprecated alias for ``'linear'``.

                ``'conservative_1st'``  First order conservative
                                        interpolation.

                                        Preserve the area integral of
                                        the data across the
                                        interpolation from source to
                                        destination. It uses the
                                        proportion of the area of the
                                        overlapping source and
                                        destination cells to determine
                                        appropriate weights.

                                        In particular, the weight of a
                                        source cell is the ratio of
                                        the area of intersection of
                                        the source and destination
                                        cells to the area of the whole
                                        destination cell.

                                        It does not account for the
                                        field gradient across the
                                        source cell, unlike the
                                        second-order conservative
                                        method (see below).

                ``'conservative_2nd'``  Second-order conservative
                                        interpolation.

                                        As with first order (see
                                        above), preserves the area
                                        integral of the field between
                                        source and destination using a
                                        weighted sum, with weights
                                        based on the proportionate
                                        area of intersection.

                                        Unlike first-order, the
                                        second-order method
                                        incorporates further terms to
                                        take into consideration the
                                        gradient of the field across
                                        the source cell, thereby
                                        typically producing a smoother
                                        result of higher accuracy.

                ``'conservative'``      Alias for ``'conservative_1st'``

                ``'patch'``             Higher-order patch recovery
                                        interpolation.

                                        A second degree polynomial
                                        regridding method, which uses
                                        a least squares algorithm to
                                        calculate the polynomial.

                                        This method gives better
                                        derivatives in the resulting
                                        destination data than the
                                        linear method.

                ``'nearest_stod'``      Nearest neighbour interpolation
                                        for which each destination point
                                        is mapped to the closest source
                                        point.

                                        Useful for extrapolation of
                                        categorical data.

                ``'nearest_dtos'``      Nearest neighbour interpolation
                                        for which each source point is
                                        mapped to the destination point.

                                        Useful for extrapolation of
                                        categorical data.

                                        A given destination point may
                                        receive input from multiple
                                        source points, but no source
                                        point will map to more than
                                        one destination point.
                ======================  ==================================

            src_cyclic: `bool`, optional
                Specifies whether the longitude for the source grid is
                periodic or not. If `None` then, if possible, this is
                determined automatically otherwise it defaults to
                False.

            dst_cyclic: `bool`, optional
                Specifies whether the longitude for the destination
                grid is periodic of not. If `None` then, if possible,
                this is determined automatically otherwise it defaults
                to False.

            use_src_mask: `bool`, optional
                For all methods other than 'nearest_stod', this must
                be True as it does not make sense to set it to
                False. For the 'nearest_stod' method if it is True
                then points in the result that are nearest to a masked
                source point are masked. Otherwise, if it is False,
                then these points are interpolated to the nearest
                unmasked source points.

            use_dst_mask: `bool`, optional
                By default the mask of the data on the destination
                grid is not taken into account when performing
                regridding. If this option is set to true then it
                is. If the destination field has more than two
                dimensions then the first 2D slice in index space is
                used for the mask e.g. for an field varying with (X,
                Y, Z, T) the mask is taken from the slice (X, Y, 0,
                0).

            fracfield: `bool`, optional
                If the method of regridding is conservative the
                fraction of each destination grid cell involved in the
                regridding is returned instead of the regridded data
                if this is True. Otherwise this is ignored.

            src_axes: `dict`, optional
                A dictionary specifying the axes of the 2D latitude
                and longitude coordinates of the source field when no
                1D dimension coordinates are present. It must have
                keys ``'X'`` and ``'Y'``. TODO

                *Parameter example:*
                  ``src_axes={'X': 'ncdim%x', 'Y': 'ncdim%y'}``

                *Parameter example:*
                  ``src_axes={'X': 1, 'Y': 0}``

            dst_axes: `dict`, optional
                A dictionary specifying the axes of the 2D latitude
                and longitude coordinates of the destination field
                when no dimension coordinates are present. It must
                have keys ``'X'`` and ``'Y'``.

                *Parameter example:*
                  ``dst_axes={'X': 'ncdim%x', 'Y': 'ncdim%y'}``

            axis_order: sequence, optional
                A sequence of items specifying dimension coordinates
                as retrieved by the `dim` method. These determine the
                order in which to iterate over the other axes of the
                field when regridding X-Y slices. The slowest moving
                axis will be the first one specified. Currently the
                regridding weights are recalculated every time the
                mask of an X-Y slice changes with respect to the
                previous one, so this option allows the user to
                minimise how frequently the mask changes.

            ignore_degenerate: `bool`, optional
                True by default. Instructs ESMPy to ignore degenerate
                cells when checking the grids for errors. Regridding
                will proceed and degenerate cells will be skipped, not
                producing a result, when set to True. Otherwise an
                error will be produced if degenerate cells are
                found. This will be present in the ESMPy log files if
                `cf.regrid_logging` is set to True. As of ESMF 7.0.0
                this only applies to conservative regridding.  Other
                methods always skip degenerate cells.


            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            _compute_field_mass: `dict`, optional
                If this is a dictionary then the field masses of the
                source and destination fields are computed and
                returned within the dictionary. The keys of the
                dictionary indicates the lat-long slice of the field
                and the corresponding value is a tuple containing the
                source field construct's mass and the destination
                field construct's mass. The calculation is only done
                if conservative regridding is being performed. This is
                for debugging purposes.

        :Returns:

            `Field`
                The regridded field construct.

        **Examples:**

        Regrid field construct ``f`` conservatively onto a grid
        contained in field construct ``g``:

        >>> h = f.regrids(g, 'conservative')

        Regrid f to the grid of g using linear regridding and forcing
        the source field f to be treated as cyclic.

        >>> h = f.regrids(g, src_cyclic=True, method='linear')

        Regrid f to the grid of g using the mask of g.

        >>> h = f.regrids(g, 'conservative_1st', use_dst_mask=True)

        Regrid f to 2D auxiliary coordinates lat and lon, which have
        their dimensions ordered "Y" first then "X".

        >>> lat
        <CF AuxiliaryCoordinate: latitude(110, 106) degrees_north>
        >>> lon
        <CF AuxiliaryCoordinate: longitude(110, 106) degrees_east>
        >>> h = f.regrids(
        ...         {'longitude': lon, 'latitude': lat, 'axes': ('Y', 'X')},
        ...         'conservative'
        ...     )

        Regrid field, f, on tripolar grid to latitude-longitude grid
        of field, g.

        >>> h = f.regrids(g, 'linear', src_axes={'X': 'ncdim%x', 'Y': 'ncdim%y'},
        ...               src_cyclic=True)

        Regrid f to the grid of g iterating over the 'Z' axis last and
        the 'T' axis next to last to minimise the number of times the
        mask is changed.

        >>> h = f.regrids(g, 'nearest_dtos', axis_order='ZT')

        """
        # Initialise ESMPy for regridding if found
        Regrid.initialize()

        f = _inplace_enabled_define_and_cleanup(self)

        # If dst is a dictionary set flag
        dst_dict = not isinstance(dst, f.__class__)

        # Retrieve the source field's latitude and longitude coordinates
        (
            src_axis_keys,
            src_axis_sizes,
            src_coord_keys,
            src_coords,
            src_coords_2D,
        ) = f._regrid_get_latlong("source", axes=src_axes)

        # Retrieve the destination field's latitude and longitude coordinates
        if dst_dict:
            # dst is a dictionary
            try:
                dst_coords = (dst["longitude"], dst["latitude"])
            except KeyError:
                raise ValueError(
                    "Keys 'longitude' and 'latitude' must be"
                    " specified for destination."
                )

            if dst_coords[0].ndim == 1:
                dst_coords_2D = False
                dst_axis_sizes = [coord.size for coord in dst_coords]
            elif dst_coords[0].ndim == 2:
                try:
                    dst_axes = dst["axes"]
                except KeyError:
                    raise ValueError(
                        "Key 'axes' must be specified for 2D"
                        " latitude/longitude coordinates."
                    )
                dst_coords_2D = True
                if dst_axes == ("X", "Y"):
                    dst_axis_sizes = dst_coords[0].shape
                elif dst_axes == ("Y", "X"):
                    dst_axis_sizes = dst_coords[0].shape[::-1]
                else:
                    raise ValueError(
                        "Keyword 'axes' must either be "
                        "('X', 'Y') or ('Y', 'X')."
                    )
                if dst_coords[0].shape != dst_coords[1].shape:
                    raise ValueError(
                        "Longitude and latitude coordinates for "
                        "destination must have the same shape."
                    )
            else:
                raise ValueError(
                    "Longitude and latitude coordinates for "
                    "destination must have 1 or 2 dimensions."
                )

            dst_axis_keys = None
        else:
            # dst is a Field
            (
                dst_axis_keys,
                dst_axis_sizes,
                dst_coord_keys,
                dst_coords,
                dst_coords_2D,
            ) = dst._regrid_get_latlong("destination", axes=dst_axes)

        # Automatically detect the cyclicity of the source longitude if
        # src_cyclic is None
        if src_cyclic is None:
            src_cyclic = f.iscyclic(src_axis_keys[0])

        # Automatically detect the cyclicity of the destination longitude if
        # dst is not a dictionary and dst_cyclic is None
        if not dst_dict and dst_cyclic is None:
            dst_cyclic = dst.iscyclic(dst_axis_keys[0])
        elif dst_dict and dst_cyclic is None:
            dst = dst.copy()
            #            dst['longitude'] = dst['longitude'].copy()
            #            dst['longitude'].autoperiod()
            dst["longitude"] = dst["longitude"].autoperiod()
            dst_cyclic = dst["longitude"].isperiodic

        # Get the axis indices and their order for the source field
        src_axis_indices, src_order = f._regrid_get_axis_indices(src_axis_keys)

        # Get the axis indices and their order for the destination field.
        if not dst_dict:
            dst = dst.copy()
            dst_axis_indices, dst_order = dst._regrid_get_axis_indices(
                dst_axis_keys
            )

        # Get the order of the X and Y axes for each 2D auxiliary coordinate.
        src_coord_order = None
        dst_coord_order = None
        if src_coords_2D:
            src_coord_order = self._regrid_get_coord_order(
                src_axis_keys, src_coord_keys
            )

        if dst_coords_2D:
            if dst_dict:
                if dst_axes == ("X", "Y"):
                    dst_coord_order = [[0, 1], [0, 1]]
                elif dst_axes == ("Y", "X"):
                    dst_coord_order = [[1, 0], [1, 0]]
                else:
                    raise ValueError(
                        "Keyword 'axes' must either be ('X', 'Y') or "
                        "('Y', 'X')."
                    )
            else:
                dst_coord_order = dst._regrid_get_coord_order(
                    dst_axis_keys, dst_coord_keys
                )

        # Get the shape of each section after it has been regridded.
        shape = self._regrid_get_section_shape(
            dst_axis_sizes, src_axis_indices
        )

        # Check the method
        self._regrid_check_method(method)

        # Check that use_src_mask is True for all methods other than
        # nearest_stod
        self._regrid_check_use_src_mask(use_src_mask, method)

        # Check the bounds of the coordinates
        self._regrid_check_bounds(src_coords, dst_coords, method)

        # Slice the source data into 2D latitude/longitude sections,
        # also getting a list of dictionary keys in the order
        # requested. If axis_order has not been set, then the order is
        # random, and so in this case the order in which sections are
        # regridded is random.
        section_keys, sections = self._regrid_get_reordered_sections(
            axis_order, src_axis_keys, src_axis_indices
        )

        # Bounds must be used if the regridding method is conservative.
        use_bounds = self._regrid_use_bounds(method)

        # Retrieve the destination field's mask if appropriate
        dst_mask = None
        if not dst_dict and use_dst_mask and dst.data.ismasked:
            dst_mask = dst._regrid_get_destination_mask(
                dst_order, axes=dst_axis_keys
            )

        # Retrieve the destination ESMPy grid and fields
        dstgrid = Regrid.create_grid(
            dst_coords,
            use_bounds,
            mask=dst_mask,
            cyclic=dst_cyclic,
            coords_2D=dst_coords_2D,
            coord_order=dst_coord_order,
        )
        # dstfield will be reused to receive the regridded source data
        # for each section, one after the other
        dstfield = Regrid.create_field(dstgrid, "dstfield")
        dstfracfield = Regrid.create_field(dstgrid, "dstfracfield")

        # Regrid each section
        old_mask = None
        unmasked_grid_created = False
        for k in section_keys:
            d = sections[k]  # d is a Data object
            # Retrieve the source field's grid, create the ESMPy grid and a
            # handle to regridding.dst_dict
            src_data = d.squeeze().transpose(src_order).array
            if not (
                method == "nearest_stod" and use_src_mask
            ) and numpy_ma_is_masked(src_data):
                mask = src_data.mask
                if not numpy_array_equal(mask, old_mask):
                    # Release old memory
                    if old_mask is not None:
                        regridSrc2Dst.destroy()  # noqa: F821
                        srcfracfield.destroy()  # noqa: F821
                        srcfield.destroy()  # noqa: F821
                        srcgrid.destroy()  # noqa: F821

                    # (Re)create the source ESMPy grid and fields
                    srcgrid = Regrid.create_grid(
                        src_coords,
                        use_bounds,
                        mask=mask,
                        cyclic=src_cyclic,
                        coords_2D=src_coords_2D,
                        coord_order=src_coord_order,
                    )
                    srcfield = Regrid.create_field(srcgrid, "srcfield")
                    srcfracfield = Regrid.create_field(srcgrid, "srcfracfield")

                    # (Re)initialise the regridder
                    regridSrc2Dst = Regrid(
                        srcfield,
                        dstfield,
                        srcfracfield,
                        dstfracfield,
                        method=method,
                        ignore_degenerate=ignore_degenerate,
                    )
                    old_mask = mask
            else:
                # The source data for this section is either a) not
                # masked or b) has the same mask as the previous
                # section.
                if not unmasked_grid_created or old_mask is not None:
                    # Create the source ESMPy grid and fields
                    srcgrid = Regrid.create_grid(
                        src_coords,
                        use_bounds,
                        cyclic=src_cyclic,
                        coords_2D=src_coords_2D,
                        coord_order=src_coord_order,
                    )
                    srcfield = Regrid.create_field(srcgrid, "srcfield")
                    srcfracfield = Regrid.create_field(srcgrid, "srcfracfield")

                    # Initialise the regridder. This also creates the
                    # weights needed for the regridding.
                    regridSrc2Dst = Regrid(
                        srcfield,
                        dstfield,
                        srcfracfield,
                        dstfracfield,
                        method=method,
                        ignore_degenerate=ignore_degenerate,
                    )
                    unmasked_grid_created = True
                    old_mask = None

            # Fill the source and destination fields (the destination
            # field gets filled with a fill value, the source field
            # with the section's data)
            self._regrid_fill_fields(src_data, srcfield, dstfield)

            # Run regridding (dstfield is an ESMF field)
            dstfield = regridSrc2Dst.run_regridding(srcfield, dstfield)

            # Compute field mass if requested for conservative regridding
            if (
                _compute_field_mass is not None
                and method in conservative_regridding_methods
            ):
                # Update the _compute_field_mass dictionary in-place,
                # thereby making the field mass available after
                # returning
                self._regrid_compute_field_mass(
                    _compute_field_mass,
                    k,
                    srcgrid,
                    srcfield,
                    srcfracfield,
                    dstgrid,
                    dstfield,
                )

            # Get the regridded data or frac field as a numpy array
            # (regridded_data is a numpy array)
            regridded_data = self._regrid_get_regridded_data(
                method, fracfield, dstfield, dstfracfield
            )

            # Insert regridded data, with axes in order of the
            # original section. This puts the regridded data back into
            # the sections dictionary, with the same key, as a new
            # Data object. Note that the reshape is necessary to
            # replace any size 1 dimensions that we squeezed out
            # earlier.
            sections[k] = Data(
                regridded_data.transpose(src_order).reshape(shape),
                units=self.Units,
            )

        # Construct new data from regridded sections
        new_data = Data.reconstruct_sectioned_data(sections)

        # Construct new field.
        # Note: cannot call `_inplace_enabled_define_and_cleanup(self)` to
        # apply this if-else logic (it deletes the decorator attribute so
        # can only be used once)
        if inplace:
            f = self
        else:
            f = self.copy()

        # Update ancillary variables of new field
        # f._conform_ancillary_variables(src_axis_keys, keep_size_1=False)

        #        for k_s, new_size in zip(src_axis_keys, dst_axis_sizes):
        #            f.domain_axes[k_s].set_size(new_size)

        # Update coordinate references of new field
        f._regrid_update_coordinate_references(
            dst,
            src_axis_keys,
            dst_axis_sizes,
            method,
            use_dst_mask,
            src_cyclic=src_cyclic,
            dst_cyclic=dst_cyclic,
        )

        # Update coordinates of new field
        f._regrid_update_coordinates(
            dst,
            dst_dict,
            dst_coords,
            src_axis_keys,
            dst_axis_keys,
            dst_axis_sizes=dst_axis_sizes,
            dst_coords_2D=dst_coords_2D,
            dst_coord_order=dst_coord_order,
        )

        # Copy across destination fields coordinate references if necessary
        if not dst_dict:
            f._regrid_copy_coordinate_references(dst, dst_axis_keys)

        # Insert regridded data into new field
        f.set_data(new_data, axes=self.get_data_axes(), copy=False)

        # Set the cyclicity of the destination longitude
        key, x = f.dimension_coordinate("X", default=(None, None), item=True)
        if x is not None and x.Units.equivalent(Units("degrees")):
            f.cyclic(
                key,
                iscyclic=dst_cyclic,
                config={"coord": x, "period": Data(360.0, "degrees")},
            )

        # Release old memory from ESMF (this ought to happen garbage
        # collection, but it doesn't seem to work there!)
        regridSrc2Dst.destroy()
        dstfracfield.destroy()
        srcfracfield.destroy()
        dstfield.destroy()
        srcfield.destroy()
        dstgrid.destroy()
        srcgrid.destroy()

        #        if f.data.fits_in_one_chunk_in_memory(f.data.dtype.itemsize):
        #            f.varray

        #        f.autocyclic()

        return f

    @_deprecated_kwarg_check("i")
    @_inplace_enabled(default=False)
    def regridc(
        self,
        dst,
        axes,
        method,
        use_src_mask=True,
        use_dst_mask=False,
        fracfield=False,
        axis_order=None,
        ignore_degenerate=True,
        inplace=False,
        i=False,
        _compute_field_mass=None,
    ):
        """Return the field with the specified Cartesian axes regridded
        onto a new grid.

        Between 1 and 3 dimensions may be regridded.

        Regridding, also called remapping or interpolation, is the
        process of changing the grid underneath field data values
        while preserving the qualities of the original data.

        The regridding method must be specified. First-order
        conservative interpolation conserves the global spatial
        integral of the field, but may not give approximations to the
        values as good as (multi)linear interpolation. Second-order
        conservative interpolation also takes into account the
        gradient across the source cells, so in general gives a
        smoother, more accurate representation of the source field
        especially when going from a coarser to a finer
        grid. (Multi)linear interpolation is available. The latter
        method is particular useful for cases when the latitude and
        longitude coordinate cell boundaries are not known nor
        inferable. Higher order patch recovery is available as an
        alternative to (multi)linear interpolation.  This typically
        results in better approximations to values and derivatives
        compared to the latter, but the weight matrix can be larger
        than the linear matrix, which can be an issue when regridding
        close to the memory limit on a machine. It is only available
        in 2D. Nearest neighbour interpolation is also
        available. Nearest source to destination is particularly
        useful for regridding integer fields such as land use.

        **Metadata**

        The field construct's domain must have axes matching those
        specified in *src_axes*. The same is true for the destination
        grid, if it provided as part of another field. Optionally the
        axes to use from the destination grid may be specified
        separately in *dst_axes*.

        The output field construct's coordinate objects which span the
        specified axes are replaced with those from the destination
        grid. Any fields contained in coordinate reference objects
        will also be regridded, if possible.


        **Mask**

        The data array mask of the field is automatically taken into
        account, such that the regridded data array will be masked in
        regions where the input data array is masked. By default the
        mask of the destination grid is not taken into account. If the
        destination field data has more dimensions than the number of
        axes specified then, if used, its mask is taken from the 1-3
        dimensional section of the data where the indices of all axes
        other than X and Y are zero.


        **Implementation**

        The interpolation is carried out using the `ESMPy` package, a
        Python interface to the Earth System Modeling Framework (ESMF)
        `regridding utility
        <https://www.earthsystemcog.org/projects/esmf/regridding>`_.


        **Logging**

        Whether ESMF logging is enabled or not is determined by
        `cf.regrid_logging`. If it is logging takes place after every
        call. By default logging is disabled.

        .. seealso:: `regrids`

        :Parameters:

            dst: `Field` or `dict`
                The field containing the new grid or a dictionary with
                the axes specifiers as keys referencing dimension
                coordinates.  If dst is a field list the first field
                in the list is used.

            axes:
                Select dimension coordinates from the source and
                destination fields for regridding. See `cf.Field.axes`
                TODO for options for selecting specific axes. However,
                the number of axes returned by `cf.Field.axes` TODO
                must be the same as the number of specifiers passed
                in.

            method: `str`
                Specify the regridding method. The *method* parameter
                must be one of the following:

                ======================  ==================================
                Method                  Description
                ======================  ==================================
                ``'linear'``            Linear interpolation in the number
                                        of dimensions being regridded.

                                        For two dimensional regridding
                                        this is bilinear
                                        interpolation, and for three
                                        dimensional regridding this is
                                        trilinear
                                        interpolation.Bilinear
                                        interpolation.

                ``'bilinear'``          Deprecated alias for ``'linear'``.

                ``'conservative_1st'``  First order conservative
                                        interpolation.

                                        Preserve the area integral of
                                        the data across the
                                        interpolation from source to
                                        destination. It uses the
                                        proportion of the area of the
                                        overlapping source and
                                        destination cells to determine
                                        appropriate weights.

                                        In particular, the weight of a
                                        source cell is the ratio of
                                        the area of intersection of
                                        the source and destination
                                        cells to the area of the whole
                                        destination cell.

                                        It does not account for the
                                        field gradient across the
                                        source cell, unlike the
                                        second-order conservative
                                        method (see below).

                ``'conservative_2nd'``  Second-order conservative
                                        interpolation.

                                        As with first order (see
                                        above), preserves the area
                                        integral of the field between
                                        source and destination using a
                                        weighted sum, with weights
                                        based on the proportionate
                                        area of intersection.

                                        Unlike first-order, the
                                        second-order method
                                        incorporates further terms to
                                        take into consideration the
                                        gradient of the field across
                                        the source cell, thereby
                                        typically producing a smoother
                                        result of higher accuracy.

                ``'conservative'``      Alias for ``'conservative_1st'``

                ``'patch'``             Higher-order patch recovery
                                        interpolation.

                                        A second degree polynomial
                                        regridding method, which uses
                                        a least squares algorithm to
                                        calculate the polynomial.

                                        This method gives better
                                        derivatives in the resulting
                                        destination data than the
                                        linear method.

                ``'nearest_stod'``      Nearest neighbour interpolation
                                        for which each destination point
                                        is mapped to the closest source
                                        point.

                                        Useful for extrapolation of
                                        categorical data.

                ``'nearest_dtos'``      Nearest neighbour interpolation
                                        for which each source point is
                                        mapped to the destination point.

                                        Useful for extrapolation of
                                        categorical data.

                                        A given destination point may
                                        receive input from multiple
                                        source points, but no source
                                        point will map to more than
                                        one destination point.
                ======================  ==================================

            use_src_mask: `bool`, optional
                For all methods other than 'nearest_stod', this must
                be True as it does not make sense to set it to
                False. For the

                'nearest_stod' method if it is True then points in the
                result that are nearest to a masked source point are
                masked. Otherwise, if it is False, then these points
                are interpolated to the nearest unmasked source
                points.

            use_dst_mask: `bool`, optional
                By default the mask of the data on the destination
                grid is not taken into account when performing
                regridding. If this option is set to True then it is.

            fracfield: `bool`, optional
                If the method of regridding is conservative the
                fraction of each destination grid cell involved in the
                regridding is returned instead of the regridded data
                if this is True. Otherwise this is ignored.

            axis_order: sequence, optional
                A sequence of items specifying dimension coordinates
                as retrieved by the `dim` method. These determine the
                order in which to iterate over the other axes of the
                field when regridding slices. The slowest moving axis
                will be the first one specified. Currently the
                regridding weights are recalculated every time the
                mask of a slice changes with respect to the previous
                one, so this option allows the user to minimise how
                frequently the mask changes.

            ignore_degenerate: `bool`, optional
                True by default. Instructs ESMPy to ignore degenerate
                cells when checking the grids for errors. Regridding
                will proceed and degenerate cells will be skipped, not
                producing a result, when set to True. Otherwise an
                error will be produced if degenerate cells are
                found. This will be present in the ESMPy log files if
                cf.regrid_logging is set to True. As of ESMF 7.0.0
                this only applies to conservative regridding.  Other
                methods always skip degenerate cells.


            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            _compute_field_mass: `dict`, optional
                If this is a dictionary then the field masses of the
                source and destination fields are computed and
                returned within the dictionary. The keys of the
                dictionary indicates the lat/long slice of the field
                and the corresponding value is a tuple containing the
                source field construct's mass and the destination
                field construct's mass. The calculation is only done
                if conservative regridding is being performed. This is
                for debugging purposes.

        :Returns:

            `Field` or `None`
                The regridded field construct, or `None` if the operation
                was in-place.

        **Examples:**

        Regrid the time axes of field ``f`` conservatively onto a grid
        contained in field ``g``:

        >>> h = f.regridc(g, axes='T', method='conservative')

        Regrid the T axis of field ``f`` conservatively onto the grid
        specified in the dimension coordinate ``t``:

        >>> h = f.regridc({'T': t}, axes=('T'), method='conservative_1st')

        Regrid the T axis of field ``f`` using linear interpolation onto
        a grid contained in field ``g``:

        >>> h = f.regridc(g, axes=('T'), method='linear')

        Regrid the X and Y axes of field ``f`` conservatively onto a grid
        contained in field ``g``:

        >>> h = f.regridc(g, axes=('X','Y'), method='conservative_1st')

        Regrid the X and T axes of field ``f`` conservatively onto a grid
        contained in field ``g`` using the destination mask:

        >>> h = f.regridc(g, axes=('X','Y'), use_dst_mask=True, method='linear')

        """
        # Initialise ESMPy for regridding if found
        Regrid.initialize()

        f = _inplace_enabled_define_and_cleanup(self)

        # If dst is a dictionary set flag
        dst_dict = not isinstance(dst, f.__class__)
        #        if isinstance(dst, self.__class__):
        #            dst_dict = False
        #            # If dst is a field list use the first field
        #            if isinstance(dst, FieldList):
        #                dst = dst[0]
        #        else:
        #            dst_dict = True

        # Get the number of axes
        if isinstance(axes, str):
            axes = (axes,)

        n_axes = len(axes)
        if n_axes < 1 or n_axes > 3:
            raise ValueError(
                "Between 1 and 3 axes must be individually specified."
            )

        # Retrieve the source axis keys and dimension coordinates
        src_axis_keys, src_coords = f._regrid_get_cartesian_coords(
            "source", axes
        )

        # Retrieve the destination axis keys and dimension coordinates
        if dst_dict:
            dst_coords = []
            for axis in axes:
                try:
                    dst_coords.append(dst[axis])
                except KeyError:
                    raise ValueError(f"Axis {axis!r} not specified in dst.")

            dst_axis_keys = None
        else:
            dst_axis_keys, dst_coords = dst._regrid_get_cartesian_coords(
                "destination", axes
            )

        # Check that the units of the source and the destination
        # coords are equivalent and if so set the units of the source
        # coords to those of the destination coords
        for src_coord, dst_coord in zip(src_coords, dst_coords):
            if src_coord.Units.equivalent(dst_coord.Units):
                src_coord.units = dst_coord.units
            else:
                raise ValueError(
                    "Units of source and destination domains are not "
                    f"equivalent: {src_coord.Units!r}, {dst_coord.Units!r}"
                )

        # Get the axis indices and their order for the source field
        src_axis_indices, src_order = f._regrid_get_axis_indices(src_axis_keys)

        # Get the axis indices and their order for the destination field.
        if not dst_dict:
            dst_axis_indices, dst_order = dst._regrid_get_axis_indices(
                dst_axis_keys
            )

        # Pad out a single dimension with an extra one (see comments
        # in _regrid_check_bounds). Variables ending in _ext pertain
        # the extra dimension.
        axis_keys_ext = []
        coords_ext = []
        src_axis_indices_ext = src_axis_indices
        src_order_ext = src_order
        # Proceed if there is only one regridding dimension, but more than
        # one dimension to the field that is not of size one.
        if n_axes == 1 and f.squeeze().ndim > 1:
            # Find the length and index of the longest axis not including
            # the axis along which regridding will be performed.
            src_shape = numpy_array(f.shape)
            tmp = src_shape.copy()
            tmp[src_axis_indices] = -1
            max_length = -1
            max_ind = -1
            for ind, length in enumerate(tmp):
                if length > max_length:
                    max_length = length
                    max_ind = ind
            # If adding this extra dimension to the regridding axis will not
            # create sections that exceed 1 chunk of memory proceed to get
            # the coordinate and associated data for the extra dimension.
            if src_shape[src_axis_indices].prod() * max_length * 8 < (
                float(chunksize())
            ):
                axis_keys_ext, coords_ext = f._regrid_get_cartesian_coords(
                    "source", [max_ind]
                )
                (
                    src_axis_indices_ext,
                    src_order_ext,
                ) = f._regrid_get_axis_indices(axis_keys_ext + src_axis_keys)

        # Calculate shape of each regridded section
        shape = f._regrid_get_section_shape(
            [coord.size for coord in coords_ext + dst_coords],
            src_axis_indices_ext,
        )

        # Check the method
        f._regrid_check_method(method)

        # Check that use_src_mask is True for all methods other than
        # nearest_stod
        f._regrid_check_use_src_mask(use_src_mask, method)

        # Check that the regridding axes span two dimensions if using
        # higher order patch recovery
        if method == "patch" and n_axes != 2:
            raise ValueError(
                "Higher order patch recovery is only available in 2D."
            )

        # Check the bounds of the coordinates
        f._regrid_check_bounds(
            src_coords, dst_coords, method, ext_coords=coords_ext
        )

        # Deal with case of 1D nonconservative regridding
        nonconservative1D = False
        if (
            method not in conservative_regridding_methods
            and n_axes == 1
            and coords_ext == []
        ):
            # Method is not conservative, regridding is to be done along
            # one dimension and that dimension has not been padded out with
            # an extra one.
            nonconservative1D = True
            coords_ext = [
                DimensionCoordinate(
                    data=Data(
                        [
                            numpy_finfo("float32").epsneg,
                            numpy_finfo("float32").eps,
                        ]
                    )
                )
            ]

        # Section the data into slices of up to three dimensions getting a
        # list of reordered keys if required. Reordering on an extended axis
        # will not have any effect as all the items in the keys will be None.
        # Therefore it is only checked if the axes specified in axis_order
        # are in the regridding axes as this is informative to the user.
        section_keys, sections = f._regrid_get_reordered_sections(
            axis_order, src_axis_keys, src_axis_indices_ext
        )

        # Use bounds if the regridding method is conservative.
        use_bounds = f._regrid_use_bounds(method)

        # Retrieve the destination field's mask if appropriate
        dst_mask = None
        if not dst_dict and use_dst_mask and dst.data.ismasked:
            dst_mask = dst._regrid_get_destination_mask(
                dst_order,
                axes=dst_axis_keys,
                cartesian=True,
                coords_ext=coords_ext,
            )

        # Create the destination ESMPy grid and fields
        dstgrid = Regrid.create_grid(
            coords_ext + dst_coords, use_bounds, mask=dst_mask, cartesian=True
        )
        dstfield = Regrid.create_field(dstgrid, "dstfield")
        dstfracfield = Regrid.create_field(dstgrid, "dstfracfield")

        # Regrid each section
        old_mask = None
        unmasked_grid_created = False
        for k in section_keys:
            d = sections[k]
            subsections = d.data.section(
                src_axis_indices_ext, chunks=True, min_step=2
            )
            for k2 in subsections.keys():
                d2 = subsections[k2]
                # Retrieve the source field's grid, create the ESMPy grid
                # and a handle to regridding.
                src_data = d2.squeeze().transpose(src_order_ext).array
                if nonconservative1D:
                    src_data = numpy_tile(src_data, (2, 1))

                if not (
                    method == "nearest_stod" and use_src_mask
                ) and numpy_ma_is_masked(src_data):
                    mask = src_data.mask
                    if not numpy_array_equal(mask, old_mask):
                        # Release old memory
                        if old_mask is not None:
                            regridSrc2Dst.destroy()  # noqa: F821
                            srcfracfield.destroy()  # noqa: F821
                            srcfield.destroy()  # noqa: F821
                            srcgrid.destroy()  # noqa: F821

                        # (Re)create the source ESMPy grid and fields
                        srcgrid = Regrid.create_grid(
                            coords_ext + src_coords,
                            use_bounds,
                            mask=mask,
                            cartesian=True,
                        )
                        srcfield = Regrid.create_field(srcgrid, "srcfield")
                        srcfracfield = Regrid.create_field(
                            srcgrid, "srcfracfield"
                        )
                        # (Re)initialise the regridder
                        regridSrc2Dst = Regrid(
                            srcfield,
                            dstfield,
                            srcfracfield,
                            dstfracfield,
                            method=method,
                            ignore_degenerate=ignore_degenerate,
                        )
                        old_mask = mask
                else:
                    if not unmasked_grid_created or old_mask is not None:
                        # Create the source ESMPy grid and fields
                        srcgrid = Regrid.create_grid(
                            coords_ext + src_coords, use_bounds, cartesian=True
                        )
                        srcfield = Regrid.create_field(srcgrid, "srcfield")
                        srcfracfield = Regrid.create_field(
                            srcgrid, "srcfracfield"
                        )
                        # Initialise the regridder
                        regridSrc2Dst = Regrid(
                            srcfield,
                            dstfield,
                            srcfracfield,
                            dstfracfield,
                            method=method,
                            ignore_degenerate=ignore_degenerate,
                        )
                        unmasked_grid_created = True
                        old_mask = None

                # Fill the source and destination fields
                f._regrid_fill_fields(src_data, srcfield, dstfield)

                # Run regridding
                dstfield = regridSrc2Dst.run_regridding(srcfield, dstfield)

                # Compute field mass if requested for conservative regridding
                if (
                    _compute_field_mass is not None
                    and method in conservative_regridding_methods
                ):
                    f._regrid_compute_field_mass(
                        _compute_field_mass,
                        k,
                        srcgrid,
                        srcfield,
                        srcfracfield,
                        dstgrid,
                        dstfield,
                    )

                # Get the regridded data or frac field as a numpy array
                regridded_data = f._regrid_get_regridded_data(
                    method, fracfield, dstfield, dstfracfield
                )

                if nonconservative1D:
                    # For nonconservative regridding along one dimension
                    # where that dimension has not been padded out take
                    # only one of the two rows of data as they should be
                    # nearly identical.
                    regridded_data = regridded_data[0]

                # Insert regridded data, with axes in correct order
                subsections[k2] = Data(
                    regridded_data.squeeze()
                    .transpose(src_order_ext)
                    .reshape(shape),
                    units=f.Units,
                )

            sections[k] = Data.reconstruct_sectioned_data(subsections)

        # Construct new data from regridded sections
        new_data = Data.reconstruct_sectioned_data(sections)

        # # Update ancillary variables of new field
        # f._conform_ancillary_variables(src_axis_keys, keep_size_1=False)

        dst_axis_sizes = [c.size for c in dst_coords]

        # Update coordinate references of new field
        f._regrid_update_coordinate_references(
            dst,
            src_axis_keys,
            dst_axis_sizes,
            method,
            use_dst_mask,
            cartesian=True,
            axes=axes,
            n_axes=n_axes,
        )

        # Update coordinates of new field
        f._regrid_update_coordinates(
            dst,
            dst_dict,
            dst_coords,
            src_axis_keys,
            dst_axis_keys,
            cartesian=True,
        )

        # Copy across destination fields coordinate references if necessary
        if not dst_dict:
            f._regrid_copy_coordinate_references(dst, dst_axis_keys)

        # Insert regridded data into new field
        f.set_data(new_data, axes=self.get_data_axes())

        # Release old memory
        regridSrc2Dst.destroy()
        dstfracfield.destroy()
        srcfracfield.destroy()
        dstfield.destroy()
        srcfield.destroy()
        dstgrid.destroy()
        srcgrid.destroy()

        return f

class XRFieldCollapseMixin():
    # ----------------------------------------------------------------
    # Worker functions for weights
    # ----------------------------------------------------------------
    def _weights_area_XY(
        self,
        comp,
        weights_axes,
        auto=False,
        measure=False,
        radius=None,
        methods=False,
    ):
        """Calculate area weights from X and Y dimension coordinate
        constructs.

        :Parameters:

            measure: `bool`
                If True then make sure that the weights represent true
                cell sizes.

            methods: `bool`, optional
                If True then add a description of the method used to
                create the weights to the *comp* dictionary, as opposed to
                the actual weights.

        :Returns:

            `bool` or `None`

        """
        xkey, xcoord = self.dimension_coordinate(
            "X", item=True, default=(None, None)
        )
        ykey, ycoord = self.dimension_coordinate(
            "Y", item=True, default=(None, None)
        )

        if xcoord is None or ycoord is None:
            if auto:
                return

            raise ValueError(
                "No unique 'X' and 'Y' dimension coordinate constructs for "
                "calculating area weights"
            )

        if xcoord.Units.equivalent(
            Units("radians")
        ) and ycoord.Units.equivalent(Units("radians")):
            pass
        elif xcoord.Units.equivalent(
            Units("metres")
        ) and ycoord.Units.equivalent(Units("metres")):
            radius = None
        else:
            if auto:
                return

            raise ValueError(
                "Insufficient coordinate constructs for calculating "
                "area weights"
            )

        xaxis = self.get_data_axes(xkey)[0]
        yaxis = self.get_data_axes(ykey)[0]

        for axis in (xaxis, yaxis):
            if axis in weights_axes:
                if auto:
                    return

                raise ValueError(
                    "Multiple weights specifications for "
                    f"{self.constructs.domain_axis_identity(axis)!r} axis"
                )

        if measure and radius is not None:
            radius = self.radius(default=radius)

        if measure or xcoord.size > 1:
            if not xcoord.has_bounds():
                if auto:
                    return

                raise ValueError(
                    "Can't create area weights: No bounds for "
                    f"{xcoord.identity()!r} axis"
                )

            if methods:
                comp[(xaxis,)] = "linear " + xcoord.identity()
            else:
                cells = xcoord.cellsize
                if xcoord.Units.equivalent(Units("radians")):
                    cells.Units = _units_radians
                    if measure:
                        cells *= radius
                        cells.override_units(radius.Units, inplace=True)
                else:
                    cells.Units = Units("metres")

                comp[(xaxis,)] = cells

            weights_axes.add(xaxis)

        if measure or ycoord.size > 1:
            if not ycoord.has_bounds():
                if auto:
                    return

                raise ValueError(
                    "Can't create area weights: No bounds for "
                    f"{ycoord.identity()!r} axis"
                )

            if ycoord.Units.equivalent(Units("radians")):
                ycoord = ycoord.clip(-90, 90, units=Units("degrees"))
                ycoord.sin(inplace=True)

                if methods:
                    comp[(yaxis,)] = "linear sine " + ycoord.identity()
                else:
                    cells = ycoord.cellsize
                    if measure:
                        cells *= radius

                    comp[(yaxis,)] = cells
            else:
                if methods:
                    comp[(yaxis,)] = "linear " + ycoord.identity()
                else:
                    cells = ycoord.cellsize
                    comp[(yaxis,)] = cells

            weights_axes.add(yaxis)

        return True

    def _weights_data(
        self,
        w,
        comp,
        weights_axes,
        axes=None,
        data=False,
        components=False,
        methods=False,
    ):
        """Creates weights for the field construct's data array.

        :Parameters:

            methods: `bool`, optional
                If True then add a description of the method used to
                create the weights to the *comp* dictionary, as
                opposed to the actual weights.

        """
        # --------------------------------------------------------
        # Data weights
        # --------------------------------------------------------
        field_data_axes = self.get_data_axes()

        if axes is not None:
            if isinstance(axes, (str, int)):
                axes = (axes,)

            if len(axes) != w.ndim:
                raise ValueError(
                    "'axes' parameter must provide an axis identifier "
                    "for each weights data dimension. Got {axes!r} for "
                    f"{w.ndim} dimension(s)."
                )

            iaxes = [
                field_data_axes.index(self.domain_axis(axis, key=True))
                for axis in axes
            ]

            if data:
                for i in range(self.ndim):
                    if i not in iaxes:
                        w = w.insert_dimension(position=i)
                        iaxes.insert(i, i)

                w = w.transpose(iaxes)

                if w.ndim > 0:
                    while w.shape[0] == 1:
                        w = w.squeeze(0)

        else:
            iaxes = list(range(self.ndim - w.ndim, self.ndim))

        if not (components or methods):
            if not self._is_broadcastable(w.shape):
                raise ValueError(
                    "The 'Data' weights (shape {}) are not broadcastable "
                    "to the field construct's data (shape {}).".format(
                        w.shape, self.shape
                    )
                )

            axes0 = field_data_axes[self.ndim - w.ndim :]
        else:
            axes0 = [field_data_axes[i] for i in iaxes]

        for axis0 in axes0:
            if axis0 in weights_axes:
                raise ValueError(
                    "Multiple weights specified for {!r} axis".format(
                        self.constructs.domain_axis_identity(axis0)
                    )
                )

        if methods:
            comp[tuple(axes0)] = "custom data"
        else:
            comp[tuple(axes0)] = w

        weights_axes.update(axes0)

        return True

    def _weights_field(self, fields, comp, weights_axes, methods=False):
        """Creates a weights field."""
        s = self.analyse_items()

        domain_axes = self.domain_axes(todict=True)
        #        domain_axes_size_1 = self.domain_axes(filter_by_size=(1,), todict=True)

        for w in fields:
            t = w.analyse_items()
            # TODO CHECK this with org

            if t["undefined_axes"]:
                #                if set(
                #                    t.domain_axes.filter_by_size(gt(1), view=True)
                #                ).intersection(t["undefined_axes"]):
                w_domain_axes_1 = w.domain_axes(
                    filter_by_size=(1,), todict=True
                )
                if set(w_domain_axes_1).intersection(t["undefined_axes"]):
                    raise ValueError("345jn456jn TODO")

            w = w.squeeze()

            w_domain_axes = w.domain_axes(todict=True)

            axis1_to_axis0 = {}

            coordinate_references = self.coordinate_references(todict=True)
            w_coordinate_references = w.coordinate_references(todict=True)

            for axis1 in w.get_data_axes():
                identity = t["axis_to_id"].get(axis1, None)
                if identity is None:
                    raise ValueError(
                        "Weights field has unmatched, size > 1 "
                        f"{w.constructs.domain_axis_identity(axis1)!r} axis"
                    )

                axis0 = s["id_to_axis"].get(identity, None)
                if axis0 is None:
                    raise ValueError(
                        f"Weights field has unmatched, size > 1 {identity!r} "
                        "axis"
                    )

                w_axis_size = w_domain_axes[axis1].get_size()
                self_axis_size = domain_axes[axis0].get_size()

                if w_axis_size != self_axis_size:
                    raise ValueError(
                        f"Weights field has incorrectly sized {identity!r} "
                        f"axis ({w_axis_size} != {self_axis_size})"
                    )

                axis1_to_axis0[axis1] = axis0

                # Check that the defining coordinate data arrays are
                # compatible
                key0 = s["axis_to_coord"][axis0]
                key1 = t["axis_to_coord"][axis1]

                if not self._equivalent_construct_data(
                    w, key0=key0, key1=key1, s=s, t=t
                ):
                    raise ValueError(
                        f"Weights field has incompatible {identity!r} "
                        "coordinates"
                    )

                # Still here? Then the defining coordinates have
                # equivalent data arrays

                # If the defining coordinates are attached to
                # coordinate references then check that those
                # coordinate references are equivalent
                refs0 = [
                    key
                    for key, ref in coordinate_references.items()
                    if key0 in ref.coordinates()
                ]
                refs1 = [
                    key
                    for key, ref in w_coordinate_references.items()
                    if key1 in ref.coordinates()
                ]

                nrefs = len(refs0)
                if nrefs > 1 or nrefs != len(refs1):
                    # The defining coordinate are associated with
                    # different numbers of coordinate references
                    equivalent_refs = False
                elif not nrefs:
                    # Neither defining coordinate is associated with a
                    # coordinate reference
                    equivalent_refs = True
                else:
                    # Each defining coordinate is associated with
                    # exactly one coordinate reference
                    equivalent_refs = self._equivalent_coordinate_references(
                        w, key0=refs0[0], key1=refs1[0], s=s, t=t
                    )

                if not equivalent_refs:
                    raise ValueError(
                        "Input weights field has an incompatible "
                        "coordinate reference"
                    )

            axes0 = tuple(
                [axis1_to_axis0[axis1] for axis1 in w.get_data_axes()]
            )

            for axis0 in axes0:
                if axis0 in weights_axes:
                    raise ValueError(
                        "Multiple weights specified for "
                        f"{self.constructs.domain_axis_identity(axis0)!r} "
                        "axis"
                    )

            comp[tuple(axes0)] = w.data

            weights_axes.update(axes0)

        return True

    def _weights_field_scalar(self, methods=False):
        """Return a scalar field of weights with long_name ``'weight'``.

        :Returns:

            `Field`

        """
        data = Data(1.0, "1")

        f = type(self)()
        f.set_data(data, copy=False)
        f.long_name = "weight"
        f.comment = "Weights for {!r}".format(self)

        return f

    def _weights_geometry_area(
        self,
        domain_axis,
        comp,
        weights_axes,
        auto=False,
        measure=False,
        radius=None,
        great_circle=False,
        return_areas=False,
        methods=False,
    ):
        """Creates area weights for polygon geometry cells.

        .. versionadded:: 3.2.0

        :Parameters:

            domain_axis : `str` or `None`

            measure: `bool`
                If True then make sure that the weights represent true
                cell sizes.

            methods: `bool`, optional
                If True then add a description of the method used to
                create the weights to the *comp* dictionary, as opposed to
                the actual weights.

        :Returns:

            `bool` or `Data`

        """
        axis, aux_X, aux_Y, aux_Z = self._weights_yyy(
            domain_axis, "polygon", methods=methods, auto=auto
        )

        if axis is None:
            if auto:
                return False

            if domain_axis is None:
                raise ValueError("No polygon geometries")

            raise ValueError(
                "No polygon geometries for {!r} axis".format(
                    self.constructs.domain_axis_identity(domain_axis)
                )
            )

        if axis in weights_axes:
            if auto:
                return False

            raise ValueError(
                "Multiple weights specifications for {!r} axis".format(
                    self.constructs.domain_axis_identity(axis)
                )
            )

        # Check for interior rings
        interior_ring_X = aux_X.get_interior_ring(None)
        interior_ring_Y = aux_Y.get_interior_ring(None)
        if interior_ring_X is None and interior_ring_Y is None:
            interior_ring = None
        elif interior_ring_X is None:
            raise ValueError(
                "Can't find weights: X coordinates have missing "
                "interior ring variable"
            )
        elif interior_ring_Y is None:
            raise ValueError(
                "Can't find weights: Y coordinates have missing "
                "interior ring variable"
            )
        elif not interior_ring_X.data.equals(interior_ring_Y.data):
            raise ValueError(
                "Can't find weights: Interior ring variables for X and Y "
                "coordinates have different data values"
            )
        else:
            interior_ring = interior_ring_X.data
            if interior_ring.shape != aux_X.bounds.shape[:-1]:
                raise ValueError(
                    "Can't find weights: Interior ring variables have "
                    "incorrect shape. Got {}, expected {}".format(
                        interior_ring.shape, aux_X.bounds.shape[:-1]
                    )
                )

        x = aux_X.bounds.data
        y = aux_Y.bounds.data

        if x.Units.equivalent(_units_metres) and y.Units.equivalent(
            _units_metres
        ):
            # ----------------------------------------------------
            # Plane polygons defined by straight lines.
            #
            # Use the shoelace formula:
            # https://en.wikipedia.org/wiki/Shoelace_formula
            #
            # Do this in preference to weights based on spherical
            # polygons, which require the great circle assumption.
            # ----------------------------------------------------
            spherical = False

            if methods:
                comp[(axis,)] = "area plane polygon geometry"
                return True

            y.Units = x.Units

            all_areas = (x[..., :-1] * y[..., 1:]).sum(-1, squeeze=True) - (
                x[..., 1:] * y[..., :-1]
            ).sum(-1, squeeze=True)

            for i, (parts_x, parts_y) in enumerate(zip(x, y)):
                for j, (nodes_x, nodes_y) in enumerate(zip(parts_x, parts_y)):
                    nodes_x = nodes_x.compressed()
                    nodes_y = nodes_y.compressed()

                    if (nodes_x.size and nodes_x[0] != nodes_x[-1]) or (
                        nodes_y.size and nodes_y[0] != nodes_y[-1]
                    ):
                        # First and last nodes of this polygon
                        # part are different => need to account
                        # for the "last" edge of the polygon that
                        # joins the first and last points.
                        all_areas[i, j] += x[-1] * y[0] - x[0] * y[-1]

            all_areas = all_areas.abs() * 0.5

        elif x.Units.equivalent(_units_radians) and y.Units.equivalent(
            _units_radians
        ):
            # ----------------------------------------------------
            # Spherical polygons defined by great circles
            #
            # The area of such a spherical polygon is given by the
            # sum of the interior angles minus (N-2)pi, where N is
            # the number of sides (Todhunter,
            # https://en.wikipedia.org/wiki/Spherical_trigonometry#Spherical_polygons):
            #
            # Area of N-sided polygon on the unit sphere =
            #     \left(\sum _{n=1}^{N}A_{n}\right) - (N - 2)\pi
            #
            # where A_{n} denotes the n-th interior angle.
            # ----------------------------------------------------
            spherical = True

            if not great_circle:
                raise ValueError(
                    "Must set great_circle=True to allow the derivation of "
                    "area weights from spherical polygons composed from "
                    "great circle segments."
                )

            if methods:
                comp[(axis,)] = "area spherical polygon geometry"
                return True

            x.Units = _units_radians
            y.Units = _units_radians

            interior_angle = self._weights_interior_angle(x, y)

            # Find the number of edges of each polygon (note that
            # this number may be one too few, but we'll adjust for
            # that later).
            N = interior_angle.sample_size(-1, squeeze=True)

            all_areas = (
                interior_angle.sum(-1, squeeze=True) - (N - 2) * numpy_pi
            )

            for i, (parts_x, parts_y) in enumerate(zip(x, y)):
                for j, (nodes_x, nodes_y) in enumerate(zip(parts_x, parts_y)):
                    nodes_x = nodes_x.compressed()
                    nodes_y = nodes_y.compressed()

                    if (nodes_x.size and nodes_x[0] != nodes_x[-1]) or (
                        nodes_y.size and nodes_y[0] != nodes_y[-1]
                    ):
                        # First and last nodes of this polygon
                        # part are different => need to account
                        # for the "last" edge of the polygon that
                        # joins the first and last points.
                        interior_angle = self._weights_interior_angle(
                            nodes_x[[0, -1]], nodes_y[[0, -1]]
                        )

                        all_areas[i, j] += interior_angle + numpy_pi

            area_min = all_areas.min()
            if area_min < 0:
                raise ValueError(
                    "A spherical polygon geometry part has negative area"
                )
        else:
            return False

        # Change the sign of areas for polygons that are interior
        # rings
        if interior_ring is not None:
            all_areas.where(interior_ring, -all_areas, inplace=True)

        # Sum the areas of each part to get the total area of each
        # cell
        areas = all_areas.sum(-1, squeeze=True)

        if measure and spherical and aux_Z is not None:
            # Multiply by radius squared, accounting for any Z
            # coordinates, to get the actual area
            z = aux_Z.get_data(None, _fill_value=False)
            if z is None:
                r = radius
            else:
                if not z.Units.equivalent(_units_metres):
                    raise ValueError(
                        "Z coordinates must have units equivalent to "
                        f"metres for area calculations. Got {z.Units!r}"
                    )

                positive = aux_Z.get_property("positive", None)
                if positive is None:
                    raise ValueError("TODO")

                if positive.lower() == "up":
                    r = radius + z
                elif positive.lower() == "down":
                    r = radius - z
                else:
                    raise ValueError(
                        "Bad value of Z coordinate 'positive' "
                        f"property: {positive!r}."
                    )

            areas *= r ** 2

        if return_areas:
            return areas

        comp[(axis,)] = areas

        weights_axes.add(axis)

        return True

    def _weights_geometry_line(
        self,
        domain_axis,
        comp,
        weights_axes,
        auto=False,
        measure=False,
        radius=None,
        great_circle=False,
        methods=False,
    ):
        """Creates line-length weights for line geometries.

        .. versionadded:: 3.2.0

        :Parameters:

            measure: `bool`
                If True then make sure that the weights represent true
                cell sizes.

            methods: `bool`, optional
                If True then add a description of the method used to
                create the weights to the *comp* dictionary, as opposed to
                the actual weights.

        """
        axis, aux_X, aux_Y, aux_Z = self._weights_yyy(
            domain_axis, "line", methods=methods, auto=auto
        )

        if axis is None:
            if auto:
                return False

            if domain_axis is None:
                raise ValueError("No line geometries")

            raise ValueError(
                "No line geometries for {!r} axis".format(
                    self.constructs.domain_axis_identity(domain_axis)
                )
            )

        if axis in weights_axes:
            if auto:
                return False

            raise ValueError(
                "Multiple weights specifications for {!r} axis".format(
                    self.constructs.domain_axis_identity(axis)
                )
            )

        x = aux_X.bounds.data
        y = aux_Y.bounds.data

        if x.Units.equivalent(_units_metres) and y.Units.equivalent(
            _units_metres
        ):
            # ----------------------------------------------------
            # Plane lines.
            #
            # Each line segment is the simple cartesian distance
            # between two adjacent nodes.
            # ----------------------------------------------------
            if methods:
                comp[(axis,)] = "linear plane line geometry"
                return True

            y.Units = x.Units

            delta_x = x.diff(axis=-1)
            delta_y = y.diff(axis=-1)

            all_lengths = (delta_x ** 2 + delta_y ** 2) ** 0.5
            all_lengths = all_lengths.sum(-1, squeeze=True)

        elif x.Units.equivalent(_units_radians) and y.Units.equivalent(
            _units_radians
        ):
            # ----------------------------------------------------
            # Spherical lines.
            #
            # Each line segment is a great circle arc between two
            # adjacent nodes.
            #
            # The length of the great circle arc is the the
            # interior angle multiplied by the radius. The
            # interior angle is calculated with a special case of
            # the Vincenty formula:
            # https://en.wikipedia.org/wiki/Great-circle_distance
            # ----------------------------------------------------
            if not great_circle:
                raise ValueError(
                    "Must set great_circle=True to allow the derivation "
                    "of line-length weights from great circle segments."
                )

            if methods:
                comp[(axis,)] = "linear spherical line geometry"
                return True

            x.Units = _units_radians
            y.Units = _units_radians

            interior_angle = self._weights_interior_angle(x, y)
            if interior_angle.min() < 0:
                raise ValueError(
                    "A spherical line geometry segment has "
                    "negative length: {!r}".format(
                        interior_angle.min() * radius
                    )
                )

            all_lengths = interior_angle.sum(-1, squeeze=True)

            if measure:
                all_lengths *= radius
        else:
            return False

        # Sum the lengths of each part to get the total length of
        # each cell
        lengths = all_lengths.sum(-1, squeeze=True)

        comp[(axis,)] = lengths

        weights_axes.add(axis)

        return True

    def _weights_geometry_volume(
        self,
        comp,
        weights_axes,
        auto=False,
        measure=False,
        radius=None,
        great_circle=False,
        methods=False,
    ):
        """Creates volume weights for polygon geometry cells.

        .. versionadded:: 3.2.0

        :Parameters:

            measure: `bool`
                If True then make sure that the weights represent true
                cell sizes.

            methods: `bool`, optional
                If True then add a description of the method used to
                create the weights to the *comp* dictionary, as opposed to
                the actual weights.

        """
        axis, aux_X, aux_Y, aux_Z = self._weights_yyy(
            "polygon", methods=methods, auto=auto
        )

        if axis is None and auto:
            return False

        if axis in weights_axes:
            if auto:
                return False

            raise ValueError(
                "Multiple weights specifications for {!r} axis".format(
                    self.constructs.domain_axis_identity(axis)
                )
            )

        x = aux_X.bounds.data
        y = aux_Y.bounds.data
        z = aux_Z.bounds.data

        if not z.Units.equivalent(_units_metres):
            if auto:
                return False

            raise ValueError("TODO")

        if not methods:
            # Initialise cell volumes as the cell areas
            volumes = self._weights_geometry_area(
                comp,
                weights_axes,
                auto=auto,
                measure=measure,
                radius=radius,
                great_circle=great_circle,
                methods=False,
                return_areas=True,
            )

            if measure:
                delta_z = abs(z[..., 1] - z[..., 0])
                delta_z.squeeze(axis=-1, inplace=True)

        if x.Units.equivalent(_units_metres) and y.Units.equivalent(
            _units_metres
        ):
            # ----------------------------------------------------
            # Plane polygons defined by straight lines.
            #
            # Do this in preference to weights based on spherical
            # polygons, which require the great circle assumption.
            # ----------------------------------------------------
            if methods:
                comp[(axis,)] = "volume plane polygon geometry"
                return True

            if measure:
                volumes *= delta_z

        elif x.Units.equivalent(_units_radians) and y.Units.equivalent(
            _units_radians
        ):
            # ----------------------------------------------------
            # Spherical polygons defined by great circles
            #
            # The area of such a spherical polygon is given by the
            # sum of the interior angles minus (N-2)pi, where N is
            # the number of sides (Todhunter):
            #
            # https://en.wikipedia.org/wiki/Spherical_trigonometry#Area_and_spherical_excess
            #
            # The interior angle of a side is calculated with a
            # special case of the Vincenty formula:
            # https://en.wikipedia.org/wiki/Great-circle_distance
            # ----------------------------------------------------
            if not great_circle:
                raise ValueError(
                    "Must set great_circle=True to allow the derivation "
                    "of volume weights from spherical polygons composed "
                    "from great circle segments."
                )

            if methods:
                comp[(axis,)] = "volume spherical polygon geometry"
                return True

            if measure:
                r = radius

                # actual_volume =
                #    [actual_area/(4*pi*r**2)]
                #    * (4/3)*pi*[(r+delta_z)**3 - r**3)]
                volumes *= (
                    delta_z ** 3 / (3 * r ** 2) + delta_z ** 2 / r + delta_z
                )
        else:
            raise ValueError("TODO")

        comp[(axis,)] = volumes

        weights_axes.add(axis)

        return True

    def _weights_interior_angle(self, data_lambda, data_phi):
        """Find the interior angle between each adjacent pair of
        geometry nodes defined on a sphere.

        The interior angle of two points on the sphere is calculated with
        a special case of the Vincenty formula
        (https://en.wikipedia.org/wiki/Great-circle_distance):

        \Delta \sigma =\arctan {
            \frac {\sqrt {\left(\cos \phi _{2}\sin(\Delta \lambda )\right)^{2} +
                   \left(\cos \phi _{1}\sin \phi _{2} -
                         \sin \phi _{1}\cos \phi _{2}\cos(\Delta \lambda )\right)^{2} } }
                  {\sin \phi _{1}\sin \phi _{2} +
                   \cos \phi _{1}\cos \phi _{2}\cos(\Delta \lambda )}
                  }

        :Parameters:

            data_lambda: `Data`
                Longitudes. Must have units of radians, which is not
                checked.

            data_phi: `Data`
                Latitudes. Must have units of radians, which is not
                checked.

        :Returns:

            `Data`
                The interior angles in units of radians.

        """
        delta_lambda = data_lambda.diff(axis=-1)

        cos_phi = data_phi.cos()
        sin_phi = data_phi.sin()

        cos_phi_1 = cos_phi[..., :-1]
        cos_phi_2 = cos_phi[..., 1:]

        sin_phi_1 = sin_phi[..., :-1]
        sin_phi_2 = sin_phi[..., 1:]

        cos_delta_lambda = delta_lambda.cos()
        sin_delta_lambda = delta_lambda.sin()

        numerator = (
            (cos_phi_2 * sin_delta_lambda) ** 2
            + (
                cos_phi_1 * sin_phi_2
                - sin_phi_1 * cos_phi_2 * cos_delta_lambda
            )
            ** 2
        ) ** 0.5

        denominator = (
            sin_phi_1 * sin_phi_2 + cos_phi_1 * cos_phi_2 * cos_delta_lambda
        )

        # TODO RuntimeWarning: overflow encountered in true_divide comes from
        # numerator/denominator with missing values

        interior_angle = (numerator / denominator).arctan()

        interior_angle.override_units(_units_1, inplace=True)

        return interior_angle

    def _weights_linear(
        self,
        axis,
        comp,
        weights_axes,
        auto=False,
        measure=False,
        methods=False,
    ):
        """1-d linear weights from dimension coordinate constructs.

        :Parameters:

            axis: `str`

            comp: `dict`

            weights_axes: `set`

            auto: `bool`

            measure: `bool`
                If True then make sure that the weights represent true
                cell sizes.

            methods: `bool`, optional
                If True then add a description of the method used to
                create the weights to the *comp* dictionary, as opposed to
                the actual weights.

        :Returns:

            `bool`

        """
        da_key = self.domain_axis(axis, key=True, default=None)
        if da_key is None:
            if auto:
                return False

            raise ValueError(
                "Can't create weights: Can't find domain axis "
                f"matching {axis!r}"
            )

        dim = self.dimension_coordinate(filter_by_axis=(da_key,), default=None)
        if dim is None:
            if auto:
                return False

            raise ValueError(
                f"Can't create linear weights for {axis!r} axis: Can't find "
                "dimension coordinate construct."
            )

        if not measure and dim.size == 1:
            return False

        if da_key in weights_axes:
            if auto:
                return False

            raise ValueError(
                f"Can't create linear weights for {axis!r} axis: Multiple "
                "axis specifications"
            )

        if not dim.has_bounds():
            # Dimension coordinate has no bounds
            if auto:
                return False

            raise ValueError(
                f"Can't create linear weights for {axis!r} axis: No bounds"
            )
        else:
            # Bounds exist
            if methods:
                comp[
                    (da_key,)
                ] = "linear " + self.constructs.domain_axis_identity(da_key)
            else:
                comp[(da_key,)] = dim.cellsize

        weights_axes.add(da_key)

        return True

    def _weights_measure(
        self, measure, comp, weights_axes, methods=False, auto=False
    ):
        """Cell measure weights.

        :Parameters:

            methods: `bool`, optional
                If True then add a description of the method used to
                create the weights to the *comp* dictionary, as opposed to
                the actual weights.

        :Returns:

            `bool`

        """
        m = self.cell_measures(filter_by_measure=(measure,), todict=True)
        len_m = len(m)

        if not len_m:
            if measure == "area":
                return False

            if auto:
                return

            raise ValueError(
                f"Can't find weights: No {measure!r} cell measure"
            )

        elif len_m > 1:
            if auto:
                return False

            raise ValueError(
                f"Can't find weights: Multiple {measure!r} cell measures"
            )

        key, clm = m.popitem()

        clm_axes0 = self.get_data_axes(key)

        clm_axes = tuple(
            [axis for axis, n in zip(clm_axes0, clm.data.shape) if n > 1]
        )

        for axis in clm_axes:
            if axis in weights_axes:
                if auto:
                    return False

                raise ValueError(
                    "Multiple weights specifications for {!r} "
                    "axis".format(self.constructs.domain_axis_identity(axis))
                )

        clm = clm.get_data(_fill_value=False).copy()
        if clm_axes != clm_axes0:
            iaxes = [clm_axes0.index(axis) for axis in clm_axes]
            clm.squeeze(iaxes, inplace=True)

        if methods:
            comp[tuple(clm_axes)] = measure + " cell measure"
        else:
            comp[tuple(clm_axes)] = clm

        weights_axes.update(clm_axes)

        return True

    def _weights_scale(self, w, scale):
        """Scale the weights so that they are <= scale.

        :Parameters:

            w: `Data

            scale: number

        :Returns:

            `Data`

        """
        scale = Data.asdata(scale).datum()
        if scale <= 0:
            raise ValueError(
                "'scale' parameter must be a positive number. " f"Got {scale}"
            )

        wmax = w.maximum()
        factor = wmax / scale
        factor.dtype = float
        if numpy_can_cast(factor.dtype, w.dtype):
            w /= factor
        else:
            w = w / factor

        return w

    def _weights_yyy(
        self, domain_axis, geometry_type, methods=False, auto=False
    ):
        """Checks whether weights can be created for given coordinates.

        .. versionadded:: 3.2.0

        :Parameters:

            domain_axis : `str` or `None`

            geometry_type: `str`
                Either ``'polygon'`` or ``'line'``.

            auto: `bool`

        :Returns:

            `tuple`

        """
        aux_X = None
        aux_Y = None
        aux_Z = None
        x_axis = None
        y_axis = None
        z_axis = None

        auxiliary_coordinates_1d = self.auxiliary_coordinates(
            filter_by_naxes=(1,), todict=True
        )

        for key, aux in auxiliary_coordinates_1d.items():
            if aux.get_geometry(None) != geometry_type:
                continue

            if aux.X:
                aux_X = aux.copy()
                x_axis = self.get_data_axes(key)[0]
                if domain_axis is not None and x_axis != domain_axis:
                    aux_X = None
                    continue
            elif aux.Y:
                aux_Y = aux.copy()
                y_axis = self.get_data_axes(key)[0]
                if domain_axis is not None and y_axis != domain_axis:
                    aux_Y = None
                    continue
            elif aux.Z:
                aux_Z = aux.copy()
                z_axis = self.get_data_axes(key)[0]
                if domain_axis is not None and z_axis != domain_axis:
                    aux_Z = None
                    continue

        if aux_X is None or aux_Y is None:
            if auto:
                return (None,) * 4

            raise ValueError(
                "Can't create weights: Need both X and Y nodes to "
                f"calculate {geometry_type} geometry weights"
            )

        if x_axis != y_axis:
            if auto:
                return (None,) * 4

            raise ValueError(
                "Can't create weights: X and Y nodes span different "
                "domain axes"
            )

        axis = x_axis

        if aux_X.get_bounds(None) is None or aux_Y.get_bounds(None) is None:
            # Not both X and Y coordinates have bounds
            if auto:
                return (None,) * 4

            raise ValueError("Not both X and Y coordinates have bounds")

        if aux_X.bounds.shape != aux_Y.bounds.shape:
            if auto:
                return (None,) * 4

            raise ValueError(
                "Can't find weights: X and Y geometry coordinate bounds "
                "must have the same shape. "
                f"Got {aux_X.bounds.shape} and {aux_Y.bounds.shape}"
            )

        if not methods:
            if aux_X.bounds.data.fits_in_one_chunk_in_memory(
                aux_X.bounds.dtype.itemsize
            ):
                aux_X.bounds.varray
            if aux_X.bounds.data.fits_in_one_chunk_in_memory(
                aux_Y.bounds.dtype.itemsize
            ):
                aux_X.bounds.varray

        if aux_Z is None:
            for key, aux in auxiliary_coordinates_1d.items():
                if aux.Z:
                    aux_Z = aux.copy()
                    z_axis = self.get_data_axes(key)[0]

        # Check Z coordinates
        if aux_Z is not None:
            if z_axis != x_axis:
                if auto:
                    return (None,) * 4

                raise ValueError(
                    "Z coordinates span different domain axis to X and Y "
                    "geometry coordinates"
                )

        return axis, aux_X, aux_Y, aux_Z

    # ----------------------------------------------------------------
    # End of worker functions for weights
    # ----------------------------------------------------------------


    def weights(
        self,
        weights=True,
        scale=None,
        measure=False,
        components=False,
        methods=False,
        radius="earth",
        data=False,
        great_circle=False,
        axes=None,
        **kwargs,
    ):
        """Return weights for the data array values.

        The weights are those used during a statistical collapse of the
        data. For example when computing a area weight average.

        Weights for any combination of axes may be returned.

        Weights are either derived from the field construct's metadata
        (such as coordinate cell sizes) or provided explicitly in the form
        of other `Field` constructs. In any case, the outer product of
        these weights components is returned in a field which is
        broadcastable to the original field (see the *components* parameter
        for returning the components individually).

        By default null, equal weights are returned.

        .. versionadded:: 1.0

        .. seealso:: `bin`, `cell_area`, `collapse`, `moving_window`,
                     `radius`

        :Parameters:

            weights: *optional*
                Specify the weights to be created. There are three
                distinct methods:

                * **Type 1** will create weights for all axes of size
                  greater than 1, raising an exception if this is not
                  possible (this is the default).;

                * **Type 2** will always succeed in creating weights for
                  all axes of the field, even if some of those weights are
                  null.

                * **Type 3** allows particular types of weights to be
                  defined for particular axes, and an exception will be
                  raised if it is not possible to the create weights.

            ..

                **Type 1** and **Type 2** come at the expense of not
                always being able to control exactly how the weights are
                created (although which methods were used can be inspected
                with use of the *methods* parameter).

                * **Type 1**: *weights* may be:

                  ==========  ============================================
                  *weights*   Description
                  ==========  ============================================
                  `True`      This is the default. Weights are created for
                              all axes (or a subset of them, see the
                              *axes* parameter). Set the *methods*
                              parameter to find out how the weights were
                              actually created.

                              The weights components are created for axes
                              of the field by one or more of the following
                              methods, in order of preference,

                                1. Volume cell measures
                                2. Area cell measures
                                3. Area calculated from (grid) latitude
                                   and (grid) longitude dimension
                                   coordinate constructs with bounds
                                4. Cell sizes of dimension coordinate
                                   constructs with bounds
                                5. Equal weights

                              and the outer product of these weights
                              components is returned in a field constructs
                              which is broadcastable to the original field
                              construct (see the *components* parameter).
                  ==========  ============================================

                * **Type 2**: *weights* may be one of:

                  ==========  ============================================
                  *weights*   Description
                  ==========  ============================================
                  `None`      Equal weights for all axes.

                  `False`     Equal weights for all axes.

                  `Data`      Explicit weights in a `Data` object that
                              must be broadcastable to the field
                              construct's data, unless the *axes*
                              parameter is also set.

                  `dict`      Explicit weights in a dictionary of the form
                              that is returned from a call to the
                              `weights` method with ``component=True``
                  ==========  ============================================

                * **Type 3**: *weights* may be one, or a sequence, of:

                  ============  ==========================================
                  *weights*     Description
                  ============  ==========================================
                  ``'area'``    Cell area weights from the field
                                construct's area cell measure construct
                                or, if one doesn't exist, from (grid)
                                latitude and (grid) longitude dimension
                                coordinate constructs. Set the *methods*
                                parameter to find out how the weights were
                                actually created.

                  ``'volume'``  Cell volume weights from the field
                                construct's volume cell measure construct.

                  `str`         Weights from the cell sizes of the
                                dimension coordinate construct with this
                                identity.

                  `Field`       Explicit weights from the data of another
                                field construct, which must be
                                broadcastable to this field construct.
                  ============  ==========================================

                If *weights* is a sequence of any combination of the
                above then the returned field contains the outer product
                of the weights defined by each element of the
                sequence. The ordering of the sequence is irrelevant.

                *Parameter example:*
                  To create to 2-dimensional weights based on cell
                  areas: ``f.weights('area')``. To create to
                  3-dimensional weights based on cell areas and linear
                  height: ``f.weights(['area', 'Z'])``.

            scale: number, optional
                If set to a positive number then scale the weights so that
                they are less than or equal to that number. If weights
                components have been requested (see the *components*
                parameter) then each component is scaled independently of
                the others.

                *Parameter example:*
                  To scale all weights so that they lie between 0 and 1:
                  ``scale=1``.

            measure: `bool`, optional
                Create weights that are cell measures, i.e. which
                describe actual cell sizes (e.g. cell areas) with
                appropriate units (e.g. metres squared).

                Cell measures can be created for any combination of
                axes. For example, cell measures for a time axis are
                the time span for each cell with canonical units of
                seconds; cell measures for the combination of four
                axes representing time and three dimensional space
                could have canonical units of metres cubed seconds.

                .. note:: Specifying cell volume weights via
                          ``weights=['X', 'Y', 'Z']`` or
                          ``weights=['area', 'Z']`` (or other
                          equivalents) will produce **an incorrect
                          result if the vertical dimension coordinates
                          do not define the actual height or depth
                          thickness of every cell in the domain**. In
                          this case, ``weights='volume'`` should be
                          used instead, which requires the field
                          construct to have a "volume" cell measure
                          construct.

                          If ``weights=True`` then care also needs to
                          be taken, as a "volume" cell measure
                          construct will be used if present, otherwise
                          the cell volumes will be calculated using
                          the size of the vertical coordinate cells.

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

            components: `bool`, optional
                If True then a dictionary of orthogonal weights components
                is returned instead of a field. Each key is a tuple of
                integers representing axis positions in the field
                construct's data, with corresponding values of weights in
                `Data` objects. The axes of weights match the axes of the
                field construct's data array in the order given by their
                dictionary keys.

            methods: `bool`, optional
                If True, then return a dictionary describing methods used
                to create the weights.

            data: `bool`, optional
                If True then return the weights in a `Data` instance that
                is broadcastable to the original data.

                .. versionadded:: 3.1.0

            great_circle: `bool`, optional
                If True then allow, if required, the derivation of i) area
                weights from polygon geometry cells by assuming that each
                cell part is a spherical polygon composed of great circle
                segments; and ii) and the derivation of line-length
                weights from line geometry cells by assuming that each
                line part is composed of great circle segments.

                .. versionadded:: 3.2.0

            axes: (sequence of) `int` or `str`, optional
                Modify the behaviour when *weights* is `True` or a `Data`
                instance. Ignored for any other value the *weights*
                parameter.

                If *weights* is `True` then weights are created only for
                the specified axes (as opposed to all
                axes). I.e. ``weight=True, axes=axes`` is identical to
                ``weights=axes``.

                If *weights* is a `Data` instance then the specified axes
                identify each dimension of the given weights. If the
                weights do not broadcast to the field construct's data
                then setting the *axes* parameter is required so that the
                broadcasting can be inferred, otherwise setting the *axes*
                is not required.

                *Parameter example:*
                  ``axes='T'``

                *Parameter example:*
                  ``axes=['longitude']``

                *Parameter example:*
                  ``axes=[3, 1]``

                .. versionadded:: 3.3.0

            kwargs: deprecated at version 3.0.0.

        :Returns:

            `Field` or `Data` or `dict`
                The weights field; or if *data* is True, weights data in
                broadcastable form; or if *components* is True, orthogonal
                weights in a dictionary.

        **Examples:**

        >>> f
        <CF Field: air_temperature(time(12), latitude(145), longitude(192)) K>
        >>> f.weights()
        <CF Field: long_name:weight(time(12), latitude(145), longitude(192)) 86400 s.rad>
        >>> f.weights(scale=1.0)
        <CF Field: long_name:weight(time(12), latitude(145), longitude(192)) 1>
        >>> f.weights(components=True)
        {(0,): <CF Data(12): [30.0, ..., 31.0] d>,
         (1,): <CF Data(145): [5.94949998503e-05, ..., 5.94949998503e-05]>,
         (2,): <CF Data(192): [0.0327249234749, ..., 0.0327249234749] radians>}
        >>> f.weights(components=True, scale=1.0)
        {(0,): <CF Data(12): [0.967741935483871, ..., 1.0] 1>,
         (1,): <CF Data(145): [0.00272710399807, ..., 0.00272710399807]>,
         (2,): <CF Data(192): [1.0, ..., 1.0]>}
        >>> f.weights(components=True, scale=2.0)
        {(0,): <CF Data(12): [1.935483870967742, ..., 2.0] 1>,
         (1,): <CF Data(145): [0.00545420799614, ..., 0.00545420799614]>,
         (2,): <CF Data(192): [2.0, ..., 2.0]>}
        >>> f.weights(methods=True)
        {(0,): 'linear time',
         (1,): 'linear sine latitude',
         (2,): 'linear longitude'}

        """
        if isinstance(weights, str) and weights == "auto":
            _DEPRECATION_ERROR_KWARG_VALUE(
                self,
                "weights",
                "weights",
                "auto",
                message="Use value True instead.",
                version="3.0.7",
            )  # pragma: no cover

        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "weights", kwargs
            )  # pragma: no cover

        if measure and scale is not None:
            raise ValueError("Can't set measure=True and scale")

        if components and data:
            raise ValueError("Can't set components=True and data=True")

        if weights is None or weights is False:
            # --------------------------------------------------------
            # All equal weights
            # --------------------------------------------------------
            if components or methods:
                # Return an empty components dictionary
                return {}

            if data:
                # Return an scalar Data instance
                return Data(1.0, "1")

            # Return a field containing a single weight of 1
            return self._weights_field_scalar()

        # Still here?
        if methods:
            components = True

        comp = {}
        field_data_axes = self.get_data_axes()

        # All axes which have weights
        weights_axes = set()

        if radius is not None:
            radius = self.radius(default=radius)

        if weights is True and axes is not None:
            # --------------------------------------------------------
            # Restrict weights to the specified axes
            # --------------------------------------------------------
            weights = axes

        if weights is True:
            # --------------------------------------------------------
            # Auto-detect all weights
            # --------------------------------------------------------
            # Volume weights
            if self._weights_measure(
                "volume", comp, weights_axes, methods=methods, auto=True
            ):
                # Found volume weights from cell measures
                pass

            elif self._weights_measure(
                "area", comp, weights_axes, methods=methods, auto=True
            ):
                # Found area weights from cell measures
                pass
            elif self._weights_area_XY(
                comp,
                weights_axes,
                measure=measure,
                radius=radius,
                methods=methods,
                auto=True,
            ):
                # Found area weights from X and Y dimension
                # coordinates
                pass

            domain_axes = self.domain_axes(todict=True)

            for da_key in domain_axes:
                if self._weights_geometry_area(
                    da_key,
                    comp,
                    weights_axes,
                    measure=measure,
                    radius=radius,
                    great_circle=great_circle,
                    methods=methods,
                    auto=True,
                ):
                    # Found area weights from polygon geometries
                    pass
                elif self._weights_geometry_line(
                    da_key,
                    comp,
                    weights_axes,
                    measure=measure,
                    radius=radius,
                    great_circle=great_circle,
                    methods=methods,
                    auto=True,
                ):
                    # Found linear weights from line geometries
                    pass
                elif self._weights_linear(
                    da_key,
                    comp,
                    weights_axes,
                    measure=measure,
                    methods=methods,
                    auto=True,
                ):
                    # Found linear weights from dimension coordinates
                    pass

            weights_axes = []
            for key in comp:
                weights_axes.extend(key)

            size_N_axes = []
            for key, domain_axis in domain_axes.items():
                if domain_axis.get_size(0) > 1:
                    size_N_axes.append(key)

            missing_axes = set(size_N_axes).difference(weights_axes)
            if missing_axes:
                raise ValueError(
                    "Can't find weights for {!r} axis.".format(
                        self.constructs.domain_axis_identity(
                            missing_axes.pop()
                        )
                    )
                )

        elif isinstance(weights, dict):
            # --------------------------------------------------------
            # Dictionary
            # --------------------------------------------------------
            for key, value in weights.items():
                key = [self.domain_axis(i, key=True) for i in key]
                for k in key:
                    if k not in field_data_axes:
                        raise ValueError("TODO {!r} domain axis".format(k))

                multiple_weights = weights_axes.intersection(key)
                if multiple_weights:
                    raise ValueError(
                        "Can't find weights: Multiple specifications for {!r} "
                        "domain axis".format(
                            self.constructs.domain_axis_identity(
                                multiple_weights.pop()
                            )
                        )
                    )

                weights_axes.update(key)

                if methods:
                    comp[tuple(key)] = "custom data"
                else:
                    comp[tuple(key)] = value.copy()

        elif isinstance(weights, self.__class__):
            # --------------------------------------------------------
            # Field
            # --------------------------------------------------------
            self._weights_field([weights], comp, weights_axes)

        elif isinstance(weights, Data):
            # --------------------------------------------------------
            # Data
            # --------------------------------------------------------
            self._weights_data(
                weights,
                comp,
                weights_axes,
                axes=axes,
                data=data,
                components=components,
                methods=methods,
            )
        else:
            # --------------------------------------------------------
            # String or sequence
            # --------------------------------------------------------
            fields = []
            axes = []
            cell_measures = []

            if isinstance(weights, str):
                if weights in ("area", "volume"):
                    cell_measures = (weights,)
                else:
                    axes.append(weights)
            else:
                # In rare edge cases, e.g. if a user sets:
                #     weights=f[0].cell_area
                # when they mean weights=f[0].cell_area(), we reach this
                # code but weights is not iterable. So check it is first:
                try:
                    weights = iter(weights)
                except TypeError:
                    raise TypeError(
                        "Invalid type of 'weights' parameter: {}".format(
                            weights
                        )
                    )

                for w in tuple(weights):
                    if isinstance(w, self.__class__):
                        fields.append(w)
                    elif isinstance(w, Data):
                        raise ValueError("TODO")
                    elif w in ("area", "volume"):
                        cell_measures.append(w)
                    else:
                        axes.append(w)

            # Field weights
            self._weights_field(fields, comp, weights_axes)

            # Volume weights
            if "volume" in cell_measures:
                self._weights_measure(
                    "volume", comp, weights_axes, methods=methods, auto=False
                )

            # Area weights
            if "area" in cell_measures:
                if self._weights_measure(
                    "area", comp, weights_axes, methods=methods, auto=True
                ):
                    # Found area weights from cell measures
                    pass
                elif self._weights_area_XY(
                    comp,
                    weights_axes,
                    measure=measure,
                    radius=radius,
                    methods=methods,
                    auto=True,
                ):
                    # Found area weights from X and Y dimension
                    # coordinates
                    pass
                else:
                    # Found area weights from polygon geometries
                    self._weights_geometry_area(
                        None,
                        comp,
                        weights_axes,
                        measure=measure,
                        radius=radius,
                        great_circle=great_circle,
                        methods=methods,
                        auto=False,
                    )

            for axis in axes:
                da_key = self.domain_axis(axis, key=True, default=None)
                if da_key is None:
                    raise ValueError(
                        "Can't create weights: "
                        "Can't find axis matching {!r}".format(axis)
                    )

                if self._weights_geometry_area(
                    da_key,
                    comp,
                    weights_axes,
                    measure=measure,
                    radius=radius,
                    great_circle=great_circle,
                    methods=methods,
                    auto=True,
                ):
                    # Found area weights from polygon geometries
                    pass
                elif self._weights_geometry_line(
                    da_key,
                    comp,
                    weights_axes,
                    measure=measure,
                    radius=radius,
                    great_circle=great_circle,
                    methods=methods,
                    auto=True,
                ):
                    # Found linear weights from line geometries
                    pass
                else:
                    self._weights_linear(
                        da_key,
                        comp,
                        weights_axes,
                        measure=measure,
                        methods=methods,
                        auto=False,
                    )

            # Check for area weights specified by X and Y axes
            # separately and replace them with area weights
            xaxis = self.domain_axis("X", key=True, default=None)
            yaxis = self.domain_axis("Y", key=True, default=None)
            if xaxis != yaxis and (xaxis,) in comp and (yaxis,) in comp:
                del comp[(xaxis,)]
                del comp[(yaxis,)]
                weights_axes.discard(xaxis)
                weights_axes.discard(yaxis)
                if not self._weights_measure(
                    "area", comp, weights_axes, methods=methods
                ):
                    self._weights_area_XY(
                        comp,
                        weights_axes,
                        measure=measure,
                        radius=radius,
                        methods=methods,
                    )

        if not methods:
            if scale is not None:
                # --------------------------------------------------------
                # Scale the weights so that they are <= scale
                # --------------------------------------------------------
                for key, w in comp.items():
                    comp[key] = self._weights_scale(w, scale)

            for w in comp.values():
                mn = w.minimum()
                if mn <= 0:
                    raise ValueError(
                        "All weights must be positive. "
                        f"Got a weight of {mn}"
                    )

        if components or methods:
            # --------------------------------------------------------
            # Return a dictionary of component weights, which may be
            # empty.
            # --------------------------------------------------------
            components = {}
            for key, v in comp.items():
                key = [field_data_axes.index(axis) for axis in key]
                if not key:
                    continue

                components[tuple(key)] = v

            return components

        # Still here?
        if not comp:
            # --------------------------------------------------------
            # No component weights have been defined so return an
            # equal weights field
            # --------------------------------------------------------
            f = self._weights_field_scalar()
            if data:
                return f.data

            return f

        # ------------------------------------------------------------
        # Still here? Return a weights field which is the outer
        # product of the component weights
        # ------------------------------------------------------------
        pp = sorted(comp.items())
        waxes, wdata = pp.pop(0)
        while pp:
            a, y = pp.pop(0)
            wdata.outerproduct(y, inplace=True)
            waxes += a

        if scale is not None:
            # --------------------------------------------------------
            # Scale the weights so that they are <= scale
            # --------------------------------------------------------
            wdata = self._weights_scale(wdata, scale)

        # ------------------------------------------------------------
        # Reorder the data so that its dimensions are in the same
        # relative order as self
        # ------------------------------------------------------------
        transpose = [
            waxes.index(axis) for axis in self.get_data_axes() if axis in waxes
        ]
        wdata = wdata.transpose(transpose)
        waxes = [waxes[i] for i in transpose]

        # Set cyclicity
        for axis in self.get_data_axes():
            if axis in waxes and self.iscyclic(axis):
                wdata.cyclic(waxes.index(axis), iscyclic=True)

        if data:
            # Insert missing size one dimensions for broadcasting
            for i, axis in enumerate(self.get_data_axes()):
                if axis not in waxes:
                    waxes.insert(i, axis)
                    wdata.insert_dimension(i, inplace=True)

            return wdata

        field = self.copy()
        field.nc_del_variable(None)
        field.del_data()
        field.del_data_axes()

        not_needed_axes = set(field.domain_axes(todict=True)).difference(
            weights_axes
        )

        for key in self.cell_methods(todict=True).copy():
            field.del_construct(key)

        for key in self.field_ancillaries(todict=True).copy():
            field.del_construct(key)

        for key in field.coordinate_references(todict=True).copy():
            if field.coordinate_reference_domain_axes(key).intersection(
                not_needed_axes
            ):
                field.del_coordinate_reference(key)

        for key in field.constructs.filter_by_axis(
            *not_needed_axes, axis_mode="or", todict=True
        ):
            field.del_construct(key)

        for key in not_needed_axes:
            field.del_construct(key)

        field.set_data(wdata, axes=waxes, copy=False)
        field.clear_properties()
        field.long_name = "weights"

        return field

    # -----------
    # collapse methods


    @_inplace_enabled(default=False)
    def digitize(
        self,
        bins,
        upper=False,
        open_ends=False,
        closed_ends=None,
        return_bins=False,
        inplace=False,
    ):
        """Return the indices of the bins to which each value belongs.

        Values (including masked values) that do not belong to any bin
        result in masked values in the output field construct of indices.

        Bins defined by percentiles are easily created with the
        `percentile` method

        *Example*:
          Find the indices for bins defined by the 10th, 50th and 90th
          percentiles:

          >>> bins = f.percentile([0, 10, 50, 90, 100], squeeze=True)
          >>> i = f.digitize(bins, closed_ends=True)

        The output field construct is given a ``long_name`` property, and
        some or all of the following properties that define the bins:

        =====================  ===========================================
        Property               Description
        =====================  ===========================================
        ``bin_count``          An integer giving the number of bins

        ``bin_bounds``         A 1-d vector giving the bin bounds. The
                               first two numbers describe the lower and
                               upper boundaries of the first bin, the
                               second two numbers describe the lower and
                               upper boundaries of the second bin, and so
                               on. The presence of left-unbounded and
                               right-unbounded bins (see the *bins* and
                               *open_ends* parameters) is deduced from the
                               ``bin_count`` property. If the
                               ``bin_bounds`` vector has 2N elements then
                               the ``bin_count`` property will be N+2 if
                               there are left-unbounded and
                               right-unbounded bins, or N if no such bins
                               are present.

        ``bin_interval_type``  A string that specifies the nature of the
                               bin boundaries, i.e. if they are closed or
                               open. For example, if the lower boundary is
                               closed and the upper boundary is open
                               (which is the case when the *upper*
                               parameter is False) then
                               ``bin_interval_type`` will have the value
                               ``'lower: closed upper: open'``.

        ``bin_units``          A string giving the units of the bin
                               boundary values (e.g. ``'Kelvin'``). If the
                               *bins* parameter is a `Data` object with
                               units then these are used to set this
                               property, otherwise the field construct's
                               units are used.

        ``bin_calendar``       A string giving the calendar of reference
                               date-time units for the bin boundary values
                               (e.g. ``'noleap'``). If the units are not
                               reference date-time units this property
                               will be omitted. If the calendar is the CF
                               default calendar, then this property may be
                               omitted. If the *bins* parameter is a
                               `Data` object with a calendar then this is
                               used to set this property, otherwise the
                               field construct's calendar is used.

        ``bin_standard_name``  A string giving the standard name of the
                               bin boundaries
                               (e.g. ``'air_temperature'``). If there is
                               no standard name then this property will be
                               omitted.

        ``bin_long_name``      A string giving the long name of the bin
                               boundaries (e.g. ``'Air Temperature'``). If
                               there is no long name, or the
                               ``bin_standard_name`` is present, then this
                               property will be omitted.
        =====================  ===========================================

        Of these properties, the ``bin_count`` and ``bin_bounds`` are
        guaranteed to be output, with the others being dependent on the
        available metadata.

        .. versionadded:: 3.0.2

        .. seealso:: `bin`, `histogram`, `percentile`

        :Parameters:

            bins: array_like
                The bin boundaries. One of:

                * An integer

                  Create this many equally sized, contiguous bins spanning
                  the range of the data. I.e. the smallest bin boundary is
                  the minimum of the data and the largest bin boundary is
                  the maximum of the data. In order to guarantee that each
                  data value lies inside a bin, the *closed_ends*
                  parameter is assumed to be True.

                * A 1-d array

                  When sorted into a monotonically increasing sequence,
                  each boundary, with the exception of the two end
                  boundaries, counts as the upper boundary of one bin and
                  the lower boundary of next. If the *open_ends* parameter
                  is True then the lowest lower bin boundary also defines
                  a left-unbounded (i.e. not bounded below) bin, and the
                  largest upper bin boundary also defines a
                  right-unbounded (i.e. not bounded above) bin.

                * A 2-d array

                  The second dimension, that must have size 2, contains
                  the lower and upper boundaries of each bin. The bins to
                  not have to be contiguous, but must not overlap. If the
                  *open_ends* parameter is True then the lowest lower bin
                  boundary also defines a left-unbounded (i.e. not bounded
                  below) bin, and the largest upper bin boundary also
                  defines a right-unbounded (i.e. not bounded above) bin.

            upper: `bool`, optional
                If True then each bin includes its upper bound but not its
                lower bound. By default the opposite is applied, i.e. each
                bin includes its lower bound but not its upper bound.

            open_ends: `bool`, optional
                If True then create left-unbounded (i.e. not bounded
                below) and right-unbounded (i.e. not bounded above) bins
                from the lowest lower bin boundary and largest upper bin
                boundary respectively. By default these bins are not
                created

            closed_ends: `bool`, optional
                If True then extend the most extreme open boundary by a
                small amount so that its bin includes values that are
                equal to the unadjusted boundary value. This is done by
                multiplying it by ``1.0 - epsilon`` or ``1.0 + epsilon``,
                whichever extends the boundary in the appropriate
                direction, where ``epsilon`` is the smallest positive
                64-bit float such that ``1.0 + epsilson != 1.0``. I.e. if
                *upper* is False then the largest upper bin boundary is
                made slightly larger and if *upper* is True then the
                lowest lower bin boundary is made slightly lower.

                By default *closed_ends* is assumed to be True if *bins*
                is a scalar and False otherwise.

            return_bins: `bool`, optional
                If True then also return the bins in their 2-d form.

            {{inplace: `bool`, optional}}

        :Returns:

            `Field` or `None`, [`Data`]
                The field construct containing indices of the bins to
                which each value belongs, or `None` if the operation was
                in-place.

                If *return_bins* is True then also return the bins in
                their 2-d form.

        **Examples:**

        >>> f = cf.example_field(0)
        >>> f
        <CF Field: specific_humidity(latitude(5), longitude(8)) 0.001 1>
        >>> f.properties()
        {'Conventions': 'CF-1.7',
         'standard_name': 'specific_humidity',
         'units': '0.001 1'}
        >>> print(f.array)
        [[  7.  34.   3.  14.  18.  37.  24.  29.]
         [ 23.  36.  45.  62.  46.  73.   6.  66.]
         [110. 131. 124. 146.  87. 103.  57.  11.]
         [ 29.  59.  39.  70.  58.  72.   9.  17.]
         [  6.  36.  19.  35.  18.  37.  34.  13.]]
        >>> g = f.digitize([0, 50, 100, 150])
        >>> g
        <CF Field: long_name=Bin index to which each 'specific_humidity' value belongs(latitude(5), longitude(8))>
        >>> print(g.array)
        [[0 0 0 0 0 0 0 0]
         [0 0 0 1 0 1 0 1]
         [2 2 2 2 1 2 1 0]
         [0 1 0 1 1 1 0 0]
         [0 0 0 0 0 0 0 0]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([  0,  50,  50, 100, 100, 150]),
         'bin_count': 3,
         'bin_interval_type': 'lower: closed upper: open',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        >>> g = f.digitize([[10, 20], [40, 60], [100, 140]])
        >>> print(g.array)
        [[-- -- --  0  0 -- -- --]
         [-- --  1 --  1 -- -- --]
         [ 2  2  2 -- --  2  1  0]
         [--  1 -- --  1 -- --  0]
         [-- --  0 --  0 -- --  0]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([ 10,  20,  40,  60, 100, 140]),
         'bin_count': 3,
         'bin_interval_type': 'lower: closed upper: open',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        >>> g = f.digitize([[10, 20], [40, 60], [100, 140]], open_ends=True)
        >>> print(g.array)
        [[ 0 --  0  1  1 -- -- --]
         [-- --  2 --  2 --  0 --]
         [ 3  3  3 -- --  3  2  1]
         [--  2 -- --  2 --  0  1]
         [ 0 --  1 --  1 -- --  1]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([ 10,  20,  40,  60, 100, 140]),
         'bin_count': 5,
         'bin_interval_type': 'lower: closed upper: open',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        >>> g = f.digitize([2, 6, 45, 100], upper=True)
        >>> g
        <CF Field: long_name=Bin index to which each 'specific_humidity' value belongs(latitude(5), longitude(8))>
        >>> print(g.array)
        [[ 1  1  0  1  1  1  1  1]
         [ 1  1  1  2  2  2  0  2]
         [-- -- -- --  2 --  2  1]
         [ 1  2  1  2  2  2  1  1]
         [ 0  1  1  1  1  1  1  1]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([  2,   6,   6,  45,  45, 100]),
         'bin_count': 3,
         'bin_interval_type': 'lower: open upper: closed',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        >>> g, bins = f.digitize(10, return_bins=True)
        >>> bins
        <CF Data(10, 2): [[3.0, ..., 146.00000000000003]] 0.001 1>
        >>> g, bins = f.digitize(10, upper=True, return_bins=True)
        <CF Data(10, 2): [[2.999999999999999, ..., 146.0]] 0.001 1>
        >>> print(g.array)
        [[0 2 0 0 1 2 1 1]
         [1 2 2 4 3 4 0 4]
         [7 8 8 9 5 6 3 0]
         [1 3 2 4 3 4 0 0]
         [0 2 1 2 1 2 2 0]]

        >>> f[1, [2, 5]] = cf.masked
        >>> print(f.array)
        [[  7.  34.   3.  14.  18.  37.  24.  29.]
         [ 23.  36.   --  62.  46.   --   6.  66.]
         [110. 131. 124. 146.  87. 103.  57.  11.]
         [ 29.  59.  39.  70.  58.  72.   9.  17.]
         [  6.  36.  19.  35.  18.  37.  34.  13.]]
        >>> g = f.digitize(10)
        >>> print(g.array)
        [[ 0  2  0  0  1  2  1  1]
         [ 1  2 --  4  3 --  0  4]
         [ 7  8  8  9  5  6  3  0]
         [ 1  3  2  4  3  4  0  0]
         [ 0  2  1  2  1  2  2  0]]
        >>> g.properties()
        {'Conventions': 'CF-1.7',
         'long_name': "Bin index to which each 'specific_humidity' value belongs",
         'bin_bounds': array([  3. ,  17.3,  17.3,  31.6,  31.6,  45.9,  45.9,  60.2,
                60.2,  74.5,  74.5,  88.8,  88.8, 103.1, 103.1, 117.4, 117.4, 131.7,
                131.7, 146. ]),
         'bin_count': 10,
         'bin_interval_type': 'lower: closed upper: open',
         'bin_standard_name': 'specific_humidity',
         'bin_units': '0.001 1'}

        """
        f = _inplace_enabled_define_and_cleanup(self)

        new_data, bins = self.data.digitize(
            bins,
            upper=upper,
            open_ends=open_ends,
            closed_ends=closed_ends,
            return_bins=True,
        )
        units = new_data.Units

        f.set_data(new_data, set_axes=False, copy=False)
        f.override_units(units, inplace=True)

        # ------------------------------------------------------------
        # Set properties
        # ------------------------------------------------------------
        f.set_property(
            "long_name",
            f"Bin index to which each {self.identity()!r} value belongs",
            copy=False,
        )

        f.set_property("bin_bounds", bins.array.flatten(), copy=False)

        bin_count = bins.shape[0]
        if open_ends:
            bin_count += 2

        f.set_property("bin_count", bin_count, copy=False)

        if upper:
            bin_interval_type = "lower: open upper: closed"
        else:
            bin_interval_type = "lower: closed upper: open"

        f.set_property("bin_interval_type", bin_interval_type, copy=False)

        standard_name = f.del_property("standard_name", None)
        if standard_name is not None:
            f.set_property("bin_standard_name", standard_name, copy=False)
        else:
            long_name = f.del_property("long_name", None)
            if long_name is not None:
                f.set_property("bin_long_name", long_name, copy=False)

        bin_units = bins.Units
        units = getattr(bin_units, "units", None)
        if units is not None:
            f.set_property("bin_units", units, copy=False)

        calendar = getattr(bin_units, "calendar", None)
        if calendar is not None:
            f.set_property("bin_calendar", calendar, copy=False)

        if return_bins:
            return f, bins

        return f

    @_manage_log_level_via_verbosity
    def bin(
        self,
        method,
        digitized,
        weights=None,
        measure=False,
        scale=None,
        mtol=1,
        ddof=1,
        radius="earth",
        great_circle=False,
        return_indices=False,
        verbose=None,
    ):
        """Collapse the data values that lie in N-dimensional bins.

        The data values of the field construct are binned according to how
        they correspond to the N-dimensional histogram bins of another set
        of variables (see `cf.histogram` for details), and each bin of
        values is collapsed with one of the collapse methods allowed by
        the *method* parameter.

        The number of dimensions of the output binned data is equal to the
        number of field constructs provided by the *digitized*
        argument. Each such field construct defines a sequence of bins and
        provides indices to the bins that each value of another field
        construct belongs. There is no upper limit to the number of
        dimensions of the output binned data.

        The output bins are defined by the exterior product of the
        one-dimensional bins of each digitized field construct. For
        example, if only one digitized field construct is provided then
        the output bins simply comprise its one-dimensional bins; if there
        are two digitized field constructs then the output bins comprise
        the two-dimensional matrix formed by all possible combinations of
        the two sets of one-dimensional bins; etc.

        An output value for a bin is formed by collapsing (using the
        method given by the *method* parameter) the elements of the data
        for which the corresponding locations in the digitized field
        constructs, taken together, index that bin. Note that it may be
        the case that not all output bins are indexed by the digitized
        field constructs, and for these bins missing data is returned.

        The returned field construct will have a domain axis construct for
        each dimension of the output bins, with a corresponding dimension
        coordinate construct that defines the bin boundaries.

        .. versionadded:: 3.0.2

        .. seealso:: `collapse`, `digitize`, `weights`, `cf.histogram`

        :Parameters:

            method: `str`
                The collapse method used to combine values that map to
                each cell of the output field construct. The following
                methods are available (see
                https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
                for precise definitions):

                ============================  ============================  ========
                *method*                      Description                   Weighted
                ============================  ============================  ========
                ``'maximum'``                 The maximum of the values.    Never

                ``'minimum'``                 The minimum of the values.    Never

                ``'maximum_absolute_value'``  The maximum of the absolute   Never
                                              values.

                ``'minimum_absolute_value'``  The minimum of the absolute   Never
                                              values.

                ``'mid_range'``               The average of the maximum    Never
                                              and the minimum of the
                                              values.

                ``'range'``                   The absolute difference       Never
                                              between the maximum and the
                                              minimum of the values.

                ``'median'``                  The median of the values.     Never

                ``'sum'``                     The sum of the values.        Never

                ``'sum_of_squares'``          The sum of the squares of     Never
                                              values.

                ``'sample_size'``             The sample size, i.e. the     Never
                                              number of non-missing
                                              values.

                ``'sum_of_weights'``          The sum of weights, as        Never
                                              would be used for other
                                              calculations.

                ``'sum_of_weights2'``         The sum of squares of         Never
                                              weights, as would be used
                                              for other calculations.

                ``'mean'``                    The weighted or unweighted    May be
                                              mean of the values.

                ``'mean_absolute_value'``     The mean of the absolute      May be
                                              values.

                ``'mean_of_upper_decile'``    The mean of the upper group   May be
                                              of data values defined by
                                              the upper tenth of their
                                              distribution.

                ``'variance'``                The weighted or unweighted    May be
                                              variance of the values, with
                                              a given number of degrees of
                                              freedom.

                ``'standard_deviation'``      The square root of the        May be
                                              weighted or unweighted
                                              variance.

                ``'root_mean_square'``        The square root of the        May be
                                              weighted or unweighted mean
                                              of the squares of the
                                              values.

                ``'integral'``                The integral of values.       Always
                ============================  ============================  ========

                * Collapse methods that are "Never" weighted ignore the
                  *weights* parameter, even if it is set.

                * Collapse methods that "May be" weighted will only be
                  weighted if the *weights* parameter is set.

                * Collapse methods that are "Always" weighted require the
                  *weights* parameter to be set.

            digitized: (sequence of) `Field`
                One or more field constructs that contain digitized data
                with corresponding metadata, as would be output by
                `cf.Field.digitize`. Each field construct contains indices
                to the one-dimensional bins to which each value of an
                original field construct belongs; and there must be
                ``bin_count`` and ``bin_bounds`` properties as defined by
                the `digitize` method (and any of the extra properties
                defined by that method are also recommended).

                The bins defined by the ``bin_count`` and ``bin_bounds``
                properties are used to create a dimension coordinate
                construct for the output field construct.

                Each digitized field construct must be transformable so
                that it is broadcastable to the input field construct's
                data. This is done by using the metadata constructs of the
                to create a mapping of physically compatible dimensions
                between the fields, and then manipulating the dimensions
                of the digitized field construct's data to ensure that
                broadcasting can occur.

            weights: optional
                Specify the weights for the collapse calculations. The
                weights are those that would be returned by this call of
                the field construct's `~cf.Field.weights` method:
                ``f.weights(weights, measure=measure, scale=scale,
                radius=radius, great_circle=great_circle,
                components=True)``. See the *measure, *scale*, *radius*
                and *great_circle* parameters and `cf.Field.weights` for
                details.

                .. note:: By default *weights* is `None`, resulting in
                          **unweighted calculations**.

                .. note:: Setting *weights* to `True` is generally a good
                          way to ensure that all collapses are
                          appropriately weighted according to the field
                          construct's metadata. In this case, if it is not
                          possible to create weights for any axis then an
                          exception will be raised.

                          However, care needs to be taken if *weights* is
                          `True` when cell volume weights are desired. The
                          volume weights will be taken from a "volume"
                          cell measure construct if one exists, otherwise
                          the cell volumes will be calculated as being
                          proportional to the sizes of one-dimensional
                          vertical coordinate cells. In the latter case
                          **if the vertical dimension coordinates do not
                          define the actual height or depth thickness of
                          every cell in the domain then the weights will
                          be incorrect**.

                If *weights* is the boolean `True` then weights are
                calculated for all of the domain axis constructs.

                *Parameter example:*
                  To specify weights based on the field construct's
                  metadata for all axes use ``weights=True``.

                *Parameter example:*
                  To specify weights based on cell areas, leaving all
                  other axes unweighted, use ``weights='area'``.

                *Parameter example:*
                  To specify weights based on cell areas and linearly in
                  time, leaving all other axes unweighted, you could set
                  ``weights=('area', 'T')``.

            measure: `bool`, optional
                Create weights, as defined by the *weights* parameter,
                which are cell measures, i.e. which describe actual cell
                sizes (e.g. cell areas) with appropriate units
                (e.g. metres squared). By default the weights are scaled
                to lie between 0 and 1 and have arbitrary units (see the
                *scale* parameter).

                Cell measures can be created for any combination of
                axes. For example, cell measures for a time axis are the
                time span for each cell with canonical units of seconds;
                cell measures for the combination of four axes
                representing time and three dimensional space could have
                canonical units of metres cubed seconds.

                When collapsing with the ``'integral'`` method, *measure*
                must be True, and the units of the weights are
                incorporated into the units of the returned field
                construct.

                .. note:: Specifying cell volume weights via
                          ``weights=['X', 'Y', 'Z']`` or
                          ``weights=['area', 'Z']`` (or other equivalents)
                          will produce **an incorrect result if the
                          vertical dimension coordinates do not define the
                          actual height or depth thickness of every cell
                          in the domain**. In this case,
                          ``weights='volume'`` should be used instead,
                          which requires the field construct to have a
                          "volume" cell measure construct.

                          If ``weights=True`` then care also needs to be
                          taken, as a "volume" cell measure construct will
                          be used if present, otherwise the cell volumes
                          will be calculated using the size of the
                          vertical coordinate cells.

            scale: number, optional
                If set to a positive number then scale the weights, as
                defined by the *weights* parameter, so that they are less
                than or equal to that number. By default the weights are
                scaled to lie between 0 and 1 (i.e.  *scale* is 1).

                *Parameter example:*
                  To scale all weights so that they lie between 0 and 0.5:
                  ``scale=0.5``.

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

            ddof: number, optional
                The delta degrees of freedom in the calculation of a
                standard deviation or variance. The number of degrees of
                freedom used in the calculation is (N-*ddof*) where N
                represents the number of non-missing elements contributing
                to the calculation. By default *ddof* is 1, meaning the
                standard deviation and variance of the population is
                estimated according to the usual formula with (N-1) in the
                denominator to avoid the bias caused by the use of the
                sample mean (Bessel's correction).

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
                If True then allow, if required, the derivation of i) area
                weights from polygon geometry cells by assuming that each
                cell part is a spherical polygon composed of great circle
                segments; and ii) and the derivation of line-length
                weights from line geometry cells by assuming that each
                line part is composed of great circle segments.

                .. versionadded:: 3.2.0

            {{verbose: `int` or `str` or `None`, optional}}

        :Returns:

            `Field`
                The field construct containing the binned values.

        **Examples:**

        Find the range of values that lie in each bin:

        >>> print(q)
        Field: specific_humidity (ncvar%q)
        ----------------------------------
        Data            : specific_humidity(latitude(5), longitude(8)) 0.001 1
        Cell methods    : area: mean
        Dimension coords: latitude(5) = [-75.0, ..., 75.0] degrees_north
                        : longitude(8) = [22.5, ..., 337.5] degrees_east
                        : time(1) = [2019-01-01 00:00:00]
        >>> print(q.array)
        [[  7.  34.   3.  14.  18.  37.  24.  29.]
         [ 23.  36.  45.  62.  46.  73.   6.  66.]
         [110. 131. 124. 146.  87. 103.  57.  11.]
         [ 29.  59.  39.  70.  58.  72.   9.  17.]
         [  6.  36.  19.  35.  18.  37.  34.  13.]]
        >>> indices = q.digitize(10)
        >>> b = q.bin('range', digitized=indices)
        >>> print(b)
        Field: specific_humidity
        ------------------------
        Data            : specific_humidity(specific_humidity(10)) 0.001 1
        Cell methods    : latitude: longitude: range
        Dimension coords: specific_humidity(10) = [10.15, ..., 138.85000000000002] 0.001 1
        >>> print(b.array)
        [14. 11. 11. 13. 11.  0.  0.  0.  7.  0.]

        Find various metrics describing how
        ``tendency_of_sea_water_potential_temperature_expressed_as_heat_content``
        data varies with ``sea_water_potential_temperature`` and
        ``sea_water_salinity``:

        >>> t
        Field: sea_water_potential_temperature (ncvar%sea_water_potential_temperature)
        ------------------------------------------------------------------------------
        Data            : sea_water_potential_temperature(time(1), depth(1), latitude(5), longitude(8)) K
        Cell methods    : area: mean time(1): mean
        Dimension coords: time(1) = [2290-06-01 00:00:00] 360_day
                        : depth(1) = [3961.89990234375] m
                        : latitude(5) = [-1.875, ..., 3.125] degrees_north
                        : longitude(8) = [75.0, ..., 83.75] degrees_east
        Auxiliary coords: model_level_number(depth(1)) = [18]
        >>> s
        Field: sea_water_salinity (ncvar%sea_water_salinity)
        ----------------------------------------------------
        Data            : sea_water_salinity(time(1), depth(1), latitude(5), longitude(8)) psu
        Cell methods    : area: mean time(1): mean
        Dimension coords: time(1) = [2290-06-01 00:00:00] 360_day
                        : depth(1) = [3961.89990234375] m
                        : latitude(5) = [-1.875, ..., 3.125] degrees_north
                        : longitude(8) = [75.0, ..., 83.75] degrees_east
        Auxiliary coords: model_level_number(depth(1)) = [18]
        >>> x
        Field: tendency_of_sea_water_potential_temperature_expressed_as_heat_content (ncvar%tend)
        -----------------------------------------------------------------------------------------
        Data            : tendency_of_sea_water_potential_temperature_expressed_as_heat_content(time(1), depth(1), latitude(5), longitude(8)) W m-2
        Cell methods    : area: mean time(1): mean
        Dimension coords: time(1) = [2290-06-01 00:00:00] 360_day
                        : depth(1) = [3961.89990234375] m
                        : latitude(5) = [-1.875, ..., 3.125] degrees_north
                        : longitude(8) = [75.0, ..., 83.75] degrees_east
        Auxiliary coords: model_level_number(depth(1)) = [18]
        >>> print(x.array)
        [[[[-209.72  340.86   94.75  154.21   38.54 -262.75  158.22  154.58]
           [ 311.67  245.91 -168.16   47.61 -219.66 -270.33  226.1    52.0 ]
           [     -- -112.34  271.67  189.22    9.92  232.39  221.17  206.0 ]
           [     --      --  -92.31 -285.57  161.55  195.89 -258.29    8.35]
           [     --      --   -7.82 -299.79  342.32 -169.38  254.5   -75.4 ]]]]

        >>> t_indices = t.digitize(6)
        >>> s_indices = s.digitize(4)

        >>> n = x.bin('sample_size', [t_indices, s_indices])
        >>> print(n)
        Field: number_of_observations
        -----------------------------
        Data            : number_of_observations(sea_water_salinity(4), sea_water_potential_temperature(6)) 1
        Cell methods    : latitude: longitude: point
        Dimension coords: sea_water_salinity(4) = [6.3054151982069016, ..., 39.09366758167744] psu
                        : sea_water_potential_temperature(6) = [278.1569468180338, ..., 303.18466695149743] K
        >>> print(n.array)
        [[ 1  2 2  2 --  2]
         [ 2  1 3  3  3  2]
         [-- -- 3 --  1 --]
         [ 1 -- 1  3  2  1]]

        >>> m = x.bin('mean', [t_indices, s_indices], weights=['X', 'Y', 'Z', 'T'])
        >>> print(m)
        Field: tendency_of_sea_water_potential_temperature_expressed_as_heat_content
        ----------------------------------------------------------------------------
        Data            : tendency_of_sea_water_potential_temperature_expressed_as_heat_content(sea_water_salinity(4), sea_water_potential_temperature(6)) W m-2
        Cell methods    : latitude: longitude: mean
        Dimension coords: sea_water_salinity(4) = [6.3054151982069016, ..., 39.09366758167744] psu
                        : sea_water_potential_temperature(6) = [278.1569468180338, ..., 303.18466695149743] K
        >>> print(m.array)
        [[ 189.22 131.36    6.75 -41.61     --  100.04]
         [-116.73 232.38   -4.82 180.47 134.25 -189.55]
         [     --     --  180.69     --  47.61      --]
         [158.22      -- -262.75  64.12 -51.83 -219.66]]

        >>> i = x.bin(
        ...         'integral', [t_indices, s_indices],
        ...         weights=['X', 'Y', 'Z', 'T'], measure=True
        ...     )
        >>> print(i)
        Field: long_name=integral of tendency_of_sea_water_potential_temperature_expressed_as_heat_content
        --------------------------------------------------------------------------------------------------
        Data            : long_name=integral of tendency_of_sea_water_potential_temperature_expressed_as_heat_content(sea_water_salinity(4), sea_water_potential_temperature(6)) 86400 m3.kg.s-2
        Cell methods    : latitude: longitude: sum
        Dimension coords: sea_water_salinity(4) = [6.3054151982069016, ..., 39.09366758167744] psu
                        : sea_water_potential_temperature(6) = [278.1569468180338, ..., 303.18466695149743] K
        >>> print(i.array)
        [[ 3655558758400.0 5070927691776.0   260864491520.0 -1605439586304.0               --  3863717609472.0]
         [-4509735059456.0 4489564127232.0  -280126521344.0 10454746267648.0  7777254113280.0 -7317268463616.0]
         [              --              -- 10470463373312.0               --   919782031360.0               --]
         [ 3055211773952.0              -- -5073676009472.0  3715958833152.0 -2000787079168.0 -4243632160768.0]]

        >>> w = x.bin('sum_of_weights', [t_indices, s_indices], weights=['X', 'Y', 'Z', 'T'], measure=True)
        Field: long_name=sum_of_weights of tendency_of_sea_water_potential_temperature_expressed_as_heat_content
        --------------------------------------------------------------------------------------------------------
        Data            : long_name=sum_of_weights of tendency_of_sea_water_potential_temperature_expressed_as_heat_content(sea_water_salinity(4), sea_water_potential_temperature(6)) 86400 m3.s
        Cell methods    : latitude: longitude: sum
        Dimension coords: sea_water_salinity(4) = [7.789749830961227, ..., 36.9842486679554] psu
                        : sea_water_potential_temperature(6) = [274.50717671712243, ..., 302.0188242594401] K
        >>> print(w.array)
        [[19319093248.0 38601412608.0 38628990976.0 38583025664.0            --  38619795456.0]
         [38628990976.0 19319093248.0 57957281792.0 57929699328.0 57929695232.0  38601412608.0]
         [         --              -- 57948086272.0            -- 19319093248.0             --]
         [19309897728.0            -- 19309897728.0 57948086272.0 38601412608.0  19319093248.0]]

        Demonstrate that the integral divided by the sum of the cell
        measures is equal to the mean:

        >>> print(i/w)
        Field:
        -------
        Data            : (sea_water_salinity(4), sea_water_potential_temperature(6)) kg.s-3
        Cell methods    : latitude: longitude: sum
        Dimension coords: sea_water_salinity(4) = [7.789749830961227, ..., 36.9842486679554] psu
                        : sea_water_potential_temperature(6) = [274.50717671712243, ..., 302.0188242594401] K
        >>> (i/w == m).all()
        True

        """
        logger.info(f"    Method: {method}")  # pragma: no cover

        if method == "integral":
            if weights is None:
                raise ValueError(
                    "Must specify weights for 'integral' calculations."
                )

            if not measure:
                raise ValueError(
                    "Must set measure=True for 'integral' calculations."
                )

            if scale is not None:
                raise ValueError(
                    "Can't set scale for 'integral' calculations."
                )

        axes = []
        bin_indices = []
        shape = []
        dims = []
        names = []

        # Initialize the output binned field
        out = type(self)(properties=self.properties())

        # Sort out its identity
        if method == "sample_size":
            out.standard_name = "number_of_observations"
        elif method in (
            "integral",
            "sum_of_squares",
            "sum_of_weights",
            "sum_of_weights2",
        ):
            out.del_property("standard_name", None)

        long_name = self.get_property("long_name", None)
        if long_name is None:
            out.long_name = (
                method + " of " + self.get_property("standard_name", "")
            )
        else:
            out.long_name = method + " of " + long_name

        # ------------------------------------------------------------
        # Create domain axes and dimension coordinates for the output
        # binned field
        # ------------------------------------------------------------
        if isinstance(digitized, self.__class__):
            digitized = (digitized,)

        for f in digitized[::-1]:
            f = self._conform_for_data_broadcasting(f)

            if not self._is_broadcastable(f.shape):
                raise ValueError(
                    "Conformed digitized field {!r} construct must have "
                    "shape broadcastable to {}.".format(f, self.shape)
                )

            bin_bounds = f.get_property("bin_bounds", None)
            bin_count = f.get_property("bin_count", None)
            bin_interval_type = f.get_property("bin_interval_type", None)
            bin_units = f.get_property("bin_units", None)
            bin_calendar = f.get_property("bin_calendar", None)
            bin_standard_name = f.get_property("bin_standard_name", None)
            bin_long_name = f.get_property("bin_long_name", None)

            if bin_count is None:
                raise ValueError(
                    "Digitized field construct {!r} must have a 'bin_count' "
                    "property.".format(f)
                )

            if bin_bounds is None:
                raise ValueError(
                    "Digitized field construct {!r} must have a "
                    "'bin_bounds' property.".format(f)
                )

            if bin_count != len(bin_bounds) / 2:
                raise ValueError(
                    "Digitized field construct {!r} bin_count must equal "
                    "len(bin_bounds)/2. Got bin_count={}, "
                    "len(bin_bounds)/2={}".format(
                        f, bin_count, len(bin_bounds) / 2
                    )
                )

            # Create dimension coordinate for bins
            dim = DimensionCoordinate()
            if bin_standard_name is not None:
                dim.standard_name = bin_standard_name
            elif bin_long_name is not None:
                dim.long_name = bin_long_name

            if bin_interval_type is not None:
                dim.set_property(
                    "bin_interval_type", bin_interval_type, copy=False
                )

            # Create units for the bins
            units = Units(bin_units, bin_calendar)

            data = Data(
                0.5 * (bin_bounds[1::2] + bin_bounds[0::2]), units=units
            )
            dim.set_data(data=data, copy=False)

            bounds_data = Data(
                numpy_reshape(bin_bounds, (bin_count, 2)), units=units
            )
            dim.set_bounds(self._Bounds(data=bounds_data))

            logger.info(
                f"                    bins     : {dim.identity()} {bounds_data!r}"  # DCH
            )  # pragma: no cover

            # Set domain axis and dimension coordinate for bins
            axis = out.set_construct(self._DomainAxis(dim.size))
            out.set_construct(dim, axes=[axis], copy=False)

            axes.append(axis)
            bin_indices.append(f.data)
            shape.append(dim.size)
            dims.append(dim)
            names.append(dim.identity())

        # ------------------------------------------------------------
        # Initialize the ouput data as a totally masked array
        # ------------------------------------------------------------
        if method == "sample_size":
            dtype = int
        else:
            dtype = self.dtype

        data = Data.masked_all(shape=tuple(shape), dtype=dtype, units=None)
        out.set_data(data, axes=axes, copy=False)
        out.hardmask = False

        c = self.copy()

        # ------------------------------------------------------------
        # Parse the weights
        # ------------------------------------------------------------
        if weights is not None:
            if not measure and scale is None:
                scale = 1.0

            weights = self.weights(
                weights,
                scale=scale,
                measure=measure,
                radius=radius,
                great_circle=great_circle,
                components=True,
            )

        # ------------------------------------------------------------
        # Find the unique multi-dimensional bin indices (TODO: can I
        # LAMA this?)
        # ------------------------------------------------------------
        y = numpy_empty((len(bin_indices), bin_indices[0].size), dtype=int)
        for i, f in enumerate(bin_indices):
            y[i, :] = f.array.flatten()

        unique_indices = numpy_unique(y, axis=1)
        del f
        del y

        # DCH
        logger.info(f"    Weights: {weights}")  # pragma: no cover
        logger.info(
            f"    Number of indexed ({', '.join(names)}) bins: "
            f"{unique_indices.shape[1]}"
        )  # pragma: no cover
        logger.info(
            f"    ({', '.join(names)}) bin indices:"  # DCH
        )  # pragma: no cover

        # Loop round unique collections of bin indices
        for i in zip(*unique_indices):
            logger.info(f"{' '.join(str(i))}")

            b = bin_indices[0] == i[0]
            for a, n in zip(bin_indices[1:], i[1:]):
                b &= a == n

            b.filled(False, inplace=True)

            c.set_data(
                self.data.where(b, None, cf_masked), set_axes=False, copy=False
            )

            result = c.collapse(
                method=method, weights=weights, measure=measure
            ).data
            out.data[i] = result.datum()

        # Set correct units (note: takes them from the last processed
        # "result" variable in the above loop)
        out.override_units(result.Units, inplace=True)
        out.hardmask = True

        # ------------------------------------------------------------
        # Create a cell method (if possible)
        # ------------------------------------------------------------
        standard_names = []
        domain_axes = self.domain_axes(filter_by_size=(ge(2),), todict=True)

        for da_key in domain_axes:
            dim = self.dimension_coordinate(
                filter_by_axis=(da_key,), default=None
            )
            if dim is None:
                continue

            standard_name = dim.get_property("standard_name", None)
            if standard_name is None:
                continue

            standard_names.append(standard_name)

        if len(standard_names) == len(domain_axes):
            cell_method = CellMethod(
                axes=sorted(standard_names),
                method=_collapse_cell_methods[method],
            )
            out.set_construct(cell_method, copy=False)

        return out


    @_deprecated_kwarg_check("i")
    @_manage_log_level_via_verbosity
    def collapse(
        self,
        method,
        axes=None,
        squeeze=False,
        mtol=1,
        weights=None,
        ddof=1,
        a=None,
        inplace=False,
        group=None,
        regroup=False,
        within_days=None,
        within_years=None,
        over_days=None,
        over_years=None,
        coordinate=None,
        group_by=None,
        group_span=None,
        group_contiguous=1,
        measure=False,
        scale=None,
        radius="earth",
        great_circle=False,
        verbose=None,
        _create_zero_size_cell_bounds=False,
        _update_cell_methods=True,
        i=False,
        _debug=False,
        **kwargs,
    ):
        """Collapse axes of the field.

        Collapsing one or more dimensions reduces their size and replaces
        the data along those axes with representative statistical
        values. The result is a new field construct with consistent
        metadata for the collapsed values.

        By default all axes with size greater than 1 are collapsed
        completely (i.e. to size 1) with a given collapse method.

        *Example:*
          Find the minimum of the entire data:

          >>> b = a.collapse('minimum')

        The collapse can also be applied to any subset of the field
        construct's dimensions. In this case, the domain axis and
        coordinate constructs for the non-collapsed dimensions remain the
        same. This is implemented either with the axes keyword, or with a
        CF-netCDF cell methods-like syntax for describing both the
        collapse dimensions and the collapse method in a single
        string. The latter syntax uses construct identities instead of
        netCDF dimension names to identify the collapse axes.

        Statistics may be created to represent variation over one
        dimension or a combination of dimensions.

        *Example:*
           Two equivalent techniques for creating a field construct of
           temporal maxima at each horizontal location:

           >>> b = a.collapse('maximum', axes='T')
           >>> b = a.collapse('T: maximum')

        *Example:*
          Find the horizontal maximum, with two equivalent techniques.

          >>> b = a.collapse('maximum', axes=['X', 'Y'])
          >>> b = a.collapse('X: Y: maximum')

        Variation over horizontal area may also be specified by the
        special identity 'area'. This may be used for any horizontal
        coordinate reference system.

        *Example:*
          Find the horizontal maximum using the special identity 'area':

          >>> b = a.collapse('area: maximum')


        **Collapse methods**

        The following collapse methods are available (see
        https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
        for precise definitions):

        ============================  ============================
        Method                        Description
        ============================  ============================
        ``'maximum'``                 The maximum of the values.

        ``'minimum'``                 The minimum of the values.

        ``'maximum_absolute_value'``  The maximum of the absolute
                                      values.

        ``'minimum_absolute_value'``  The minimum of the absolute
                                      values.

        ``'mid_range'``               The average of the maximum
                                      and the minimum of the
                                      values.

        ``'median'``                  The median of the values.

        ``'range'``                   The absolute difference
                                      between the maximum and the
                                      minimum of the values.

        ``'sum'``                     The sum of the values.

        ``'sum_of_squares'``          The sum of the squares of
                                      values.

        ``'sample_size'``             The sample size, i.e. the
                                      number of non-missing
                                      values.

        ``'sum_of_weights'``          The sum of weights, as
                                      would be used for other
                                      calculations.

        ``'sum_of_weights2'``         The sum of squares of
                                      weights, as would be used
                                      for other calculations.

        ``'mean'``                    The weighted or unweighted
                                      mean of the values.

        ``'mean_absolute_value'``     The mean of the absolute
                                      values.

        ``'mean_of_upper_decile'``    The mean of the upper group
                                      of data values defined by
                                      the upper tenth of their
                                      distribution.

        ``'variance'``                The weighted or unweighted
                                      variance of the values, with
                                      a given number of degrees of
                                      freedom.

        ``'standard_deviation'``      The weighted or unweighted
                                      standard deviation of the
                                      values, with a given number
                                      of degrees of freedom.

        ``'root_mean_square'``        The square root of the
                                      weighted or unweighted mean
                                      of the squares of the
                                      values.

        ``'integral'``                The integral of values.
        ============================  ============================


        **Data type and missing data**

        In all collapses, missing data array elements are accounted for in
        the calculation.

        Any collapse method that involves a calculation (such as
        calculating a mean), as opposed to just selecting a value (such as
        finding a maximum), will return a field containing double
        precision floating point numbers. If this is not desired then the
        data type can be reset after the collapse with the `dtype`
        attribute of the field construct.


        **Collapse weights**

        The calculations of means, standard deviations and variances are,
        by default, **not weighted**. For weights to be incorporated in
        the collapse, the axes to be weighted must be identified with the
        *weights* keyword.

        Weights are either derived from the field construct's metadata
        (such as cell sizes), or may be provided explicitly in the form of
        other field constructs containing data of weights values. In
        either case, the weights actually used are those derived by the
        `weights` method of the field construct with the same weights
        keyword value. Collapsed axes that are not identified by the
        *weights* keyword are unweighted during the collapse operation.

        *Example:*
          Create a weighted time average:

          >>> b = a.collapse('T: mean', weights=True)

        *Example:*
          Calculate the mean over the time and latitude axes, with
          weights only applied to the latitude axis:

          >>> b = a.collapse('T: Y: mean', weights='Y')

        *Example*
          Alternative syntax for specifying area weights:

          >>> b = a.collapse('area: mean', weights=True)

        An alternative technique for specifying weights is to set the
        *weights* keyword to the output of a call to the `weights` method.

        *Example*
          Alternative syntax for specifying weights:

          >>> b = a.collapse('area: mean', weights=a.weights('area'))

        **Multiple collapses**

        Multiple collapses normally require multiple calls to `collapse`:
        one on the original field construct and then one on each interim
        field construct.

        *Example:*
          Calculate the temporal maximum of the weighted areal means
          using two independent calls:

          >>> b = a.collapse('area: mean', weights=True).collapse('T: maximum')

        If preferred, multiple collapses may be carried out in a single
        call by using the CF-netCDF cell methods-like syntax (note that
        the colon (:) is only used after the construct identity that
        specifies each axis, and a space delimits the separate collapses).

        *Example:*
          Calculate the temporal maximum of the weighted areal means in
          a single call, using the cf-netCDF cell methods-like syntax:

          >>> b =a.collapse('area: mean T: maximum', weights=True)


        **Grouped collapses**

        A grouped collapse is one for which as axis is not collapsed
        completely to size 1. Instead the collapse axis is partitioned
        into non-overlapping groups and each group is collapsed to size
        1. The resulting axis will generally have more than one
        element. For example, creating 12 annual means from a timeseries
        of 120 months would be a grouped collapse.

        Selected statistics for overlapping groups can be calculated with
        the `moving_window` method.

        The *group* keyword defines the size of the groups. Groups can be
        defined in a variety of ways, including with `Query`,
        `TimeDuration` and `Data` instances.

        An element of the collapse axis can not be a member of more than
        one group, and may be a member of no groups. Elements that are not
        selected by the *group* keyword are excluded from the result.

        *Example:*
          Create annual maxima from a time series, defining a year to
          start on 1st December.

          >>> b = a.collapse('T: maximum', group=cf.Y(month=12))

        *Example:*
          Find the maximum of each group of 6 elements along an axis.

          >>> b = a.collapse('T: maximum', group=6)

        *Example:*
          Create December, January, February maxima from a time series.

          >>> b = a.collapse('T: maximum', group=cf.djf())

        *Example:*
          Create maxima for each 3-month season of a timeseries (DJF, MAM,
          JJA, SON).

          >>> b = a.collapse('T: maximum', group=cf.seasons())

        *Example:*
          Calculate zonal means for the western and eastern hemispheres.

          >>> b = a.collapse('X: mean', group=cf.Data(180, 'degrees'))

        Groups can be further described with the *group_span* parameter
        (to include groups whose actual span is not equal to a given
        value) and the *group_contiguous* parameter (to include
        non-contiguous groups, or any contiguous group containing
        overlapping cells).


        **Climatological statistics**

        Climatological statistics may be derived from corresponding
        portions of the annual cycle in a set of years (e.g. the average
        January temperatures in the climatology of 1961-1990, where the
        values are derived by averaging the 30 Januarys from the separate
        years); or from corresponding portions of the diurnal cycle in a
        set of days (e.g. the average temperatures for each hour in the
        day for May 1997). A diurnal climatology may also be combined with
        a multiannual climatology (e.g. the minimum temperature for each
        hour of the average day in May from a 1961-1990 climatology).

        Calculation requires two or three collapses, depending on the
        quantity being created, all of which are grouped collapses. Each
        collapse method needs to indicate its climatological nature with
        one of the following qualifiers,

        ================  =======================
        Method qualifier  Associated keyword
        ================  =======================
        ``within years``  *within_years*
        ``within days``   *within_days*
        ``over years``    *over_years* (optional)
        ``over days``     *over_days* (optional)
        ================  =======================

        and the associated keyword specifies how the method is to be
        applied.

        *Example*
          Calculate the multiannual average of the seasonal means:

          >>> b = a.collapse('T: mean within years T: mean over years',
          ...                within_years=cf.seasons(), weights=True)

        *Example:*
          Calculate the multiannual variance of the seasonal
          minima. Note that the units of the result have been changed
          from 'K' to 'K2':

          >>> b = a.collapse('T: minimum within years T: variance over years',
          ...                within_years=cf.seasons(), weights=True)

        When collapsing over years, it is assumed by default that each
        portion of the annual cycle is collapsed over all years that are
        present. This is the case in the above two examples. It is
        possible, however, to restrict the years to be included, or group
        them into chunks, with the *over_years* keyword.

        *Example:*
          Calculate the multiannual average of the seasonal means in 5
          year chunks:

          >>> b = a.collapse(
          ...     'T: mean within years T: mean over years', weights=True,
          ...     within_years=cf.seasons(), over_years=cf.Y(5)
          ... )

        *Example:*
          Calculate the multiannual average of the seasonal means,
          restricting the years from 1963 to 1968:

          >>> b = a.collapse(
          ...     'T: mean within years T: mean over years', weights=True,
          ...     within_years=cf.seasons(),
          ...     over_years=cf.year(cf.wi(1963, 1968))
          ... )

        Similarly for collapses over days, it is assumed by default that
        each portion of the diurnal cycle is collapsed over all days that
        are present, But it is possible to restrict the days to be
        included, or group them into chunks, with the *over_days* keyword.

        The calculation can be done with multiple collapse calls, which
        can be useful if the interim stages are needed independently, but
        be aware that the interim field constructs will have
        non-CF-compliant cell method constructs.

        *Example:*
          Calculate the multiannual maximum of the seasonal standard
          deviations with two separate collapse calls:

          >>> b = a.collapse('T: standard_deviation within years',
          ...                within_years=cf.seasons(), weights=True)


        .. versionadded:: 1.0

        .. seealso:: `bin`, `cell_area`, `convolution_filter`,
                     `moving_window`, `radius`, `weights`

        :Parameters:

            method: `str`
                Define the collapse method. All of the axes specified by
                the *axes* parameter are collapsed simultaneously by this
                method. The method is given by one of the following
                strings (see
                https://ncas-cms.github.io/cf-python/analysis.html#collapse-methods
                for precise definitions):

                ============================  ============================  ========
                *method*                      Description                   Weighted
                ============================  ============================  ========
                ``'maximum'``                 The maximum of the values.    Never

                ``'minimum'``                 The minimum of the values.    Never

                ``'maximum_absolute_value'``  The maximum of the absolute   Never
                                              values.

                ``'minimum_absolute_value'``  The minimum of the absolute   Never
                                              values.

                ``'mid_range'``               The average of the maximum    Never
                                              and the minimum of the
                                              values.

                ``'median'``                  The median of the values.     Never

                ``'range'``                   The absolute difference       Never
                                              between the maximum and the
                                              minimum of the values.

                ``'sum'``                     The sum of the values.        Never

                ``'sum_of_squares'``          The sum of the squares of     Never
                                              values.

                ``'sample_size'``             The sample size, i.e. the     Never
                                              number of non-missing
                                              values.

                ``'sum_of_weights'``          The sum of weights, as        Never
                                              would be used for other
                                              calculations.

                ``'sum_of_weights2'``         The sum of squares of         Never
                                              weights, as would be used
                                              for other calculations.

                ``'mean'``                    The weighted or unweighted    May be
                                              mean of the values.

                ``'mean_absolute_value'``     The mean of the absolute      May be
                                              values.

                ``'mean_of_upper_decile'``    The mean of the upper group   May be
                                              of data values defined by
                                              the upper tenth of their
                                              distribution.

                ``'variance'``                The weighted or unweighted    May be
                                              variance of the values, with
                                              a given number of degrees of
                                              freedom.

                ``'standard_deviation'``      The weighted or unweighted    May be
                                              standard deviation of the
                                              values, with a given number
                                              of degrees of freedom.

                ``'root_mean_square'``        The square root of the        May be
                                              weighted or unweighted mean
                                              of the squares of the
                                              values.

                ``'integral'``                The integral of values.       Always
                ============================  ============================  ========

                * Collapse methods that are "Never" weighted ignore the
                  *weights* parameter, even if it is set.

                * Collapse methods that "May be" weighted will only be
                  weighted if the *weights* parameter is set.

                * Collapse methods that are "Always" weighted require the
                  *weights* parameter to be set.

                An alternative form of providing the collapse method is to
                provide a CF cell methods-like string. In this case an
                ordered sequence of collapses may be defined and both the
                collapse methods and their axes are provided. The axes are
                interpreted as for the *axes* parameter, which must not
                also be set. For example:

                >>> g = f.collapse(
                ...     'time: max (interval 1 hr) X: Y: mean dim3: sd')

                is equivalent to:

                >>> g = f.collapse('max', axes='time')
                >>> g = g.collapse('mean', axes=['X', 'Y'])
                >>> g = g.collapse('sd', axes='dim3')

                Climatological collapses are carried out if a *method*
                string contains any of the modifiers ``'within days'``,
                ``'within years'``, ``'over days'`` or ``'over
                years'``. For example, to collapse a time axis into
                multiannual means of calendar monthly minima:

                >>> g = f.collapse(
                ...     'time: minimum within years T: mean over years',
                ...     within_years=cf.M()
                ... )

                which is equivalent to:

                >>> g = f.collapse(
                ...     'time: minimum within years', within_years=cf.M())
                >>> g = g.collapse('mean over years', axes='T')

            axes: (sequence of) `str`, optional
                The axes to be collapsed, defined by those which would be
                selected by passing each given axis description to a call
                of the field construct's `domain_axis` method. For
                example, for a value of ``'X'``, the domain axis construct
                returned by ``f.domain_axis('X')`` is selected. If a
                selected axis has size 1 then it is ignored. By default
                all axes with size greater than 1 are collapsed.

                *Parameter example:*
                  ``axes='X'``

                *Parameter example:*
                  ``axes=['X']``

                *Parameter example:*
                  ``axes=['X', 'Y']``

                *Parameter example:*
                  ``axes=['Z', 'time']``

                If the *axes* parameter has the special value ``'area'``
                then it is assumed that the X and Y axes are intended.

                *Parameter example:*
                  ``axes='area'`` is equivalent to ``axes=['X', 'Y']``.

                *Parameter example:*
                  ``axes=['area', Z']`` is equivalent to ``axes=['X', 'Y',
                  'Z']``.

            weights: optional
                Specify the weights for the collapse axes. The weights
                are, in general, those that would be returned by this call
                of the field construct's `weights` method:
                ``f.weights(weights, axes=axes, measure=measure,
                scale=scale, radius=radius, great_circle=great_circle,
                components=True)``. See the *axes*, *measure*, *scale*,
                *radius* and *great_circle* parameters and
                `cf.Field.weights` for details.

                .. note:: By default *weights* is `None`, resulting in
                          **unweighted calculations**.

                If the alternative form of providing the collapse method
                and axes combined as a CF cell methods-like string via the
                *method* parameter has been used, then the *axes*
                parameter is ignored and the axes are derived from the
                *method* parameter. For example, if *method* is ``'T:
                area: minimum'`` then this defines axes of ``['T',
                'area']``. If *method* specifies multiple collapses,
                e.g. ``'T: minimum area: mean'`` then this implies axes of
                ``'T'`` for the first collapse, and axes of ``'area'`` for
                the second collapse.

                .. note:: Setting *weights* to `True` is generally a good
                          way to ensure that all collapses are
                          appropriately weighted according to the field
                          construct's metadata. In this case, if it is not
                          possible to create weights for any axis then an
                          exception will be raised.

                          However, care needs to be taken if *weights* is
                          `True` when cell volume weights are desired. The
                          volume weights will be taken from a "volume"
                          cell measure construct if one exists, otherwise
                          the cell volumes will be calculated as being
                          proportional to the sizes of one-dimensional
                          vertical coordinate cells. In the latter case
                          **if the vertical dimension coordinates do not
                          define the actual height or depth thickness of
                          every cell in the domain then the weights will
                          be incorrect**.

                *Parameter example:*
                  To specify weights based on the field construct's
                  metadata for all collapse axes use ``weights=True``.

                *Parameter example:*
                  To specify weights based on cell areas use
                  ``weights='area'``.

                *Parameter example:*
                  To specify weights based on cell areas and linearly in
                  time you could set ``weights=('area', 'T')``.

            measure: `bool`, optional
                Create weights which are cell measures, i.e. which
                describe actual cell sizes (e.g. cell area) with
                appropriate units (e.g. metres squared). By default the
                weights are normalized and have arbitrary units.

                Cell measures can be created for any combination of
                axes. For example, cell measures for a time axis are the
                time span for each cell with canonical units of seconds;
                cell measures for the combination of four axes
                representing time and three dimensional space could have
                canonical units of metres cubed seconds.

                When collapsing with the ``'integral'`` method, *measure*
                must be True, and the units of the weights are
                incorporated into the units of the returned field
                construct.

                .. note:: Specifying cell volume weights via
                          ``weights=['X', 'Y', 'Z']`` or
                          ``weights=['area', 'Z']`` (or other equivalents)
                          will produce **an incorrect result if the
                          vertical dimension coordinates do not define the
                          actual height or depth thickness of every cell
                          in the domain**. In this case,
                          ``weights='volume'`` should be used instead,
                          which requires the field construct to have a
                          "volume" cell measure construct.

                          If ``weights=True`` then care also needs to be
                          taken, as a "volume" cell measure construct will
                          be used if present, otherwise the cell volumes
                          will be calculated using the size of the
                          vertical coordinate cells.

                .. versionadded:: 3.0.2

            scale: number, optional
                If set to a positive number then scale the weights so that
                they are less than or equal to that number. By default the
                weights are scaled to lie between 0 and 1 (i.e.  *scale*
                is 1).

                *Parameter example:*
                  To scale all weights so that they lie between 0 and 0.5:
                  ``scale=0.5``.

                .. versionadded:: 3.0.2

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

                .. versionadded:: 3.0.2

            great_circle: `bool`, optional
                If True then allow, if required, the derivation of i) area
                weights from polygon geometry cells by assuming that each
                cell part is a spherical polygon composed of great circle
                segments; and ii) and the derivation of line-length
                weights from line geometry cells by assuming that each
                line part is composed of great circle segments.

                .. versionadded:: 3.2.0

            squeeze: `bool`, optional
                If True then size 1 collapsed axes are removed from the
                output data array. By default the axes which are collapsed
                are retained in the result's data array.

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

            ddof: number, optional
                The delta degrees of freedom in the calculation of a
                standard deviation or variance. The number of degrees of
                freedom used in the calculation is (N-*ddof*) where N
                represents the number of non-missing elements. By default
                *ddof* is 1, meaning the standard deviation and variance
                of the population is estimated according to the usual
                formula with (N-1) in the denominator to avoid the bias
                caused by the use of the sample mean (Bessel's
                correction).

            coordinate: optional
                Specify how the cell coordinate values for collapsed axes
                are placed. This has no effect on the cell bounds for the
                collapsed axes, which always represent the extrema of the
                input coordinates.

                The *coordinate* parameter may be one of:

                ===============  =========================================
                *coordinate*     Description
                ===============  =========================================
                `None`           This is the default.

                                 If the collapse is a climatological time
                                 collapse over years or over days then
                                 assume a value of ``'min'``, otherwise
                                 assume value of ``'mid_range'``.

                ``'mid_range'``  An output coordinate is the mean of
                                 first and last input coordinate bounds
                                 (or the first and last coordinates if
                                 there are no bounds). This is the
                                 default.

                ``'minimum'``    An output coordinate is the minimum of
                                 the input coordinates.

                ``'maximum'``    An output coordinate is the maximum of
                                 the input coordinates.
                ===============  =========================================

                *Parameter example:*
                  ``coordinate='minimum'``

            group: optional
                A grouped collapse is one for which an axis is not
                collapsed completely to size 1. Instead, the collapse axis
                is partitioned into non-overlapping groups and each group
                is collapsed to size 1, independently of the other
                groups. The results of the collapses are concatenated so
                that the output axis has a size equal to the number of
                groups.

                An element of the collapse axis can not be a member of
                more than one group, and may be a member of no
                groups. Elements that are not selected by the *group*
                parameter are excluded from the result.

                The *group* parameter defines how the axis elements are
                partitioned into groups, and may be one of:

                ===============  =========================================
                *group*          Description
                ===============  =========================================
                `Data`           Define groups by coordinate values that
                                 span the given range. The first group
                                 starts at the first coordinate bound of
                                 the first axis element (or its coordinate
                                 if there are no bounds) and spans the
                                 defined group size. Each subsequent
                                 group immediately follows the preceding
                                 one. By default each group contains the
                                 consecutive run of elements whose
                                 coordinate values lie within the group
                                 limits (see the *group_by* parameter).

                                 * By default each element will be in
                                   exactly one group (see the *group_by*,
                                   *group_span* and *group_contiguous*
                                   parameters).

                                 * By default groups may contain different
                                   numbers of elements.

                                 * If no units are specified then the
                                   units of the coordinates are assumed.

                `TimeDuration`   Define groups by a time interval spanned
                                 by the coordinates. The first group
                                 starts at or before the first coordinate
                                 bound of the first axis element (or its
                                 coordinate if there are no bounds) and
                                 spans the defined group size. Each
                                 subsequent group immediately follows the
                                 preceding one. By default each group
                                 contains the consecutive run of elements
                                 whose coordinate values lie within the
                                 group limits (see the *group_by*
                                 parameter).

                                 * By default each element will be in
                                   exactly one group (see the *group_by*,
                                   *group_span* and *group_contiguous*
                                   parameters).

                                 * By default groups may contain different
                                   numbers of elements.

                                 * The start of the first group may be
                                   before the first first axis element,
                                   depending on the offset defined by the
                                   time duration. For example, if
                                   ``group=cf.Y(month=12)`` then the first
                                   group will start on the closest 1st
                                   December to the first axis element.

                `Query`          Define groups from elements whose
                                 coordinates satisfy the query
                                 condition. Multiple groups are created:
                                 one for each maximally consecutive run
                                 within the selected elements.

                                 If a sequence of `Query` is provided then
                                 groups are defined for each query.

                                 * If a coordinate does not satisfy any of
                                   the query conditions then its element
                                   will not be in a group.

                                 * By default groups may contain different
                                   numbers of elements.

                                 * If no units are specified then the
                                   units of the coordinates are assumed.

                                 * If an element is selected by two or
                                   more queries then the latest one in the
                                   sequence defines which group it will be
                                   in.

                `int`            Define groups that contain the given
                                 number of elements. The first group
                                 starts with the first axis element and
                                 spans the defined number of consecutive
                                 elements. Each subsequent group
                                 immediately follows the preceding one.

                                 * By default each group has the defined
                                   number of elements, apart from the last
                                   group which may contain fewer elements
                                   (see the *group_span* parameter).

                `numpy.ndarray`  Define groups by selecting elements that
                                 map to the same value in the `numpy`
                                 array. The array must contain integers
                                 and have the same length as the axis to
                                 be collapsed and its sequence of values
                                 correspond to the axis elements. Each
                                 group contains the elements which
                                 correspond to a common non-negative
                                 integer value in the numpy array. Upon
                                 output, the collapsed axis is arranged in
                                 order of increasing group number. See the
                                 *regroup* parameter, which allows the
                                 creation of such a `numpy.array` for a
                                 given grouped collapse.

                                 * The groups do not have to be in runs of
                                   consecutive elements; they may be
                                   scattered throughout the axis.

                                 * An element which corresponds to a
                                   negative integer in the array will not
                                   be in any group.
                ===============  =========================================

                *Parameter example:*
                  To define groups of 10 kilometres: ``group=cf.Data(10,
                  'km')``.

                *Parameter example:*
                  To define groups of 5 days, starting and ending at
                  midnight on each day: ``group=cf.D(5)`` (see `cf.D`).

                *Parameter example:*
                  To define groups of 1 calendar month, starting and
                  ending at day 16 of each month: ``group=cf.M(day=16)``
                  (see `cf.M`).

                *Parameter example:*
                  To define groups of the season MAM in each year:
                  ``group=cf.mam()`` (see `cf.mam`).

                *Parameter example:*
                  To define groups of the seasons DJF and JJA in each
                  year: ``group=[cf.jja(), cf.djf()]``. To define groups
                  for seasons DJF, MAM, JJA and SON in each year:
                  ``group=cf.seasons()`` (see `cf.djf`, `cf.jja` and
                  `cf.season`).

                *Parameter example:*
                  To define groups for longitude elements less than or
                  equal to 90 degrees and greater than 90 degrees:
                  ``group=[cf.le(90, 'degrees'), cf.gt(90, 'degrees')]``
                  (see `cf.le` and `cf.gt`).

                *Parameter example:*
                  To define groups of 5 elements: ``group=5``.

                *Parameter example:*
                  For an axis of size 8, create two groups, the first
                  containing the first and last elements and the second
                  containing the 3rd, 4th and 5th elements, whilst
                  ignoring the 2nd, 6th and 7th elements:
                  ``group=numpy.array([0, -1, 4, 4, 4, -1, -2, 0])``.

            regroup: `bool`, optional
                If True then, for grouped collapses, do not collapse the
                field construct, but instead return a `numpy.array` of
                integers which identifies the groups defined by the
                *group* parameter. Each group contains the elements which
                correspond to a common non-negative integer value in the
                numpy array. Elements corresponding to negative integers
                are not in any group. The array may subsequently be used
                as the value of the *group* parameter in a separate
                collapse.

                For example:

                >>> groups = f.collapse('time: mean', group=10, regroup=True)
                >>> g = f.collapse('time: mean', group=groups)

                is equivalent to:

                >>> g = f.collapse('time: mean', group=10)

            group_by: optional
                Specify how coordinates are assigned to the groups defined
                by the *group*, *within_days* or *within_years*
                parameters. Ignored unless one of these parameters is set
                to a `Data` or `TimeDuration` object.

                The *group_by* parameter may be one of:

                ============  ============================================
                *group_by*    Description
                ============  ============================================
                `None`        This is the default.

                              If the groups are defined by the *group*
                              parameter (i.e. collapses other than
                              climatological time collapses) then assume a
                              value of ``'coords'``.

                              If the groups are defined by the
                              *within_days* or *within_years* parameter
                              (i.e. climatological time collapses) then
                              assume a value of ``'bounds'``.

                ``'coords'``  Each group contains the axis elements whose
                              coordinate values lie within the group
                              limits. Every element will be in a group.

                ``'bounds'``  Each group contains the axis elements whose
                              upper and lower coordinate bounds both lie
                              within the group limits. Some elements may
                              not be inside any group, either because the
                              group limits do not coincide with coordinate
                              bounds or because the group size is
                              sufficiently small.
                ============  ============================================

            group_span: optional
                Specify how to treat groups that may not span the desired
                range. For example, when creating 3-month means, the
                *group_span* parameter can be used to allow groups which
                only contain 1 or 2 months of data.

                By default, *group_span* is `None`. This means that only
                groups whose span equals the size specified by the
                definition of the groups are collapsed; unless the groups
                have been defined by one or more `Query` objects, in which
                case then the default behaviour is to collapse all groups,
                regardless of their size.

                In effect, the *group_span* parameter defaults to `True`
                unless the groups have been defined by one or more `Query`
                objects, in which case *group_span* defaults to `False`.

                The different behaviour when the groups have been defined
                by one or more `Query` objects is necessary because a
                `Query` object can only define the composition of a group,
                and not its size (see the parameter examples below for how
                to specify a group span in this case).

                .. note:: Prior to version 3.1.0, the default value of
                          *group_span* was effectively `False`.

                In general, the span of a group is the absolute difference
                between the lower bound of its first element and the upper
                bound of its last element. The only exception to this
                occurs if *group_span* is (by default or by explicit
                setting) an integer, in which case the span of a group is
                the number of elements in the group. See also the
                *group_contiguous* parameter for how to deal with groups
                that have gaps in their coverage.

                The *group_span* parameter is only applied to groups
                defined by the *group*, *within_days* or *within_years*
                parameters, and is otherwise ignored.

                The *group_span* parameter may be one of:

                ==============  ==========================================
                *group_span*    Description
                ==============  ==========================================
                `None`          This is the default. Apply a value of
                                `True` or `False` depending on how the
                                groups have been defined.

                `True`          Ignore groups whose span is not equal to
                                the size specified by the definition of
                                the groups. Only applicable if the groups
                                are defined by a `Data`, `TimeDuration` or
                                `int` object, and this is the default in
                                this case.

                `False`         Collapse all groups, regardless of their
                                size. This is the default if the groups
                                are defined by one to more `Query`
                                objects.

                `Data`          Ignore groups whose span is not equal to
                                the given size. If no units are specified
                                then the units of the coordinates are
                                assumed.

                `TimeDuration`  Ignore groups whose span is not equals to
                                the given time duration.

                `int`           Ignore groups that contain fewer than the
                                given number of elements
                ==============  ==========================================

                *Parameter example:*
                  To collapse into groups of 10km, ignoring any groups
                  that span less than that distance: ``group=cf.Data(10,
                  'km'), group_span=True``.

                *Parameter example:*
                  To collapse a daily timeseries into monthly groups,
                  ignoring any groups that span less than 1 calendar
                  month: monthly values: ``group=cf.M(), group_span=True``
                  (see `cf.M`).

                *Parameter example:*
                  To collapse a timeseries into seasonal groups, ignoring
                  any groups that span less than three months:
                  ``group=cf.seasons(), group_span=cf.M(3)`` (see
                  `cf.seasons` and `cf.M`).

            group_contiguous: `int`, optional
                Specify how to treat groups whose elements are not
                contiguous or have overlapping cells. For example, when
                creating a December to February means, the
                *group_contiguous* parameter can be used to allow groups
                which have no data for January.

                A group is considered to be contiguous unless it has
                coordinates with bounds that do not coincide for adjacent
                cells. The definition may be expanded to include groups
                whose coordinate bounds that overlap.

                By default *group_contiguous* is ``1``, meaning that
                non-contiguous groups, and those whose coordinate bounds
                overlap, are not collapsed

                .. note:: Prior to version 3.1.0, the default value of
                          *group_contiguous* was ``0``.

                The *group_contiguous* parameter is only applied to groups
                defined by the *group*, *within_days* or *within_years*
                parameters, and is otherwise ignored.

                The *group_contiguous* parameter may be one of:

                ===================  =====================================
                *group_contiguous*   Description
                ===================  =====================================
                ``0``                Allow non-contiguous groups, and
                                     those containing overlapping cells.

                ``1``                This is the default. Ignore
                                     non-contiguous groups, as well as
                                     contiguous groups containing
                                     overlapping cells.

                ``2``                Ignore non-contiguous groups,
                                     allowing contiguous groups containing
                                     overlapping cells.
                ===================  =====================================

                *Parameter example:*
                  To allow non-contiguous groups, and those containing
                  overlapping cells: ``group_contiguous=0``.

            within_days: optional
                Define the groups for creating CF "within days"
                climatological statistics.

                Each group contains elements whose coordinates span a time
                interval of up to one day. The results of the collapses
                are concatenated so that the output axis has a size equal
                to the number of groups.

                .. note:: For CF compliance, a "within days" collapse
                          should be followed by an "over days" collapse.

                The *within_days* parameter defines how the elements are
                partitioned into groups, and may be one of:

                ==============  ==========================================
                *within_days*   Description
                ==============  ==========================================
                `TimeDuration`  Defines the group size in terms of a time
                                interval of up to one day. The first group
                                starts at or before the first coordinate
                                bound of the first axis element (or its
                                coordinate if there are no bounds) and
                                spans the defined group size. Each
                                subsequent group immediately follows the
                                preceding one. By default each group
                                contains the consecutive run of elements
                                whose coordinate cells lie within the
                                group limits (see the *group_by*
                                parameter).

                                * Groups may contain different numbers of
                                  elements.

                                * The start of the first group may be
                                  before the first first axis element,
                                  depending on the offset defined by the
                                  time duration. For example, if
                                  ``group=cf.D(hour=12)`` then the first
                                  group will start on the closest midday
                                  to the first axis element.

                `Query`         Define groups from elements whose
                                coordinates satisfy the query
                                condition. Multiple groups are created:
                                one for each maximally consecutive run
                                within the selected elements.

                                If a sequence of `Query` is provided then
                                groups are defined for each query.

                                * Groups may contain different numbers of
                                  elements.

                                * If no units are specified then the units
                                  of the coordinates are assumed.

                                * If a coordinate does not satisfy any of
                                  the conditions then its element will not
                                  be in a group.

                                * If an element is selected by two or more
                                  queries then the latest one in the
                                  sequence defines which group it will be
                                  in.
                ==============  ==========================================

                *Parameter example:*
                  To define groups of 6 hours, starting at 00:00, 06:00,
                  12:00 and 18:00: ``within_days=cf.h(6)`` (see `cf.h`).

                *Parameter example:*
                  To define groups of 1 day, starting at 06:00:
                  ``within_days=cf.D(1, hour=6)`` (see `cf.D`).

                *Parameter example:*
                  To define groups of 00:00 to 06:00 within each day,
                  ignoring the rest of each day:
                  ``within_days=cf.hour(cf.le(6))`` (see `cf.hour` and
                  `cf.le`).

                *Parameter example:*
                  To define groups of 00:00 to 06:00 and 18:00 to 24:00
                  within each day, ignoring the rest of each day:
                  ``within_days=[cf.hour(cf.le(6)), cf.hour(cf.gt(18))]``
                  (see `cf.gt`, `cf.hour` and `cf.le`).

            within_years: optional
                Define the groups for creating CF "within years"
                climatological statistics.

                Each group contains elements whose coordinates span a time
                interval of up to one calendar year. The results of the
                collapses are concatenated so that the output axis has a
                size equal to the number of groups.

                .. note:: For CF compliance, a "within years" collapse
                          should be followed by an "over years" collapse.

                The *within_years* parameter defines how the elements are
                partitioned into groups, and may be one of:

                ==============  ==========================================
                *within_years*  Description
                ==============  ==========================================
                `TimeDuration`  Define the group size in terms of a time
                                interval of up to one calendar year. The
                                first group starts at or before the first
                                coordinate bound of the first axis element
                                (or its coordinate if there are no bounds)
                                and spans the defined group size. Each
                                subsequent group immediately follows the
                                preceding one. By default each group
                                contains the consecutive run of elements
                                whose coordinate cells lie within the
                                group limits (see the *group_by*
                                parameter).

                                * Groups may contain different numbers of
                                  elements.

                                * The start of the first group may be
                                  before the first first axis element,
                                  depending on the offset defined by the
                                  time duration. For example, if
                                  ``group=cf.Y(month=12)`` then the first
                                  group will start on the closest 1st
                                  December to the first axis element.

                 `Query`        Define groups from elements whose
                                coordinates satisfy the query
                                condition. Multiple groups are created:
                                one for each maximally consecutive run
                                within the selected elements.

                                If a sequence of `Query` is provided then
                                groups are defined for each query.

                                * The first group may start outside of the
                                  range of coordinates (the start of the
                                  first group is controlled by parameters
                                  of the `TimeDuration`).

                                * If group boundaries do not coincide with
                                  coordinate bounds then some elements may
                                  not be inside any group.

                                * If the group size is sufficiently small
                                  then some elements may not be inside any
                                  group.

                                * Groups may contain different numbers of
                                  elements.
                ==============  ==========================================

                *Parameter example:*
                  To define groups of 90 days: ``within_years=cf.D(90)``
                  (see `cf.D`).

                *Parameter example:*
                  To define groups of 3 calendar months, starting on the
                  15th of a month: ``within_years=cf.M(3, day=15)`` (see
                  `cf.M`).

                *Parameter example:*
                  To define groups for the season MAM within each year:
                  ``within_years=cf.mam()`` (see `cf.mam`).

                *Parameter example:*
                  To define groups for February and for November to
                  December within each year: ``within_years=[cf.month(2),
                  cf.month(cf.ge(11))]`` (see `cf.month` and `cf.ge`).

            over_days: optional
                Define the groups for creating CF "over days"
                climatological statistics.

                By default (or if *over_days* is `None`) each group
                contains all elements for which the time coordinate cell
                lower bounds have a common time of day but different
                dates, and for which the time coordinate cell upper bounds
                also have a common time of day but different dates. The
                collapsed dime axis will have a size equal to the number
                of groups that were found.

                For example, elements corresponding to the two time
                coordinate cells

                  | ``1999-12-31 06:00:00/1999-12-31 18:00:00``
                  | ``2000-01-01 06:00:00/2000-01-01 18:00:00``

                would be together in a group; and elements corresponding
                to the two time coordinate cells

                  | ``1999-12-31 00:00:00/2000-01-01 00:00:00``
                  | ``2000-01-01 00:00:00/2000-01-02 00:00:00``

                would also be together in a different group.

                .. note:: For CF compliance, an "over days" collapse
                          should be preceded by a "within days" collapse.

                The default groups may be split into smaller groups if the
                *over_days* parameter is one of:

                ==============  ==========================================
                *over_days*     Description
                ==============  ==========================================
                `TimeDuration`  Split each default group into smaller
                                groups which span the given time duration,
                                which must be at least one day.

                                * Groups may contain different numbers of
                                  elements.

                                * The start of the first group may be
                                  before the first first axis element,
                                  depending on the offset defined by the
                                  time duration. For example, if
                                  ``group=cf.M(day=15)`` then the first
                                  group will start on the closest 15th of
                                  a month to the first axis element.

                `Query`         Split each default group into smaller
                                groups whose coordinate cells satisfy the
                                query condition.

                                If a sequence of `Query` is provided then
                                groups are defined for each query.

                                * Groups may contain different numbers of
                                  elements.

                                * If a coordinate does not satisfy any of
                                  the conditions then its element will not
                                  be in a group.

                                * If an element is selected by two or more
                                  queries then the latest one in the
                                  sequence defines which group it will be
                                  in.
                ==============  ==========================================

                *Parameter example:*
                  To define groups for January and for June to December,
                  ignoring all other months: ``over_days=[cf.month(1),
                  cf.month(cf.wi(6, 12))]`` (see `cf.month` and `cf.wi`).

                *Parameter example:*
                  To define groups spanning 90 days:
                  ``over_days=cf.D(90)`` or ``over_days=cf.h(2160)``. (see
                  `cf.D` and `cf.h`).

                *Parameter example:*
                  To define groups that each span 3 calendar months,
                  starting and ending at 06:00 in the first day of each
                  month: ``over_days=cf.M(3, hour=6)`` (see `cf.M`).

                *Parameter example:*
                  To define groups that each span a calendar month
                  ``over_days=cf.M()`` (see `cf.M`).

                *Parameter example:*
                  To define groups for January and for June to December,
                  ignoring all other months: ``over_days=[cf.month(1),
                  cf.month(cf.wi(6, 12))]`` (see `cf.month` and `cf.wi`).

            over_years: optional
                Define the groups for creating CF "over years"
                climatological statistics.

                By default (or if *over_years* is `None`) each group
                contains all elements for which the time coordinate cell
                lower bounds have a common date of the year but different
                years, and for which the time coordinate cell upper bounds
                also have a common date of the year but different
                years. The collapsed dime axis will have a size equal to
                the number of groups that were found.

                For example, elements corresponding to the two time
                coordinate cells

                  | ``1999-12-01 00:00:00/2000-01-01 00:00:00``
                  | ``2000-12-01 00:00:00/2001-01-01 00:00:00``

                would be together in a group.

                .. note:: For CF compliance, an "over years" collapse
                          should be preceded by a "within years" or "over
                          days" collapse.

                The default groups may be split into smaller groups if the
                *over_years* parameter is one of:

                ==============  ==========================================
                *over_years*    Description
                ==============  ==========================================
                `TimeDuration`  Split each default group into smaller
                                groups which span the given time duration,
                                which must be at least one day.

                                * Groups may contain different numbers of
                                  elements.

                                * The start of the first group may be
                                  before the first first axis element,
                                  depending on the offset defined by the
                                  time duration. For example, if
                                  ``group=cf.Y(month=12)`` then the first
                                  group will start on the closest 1st
                                  December to the first axis element.

                `Query`         Split each default group into smaller
                                groups whose coordinate cells satisfy the
                                query condition.

                                If a sequence of `Query` is provided then
                                groups are defined for each query.

                                * Groups may contain different numbers of
                                  elements.

                                * If a coordinate does not satisfy any of
                                  the conditions then its element will not
                                  be in a group.

                                * If an element is selected by two or more
                                  queries then the latest one in the
                                  sequence defines which group it will be
                                  in.
                ==============  ==========================================

                *Parameter example:*
                  An element with coordinate bounds {1999-06-01 06:00:00,
                  1999-09-01 06:00:00} **matches** an element with
                  coordinate bounds {2000-06-01 06:00:00, 2000-09-01
                  06:00:00}.

                *Parameter example:*
                  An element with coordinate bounds {1999-12-01 00:00:00,
                  2000-12-01 00:00:00} **matches** an element with
                  coordinate bounds {2000-12-01 00:00:00, 2001-12-01
                  00:00:00}.

                *Parameter example:*
                  To define groups spanning 10 calendar years:
                  ``over_years=cf.Y(10)`` or ``over_years=cf.M(120)`` (see
                  `cf.M` and `cf.Y`).

                *Parameter example:*
                  To define groups spanning 5 calendar years, starting and
                  ending at 06:00 on 01 December of each year:
                  ``over_years=cf.Y(5, month=12, hour=6)`` (see `cf.Y`).

                *Parameter example:*
                  To define one group spanning 1981 to 1990 and another
                  spanning 2001 to 2005: ``over_years=[cf.year(cf.wi(1981,
                  1990), cf.year(cf.wi(2001, 2005)]`` (see `cf.year` and
                  `cf.wi`).

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

            kwargs: deprecated at version 3.0.0

        :Returns:

            `Field` or `numpy.ndarray`
                 The collapsed field construct. Alternatively, if the
                 *regroup* parameter is True then a `numpy` array is
                 returned.

        **Examples:**

        There are further worked examples in
        https://ncas-cms.github.io/cf-python/analysis.html#statistical-collapses

        """
        if _debug:
            _DEPRECATION_ERROR_KWARGS(
                self,
                "collapse",
                {"_debug": _debug},
                "Use keyword 'verbose' instead.",
            )  # pragma: no cover

        if kwargs:
            _DEPRECATION_ERROR_KWARGS(
                self, "collapse", kwargs
            )  # pragma: no cover

        if inplace:
            f = self
        else:
            f = self.copy()

        # Whether or not to create null bounds for null
        # collapses. I.e. if the collapse axis has size 1 and no
        # bounds, whether or not to create upper and lower bounds to
        # the coordinate value. If this occurs it's because the null
        # collapse is part of a grouped collapse and so will be
        # concatenated to other collapses for the final result: bounds
        # will be made for the grouped collapse, so all elements need
        # bounds.
        #        _create_zero_size_cell_bounds = kwargs.get(
        #            '_create_zero_size_cell_bounds', False)

        # ------------------------------------------------------------
        # Parse the methods and axes
        # ------------------------------------------------------------
        if ":" in method:
            # Convert a cell methods string (such as 'area: mean dim3:
            # dim2: max T: minimum height: variance') to a CellMethod
            # construct
            if axes is not None:
                raise ValueError(
                    "Can't collapse: Can't set 'axes' when 'method' is "
                    "CF-like cell methods string"
                )

            all_methods = []
            all_axes = []
            all_within = []
            all_over = []

            for cm in CellMethod.create(method):
                all_methods.append(cm.get_method(None))
                all_axes.append(cm.get_axes(()))
                all_within.append(cm.get_qualifier("within", None))
                all_over.append(cm.get_qualifier("over", None))
        else:
            x = method.split(" within ")
            if method == x[0]:
                within = None
                x = method.split(" over ")
                if method == x[0]:
                    over = None
                else:
                    method, over = x
            else:
                method, within = x

            if isinstance(axes, (str, int)):
                axes = (axes,)

            all_methods = (method,)
            all_within = (within,)
            all_over = (over,)
            all_axes = (axes,)

        # ------------------------------------------------------------
        # Convert axes into domain axis construct keys
        # ------------------------------------------------------------
        domain_axes = None

        input_axes = all_axes
        all_axes = []
        for axes in input_axes:
            if axes is None:
                domain_axes = self.domain_axes(
                    todict=False, cached=domain_axes
                )
                all_axes.append(list(domain_axes))
                continue

            axes2 = []
            for axis in axes:
                msg = (
                    "Must have '{}' axes for an '{}' collapse. Can't "
                    "find {{!r}} axis"
                )
                if axis == "area":
                    iterate_over = ("X", "Y")
                    msg = msg.format("', '".join(iterate_over), axis)
                elif axis == "volume":
                    iterate_over = ("X", "Y", "Z")
                    msg = msg.format("', '".join(iterate_over), axis)
                else:
                    iterate_over = (axis,)
                    msg = "Can't find the collapse axis identified by {!r}"

                for x in iterate_over:
                    a = self.domain_axis(x, key=True, default=None)
                    if a is None:
                        raise ValueError(msg.format(x))
                    axes2.append(a)

            all_axes.append(axes2)

        logger.info(
            "    all_methods, all_axes, all_within, all_over = "
            "{} {} {} {}".format(all_methods, all_axes, all_within, all_over)
        )  # pragma: no cover

        if group is not None and len(all_axes) > 1:
            raise ValueError(
                "Can't use the 'group' parameter for multiple collapses"
            )

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        domain_axes = f.domain_axes(todict=False, cached=domain_axes)
        #        auxiliary_coordinates = f.auxiliary_coordinates(view=True)
        #        dimension_coordinates = f.dimension_coordinates(view=True)

        for method, axes, within, over, axes_in in zip(
            all_methods, all_axes, all_within, all_over, input_axes
        ):

            method2 = _collapse_methods.get(method, None)
            if method2 is None:
                raise ValueError(
                    "Unknown collapse method: {!r}".format(method)
                )

            method = method2

            #            collapse_axes_all_sizes = domain_axes.filter_by_key(*axes)
            collapse_axes_all_sizes = f.domain_axes(
                filter_by_key=axes, todict=False
            )

            logger.info(
                "    axes                    = {}".format(axes)
            )  # pragma: no cover
            logger.info(
                "    method                  = {}".format(method)
            )  # pragma: no cover
            logger.info(
                "    collapse_axes_all_sizes = {}".format(
                    collapse_axes_all_sizes
                )
            )  # pragma: no cover

            if not collapse_axes_all_sizes:
                raise ValueError(
                    "Can't collapse: Can not identify collapse axes"
                )

            if method in (
                "sum_of_weights",
                "sum_of_weights2",
                "sample_size",
                "integral",
                "maximum_absolute_value",
                "minimum_absolute_value",
                "mean_absolute_value",
                "range",
                "root_mean_square",
                "sum_of_squares",
            ):
                collapse_axes = collapse_axes_all_sizes.todict()  # copy()
            else:
                collapse_axes = collapse_axes_all_sizes.filter_by_size(
                    gt(1), todict=True
                )

            logger.info(
                "    collapse_axes           = {}".format(collapse_axes)
            )  # pragma: no cover

            if not collapse_axes:
                # Do nothing if there are no collapse axes
                if _create_zero_size_cell_bounds:
                    # Create null bounds if requested
                    for axis in axes:
                        #                        dc = f.dimension_coordinates(
                        #                            filter_by_axis=(axis,), axis_mode="and", todict=Tru#e
                        #                        ).value(None)
                        dc = f.dimension_coordinate(
                            filter_by_axis=(axis,),
                            default=None,
                        )
                        if dc is not None and not dc.has_bounds():
                            dc.set_bounds(dc.create_bounds(cellsize=0))

                continue

            # Check that there are enough elements to collapse
            collapse_axes_sizes = [
                da.get_size() for da in collapse_axes.values()
            ]
            size = reduce(operator_mul, collapse_axes_sizes, 1)

            logger.info(
                "    collapse_axes_sizes     = {}".format(collapse_axes_sizes)
            )  # pragma: no cover

            grouped_collapse = (
                within is not None or over is not None or group is not None
            )

            # --------------------------------------------------------
            # Set the group_by parameter
            # --------------------------------------------------------
            if group_by is None:
                if within is None and over is None:
                    group_by = "coords"
                else:
                    group_by = "bounds"
            elif (
                within is not None or over is not None
            ) and group_by == "coords":
                raise ValueError(
                    "Can't collapse: group_by parameter can't be "
                    "'coords' for a climatological time collapse."
                )

            # --------------------------------------------------------
            # Set the coordinate parameter
            # --------------------------------------------------------
            if coordinate is None and over is None:
                coordinate = "mid_range"

            if grouped_collapse:
                if len(collapse_axes) > 1:
                    raise ValueError(
                        "Can't do a grouped collapse on multiple axes "
                        "simultaneously"
                    )

                # ------------------------------------------------------------
                # Grouped collapse: Calculate weights
                # ------------------------------------------------------------
                g_weights = weights
                if method not in _collapse_weighted_methods:
                    g_weights = None
                else:
                    # if isinstance(weights, (dict, self.__class__, Data)):
                    #     if measure:
                    #         raise ValueError(
                    #             "TODO")
                    #
                    #     if scale is not None:
                    #         raise ValueError(
                    #             "TODO")
                    if method == "integral":
                        if not measure:
                            raise ValueError(
                                "Must set measure=True for 'integral' "
                                "collapses."
                            )

                        if scale is not None:
                            raise ValueError(
                                "Can't set scale for 'integral' collapses."
                            )
                    elif not measure and scale is None:
                        scale = 1.0
                    elif measure and scale is not None:
                        raise ValueError("TODO")

                    #                    if weights is True:
                    #                        weights = tuple(collapse_axes.keys())

                    g_weights = f.weights(
                        weights,
                        components=True,
                        axes=list(collapse_axes),  # .keys()),
                        scale=scale,
                        measure=measure,
                        radius=radius,
                        great_circle=great_circle,
                    )

                    if not g_weights:
                        g_weights = None

                #                axis = collapse_axes.key()
                axis = [a for a in collapse_axes][0]

                f = f._collapse_grouped(
                    method,
                    axis,
                    within=within,
                    over=over,
                    within_days=within_days,
                    within_years=within_years,
                    over_days=over_days,
                    over_years=over_years,
                    group=group,
                    group_span=group_span,
                    group_contiguous=group_contiguous,
                    regroup=regroup,
                    mtol=mtol,
                    ddof=ddof,
                    measure=measure,
                    weights=g_weights,
                    squeeze=squeeze,
                    coordinate=coordinate,
                    group_by=group_by,
                    axis_in=axes_in[0],
                    verbose=verbose,
                )

                if regroup:
                    # Grouped collapse: Return the numpy array
                    return f

                # ----------------------------------------------------
                # Grouped collapse: Update the cell methods
                # ----------------------------------------------------
                f._update_cell_methods(
                    method=method,
                    domain_axes=collapse_axes,
                    input_axes=axes_in,
                    within=within,
                    over=over,
                    verbose=verbose,
                )
                continue

            elif regroup:
                raise ValueError(
                    "Can't return an array of groups for a non-grouped "
                    "collapse"
                )

            data_axes = f.get_data_axes()
            iaxes = [
                data_axes.index(axis)
                for axis in collapse_axes
                if axis in data_axes
            ]

            # ------------------------------------------------------------
            # Calculate weights
            # ------------------------------------------------------------
            logger.info(
                "    Input weights           = {!r}".format(weights)
            )  # pragma: no cover

            if method not in _collapse_weighted_methods:
                weights = None

            d_kwargs = {}
            if weights is not None:
                # if isinstance(weights, (dict, self.__class__, Data)):
                #     if measure:
                #         raise ValueError("TODO")
                #
                #     if scale is not None:
                #         raise ValueError("TODO")

                if method == "integral":
                    if not measure:
                        raise ValueError(
                            f"Must set measure=True for {method!r} collapses"
                        )

                    if scale is not None:
                        raise ValueError(
                            "Can't set scale for 'integral' collapses."
                        )
                elif not measure and scale is None:
                    scale = 1.0
                elif measure and scale is not None:
                    raise ValueError("TODO")

                d_weights = f.weights(
                    weights,
                    components=True,
                    axes=list(collapse_axes.keys()),
                    scale=scale,
                    measure=measure,
                    radius=radius,
                    great_circle=great_circle,
                )

                if d_weights:
                    d_kwargs["weights"] = d_weights

                logger.info(
                    f"    Output weights          = {d_weights!r}"
                )  # pragma: no cover

            elif method == "integral":
                raise ValueError(
                    f"Must set the 'weights' parameter for {method!r} "
                    "collapses"
                )

            if method in _collapse_ddof_methods:
                d_kwargs["ddof"] = ddof

            # ========================================================
            # Collapse the data array
            # ========================================================
            logger.info(
                "  Before collapse of data:\n"
                f"    iaxes, d_kwargs = {iaxes} {d_kwargs}\n"
                f"    f.shape = {f.shape}\n"
                f"    f.dtype = {f.dtype}\n"
            )  # pragma: no cover

            getattr(f.data, method)(
                axes=iaxes,
                squeeze=squeeze,
                mtol=mtol,
                inplace=True,
                **d_kwargs,
            )

            if squeeze:
                # ----------------------------------------------------
                # Remove the collapsed axes from the field's list of
                # data array axes
                # ----------------------------------------------------
                f.set_data_axes(
                    [axis for axis in data_axes if axis not in collapse_axes]
                )

            logger.info(
                "  After collapse of data:\n"
                f"    f.shape = {f.shape}\n"
                f"    f.dtype = {f.dtype}\n",
                f"collapse_axes = {collapse_axes}",
            )  # pragma: no cover

            # ---------------------------------------------------------
            # Update dimension coordinates, auxiliary coordinates,
            # cell measures and domain ancillaries
            # ---------------------------------------------------------
            for axis, domain_axis in collapse_axes.items():
                # Ignore axes which are already size 1
                size = domain_axis.get_size()
                if size == 1:
                    continue

                # REMOVE all cell measures and domain ancillaries
                # which span this axis
                c = f.constructs.filter(
                    filter_by_type=("cell_measure", "domain_ancillary"),
                    filter_by_axis=(axis,),
                    axis_mode="or",
                    todict=True,
                )
                for key, value in c.items():
                    logger.info(
                        f"    Removing {value.construct_type}"
                    )  # pragma: no cover

                    f.del_construct(key)

                # REMOVE all 2+ dimensional auxiliary coordinates
                # which span this axis
                #                c = auxiliary_coordinates.filter_by_naxes(gt(1), view=True)
                c = f.auxiliary_coordinates(
                    filter_by_naxes=(
                        gt(
                            1,
                        ),
                    ),
                    filter_by_axis=(axis,),
                    axis_mode="or",
                    todict=True,
                )
                for key, value in c.items():
                    logger.info(
                        f"    Removing {value.construct_type} {key!r}"
                    )  # pragma: no cover

                    f.del_construct(key)

                # REMOVE all 1 dimensional auxiliary coordinates which
                # span this axis and have different values in their
                # data array and bounds.
                #
                # KEEP, after changing their data arrays, all
                # one-dimensional auxiliary coordinates which span
                # this axis and have the same values in their data
                # array and bounds.
                c = f.auxiliary_coordinates(
                    filter_by_axis=(axis,), axis_mode="exact", todict=True
                )
                for key, aux in c.items():
                    logger.info(f"key = {key}")  # pragma: no cover

                    d = aux[0]

                    # TODODASK: remove once dask. For some reason,
                    # without this we now get LAMA related failures in
                    # Partition.nbytes ...
                    _ = aux.dtype

                    if aux.has_bounds() or (aux[:-1] != aux[1:]).any():
                        logger.info(
                            f"    Removing {aux.construct_type} {key!r}"
                        )  # pragma: no cover

                        f.del_construct(key)
                    else:
                        # Change the data array for this auxiliary
                        # coordinate
                        aux.set_data(d.data, copy=False)
                        if d.has_bounds():
                            aux.bounds.set_data(d.bounds.data, copy=False)

                # Reset the axis size
                f.domain_axes(todict=True)[axis].set_size(1)
                logger.info(
                    f"Changing axis size to 1: {axis}"
                )  # pragma: no cover

                dim = f.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if dim is None:
                    continue

                # Create a new dimension coordinate for this axis
                if dim.has_bounds():
                    bounds_data = [dim.bounds.datum(0), dim.bounds.datum(-1)]
                else:
                    bounds_data = [dim.datum(0), dim.datum(-1)]

                units = dim.Units

                if coordinate == "min":
                    coordinate = "minimum"
                    print(
                        "WARNING: coordinate='min' has been deprecated. "
                        "Use coordinate='minimum' instead."
                    )
                elif coordinate == "max":
                    coordinate = "maximum"
                    print(
                        "WARNING: coordinate='max' has been deprecated. "
                        "Use coordinate='maximum' instead."
                    )

                if coordinate == "mid_range":
                    data = Data(
                        [(bounds_data[0] + bounds_data[1]) * 0.5], units=units
                    )
                elif coordinate == "minimum":
                    data = dim.data.min()
                elif coordinate == "maximum":
                    data = dim.data.max()
                else:
                    raise ValueError(
                        "Can't collapse: Bad parameter value: "
                        f"coordinate={coordinate!r}"
                    )

                bounds = self._Bounds(data=Data([bounds_data], units=units))

                dim.set_data(data, copy=False)
                dim.set_bounds(bounds, copy=False)

            # --------------------------------------------------------
            # Update the cell methods
            # --------------------------------------------------------
            if _update_cell_methods:
                f._update_cell_methods(
                    method,
                    domain_axes=collapse_axes,
                    input_axes=axes_in,
                    within=within,
                    over=over,
                    verbose=verbose,
                )

        # ------------------------------------------------------------
        # Return the collapsed field (or the classification array)
        # ------------------------------------------------------------
        return f


    def _update_cell_methods(
        self,
        method=None,
        domain_axes=None,
        input_axes=None,
        within=None,
        over=None,
        verbose=None,
    ):
        """Update the cell methods.

        :Parameters:

            method: `str`

            domain_axes: `Constructs` or `dict`

            {{verbose: `int` or `str` or `None`, optional}}

        :Returns:

            `None`

        """
        original_cell_methods = self.cell_methods(todict=True)  # .ordered()
        logger.info("  Update cell methods:")  # pragma: no cover
        logger.info(
            "    Original cell methods = {}".format(original_cell_methods)
        )  # pragma: no cover
        logger.info(
            "    method        = {!r}".format(method)
        )  # pragma: no cover
        logger.info(
            "    within        = {!r}".format(within)
        )  # pragma: no cover
        logger.info(
            "    over          = {!r}".format(over)
        )  # pragma: no cover

        if input_axes and tuple(input_axes) == ("area",):
            axes = ("area",)
        else:
            axes = tuple(domain_axes)

        comment = None

        method = _collapse_cell_methods.get(method, method)

        cell_method = CellMethod(axes=axes, method=method)
        if within:
            cell_method.set_qualifier("within", within)
        elif over:
            cell_method.set_qualifier("over", over)

        if comment:
            cell_method.set_qualifier("comment", comment)

        if original_cell_methods:
            # There are already some cell methods
            if len(domain_axes) == 1:
                # Only one axis has been collapsed
                key, original_domain_axis = tuple(domain_axes.items())[0]

                lastcm = tuple(original_cell_methods.values())[-1]
                lastcm_method = _collapse_cell_methods.get(
                    lastcm.get_method(None), lastcm.get_method(None)
                )

                if (
                    original_domain_axis.get_size()
                    == self.domain_axes(todict=True)[key].get_size()
                ):
                    if (
                        lastcm.get_axes(None) == axes
                        and lastcm_method == method
                        and lastcm_method
                        in (
                            "mean",
                            "maximum",
                            "minimum",
                            "point",
                            "sum",
                            "median",
                            "mode",
                            "minimum_absolute_value",
                            "maximum_absolute_value",
                        )
                        and not lastcm.get_qualifier("within", None)
                        and not lastcm.get_qualifier("over", None)
                    ):
                        # It was a null collapse (i.e. the method is
                        # the same as the last one and the size of the
                        # collapsed axis hasn't changed).
                        if within:
                            lastcm.within = within
                        elif over:
                            lastcm.over = over

                        cell_method = None

        if cell_method is not None:
            self.set_construct(cell_method)

        logger.info(
            f"    Modified cell methods = {self.cell_methods()}"
        )  # pragma: no cover


    @_manage_log_level_via_verbosity
    def _collapse_grouped(
        self,
        method,
        axis,
        within=None,
        over=None,
        within_days=None,
        within_years=None,
        over_days=None,
        over_years=None,
        group=None,
        group_span=None,
        group_contiguous=False,
        mtol=None,
        ddof=None,
        regroup=None,
        coordinate=None,
        measure=False,
        weights=None,
        squeeze=None,
        group_by=None,
        axis_in=None,
        verbose=None,
    ):
        """Implements a grouped collapse on a field.

        A grouped collapse is one for which an axis is not collapsed
        completely to size 1.

        :Parameters:

            method: `str`
                See `collapse` for details.

            measure: `bool`, optional
                See `collapse` for details.

            over: `str`
                See `collapse` for details.

            within: `str`
                See `collapse` for details.

        """

        def _ddddd(
            classification,
            n,
            lower,
            upper,
            increasing,
            coord,
            group_by_coords,
            extra_condition,
        ):
            """Returns configuration for a general collapse.

            :Parameter:

                extra_condition: `Query`

            :Returns:

                `numpy.ndarray`, `int`, date-time, date-time

            """
            if group_by_coords:
                q = ge(lower) & lt(upper)
            else:
                q = ge(lower, attr="lower_bounds") & le(
                    upper, attr="upper_bounds"
                )

            if extra_condition:
                q &= extra_condition

            index = q.evaluate(coord).array
            classification[index] = n

            if increasing:
                lower = upper
            else:
                upper = lower

            n += 1

            return classification, n, lower, upper

        def _time_interval(
            classification,
            n,
            coord,
            interval,
            lower,
            upper,
            lower_limit,
            upper_limit,
            group_by,
            extra_condition=None,
        ):
            """Prepares for a collapse where the group is a
            TimeDuration.

            :Parameters:

                classification: `numpy.ndarray`

                n: `int`

                coord: `DimensionCoordinate`

                interval: `TimeDuration`

                lower: date-time object

                upper: date-time object

                lower_limit: `datetime`

                upper_limit: `datetime`

                group_by: `str`

                extra_condition: `Query`, optional

            :Returns:

                (`numpy.ndarray`, `int`)

            """
            group_by_coords = group_by == "coords"

            if coord.increasing:
                # Increasing dimension coordinate
                lower, upper = interval.bounds(lower)
                while lower <= upper_limit:
                    lower, upper = interval.interval(lower)
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        True,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )
            else:
                # Decreasing dimension coordinate
                lower, upper = interval.bounds(upper)
                while upper >= lower_limit:
                    lower, upper = interval.interval(upper, end=True)
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        False,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )

            return classification, n

        def _time_interval_over(
            classification,
            n,
            coord,
            interval,
            lower,
            upper,
            lower_limit,
            upper_limit,
            group_by,
            extra_condition=None,
        ):
            """Prepares for a collapse over some TimeDuration.

            :Parameters:

                classification: `numpy.ndarray`

                n: `int`

                coord: `DimensionCoordinate`

                interval: `TimeDuration`

                lower: date-time

                upper: date-time

                lower_limit: date-time

                upper_limit: date-time

                group_by: `str`

                extra_condition: `Query`, optional

            :Returns:

                (`numpy.ndarray`, `int`)

            """
            group_by_coords = group_by == "coords"

            if coord.increasing:
                # Increasing dimension coordinate
                # lower, upper = interval.bounds(lower)
                upper = interval.interval(upper)[1]
                while lower <= upper_limit:
                    lower, upper = interval.interval(lower)
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        True,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )
            else:
                # Decreasing dimension coordinate
                # lower, upper = interval.bounds(upper)
                lower = interval.interval(upper, end=True)[0]
                while upper >= lower_limit:
                    lower, upper = interval.interval(upper, end=True)
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        False,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )

            return classification, n

        def _data_interval(
            classification,
            n,
            coord,
            interval,
            lower,
            upper,
            lower_limit,
            upper_limit,
            group_by,
            extra_condition=None,
        ):
            """Prepares for a collapse where the group is a data
            interval.

            :Returns:

                `numpy.ndarray`, `int`

            """
            group_by_coords = group_by == "coords"

            if coord.increasing:
                # Increasing dimension coordinate
                lower = lower.squeeze()
                while lower <= upper_limit:
                    upper = lower + interval
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        True,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )
            else:
                # Decreasing dimension coordinate
                upper = upper.squeeze()
                while upper >= lower_limit:
                    lower = upper - interval
                    classification, n, lower, upper = _ddddd(
                        classification,
                        n,
                        lower,
                        upper,
                        False,
                        coord,
                        group_by_coords,
                        extra_condition,
                    )

            return classification, n

        def _selection(
            classification,
            n,
            coord,
            selection,
            parameter,
            extra_condition=None,
            group_span=None,
            within=False,
        ):
            """Processes a group selection.

            :Parameters:

                classification: `numpy.ndarray`

                n: `int`

                coord: `DimensionCoordinate`

                selection: sequence of `Query`

                parameter: `str`
                    The name of the `cf.Field.collapse` parameter which
                    defined *selection*. This is used in error messages.

                    *Parameter example:*
                      ``parameter='within_years'``

                extra_condition: `Query`, optional

            :Returns:

                `numpy.ndarray`, `int`

            """
            # Create an iterator for stepping through each Query in
            # the selection sequence
            try:
                iterator = iter(selection)
            except TypeError:
                raise ValueError(
                    "Can't collapse: Bad parameter value: {}={!r}".format(
                        parameter, selection
                    )
                )

            for condition in iterator:
                if not isinstance(condition, Query):
                    raise ValueError(
                        "Can't collapse: {} sequence contains a non-{} "
                        "object: {!r}".format(
                            parameter, Query.__name__, condition
                        )
                    )

                if extra_condition is not None:
                    condition &= extra_condition

                boolean_index = condition.evaluate(coord).array

                classification[boolean_index] = n
                n += 1

            #                if group_span is not None:
            #                    x = numpy_where(classification==n)[0]
            #                    for i in range(1, max(1, int(float(len(x))/group_span))):
            #                        n += 1
            #                        classification[x[i*group_span:(i + 1)*group_span]] = n
            #                n += 1

            return classification, n

        def _discern_runs(classification, within=False):
            """Processes a group classification.

            :Parameters:

                classification: `numpy.ndarray`

            :Returns:

                `numpy.ndarray`

            """
            x = numpy_where(numpy_diff(classification))[0] + 1
            if not x.size:
                if classification[0] >= 0:
                    classification[:] = 0

                return classification

            if classification[0] >= 0:
                classification[0 : x[0]] = 0

            n = 1
            for i, j in zip(x[:-1], x[1:]):
                if classification[i] >= 0:
                    classification[i:j] = n
                    n += 1

            if classification[x[-1]] >= 0:
                classification[x[-1] :] = n
                n += 1

            return classification

        def _discern_runs_within(classification, coord):
            """Processes group classification for a 'within'
            collapse."""
            size = classification.size
            if size < 2:
                return classification

            n = classification.max() + 1

            start = 0
            for i, c in enumerate(classification[: size - 1]):
                if c < 0:
                    continue

                if not coord[i : i + 2].contiguous(overlap=False):
                    classification[start : i + 1] = n
                    start = i + 1
                    n += 1

            return classification

        def _tyu(coord, group_by, time_interval):
            """Returns bounding values and limits for a general
            collapse.

            :Parameters:

                coord: `DimensionCoordinate`
                    The dimension coordinate construct associated with
                    the collapse.

                group_by: `str`
                    As for the *group_by* parameter of the `collapse` method.

                time_interval: `bool`
                    If True then then return a tuple of date-time
                    objects. If False return a tuple of `Data` objects.

            :Returns:

                `tuple`
                    A tuple of 4 `Data` object or, if *time_interval* is
                    True, a tuple of 4 date-time objects.

            """
            bounds = coord.get_bounds(None)
            if bounds is not None:
                lower_bounds = coord.lower_bounds
                upper_bounds = coord.upper_bounds
                lower = lower_bounds[0]
                upper = upper_bounds[0]
                lower_limit = lower_bounds[-1]
                upper_limit = upper_bounds[-1]
            elif group_by == "coords":
                if coord.increasing:
                    lower = coord.data[0]
                    upper = coord.data[-1]
                else:
                    lower = coord.data[-1]
                    upper = coord.data[0]

                lower_limit = lower
                upper_limit = upper
            else:
                raise ValueError(
                    "Can't collapse: {!r} coordinate bounds are required "
                    "with group_by={!r}".format(coord.identity(), group_by)
                )

            if time_interval:
                units = coord.Units
                if units.isreftime:
                    lower = lower.datetime_array[0]
                    upper = upper.datetime_array[0]
                    lower_limit = lower_limit.datetime_array[0]
                    upper_limit = upper_limit.datetime_array[0]
                elif not units.istime:
                    raise ValueError(
                        "Can't group by {} when coordinates have units "
                        "{!r}".format(
                            TimeDuration.__class__.__name__, coord.Units
                        )
                    )

            return (lower, upper, lower_limit, upper_limit)

        def _group_weights(weights, iaxis, index):
            """Subspaces weights components.

            :Parameters:

                weights: `dict` or `None`

                iaxis: `int`

                index: `list`

            :Returns:

                `dict` or `None`

            **Examples:**

            >>> print(weights)
            None
            >>> print(_group_weights(weights, 2, [2, 3, 40]))
            None
            >>> print(_group_weights(weights, 1, slice(2, 56)))
            None

            >>> weights

            >>> _group_weights(weights, 2, [2, 3, 40])

            >>> _group_weights(weights, 1, slice(2, 56))

            """
            if not isinstance(weights, dict):
                return weights

            weights = weights.copy()
            for iaxes, value in weights.items():
                if iaxis in iaxes:
                    indices = [slice(None)] * len(iaxes)
                    indices[iaxes.index(iaxis)] = index
                    weights[iaxes] = value[tuple(indices)]
                    break

            return weights

        # START OF MAIN CODE

        logger.info("    Grouped collapse:")  # pragma: no cover
        logger.info(
            "        method            = {!r}".format(method)
        )  # pragma: no cover
        logger.info(
            "        axis_in           = {!r}".format(axis_in)
        )  # pragma: no cover
        logger.info(
            "        axis              = {!r}".format(axis)
        )  # pragma: no cover
        logger.info(
            "        over              = {!r}".format(over)
        )  # pragma: no cover
        logger.info(
            "        over_days         = {!r}".format(over_days)
        )  # pragma: no cover
        logger.info(
            "        over_years        = {!r}".format(over_years)
        )  # pragma: no cover
        logger.info(
            "        within            = {!r}".format(within)
        )  # pragma: no cover
        logger.info(
            "        within_days       = {!r}".format(within_days)
        )  # pragma: no cover
        logger.info(
            "        within_years      = {!r}".format(within_years)
        )  # pragma: no cover
        logger.info(
            "        regroup           = {!r}".format(regroup)
        )  # pragma: no cover
        logger.info(
            "        group             = {!r}".format(group)
        )  # pragma: no cover
        logger.info(
            "        group_span        = {!r}".format(group_span)
        )  # pragma: no cover
        logger.info(
            "        group_contiguous  = {!r}".format(group_contiguous)
        )  # pragma: no cover

        # Size of uncollapsed axis
        axis_size = self.domain_axes(todict=True)[axis].get_size()
        # Integer position of collapse axis
        iaxis = self.get_data_axes().index(axis)

        fl = []

        # If group, rolling window, classification, etc, do something
        # special for size one axes - either return unchanged
        # (possibly mofiying cell methods with , e.g, within_days', or
        # raising an exception for 'can't match', I suppose.

        classification = None

        if group is not None:
            if within is not None or over is not None:
                raise ValueError(
                    "Can't set 'group' parameter for a climatological "
                    "collapse"
                )

            if isinstance(group, numpy_ndarray):
                classification = numpy_squeeze(group.copy())

                if classification.dtype.kind != "i":
                    raise ValueError(
                        "Can't group by numpy array of type {}".format(
                            classification.dtype.name
                        )
                    )
                elif classification.shape != (axis_size,):
                    raise ValueError(
                        "Can't group by numpy array with incorrect "
                        "shape: {}".format(classification.shape)
                    )

                # Set group to None
                group = None

        if group is not None:
            if isinstance(group, Query):
                group = (group,)

            if isinstance(group, int):
                # ----------------------------------------------------
                # E.g. group=3
                # ----------------------------------------------------
                coord = None
                classification = numpy_empty((axis_size,), int)

                start = 0
                end = group
                n = 0
                while start < axis_size:
                    classification[start:end] = n
                    start = end
                    end += group
                    n += 1

                if group_span is True or group_span is None:
                    # Use the group definition as the group span
                    group_span = group

            elif isinstance(group, TimeDuration):
                # ----------------------------------------------------
                # E.g. group=cf.M()
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None:
                    raise ValueError("dddddd siduhfsuildfhsuil dhfdui TODO")

                classification = numpy_empty((axis_size,), int)
                classification.fill(-1)

                lower, upper, lower_limit, upper_limit = _tyu(
                    coord, group_by, True
                )

                classification, n = _time_interval(
                    classification,
                    0,
                    coord=coord,
                    interval=group,
                    lower=lower,
                    upper=upper,
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                    group_by=group_by,
                )

                if group_span is True or group_span is None:
                    # Use the group definition as the group span
                    group_span = group

            elif isinstance(group, Data):
                # ----------------------------------------------------
                # Chunks of
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None:
                    raise ValueError("TODO asdas 4444444  dhfdui ")

                if coord.Units.isreftime:
                    raise ValueError(
                        "Can't group a reference-time axis with {!r}. Use "
                        "a TimeDuration instance instead.".format(group)
                    )

                if group.size != 1:
                    raise ValueError(
                        "Group must have only one element: "
                        "{!r}".format(group)
                    )

                if group.Units and not group.Units.equivalent(coord.Units):
                    raise ValueError(
                        "Can't group by {!r} when coordinates have "
                        "non-equivalent units {!r}".format(group, coord.Units)
                    )

                classification = numpy_empty((axis_size,), int)
                classification.fill(-1)

                group = group.squeeze()

                lower, upper, lower_limit, upper_limit = _tyu(
                    coord, group_by, False
                )

                classification, n = _data_interval(
                    classification,
                    0,
                    coord=coord,
                    interval=group,
                    lower=lower,
                    upper=upper,
                    lower_limit=lower_limit,
                    upper_limit=upper_limit,
                    group_by=group_by,
                )

                if group_span is True or group_span is None:
                    # Use the group definition as the group span
                    group_span = group

            else:
                # ----------------------------------------------------
                # E.g. group=[cf.month(4), cf.month(cf.wi(9, 11))]
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None:
                    coord = self.auxiliary_coordinate(
                        filter_by_axis=(axis,), axis_mode="exact", default=None
                    )
                    if coord is None:
                        raise ValueError("asdad8777787 TODO")

                classification = numpy_empty((axis_size,), int)
                classification.fill(-1)

                classification, n = _selection(
                    classification,
                    0,
                    coord=coord,
                    selection=group,
                    parameter="group",
                )

                classification = _discern_runs(classification)

                if group_span is None:
                    group_span = False
                elif group_span is True:
                    raise ValueError(
                        "Can't collapse: Can't set group_span=True when "
                        f"group={group!r}"
                    )

        if classification is None:
            if over == "days":
                # ----------------------------------------------------
                # Over days
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None or not coord.Units.isreftime:
                    raise ValueError(
                        "Reference-time dimension coordinates are required "
                        "for an 'over days' collapse"
                    )

                if not coord.has_bounds():
                    raise ValueError(
                        "Reference-time dimension coordinate bounds are "
                        "required for an 'over days' collapse"
                    )

                cell_methods = self.cell_methods(todict=True)
                w = [
                    cm.get_qualifier("within", None)
                    for cm in cell_methods.values()
                ]
                if "days" not in w:
                    raise ValueError(
                        "An 'over days' collapse must come after a "
                        "'within days' cell method"
                    )

                # Parse the over_days parameter
                if isinstance(over_days, Query):
                    over_days = (over_days,)
                elif isinstance(over_days, TimeDuration):
                    if over_days.Units.istime and over_days < Data(1, "day"):
                        raise ValueError(
                            f"Bad parameter value: over_days={over_days!r}"
                        )

                coordinate = "minimum"

                classification = numpy_empty((axis_size,), int)
                classification.fill(-1)

                if isinstance(over_days, TimeDuration):
                    _, _, lower_limit, upper_limit = _tyu(
                        coord, "bounds", True
                    )

                bounds = coord.bounds
                lower_bounds = coord.lower_bounds.datetime_array
                upper_bounds = coord.upper_bounds.datetime_array

                HMS0 = None

                n = 0
                for lower, upper in zip(lower_bounds, upper_bounds):
                    HMS_l = (
                        eq(lower.hour, attr="hour")
                        & eq(lower.minute, attr="minute")
                        & eq(lower.second, attr="second")
                    ).addattr("lower_bounds")
                    HMS_u = (
                        eq(upper.hour, attr="hour")
                        & eq(upper.minute, attr="minute")
                        & eq(upper.second, attr="second")
                    ).addattr("upper_bounds")
                    HMS = HMS_l & HMS_u

                    if not HMS0:
                        HMS0 = HMS
                    elif HMS.equals(HMS0):
                        # We've got repeat of the first cell, which
                        # means that we must have now classified all
                        # cells. Therefore we can stop.
                        break

                    logger.info(
                        "          HMS  = {!r}".format(HMS)
                    )  # pragma: no cover

                    if over_days is None:
                        # --------------------------------------------
                        # over_days=None
                        # --------------------------------------------
                        # Over all days
                        index = HMS.evaluate(coord).array
                        classification[index] = n
                        n += 1
                    elif isinstance(over_days, TimeDuration):
                        # --------------------------------------------
                        # E.g. over_days=cf.M()
                        # --------------------------------------------
                        classification, n = _time_interval_over(
                            classification,
                            n,
                            coord=coord,
                            interval=over_days,
                            lower=lower,
                            upper=upper,
                            lower_limit=lower_limit,
                            upper_limit=upper_limit,
                            group_by="bounds",
                            extra_condition=HMS,
                        )
                    else:
                        # --------------------------------------------
                        # E.g. over_days=[cf.month(cf.wi(4, 9))]
                        # --------------------------------------------
                        classification, n = _selection(
                            classification,
                            n,
                            coord=coord,
                            selection=over_days,
                            parameter="over_days",
                            extra_condition=HMS,
                        )

            elif over == "years":
                # ----------------------------------------------------
                # Over years
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None or not coord.Units.isreftime:
                    raise ValueError(
                        "Reference-time dimension coordinates are required "
                        "for an 'over years' collapse"
                    )

                bounds = coord.get_bounds(None)
                if bounds is None:
                    raise ValueError(
                        "Reference-time dimension coordinate bounds are "
                        "required for an 'over years' collapse"
                    )

                cell_methods = self.cell_methods(todict=True)
                w = [
                    cm.get_qualifier("within", None)
                    for cm in cell_methods.values()
                ]
                o = [
                    cm.get_qualifier("over", None)
                    for cm in cell_methods.values()
                ]
                if "years" not in w and "days" not in o:
                    raise ValueError(
                        "An 'over years' collapse must come after a "
                        "'within years' or 'over days' cell method"
                    )

                # Parse the over_years parameter
                if isinstance(over_years, Query):
                    over_years = (over_years,)
                elif isinstance(over_years, TimeDuration):
                    if over_years.Units.iscalendartime:
                        over_years.Units = Units("calendar_years")
                        if not over_years.isint or over_years < 1:
                            raise ValueError(
                                "over_years is not a whole number of "
                                "calendar years: {!r}".format(over_years)
                            )
                    else:
                        raise ValueError(
                            "over_years is not a whole number of calendar "
                            f"years: {over_years!r}"
                        )

                coordinate = "minimum"

                classification = numpy_empty((axis_size,), int)
                classification.fill(-1)

                if isinstance(over_years, TimeDuration):
                    _, _, lower_limit, upper_limit = _tyu(
                        coord, "bounds", True
                    )

                lower_bounds = coord.lower_bounds.datetime_array
                upper_bounds = coord.upper_bounds.datetime_array
                mdHMS0 = None

                n = 0
                for lower, upper in zip(lower_bounds, upper_bounds):
                    mdHMS_l = (
                        eq(lower.month, attr="month")
                        & eq(lower.day, attr="day")
                        & eq(lower.hour, attr="hour")
                        & eq(lower.minute, attr="minute")
                        & eq(lower.second, attr="second")
                    ).addattr("lower_bounds")
                    mdHMS_u = (
                        eq(upper.month, attr="month")
                        & eq(upper.day, attr="day")
                        & eq(upper.hour, attr="hour")
                        & eq(upper.minute, attr="minute")
                        & eq(upper.second, attr="second")
                    ).addattr("upper_bounds")
                    mdHMS = mdHMS_l & mdHMS_u

                    if not mdHMS0:
                        # Keep a record of the first cell
                        mdHMS0 = mdHMS
                        logger.info(
                            f"        mdHMS0 = {mdHMS0!r}"
                        )  # pragma: no cover
                    elif mdHMS.equals(mdHMS0):
                        # We've got repeat of the first cell, which
                        # means that we must have now classified all
                        # cells. Therefore we can stop.
                        break

                    logger.info(
                        f"        mdHMS  = {mdHMS!r}"
                    )  # pragma: no cover

                    if over_years is None:
                        # --------------------------------------------
                        # over_years=None
                        # --------------------------------------------
                        # Over all years
                        index = mdHMS.evaluate(coord).array
                        classification[index] = n
                        n += 1
                    elif isinstance(over_years, TimeDuration):
                        # --------------------------------------------
                        # E.g. over_years=cf.Y(2)
                        # --------------------------------------------
                        classification, n = _time_interval_over(
                            classification,
                            n,
                            coord=coord,
                            interval=over_years,
                            lower=lower,
                            upper=upper,
                            lower_limit=lower_limit,
                            upper_limit=upper_limit,
                            group_by="bounds",
                            extra_condition=mdHMS,
                        )
                    else:
                        # --------------------------------------------
                        # E.g. over_years=cf.year(cf.lt(2000))
                        # --------------------------------------------
                        classification, n = _selection(
                            classification,
                            n,
                            coord=coord,
                            selection=over_years,
                            parameter="over_years",
                            extra_condition=mdHMS,
                        )

            elif within == "days":
                # ----------------------------------------------------
                # Within days
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None or not coord.Units.isreftime:
                    raise ValueError(
                        "Reference-time dimension coordinates are required "
                        "for an 'over years' collapse"
                    )

                bounds = coord.get_bounds(None)
                if bounds is None:
                    raise ValueError(
                        "Reference-time dimension coordinate bounds are "
                        "required for a 'within days' collapse"
                    )

                classification = numpy_empty((axis_size,), int)
                classification.fill(-1)

                # Parse the within_days parameter
                if isinstance(within_days, Query):
                    within_days = (within_days,)
                elif isinstance(within_days, TimeDuration):
                    if (
                        within_days.Units.istime
                        and TimeDuration(24, "hours") % within_days
                    ):
                        # % Data(1, 'day'): # % within_days:
                        raise ValueError(
                            f"Can't collapse: within_days={within_days!r} "
                            "is not an exact factor of 1 day"
                        )

                if isinstance(within_days, TimeDuration):
                    # ------------------------------------------------
                    # E.g. within_days=cf.h(6)
                    # ------------------------------------------------
                    lower, upper, lower_limit, upper_limit = _tyu(
                        coord, "bounds", True
                    )

                    classification, n = _time_interval(
                        classification,
                        0,
                        coord=coord,
                        interval=within_days,
                        lower=lower,
                        upper=upper,
                        lower_limit=lower_limit,
                        upper_limit=upper_limit,
                        group_by=group_by,
                    )

                    if group_span is True or group_span is None:
                        # Use the within_days definition as the group
                        # span
                        group_span = within_days

                else:
                    # ------------------------------------------------
                    # E.g. within_days=cf.hour(cf.lt(12))
                    # ------------------------------------------------
                    classification, n = _selection(
                        classification,
                        0,
                        coord=coord,
                        selection=within_days,
                        parameter="within_days",
                    )

                    classification = _discern_runs(classification)

                    classification = _discern_runs_within(
                        classification, coord
                    )

                    if group_span is None:
                        group_span = False
                    elif group_span is True:
                        raise ValueError(
                            "Can't collapse: Can't set group_span=True when "
                            f"within_days={within_days!r}"
                        )

            elif within == "years":
                # ----------------------------------------------------
                # Within years
                # ----------------------------------------------------
                coord = self.dimension_coordinate(
                    filter_by_axis=(axis,), default=None
                )
                if coord is None or not coord.Units.isreftime:
                    raise ValueError(
                        "Can't collapse: Reference-time dimension "
                        'coordinates are required for a "within years" '
                        "collapse"
                    )

                if not coord.has_bounds():
                    raise ValueError(
                        "Can't collapse: Reference-time dimension coordinate "
                        'bounds are required for a "within years" collapse'
                    )

                classification = numpy_empty((axis_size,), int)
                classification.fill(-1)

                # Parse within_years
                if isinstance(within_years, Query):
                    within_years = (within_years,)
                elif within_years is None:
                    raise ValueError(
                        "Must set the within_years parameter for a "
                        '"within years" climatalogical time collapse'
                    )

                if isinstance(within_years, TimeDuration):
                    # ------------------------------------------------
                    # E.g. within_years=cf.M()
                    # ------------------------------------------------
                    lower, upper, lower_limit, upper_limit = _tyu(
                        coord, "bounds", True
                    )

                    classification, n = _time_interval(
                        classification,
                        0,
                        coord=coord,
                        interval=within_years,
                        lower=lower,
                        upper=upper,
                        lower_limit=lower_limit,
                        upper_limit=upper_limit,
                        group_by=group_by,
                    )

                    if group_span is True or group_span is None:
                        # Use the within_years definition as the group
                        # span
                        group_span = within_years

                else:
                    # ------------------------------------------------
                    # E.g. within_years=cf.season()
                    # ------------------------------------------------
                    classification, n = _selection(
                        classification,
                        0,
                        coord=coord,
                        selection=within_years,
                        parameter="within_years",
                        within=True,
                    )

                    classification = _discern_runs(classification, within=True)

                    classification = _discern_runs_within(
                        classification, coord
                    )

                    if group_span is None:
                        group_span = False
                    elif group_span is True:
                        raise ValueError(
                            "Can't collapse: Can't set group_span=True when "
                            "within_years={!r}".format(within_years)
                        )

            elif over is not None:
                raise ValueError(
                    f"Can't collapse: Bad 'over' syntax: {over!r}"
                )

            elif within is not None:
                raise ValueError(
                    f"Can't collapse: Bad 'within' syntax: {within!r}"
                )

        if classification is not None:
            # ---------------------------------------------------------
            # Collapse each group
            # ---------------------------------------------------------
            logger.info(
                f"        classification    = {classification}"
            )  # pragma: no cover

            unique = numpy_unique(classification)
            unique = unique[numpy_where(unique >= 0)[0]]
            unique.sort()

            ignore_n = -1
            for u in unique:
                index = numpy_where(classification == u)[0].tolist()

                pc = self.subspace(**{axis: index})

                # ----------------------------------------------------
                # Ignore groups that don't meet the specified criteria
                # ----------------------------------------------------
                if over is None:
                    coord = pc.coordinate(axis_in, default=None)

                    if group_span is not False:
                        if isinstance(group_span, int):
                            if (
                                pc.domain_axes(todict=True)[axis].get_size()
                                != group_span
                            ):
                                classification[index] = ignore_n
                                ignore_n -= 1
                                continue
                        else:
                            if coord is None:
                                raise ValueError(
                                    "Can't collapse: Need an unambiguous 1-d "
                                    "coordinate construct when "
                                    f"group_span={group_span!r}"
                                )

                            bounds = coord.get_bounds(None)
                            if bounds is None:
                                raise ValueError(
                                    "Can't collapse: Need unambiguous 1-d "
                                    "coordinate cell bounds when "
                                    f"group_span={group_span!r}"
                                )

                            lb = bounds[0, 0].get_data(_fill_value=False)
                            ub = bounds[-1, 1].get_data(_fill_value=False)
                            if coord.T:
                                lb = lb.datetime_array.item()
                                ub = ub.datetime_array.item()

                            if not coord.increasing:
                                lb, ub = ub, lb
                            if group_span + lb != ub:
                                # The span of this group is not the
                                # same as group_span, so don't
                                # collapse it.
                                classification[index] = ignore_n
                                ignore_n -= 1
                                continue

                    if (
                        group_contiguous
                        and coord is not None
                        and coord.has_bounds()
                        and not coord.bounds.contiguous(
                            overlap=(group_contiguous == 2)
                        )
                    ):
                        # This group is not contiguous, so don't
                        # collapse it.
                        classification[index] = ignore_n
                        ignore_n -= 1
                        continue

                if regroup:
                    continue

                # ----------------------------------------------------
                # Still here? Then collapse the group
                # ----------------------------------------------------
                w = _group_weights(weights, iaxis, index)
                logger.info(
                    f"        Collapsing group {u}:"
                )  # pragma: no cover

                fl.append(
                    pc.collapse(
                        method,
                        axis,
                        weights=w,
                        measure=measure,
                        mtol=mtol,
                        ddof=ddof,
                        coordinate=coordinate,
                        squeeze=False,
                        inplace=True,
                        _create_zero_size_cell_bounds=True,
                        _update_cell_methods=False,
                    )
                )

            if regroup:
                # return the numpy array
                return classification

        elif regroup:
            raise ValueError("Can't return classification 2453456 ")

        # Still here?
        if not fl:
            c = "contiguous " if group_contiguous else ""
            s = f" spanning {group_span}" if group_span is not False else ""
            if within is not None:
                s = f" within {within}{s}"

            raise ValueError(
                f"Can't collapse: No {c}groups{s} were identified"
            )

        if len(fl) == 1:
            f = fl[0]
        else:
            # Hack to fix missing bounds!
            for g in fl:
                try:
                    c = g.dimension_coordinate(
                        filter_by_axis=(axis,), default=None
                    )
                    if not c.has_bounds():
                        c.set_bounds(c.create_bounds())
                except Exception:
                    pass

            # --------------------------------------------------------
            # Sort the list of collapsed fields
            # --------------------------------------------------------
            if (
                coord is not None
                and coord.construct_type == "dimension_coordinate"
            ):
                fl.sort(
                    key=lambda g: g.dimension_coordinate(
                        filter_by_axis=(axis,)
                    ).datum(0),
                    reverse=coord.decreasing,
                )

            # --------------------------------------------------------
            # Concatenate the partial collapses
            # --------------------------------------------------------
            try:
                f = self.concatenate(fl, axis=iaxis, _preserve=False)
            except ValueError as error:
                raise ValueError(f"Can't collapse: {error}")

        if squeeze and f.domain_axes(todict=True)[axis].get_size() == 1:
            # Remove a totally collapsed axis from the field's
            # data array
            f.squeeze(axis, inplace=True)

        # ------------------------------------------------------------
        # Return the collapsed field
        # ------------------------------------------------------------
        self.__dict__ = f.__dict__
        logger.info("    End of grouped collapse")  # pragma: no cover

        return self

