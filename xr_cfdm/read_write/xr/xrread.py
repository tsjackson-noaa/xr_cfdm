"""Implements parsing metadata on file load.

XRReadBase is based on cfdm's ``cfdm.read_write.netcdf.NetCDFRead``.
XRRead is based on cf-python's ``cf.read_write.netcdf.NetCDFRead``.
"""

import ast
import json
import logging
import operator
import os
import re
import struct
import subprocess
import tempfile

from collections import OrderedDict
from copy import deepcopy
from distutils.version import LooseVersion
from functools import reduce

import numpy
import netCDF4
import cfdm
from cfdm.functions import is_log_level_debug
import netcdf_flattener

from cf.constants import _file_to_fh
from cf.functions import pathjoin, dirname
from cf.units import Units

logger = logging.getLogger(__name__)

_cached_temporary_files = {}

_flattener_separator = netcdf_flattener._Flattener._Flattener__new_separator


class XRReadBase(cfdm.read_write.netcdf.NetCDFRead):
    """Overwrite the subset of methods in cfdm NetCDFRead which call methods in
    netCDF4, replacing them with xarray methods.
    """

    def file_open(self, filename, flatten=True, verbose=None):
        """Open the netCDf file for reading.

        If the file has hierarchical groups then a flattened version of it
        is returned, and the original grouped file remains open.

        .. versionadded:: (cfdm) 1.7.0

        :Paramters:

            filename: `str`
                As for the *filename* parameter for initialising a
                `netCDF.Dataset` instance.

            flatten: `bool`, optional
                If False then do not flatten a grouped file. Ignored if
                the file has no groups.

                .. versionadded:: (cfdm) 1.8.6

        :Returns:

            `netCDF4.Dataset`
                A `netCDF4.Dataset` object for the file.

        **Examples:**

        >>> r.file_open('file.nc')

        """
        try:
            nc = netCDF4.Dataset(filename, "r")
        except RuntimeError as error:
            raise RuntimeError(f"{error}: {filename}")

        # ------------------------------------------------------------
        # If the file has a group structure then flatten it (CF>=1.8)
        # ------------------------------------------------------------
        g = self.read_vars

        if flatten and nc.groups:
            # Create a diskless, non-persistent container for the
            # flattened file
            flat_file = tempfile.NamedTemporaryFile(
                mode="wb",
                dir=tempfile.gettempdir(),
                prefix="cfdm_flat_",
                suffix=".nc",
                delete=True,
            )

            flat_nc = netCDF4.Dataset(
                flat_file, "w", diskless=True, persist=False
            )
            flat_nc.set_fill_off()

            # Flatten the file
            netcdf_flattener.flatten(
                nc, flat_nc, lax_mode=True, _copy_data=False
            )

            # Store the original grouped file. This is primarily
            # because the unlimited dimensions in the flattened
            # dataset have size 0, since it contains no
            # data. (v1.8.8.1)
            g["nc_grouped"] = nc

            nc = flat_nc

            g["has_groups"] = True
            g["flat_files"].append(flat_file)

        g["nc"] = nc
        return nc

    @cfdm.decorators._manage_log_level_via_verbosity
    def read(
        self,
        filename,
        extra=None,
        default_version=None,
        external=None,
        extra_read_vars=None,
        _scan_only=False,
        verbose=None,
        mask=True,
        warnings=True,
        warn_valid=False,
    ):
        """Reads a netCDF dataset from file or OPenDAP URL.

        Read fields from a netCDF file on disk or from an OPeNDAP
        server location.

        The file may be big or little endian.

        NetCDF dimension names are stored in the `ncdim` attributes of the
        field's DomainAxis objects and netCDF variable names are stored in
        the `ncvar` attributes of the field and its components
        (coordinates, coordinate bounds, cell measures and coordinate
        references, domain ancillaries, field ancillaries).

        :Parameters:

            filename: `str`
                The file name or OPenDAP URL of the dataset.

                Relative paths are allowed, and standard tilde and shell
                parameter expansions are applied to the string.

                *Parameter example:*
                  The file ``file.nc`` in the user's home directory could
                  be described by any of the following:
                  ``'$HOME/file.nc'``, ``'${HOME}/file.nc'``,
                  ``'~/file.nc'``, ``'~/tmp/../file.nc'``.

            extra: sequence of `str`, optional
                Create extra, independent fields from the particular types
                of metadata constructs. The *extra* parameter may be one,
                or a sequence, of:

                ==========================  ================================
                *extra*                     Metadata constructs
                ==========================  ================================
                ``'field_ancillary'``       Field ancillary constructs
                ``'domain_ancillary'``      Domain ancillary constructs
                ``'dimension_coordinate'``  Dimension coordinate constructs
                ``'auxiliary_coordinate'``  Auxiliary coordinate constructs
                ``'cell_measure'``          Cell measure constructs
                ==========================  ================================

                *Parameter example:*
                  To create fields from auxiliary coordinate constructs:
                  ``extra='auxiliary_coordinate'`` or
                  ``extra=['auxiliary_coordinate']``.

                *Parameter example:*
                  To create fields from domain ancillary and cell measure
                  constructs: ``extra=['domain_ancillary',
                  'cell_measure']``.

            warnings: `bool`, optional
                If False then do not print warnings when an output field
                construct is incomplete due to "structural
                non-CF-compliance" of the dataset. By default such
                warnings are displayed.

                Structural non-CF-compliance occurs when it is not
                possible to unambiguously map an element of the netCDF
                dataset to an element of the CF data model. Other type on
                non-CF-compliance are not checked, for example, whether or
                not controlled vocabularies have been adhered to is not
                checked.

            mask: `bool`, optional
                If False then do not mask by convention when reading the
                data of field or metadata constructs from disk. By default
                data is masked by convention.

                The masking by convention of a netCDF array depends on the
                values of any of the netCDF variable attributes
                ``_FillValue`` and ``missing_value``,``valid_min``,
                ``valid_max``, ``valid_range``. See the CF conventions for
                details.

                .. versionadded:: (cfdm) 1.8.2

            warn_valid: `bool`, optional
                If True then print a warning for the presence of
                ``valid_min``, ``valid_max`` or ``valid_range`` properties
                on field constructs and metadata constructs that have
                data. By default no such warning is printed

                "Out-of-range" data values in the file, as defined by any
                of these properties, are by default automatically masked,
                which may not be as intended. See the *mask* parameter for
                turning off all automatic masking.

                .. versionadded:: (cfdm) 1.8.3

        :Returns:

            `list`
                The fields in the file.

        """
        # ------------------------------------------------------------
        # Initialise netCDF read parameters
        # ------------------------------------------------------------
        self.read_vars = {
            "new_dimensions": {},
            "formula_terms": {},
            "compression": {},
            # Verbose?
            "verbose": verbose,
            # Warnings?
            "warnings": warnings,
            "dataset_compliance": {None: {"non-compliance": {}}},
            "component_report": {},
            "auxiliary_coordinate": {},
            "cell_measure": {},
            "dimension_coordinate": {},
            "domain_ancillary": {},
            "domain_ancillary_key": None,
            "field_ancillary": {},
            "coordinates": {},
            "bounds": {},
            # --------------------------------------------------------
            # Geometry containers, keyed by their netCDF geometry
            # container variable names.
            # --------------------------------------------------------
            "geometries": {},
            # Map data variables to their geometry variable names
            "variable_geometry": {},
            "do_not_create_field": set(),
            "references": {},
            "referencers": {},
            # --------------------------------------------------------
            # External variables
            # --------------------------------------------------------
            # Variables listed by the global external_variables
            # attribute
            "external_variables": set(),
            # External variables that are actually referenced from
            # within the parent file
            "referenced_external_variables": set(),
            # --------------------------------------------------------
            # Coordinate references
            # --------------------------------------------------------
            # Grid mapping attributes that describe horizontal datum
            "datum_parameters": self.cf_datum_parameters(),
            # Vertical coordinate reference constructs, keyed by the
            # netCDF variable name of their parent parametric vertical
            # coordinate variable.
            #
            # E.g. {'ocean_s_coordinate':
            #        <CoordinateReference: ocean_s_coordinate>}
            "vertical_crs": {},
            #
            "version": {},
            # Auto mask?
            "mask": bool(mask),
            # Warn for the presence of valid_[min|max|range]
            # attributes?
            "warn_valid": bool(warn_valid),
            "valid_properties": set(("valid_min", "valid_max", "valid_range")),
            # Assume a priori that the dataset does not have a group
            # structure
            "has_groups": False,
            # Keep a list of flattened file names
            "flat_files": [],
        }

        g = self.read_vars

        # Set versions
        for version in ("1.6", "1.7", "1.8", "1.9"):
            g["version"][version] = LooseVersion(version)

        # ------------------------------------------------------------
        # Add custom read vars
        # ------------------------------------------------------------
        if extra_read_vars:
            g.update(deepcopy(extra_read_vars))

        # ----------------------------------------------------------------
        # Parse field parameter
        # ----------------------------------------------------------------
        g["get_constructs"] = {
            "auxiliary_coordinate": self.implementation.get_auxiliary_coordinates,
            "cell_measure": self.implementation.get_cell_measures,
            "dimension_coordinate": self.implementation.get_dimension_coordinates,
            "domain_ancillary": self.implementation.get_domain_ancillaries,
            "field_ancillary": self.implementation.get_field_ancillaries,
        }

        # Parse the 'external' keyword parameter
        if external:
            if isinstance(external, str):
                external = (external,)
        else:
            external = ()

        g["external_files"] = set(external)

        # Parse 'extra' keyword parameter
        if extra:
            if isinstance(extra, str):
                extra = (extra,)

            for f in extra:
                if f not in g["get_constructs"]:
                    raise ValueError(
                        f"Can't read: Bad parameter value: extra={extra!r}"
                    )

        g["extra"] = extra

        filename = os.path.expanduser(os.path.expandvars(filename))

        if os.path.isdir(filename):
            raise IOError(f"Can't read directory {filename}")

        if not os.path.isfile(filename):
            raise IOError(f"Can't read non-existent file {filename}")

        g["filename"] = filename

        # ------------------------------------------------------------
        # Open the netCDF file to be read
        # ------------------------------------------------------------
        nc = self.file_open(filename, flatten=True, verbose=None)
        logger.info(f"Reading netCDF file: {filename}\n")  # pragma: no cover
        if is_log_level_debug(logger):
            logger.debug(
                f"    Input netCDF dataset:\n        {nc}\n"
            )  # pragma: no cover

        # ----------------------------------------------------------------
        # Put the file's global attributes into the global
        # 'global_attributes' dictionary
        # ----------------------------------------------------------------
        global_attributes = {}
        for attr in map(str, nc.ncattrs()):
            try:
                value = nc.getncattr(attr)
                if isinstance(value, str):
                    try:
                        global_attributes[attr] = str(value)
                    except UnicodeEncodeError:
                        global_attributes[attr] = value.encode(errors="ignore")
                else:
                    global_attributes[attr] = value
            except UnicodeDecodeError:
                pass

        g["global_attributes"] = global_attributes
        if is_log_level_debug(logger):
            logger.debug(
                f"    Global attributes:\n        {g['global_attributes']}"
            )  # pragma: no cover

        # ------------------------------------------------------------
        # Find the CF version for the file
        # ------------------------------------------------------------
        Conventions = g["global_attributes"].get("Conventions", "")

        all_conventions = re.split(",", Conventions)
        if all_conventions[0] == Conventions:
            all_conventions = re.split(r"\s+", Conventions)

        file_version = None
        for c in all_conventions:
            if not re.match(r"^CF-\d", c):
                continue

            file_version = re.sub("^CF-", "", c)

        if not file_version:
            if default_version is not None:
                # Assume the default version provided by the user
                file_version = default_version
            else:
                # Assume the file has the same version of the CFDM
                # implementation
                file_version = self.implementation.get_cf_version()

        g["file_version"] = LooseVersion(file_version)

        # Set minimum versions
        for vn in ("1.6", "1.7", "1.8", "1.9"):
            g["CF>=" + vn] = g["file_version"] >= g["version"][vn]

        # ------------------------------------------------------------
        # Create a dictionary keyed by netCDF variable names where
        # each key's value is a dictionary of that variable's netCDF
        # attributes. E.g. attributes['tas']['units']='K'
        # ------------------------------------------------------------
        variable_attributes = {}
        variable_dimensions = {}
        variable_dataset = {}
        variable_filename = {}
        variables = {}
        variable_groups = {}
        variable_group_attributes = {}
        variable_basename = {}
        variable_grouped_dataset = {}

        dimension_groups = {}
        dimension_basename = {}

        dimension_isunlimited = {}

        # ------------------------------------------------------------
        # For grouped files (CF>=1.8) map:
        #
        # * each flattened variable name to its absolute path
        # * each flattened dimension name to its absolute path
        # * each group to its group attributes
        #
        # ------------------------------------------------------------
        has_groups = g["has_groups"]

        flattener_variables = {}
        flattener_dimensions = {}
        flattener_attributes = {}

        if has_groups:
            flattener_name_mapping_variables = getattr(
                nc, "__flattener_name_mapping_variables", None
            )
            if flattener_name_mapping_variables is not None:
                if isinstance(flattener_name_mapping_variables, str):
                    flattener_name_mapping_variables = [
                        flattener_name_mapping_variables
                    ]
                flattener_variables = dict(
                    tuple(x.split(": "))
                    for x in flattener_name_mapping_variables
                )

            flattener_name_mapping_dimensions = getattr(
                nc, "__flattener_name_mapping_dimensions", None
            )
            if flattener_name_mapping_dimensions is not None:
                if isinstance(flattener_name_mapping_dimensions, str):
                    flattener_name_mapping_dimensions = [
                        flattener_name_mapping_dimensions
                    ]
                flattener_dimensions = dict(
                    tuple(x.split(": "))
                    for x in flattener_name_mapping_dimensions
                )

                # Remove a leading / (slash) from dimensions in the
                # root group
                for key, value in flattener_dimensions.items():
                    if value.startswith("/") and value.count("/") == 1:
                        flattener_dimensions[key] = value[1:]

            flattener_name_mapping_attributes = getattr(
                nc, "__flattener_name_mapping_attributes", None
            )
            if flattener_name_mapping_attributes is not None:
                if isinstance(flattener_name_mapping_attributes, str):
                    flattener_name_mapping_attributes = [
                        flattener_name_mapping_attributes
                    ]
                flattener_attributes = dict(
                    tuple(x.split(": "))
                    for x in flattener_name_mapping_attributes
                )

                # Remove group attributes from the global attributes,
                # and vice versa.
                for flat_attr in flattener_attributes.copy():
                    attr = flattener_attributes.pop(flat_attr)

                    x = attr.split("/")
                    groups = x[1:-1]

                    if groups:
                        g["global_attributes"].pop(flat_attr)

                        group_attr = x[-1]
                        flattener_attributes.setdefault(tuple(groups), {})[
                            group_attr
                        ] = nc.getncattr(flat_attr)

            # Remove flattener attributes from the global attributes
            for attr in (
                "__flattener_name_mapping_variables",
                "__flattener_name_mapping_dimensions",
                "__flattener_name_mapping_attributes",
            ):
                g["global_attributes"].pop(attr, None)

        for ncvar in nc.variables:
            ncvar_basename = ncvar
            groups = ()
            group_attributes = {}

            variable = nc.variables[ncvar]

            # --------------------------------------------------------
            # Specify the group structure for each variable (CF>=1.8)
            # TODO
            # If the file only has the root group then this dictionary
            # will be empty. Variables in the root group when there
            # are sub-groups will have dictionary values of None.
            # --------------------------------------------------------
            if has_groups:
                # Replace the flattened variable name with its
                # absolute path.
                ncvar_flat = ncvar
                ncvar = flattener_variables[ncvar]

                groups = tuple(ncvar.split("/")[1:-1])

                if groups:
                    # This variable is in a group. Remove the group
                    # structure that was prepended to the netCDF
                    # variable name by the netCDF flattener.
                    ncvar_basename = re.sub(
                        f"^{_flattener_separator.join(groups)}{_flattener_separator}",
                        "",
                        ncvar_flat,
                    )

                    # ------------------------------------------------
                    # Group attributes. Note that, currently,
                    # sub-group attributes supercede all parent group
                    # attributes (but not global attributes).
                    # ------------------------------------------------
                    group_attributes = {}
                    for i in range(1, len(groups) + 1):
                        hierarchy = groups[:i]
                        if hierarchy not in flattener_attributes:
                            continue

                        group_attributes.update(
                            flattener_attributes[hierarchy]
                        )
                else:
                    # Remove the leading / from the absolute netCDF
                    # variable path
                    ncvar = ncvar[1:]
                    flattener_variables[ncvar] = ncvar

                variable_grouped_dataset[ncvar] = g["nc_grouped"]

            variable_attributes[ncvar] = {}
            for attr in map(str, variable.ncattrs()):
                try:
                    variable_attributes[ncvar][attr] = variable.getncattr(attr)
                    if isinstance(variable_attributes[ncvar][attr], str):
                        try:
                            variable_attributes[ncvar][attr] = str(
                                variable_attributes[ncvar][attr]
                            )
                        except UnicodeEncodeError:
                            variable_attributes[ncvar][
                                attr
                            ] = variable_attributes[ncvar][attr].encode(
                                errors="ignore"
                            )
                except UnicodeDecodeError:
                    pass

            variable_dimensions[ncvar] = tuple(variable.dimensions)
            variable_dataset[ncvar] = nc
            variable_filename[ncvar] = g["filename"]
            variables[ncvar] = variable

            variable_basename[ncvar] = ncvar_basename
            variable_groups[ncvar] = groups
            variable_group_attributes[ncvar] = group_attributes

        # Populate dimensions_groups abd dimension_basename
        # dictionaries
        for ncdim in nc.dimensions:
            ncdim_org = ncdim
            ncdim_basename = ncdim
            groups = ()
            ncdim_basename = ncdim

            if has_groups:
                # Replace the flattened variable name with its
                # absolute path.
                ncdim_flat = ncdim
                ncdim = flattener_dimensions[ncdim_flat]

                groups = tuple(ncdim.split("/")[1:-1])

                if groups:
                    # This dimension is in a group.
                    ncdim_basename = re.sub(
                        "^{_flattener_separator.join(groups)}{_flattener_separator}",
                        "",
                        ncdim_flat,
                    )

            dimension_groups[ncdim] = groups
            dimension_basename[ncdim] = ncdim_basename

            dimension_isunlimited[ncdim] = nc.dimensions[
                ncdim_org
            ].isunlimited()

        if has_groups:
            variable_dimensions = {
                name: tuple([flattener_dimensions[ncdim] for ncdim in value])
                for name, value in variable_dimensions.items()
            }

        if is_log_level_debug(logger):
            logger.debug(
                "    General read variables:\n"
                "        read_vars['variable_dimensions'] =\n"
                f"            {variable_dimensions}"
            )  # pragma: no cover

        # The netCDF attributes for each variable
        #
        # E.g. {'grid_lon': {'standard_name': 'grid_longitude'}}
        g["variable_attributes"] = variable_attributes

        # The netCDF dimensions for each variable
        #
        # E.g. {'grid_lon_bounds': ('grid_longitude', 'bounds2')}
        g["variable_dimensions"] = variable_dimensions

        # The netCDF4 dataset object for each variable
        g["variable_dataset"] = variable_dataset

        # The original gouped dataset for each variable (empty if the
        # original dataset is not grouped) v1.8.8.1
        g["variable_grouped_dataset"] = variable_grouped_dataset

        # The name of the file containing the each variable
        g["variable_filename"] = variable_filename

        # The netCDF4 variable object for each variable
        g["variables"] = variables

        # The netCDF4 dataset objects that have been opened (i.e. the
        # for parent file and any external files)
        g["datasets"] = [nc]

        # The names of the variable in the parent files
        # (i.e. excluding any external variables)
        g["internal_variables"] = set(variables)

        # The netCDF dimensions of the parent file
        internal_dimension_sizes = {}
        for name, dimension in nc.dimensions.items():
            if (
                has_groups
                and dimension_isunlimited[flattener_dimensions[name]]
            ):
                # For grouped datasets, get the unlimited dimension
                # size from the original grouped dataset, because
                # unlimited dimensions have size 0 in the flattened
                # dataset (because it contains no data) (v1.8.8.1)
                group, ncdim = self._netCDF4_group(
                    g["nc_grouped"], flattener_dimensions[name]
                )
                internal_dimension_sizes[name] = group.dimensions[ncdim].size
            else:
                internal_dimension_sizes[name] = dimension.size

        if g["has_groups"]:
            internal_dimension_sizes = {
                flattener_dimensions[name]: value
                for name, value in internal_dimension_sizes.items()
            }

        g["internal_dimension_sizes"] = internal_dimension_sizes

        # The group structure for each variable. Variables in the root
        # group have a group structure of ().
        #
        # E.g. {'lat': (),
        #       '/forecasts/lon': ('forecasts',)
        #       '/forecasts/model/t': 'forecasts', 'model')}
        g["variable_groups"] = variable_groups

        # The group attributes that apply to each variable
        #
        # E.g. {'latitude': {},
        #       'eastward_wind': {'model': 'climate1'}}
        g["variable_group_attributes"] = variable_group_attributes

        # Mapped components of a flattened version of the netCDF file
        g["flattener_variables"] = flattener_variables
        g["flattener_dimensions"] = flattener_dimensions
        g["flattener_attributes"] = flattener_attributes

        # The basename of each variable. I.e. the dimension name
        # without its prefixed group structure.
        #
        # E.g. {'lat': 'lat',
        #       '/forecasts/lon': 'lon',
        #       '/forecasts/model/t': 't'}
        g["variable_basename"] = variable_basename

        # The unlimited status of each dimension
        #
        # E.g. {'/forecast/lat': False, 'bounds2': False, 'lon':
        #       False}
        g["dimension_isunlimited"] = dimension_isunlimited

        # The group structure for each dimension. Dimensions in the
        # root group have a group structure of ().
        #
        # E.g. {'lat': (),
        #       '/forecasts/lon': ('forecasts',)
        #       '/forecasts/model/t': 9'forecasts', 'model')}
        g["dimension_groups"] = dimension_groups

        # The basename of each dimension. I.e. the dimension name
        # without its prefixed group structure.
        #
        # E.g. {'lat': 'lat',
        #       '/forecasts/lon': 'lon',
        #       '/forecasts/model/t': 't'}
        g["dimension_basename"] = dimension_basename

        if is_log_level_debug(logger):
            logger.debug(
                "        read_vars['dimension_isunlimited'] =\n"
                f"            {g['dimension_isunlimited']}\n"
                "        read_vars['internal_dimension_sizes'] =\n"
                f"            {g['internal_dimension_sizes']}\n"
                "    Groups read vars:\n"
                "        read_vars['variable_groups'] =\n"
                f"            {g['variable_groups']}\n"
                "        read_vars['variable_basename'] =\n"
                f"            {variable_basename}\n"
                "        read_vars['dimension_groups'] =\n"
                f"            {g['dimension_groups']}\n"
                "        read_vars['dimension_basename'] =\n"
                f"            {g['dimension_basename']}\n"
                "        read_vars['flattener_variables'] =\n"
                f"            {g['flattener_variables']}\n"
                "        read_vars['flattener_dimensions'] =\n"
                f"            {g['flattener_dimensions']}\n"
                "        read_vars['flattener_attributes'] =\n"
                f"            {g['flattener_attributes']}\n"
                f"    netCDF dimensions: {internal_dimension_sizes}"
            )  # pragma: no cover

        # ------------------------------------------------------------
        # List variables
        #
        # Identify and parse all list variables
        # ------------------------------------------------------------
        for ncvar, dimensions in variable_dimensions.items():
            if dimensions != (ncvar,):
                continue

            # This variable is a Unidata coordinate variable
            compress = variable_attributes[ncvar].get("compress")
            if compress is None:
                continue

            # This variable is a list variable for gathering
            # arrays
            self._parse_compression_gathered(ncvar, compress)

            # Do not attempt to create a field from a list
            # variable
            g["do_not_create_field"].add(ncvar)

        # ------------------------------------------------------------
        # DSG variables (CF>=1.6)
        #
        # Identify and parse all DSG count and DSG index variables
        # ------------------------------------------------------------
        if g["CF>=1.6"]:
            featureType = g["global_attributes"].get("featureType")
            if featureType is not None:
                g["featureType"] = featureType

                sample_dimension = None
                for ncvar, attributes in variable_attributes.items():
                    if "sample_dimension" not in attributes:
                        continue

                    # ------------------------------------------------
                    # This variable is a count variable for DSG
                    # contiguous ragged arrays
                    # ------------------------------------------------
                    sample_dimension = attributes["sample_dimension"]

                    if has_groups:
                        sample_dimension = g["flattener_dimensions"].get(
                            sample_dimension, sample_dimension
                        )

                    cf_compliant = self._check_sample_dimension(
                        ncvar, sample_dimension
                    )
                    if not cf_compliant:
                        sample_dimension = None
                    else:
                        self._parse_ragged_contiguous_compression(
                            ncvar, sample_dimension
                        )

                        # Do not attempt to create a field from a
                        # count variable
                        g["do_not_create_field"].add(ncvar)

                instance_dimension = None
                for ncvar, attributes in variable_attributes.items():
                    if "instance_dimension" not in attributes:
                        continue

                    # ------------------------------------------------
                    # This variable is an index variable for DSG
                    # indexed ragged arrays
                    # ------------------------------------------------
                    instance_dimension = attributes["instance_dimension"]

                    if has_groups:
                        instance_dimension = g["flattener_dimensions"].get(
                            instance_dimension, instance_dimension
                        )

                    cf_compliant = self._check_instance_dimension(
                        ncvar, instance_dimension
                    )
                    if not cf_compliant:
                        instance_dimension = None
                    else:
                        self._parse_indexed_compression(
                            ncvar, instance_dimension
                        )

                        # Do not attempt to create a field from a
                        # index variable
                        g["do_not_create_field"].add(ncvar)

                if (
                    sample_dimension is not None
                    and instance_dimension is not None
                ):
                    # ------------------------------------------------
                    # There are DSG indexed contiguous ragged arrays
                    # ------------------------------------------------
                    self._parse_indexed_contiguous_compression(
                        sample_dimension, instance_dimension
                    )

        # ------------------------------------------------------------
        # Identify and parse all geometry container variables
        # (CF>=1.8)
        # ------------------------------------------------------------
        if g["CF>=1.8"]:
            for ncvar, attributes in variable_attributes.items():
                if "geometry" not in attributes:
                    # This data variable does not have a geometry
                    # container
                    continue

                geometry_ncvar = self._parse_geometry(
                    ncvar, variable_attributes
                )

                if not geometry_ncvar:
                    # The geometry container has already been parsed,
                    # or a sufficiently compliant geometry container
                    # could not be found.
                    continue

                # Do not attempt to create a field construct from a
                # node coordinate variable
                g["do_not_create_field"].add(geometry_ncvar)

        if is_log_level_debug(logger):
            logger.debug(
                "    Compression read vars:\n"
                "        read_vars['compression'] =\n"
                f"            {g['compression']}"
            )  # pragma: no cover

        # ------------------------------------------------------------
        # Parse external variables (CF>=1.7)
        # ------------------------------------------------------------
        if g["CF>=1.7"]:
            netcdf_external_variables = global_attributes.pop(
                "external_variables", None
            )
            parsed_external_variables = self._split_string_by_white_space(
                None, netcdf_external_variables
            )
            parsed_external_variables = self._check_external_variables(
                netcdf_external_variables, parsed_external_variables
            )
            g["external_variables"] = set(parsed_external_variables)

        # Now that all of the variables have been scanned, customize
        # the read parameters.
        self._customize_read_vars()

        if _scan_only:
            return self.read_vars

        # ------------------------------------------------------------
        # Get external variables (CF>=1.7)
        # ------------------------------------------------------------
        if g["CF>=1.7"]:
            logger.info(
                f"    External variables: {g['external_variables']}\n"
                f"    External files    : {g['external_files']}"
            )  # pragma: no cover

            if g["external_files"] and g["external_variables"]:
                self._get_variables_from_external_files(
                    netcdf_external_variables
                )

        # ------------------------------------------------------------
        # Create a field from every netCDF variable (apart from
        # special variables that have already been identified as such)
        # ------------------------------------------------------------
        all_fields = OrderedDict()
        for ncvar in g["variables"]:
            if ncvar not in g["do_not_create_field"]:
                all_fields[ncvar] = self._create_field(ncvar)

        # ------------------------------------------------------------
        # Check for unreferenced external variables (CF>=1.7)
        # ------------------------------------------------------------
        if g["CF>=1.7"]:
            unreferenced_external_variables = g[
                "external_variables"
            ].difference(g["referenced_external_variables"])
            for ncvar in unreferenced_external_variables:
                self._add_message(
                    None,
                    ncvar,
                    message=("External variable", "is not referenced in file"),
                    attribute={
                        "external_variables": netcdf_external_variables
                    },
                )

        if is_log_level_debug(logger):
            logger.debug(
                "    Reference read vars:\n"
                "        read_vars['references'] =\n"
                f"            {g['references']}\n"
                "        read_vars['referencers'] =\n"
                f"            {g['referencers']}"
            )  # pragma: no cover

        # ------------------------------------------------------------
        # Discard fields created from netCDF variables that are
        # referenced by other netCDF variables
        # ------------------------------------------------------------
        fields = OrderedDict()
        for ncvar, f in all_fields.items():
            if self._is_unreferenced(ncvar):
                fields[ncvar] = f

        referenced_variables = [
            ncvar
            for ncvar in sorted(all_fields)
            if not self._is_unreferenced(ncvar)
        ]
        unreferenced_variables = [
            ncvar
            for ncvar in sorted(all_fields)
            if self._is_unreferenced(ncvar)
        ]

        for ncvar in referenced_variables[:]:
            if all(
                referencer in referenced_variables
                for referencer in g["referencers"][ncvar]
            ):
                referenced_variables.remove(ncvar)
                unreferenced_variables.append(ncvar)
                fields[ncvar] = all_fields[ncvar]

        logger.info(
            "    Referenced netCDF variables:\n        "
            + "\n        ".join(referenced_variables)
        )  # pragma: no cover
        if g["do_not_create_field"]:
            logger.info(
                "        "
                + "\n        ".join(
                    [ncvar for ncvar in sorted(g["do_not_create_field"])]
                )
            )  # pragma: no cover
        logger.info(
            "    Unreferenced netCDF variables:\n        "
            + "\n        ".join(unreferenced_variables)
        )  # pragma: no cover

        # ------------------------------------------------------------
        # If requested, reinstate fields created from netCDF variables
        # that are referenced by other netCDF variables.
        # ------------------------------------------------------------
        self_referenced = {}
        if g["extra"]:
            fields0 = list(fields.values())
            for construct_type in g["extra"]:
                for f in fields0:
                    for construct in g["get_constructs"][construct_type](
                        f
                    ).values():
                        ncvar = self.implementation.nc_get_variable(construct)
                        if ncvar not in all_fields:
                            continue

                        if ncvar not in fields:
                            fields[ncvar] = all_fields[ncvar]
                        else:
                            self_referenced[ncvar] = all_fields[ncvar]

        if not self_referenced:
            items = fields.items()
        else:
            items = tuple(fields.items()) + tuple(self_referenced.items())

        out = [x[1] for x in sorted(items)]

        if warnings:
            for x in out:
                qq = x.dataset_compliance()
                if qq:
                    logger.warning(
                        f"WARNING: {x.__class__.__name__} incomplete due to "
                        f"non-CF-compliant dataset. Report:\n{qq}"
                    )  # pragma: no cover

        if warn_valid:
            # --------------------------------------------------------
            # Warn for the presence of 'valid_min', 'valid_max'or
            # 'valid_range' properties. (Introduced at v1.8.3)
            # --------------------------------------------------------
            for f in out:
                # Check field constructs
                self._check_valid(f, f)

                # Check constructs with data
                for c in self.implementation.get_constructs(
                    f, data=True
                ).values():
                    self._check_valid(f, c)

        # Close all opened netCDF files
        self.file_close()

        # Return the fields
        return out

    def file_close(self):
        """Close all netCDF files that have been opened.

        Includes the input file being read, any external files, and any
        temporary flattened files.

        :Returns:

            `None`

        **Examples:**

        >>> r.file_close()

        """
        for nc in self.read_vars["datasets"]:
            nc.close()

        # Close temporary flattened files
        for flat_file in self.read_vars["flat_files"]:
            flat_file.close()

        # Close the original grouped file (v1.8.8.1)
        if "nc_grouped" in self.read_vars:
            self.read_vars["nc_grouped"].close()


class XRRead(XRReadBase):
    """TODO.

    .. versionadded:: 3.0.0

    """

    def _ncdimensions(self, ncvar):
        """Return a list of the netCDF dimensions corresponding to a
        netCDF variable.

        If the variable has been compressed then the *implied
        uncompressed* dimensions are returned.

        For a CFA variable, the netCDF dimensions are taken from the
        'cfa_dimensions' netCDF attribute.

        .. versionadded:: 3.0.0

        :Parameters:

            ncvar: `str`
                The netCDF variable name.

        :Returns:

            `list`
                The netCDF dimension names spanned by the netCDF variable.

        **Examples:**

        >>> n._ncdimensions('humidity')
        ['time', 'lat', 'lon']

        """
        g = self.read_vars

        cfa = (
            g["cfa"]
            and ncvar not in g["external_variables"]
            and g["variable_attributes"][ncvar].get("cf_role")
            == "cfa_variable"
        )

        if not cfa:
            return super()._ncdimensions(ncvar)

        # Still here?
        ncdimensions = (
            g["variable_attributes"][ncvar].get("cfa_dimensions", "").split()
        )

        return list(map(str, ncdimensions))

    def _get_domain_axes(self, ncvar, allow_external=False):
        """Return the domain axis identifiers that correspond to a
        netCDF variable's netCDF dimensions.

        For a CFA variable, the netCDF dimensions are taken from the
        'cfa_dimensions' netCDF attribute.

        :Parameter:

            ncvar: `str`
                The netCDF variable name.

            allow_external: `bool`
                If `True` and *ncvar* is an external variable then return an
                empty list.

        :Returns:

            `list`

        **Examples:**

        >>> r._get_domain_axes('areacello')
        ['domainaxis0', 'domainaxis1']

        >>> r._get_domain_axes('areacello', allow_external=True)
        []

        """
        g = self.read_vars

        cfa = (
            g["cfa"]
            and ncvar not in g["external_variables"]
            and g["variable_attributes"][ncvar].get("cf_role")
            == "cfa_variable"
        )

        if not cfa:
            return super()._get_domain_axes(
                ncvar=ncvar, allow_external=allow_external
            )

        # Still here?
        cfa_dimensions = (
            g["variable_attributes"][ncvar].get("cfa_dimensions", "").split()
        )

        ncdim_to_axis = g["ncdim_to_axis"]
        axes = [
            ncdim_to_axis[ncdim]
            for ncdim in cfa_dimensions
            if ncdim in ncdim_to_axis
        ]

        return axes

    def _create_data(
        self,
        ncvar,
        construct=None,
        unpacked_dtype=False,
        uncompress_override=None,
        parent_ncvar=None,
    ):
        """TODO.

        .. versionadded:: 3.0.0

        :Parameters:

            ncvar: `str`
                The name of the netCDF variable that contains the data.

            construct: optional

            unpacked_dtype: `False` or `numpy.dtype`, optional

            uncompress_override: `bool`, optional

        :Returns:

            `Data`

        """
        g = self.read_vars

        is_cfa_variable = (
            g["cfa"]
            and construct.get_property("cf_role", None) == "cfa_variable"
        )

        if not is_cfa_variable:
            # --------------------------------------------------------
            # Create data for a normal netCDF variable
            # --------------------------------------------------------
            return super()._create_data(
                ncvar=ncvar,
                construct=construct,
                unpacked_dtype=unpacked_dtype,
                uncompress_override=uncompress_override,
                parent_ncvar=parent_ncvar,
            )

        # ------------------------------------------------------------
        # Still here? Then create data for a CFA netCDF variable
        # ------------------------------------------------------------
        #        print ('    Creating data from CFA variable', repr(ncvar),
        #               repr(construct))
        try:
            cfa_data = json.loads(construct.get_property("cfa_array"))
        except ValueError as error:
            raise ValueError(
                "Error during JSON-decoding of netCDF attribute 'cfa_array': "
                "{}".format(error)
            )

        variable = g["variables"][ncvar]

        cfa_data["file"] = g["filename"]
        cfa_data["Units"] = construct.Units
        cfa_data["fill_value"] = construct.fill_value()
        cfa_data["_pmshape"] = cfa_data.pop("pmshape", ())
        cfa_data["_pmaxes"] = cfa_data.pop("pmdimensions", ())

        base = cfa_data.get("base", None)
        if base is not None:
            cfa_data["base"] = pathjoin(dirname(g["filename"]), base)

        ncdimensions = construct.get_property("cfa_dimensions", "").split()
        dtype = variable.dtype

        if dtype is str:
            # netCDF string types have a dtype of `str`, which needs
            # to be reset as a numpy.dtype, but we don't know what
            # without reading the data, so set it to None for now.
            dtype = None

        # UNICODE???? TODO
        if self._is_char(ncvar) and dtype.kind in "SU" and ncdimensions:
            strlen = g["nc"].dimensions[ncdimensions[-1]].size
            if strlen > 1:
                ncdimensions.pop()
                dtype = numpy.dtype("S{0}".format(strlen))

        cfa_data["dtype"] = dtype
        cfa_data["_axes"] = ncdimensions
        cfa_data["shape"] = [
            g["nc"].dimensions[ncdim].size for ncdim in ncdimensions
        ]

        for attrs in cfa_data["Partitions"]:
            # FORMAT
            sformat = attrs.get("subarray", {}).pop("format", "netCDF")
            if sformat is not None:
                attrs["format"] = sformat

            # DTYPE
            dtype = attrs.get("subarray", {}).pop("dtype", None)
            if dtype not in (None, "char"):
                attrs["subarray"]["dtype"] = numpy.dtype(dtype)

            # UNITS and CALENDAR
            units = attrs.pop("punits", None)
            calendar = attrs.pop("pcalendar", None)
            if units is not None or calendar is not None:
                attrs["Units"] = Units(units, calendar)

            # AXES
            pdimensions = attrs.pop("pdimensions", None)
            if pdimensions is not None:
                attrs["axes"] = pdimensions

            # REVERSE
            reverse = attrs.pop("reverse", None)
            if reverse is not None:
                attrs["reverse"] = reverse

            # LOCATION: Change to python indexing (i.e. range does not
            #           include the final index)
            for r in attrs["location"]:
                r[1] += 1

            # PART: Change to python indexing (i.e. slice range does
            #       not include the final index)
            part = attrs.get("part", None)
            if part:
                p = []
                for x in ast.literal_eval(part):
                    if isinstance(x, list):
                        if x[2] > 0:
                            p.append(slice(x[0], x[1] + 1, x[2]))
                        elif x[1] == 0:
                            p.append(slice(x[0], None, x[2]))
                        else:
                            p.append(slice(x[0], x[1] - 1, x[2]))
                    else:
                        p.append(list(x))

                attrs["part"] = p

        construct.del_property("cf_role")
        construct.del_property("cfa_array")
        construct.del_property("cfa_dimensions", None)

        out = self._create_Data(loadd=cfa_data)

        return out

    def _create_Data(
        self,
        array=None,
        units=None,
        calendar=None,
        ncvar=None,
        loadd=None,
        **kwargs
    ):
        """TODO.

        .. versionadded:: 3.0.0

        :Parameters:

            ncvar: `str`
                The netCDF variable from which to get units and calendar.

        """
        try:
            compressed = array.get_compression_type()  # TODO
        except AttributeError:
            compressed = False

        if not compressed:
            # Do not chunk compressed data (for now ...)
            chunk = False
        else:
            chunk = self.read_vars.get("chunk", True)

        return super()._create_Data(
            array=array,
            units=units,
            calendar=calendar,
            ncvar=ncvar,
            loadd=loadd,
            chunk=chunk,
            **kwargs
        )

    def _customize_read_vars(self):
        """TODO.

        .. versionadded:: 3.0.0

        """
        super()._customize_read_vars()

        g = self.read_vars

        # ------------------------------------------------------------
        # Find out if this is a CFA file
        # ------------------------------------------------------------
        g["cfa"] = "CFA" in g["global_attributes"].get(
            "Conventions", ()
        )  # TODO

        # ------------------------------------------------------------
        # Do not create fields from CFA private variables
        # ------------------------------------------------------------
        if g["cfa"]:
            for ncvar in g["variables"]:
                if (
                    g["variable_attributes"][ncvar].get("cf_role", None)
                    == "cfa_private"
                ):
                    g["do_not_create_field"].add(ncvar)

        # ------------------------------------------------------------
        #
        # ------------------------------------------------------------
        if g["cfa"]:
            for ncvar, ncdims in tuple(g["variable_dimensions"].items()):
                if ncdims != ():
                    continue

                if not (
                    ncvar not in g["external_variables"]
                    and g["variable_attributes"][ncvar].get("cf_role")
                    == "cfa_variable"
                ):
                    continue

                ncdimensions = (
                    g["variable_attributes"][ncvar]
                    .get("cfa_dimensions", "")
                    .split()
                )
                if ncdimensions:
                    g["variable_dimensions"][ncvar] = tuple(
                        map(str, ncdimensions)
                    )

    def file_open(self, filename, flatten=True, verbose=None):
        """Open the netCDf file for reading.

        :Paramters:

            filename: `str`
                The netCDF file to be read.

            flatten: `bool`, optional
                If False then do not flatten a grouped file. Ignored if
                the file has no groups.

                .. versionadded:: 3.6.0

        :Returns:

            `netCDF4.Dataset`
                The object for the file.

        """
        out = super().file_open(filename, flatten=flatten, verbose=verbose)
        _file_to_fh["netCDF"].pop(filename, None)
        return out
