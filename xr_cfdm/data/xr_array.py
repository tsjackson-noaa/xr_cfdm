import cfdm
import numpy
import netCDF4

from .functions import _open_netcdf_file, _close_netcdf_file

import logging
logger = logging.getLogger(__name__)

class XRArrayBase(cfdm.data.NetCDFArray):
    def __getitem__(self, indices):
        """Returns a subspace of the array as a numpy array.

        x.__getitem__(indices) <==> x[indices]

        The indices that define the subspace must be either `Ellipsis` or
        a sequence that contains an index for each dimension. In the
        latter case, each dimension's index must either be a `slice`
        object or a sequence of two or more integers.

        Indexing is similar to numpy indexing. The only difference to
        numpy indexing (given the restrictions on the type of indices
        allowed) is:

          * When two or more dimension's indices are sequences of integers
            then these indices work independently along each dimension
            (similar to the way vector subscripts work in Fortran).

        .. versionadded:: (cfdm) 1.7.0

        """
        netcdf = self.open()

        # Traverse the group structure, if there is one (CF>=1.8).
        group = self.get_group()
        if group:
            for g in group[:-1]:
                netcdf = netcdf.groups[g]

            netcdf = netcdf.groups[group[-1]]

        ncvar = self.get_ncvar()
        mask = self.get_mask()

        if ncvar is not None:
            # Get the variable by netCDF name
            variable = netcdf.variables[ncvar]
            variable.set_auto_mask(mask)
            array = variable[indices]
        else:
            # Get the variable by netCDF ID
            varid = self.get_varid()

            for variable in netcdf.variables.values():
                if variable._varid == varid:
                    variable.set_auto_mask(mask)
                    array = variable[indices]
                    break

        if self._get_component("close"):
            # Close the netCDF file
            self.close()

        string_type = isinstance(array, str)
        if string_type:
            # --------------------------------------------------------
            # A netCDF string type scalar variable comes out as Python
            # str object, so convert it to a numpy array.
            # --------------------------------------------------------
            array = numpy.array(array, dtype="S{0}".format(len(array)))

        if not self.ndim:
            # Hmm netCDF4 has a thing for making scalar size 1 , 1d
            array = array.squeeze()

        kind = array.dtype.kind
        if not string_type and kind in "SU":
            #     == 'S' and array.ndim > (self.ndim -
            #     getattr(self, 'gathered', 0) -
            #     getattr(self, 'ragged', 0)):
            # --------------------------------------------------------
            # Collapse (by concatenation) the outermost (fastest
            # varying) dimension of char array into
            # memory. E.g. [['a','b','c']] becomes ['abc']
            # --------------------------------------------------------
            if kind == "U":
                array = array.astype("S")

            array = netCDF4.chartostring(array)
            shape = array.shape
            array = numpy.array([x.rstrip() for x in array.flat], dtype="S")
            array = numpy.reshape(array, shape)
            array = numpy.ma.masked_where(array == b"", array)

        elif not string_type and kind == "O":
            # --------------------------------------------------------
            # A netCDF string type N-d (N>=1) variable comes out as a
            # numpy object array, so convert it to numpy string array.
            # --------------------------------------------------------
            array = array.astype("S")  # , copy=False)

            # --------------------------------------------------------
            # netCDF4 does not auto-mask VLEN variable, so do it here.
            # --------------------------------------------------------
            array = numpy.ma.where(array == b"", numpy.ma.masked, array)

        return array

    def open(self):
        """Returns an open `netCDF4.Dataset` for the array's file.

        .. versionadded:: (cfdm) 1.7.0

        :Returns:

            `netCDF4.Dataset`

        **Examples:**

        >>> netcdf = a.open()
        >>> variable = netcdf.variables[a.get_ncvar()]
        >>> variable.getncattr('standard_name')
        'eastward_wind'

        """
        if self._get_component("netcdf") is None:
            try:
                netcdf = netCDF4.Dataset(self.get_filename(), "r")
            except RuntimeError as error:
                raise RuntimeError(f"{error}: {self.get_filename()}")

            self._set_component("netcdf", netcdf, copy=False)

        return netcdf


class XRArray(XRArrayBase):
    """A sub-array stored in a netCDF file."""

    def __init__(
        self,
        filename=None,
        ncvar=None,
        varid=None,
        group=None,
        dtype=None,
        ndim=None,
        shape=None,
        size=None,
        mask=True,
    ):
        """**Initialization**

        :Parameters:

            filename: `str`
                The name of the netCDF file containing the array.

            ncvar: `str`, optional
                The name of the netCDF variable containing the array. Required
                unless *varid* is set.

            varid: `int`, optional
                The UNIDATA netCDF interface ID of the variable containing the
                array. Required if *ncvar* is not set, ignored if *ncvar* is
                set.

            group: `None` or sequence of `str`, optional
                Specify the netCDF4 group to which the netCDF variable
                belongs. By default, or if *group* is `None` or an empty
                sequence, it assumed to be in the root group. The last
                element in the sequence is the name of the group in which
                the variable lies, with other elements naming any parent
                groups (excluding the root group).

                :Parameter example:
                  To specify that a variable is in the root group:
                  ``group=()`` or ``group=None``

                :Parameter example:
                  To specify that a variable is in the group '/forecasts':
                  ``group=['forecasts']``

                :Parameter example:
                  To specify that a variable is in the group
                  '/forecasts/model2': ``group=['forecasts', 'model2']``

                .. versionadded:: 3.6.0

            dtype: `numpy.dtype`
                The data type of the array in the netCDF file. May be
                `None` if the numpy data-type is not known (which can be
                the case for netCDF string types, for example).

            shape: `tuple`
                The array dimension sizes in the netCDF file.

            size: `int`
                Number of elements in the array in the netCDF file.

            ndim: `int`
                The number of array dimensions in the netCDF file.

            mask: `bool`, optional
                If False then do not mask by convention when reading data
                from disk. By default data is masked by convention.

                A netCDF array is masked depending on the values of any of
                the netCDF variable attributes ``valid_min``,
                ``valid_max``, ``valid_range``, ``_FillValue`` and
                ``missing_value``.

                .. versionadded:: 3.4.0

        **Examples:**

        >>> import netCDF4
        >>> nc = netCDF4.Dataset('file.nc', 'r')
        >>> v = nc.variable['tas']
        >>> a = NetCDFFileArray(filename='file.nc', ncvar='tas',
        ...                     group=['forecast'], dtype=v.dtype,
        ...                     ndim=v.ndim, shape=v.shape, size=v.size)

        """
        super().__init__(
            filename=filename,
            ncvar=ncvar,
            varid=varid,
            group=group,
            dtype=dtype,
            ndim=ndim,
            shape=shape,
            size=size,
            mask=mask,
        )

        # By default, keep the netCDF file open after data array
        # access
        self._set_component("close", False, copy=False)

    @property
    def file_pointer(self):
        """The file pointer starting at the position of the netCDF
        variable."""
        offset = getattr(self, "ncvar", None)
        if offset is None:
            offset = self.varid

        return (self.get_filename(), offset)

    def close(self):
        """Close the file containing the data array.

        If the file is not open then no action is taken.

        :Returns:

            `None`

        **Examples:**

        >>> f.close()

        """
        _close_netcdf_file(self.get_filename())

    def open(self):
        """Return a `netCDF4.Dataset` object for the file containing the
        data array.

        :Returns:

            `netCDF4.Dataset`

        **Examples:**

        >>> f.open()
        <netCDF4.Dataset at 0x115a4d0>

        """
        return _open_netcdf_file(self.get_filename(), "r")
