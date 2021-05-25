import functools
import itertools

from operator import mul as operator_mul
from json import dumps as json_dumps
from json import loads as json_loads

import numpy as np
import cfdm

from ..cf_python.mixin_container import Container
from ..cf_python.constants import masked as cf_masked
from .data_aux import (
    # module-level functions/constants from cf-python
    _array_getattr,
    _convert_to_builtin_type,
    _initialise_axes,
    _seterr,
    _seterr_raise_to_ignore,
    _size_of_index,
    _year_length,
    _month_length
)
from ..cf_python.data.filledarray import FilledArray
from ..cf_python.cfdatetime import dt2rt, rt2dt, st2rt
from ..cf_python.cfdatetime import dt as cf_dt
from ..cf_python.decorators import (
    _inplace_enabled,
    _inplace_enabled_define_and_cleanup,
    _deprecated_kwarg_check,
    _manage_log_level_via_verbosity,
    _display_or_return,
)
from ..cf_python.functions import (
    parse_indices,
    pathjoin,
    hash_array,
)
from ..cf_python.functions import inspect as cf_inspect
from ..units import Units
from . import XRArray

import logging
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------
# _mask_fpe[0] = Whether or not to automatically set
#                FloatingPointError exceptions to masked values in
#                arimthmetic.
# --------------------------------------------------------------------
_mask_fpe = [False]
_empty_set = set()
_units_None = Units()

class Data(Container, cfdm.Data):
    """
    """
    def __init__(
        self,
        array=None,
        units=None,
        calendar=None,
        fill_value=None,
        hardmask=True,
        chunk=True,
        loadd=None,
        loads=None,
        dt=False,
        source=None,
        copy=True,
        dtype=None,
        mask=None,
        _use_array=True,
    ):
        """**Initialization**

        :Parameters:

            array: optional
                The array of values. May be any scalar or array-like
                object, including another `Data` instance. Ignored if the
                *source* parameter is set.

                *Parameter example:*
                  ``array=[34.6]``

                *Parameter example:*
                  ``array=[[1, 2], [3, 4]]``

                *Parameter example:*
                  ``array=np.ma.arange(10).reshape(2, 1, 5)``

            units: `str` or `Units`, optional
                The physical units of the data. if a `Units` object is
                provided then this an also set the calendar. Ignored if
                the *source* parameter is set.

                The units (without the calendar) may also be set after
                initialisation with the `set_units` method.

                *Parameter example:*
                  ``units='km hr-1'``

                *Parameter example:*
                  ``units='days since 2018-12-01'``

            calendar: `str`, optional
                The calendar for reference time units. Ignored if the
                *source* parameter is set.

                The calendar may also be set after initialisation with the
                `set_calendar` method.

                *Parameter example:*
                  ``calendar='360_day'``

            fill_value: optional
                The fill value of the data. By default, or if set to
                `None`, the `numpy` fill value appropriate to the array's
                data-type will be used (see
                `np.ma.default_fill_value`). Ignored if the *source*
                parameter is set.

                The fill value may also be set after initialisation with
                the `set_fill_value` method.

                *Parameter example:*
                  ``fill_value=-999.``

            dtype: data-type, optional
                The desired data-type for the data. By default the
                data-type will be inferred form the *array* parameter.

                The data-type may also be set after initialisation with
                the `dtype` attribute.

                *Parameter example:*
                    ``dtype=float``

                *Parameter example:*
                    ``dtype='float32'``

                *Parameter example:*
                    ``dtype=np.dtype('i2')``

                .. versionadded:: 3.0.4

            mask: optional
                Apply this mask to the data given by the *array*
                parameter. By default, or if *mask* is `None`, no mask is
                applied. May be any scalar or array-like object (such as a
                `list`, `numpy` array or `Data` instance) that is
                broadcastable to the shape of *array*. Masking will be
                carried out where the mask elements evaluate to `True`.

                This mask will applied in addition to any mask already
                defined by the *array* parameter.

                .. versionadded:: 3.0.5

            source: optional
                Initialize the array, units, calendar and fill value from
                those of *source*.

            hardmask: `bool`, optional
                If False then the mask is soft. By default the mask is
                hard.

            dt: `bool`, optional
                If True then strings (such as ``'1990-12-01 12:00'``)
                given by the *array* parameter are re-interpreted as
                date-time objects. By default they are not.

            loadd: `dict`, optional
                Initialise the data from a dictionary serialization of a
                `cf.Data` object. All other arguments are ignored. See the
                `dumpd` and `loadd` methods.

            loads: `str`, optional
                Initialise the data array from a string serialization of a
                `Data` object. All other arguments are ignored. See the
                `dumps` and `loads` methods.

            copy: `bool`, optional
                If False then do not deep copy input parameters prior to
                initialization. By default arguments are deep copied.

            chunk: `bool`, optional
                If False then the data array will be stored in a single
                partition. By default the data array will be partitioned
                if it is larger than the chunk size, as returned by the
                `cf.chunksize` function.

        **Examples:**

        >>> d = cf.Data(5)
        >>> d = cf.Data([1,2,3], units='K')
        >>> import numpy
        >>> d = cf.Data(np.arange(10).reshape(2,5),
        ...             units=Units('m/s'), fill_value=-999)
        >>> d = cf.Data(tuple('fly'))

        """
        data = array

        super().__init__(source=source, fill_value=fill_value)

        if source is not None:
            partitions = self._custom.get("partitions")
            if partitions is not None:
                self.partitions = partitions.copy()

            auxiliary_mask = self._custom.get("_auxiliary_mask")
            if auxiliary_mask is not None:
                self._auxiliary_mask = [mask.copy() for mask in auxiliary_mask]

            return

        if not (loadd or loads):
            units = Units(units, calendar=calendar)
            self._Units = units

        empty_list = []

        # The _flip attribute is an unordered subset of the data
        # array's axis names. It is a subset of the axes given by the
        # _axes attribute. It is used to determine whether or not to
        # reverse an axis in each partition's sub-array during the
        # creation of the partition's data array. DO NOT CHANGE IN
        # PLACE.
        self._flip(empty_list)

        # The _all_axes attribute must be None or a tuple
        self._all_axes = None

        self.hardmask = hardmask

        # The _HDF_chunks attribute is.... Is either None or a
        # dictionary. DO NOT CHANGE IN PLACE.
        self._HDF_chunks = None

        # ------------------------------------------------------------
        # Attribute: _auxiliary_mask
        #
        # Must be None or a (possibly empty) list of Data objects.
        # ------------------------------------------------------------
        self._auxiliary_mask = None

        if loadd is not None:
            self.loadd(loadd, chunk=chunk)
            return

        if loads is not None:
            self.loads(loads, chunk=chunk)
            return

        # The _cyclic attribute contains the axes of the data array
        # which are cyclic (and therefore allow cyclic slicing). It is
        # a subset of the axes given by the _axes attribute. DO NOT
        # CHANGE IN PLACE.
        self._cyclic = _empty_set

        data = array

        if data is None:
            if dtype is not None:
                dtype = np.dtype(dtype)

            self._dtype = dtype
            return

        #        if not isinstance(data, Array):
        if not self._is_abstract_Array_subclass(data):
            check_free_memory = True

            if isinstance(data, self.__class__):
                # self.loadd(data.dumpd(), chunk=chunk)
                self.__dict__ = data.copy().__dict__
                if chunk:
                    self.chunk()

                if mask is not None:
                    self.where(mask, cf_masked, inplace=True)

                return

            if not isinstance(data, np.ndarray):
                data = np.asanyarray(data)

            if (
                data.dtype.kind == "O"
                and not dt
                and hasattr(data.item((0,) * data.ndim), "timetuple")
            ):
                # We've been given one or more date-time objects
                dt = True
        else:
            check_free_memory = False

        _dtype = data.dtype

        if dt or units.isreftime:
            # TODO raise exception if compressed
            kind = _dtype.kind
            if kind in "US":
                # Convert date-time strings to reference time floats
                if not units:
                    YMD = str(data.item((0,) * data.ndim)).partition("T")[0]
                    units = Units("days since " + YMD, units._calendar)
                    self._Units = units

                data = st2rt(data, units, units)
                _dtype = data.dtype
            elif kind == "O":
                # Convert date-time objects to reference time floats
                x = data.item(0)
                x_since = "days since " + "-".join(
                    map(str, (x.year, x.month, x.day))
                )
                x_calendar = getattr(x, "calendar", "gregorian")

                d_calendar = getattr(self.Units, "calendar", None)
                d_units = getattr(self.Units, "units", None)

                if x_calendar != "":
                    if d_calendar is not None:
                        if not self.Units.equivalent(
                            Units(x_since, x_calendar)
                        ):
                            raise ValueError(
                                "Incompatible units: {!r}, {!r}".format(
                                    self.Units, Units(x_since, x_calendar)
                                )
                            )
                    else:
                        d_calendar = x_calendar
                # --- End: if

                if not units:
                    # Set the units to something that is (hopefully)
                    # close to all of the datetimes, in an attempt to
                    # reduce errors arising from the conversion to
                    # reference times
                    units = Units(x_since, calendar=d_calendar)
                else:
                    units = Units(d_units, calendar=d_calendar)

                self._Units = units

                # Check that all date-time objects have correct and
                # equivalent calendars
                calendars = set(
                    [getattr(x, "calendar", "gregorian") for x in data.flat]
                )
                if len(calendars) > 1:
                    raise ValueError(
                        "Not all date-time objects have equivalent "
                        "calendars: {}".format(tuple(calendars))
                    )

                # If the date-times are calendar-agnostic, assign the
                # given calendar, defaulting to Gregorian.
                if calendars.pop() == "":
                    calendar = getattr(self.Units, "calendar", "gregorian")

                    new_data = np.empty(np.shape(data), dtype="O")
                    for i in np.ndindex(new_data.shape):
                        new_data[i] = cf_dt(data[i], calendar=calendar)

                    data = new_data

                # Convert the date-time objects to reference times
                data = dt2rt(data, None, units)

            _dtype = data.dtype

            if not units.isreftime:
                raise ValueError(
                    "Can't initialise a reference time array with "
                    "units {!r}".format(units)
                )
        # --- End: if

        shape = data.shape
        ndim = data.ndim
        size = data.size
        axes = _initialise_axes(ndim)

        # The _axes attribute is the ordered list of the data array's
        # axis names. Each axis name is an arbitrary, unique
        # string. DO NOT CHANGE IN PLACE.
        self._axes = axes

        self._ndim = ndim
        self._shape = shape
        self._size = size

        if dtype is not None:
            _dtype = np.dtype(dtype)

        self._dtype = _dtype

        self._set_partition_matrix(
            data, chunk=chunk, check_free_memory=check_free_memory
        )

        if mask is not None:
            self.where(mask, cf_masked, inplace=True)

    def __contains__(self, value):
        """Membership test operator ``in``

        x.__contains__(y) <==> y in x

        Returns True if the value is contained anywhere in the data
        array. The value may be a `cf.Data` object.

        **Examples:**

        >>> d = Data([[0.0, 1,  2], [3, 4, 5]], 'm')
        >>> 4 in d
        True
        >>> Data(3) in d
        True
        >>> Data([2.5], units='2 m') in d
        True
        >>> [[2]] in d
        True
        >>> np.array([[[2]]]) in d
        True
        >>> Data(2, 'seconds') in d
        False

        """
        if isinstance(value, self.__class__):
            self_units = self.Units
            value_units = value.Units
            if value_units.equivalent(self_units):
                if not value_units.equals(self_units):
                    value = value.copy()
                    value.Units = self_units
            elif value_units:
                return False

            value = value.array

        config = self.partition_configuration(readonly=True)

        for partition in self.partitions.matrix.flat:
            partition.open(config)
            array = partition.array
            partition.close()

            if value in array:
                return True
        # --- End: for

        return False

    def _is_abstract_Array_subclass(self, array):
        """Whether or not an array is a type of abstract Array.

        :Parameters:

            array:

        :Returns:

            `bool`

        """
        return isinstance(array, cfdm.Array)

    def _auxiliary_mask_from_1d_indices(self, compressed_indices):
        """Returns the auxiliary masks corresponding to given indices.

        :Parameters:

            compressed_indices:

        :Returns:

            `list` of `Data`
                The auxiliary masks in a list.

        """
        auxiliary_mask = []

        for i, (compressed_index, size) in enumerate(
            zip(compressed_indices, self._shape)
        ):

            if isinstance(
                compressed_index, slice
            ) and compressed_index.step in (-1, 1):
                # Compressed index is a slice object with a step of
                # +-1 => no auxiliary mask required for this axis
                continue

            index = np.zeros(size, dtype=bool)
            index[compressed_index] = True

            compressed_size = index.sum()

            ind = np.where(index)

            ind0 = ind[0]
            start = ind0[0]
            envelope_size = ind0[-1] - start + 1

            if 0 < compressed_size < envelope_size:
                jj = [None] * self._ndim
                jj[i] = envelope_size

                if start:
                    ind0 -= start

                mask = self._auxiliary_mask_component(jj, ind, True)
                auxiliary_mask.append(mask)
        # --- End: for

        return auxiliary_mask

    def _auxiliary_mask_return(self):
        """Return the auxiliary mask.

        :Returns:

            `Data` or `None`
                The auxiliary mask, or `None` if there isn't one.

        **Examples:**

        >>> m = d._auxiliary_mask_return()

        """
        _auxiliary_mask = self._auxiliary_mask
        if not _auxiliary_mask:
            shape = getattr(self, "shape", None)
            if shape is not None:
                return type(self).full(shape, fill_value=False, dtype=bool)
            else:
                return None
        # --- End: if

        mask = _auxiliary_mask[0]
        for m in _auxiliary_mask[1:]:
            mask = mask | m

        return mask

    def _auxiliary_mask_add_component(self, mask):
        """Add a new auxiliary mask.

        :Parameters:

            mask: `cf.Data` or `None`

        :Returns:

            `None`

        **Examples:**

        >>> d._auxiliary_mask_add_component(m)

        """
        if mask is None:
            return

        # Check that this mask component has the correct number of
        # dimensions
        if mask.ndim != self._ndim:
            raise ValueError(
                "Auxiliary mask must have same number of axes as the data "
                "array ({}!={})".format(mask.ndim, self.ndim)
            )

        # Check that this mask component has an appropriate shape
        mask_shape = mask.shape
        for i, j in zip(mask_shape, self._shape):
            if not (i == j or i == 1):
                raise ValueError(
                    "Auxiliary mask shape {} is not broadcastable to data "
                    "array shape {}".format(mask.shape, self._shape)
                )

        # Merge this mask component with another, if possible.
        append = True
        if self._auxiliary_mask is not None:
            for m0 in self._auxiliary_mask:
                if m0.shape == mask_shape:
                    # Merging the new mask with an existing auxiliary
                    # mask component
                    m0 |= mask
                    append = False
                    break
        # --- End: if

        if append:
            mask = mask.copy()

            # Make sure that the same axes are cyclic for the data
            # array and the auxiliary mask
            indices = [self._axes.index(axis) for axis in self._cyclic]
            mask._cyclic = set([mask._axes[i] for i in indices])

            if self._auxiliary_mask is None:
                self._auxiliary_mask = [mask]
            else:
                self._auxiliary_mask.append(mask)

    def _auxiliary_mask_subspace(self, indices):
        """Subspace the new auxiliary mask.

        :Returns:

            `None`

        **Examples:**

        >>> d._auxiliary_mask_subspace((slice(0, 9, 2)))

        """
        if not self._auxiliary_mask:
            # There isn't an auxiliary mask
            return

        new = []
        for mask in self._auxiliary_mask:
            mask_indices = [
                (slice(None) if n == 1 else index)
                for n, index in zip(mask.shape, indices)
            ]
            new.append(mask[tuple(mask_indices)])

        self._auxiliary_mask = new

    def _create_auxiliary_mask_component(self, mask_shape, ind, compress):
        """Create a new auxiliary mask component of given shape.

        :Parameters:

            mask_shape: `tuple`
                The shape of the mask component to be created. This will
                contain `None` for axes not spanned by the *ind*
                parameter.

                  *Parameter example*
                      ``mask_shape=(3, 11, None)``

            ind: `np.ndarray`
                As returned by a single argument call of
                ``np.array(numpy[.ma].where(....))``.

            compress: `bool`
                If True then remove whole slices which only contain masked
                points.

        :Returns:

            `Data`

        """
        # --------------------------------------------------------
        # Find the shape spanned by ind
        # --------------------------------------------------------
        shape = [n for n in mask_shape if n]

        # Note that, for now, auxiliary_mask has to be numpy array
        # (rather than a cf.Data object) because we're going to index
        # it with fancy indexing which a cf.Data object might not
        # support - namely a non-monotonic list of integers.
        auxiliary_mask = np.ones(shape, dtype=bool)

        auxiliary_mask[tuple(ind)] = False

        if compress:
            # For compressed indices, remove slices which only
            # contain masked points. (Note that we only expect to
            # be here if there were N-d item criteria.)
            for iaxis, (index, n) in enumerate(zip(ind, shape)):
                index = set(index)
                if len(index) < n:
                    auxiliary_mask = auxiliary_mask.take(
                        sorted(index), axis=iaxis
                    )
        # --- End: if

        # Add missing size 1 axes to the auxiliary mask
        if auxiliary_mask.ndim < self.ndim:
            i = [(slice(None) if n else np.newaxis) for n in mask_shape]
            auxiliary_mask = auxiliary_mask[tuple(i)]

        return type(self)(auxiliary_mask)

    def _auxiliary_mask_tidy(self):
        """Remove unnecessary auxiliary mask components.

        :Returns:

            `None`

        **Examples:**

        >>> d._auxiliary_mask_tidy()

        """
        auxiliary_mask = self._auxiliary_mask
        if auxiliary_mask:
            # Get rid of auxiliary mask components which are all False
            auxiliary_mask = [m for m in auxiliary_mask if m.any()]
            if not auxiliary_mask:
                auxiliary_mask = None
        else:
            auxiliary_mask = None

        self._auxiliary_mask = auxiliary_mask

    def __data__(self):
        """Returns a new reference to self."""
        return self

    def __hash__(self):
        """The built-in function `hash`

        Generating the hash temporarily realizes the entire array in
        memory, which may not be possible for large arrays.

        The hash value is dependent on the data-type and shape of the data
        array. If the array is a masked array then the hash value is
        independent of the fill value and of data array values underlying
        any masked elements.

        The hash value may be different if regenerated after the data
        array has been changed in place.

        The hash value is not guaranteed to be portable across versions of
        Python, numpy and cf.

        :Returns:

            `int`
                The hash value.

        **Examples:**

        >>> print(d.array)
        [[0 1 2 3]]
        >>> d.hash()
        -8125230271916303273
        >>> d[1, 0] = np.ma.masked
        >>> print(d.array)
        [[0 -- 2 3]]
        >>> hash(d)
        791917586613573563
        >>> d.hardmask = False
        >>> d[0, 1] = 999
        >>> d[0, 1] = np.ma.masked
        >>> d.hash()
        791917586613573563
        >>> d.squeeze()
        >>> print(d.array)
        [0 -- 2 3]
        >>> hash(d)
        -7007538450787927902
        >>> d.dtype = float
        >>> print(d.array)
        [0.0 -- 2.0 3.0]
        >>> hash(d)
        -4816859207969696442

        """
        return hash_array(self.array)

    def __float__(self):
        """Called to implement the built-in function `float`

        x.__float__() <==> float(x)

        """
        if self.size != 1:
            raise TypeError(
                "only length-1 arrays can be converted to Python scalars"
            )

        return float(self.datum())

    def __round__(self, *ndigits):
        """Called to implement the built-in function `round`

        x.__round__(*ndigits) <==> round(x, *ndigits)

        """
        if self.size != 1:
            raise TypeError(
                "only length-1 arrays can be converted to Python scalars"
            )

        return round(self.datum(), *ndigits)

    def __int__(self):
        """Called to implement the built-in function `int`

        x.__int__() <==> int(x)

        """
        if self.size != 1:
            raise TypeError(
                "only length-1 arrays can be converted to Python scalars"
            )

        return int(self.datum())

    def __iter__(self):
        """Called when an iterator is required.

        x.__iter__() <==> iter(x)

        **Examples:**

        >>> d = cf.Data([1, 2, 3], 'metres')
        >>> for e in d:
        ...    print(repr(e))
        ...
        1
        2
        3

        >>> d = cf.Data([[1, 2], [4, 5]], 'metres')
        >>> for e in d:
        ...    print(repr(e))
        ...
        <CF Data: [1, 2] metres>
        <CF Data: [4, 5] metres>

        >>> d = cf.Data(34, 'metres')
        >>> for e in d:
        ...     print(repr(e))
        ..
        TypeError: iteration over a 0-d Data

        """
        ndim = self._ndim

        if not ndim:
            raise TypeError(
                "Iteration over 0-d {}".format(self.__class__.__name__)
            )

        elif ndim == 1:
            if self.fits_in_memory(self.dtype.itemsize):
                i = iter(self.array)
                while 1:
                    try:
                        yield next(i)
                    except StopIteration:
                        return
            else:
                for n in range(self._size):
                    yield self[n].array[0]

        else:
            # ndim > 1
            for n in range(self._shape[0]):
                out = self[n, ...]
                out.squeeze(0, inplace=True)
                yield out

    def __len__(self):
        """The built-in function `len`

        x.__len__() <==> len(x)

        **Examples:**

        >>> len(Data([1, 2, 3]))
        3
        >>> len(Data([[1, 2, 3]]))
        1
        >>> len(Data([[1, 2, 3], [4, 5, 6]]))
        2
        >>> len(Data(1))
        TypeError: len() of scalar Data

        """
        shape = self._shape
        if shape:
            return shape[0]

        raise TypeError("len() of scalar {}".format(self.__class__.__name__))

    def __bool__(self):
        """Truth value testing and the built-in operation `bool`

        x.__bool__() <==> bool(x)

        **Examples:**

        >>> bool(Data(1))
        True
        >>> bool(Data([[False]]))
        False
        >>> bool(Data([1, 2]))
        ValueError: The truth value of Data with more than one element is ambiguous. Use d.any() or d.all()

        """
        if self._size == 1:
            return bool(self.array)

        raise ValueError(
            "The truth value of Data with more than one element is "
            "ambiguous. Use d.any() or d.all()"
        )

    def __repr__(self):
        """Called by the `repr` built-in function.

        x.__repr__() <==> repr(x)

        """
        return super().__repr__().replace("<", "<CF ", 1)

    def __getitem__(self, indices):
        """Return a subspace of the data defined by indices.

        d.__getitem__(indices) <==> d[indices]

        Indexing follows rules that are very similar to the numpy indexing
        rules, the only differences being:

        * An integer index i takes the i-th element but does not reduce
          the rank by one.

        * When two or more dimensions' indices are sequences of integers
          then these indices work independently along each dimension
          (similar to the way vector subscripts work in Fortran). This is
          the same behaviour as indexing on a `netCDF4.Variable` object.

        . seealso:: `__setitem__`, `_parse_indices`

        :Returns:

            `Data`
                The subspace of the data.

        **Examples:**

        >>> import numpy
        >>> d = Data(np.arange(100, 190).reshape(1, 10, 9))
        >>> d.shape
        (1, 10, 9)
        >>> d[:, :, 1].shape
        (1, 10, 1)
        >>> d[:, 0].shape
        (1, 1, 9)
        >>> d[..., 6:3:-1, 3:6].shape
        (1, 3, 3)
        >>> d[0, [2, 9], [4, 8]].shape
        (1, 2, 2)
        >>> d[0, :, -2].shape
        (1, 10, 1)

        """
        size = self._size

        if indices is Ellipsis:
            return self.copy()

        d = self

        axes = d._axes
        flip = d._flip()
        shape = d._shape
        cyclic_axes = d._cyclic
        auxiliary_mask = []

        try:
            arg0 = indices[0]
        except (IndexError, TypeError):
            pass
        else:
            if isinstance(arg0, str) and arg0 == "mask":
                auxiliary_mask = indices[1]
                indices = tuple(indices[2:])
            else:
                pass

        indices, roll, flip_axes = parse_indices(shape, indices, True, True)

        if roll:
            for axis, shift in roll.items():
                if axes[axis] not in cyclic_axes:
                    raise IndexError(
                        "Can't take a cyclic slice of a non-cyclic "
                        "axis (axis position {})".format(axis)
                    )

                d = d.roll(axis, shift)
        # --- End: if

        new_shape = tuple(map(_size_of_index, indices, shape))
        new_size = functools.reduce(operator_mul, new_shape, 1)

        new = d.copy()  # Data.__new__(Data)

        source = new.source(None)
        if source is not None and source.get_compression_type():
            new._del_Array(None)

        #        new.get_fill_value = d.get_fill_value(None)

        new._shape = new_shape
        new._size = new_size
        #        new._ndim       = d._ndim
        #        new.hardmask   = d.hardmask
        #        new._all_axes   = d._all_axes
        #        new._cyclic     = cyclic_axes
        #        new._axes       = axes
        #        new._flip       = flip
        #        new._dtype      = d.dtype
        #        new._HDF_chunks = d._HDF_chunks
        #        new._Units      = d._Units
        #        new._auxiliary_mask = d._auxiliary_mask

        partitions = d.partitions

        new_partitions = PartitionMatrix(
            _overlapping_partitions(partitions, indices, axes, flip),
            partitions.axes,
        )

        if new_size != size:
            new_partitions.set_location_map(axes)

        new.partitions = new_partitions

        # ------------------------------------------------------------
        # Index an existing auxiliary mask. (Note: By now indices are
        # strictly monotonically increasing and don't roll.)
        # ------------------------------------------------------------
        new._auxiliary_mask_subspace(indices)

        # --------------------------------------------------------
        # Apply the input auxiliary mask
        # --------------------------------------------------------
        for mask in auxiliary_mask:
            new._auxiliary_mask_add_component(mask)

        # ------------------------------------------------------------
        # Tidy up the auxiliary mask
        # ------------------------------------------------------------
        new._auxiliary_mask_tidy()

        # ------------------------------------------------------------
        # Flip axes
        # ------------------------------------------------------------
        if flip_axes:
            new.flip(flip_axes, inplace=True)

        # ------------------------------------------------------------
        # Mark cyclic axes which have been reduced in size as
        # non-cyclic
        # ------------------------------------------------------------
        if cyclic_axes:
            x = [
                i
                for i, (axis, n0, n1) in enumerate(zip(axes, shape, new_shape))
                if n1 != n0 and axis in cyclic_axes
            ]
            if x:
                new._cyclic = cyclic_axes.difference(x)
        # --- End: if

        # -------------------------------------------------------------
        # Remove size 1 axes from the partition matrix
        # -------------------------------------------------------------
        new_partitions.squeeze(inplace=True)

        return new

    def __setitem__(self, indices, value):
        """Implement indexed assignment.

        x.__setitem__(indices, y) <==> x[indices]=y

        Assignment to data array elements defined by indices.

        Elements of a data array may be changed by assigning values to a
        subspace. See `__getitem__` for details on how to define subspace
        of the data array.

        **Missing data**

        The treatment of missing data elements during assignment to a
        subspace depends on the value of the `hardmask` attribute. If it
        is True then masked elements will not be unmasked, otherwise masked
        elements may be set to any value.

        In either case, unmasked elements may be set, (including missing
        data).

        Unmasked elements may be set to missing data by assignment to the
        `cf.masked` constant or by assignment to a value which contains
        masked elements.

        .. seealso:: `cf.masked`, `hardmask`, `where`

        **Examples:**

        """

        def _mirror_slice(index, size):
            """Return a slice object which creates the reverse of the
            input slice.

            The step of the input slice must have a step of `.

            :Parameters:

                index: `slice`
                    A slice object with a step of 1.

                size: `int`

            :Returns:

                `slice`

            **Examples:**

            >>> s = slice(2, 6)
            >>> t = _mirror_slice(s, 8)
            >>> s, t
            slice(2, 6), slice(5, 1, -1)
            >>> range(8)[s]
            [2, 3, 4, 5]
            >>> range(8)[t]
            [5, 4, 3, 2]
            >>> range(7, -1, -1)[s]
            [5, 4, 3, 2]
            >>> range(7, -1, -1)[t]
            [2, 3, 4, 5]

            """
            start, stop, step = index.indices(size)
            size -= 1
            start = size - start
            stop = size - stop
            if stop < 0:
                stop = None

            return slice(start, stop, -1)

        # --- End: def

        config = self.partition_configuration(readonly=False)

        # ------------------------------------------------------------
        # parse the indices
        # ------------------------------------------------------------
        indices_in = indices
        indices, roll, flip_axes, mask = parse_indices(
            self._shape, indices_in, cyclic=True, reverse=True, mask=True
        )

        if roll:
            for iaxis, shift in roll.items():
                self.roll(iaxis, shift, inplace=True)
        # --- End: if

        if mask:
            original_self = self.copy()

        scalar_value = False
        if value is cf_masked:
            scalar_value = True
        else:
            if not isinstance(value, Data):
                # Convert to the value to a Data object
                value = type(self)(value, self.Units)
            else:
                if value.Units.equivalent(self.Units):
                    if not value.Units.equals(self.Units):
                        value = value.copy()
                        value.Units = self.Units
                elif not value.Units:
                    value = value.override_units(self.Units)
                else:
                    raise ValueError(
                        "Can't assign values with units {!r} to data with "
                        "units {!r}".format(value.Units, self.Units)
                    )
            # --- End: if

            if value._size == 1:
                scalar_value = True
                value = value.datum(0)
        # --- End: if

        source = self.source(None)
        if source is not None and source.get_compression_type():
            self._del_Array(None)

        if scalar_value:
            # --------------------------------------------------------
            # The value is logically scalar
            # --------------------------------------------------------
            for partition in self.partitions.matrix.flat:
                p_indices, shape = partition.overlaps(indices)
                if p_indices is None:
                    # This partition does not overlap the indices
                    continue

                partition.open(config)
                array = partition.array

                if value is cf_masked and not partition.masked:
                    # The assignment is masking elements, so turn a
                    # non-masked sub-array into a masked one.
                    array = array.view(np.ma.MaskedArray)
                    partition.subarray = array

                self._set_subspace(array, p_indices, value)
                partition.close()

            if roll:
                for iaxis, shift in roll.items():
                    self.roll(iaxis, -shift, inplace=True)
            # --- End: if

            if mask:
                indices = tuple(indices)
                original_self = original_self[indices]
                u = self[indices]
                for m in mask:
                    u.where(m, original_self, inplace=True)

                self[indices] = u
            # --- End: if

            return

        # ------------------------------------------------------------
        # Still here? Then the value is not logically scalar.
        # ------------------------------------------------------------
        data0_shape = self._shape
        value_shape = value._shape

        shape00 = list(map(_size_of_index, indices, data0_shape))
        shape0 = shape00[:]

        self_ndim = self._ndim
        value_ndim = value._ndim
        align_offset = self_ndim - value_ndim
        if align_offset >= 0:
            # self has more dimensions than other
            shape0 = shape0[align_offset:]
            shape1 = value_shape
            ellipsis = None

            flip_axes = [
                i - align_offset for i in flip_axes if i >= align_offset
            ]
        else:
            # value has more dimensions than self
            v_align_offset = -align_offset
            if value_shape[:v_align_offset] != [1] * v_align_offset:
                # Can only allow value to have more dimensions then
                # self if the extra dimensions all have size 1.
                raise ValueError(
                    "Can't broadcast shape %r across shape %r"
                    % (value_shape, data0_shape)
                )

            shape1 = value_shape[v_align_offset:]
            ellipsis = Ellipsis
            align_offset = 0

        # Find out which of the dimensions of value are to be
        # broadcast, and those which are not. Note that, as in numpy,
        # it is not allowed for a dimension in value to be larger than
        # a size 1 dimension in self
        base_value_indices = []
        non_broadcast_dimensions = []

        for i, (a, b) in enumerate(zip(shape0, shape1)):
            if b == 1:
                base_value_indices.append(slice(None))
            elif a == b and b != 1:
                base_value_indices.append(None)
                non_broadcast_dimensions.append(i)
            else:
                raise ValueError(
                    "Can't broadcast data with shape {!r} across "
                    "shape {!r}".format(shape1, tuple(shape00))
                )
        # --- End: for

        previous_location = ((-1,),) * self_ndim
        start = [0] * value_ndim

        #        save = pda_args['save']
        #        keep_in_memory = pda_args['keep_in_memory']

        value.to_memory()

        for partition in self.partitions.matrix.flat:
            p_indices, shape = partition.overlaps(indices)

            if p_indices is None:
                # This partition does not overlap the indices
                continue

            # --------------------------------------------------------
            # Find which elements of value apply to this partition
            # --------------------------------------------------------
            value_indices = base_value_indices[:]

            for i in non_broadcast_dimensions:
                j = i + align_offset
                location = partition.location[j][0]
                reference_location = previous_location[j][0]

                if location > reference_location:
                    stop = start[i] + shape[j]
                    value_indices[i] = slice(start[i], stop)
                    start[i] = stop

                elif location == reference_location:
                    value_indices[i] = previous_slice[i]  # noqa F821

                elif location < reference_location:
                    stop = shape[j]
                    value_indices[i] = slice(0, stop)
                    start[i] = stop
            # --- End: for

            previous_location = partition.location
            previous_slice = value_indices[:]  # noqa F821

            for i in flip_axes:
                value_indices[i] = _mirror_slice(value_indices[i], shape1[i])

            if ellipsis:
                value_indices.insert(0, ellipsis)

            # --------------------------------------------------------
            #
            # --------------------------------------------------------
            v = value[tuple(value_indices)].varray

            #            if keep_in_memory: #not save:
            #                v = v.copy()

            # Update the partition's data
            partition.open(config)
            array = partition.array

            if not partition.masked and np.ma.isMA(v):
                # The sub-array is not masked and the assignment is
                # masking elements, so turn the non-masked sub-array
                # into a masked one.
                array = array.view(np.ma.MaskedArray)
                partition.subarray = array

            self._set_subspace(array, p_indices, v)

            partition.close()
        # --- End: For

        if roll:
            # Unroll
            for iaxis, shift in roll.items():
                self.roll(iaxis, -shift, inplace=True)
        # --- End: if

        if mask:
            indices = tuple(indices)
            original_self = original_self[indices]
            u = self[indices]
            for m in mask:
                u.where(m, original_self, inplace=True)

            self[indices] = u

    def dumps(self):
        """Return a JSON string serialization of the data array."""
        d = self.dumpd()

        # Change a set to a list
        if "_cyclic" in d:
            d["_cyclic"] = list(d["_cyclic"])

        # Change np.dtype object to a data-type string
        if "dtype" in d:
            d["dtype"] = str(d["dtype"])

        # Change a Units object to a units string
        if "Units" in d:
            d["units"] = str(d.pop("Units"))

        #
        for p in d["Partitions"]:
            if "Units" in p:
                p["units"] = str(p.pop("Units"))
        # --- End: for

        return json_dumps(d, default=_convert_to_builtin_type)


    def loads(self, j, chunk=True):
        """Reset the data in place from a string serialization.

        .. seealso:: `dumpd`, `loadd`

        :Parameters:

            j: `str`
                A JSON document string serialization of a `cf.Data` object.

            chunk: `bool`, optional
                If True (the default) then the reset data array will be
                re-partitioned according the current chunk size, as defined
                by the `cf.chunksize` function.

        :Returns:

            `None`

        """
        d = json_loads(j)

        # Convert _cyclic to a set
        if "_cyclic" in d:
            d["_cyclic"] = set(d["_cyclic"])

        # Convert dtype to np.dtype
        if "dtype" in d:
            d["dtype"] = np.dtype(d["dtype"])

        # Convert units to Units
        if "units" in d:
            d["Units"] = Units(d.pop("units"))

        # Convert partition location elements to tuples
        for p in d["Partitions"]:
            p["location"] = [tuple(x) for x in p["location"]]

            if "units" in p:
                p["Units"] = Units(p.pop("units"))
        # --- End: for

        self.loadd(d, chunk=chunk)

    def dumpd(self):
        """Return a serialization of the data array.

        The serialization may be used to reconstruct the data array as it
        was at the time of the serialization creation.

        .. seealso:: `loadd`, `loads`

        :Returns:

            `dict`
                The serialization.

        **Examples:**

        >>> d = cf.Data([[1, 2, 3]], 'm')
        >>> d.dumpd()
        {'Partitions': [{'location': [(0, 1), (0, 3)],
                         'subarray': array([[1, 2, 3]])}],
         'units': 'm',
         '_axes': ['dim0', 'dim1'],
         '_pmshape': (),
         'dtype': dtype('int64'),
         'shape': (1, 3)}

        >>> d.flip(1)
        >>> d.transpose()
        >>> d.Units *= 1000
        >>> d.dumpd()
        {'Partitions': [{'units': 'm',
                         'axes': ['dim0', 'dim1'],
                         'location': [(0, 3), (0, 1)],
                         'subarray': array([[1, 2, 3]])}],
        ` 'units': '1000 m',
         '_axes': ['dim1', 'dim0'],
         '_flip': ['dim1'],
         '_pmshape': (),
         'dtype': dtype('int64'),
         'shape': (3, 1)}

        >>> d.dumpd()
        {'Partitions': [{'units': 'm',
                         'location': [(0, 1), (0, 3)],
                         'subarray': array([[1, 2, 3]])}],
         'units': '10000 m',
         '_axes': ['dim0', 'dim1'],
         '_flip': ['dim1'],
         '_pmshape': (),
         'dtype': dtype('int64'),
         'shape': (1, 3)}

        >>> e = cf.Data(loadd=d.dumpd())
        >>> e.equals(d)
        True

        """
        axes = self._axes
        units = self.Units
        dtype = self.dtype

        cfa_data = {
            "dtype": dtype,
            "Units": str(units),
            "shape": self._shape,
            "_axes": axes[:],
            "_pmshape": self._pmshape,
        }

        pmaxes = self._pmaxes
        if pmaxes:
            cfa_data["_pmaxes"] = pmaxes[:]

        #        flip = self._flip
        flip = self._flip()
        if flip:
            cfa_data["_flip"] = flip[:]

        fill_value = self.get_fill_value(None)
        if fill_value is not None:
            cfa_data["fill_value"] = fill_value

        cyclic = self._cyclic
        if cyclic:
            cfa_data["_cyclic"] = cyclic.copy()

        HDF_chunks = self._HDF_chunks
        if HDF_chunks:
            cfa_data["_HDF_chunks"] = HDF_chunks.copy()

        partitions = []
        for index, partition in self.partitions.ndenumerate():

            attrs = {}

            p_subarray = partition.subarray
            p_dtype = p_subarray.dtype

            # Location in partition matrix
            if index:
                attrs["index"] = index

            # Sub-array location
            attrs["location"] = partition.location[:]

            # Sub-array part
            p_part = partition.part
            if p_part:
                attrs["part"] = p_part[:]

            # Sub-array axes
            p_axes = partition.axes
            if p_axes != axes:
                attrs["axes"] = p_axes[:]

            # Sub-array units
            p_Units = partition.Units
            if p_Units != units:
                attrs["Units"] = str(p_Units)

            # Sub-array flipped axes
            p_flip = partition.flip
            if p_flip:
                attrs["flip"] = p_flip[:]

            # --------------------------------------------------------
            # File format specific stuff
            # --------------------------------------------------------
            if isinstance(p_subarray, NetCDFArray):
                # if isinstance(p_subarray.array, NetCDFFileArray):
                # ----------------------------------------------------
                # NetCDF File Array
                # ----------------------------------------------------
                attrs["format"] = "netCDF"

                subarray = {}

                subarray["file"] = p_subarray.get_filename()
                subarray["shape"] = p_subarray.shape

                #                for attr in ('file', 'shape'):
                #                    subarray[attr] = getattr(p_subarray, attr)

                subarray["ncvar"] = p_subarray.get_ncvar()
                subarray["varid"] = p_subarray.get_varid()
                #                for attr in ('ncvar', 'varid'):
                #                    value = getattr(p_subarray, attr, None)
                # #                    value = getattr(p_subarray.array, attr, None)
                # #                    p_subarray.array.inspect()
                #
                #                    if value is not None:
                #                        subarray[attr] = value
                # --- End: for

                if p_dtype != dtype:
                    subarray["dtype"] = p_dtype

                attrs["subarray"] = subarray

            elif isinstance(p_subarray, UMArray):
                # elif isinstance(p_subarray.array, UMFileArray):
                # ----------------------------------------------------
                # UM File Array
                # ----------------------------------------------------
                attrs["format"] = "UM"

                subarray = {}
                for attr in (
                    "filename",
                    "shape",
                    "header_offset",
                    "data_offset",
                    "disk_length",
                ):
                    subarray[attr] = getattr(p_subarray, attr)

                if p_dtype != dtype:
                    subarray["dtype"] = p_dtype

                attrs["subarray"] = subarray
            else:
                attrs["subarray"] = p_subarray
            #                attrs['subarray'] = p_subarray.array

            partitions.append(attrs)
        # --- End: for

        cfa_data["Partitions"] = partitions

        # ------------------------------------------------------------
        # Auxiliary mask
        # ------------------------------------------------------------
        if self._auxiliary_mask:
            cfa_data["_auxiliary_mask"] = [
                m.copy() for m in self._auxiliary_mask
            ]

        return cfa_data

    def loadd(self, d, chunk=True):
        """Reset the data in place from a dictionary serialization.

        .. seealso:: `dumpd`, `loads`

        :Parameters:

            d: `dict`
                A dictionary serialization of a `cf.Data` object, such as
                one as returned by the `dumpd` method.

            chunk: `bool`, optional
                If True (the default) then the reset data array will be
                re-partitioned according the current chunk size, as
                defined by the `cf.chunksize` function.

        :Returns:

            `None`

        **Examples:**

        >>> d = Data([[1, 2, 3]], 'm')
        >>> e = Data([6, 7, 8, 9], 's')
        >>> e.loadd(d.dumpd())
        >>> e.equals(d)
        True
        >>> e is d
        False

        >>> e = Data(loadd=d.dumpd())
        >>> e.equals(d)
        True

        """
        axes = list(d.get("_axes", ()))
        shape = tuple(d.get("shape", ()))

        units = d.get("Units", None)
        if units is None:
            units = Units()
        else:
            units = Units(units)

        dtype = d["dtype"]
        self._dtype = dtype
        #        print ('P45 asdasdasds', dtype)
        self.Units = units
        self._axes = axes

        self._flip(list(d.get("_flip", ())))
        self.set_fill_value(d.get("fill_value", None))

        self._shape = shape
        self._ndim = len(shape)
        self._size = functools.reduce(operator_mul, shape, 1)

        cyclic = d.get("_cyclic", None)
        if cyclic:
            self._cyclic = cyclic.copy()
        else:
            self._cyclic = _empty_set

        HDF_chunks = d.get("_HDF_chunks", None)
        if HDF_chunks:
            self._HDF_chunks = HDF_chunks.copy()
        else:
            self._HDF_chunks = None

        filename = d.get("file", None)
        #        if filename is not None:

        #            filename = abspath(filename)

        base = d.get("base", None)
        #        if base is not None:
        #            base = abspath(base)

        # ------------------------------------------------------------
        # Initialise an empty partition array
        # ------------------------------------------------------------
        partition_matrix = PartitionMatrix(
            np.empty(d.get("_pmshape", ()), dtype=object),
            list(d.get("_pmaxes", ())),
        )
        pmndim = partition_matrix.ndim

        # ------------------------------------------------------------
        # Fill the partition array with partitions
        # ------------------------------------------------------------
        for attrs in d["Partitions"]:

            # Find the position of this partition in the partition
            # matrix
            if "index" in attrs:
                index = attrs["index"]
                if len(index) == 1:
                    index = index[0]
                else:
                    index = tuple(index)
            else:
                index = (0,) * pmndim

            location = attrs.get("location", None)
            if location is not None:
                location = location[:]
            else:
                # Default location
                location = [[0, i] for i in shape]

            p_units = attrs.get("p_units", None)
            if p_units is None:
                p_units = units
            else:
                p_units = Units(p_units)

            partition = Partition(
                location=location,
                axes=attrs.get("axes", axes)[:],
                flip=attrs.get("flip", [])[:],
                Units=p_units,
                part=attrs.get("part", [])[:],
            )

            fmt = attrs.get("format", None)
            if fmt is None:
                # ----------------------------------------------------
                # Subarray is effectively a numpy array in memory
                # ----------------------------------------------------
                partition.subarray = attrs["subarray"]

            else:
                # ----------------------------------------------------
                # Subarray is in a file on disk
                # ----------------------------------------------------
                partition.subarray = attrs["subarray"]
                if fmt not in ("netCDF", "UM"):
                    raise TypeError(
                        "Don't know how to load sub-array from file "
                        "format {!r}".format(fmt)
                    )

                # Set the 'subarray' attribute
                kwargs = attrs["subarray"].copy()

                kwargs["shape"] = tuple(kwargs["shape"])

                kwargs["ndim"] = len(kwargs["shape"])
                kwargs["size"] = functools.reduce(
                    operator_mul, kwargs["shape"], 1
                )

                kwargs.setdefault("dtype", dtype)

                if "file" in kwargs:
                    f = kwargs["file"]
                    if f == "":
                        kwargs["filename"] = filename
                    else:
                        if base is not None:
                            f = pathjoin(base, f)

                        kwargs["filename"] = f
                else:
                    kwargs["filename"] = filename

                del kwargs["file"]

                if fmt == "netCDF":
                    partition.subarray = NetCDFArray(**kwargs)
                elif fmt == "UM":
                    partition.subarray = UMArray(**kwargs)
            # --- End: if

            # Put the partition into the partition array
            partition_matrix[index] = partition
        # --- End: for

        # Save the partition array
        self.partitions = partition_matrix

        if chunk:
            self.chunk()

        # ------------------------------------------------------------
        # Auxiliary mask
        # ------------------------------------------------------------
        _auxiliary_mask = d.get("_auxiliary_mask", None)
        if _auxiliary_mask:
            self._auxiliary_mask = [m.copy() for m in _auxiliary_mask]
        else:
            self._auxiliary_mask = None

    @_inplace_enabled(default=False)
    def _asdatetime(self, inplace=False):
        """Change the internal representation of data array elements
        from numeric reference times to datetime-like objects.

        If the calendar has not been set then the default CF calendar will
        be used and the units' and the `calendar` attribute will be
        updated accordingly.

        If the internal representations are already datetime-like objects
        then no change occurs.

        .. versionadded:: 1.3

        .. seealso:: `_asreftime`, `_isdatetime`

        :Parameters:

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples:**

        >>> d._asdatetime()

        """
        d = _inplace_enabled_define_and_cleanup(self)
        units = self.Units

        if not units.isreftime:
            raise ValueError(
                "Can't convert {!r} data to date-time objects".format(units)
            )

        if d._isdatetime():
            if inplace:
                d = None
            return d

        config = d.partition_configuration(
            readonly=False, func=rt2dt, dtype=None
        )

        for partition in d.partitions.matrix.flat:
            partition.open(config)
            array = partition.array
            p_units = partition.Units
            partition.Units = Units(p_units.units, p_units._utime.calendar)
            partition.close()

        d.Units = Units(units.units, units._utime.calendar)

        d._dtype = array.dtype

        return d

    def _isdatetime(self):
        """True if the internal representation is a datetime-like
        object."""
        return self.dtype.kind == "O" and self.Units.isreftime

    @_inplace_enabled(default=False)
    def _asreftime(self, inplace=False):
        """Change the internal representation of data array elements
        from datetime-like objects to numeric reference times.

        If the calendar has not been set then the default CF calendar will
        be used and the units' and the `calendar` attribute will be
        updated accordingly.

        If the internal representations are already numeric reference
        times then no change occurs.

        .. versionadded:: 1.3

        .. seealso:: `_asdatetime`, `_isdatetime`

        :Parameters:

            {{inplace: `bool`, optional}}

        :Returns:

            `Data` or `None`

        **Examples:**

        >>> d._asreftime()

        """
        d = _inplace_enabled_define_and_cleanup(self)
        units = d.Units

        if not d._isdatetime():
            if units.isreftime:
                if inplace:
                    d = None
                return d
            else:
                raise ValueError(
                    "Can't convert {!r} data to numeric reference "
                    "times".format(units)
                )
        # --- End: if

        config = d.partition_configuration(
            readonly=False, func=dt2rt, dtype=None
        )

        for partition in d.partitions.matrix.flat:
            partition.open(config)
            array = partition.array
            p_units = partition.Units
            partition.Units = Units(p_units.units, p_units._utime.calendar)
            partition.close()

        d.Units = Units(units.units, units._utime.calendar)

        d._dtype = array.dtype

        return d

    def __query_set__(self, values):
        """Implements the “member of set” condition."""
        i = iter(values)
        v = next(i)

        out = self == v
        for v in i:
            out |= self == v

        return out

    #        new = self.copy()
    #
    #        pda_args = new.pda_args(revert_to_file=True)
    #
    #        for partition in new.partitions.matrix.flat:
    #            array = partition.dataarray(**pda_args)
    #
    #            i = iter(values)
    #            value = next(i)
    #            out = (array == value)
    #            for value in i:
    #                out |= (array == value)
    #
    #            partition.subarray = out
    #            partition.close()
    #        # --- End: for
    #
    #        new.dtype = bool
    #
    #        return new

    def __query_wi__(self, value):
        """Implements the “within a range” condition."""
        return (self >= value[0]) & (self <= value[1])

    #        new = self.copy()
    #
    #        pda_args = new.pda_args(revert_to_file=True)
    #
    #        for partition in new.partitions.matrix.flat:
    #            array = partition.dataarray(**pda_args)
    #            print(array, new.Units, type(value0), value1)
    #            partition.subarray = (array >= value0) & (array <= value1)
    #            partition.close()
    #        # --- End: for
    #
    #        new.dtype = bool
    #
    #        return new

    def __query_wo__(self, value):
        """Implements the “without a range” condition."""
        return (self < value[0]) | (self > value[1])

    #        new = self.copy()
    #
    #        pda_args = new.pda_args(revert_to_file=True)
    #
    #        for partition in new.partitions.matrix.flat:
    #            array = partition.dataarray(**pda_args)
    #            partition.subarray = (array < value0) | (array > value1)
    #            partition.close()
    #        # --- End: for
    #
    #        new.dtype = bool
    #
    #        return new


    def _all_axis_names(self):
        """Return a set of all the dimension names in use by the data
        array.

        Note that the output set includes dimensions of individual
        partitions which are not dimensions of the master data array.

        :Returns:

            `list` of `str`
                The axis names.

        **Examples:**

        >>> d._axes
        ['dim1', 'dim0']
        >>> d.partitions.info('_dimensions')
        [['dim0', 'dim0'],
         ['dim1', 'dim0', 'dim2']]
        >>> d._all_axis_names()
        ['dim2', dim0', 'dim1']

        """
        all_axes = self._all_axes
        if not all_axes:
            return list(self._axes)
        else:
            return list(all_axes)

    def _change_axis_names(self, axis_map):
        """Change the axis names.

        The axis names are arbitrary, so mapping them to another
        arbitrary collection does not change the data array values,
        units, nor axis order.

        """
        # Find any axis names which are not mapped. If there are any,
        # then update axis_map.
        all_axes = self._all_axes
        if all_axes:
            d = set(all_axes).difference(axis_map)
            if d:
                axis_map = axis_map.copy()
                existing_axes = list(all_axes)
                for axis in d:
                    if axis in axis_map.values():
                        axis_map[axis] = self._new_axis_identifier(
                            existing_axes
                        )
                        existing_axes.append(axis)
                    else:
                        axis_map[axis] = axis
        # --- End: if

        if all([axis0 == axis1 for axis0, axis1 in axis_map.items()]):
            # Return without doing anything if the mapping is null
            return

        # Axes
        self._axes = [axis_map[axis] for axis in self._axes]

        # All axes
        if all_axes:
            self._all_axes = tuple([axis_map[axis] for axis in all_axes])

        # Flipped axes
        #        flip = self._flip
        flip = self._flip()
        if flip:
            self._flip([axis_map[axis] for axis in flip])
        #            self._flip = [axis_map[axis] for axis in flip]

        # HDF chunks
        chunks = self._HDF_chunks
        if chunks:
            self._HDF_chunks = dict(
                [(axis_map[axis], size) for axis, size in chunks.items()]
            )

        # Partitions in the partition matrix
        self.partitions.change_axis_names(axis_map)

    def _new_axis_identifier(self, existing_axes=None):
        """Return an axis name not being used by the data array.

        The returned axis name will also not be referenced by partitions
        of the partition matrix.

        :Parameters:

            existing_axes: sequence of `str`, optional

        :Returns:

            `str`
                The new axis name.

        **Examples:**

        >>> d._all_axis_names()
        ['dim1', 'dim0']
        >>> d._new_axis_identifier()
        'dim2'

        >>> d._all_axis_names()
        ['dim1', 'dim0', 'dim3']
        >>> d._new_axis_identifier()
        'dim4'

        >>> d._all_axis_names()
        ['dim5', 'dim6', 'dim7']
        >>> d._new_axis_identifier()
        'dim3'

        """
        if existing_axes is None:
            existing_axes = self._all_axis_names()

        n = len(existing_axes)
        axis = "dim%d" % n
        while axis in existing_axes:
            n += 1
            axis = "dim%d" % n

        return axis

    # ----------------------------------------------------------------
    # Private attributes
    # ----------------------------------------------------------------
    @property
    def _Units(self):
        """Storage for the units."""
        try:
            return self._custom["_Units"]
        except KeyError:
            raise AttributeError()

    @_Units.setter
    def _Units(self, value):
        self._custom["_Units"] = value

    @_Units.deleter
    def _Units(self):
        self._custom["_Units"] = _units_None

    @property
    def _auxiliary_mask(self):
        """Storage for the auxiliary mask."""
        return self._custom["_auxiliary_mask"]

    @_auxiliary_mask.setter
    def _auxiliary_mask(self, value):
        self._custom["_auxiliary_mask"] = value

    @_auxiliary_mask.deleter
    def _auxiliary_mask(self):
        del self._custom["_auxiliary_mask"]

    @property
    def _cyclic(self):
        """Storage for axis cyclicity."""
        return self._custom["_cyclic"]

    @_cyclic.setter
    def _cyclic(self, value):
        self._custom["_cyclic"] = value

    @_cyclic.deleter
    def _cyclic(self):
        del self._custom["_cyclic"]

    @property
    def _dtype(self):
        """Storage for the data type."""
        return self._custom["_dtype"]

    @_dtype.setter
    def _dtype(self, value):
        self._custom["_dtype"] = value

    @_dtype.deleter
    def _dtype(self):
        del self._custom["_dtype"]

    @property
    def _HDF_chunks(self):
        """The HDF chunksizes.

        DO NOT CHANGE IN PLACE.

        """
        return self._custom["_HDF_chunks"]

    @_HDF_chunks.setter
    def _HDF_chunks(self, value):
        self._custom["_HDF_chunks"] = value

    @_HDF_chunks.deleter
    def _HDF_chunks(self):
        del self._custom["_HDF_chunks"]

    @property
    def partitions(self):
        """Storage for the partitions matrix."""
        return self._custom["partitions"]

    @partitions.setter
    def partitions(self, value):
        self._custom["partitions"] = value

    @partitions.deleter
    def partitions(self):
        del self._custom["partitions"]

    @property
    def _ndim(self):
        """Storage for the number of dimensions."""
        return self._custom["_ndim"]

    @_ndim.setter
    def _ndim(self, value):
        self._custom["_ndim"] = value

    @_ndim.deleter
    def _ndim(self):
        del self._custom["_ndim"]

    @property
    def _size(self):
        """Storage for the number of elements."""
        return self._custom["_size"]

    @_size.setter
    def _size(self, value):
        self._custom["_size"] = value

    @_size.deleter
    def _size(self):
        del self._custom["_size"]

    @property
    def _shape(self):
        """Storage for the data shape."""
        return self._custom["_shape"]

    @_shape.setter
    def _shape(self, value):
        self._custom["_shape"] = value

    @_shape.deleter
    def _shape(self):
        del self._custom["_shape"]

    @property
    def _axes(self):
        """Storage for the axes names."""
        return self._custom["_axes"]

    @_axes.setter
    def _axes(self, value):
        self._custom["_axes"] = value

    @_axes.deleter
    def _axes(self):
        del self._custom["_axes"]

    @property
    def _all_axes(self):
        """Storage for the full collection of axes names.

        :Returns:

            `None` or `tuple`.

        """
        return self._custom["_all_axes"]

    @_all_axes.setter
    def _all_axes(self, value):
        self._custom["_all_axes"] = value

    @_all_axes.deleter
    def _all_axes(self):
        del self._custom["_all_axes"]

    def _flip(self, *flip):
        """"""
        if flip:
            self._custom["flip"] = flip[0]
        else:
            return self._custom["flip"]

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def Units(self):
        """The `cf.Units` object containing the units of the data array.

        Deleting this attribute is equivalent to setting it to an
        undefined units object, so this attribute is guaranteed to always
        exist.

        **Examples:**

        >>> d.Units = Units('m')
        >>> d.Units
        <Units: m>
        >>> del d.Units
        >>> d.Units
        <Units: >

        """
        return self._Units

    @Units.setter
    def Units(self, value):
        units = getattr(self, "_Units", _units_None)
        if units and not self._Units.equivalent(value, verbose=1):
            raise ValueError(
                "Can't set units (currently {!r}) to non-equivalent "
                "units {!r}. Consider the override_units method.".format(
                    units, value
                )
            )

        dtype = self.dtype

        if dtype is not None:
            if dtype.kind == "i":
                char = dtype.char
                if char == "i":
                    old_units = getattr(self, "_Units", None)
                    if old_units is not None and not old_units.equals(value):
                        self.dtype = "float32"
                elif char == "l":
                    old_units = getattr(self, "_Units", None)
                    if old_units is not None and not old_units.equals(value):
                        self.dtype = float
        # --- End: if

        self._Units = value

    @Units.deleter
    def Units(self):
        del self._Units  # = _units_None

    @property
    def data(self):
        """The data as an object identity.

        **Examples:**

        >>> d.data is d
        True

        """
        return self

    @property
    def fill_value(self):
        """The data array missing data value.

        If set to `None` then the default `numpy` fill value appropriate to
        the data array's data-type will be used.

        Deleting this attribute is equivalent to setting it to None, so
        this attribute is guaranteed to always exist.

        **Examples:**

        >>> d.fill_value = 9999.0
        >>> d.fill_value
        9999.0
        >>> del d.fill_value
        >>> d.fill_value
        None

        """
        return self.get_fill_value(None)

    @fill_value.setter
    def fill_value(self, value):
        self.set_fill_value(value)

    @fill_value.deleter
    def fill_value(self):
        self.del_fill_value(None)


    @property
    def datetime_array(self):
        """An independent numpy array of date-time objects.

        Only applicable to data arrays with reference time units.

        If the calendar has not been set then the CF default calendar will
        be used and the units will be updated accordingly.

        The data-type of the data array is unchanged.

        .. seealso:: `array`, `varray`

        **Examples:**

        """
        if not self.Units.isreftime:
            raise ValueError(
                "Can't create date-time array from units "
                "{!r}".format(self.Units)
            )

        if getattr(self.Units, "calendar", None) == "none":
            raise ValueError(
                "Can't create date-time array from units {!r} because "
                "calendar is 'none'".format(self.Units)
            )

        units, reftime = self.Units.units.split(" since ")

        d = self

        # Convert months and years to days, because cftime won't work
        # otherwise.
        if units in ("months", "month"):
            d = self * _month_length
            d.override_units(
                Units(
                    "days since " + reftime,
                    calendar=getattr(self.Units, "calendar", None),
                ),
                inplace=True,
            )
        elif units in ("years", "year", "yr"):
            d = self * _year_length
            d.override_units(
                Units(
                    "days since " + reftime,
                    calendar=getattr(self.Units, "calendar", None),
                ),
                inplace=True,
            )

        d._dtarray = True
        return d.array

    @property
    def varray(self):
        """A numpy array view the data array.

        Note that making changes to elements of the returned view changes
        the underlying data.

        .. seealso:: `array`, `datetime_array`

        **Examples:**

        >>> a = d.varray
        >>> type(a)
        <type 'np.ndarray'>
        >>> a
        array([0, 1, 2, 3, 4])
        >>> a[0] = 999
        >>> d.varray
        array([999, 1, 2, 3, 4])

        """
        config = self.partition_configuration(readonly=False)

        data_type = self.dtype

        if getattr(self, "_dtarray", False):
            del self._dtarray
        elif self._isdatetime():  # self._isdt:
            data_type = np.dtype(float)
            config["func"] = dt2rt
            # Turn off data-type checking and partition updating
            config["dtype"] = None

        if self.partitions.size == 1:
            # If there is only one partition, then we can return a
            # view of the partition's data array without having to
            # create an empty array and then filling it up partition
            # by partition.
            partition = self.partitions.matrix.item()
            partition.open(config)
            array = partition.array
            # Note that there is no need to close the partition here.
            self._dtype = data_type

            #            source = self.source(None)
            #            if source is not None and source.get_compression_type():
            #                self._del_Array(None)

            # Flip to []?
            return array

        # Still here?
        shape = self._shape
        array_out = np.empty(shape, dtype=data_type)
        masked = False

        config["readonly"] = True

        for partition in self.partitions.matrix.flat:
            partition.open(config)
            p_array = partition.array

            if not masked and partition.masked:
                array_out = array_out.view(np.ma.MaskedArray)
                array_out.set_fill_value(self.get_fill_value(None))
                masked = True

            array_out[partition.indices] = p_array

            # Note that there is no need to close the partition here
        # --- End: for

        # ------------------------------------------------------------
        # Apply an auxiliary mask
        # ------------------------------------------------------------
        if self._auxiliary_mask:
            if not masked:
                # Convert the output array to a masked array
                array_out = array_out.view(np.ma.MaskedArray)
                array_out.set_fill_value(self.get_fill_value(None))
                masked = True

            self._auxiliary_mask_tidy()

            for mask in self._auxiliary_mask:
                array_out.mask = array_out.mask | mask.array

            if array_out.mask is np.ma.nomask:
                # There are no masked points, so convert back to a
                # non-masked array.
                array_out = array_out.data
                masked = False

            self._auxiliary_mask = None
        # --- End: if

        if masked and self.hardmask:
            # Harden the mask of the output array
            array_out.harden_mask()

        #        matrix = _xxx.copy()

        if not array_out.ndim and not isinstance(array_out, np.ndarray):
            array_out = np.asanyarray(array_out)

        self._set_partition_matrix(
            array_out, chunk=False, check_free_memory=False
        )

        #        matrix[()] = Partition(subarray = array_out,
        #                               location = [(0, n) for n in shape],
        #                               axes     = self._axes,
        #                               flip     = [],
        #                               shape    = list(shape),
        #                               Units    = self.Units,
        #                               part     = []
        #                               )
        #
        #        self.partitions = PartitionMatrix(matrix, [])

        self._dtype = data_type

        #        self._flip  = []
        self._flip([])

        #        source = self.source(None)
        #        if source is not None and source.get_compression_type():
        #            self._del_Array(None)

        return array_out

    @staticmethod
    def mask_fpe(*arg):
        """Masking of floating-point errors in the results of arithmetic
        operations.

        If masking is allowed then only floating-point errors which would
        otherwise be raised as `FloatingPointError` exceptions are
        masked. Whether `FloatingPointError` exceptions may be raised is
        determined by `cf.Data.seterr`.

        If called without an argument then the current behaviour is
        returned.

        Note that if the raising of `FloatingPointError` exceptions has
        suppressed then invalid values in the results of arithmetic
        operations may be subsequently converted to masked values with the
        `mask_invalid` method.

        .. seealso:: `cf.Data.seterr`, `mask_invalid`

        :Parameters:

            arg: `bool`, optional
                The new behaviour. True means that `FloatingPointError`
                exceptions are suppressed and replaced with masked
                values. False means that `FloatingPointError` exceptions
                are raised. The default is not to change the current
                behaviour.

        :Returns:

            `bool`
                The behaviour prior to the change, or the current
                behaviour if no new value was specified.

        **Examples:**

        >>> d = cf.Data([0., 1])
        >>> e = cf.Data([1., 2])

        >>> old = cf.Data.mask_fpe(False)
        >>> old = cf.Data.seterr('raise')
        >>> e/d
        FloatingPointError: divide by zero encountered in divide
        >>> e**123456
        FloatingPointError: overflow encountered in power

        >>> old = cf.Data.mask_fpe(True)
        >>> old = cf.Data.seterr('raise')
        >>> e/d
        <CF Data: [--, 2.0] >
        >>> e**123456
        <CF Data: [1.0, --] >

        >>> old = cf.Data.mask_fpe(True)
        >>> old = cf.Data.seterr('ignore')
        >>> e/d
        <CF Data: [inf, 2.0] >
        >>> e**123456
        <CF Data: [1.0, inf] >

        """
        old = _mask_fpe[0]

        if arg:
            _mask_fpe[0] = bool(arg[0])

        return old

    @staticmethod
    def seterr(all=None, divide=None, over=None, under=None, invalid=None):
        """Set how floating-point errors in the results of arithmetic
        operations are handled.

        The options for handling floating-point errors are:

        ============  ========================================================
        Treatment     Action
        ============  ========================================================
        ``'ignore'``  Take no action. Allows invalid values to occur in the
                      result data array.

        ``'warn'``    Print a `RuntimeWarning` (via the Python `warnings`
                      module). Allows invalid values to occur in the result
                      data array.

        ``'raise'``   Raise a `FloatingPointError` exception.
        ============  ========================================================

        The different types of floating-point errors are:

        =================  =================================  =================
        Error              Description                        Default treatment
        =================  =================================  =================
        Division by zero   Infinite result obtained from      ``'warn'``
                           finite numbers.

        Overflow           Result too large to be expressed.  ``'warn'``

        Invalid operation  Result is not an expressible       ``'warn'``
                           number, typically indicates that
                           a NaN was produced.

        Underflow          Result so close to zero that some  ``'ignore'``
                           precision was lost.
        =================  =================================  =================

        Note that operations on integer scalar types (such as int16) are
        handled like floating point, and are affected by these settings.

        If called without any arguments then the current behaviour is
        returned.

        .. seealso:: `cf.Data.mask_fpe`, `mask_invalid`

        :Parameters:

            all: `str`, optional
                Set the treatment for all types of floating-point errors
                at once. The default is not to change the current
                behaviour.

            divide: `str`, optional
                Set the treatment for division by zero. The default is not
                to change the current behaviour.

            over: `str`, optional
                Set the treatment for floating-point overflow. The default
                is not to change the current behaviour.

            under: `str`, optional
                Set the treatment for floating-point underflow. The
                default is not to change the current behaviour.

            invalid: `str`, optional
                Set the treatment for invalid floating-point
                operation. The default is not to change the current
                behaviour.

        :Returns:

            `dict`
                The behaviour prior to the change, or the current
                behaviour if no new values are specified.

        **Examples:**

        Set treatment for all types of floating-point errors to
        ``'raise'`` and then reset to the previous behaviours:

        >>> cf.Data.seterr()
        {'divide': 'warn', 'invalid': 'warn', 'over': 'warn', 'under': 'ignore'}
        >>> old = cf.Data.seterr('raise')
        >>> cf.Data.seterr(**old)
        {'divide': 'raise', 'invalid': 'raise', 'over': 'raise', 'under': 'raise'}
        >>> cf.Data.seterr()
        {'divide': 'warn', 'invalid': 'warn', 'over': 'warn', 'under': 'ignore'}

        Set the treatment of division by zero to ``'ignore'`` and overflow
        to ``'warn'`` without changing the treatment of underflow and
        invalid operation:

        >>> cf.Data.seterr(divide='ignore', over='warn')
        {'divide': 'warn', 'invalid': 'warn', 'over': 'warn', 'under': 'ignore'}
        >>> cf.Data.seterr()
        {'divide': 'ignore', 'invalid': 'warn', 'over': 'ignore', 'under': 'ignore'}

        Some examples with data arrays:

        >>> d = cf.Data([0., 1])
        >>> e = cf.Data([1., 2])

        >>> old = cf.Data.seterr('ignore')
        >>> e/d
        <CF Data: [inf, 2.0] >
        >>> e**12345
        <CF Data: [1.0, inf] >

        >>> cf.Data.seterr(divide='warn')
        {'divide': 'ignore', 'invalid': 'ignore', 'over': 'ignore', 'under': 'ignore'}
        >>> e/d
        RuntimeWarning: divide by zero encountered in divide
        <CF Data: [inf, 2.0] >
        >>> e**12345
        <CF Data: [1.0, inf] >

        >>> old = cf.Data.mask_fpe(False)
        >>> cf.Data.seterr(over='raise')
        {'divide': 'warn', 'invalid': 'ignore', 'over': 'ignore', 'under': 'ignore'}
        >>> e/d
        RuntimeWarning: divide by zero encountered in divide
        <CF Data: [inf, 2.0] >
        >>> e**12345
        FloatingPointError: overflow encountered in power

        >>> cf.Data.mask_fpe(True)
        False
        >>> cf.Data.seterr(divide='ignore')
        {'divide': 'warn', 'invalid': 'ignore', 'over': 'raise', 'under': 'ignore'}
        >>> e/d
        <CF Data: [inf, 2.0] >
        >>> e**12345
        <CF Data: [1.0, --] >

        """
        old = _seterr.copy()

        if all:
            _seterr.update(
                {"divide": all, "invalid": all, "under": all, "over": all}
            )
            if all == "raise":
                _seterr_raise_to_ignore.update(
                    {
                        "divide": "ignore",
                        "invalid": "ignore",
                        "under": "ignore",
                        "over": "ignore",
                    }
                )

        else:
            if divide:
                _seterr["divide"] = divide
                if divide == "raise":
                    _seterr_raise_to_ignore["divide"] = "ignore"

            if over:
                _seterr["over"] = over
                if over == "raise":
                    _seterr_raise_to_ignore["over"] = "ignore"

            if under:
                _seterr["under"] = under
                if under == "raise":
                    _seterr_raise_to_ignore["under"] = "ignore"

            if invalid:
                _seterr["invalid"] = invalid
                if invalid == "raise":
                    _seterr_raise_to_ignore["invalid"] = "ignore"
        # --- End: if

        return old

    def get_data(self, default=ValueError(), _units=None, _fill_value=None):
        """Returns the data.

        .. versionadded:: 3.0.0

        :Returns:

                `Data`

        """
        return self

    def get_calendar(self, default=ValueError()):
        """Return the calendar.

        .. seealso:: `del_calendar`, `set_calendar`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                calendar has not been set. If set to an `Exception`
                instance then it will be raised instead.

        :Returns:

                The calendar.

        **Examples:**

        >>> d.set_calendar('julian')
        >>> d.get_calendar
        'metres'
        >>> d.del_calendar()
        >>> d.get_calendar()
        ValueError: Can't get non-existent calendar
        >>> print(d.get_calendar(None))
        None

        """
        try:
            return self.Units.calendar
        except AttributeError:
            return super().get_calendar(default=default)

    def set_calendar(self, calendar):
        """Set the calendar.

        .. seealso:: `del_calendar`, `get_calendar`

        :Parameters:

            value: `str`
                The new calendar.

        :Returns:

            `None`

        **Examples:**

        >>> d.set_calendar('none')
        >>> d.get_calendar
        'none'
        >>> d.del_calendar()
        >>> d.get_calendar()
        ValueError: Can't get non-existent calendar
        >>> print(d.get_calendar(None))
        None

        """
        self.Units = Units(self.get_units(default=None), calendar)

    @classmethod
    def asdata(cls, d, dtype=None, copy=False):
        """Convert the input to a `Data` object.

        :Parameters:

            d: data-like
                Input data in any form that can be converted to an cf.Data
                object. This includes `cf.Data` and `cf.Field` objects,
                numpy arrays and any object which may be converted to a
                numpy array.

           dtype: data-type, optional
                By default, the data-type is inferred from the input data.

           copy:

        :Returns:

            `Data`
                `Data` interpretation of *d*. No copy is performed on the
                input if it is already a `Data` object with matching dtype
                and *copy* is False.

        **Examples:**

        >>> d = cf.Data([1, 2])
        >>> cf.Data.asdata(d) is d
        True
        >>> d.asdata(d) is d
        True

        >>> cf.Data.asdata([1, 2])
        <CF Data: [1, 2]>

        >>> cf.Data.asdata(np.array([1, 2]))
        <CF Data: [1, 2]>

        """
        data = getattr(d, "__data__", None)
        if data is None:
            # d does not have a Data interface
            data = cls(d)
            if dtype is not None:
                data.dtype = dtype

            return data

        data = data()
        if copy:
            data = data.copy()
            if dtype is not None and np.dtype(dtype) != data.dtype:
                data.dtype = dtype
        else:
            if dtype is not None and np.dtype(dtype) != data.dtype:
                data = data.copy()
                data.dtype = dtype
        # --- End: if

        return data

    def cyclic(self, axes=None, iscyclic=True):
        """Returns or sets the axes of the data array which are cyclic.

        :Parameters:

            axes: (sequence of) `int`, optional

            iscyclic: `bool`

        :Returns:

            `set`

        **Examples:**

        """
        cyclic_axes = self._cyclic
        data_axes = self._axes

        old = set([data_axes.index(axis) for axis in cyclic_axes])

        if axes is None:
            return old

        parsed_axes = self._parse_axes(axes)
        axes = [data_axes[i] for i in parsed_axes]

        if iscyclic:
            self._cyclic = cyclic_axes.union(axes)
        else:
            self._cyclic = cyclic_axes.difference(axes)

        # Make sure that the auxiliary mask has the same cyclicity
        auxiliary_mask = self._custom.get("_auxiliary_mask")
        if auxiliary_mask is not None:
            self._auxiliary_mask = [mask.copy() for mask in auxiliary_mask]
            for mask in self._auxiliary_mask:
                mask.cyclic(parsed_axes, iscyclic)

        return old

    def _YMDhms(self, attr):
        """Provides datetime components of the data array elements.

        .. seealso:: `~cf.Data.year`, ~cf.Data.month`, `~cf.Data.day`,
        `~cf.Data.hour`, `~cf.Data.minute`, `~cf.Data.second`

        """

        def _func(array, units_in, dummy0, dummy1):
            """The returned array is always independent.

            :Parameters:

                array: numpy array

                units_in: `Units`

                dummy0:
                    Ignored.

                dummy1:
                    Ignored.

            :Returns:

                numpy array

            """
            if not self._isdatetime():
                array = rt2dt(array, units_in)

            return _array_getattr(array, attr)

        # --- End: def

        if not self.Units.isreftime:
            raise ValueError(
                "Can't get {}s from data with {!r}".format(attr, self.Units)
            )

        new = self.copy()

        new._Units = _units_None

        config = new.partition_configuration(
            readonly=False, func=_func, dtype=None
        )

        for partition in new.partitions.matrix.flat:
            partition.open(config)
            array = partition.array
            new_dtype = array.dtype
            partition.close()

        new._dtype = new_dtype

        return new

    @property
    def year(self):
        """The year of each data array element.

        Only applicable for reference time units.

        .. seealso:: `~cf.Data.month`, `~cf.Data.day`, `~cf.Data.hour`,
                     `~cf.Data.minute`, `~cf.Data.second`

        **Examples:**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data: [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.year
        <CF Data: [[2000, 2001]] >

        """
        return self._YMDhms("year")

    @property
    def month(self):
        """The month of each data array element.

        Only applicable for reference time units.

        .. seealso:: `~cf.Data.year`, `~cf.Data.day`, `~cf.Data.hour`,
                     `~cf.Data.minute`, `~cf.Data.second`

        **Examples:**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data: [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.month
        <CF Data: [[12, 1]] >

        """
        return self._YMDhms("month")

    @property
    def day(self):
        """The day of each data array element.

        Only applicable for reference time units.

        .. seealso:: `~cf.Data.year`, `~cf.Data.month`, `~cf.Data.hour`,
                     `~cf.Data.minute`, `~cf.Data.second`

        **Examples:**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data: [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.day
        <CF Data: [[30, 3]] >

        """
        return self._YMDhms("day")

    @property
    def hour(self):
        """The hour of each data array element.

        Only applicable for reference time units.

        .. seealso:: `~cf.Data.year`, `~cf.Data.month`, `~cf.Data.day`,
                     `~cf.Data.minute`, `~cf.Data.second`

        **Examples:**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data: [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.hour
        <CF Data: [[22, 4]] >

        """
        return self._YMDhms("hour")

    @property
    def minute(self):
        """The minute of each data array element.

        Only applicable for reference time units.

        .. seealso:: `~cf.Data.year`, `~cf.Data.month`, `~cf.Data.day`,
                     `~cf.Data.hour`, `~cf.Data.second`

        **Examples:**

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data: [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.minute
        <CF Data: [[19, 4]] >

        """
        return self._YMDhms("minute")

    @property
    def second(self):
        """The second of each data array element.

        Only applicable for reference time units.

        .. seealso:: `~cf.Data.year`, `~cf.Data.month`, `~cf.Data.day`,
                     `~cf.Data.hour`, `~cf.Data.minute`

        >>> d = cf.Data([[1.93, 5.17]], 'days since 2000-12-29')
        >>> d
        <CF Data: [[2000-12-30 22:19:12, 2001-01-03 04:04:48]] >
        >>> d.second
        <CF Data: [[12, 48]] >

        """
        return self._YMDhms("second")

    @_display_or_return
    def dump(self, display=True, prefix=None):
        """Return a string containing a full description of the
        instance.

        :Parameters:

            display: `bool`, optional
                If False then return the description as a string. By
                default the description is printed, i.e. ``d.dump()`` is
                equivalent to ``print(d.dump(display=False))``.

            prefix: `str`, optional
               Set the common prefix of component names. By default the
               instance's class name is used.

        :Returns:

            `None` or `str`
                A string containing the description.

        """
        if prefix is None:
            prefix = self.__class__.__name__

        string = ["{0}.shape = {1}".format(prefix, self._shape)]

        if self._size == 1:
            string.append(
                "{0}.first_datum = {1}".format(prefix, self.datum(0))
            )
        else:
            string.append(
                "{0}.first_datum = {1}".format(prefix, self.datum(0))
            )
            string.append(
                "{0}.last_datum  = {1}".format(prefix, self.datum(-1))
            )

        for attr in ("fill_value", "Units"):
            string.append(
                "{0}.{1} = {2!r}".format(prefix, attr, getattr(self, attr))
            )
        # --- End: for

        return "\n".join(string)

    def ndindex(self):
        """Return an iterator over the N-dimensional indices of the data
        array.

        At each iteration a tuple of indices is returned, the last
        dimension is iterated over first.

        :Returns:

            `itertools.product`
                An iterator over tuples of indices of the data array.

        **Examples:**

        >>> d.shape
        (2, 1, 3)
        >>> for i in d.ndindex():
        ...     print(i)
        ...
        (0, 0, 0)
        (0, 0, 1)
        (0, 0, 2)
        (1, 0, 0)
        (1, 0, 1)
        (1, 0, 2)

        > d.shape
        ()
        >>> for i in d.ndindex():
        ...     print(i)
        ...
        ()

        """
        return itertools.product(*[range(0, r) for r in self._shape])

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
        """Expand the data by adding a halo.

        The halo may be applied over a subset of the data dimensions and
        each dimension may have a different halo size (including
        zero). The halo region is populated with a copy of the proximate
        values from the original data.

        **Cyclic axes**

        A cyclic axis that is expanded with a halo of at least size 1 is
        no longer considered to be cyclic.

        **Tripolar domains**

        Data for global tripolar domains are a special case in that a halo
        added to the northern end of the "Y" axis must be filled with
        values that are flipped in "X" direction. Such domains need to be
        explicitly indicated with the *tripolar* parameter.

        .. versionadded:: 3.5.0

        :Parameters:

            size: `int` or `dict`
                Specify the size of the halo for each axis.

                If *size* is a non-negative `int` then this is the halo
                size that is applied to all of the axes defined by the
                *axes* parameter.

                Alternatively, halo sizes may be assigned to axes
                individually by providing a `dict` for which a key
                specifies an axis (defined by its integer position in the
                data) with a corresponding value of the halo size for that
                axis. Axes not specified by the dictionary are not
                expanded, and the *axes* parameter must not also be set.

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
                  Specify a halo size of 2 for the first and last
                  dimensions `size=2, axes=[0, -1]`` or equivalently
                  ``size={0: 2, -1: 2}``.

            axes: (sequence of) `int`
                Select the domain axes to be expanded, defined by their
                integer positions in the data. By default, or if *axes* is
                `None`, all axes are selected. No axes are expanded if
                *axes* is an empty sequence.

            tripolar: `dict`, optional
                A dictionary defining the "X" and "Y" axes of a global
                tripolar domain. This is necessary because in the global
                tripolar case the "X" and "Y" axes need special treatment,
                as described above. It must have keys ``'X'`` and ``'Y'``,
                whose values identify the corresponding domain axis
                construct by their integer positions in the data.

                The "X" and "Y" axes must be a subset of those identified
                by the *size* or *axes* parameter.

                See the *fold_index* parameter.

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

            `Data` or `None`
                The expanded data, or `None` if the operation was
                in-place.

        **Examples:**

        >>> d = cf.Data(np.arange(12).reshape(3, 4), 'm')
        >>> d[-1, -1] = cf.masked
        >>> d[1, 1] = cf.masked
        >>> print(d.array)
        [[ 0  1  2  3]
         [ 4 --  6  7]
         [ 8  9 10 --]]

        >>> e = d.halo(1)
        >>> print(e.array)
        [[ 0  0  1  2  3  3]
         [ 0  0  1  2  3  3]
         [ 4  4 --  6  7  7]
         [ 8  8  9 10 -- --]
         [ 8  8  9 10 -- --]]
        >>> d.equals(e[1:-1, 1:-1])
        True

        >>> e = d.halo(2)
        >>> print(e.array)
        [[ 0  1  0  1  2  3  2  3]
         [ 4 --  4 --  6  7  6  7]
         [ 0  1  0  1  2  3  2  3]
         [ 4 --  4 --  6  7  6  7]
         [ 8  9  8  9 10 -- 10 --]
         [ 4 --  4 --  6  7  6  7]
         [ 8  9  8  9 10 -- 10 --]]
        >>> d.equals(e[2:-2, 2:-2])
        True

        >>> e = d.halo(0)
        >>> d.equals(e)
        True

        >>> e = d.halo(1, axes=0)
        >>> print(e.array)
        [[ 0  1  2  3]
         [ 0  1  2  3]
         [ 4 --  6  7]
         [ 8  9 10 --]
         [ 8  9 10 --]]
        >>> d.equals(e[1:-1, :])
        True
        >>> f = d.halo({0: 1})
        >>> f.equals(e)
        True

        >>> e = d.halo(1, tripolar={'X': 1, 'Y': 0})
        >>> print(e.array)
        [[ 0  0  1  2  3  3]
         [ 0  0  1  2  3  3]
         [ 4  4 --  6  7  7]
         [ 8  8  9 10 -- --]
         [-- -- 10  9  8  8]]

        >>> e = d.halo(1, tripolar={'X': 1, 'Y': 0}, fold_index=0)
        >>> print(e.array)
        [[ 3  3  2  1  0  0]
         [ 0  0  1  2  3  3]
         [ 4  4 --  6  7  7]
         [ 8  8  9 10 -- --]
         [ 8  8  9 10 -- --]]

        """
        _kwargs = ["{}={!r}".format(k, v) for k, v in locals().items()]
        _ = "{}.halo(".format(self.__class__.__name__)
        logger.info("{}{})".format(_, (",\n" + " " * len(_)).join(_kwargs)))

        d = _inplace_enabled_define_and_cleanup(self)

        ndim = d.ndim
        shape0 = d.shape

        # ------------------------------------------------------------
        # Parse the size and axes parameters
        # ------------------------------------------------------------
        if isinstance(size, dict):
            if axes is not None:
                raise ValueError(
                    "Can't set the axes parameter when the "
                    "size parameter is a dictionary"
                )

            axes = self._parse_axes(tuple(size))
            size = [size[i] if i in axes else 0 for i in range(ndim)]
        else:
            if axes is None:
                axes = list(range(ndim))

            axes = d._parse_axes(axes)
            size = [size if i in axes else 0 for i in range(ndim)]

        # ------------------------------------------------------------
        # Parse the tripolar parameter
        # ------------------------------------------------------------
        if tripolar:
            if fold_index not in (0, -1):
                raise ValueError(
                    "fold_index parameter must be -1 or 0. "
                    "Got {!r}".format(fold_index)
                )

            # Find the X and Y axes of a tripolar grid
            tripolar = tripolar.copy()
            X_axis = tripolar.pop("X", None)
            Y_axis = tripolar.pop("Y", None)

            if tripolar:
                raise ValueError(
                    "Can not set key {!r} in the tripolar "
                    "dictionary.".format(tripolar.popitem()[0])
                )

            if X_axis is None:
                raise ValueError("Must provide a tripolar 'X' axis.")

            if Y_axis is None:
                raise ValueError("Must provide a tripolar 'Y' axis.")

            X = d._parse_axes(X_axis)
            Y = d._parse_axes(Y_axis)

            if len(X) != 1:
                raise ValueError(
                    "Must provide exactly one tripolar 'X' axis. "
                    "Got {!r}".format(X_axis)
                )

            if len(Y) != 1:
                raise ValueError(
                    "Must provide exactly one tripolar 'Y' axis. "
                    "Got {!r}".format(Y_axis)
                )

            X_axis = X[0]
            Y_axis = Y[0]

            if X_axis == Y_axis:
                raise ValueError(
                    "Tripolar 'X' and 'Y' axes must be different. "
                    "Got {!r}, {!r}".format(X_axis, Y_axis)
                )

            for A, axis in zip(
                (
                    "X",
                    "Y",
                ),
                (X_axis, Y_axis),
            ):
                if axis not in axes:
                    raise ValueError(
                        "If dimensions have been identified with the "
                        "axes or size parameters then they must include "
                        "the tripolar {!r} axis: {!r}".format(A, axis)
                    )
            # --- End: for

            tripolar = True
        # --- End: if

        # Remove axes with a size 0 halo
        axes = [i for i in axes if size[i]]

        if not axes:
            # Return now if all halos are of size 0
            return d

        # Check that the halos are not too large
        for i, (h, n) in enumerate(zip(size, shape0)):
            if h > n:
                raise ValueError(
                    "Halo size {!r} is too big for axis of size {!r}".format(
                        h, n
                    )
                )
        # --- End: for

        # Initialise the expanded data
        shape1 = [
            n + size[i] * 2 if i in axes else n for i, n in enumerate(shape0)
        ]
        out = type(d).empty(
            shape1,
            dtype=d.dtype,
            units=d.Units,
            fill_value=d.get_fill_value(None),
        )

        # ------------------------------------------------------------
        # Body (not edges nor corners)
        # ------------------------------------------------------------
        indices = [
            slice(h, h + n) if (h and i in axes) else slice(None)
            for i, (h, n) in enumerate(zip(size, shape0))
        ]
        out[tuple(indices)] = d

        # ------------------------------------------------------------
        # Edges (not corners)
        # ------------------------------------------------------------
        for i in axes:
            size_i = size[i]

            for edge in (0, -1):
                # Initialise indices to the expanded data
                indices1 = [slice(None)] * ndim

                if edge == -1:
                    indices1[i] = slice(-size_i, None)
                else:
                    indices1[i] = slice(0, size_i)

                # Initialise indices to the original data
                indices0 = indices1[:]

                for j in axes:
                    if j == i:
                        continue

                    size_j = size[j]
                    indices1[j] = slice(size_j, -size_j)

                out[tuple(indices1)] = d[tuple(indices0)]
        # --- End: for

        # ------------------------------------------------------------
        # Corners
        # ------------------------------------------------------------
        if len(axes) > 1:
            for indices in itertools.product(
                *[
                    (slice(0, size[i]), slice(-size[i], None))
                    if i in axes
                    else (slice(None),)
                    for i in range(ndim)
                ]
            ):
                out[indices] = d[indices]

        hardmask = d.hardmask

        # ------------------------------------------------------------
        # Special case for tripolar: The northern "Y" axis halo
        # contains the values that have been flipped in the "X"
        # direction.
        # ------------------------------------------------------------
        if tripolar and size[Y_axis]:
            indices1 = [slice(None)] * ndim

            if fold_index == -1:
                # The last index of the "Y" axis corresponds to the
                # fold in "X" axis of a tripolar grid
                indices1[Y_axis] = slice(-size[Y_axis], None)
            else:
                # The first index of the "Y" axis corresponds to the
                # fold in "X" axis of a tripolar grid
                indices1[Y_axis] = slice(0, size[Y_axis])

            indices2 = indices1[:]
            indices2[X_axis] = slice(None, None, -1)

            out.hardmask = False
            out[tuple(indices1)] = out[tuple(indices2)]

        out.hardmask = True

        # Set expanded axes to be non-cyclic
        out.cyclic(axes=axes, iscyclic=False)

        if inplace:
            d.__dict__ = out.__dict__
            d.hardmask = hardmask
        else:
            d = out

        return d

    def has_calendar(self):
        """Whether a calendar has been set.

        .. seealso:: `del_calendar`, `get_calendar`, `set_calendar`,
                     `has_units`

        :Returns:

            `bool`
                True if the calendar has been set, otherwise False.

        **Examples:**

        >>> d.set_calendar('360_day')
        >>> d.has_calendar()
        True
        >>> d.get_calendar()
        '360_day'
        >>> d.del_calendar()
        >>> d.has_calendar()
        False
        >>> d.get_calendar()
        ValueError: Can't get non-existent calendar
        >>> print(d.get_calendar(None))
        None
        >>> print(d.del_calendar(None))
        None

        """
        return hasattr(self.Units, "calendar")

    def has_units(self):
        """Whether units have been set.

        .. seealso:: `del_units`, `get_units`, `set_units`, `has_calendar`

        :Returns:

            `bool`
                True if units have been set, otherwise False.

        **Examples:**

        >>> d.set_units('metres')
        >>> d.has_units()
        True
        >>> d.get_units()
        'metres'
        >>> d.del_units()
        >>> d.has_units()
        False
        >>> d.get_units()
        ValueError: Can't get non-existent units
        >>> print(d.get_units(None))
        None
        >>> print(d.del_units(None))
        None

        """
        return hasattr(self.Units, "units")

    def flat(self, ignore_masked=True):
        """Return a flat iterator over elements of the data array.

        :Parameters:

            ignore_masked: `bool`, optional
                If False then masked and unmasked elements will be
                returned. By default only unmasked elements are returned

        :Returns:

            generator
                An iterator over elements of the data array.

        **Examples:**

        >>> print(d.array)
        [[1 -- 3]]
        >>> for x in d.flat():
        ...     print(x)
        ...
        1
        3

        >>> for x in d.flat(ignore_masked=False):
        ...     print(x)
        ...
        1
        --
        3

        """
        self.to_memory()

        mask = self.mask

        if ignore_masked:
            for index in self.ndindex():
                if not mask[index]:
                    yield self[index].array.item()
        else:
            for index in self.ndindex():
                if not mask[index]:
                    yield self[index].array.item()
                else:
                    yield cf_masked

    @_deprecated_kwarg_check("i")
    @_inplace_enabled(default=False)
    def change_calendar(self, calendar, inplace=False, i=False):
        """Change the calendar of the data array elements.

        Changing the calendar could result in a change of reference time
        data array values.

        Not to be confused with using the `override_calendar` method or
        resetting `d.Units`. `override_calendar` is different because the
        new calendar need not be equivalent to the original ones and the
        data array elements will not be changed to reflect the new
        units. Resetting `d.Units` will

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if not self.Units.isreftime:
            raise ValueError(
                "Can't change calendar of non-reference time "
                "units: {!r}".format(self.Units)
            )

        d._asdatetime(inplace=True)
        d.override_units(Units(self.Units.units, calendar), inplace=True)
        d._asreftime(inplace=True)

        return d

    @_deprecated_kwarg_check("i")
    @_inplace_enabled(default=False)
    def override_units(self, units, inplace=False, i=False):
        """Override the data array units.

        Not to be confused with setting the `Units` attribute to units
        which are equivalent to the original units. This is different
        because in this case the new units need not be equivalent to the
        original ones and the data array elements will not be changed to
        reflect the new units.

        :Parameters:

            units: `str` or `Units`
                The new units for the data array.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples:**

        >>> d = cf.Data(1012.0, 'hPa')
        >>> d.override_units('km')
        >>> d.Units
        <Units: km>
        >>> d.datum(0)
        1012.0
        >>> d.override_units(Units('watts'))
        >>> d.Units
        <Units: watts>
        >>> d.datum(0)
        1012.0

        """
        d = _inplace_enabled_define_and_cleanup(self)
        units = Units(units)

        config = self.partition_configuration(readonly=False)

        for partition in d.partitions.matrix.flat:
            p_units = partition.Units
            if not p_units or p_units == units:
                # No need to create the data array if the sub-array
                # units are the same as the master data array units or
                # the partition units are not set
                partition.Units = units
                continue

            partition.open(config)
            partition.array
            partition.Units = units
            partition.close()

        d._Units = units

        return d

    @_deprecated_kwarg_check("i")
    @_inplace_enabled(default=False)
    def override_calendar(self, calendar, inplace=False, i=False):
        """Override the calendar of the data array elements.

        Not to be confused with using the `change_calendar` method or
        setting the `d.Units.calendar`. `override_calendar` is different
        because the new calendar need not be equivalent to the original
        ones and the data array elements will not be changed to reflect
        the new units.

        :Parameters:

            calendar: `str`
                The new calendar.

            {{inplace: `bool`, optional}}

            {{i: deprecated at version 3.0.0}}

        :Returns:

            `Data` or `None`

        **Examples:**

        """
        d = _inplace_enabled_define_and_cleanup(self)

        if not self.Units.isreftime:
            raise ValueError(
                "Can't override the calendar of non-reference-time "
                "units: {0!r}".format(self.Units)
            )

        for partition in d.partitions.matrix.flat:
            partition.Units = Units(partition.Units._units, calendar)
            partition.close()

        d._Units = Units(d.Units._units, calendar)

        return d

    def datum(self, *index):
        """Return an element of the data array as a standard Python
        scalar.

        The first and last elements are always returned with
        ``d.datum(0)`` and ``d.datum(-1)`` respectively, even if the data
        array is a scalar array or has two or more dimensions.

        The returned object is of the same type as is stored internally.

        .. seealso:: `array`, `datetime_array`

        :Parameters:

            index: *optional*
                Specify which element to return. When no positional
                arguments are provided, the method only works for data
                arrays with one element (but any number of dimensions),
                and the single element is returned. If positional
                arguments are given then they must be one of the
                fdlowing:

                * An integer. This argument is interpreted as a flat index
                  into the array, specifying which element to copy and
                  return.

                  *Parameter example:*
                    If the data array shape is ``(2, 3, 6)`` then:
                    * ``d.datum(0)`` is equivalent to ``d.datum(0, 0, 0)``.
                    * ``d.datum(-1)`` is equivalent to ``d.datum(1, 2, 5)``.
                    * ``d.datum(16)`` is equivalent to ``d.datum(0, 2, 4)``.

                  If *index* is ``0`` or ``-1`` then the first or last data
                  array element respectively will be returned, even if the
                  data array is a scalar array.

                * Two or more integers. These arguments are interpreted as a
                  multidimensional index to the array. There must be the
                  same number of integers as data array dimensions.

                * A tuple of integers. This argument is interpreted as a
                  multidimensional index to the array. There must be the
                  same number of integers as data array dimensions.

                  *Parameter example:*
                    ``d.datum((0, 2, 4))`` is equivalent to ``d.datum(0,
                    2, 4)``; and ``d.datum(())`` is equivalent to
                    ``d.datum()``.

        :Returns:

                A copy of the specified element of the array as a suitable
                Python scalar.

        **Examples:**

        >>> d = cf.Data(2)
        >>> d.datum()
        2
        >>> 2 == d.datum(0) == d.datum(-1) == d.datum(())
        True

        >>> d = cf.Data([[2]])
        >>> 2 == d.datum() == d.datum(0) == d.datum(-1)
        True
        >>> 2 == d.datum(0, 0) == d.datum((-1, -1)) == d.datum(-1, 0)
        True

        >>> d = cf.Data([[4, 5, 6], [1, 2, 3]], 'metre')
        >>> d[0, 1] = cf.masked
        >>> print(d)
        [[4 -- 6]
         [1  2 3]]
        >>> d.datum(0)
        4
        >>> d.datum(-1)
        3
        >>> d.datum(1)
        masked
        >>> d.datum(4)
        2
        >>> d.datum(-2)
        2
        >>> d.datum(0, 0)
        4
        >>> d.datum(-2, -1)
        6
        >>> d.datum(1, 2)
        3
        >>> d.datum((0, 2))
        6

        """
        if index:
            n_index = len(index)
            if n_index == 1:
                index = index[0]
                if index == 0:
                    # This also works for scalar arrays
                    index = (slice(0, 1),) * self._ndim
                elif index == -1:
                    # This also works for scalar arrays
                    index = (slice(-1, None),) * self._ndim
                elif isinstance(index, int):
                    if index < 0:
                        index += self._size

                    index = np.unravel_index(index, self._shape)
                elif len(index) == self._ndim:
                    index = tuple(index)
                else:
                    raise ValueError(
                        "Incorrect number of indices for {} array".format(
                            self.__class__.__name__
                        )
                    )
            elif n_index != self._ndim:
                raise ValueError(
                    "Incorrect number of indices for {} array".format(
                        self.__class__.__name__
                    )
                )

            array = self[index].array

        elif self._size == 1:
            array = self.array

        else:
            raise ValueError(
                "Can only convert a {} array of size 1 to a "
                "Python scalar".format(self.__class__.__name__)
            )

        if not np.ma.isMA(array):
            return array.item()

        mask = array.mask
        if mask is np.ma.nomask or not mask.item():
            return array.item()

        return cf_masked

    def del_calendar(self, default=ValueError()):
        """Delete the calendar.

        .. seealso:: `get_calendar`, `has_calendar`, `set_calendar`,
                     `del_units`

        :Parameters:

            default: optional
                Return the value of the *default* parameter if the
                calendar has not been set.

                {{default Exception}}

        :Returns:

            `str`
                The value of the deleted calendar.

        **Examples:**

        >>> d.set_calendar('360_day')
        >>> d.has_calendar()
        True
        >>> d.get_calendar()
        '360_day'
        >>> d.del_calendar()
        >>> d.has_calendar()
        False
        >>> d.get_calendar()
        ValueError: Can't get non-existent calendar
        >>> print(d.get_calendar(None))
        None
        >>> print(d.del_calendar(None))
        None

        """
        calendar = getattr(self.Units, "calendar", None)

        if calendar is not None:
            self.override_calendar(None, inplace=True)
            return calendar

        raise self._default(
            default, f"{self.__class__.__name__} has no 'calendar' component"
        )

    @classmethod
    def masked_all(cls, shape, dtype=None, units=None, chunk=True):
        """Return a new data array of given shape and type with all
        elements masked.

        .. seealso:: `empty`, `ones`, `zeros`

        :Parameters:

            shape: `int` or `tuple` of `int`
                The shape of the new array.

            dtype: data-type
                The data-type of the new array. By default the data-type
                is ``float``.

            units: `str` or `Units`
                The units for the new data array.

        :Returns:

            `Data`
                The new data array having all elements masked.

        **Examples:**

        >>> d = cf.Data.masked_all((96, 73))

        """
        array = FilledArray(
            shape=tuple(shape),
            size=functools.reduce(operator_mul, shape, 1),
            ndim=len(shape),
            dtype=np.dtype(dtype),
            fill_value=cf_masked,
        )

        return cls(array, units=units, chunk=chunk)

    def HDF_chunks(self, *chunks):
        """"""
        _HDF_chunks = self._HDF_chunks

        if _HDF_chunks is None:
            _HDF_chunks = {}
        else:
            _HDF_chunks = _HDF_chunks.copy()

        org_HDF_chunks = dict(
            [(i, _HDF_chunks.get(axis)) for i, axis in enumerate(self._axes)]
        )

        if not chunks:
            return org_HDF_chunks

        chunks = chunks[0]

        if chunks is None:
            # Clear all chunking
            self._HDF_chunks = None
            return org_HDF_chunks

        axes = self._axes
        for axis, size in chunks.items():
            _HDF_chunks[axes[axis]] = size

        if _HDF_chunks.values() == [None] * self._ndim:
            _HDF_chunks = None

        self._HDF_chunks = _HDF_chunks

        return org_HDF_chunks

    def inspect(self):
        """Inspect the object for debugging.

        .. seealso:: `cf.inspect`

        :Returns:

            `None`

        """
        print(cf_inspect(self))  # pragma: no cover

    def tolist(self):
        """Return the array as a (possibly nested) list.

        Return a copy of the array data as a (nested) Python list. Data
        items are converted to the nearest compatible Python type.

        :Returns:

            `list`
                The possibly nested list of array elements.

        **Examples:**

        >>> d = cf.Data([1, 2])
        >>> d.tolist()
        [1, 2]

        >>> d = cf.Data(([[1, 2], [3, 4]]))
        >>> list(d)
        [array([1, 2]), array([3, 4])]      # DCH CHECK
        >>> d.tolist()
        [[1, 2], [3, 4]]

        >>> d.equals(cf.Data(d.tolist()))
        True

        """
        return self.array.tolist()


    @classmethod
    def empty(
        cls,
        shape,
        dtype=None,
        units=None,
        calendar=None,
        fill_value=None,
        chunk=True,
    ):
        """Create a new data array without initializing the elements.

        Note that the mask of the returned empty data is hard.

        .. seealso:: `full`, `hardmask`, `ones`, `zeros`

        :Parameters:

            shape: `int` or `tuple` of `int`
                The shape of the new array.

            dtype: `np.dtype` or any object convertible to `np.dtype`
                The data-type of the new array. By default the data-type
                is ``float``.

            units: `str` or `Units`
                The units for the new data array.

            calendar: `str`, optional
                The calendar for reference time units.

            fill_value: optional
                The fill value of the data. By default, or if set to
                `None`, the `numpy` fill value appropriate to the array's
                data-type will be used (see
                `np.ma.default_fill_value`). Ignored if the *source*
                parameter is set.

                The fill value may also be set after initialisation with
                the `set_fill_value` method.

                *Parameter example:*
                  ``fill_value=-999.``

        :Returns:

            `Data`

        **Examples:**

        >>> d = cf.Data.empty((96, 73))

        """
        out = cls.full(
            shape,
            fill_value=None,
            dtype=dtype,
            units=units,
            calendar=calendar,
            chunk=chunk,
        )
        out.fill_value = fill_value
        return out

    @classmethod
    def full(
        cls,
        shape,
        fill_value,
        dtype=None,
        units=None,
        calendar=None,
        chunk=True,
    ):
        """Returns a new data array of given shape and type, filled with
        the given value.

        .. seealso:: `empty`, `ones`, `zeros`

        :Parameters:

            shape: `int` or `tuple` of `int`
                The shape of the new array.

            fill_value: scalar
                The fill value.

            dtype: data-type
                The data-type of the new array. By default the data-type
                is ``float``.

            units: `str` or `Units`
                The units for the new data array.

            calendar: `str`, optional
                The calendar for reference time units.

        :Returns:

            `Data`

        **Examples:**

        >>> d = cf.Data.full((96, 73), -99)

        """
        array = FilledArray(
            shape=tuple(shape),
            size=functools.reduce(operator_mul, shape, 1),
            ndim=len(shape),
            dtype=np.dtype(dtype),
            fill_value=fill_value,
        )

        return cls(array, units=units, calendar=calendar, chunk=chunk)

    @classmethod
    def ones(cls, shape, dtype=None, units=None, calendar=None, chunk=True):
        """Returns a new array filled with ones of set shape and
        type."""
        return cls.full(
            shape, 1, dtype=dtype, units=units, calendar=calendar, chunk=chunk
        )

    @classmethod
    def zeros(cls, shape, dtype=None, units=None, calendar=None, chunk=True):
        """Returns a new array filled with zeros of set shape and
        type."""
        return cls.full(
            shape, 0, dtype=dtype, units=units, calendar=calendar, chunk=chunk
        )



    # ----------------------------------------------------------------
    # Alias
    # ----------------------------------------------------------------
    @property
    def dtarray(self):
        """Alias for `datetime_array`"""
        return self.datetime_array
