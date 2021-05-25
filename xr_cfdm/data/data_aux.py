"""Module-level functions unchanged from cf-python.data.
"""

from numpy import dtype as numpy_dtype
from numpy import empty as numpy_empty
from numpy import shape as numpy_shape
from numpy import tile as numpy_tile
from numpy import unravel_index as numpy_unravel_index
from numpy import vectorize as numpy_vectorize
from numpy import floating as numpy_floating
from numpy import bool_ as numpy_bool_
from numpy import integer as numpy_integer

from ..units import Units

import logging
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------
_year_length = 365.242198781
_month_length = _year_length / 12


def _convert_to_builtin_type(x):
    """Convert a non-JSON-encodable object to a JSON-encodable built-in
    type.

    Possible conversions are:

    ================  =======  ================================
    Input             Output   `numpy` data-types covered
    ================  =======  ================================
    `numpy.bool_`     `bool`   bool
    `numpy.integer`   `int`    int, int8, int16, int32, int64,
                               uint8, uint16, uint32, uint64
    `numpy.floating`  `float`  float, float16, float32, float64
    ================  =======  ================================

    :Parameters:

        x:
            `numpy.bool_` or `numpy.integer` or `numpy.floating`
                The object of some numpy primitive data type.

    :Returns:

            `bool` or `int` or `float`
                 The object converted to a JSON-encodable type.

    **Examples:**

    >>> type(_convert_to_builtin_type(numpy.bool_(True)))
    bool
    >>> type(_convert_to_builtin_type(numpy.array([1.0])[0]))
    double
    >>> type(_convert_to_builtin_type(numpy.array([2])[0]))
    int

    """
    if isinstance(x, numpy_bool_):
        return bool(x)

    if isinstance(x, numpy_integer):
        return int(x)

    if isinstance(x, numpy_floating):
        return float(x)

    raise TypeError(
        "{0!r} object is not JSON serializable: {1!r}".format(type(x), x)
    )


# --------------------------------------------------------------------
# _seterr = How floating-point errors in the results of arithmetic
#           operations are handled. These defaults are those of
#           numpy 1.10.1.
# --------------------------------------------------------------------
_seterr = {
    "divide": "warn",
    "invalid": "warn",
    "over": "warn",
    "under": "ignore",
}

# --------------------------------------------------------------------
# _seterr_raise_to_ignore = As _seterr but with any values of 'raise'
#                           changed to 'ignore'.
# --------------------------------------------------------------------
_seterr_raise_to_ignore = _seterr.copy()


for key, value in _seterr.items():
    if value == "raise":
        _seterr_raise_to_ignore[key] = "ignore"
# --- End: for

# --------------------------------------------------------------------
# _mask_fpe[0] = Whether or not to automatically set
#                FloatingPointError exceptions to masked values in
#                arimthmetic.
# --------------------------------------------------------------------
_mask_fpe = [False]

_xxx = numpy_empty((), dtype=object)

_empty_set = set()

_units_None = Units()
_units_1 = Units("1")
_units_radians = Units("radians")

_dtype_object = numpy_dtype(object)
_dtype_float = numpy_dtype(float)
_dtype_bool = numpy_dtype(bool)

_cached_axes = {0: []}


def _initialise_axes(ndim):
    """Initialise dimension identifiers of N-d data.

    :Parameters:

        ndim: `int`
            The number of dimensions in the data.

    :Returns:

        `list`
             The dimension identifiers, one of each dimension in the
             array. If the data is scalar thn the list will be empty.

    **Examples:**

    >>> _initialise_axes(0)
    []
    >>> _initialise_axes(1)
    ['dim1']
    >>> _initialise_axes(3)
    ['dim1', 'dim2', 'dim3']
    >>> _initialise_axes(3) is _initialise_axes(3)
    True

    """
    axes = _cached_axes.get(ndim, None)
    if axes is None:
        axes = ["dim%d" % i for i in range(ndim)]
        _cached_axes[ndim] = axes

    return axes


def _size_of_index(index, size=None):
    """Return the number of elements resulting in applying an index to a
    sequence.

    :Parameters:

        index: `slice` or `list` of `int`
            The index being applied to the sequence.

        size: `int`, optional
            The number of elements in the sequence being indexed. Only
            required if *index* is a slice object.

    :Returns:

        `int`
            The length of the sequence resulting from applying the index.

    **Examples:**

    >>> _size_of_index(slice(None, None, -2), 10)
    5
    >>> _size_of_index([1, 4, 9])
    3

    """
    if isinstance(index, slice):
        # Index is a slice object
        start, stop, step = index.indices(size)
        div, mod = divmod(stop - start, step)
        if mod != 0:
            div += 1
        return div
    else:
        # Index is a list of integers
        return len(index)


def _overlapping_partitions(partitions, indices, axes, master_flip):
    """Return the nested list of (modified) partitions which overlap the
    given indices to the master array.

    :Parameters:

        partitions : cf.PartitionMatrix

        indices : tuple

        axes : sequence of str

        master_flip : list

    :Returns:

        numpy array
            A numpy array of cf.Partition objects.

    **Examples:**

    >>> type(f.Data)
    <class 'cf.data.Data'>
    >>> d._axes
    ['dim1', 'dim2', 'dim0']
    >>> axis_to_position = {'dim0': 2, 'dim1': 0, 'dim2' : 1}
    >>> indices = (slice(None), slice(5, 1, -2), [1,3,4,8])
    >>> x = _overlapping_partitions(d.partitions, indices, axis_to_position, master_flip)

    """

    axis_to_position = {}
    for i, axis in enumerate(axes):
        axis_to_position[axis] = i

    if partitions.size == 1:
        partition = partitions.matrix.item()

        # Find out if this partition overlaps the original slice
        p_indices, shape = partition.overlaps(indices)

        if p_indices is None:
            # This partition is not in the slice out of bounds - raise
            # error?
            return

        # Still here? Create a new partition
        partition = partition.copy()
        partition.new_part(p_indices, axis_to_position, master_flip)
        partition.shape = shape

        new_partition_matrix = numpy_empty(partitions.shape, dtype=object)
        new_partition_matrix[...] = partition

        return new_partition_matrix
    # --- End: if

    # Still here? Then there are 2 or more partitions.

    partitions_list = []
    partitions_list_append = partitions_list.append

    flat_pm_indices = []
    flat_pm_indices_append = flat_pm_indices.append

    partitions_flat = partitions.matrix.flat

    i = partitions_flat.index

    for partition in partitions_flat:
        # Find out if this partition overlaps the original slice
        p_indices, shape = partition.overlaps(indices)

        if p_indices is None:
            # This partition is not in the slice
            i = partitions_flat.index
            continue

        # Still here? Then this partition overlaps the slice, so
        # create a new partition.
        partition = partition.copy()
        partition.new_part(p_indices, axis_to_position, master_flip)
        partition.shape = shape

        partitions_list_append(partition)

        flat_pm_indices_append(i)

        i = partitions_flat.index
    # --- End: for

    new_shape = [
        len(set(s))
        for s in numpy_unravel_index(flat_pm_indices, partitions.shape)
    ]

    new_partition_matrix = numpy_empty((len(flat_pm_indices),), dtype=object)
    new_partition_matrix[...] = partitions_list
    new_partition_matrix.resize(new_shape)

    return new_partition_matrix


# --------------------------------------------------------------------
#
# --------------------------------------------------------------------
def _getattr(x, attr):
    if not x:
        return False
    return getattr(x, attr)


_array_getattr = numpy_vectorize(_getattr)


def _broadcast(a, shape):
    """Broadcast an array to a given shape.

    It is assumed that ``len(array.shape) <= len(shape)`` and that the
    array is broadcastable to the shape by the normal numpy
    boradcasting rules, but neither of these things are checked.

    For example, ``d[...] = d._broadcast(e, d.shape)`` gives the same
    result as ``d[...] = e``

    :Parameters:

        a: numpy array-like

        shape: `tuple`

    :Returns:

        `numpy.ndarray`

    """
    # Replace with numpy.broadcast_to v1.10 ??/ TODO

    a_shape = numpy_shape(a)
    if a_shape == shape:
        return a

    tile = [(m if n == 1 else 1) for n, m in zip(a_shape[::-1], shape[::-1])]
    tile = shape[0 : len(shape) - len(a_shape)] + tuple(tile[::-1])

    return numpy_tile(a, tile)


class AuxiliaryMask:
    """TODO."""

    def __init__(self):
        """TODO."""
        self._mask = []

    def __getitem__(self, indices):
        """TODO."""
        new = type(self)()

        for mask in self._mask:
            mask_indices = [
                (slice(None) if n == 1 else index)
                for n, index in zip(mask.shape, indices)
            ]
            new._mask.append(mask[tuple(mask_indices)])

        return new

    # ----------------------------------------------------------------
    # Attributes
    # ----------------------------------------------------------------
    @property
    def ndim(self):
        """TODO."""
        return self._mask[0].ndim

    @property
    def dtype(self):
        """TODO."""
        return self._mask[0].dtype

    # ----------------------------------------------------------------
    # Methods
    # ----------------------------------------------------------------
    def append(self, mask):
        """TODO."""
        self._mask.append(mask)


# --- End: class
