"""Implements writing file with our metadata annotations.

XRReadWrite is based on cfdm's ``cfdm.read_write.netcdf.NetCDFWrite``.
XRWrite is based on cf-python's ``cf.read_write.netcdf.NetCDFWrite``.
"""
import os

import netCDF4
import cfdm

import logging
logger = logging.getLogger(__name__)

class XRWriteBase(cfdm.read_write.netcdf.NetCDFWrite):
    """Overwrite the subset of methods in cfdm NetCDFWrite which call methods in
    netCDF4, replacing them with xarray methods.
    """
    def _create_netcdf_group(self, nc, group_name):
        """Creates a new netCDF4 group object.

        .. versionadded:: (cfdm) 1.8.6

        :Parameters:

            nc: `netCDF4._netCDF4.Group` or `netCDF4.Dataset`

            group_name: `str`
                The name of the group.

        :Returns:

            `netCDF4._netCDF4.Group`
                The new group object.

        """
        return nc.createGroup(group_name)

    def file_open(self, filename, mode, fmt, fields):
        """Open the netCDF file for writing.

        :Parameters:

            filename: `str`
                As for the *filename* parameter for initialising a
                `netCDF.Dataset` instance.

            mode: `str`
                As for the *mode* parameter for initialising a
                `netCDF.Dataset` instance.

            fmt: `str`
                As for the *format* parameter for initialising a
                `netCDF.Dataset` instance.

            fields: sequence of `Field`
                The field constructs to be written.

        :Returns:

            `netCDF.Dataset`
                A `netCDF4.Dataset` object for the file.

        """
        if fields:
            filename = os.path.abspath(filename)
            for f in fields:
                if filename in self.implementation.get_filenames(f):
                    raise ValueError(
                        "Can't write to a file that contains data "
                        "that needs to be read: {}".format(filename)
                    )
        # --- End: if

        if self.write_vars["overwrite"]:
            os.remove(filename)

        try:
            nc = netCDF4.Dataset(filename, mode, format=fmt)
        except RuntimeError as error:
            raise RuntimeError("{}: {}".format(error, filename))

        return nc


class XRWrite(XRWriteBase, cf.read_write.netcdf.NetCDFWrite):
    pass
