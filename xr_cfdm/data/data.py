import cfdm
import cf

class XRDataBase(cfdm.mixin.Container, cfdm.mixin.netcdf.NetCDFHDF5, cfdm.core.Data):
    # NetCDFHDF5 only for keeping track of chunksizes
    pass

class XRData(cf.mixin_container.Container, XRDataBase):
    pass