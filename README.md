# xr_cfdm

This package attempts to implement a complete object model of the CF (Climate and Forecast) metadata standard for data presented as an xarray DataSet.

In terms of functionality, what's done here is almost fully redundant with NCAS-CMS's [cf-python](https://github.com/NCAS-CMS/cf-python) package, which similarly builds on cfdm and offers support for mathematical operations and lazy, out-of-core evaulation. We feel it's worth reinventing this particular wheel in order to take advantage of the greater developer momentum behind the xarray ecosystem, in particular support for distributed processing via [dask](https://github.com/dask/dask) and support for non-netCDF binary formats such as [zarr](https://zarr.readthedocs.io/en/stable/).

References:

- The [CF standard](http://cfconventions.org/Data/cf-conventions/cf-conventions-1.8/cf-conventions.html);
- The [NCAS-CMS](http://cms.ncas.ac.uk/) implementation of the CF standard's data model (`cfdm`): [journal article](https://doi.org/10.5194/gmd-10-4619-2017), [code](https://github.com/NCAS-CMS/cfdm), [docs](https://ncas-cms.github.io/cfdm/index.html).
- Xarray [code](https://github.com/pydata/xarray) and [docs](http://xarray.pydata.org/en/stable/index.html).

