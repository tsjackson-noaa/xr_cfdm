import cfdm
from .cf_python import mixin
from . import PropertiesData

import logging
logger = logging.getLogger(__name__)

# don't *think* we need to mess with cfdm.Field, or any of its parents

class XRField(mixin.FieldDomain, PropertiesData, cfdm.Field):
    pass