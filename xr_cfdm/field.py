import cfdm
import cf

import logging
logger = logging.getLogger(__name__)

# don't *think* we need to mess with cfdm.Field, or any of its parents

class XRField(cf.mixin.FieldDomain, cf.mixin.PropertiesData, cfdm.Field):
    pass