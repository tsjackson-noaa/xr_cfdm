import cfdm
import cf

# don't *think* we need to mess with cfdm.Field, or any of its parents

class XRField(cf.mixin.FieldDomain, cf.mixin.PropertiesData, cfdm.Field):
    pass