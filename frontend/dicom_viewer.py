import pydicom
from pydicom import dcmread

fp = "frontend/static/dicoms/1.2.276.0.7230010.3.1.4.16777343.2068.1685452558.707970.dcm"

ds = dcmread(fp)

print(ds)