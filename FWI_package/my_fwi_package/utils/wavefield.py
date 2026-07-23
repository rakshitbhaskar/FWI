import scipy.sparse.linalg as spla
import numpy as np
import matplotlib.pyplot as plt
from my_fwi_package.utils import create_geometry
from my_fwi_package.utils import point_source
geometry = create_geometry()
q=point_source(s_id=4)
def forward_wavefield(Forward_operator):
    LU=spla.splu(Forward_operator.tocsc())
    u=LU.solve(q).reshape(geometry["nz"], geometry["nx"])
    return u
