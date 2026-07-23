import numpy as np

from ..utils.geometry import create_geometry
geom = create_geometry()
def point_source(s_id):
    q = np.zeros(geom["n"], dtype=complex)
    q[geom["idx"](geom["src_z"], geom["src_x"][s_id])] =1
    return q