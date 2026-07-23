import numpy as np
import scipy.sparse.linalg as spla
from ..modelling.hel9abc import helmholtzabc
from my_fwi_package.utils import create_geometry
geom = create_geometry()
from my_fwi_package.utils import create_velocity_model
from my_fwi_package.utils import projection_matrix
c_true, c_init, m_true, m_init, vmax = create_velocity_model()

frequencies = [
    3,4,
    5,6,
    7,15
]
P = projection_matrix()

def generate_observed_data(
    m_true=m_true,
    frequencies=frequencies,
    P=P,
    helmholtz_solver=helmholtzabc,
    vmax=vmax,
):
    """
    Generate observed seismic data from the true model.

    Parameters
    ----------
    m_true : ndarray
        True squared-slowness model.

    frequencies : list
        Frequencies (Hz).

    geometry : Geometry
        Acquisition geometry.

    P : scipy.sparse.csr_matrix
        Projection operator.

    helmholtz_solver : callable
        Helmholtz matrix constructor.

    vmax : float
        Maximum velocity.

    Returns
    -------
    dict
        Dictionary containing observed data for each frequency.
    """

    d_obs = {}

    for f in frequencies:

        omega = 2 * np.pi * f

        A = helmholtz_solver(
            m_true,
            omega,
            geom["dx"],
            geom["dz"],
            geom["npml"],
            vmax,
        )

        LU = spla.splu(A.tocsc())

        d_obs[f] = []

        for sx in geom["src_x"]:

            q = np.zeros(geom["n"], dtype=complex)

            q[geom["idx"](geom["src_z"], sx)] = 1.0

            u = LU.solve(q)

            d_obs[f].append(P @ u)

    return d_obs