import numpy as np
from ..utils.geometry import create_geometry
geometry = create_geometry()
def create_velocity_model(
    nx=geometry["nx"],
    nz=geometry["nz"],
    background_velocity=2000,
    anomaly_velocity=2500,
    anomaly=((30, 50), (30, 50)),
):
    """
    Create true and initial velocity models.
    """

    c_true = background_velocity * np.ones((nz, nx))

    (z1, z2), (x1, x2) = anomaly

    c_true[z1:z2, x1:x2] = anomaly_velocity

    c_init = background_velocity * np.ones_like(c_true)

    m_true = 1 / c_true**2
    m_init = 1 / c_init**2

    vmax = np.max(c_true)

    return c_true, c_init, m_true, m_init, vmax