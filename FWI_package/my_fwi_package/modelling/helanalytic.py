import numpy as np
from scipy.special import hankel1

def analytic_2d_helmholtz(nx, nz, dx, dz,
                          omega, velocity,
                          xs, zs):
    """
    Analytic 2D Helmholtz Green's function.

    Parameters
    ----------
    nx, nz : grid size
    dx, dz : spacing
    omega  : angular frequency
    velocity : constant velocity
    xs, zs : source indices

    Returns
    -------
    u : complex ndarray (nz, nx)
    """

    k = omega / velocity

    x = np.arange(nx) * dx
    z = np.arange(nz) * dz

    X, Z = np.meshgrid(x, z)

    x0 = xs * dx
    z0 = zs * dz

    r = np.sqrt((X - x0)**2 + (Z - z0)**2)

    # avoid singularity at source
    r[r == 0] = 1e-10

    u = -1j/4 * hankel1(0, k * r)

    return u