import numpy as np
import scipy.sparse as sp
def helmholtz5(m: np.ndarray,
                    omega: float,
                    dx: float,
                    dz: float,
                    npml: int,
                    vmax: float) -> sp.csr_matrix:
    """
    Build 2D Helmholtz matrix with quadratic PML.

    Parameters
    ----------
    m : (nz, nx) array
        Squared slowness model (1/v^2)
    omega : float
        Angular frequency
    dx, dz : float
        Grid spacing
    npml : int
        Number of PML grid points
    vmax : float
        Maximum velocity (for sigma_max)

    Returns
    -------
    A : sparse CSR matrix (n x n)
    """

    nz, nx = m.shape
    n = nx * nz

    # ------------------------------------------------------
    # Index mapping
    # ------------------------------------------------------
    def idx(i, j):
        return i * nx + j

    # ------------------------------------------------------
    # Construct PML profiles
    # ------------------------------------------------------
    def pml_sigma(npts, d):
        sigma = np.zeros(npts)
        L = npml * d
        sigma_max = 3 * vmax * np.log(100) / (2 * L)

        for i in range(npml):
            x = (npml - i) / npml
            val = sigma_max * x**2
            sigma[i] = val
            sigma[npts - 1 - i] = val

        return sigma

    sigma_x = pml_sigma(nx, dx)
    sigma_z = pml_sigma(nz, dz)

    # ------------------------------------------------------
    # Assemble matrix
    # ------------------------------------------------------
    rows, cols, data = [], [], []

    for i in range(nz):
        for j in range(nx):

            k = idx(i, j)

            sx = 1.0 + 1j * sigma_x[j] / omega
            sz = 1.0 + 1j * sigma_z[i] / omega

            ax = 1.0 / (sx * dx**2)
            az = 1.0 / (sz * dz**2)

            main = -2.0 * ax - 2.0 * az + omega**2 * m[i, j]

            # center
            rows.append(k); cols.append(k); data.append(main)

            # left
            if j > 0:
                rows.append(k); cols.append(idx(i, j-1)); data.append(ax)

            # right
            if j < nx-1:
                rows.append(k); cols.append(idx(i, j+1)); data.append(ax)

            # up
            if i > 0:
                rows.append(k); cols.append(idx(i-1, j)); data.append(az)

            # down
            if i < nz-1:
                rows.append(k); cols.append(idx(i+1, j)); data.append(az)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    return A.tocsr()