import numpy as np
import scipy.sparse as sp
def helmholtz9(m: np.ndarray,
               omega: float,
               dx: float,
               dz: float,
               npml: int,
               vmax: float) -> sp.csr_matrix:
    """
    2D Helmholtz with quadratic PML
    using correct 9-point 4th-order stencil.
    """

    nz, nx = m.shape
    n = nx * nz

    def idx(i, j):
        return i * nx + j

    # ---------------- PML ----------------
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

    rows, cols, data = [], [], []

    for i in range(nz):
        for j in range(nx):

            k = idx(i, j)

            sx = 1.0 + 1j * sigma_x[j] / omega
            sz = 1.0 + 1j * sigma_z[i] / omega

            ax = 1.0 / (sx * dx**2)
            az = 1.0 / (sz * dz**2)

            # 9-point coefficients
            cx = 4.0 / 6.0 * ax
            cz = 4.0 / 6.0 * az
            cdiag = 1.0 / 6.0 * (ax + az) / 2.0

            main = (
                -20.0 / 6.0 * (ax + az) / 2.0
                + omega**2 * m[i, j]
            )

            # Center
            rows.append(k); cols.append(k); data.append(main)

            # Cross neighbors
            for (ii, jj, val) in [
                (i, j-1, cx),
                (i, j+1, cx),
                (i-1, j, cz),
                (i+1, j, cz),
            ]:
                if 0 <= ii < nz and 0 <= jj < nx:
                    rows.append(k)
                    cols.append(idx(ii, jj))
                    data.append(val)

            # Diagonal neighbors
            for (ii, jj) in [
                (i-1, j-1),
                (i-1, j+1),
                (i+1, j-1),
                (i+1, j+1),
            ]:
                if 0 <= ii < nz and 0 <= jj < nx:
                    rows.append(k)
                    cols.append(idx(ii, jj))
                    data.append(cdiag)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    return A.tocsr()