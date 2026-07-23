import numpy as np
import scipy.sparse as sp

def helmholtzabc(m: np.ndarray,
               omega: float,
               dx: float,
               dz: float,
               npml: int,
               vmax: float) -> sp.csr_matrix:
    """
    2D Helmholtz operator with Clayton–Engquist absorbing boundary condition.

    Parameters
    ----------
    m : (nz, nx) squared slowness (1/v^2)
    omega : angular frequency
    dx, dz : grid spacing
    npml : unused (kept for compatibility)
    vmax : unused (kept for compatibility)

    Returns
    -------
    A : sparse CSR matrix
    """

    nz, nx = m.shape
    n = nx * nz

    def idx(ix, iz):
        return iz * nx + ix

    rows, cols, data = [], [], []

    k_local = omega * np.sqrt(m)

    for iz in range(nz):
        for ix in range(nx):

            row_id = idx(ix, iz)

            # --------------------------------------------------
            # Interior nodes (5-point stencil)
            # --------------------------------------------------
            if 0 < ix < nx-1 and 0 < iz < nz-1:

                main = -2/dx**2 - 2/dz**2 + omega**2 * m[iz, ix]

                rows.append(row_id); cols.append(row_id); data.append(main)

                # Left
                rows.append(row_id); cols.append(idx(ix-1, iz)); data.append(1/dx**2)

                # Right
                rows.append(row_id); cols.append(idx(ix+1, iz)); data.append(1/dx**2)

                # Up
                rows.append(row_id); cols.append(idx(ix, iz-1)); data.append(1/dz**2)

                # Down
                rows.append(row_id); cols.append(idx(ix, iz+1)); data.append(1/dz**2)

            # --------------------------------------------------
            # Clayton–Engquist ABC
            # ∂u/∂n = i k u
            # --------------------------------------------------
            else:

                rows.append(row_id)
                cols.append(row_id)

                # Left boundary
                if ix == 0:
                    data.append(1/dx - 1j * k_local[iz, ix])
                    rows.append(row_id)
                    cols.append(idx(ix+1, iz))
                    data.append(-1/dx)

                # Right boundary
                elif ix == nx-1:
                    data.append(1/dx - 1j * k_local[iz, ix])
                    rows.append(row_id)
                    cols.append(idx(ix-1, iz))
                    data.append(-1/dx)

                # Top boundary
                elif iz == 0:
                    data.append(1/dz - 1j * k_local[iz, ix])
                    rows.append(row_id)
                    cols.append(idx(ix, iz+1))
                    data.append(-1/dz)

                # Bottom boundary
                elif iz == nz-1:
                    data.append(1/dz - 1j * k_local[iz, ix])
                    rows.append(row_id)
                    cols.append(idx(ix, iz-1))
                    data.append(-1/dz)

    A = sp.coo_matrix((data, (rows, cols)), shape=(n, n))
    return A.tocsr()