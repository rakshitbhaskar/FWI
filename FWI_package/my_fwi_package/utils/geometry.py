import numpy as np


def create_geometry(
    nx=80,
    nz=80,
    dx=10.0,
    dz=10.0,
    npml=10,
    nsrc=7,
    src_depth=2,
    rec_depth=0,
    rec_spacing=2,
    rec_start=5,
    rec_end=None,
):
    """
    Create computational grid, source and receiver geometry.

    Returns
    -------
    geometry : dict
        Dictionary containing grid information,
        source positions and receiver positions.
    """

    if rec_end is None:
        rec_end = nx - 5

    n = nx * nz

    idx = lambda i, j: i * nx + j

    # ---------------- Sources ----------------
    src_x = np.linspace(10, nx - 15, nsrc).astype(int)
    src_z = src_depth

    # ---------------- Receivers ----------------
    rec_x = np.arange(rec_start, rec_end, rec_spacing)
    rec_z = rec_depth

    rec_idx = [idx(rec_z, x) for x in rec_x]

    # ---------------- Physical coordinates ----------------
    x_phys = np.arange(nx) * dx
    z_phys = np.arange(nz) * dz

    geometry = {
        "nx": nx,
        "nz": nz,
        "dx": dx,
        "dz": dz,
        "npml": npml,
        "n": n,
        "idx": idx,
        "src_x": src_x,
        "src_z": src_z,
        "rec_x": rec_x,
        "rec_z": rec_z,
        "rec_idx": rec_idx,
        "nsrc": len(src_x),
        "nrec": len(rec_x),
        "x_phys": x_phys,
        "z_phys": z_phys,
        "src_x_phys": src_x * dx,
        "src_z_phys": src_z * dz,
        "rec_x_phys": rec_x * dx,
        "rec_z_phys": rec_z * dz,
        "model_extent": [
            x_phys[0],
            x_phys[-1],
            z_phys[-1],
            z_phys[0],
        ],
    }

    return geometry