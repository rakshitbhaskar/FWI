import scipy.sparse as sp

from ..utils.geometry import create_geometry
geometry = create_geometry()
def projection_matrix(nrec=geometry["nrec"], n=geometry["n"], rec_idx=geometry["rec_idx"]):

    P = sp.lil_matrix((nrec, n))

    for i, r in enumerate(rec_idx):
        P[i, r] = 1

    return P.tocsr()