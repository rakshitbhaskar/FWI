import numpy as np
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter
from ..modelling.hel9abc import helmholtzabc

def Gradient(c_init, frequencies, d_obs,
             P,geometry, vmax,
             n_iterations):
    """
    Computes Full Waveform Inversion (FWI) Gradient Descent.
    """
    # ----------------------------------------
    # Parameterization: m = 1 / c^2
    # ----------------------------------------
    m = 1.0 / (c_init.copy() ** 2)

    # Bounds converted from velocity limits
    m_min = 1.0 / (2600.0 ** 2)
    m_max = 1.0 / (1500.0 ** 2)

    misfits = []

    for f in frequencies:
        print(f"\n===== Frequency: {f} Hz =====")
        omega = 2 * np.pi * f

        for it in range(n_iterations):
            g_m_total = np.zeros((geometry["nz"], geometry["nx"]))
            misfit = 0.0

            # Build operator directly with m
            A = helmholtzabc(m, omega, geometry["dx"], geometry["dz"], geometry["npml"], vmax)
            LU = spla.splu(A.tocsc())

            # -------- Forward + Adjoint --------
            for isrc, s in enumerate(geometry["src_x"]):
                q = np.zeros(geometry["n"], dtype=complex)
                q[geometry["idx"](geometry["src_z"], s)] = 1.0

                # Forward
                u = LU.solve(q)

                # Residual
                r = P @ u - d_obs[f][isrc]
                misfit += 0.5 * np.linalg.norm(r)**2

                # Adjoint
                rhs = np.zeros(geometry["n"], dtype=complex)
                rhs[geometry["rec_idx"]] = r
                lam = LU.solve(rhs, trans='H')

                # Gradient directly wrt squared slowness
                g_m = -omega**2 * np.real(u.conj() * lam).reshape(geometry["nz"], geometry["nx"])
                g_m_total += g_m

            misfits.append(misfit)
            print(f"Iter: {it+1} | Misfit: {misfit:.4e}")

            # -------- Smoothing --------
            g_m_total = gaussian_filter(g_m_total, sigma=2)

            # -------- Normalization --------
            g_m_total /= (np.max(np.abs(g_m_total)) + 1e-12)

            # -------- Backtracking line search --------
            step = 1e-6
            beta = 0.5

            while step > 1e-12:
                m_trial = m - step * g_m_total

                # Apply squared-slowness bounds
                m_trial = np.clip(m_trial, m_min, m_max)

                A_trial = helmholtzabc(m_trial, omega, geometry["dx"], geometry["dz"], geometry["npml"], vmax)
                LU_trial = spla.splu(A_trial.tocsc())

                misfit_trial = 0.0

                for isrc, s in enumerate(geometry["src_x"]):
                    q = np.zeros(geometry["n"], dtype=complex)
                    q[geometry["idx"](geometry["src_z"] , s)] = 1.0

                    u_trial = LU_trial.solve(q)
                    r_trial = P @ u_trial - d_obs[f][isrc]
                    misfit_trial += 0.5 * np.linalg.norm(r_trial)**2

                if misfit_trial < misfit:
                    break

                step *= beta

            m = m_trial

    return m, misfits
