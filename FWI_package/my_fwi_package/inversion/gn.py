##### adaptive damping
import numpy as np
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter
from ..modelling.hel9abc import helmholtzabc
def Gauss_newton(c_init, frequencies, d_obs,
                 P,geometry, vmax,
                 beta0,
                 n_iterations,
                 cg_maxiter):

    m = 1 / c_init**2
    misfits = []

    for f in frequencies:

        print("\n===== Frequency:", f, "Hz =====")

        omega = 2 * np.pi * f

        # fresh damping per frequency (scaled if desired)
        beta = beta0 / (f**2)

        for iteration in range(n_iterations):

            g_total = np.zeros(geometry["n"])
            H_data = []
            misfit = 0.0

            A = helmholtzabc(m, omega, geometry["dx"], geometry["dz"], geometry["npml"], vmax)
            LU = spla.splu(A.tocsc())

            # -------- Forward + Adjoint --------
            for isrc, s in enumerate(geometry["src_x"]):

                q = np.zeros(geometry["n"], dtype=complex)
                q[geometry["idx"](geometry["src_z"] , s)] = 1.0

                u = LU.solve(q)

                r = P @ u - d_obs[f][isrc]
                misfit += 0.5 * np.linalg.norm(r)**2

                rhs = np.zeros(geometry["n"], dtype=complex)
                rhs[geometry["rec_idx"]] = r
                lam = LU.solve(rhs, trans='H')

                g_total += -omega**2 * np.real(u.conj() * lam)

                H_data.append((u, LU))

            misfits.append(misfit)

            print("Iteration:", iteration+1,
                  "Misfit:", misfit,
                  "beta:", beta)

            # -------- Hessian-vector product --------
            def H_matvec(v):

                Hv = np.zeros(geometry["n"])

                for (u, LU_local) in H_data:

                    w = LU_local.solve(-omega**2 * (v * u))
                    Jv = P @ w

                    rhs2 = np.zeros(geometry["n"], dtype=complex)
                    rhs2[geometry["rec_idx"]] = Jv
                    lam2 = LU_local.solve(rhs2, trans='H')

                    Hv += -omega**2 * np.real(u.conj() * lam2)

                Hv += beta * v
                return Hv

            H_linop = spla.LinearOperator((geometry["n"], geometry["n"]), matvec=H_matvec)

            dm, info = spla.cg(H_linop, -g_total, maxiter=cg_maxiter)

            # -------- Line search (FIXED) --------
            step = 1.0
            accepted = False

            for _ in range(8):

                m_trial = m + step * dm.reshape(geometry["nz"], geometry["nx"])

                A_trial = helmholtzabc(
                    m_trial, omega, geometry["dx"], geometry["dz"], geometry["npml"], vmax
                )
                LU_trial = spla.splu(A_trial.tocsc())

                misfit_trial = 0.0

                for isrc, s in enumerate(geometry["src_x"] ):
                    q = np.zeros(geometry["n"] , dtype=complex)
                    q[geometry["idx"](geometry["src_z"] , s)] = 1.0
                    u_trial = LU_trial.solve(q)
                    r_trial = P @ u_trial - d_obs[f][isrc]
                    misfit_trial += 0.5 * np.linalg.norm(r_trial)**2

                if misfit_trial < misfit:
                    accepted = True
                    break

                step *= 0.5

            # -------- LM damping update (FIXED LOCATION) --------
            if accepted:
                m = m_trial
                beta *= 0.5
            else:
                beta *= 5.0

            # Stabilization
            m = gaussian_filter(m, sigma=1)

    return m, misfits