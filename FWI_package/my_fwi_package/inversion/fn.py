import numpy as np
import scipy.sparse.linalg as spla
from scipy.ndimage import gaussian_filter
from ..modelling.hel9abc import helmholtzabc
def Full_newton(c_init, frequencies, d_obs,
                  P,geometry, vmax,
                beta0,
                n_iterations,
                minres_maxiter):

    m = 1/c_init**2
    misfits = []

    for f in frequencies:

        print("\n===== Frequency:", f, "Hz =====")
        omega = 2*np.pi*f
        beta = beta0

        for iteration in range(n_iterations):

            g = np.zeros(geometry["n"])
            H_data = []
            misfit = 0.0

            # ----- Build operator -----
            A = helmholtzabc(m,omega,geometry["dx"],geometry["dz"],geometry["npml"],vmax)
            LU = spla.splu(A.tocsc())

            # ----- Forward & Adjoint -----
            for isrc, s in enumerate(geometry["src_x"]):

                q = np.zeros(geometry["n"], dtype=complex)
                q[geometry["idx"](geometry["src_z"] , s)] = 1.0

                u = LU.solve(q)
                r = P @ u - d_obs[f][isrc]
                misfit += 0.5 * np.linalg.norm(r)**2

                rhs = np.zeros(geometry["n"], dtype=complex)
                rhs[geometry["rec_idx"]] = r
                lam = LU.solve(rhs, trans='H')

                g += -omega**2 * np.real(u.conj() * lam)

                # Store for full Hessian
                H_data.append((u, lam, LU))

            misfits.append(misfit)
            print("Iter:", iteration+1, "Misfit:", misfit, "beta:", beta)

            # ----- Full Newton Hessian-vector product -----
            def H_matvec(v):

                Hv = np.zeros(geometry["n"])

                for (u, lam, LU_local) in H_data:

                    # First variation of wavefield
                    w = LU_local.solve(-omega**2 * (v * u))
                    Jv = P @ w

                    # Adjoint of Jv
                    rhs2 = np.zeros(geometry["n"], dtype=complex)
                    rhs2[geometry["rec_idx"]] = Jv
                    lam2 = LU_local.solve(rhs2, trans='H')

                    term1 = -omega**2 * np.real(u.conj() * lam2)
                    term2 = -omega**2 * np.real(w.conj() * lam)

                    Hv += term1 + term2

                Hv += beta * v
                return Hv

            H_linop = spla.LinearOperator((geometry["n"], geometry["n"]), matvec=H_matvec)

            # Solve (H + βI) dm = -g   (indefinite-safe)
            dm, info = spla.minres(H_linop, -g, maxiter=minres_maxiter)

            # ----- Trial step -----
            m_trial = m + dm.reshape(geometry["nz"], geometry["nx"])

            A_trial = helmholtzabc(m_trial, omega, geometry["dx"], geometry["dz"], geometry["npml"], vmax)
            LU_trial = spla.splu(A_trial.tocsc())

            misfit_trial = 0.0
            for isrc, s in enumerate(geometry["src_x"]):

                q = np.zeros(geometry["n"], dtype=complex)
                q[geometry["idx"](geometry["src_z"] , s)] = 1.0

                u_trial = LU_trial.solve(q)
                r_trial = P @ u_trial - d_obs[f][isrc]
                misfit_trial += 0.5 * np.linalg.norm(r_trial)**2

            # ----- Adaptive Levenberg–Marquardt damping -----
            if misfit_trial < misfit:
                m = m_trial
                beta *= 0.5
            else:
                beta *= 5.0

            # Stabilization
            m = gaussian_filter(m, sigma=1)

            # Physical bounds (slowness squared)
            m = np.clip(m, 1/2600**2, 1/1500**2)

    return m, misfits
