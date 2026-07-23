from my_fwi_package.inversion import Gradient
from my_fwi_package.inversion.fn import Full_newton
from my_fwi_package.inversion.gn import Gauss_newton
from my_fwi_package.utils import create_geometry
from my_fwi_package.utils import create_velocity_model
from my_fwi_package.utils import projection_matrix
from my_fwi_package.modelling.dobs import generate_observed_data
from my_fwi_package.plotting import plot_velocity, plot_misfit
from my_fwi_package.utils import save_results
geometry = create_geometry()
c_true, c_init, m_true, m_init, vmax = create_velocity_model()
P = projection_matrix()
d_obs = generate_observed_data()
frequencies = [
    3,4,
    5,6,
    7,15
]

m_gd,misfits_gd=Gradient(c_init, frequencies, d_obs,
                         P, geometry, vmax,
                        n_iterations=2)

m_gn,misfits_gn = Gauss_newton(c_init, frequencies, d_obs,
                 P,geometry, vmax,
                 beta0=5e-3,
                 n_iterations=2,
                 cg_maxiter=5)


m_fn,misfits_fn = Full_newton(c_init, frequencies, d_obs,
                 P,geometry, vmax,
                 beta0=5e-3,
                 n_iterations=2,
                 minres_maxiter=5)


save_results(
    "results/gd_run1.npz",
    model=m_gd,
    misfit=misfits_gd,
)
save_results(
    "results/gn_run1.npz",
    model=m_gn,
    misfit=misfits_gn,
)
save_results(
    "results/fn_run1.npz",
    model=m_fn,
    misfit=misfits_fn,
)