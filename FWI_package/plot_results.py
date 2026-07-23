from my_fwi_package.utils import geometry, load_results
from my_fwi_package.utils import load_wavefield_results
from my_fwi_package.plotting import plot_velocity, plot_misfit
from my_fwi_package.plotting import plot_wavefield

# m_fn, misfits_fn = load_results("results/fn_run1.npz")

# plot_velocity(m_fn)
# plot_misfit(misfits_fn)

u_abc, u_5, u_9, u_analytic = load_wavefield_results("results/wavefield_results.npz")
plot_wavefield(u_abc, u_5, u_9, u_analytic, f_plot=3,s_id=4)

