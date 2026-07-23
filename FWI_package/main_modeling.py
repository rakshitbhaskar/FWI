import numpy as np
import scipy.sparse.linalg as spla
from my_fwi_package.modelling import helmholtzabc
from my_fwi_package.modelling import helmholtz5
from my_fwi_package.modelling import helmholtz9
from my_fwi_package.modelling import analytic_2d_helmholtz
from my_fwi_package.utils import create_geometry
from my_fwi_package.utils import create_velocity_model
from my_fwi_package.utils.io import save_wavefield_results
from my_fwi_package.utils.operator import projection_matrix
from my_fwi_package.utils import forward_wavefield
from my_fwi_package.utils import save_wavefield_results
geometry = create_geometry()
c_true, c_init, m_true, m_init, vmax = create_velocity_model()
A=helmholtzabc(m_true,omega=2*np.pi*3, dx=geometry["dx"],dz=geometry["dz"], npml=geometry["npml"], vmax=vmax)
B=helmholtz5(m_true,omega=2*np.pi*3, dx=geometry["dx"],dz=geometry["dz"], npml=geometry["npml"], vmax=vmax)
C=helmholtz9(m_true,omega=2*np.pi*3, dx=geometry["dx"],dz=geometry["dz"], npml=geometry["npml"], vmax=vmax)
D=analytic_2d_helmholtz(nx=geometry["nx"], nz=geometry["nz"], dx=geometry["dx"], dz=geometry["dz"], omega=2*np.pi*3, velocity=vmax, xs=geometry["src_x"][len(geometry["src_x"])//2], zs=geometry["src_z"])

u_abc = forward_wavefield(A)
u_5 = forward_wavefield(B)
u_9 = forward_wavefield(C)
u_analytic = D

save_wavefield_results("results/wavefield_results.npz", [u_abc, u_5, u_9, u_analytic])