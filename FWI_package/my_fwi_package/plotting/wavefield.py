import scipy.sparse.linalg as spla
import numpy as np
import matplotlib.pyplot as plt
from my_fwi_package.utils import create_geometry
geometry = create_geometry()

def plot_wavefield(uabc, u5, u9, u_analytic, f_plot,s_id):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))

    titles = [
    "Helmholtz 5-point",
    "Helmholtz 9-point",
    "Helmholtz ABC",
    "Analytic Solution"
    ]

    wavefields = [
    np.real(u5),
    np.real(u9),
    np.real(uabc),
    np.real(u_analytic)
    ]

    for ax, wf, title in zip(axs, wavefields, titles):

        im = ax.imshow(
        wf,
        cmap='RdBu',
        extent=geometry["model_extent"],
        aspect='equal'
        )

        ax.scatter(
        geometry["src_x_phys"][s_id],
        geometry["src_z_phys"],
        c='red',
        marker='*',
        s=80
        )

        ax.scatter(
        geometry["rec_x_phys"],
        [geometry["rec_z_phys"]] * len(geometry["rec_x_phys"]),
        c='black',
        marker='v',
        s=10
        )

        ax.set_title(title)
        ax.set_xlabel("Distance (m)")
        ax.set_ylabel("Depth (m)")

        fig.colorbar(im, ax=ax, fraction=0.046)

    plt.suptitle(f"Wavefield Comparison at {f_plot} Hz")
    plt.tight_layout()
    plt.show()