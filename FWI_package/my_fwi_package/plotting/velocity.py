import matplotlib.pyplot as plt
import numpy as np

def plot_velocity(m, title="Velocity Model"):
    """
    Plot velocity model from squared slowness.
    """

    c = 1 / np.sqrt(m)

    plt.figure(figsize=(6,5))

    plt.imshow(
        c,
        origin="upper",
        cmap="jet",
        aspect="auto"
    )

    plt.colorbar(label="Velocity (m/s)")
    plt.title(title)
    plt.xlabel("X Grid")
    plt.ylabel("Z Grid")

    plt.tight_layout()
    plt.show()