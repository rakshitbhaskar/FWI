import matplotlib.pyplot as plt

def plot_misfit(misfits):
    """
    Plot convergence history.
    """

    plt.figure(figsize=(6,4))

    plt.plot(
        range(1, len(misfits)+1),
        misfits,
        "-o"
    )

    plt.xlabel("Iteration")
    plt.ylabel("Misfit")
    plt.title("FWI Convergence")

    plt.grid(True)

    plt.tight_layout()

    plt.show()