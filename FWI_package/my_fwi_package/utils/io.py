from pathlib import Path
import numpy as np

def save_results(filename, model, misfit):
    # Create the parent directory if it doesn't exist
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        filename,
        model=model,
        misfit=np.array(misfit),
    )


def load_results(filename):
    data = np.load(filename)

    return data["model"], data["misfit"]

def save_wavefield_results(filename, forward_wavefield):
    # Create the parent directory if it doesn't exist
    Path(filename).parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        filename,
        forward_wavefield=forward_wavefield
    )

def load_wavefield_results(filename):
    data = np.load(filename)

    return data["forward_wavefield"]