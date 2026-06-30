# Seismic Forward Modeling and Inversion Framework

This repository contains a comprehensive suite of tools for seismic wavefield simulation and geophysical optimization, ranging from basic finite-difference experiments to complex model reconstructions.


### 1. Forward Modeling
**Focus:** Wavefield simulation and boundary physics.
- Implementation of various finite-difference algorithms.
- Comparative analysis of boundary conditions:
    - **ABC** (Absorbing Boundary Conditions)
    - **PML** (Perfectly Matched Layers) for superior edge-reflection suppression.

### 2.Inversion (Optimization Algorithms)
**Focus:** The mathematical core of Full Waveform Inversion (FWI).
- **Gradient Descent:** Basic first-order optimization.
- **Gauss-Newton:** Approximating the Hessian for faster convergence.
- **Full Newton Method:** Utilizing the complete Hessian for high-precision inversion.

### 3.Visualization (Square Anomaly Results)
**Focus:** Benchmarking and Proof of Concept.
- Testing the inversion workflow on a synthetic square anomaly.
- Visualizing convergence rates and reconstruction accuracy across different optimization methods.

### 4. Marmousi Model Application
**Focus:** Real-world complexity.
- Applying the finalized algorithms (GN) to the classic **Marmousi Model**.
- Demonstrating the robustness of the forward engine and optimizer on complex structural velocity models.

##  Getting Started
*   **Prerequisites:** Python 3.x, NumPy, Matplotlib, SciPy.
*   **Usage:** Each notebook is self-contained. It is recommended to follow the numerical order (1–4) to understand the progression from modeling to inversion.
