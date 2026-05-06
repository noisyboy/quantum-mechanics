# quantum-mechanics
Quantum mechanics problems in python using numpy

Let's walk through the mathematics of the finite difference method (FDM) applied to the 1D Time-Independent Schrödinger Equation (TISE).

## 1. The Continuous Equation
We begin with the standard 1D TISE:

$$-\frac{\hbar^2}{2m} \frac{d^2\psi(x)}{dx^2} + V(x)\psi(x) = E\psi(x)$$

Here, we are looking for the eigenvalues $E$ (allowed energy states) and the eigenfunctions $\psi(x)$ (wavefunctions) for a given potential $V(x)$.

## 2. Discretizing Space
Computers cannot inherently process continuous domains, so we must restrict our problem to a bounded region and slice it into a discrete grid.

Imagine bounding the particle in a region from $x_{min}$ to $x_{max}$. We divide this region into $N$ internal grid points, each separated by a small, uniform step size $\Delta x$.

* **Grid points:** $x_i = x_{min} + i\Delta x$ for $i = 0, 1, 2, \dots, N, N+1$.
* **Step size:** $\Delta x = \frac{x_{max} - x_{min}}{N+1}$.
* **Discrete wavefunction:** $\psi(x_i)$ becomes $\psi_i$.
* **Discrete potential:** $V(x_i)$ becomes $V_i$.

## 3. The Finite Difference Approximation
The core of FDM is replacing exact derivatives with algebraic approximations. For the kinetic energy term, we need the second derivative of the wavefunction. Using the central finite difference approximation (derived from Taylor series expansion), we get:

$$\frac{d^2\psi_i}{dx^2} \approx \frac{\psi_{i+1} - 2\psi_i + \psi_{i-1}}{\Delta x^2}$$

This formula tells us that the curvature of the wavefunction at point $x_i$ depends strictly on its value at that point and its immediate neighbors.

## 4. Constructing the Discrete TISE
Substitute this approximation back into the original Schrödinger equation at a specific grid point $i$:

$$-\frac{\hbar^2}{2m} \left( \frac{\psi_{i+1} - 2\psi_i + \psi_{i-1}}{\Delta x^2} \right) + V_i\psi_i = E\psi_i$$

Now, let's group the terms by $\psi_{i-1}$, $\psi_i$, and $\psi_{i+1}$. To keep the notation clean, let's define a constant $t = \frac{\hbar^2}{2m\Delta x^2}$ (which represents the kinetic energy coupling between adjacent points). The equation reorganizes to:

$$-t\psi_{i-1} + (2t + V_i)\psi_i - t\psi_{i+1} = E\psi_i$$

## 5. Applying Boundary Conditions
To close the system, we need boundary conditions. The simplest and most common approach for bound states is to assume the particle is trapped in an infinitely deep box spanning from $x_{min}$ to $x_{max}$. This enforces Dirichlet boundary conditions:

* $\psi_0 = 0$
* $\psi_{N+1} = 0$

Because the wavefunction is zero at the boundaries, we only need to solve for the $N$ internal points ($\psi_1$ through $\psi_N$).

## 6. The Matrix Formulation
We now have a system of $N$ linear equations. In linear algebra terms, this translates perfectly into an eigenvalue problem:

$$H\vec{\psi} = E\vec{\psi}$$

Where $H$ is the Hamiltonian matrix and $\vec{\psi}$ is the column vector of our discrete wavefunction values.

Because our finite difference formula only couples a point $i$ to its immediate neighbors $i-1$ and $i+1$, the Hamiltonian matrix $H$ is **tridiagonal**. All elements are zero except for the main diagonal and the two adjacent diagonals:

$$H = \begin{pmatrix}
2t + V_1 & -t & 0 & \dots & 0 \\
-t & 2t + V_2 & -t & \dots & 0 \\
0 & -t & 2t + V_3 & \dots & 0 \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
0 & 0 & 0 & -t & 2t + V_N
\end{pmatrix}$$

* **Main Diagonal:** $2t + V_i$ (Contains both the self-energy from the kinetic term and the local potential energy).
* **Off-Diagonals:** $-t$ (Represents the kinetic energy transition amplitude between neighboring spatial points).

By reducing the differential equation to this tridiagonal matrix $H$, the entire physics problem shifts. We simply need to construct this matrix in Python and use a linear algebra solver to extract its eigenvalues (the energy levels $E$) and eigenvectors (the wavefunctions $\vec{\psi}$).
