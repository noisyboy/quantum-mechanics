import numpy as np
import matplotlib.pyplot as plt

# 1. Spatial Grid Setup

L = 6.0          # Spatial boundaries [-L, L]
N = 1000         # Grid points
x = np.linspace(-L, L, N)
h = x[1] - x[0]

def solve_pt_symmetric_dense(alpha):
    """
    Constructs the dense non-Hermitian Hamiltonian using np.diag 
    and computes the eigenvalues using np.linalg.eig.
    """
    #  complex potential across the grid
    V = 0.5 * x**2 + 1j * alpha * x**3

    # Construct the diagonal arrays
    main_diag = np.full(N, 1.0 / h**2, dtype=complex) + V
    off_diag = np.full(N - 1, -1.0 / (2.0 * h**2), dtype=complex)

    # Assemble the dense matrix 
    H = np.diag(main_diag) + np.diag(off_diag, k=1) + np.diag(off_diag, k=-1)

    # Solve for ALL eigenvalues using the general complex solver
    evals, evecs = np.linalg.eig(H)

    # Sort the output based on the real part of the eigenvalues
    idx = np.argsort(np.real(evals))
    return evals[idx], evecs[:, idx]

# ==========================================
# 2. Phase Transition Sweep (Finding the EP)
# ==========================================
alpha_values = np.linspace(0.0, 0.3, 80)
num_levels = 4  # Number of lowest energy levels to track

# Arrays to store the progression of the eigenvalues
real_energies = np.zeros((len(alpha_values), num_levels))
imag_energies = np.zeros((len(alpha_values), num_levels))

print("Assembling dense matrices and sweeping alpha values...")

for i, alpha in enumerate(alpha_values):
    # Retrieve sorted eigenvalues and eigenvectors
    evals, _ = solve_pt_symmetric_dense(alpha)
    
    # Store the first 'num_levels' energies
    real_energies[i, :] = np.real(evals[:num_levels])
    imag_energies[i, :] = np.imag(evals[:num_levels])

# ==========================================
# 3. Plotting the Results
# ==========================================
# Set up a dark-themed plot to match a sleek presentation aesthetic
plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

colors = ['cyan', 'gold', 'hotpink', 'springgreen']

# Left Plot: Real part of the eigenvalues
for n in range(num_levels):
    ax1.plot(alpha_values, real_energies[:, n], color=colors[n], 
             linewidth=2.5, label=f'$E_{n}$')
    
ax1.set_title("Real Part of Eigenvalues", fontsize=14)
ax1.set_xlabel(r"Coupling Strength ($\alpha$)", fontsize=12)
ax1.set_ylabel(r"Re($E$)", fontsize=12)
ax1.grid(True, alpha=0.2)
ax1.legend(loc='upper left')

# Right Plot: Imaginary part of the eigenvalues
for n in range(num_levels):
    ax2.plot(alpha_values, imag_energies[:, n], color=colors[n], 
             linewidth=2.5, label=f'$E_{n}$')

ax2.set_title("Imaginary Part of Eigenvalues", fontsize=14)
ax2.set_xlabel(r"Coupling Strength ($\alpha$)", fontsize=12)
ax2.set_ylabel(r"Im($E$)", fontsize=12)
ax2.grid(True, alpha=0.2)

# Estimate and draw a vertical line for the Exceptional Point (EP)
# We find where the imaginary part of the ground state becomes non-zero
threshold = 1e-3
broken_indices = np.where(np.abs(imag_energies[:, 0]) > threshold)[0]
if len(broken_indices) > 0:
    ep_alpha = alpha_values[broken_indices[0]]
    ax1.axvline(ep_alpha, color='red', linestyle='--', alpha=0.7)
    ax2.axvline(ep_alpha, color='red', linestyle='--', alpha=0.7)
    ax1.text(ep_alpha + 0.01, 3.5, "Exceptional Point", color='red', fontsize=11)

plt.suptitle(r"$\mathcal{PT}$-Symmetry Spontaneous Breaking Phase Transition", fontsize=16)
plt.tight_layout()
plt.show()