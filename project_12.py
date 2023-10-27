import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
import itertools

# Define the spin matrices
Sx = np.array([[0, 1], [1, 0]])
Sy = np.array([[0, -1j], [1j, 0]])
Sz = np.array([[1, 0], [0, -1]])

def set_data(i):
    scenarios = {
        0: {'J12': -1.1, 'J13': -2.1, 'J23': -3.8, 'h1': 0.6, 'h2': 0, 'h3': 0},
        1: {'J12': 1, 'J13': 2, 'J23': 2, 'h1': 1, 'h2': 0, 'h3': 0}
    }
    globals().update(scenarios.get(i))

set_data(0)

spin_states = [1, -1]
basis = np.array(list(itertools.product(spin_states, repeat=3)))
#print(basis)

def H_1():
    H1 = np.zeros((8,8))
    for i in range(8):
        H1[i,i] = -(J12*basis[i][0]*basis[i][1] +J13*basis[i][0]*basis[i][2] + J23*basis[i][1]*basis[i][2]+h1*basis[i][0] + h2*basis[i][1]+ h3*basis[i][2])
    return H1

def H_0():
    N = len(basis)
    H0 = np.zeros((N, N))

    for i, state_i in enumerate(basis):
        for j, state_j in enumerate(basis):
            for k in range(3):
                flipped_state = np.array(state_i)
                flipped_state[k] *= -1

                if np.array_equal(flipped_state, state_j):
                    H0[i][j] = -1

    return H0



def H(par):
    return par*H_1() + (1-par)*H_0()

#print(H(0))

def calculate_eigenvalues_and_eigenvectors(k):
    # Generate the Hamiltonian for the given k
    Hamiltonian = H(k)

    # Calculate the eigenvalues and eigenvectors of the Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(Hamiltonian)

    return eigenvalues, eigenvectors

# Test the function
eigenvalues, eigenvectors = calculate_eigenvalues_and_eigenvectors(1)
#print('Eigenvalues:', eigenvalues)
#print('Eigenvectors:', eigenvectors)

def calculate_energy_difference(eigenvalues):
    sorted_eigenvalues = np.sort(eigenvalues)
    energy_difference = sorted_eigenvalues[1] - sorted_eigenvalues[0]
    return energy_difference

energy_difference = calculate_energy_difference(eigenvalues)
#print('Energy difference:', energy_difference)

def calculate_energy_gap(basis):
    # Define the range of k values
    k_values = np.linspace(0, 1, 100)

    # Calculate the energy gap for each k value
    energy_gaps = [calculate_energy_difference(calculate_eigenvalues_and_eigenvectors(k)[0]) for k in k_values]

    return k_values, energy_gaps

# Test the function
k_values, energy_gaps = calculate_energy_gap(basis)

# Plot the energy gap as a function of k
plt.plot(k_values, energy_gaps)
plt.xlabel('k')
plt.ylabel('Energy gap')
plt.title('Energy gap as a function of k')
plt.grid(True)
plt.show()
