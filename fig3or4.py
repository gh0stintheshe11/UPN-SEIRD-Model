import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class GIRG:
    def __init__(self, n, d, tau, alpha, expected_weight):
        self.n = n
        self.d = d
        self.tau = tau
        self.alpha = alpha
        self.expected_weight = expected_weight
        self.positions = np.random.uniform(size=(n, d))
        self.weights = self._generate_weights(tau, expected_weight, n)
        self.graph = self._generate_graph()

    def _generate_weights(self, tau, expected_weight, size):
        return np.random.random(size)**(-1/(tau-1))*(expected_weight)*(tau-2)/(tau-1)

    def _distance_matrix(self):
        return squareform(pdist(self.positions))

    def _generate_graph(self):
        dist_matrix = self._distance_matrix()
        graph = nx.Graph()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                p = 1 - np.exp(- (self.weights[i] * self.weights[j]) / (dist_matrix[i, j] ** self.d) ** self.alpha)
                if np.random.random() < p:
                    graph.add_edge(i, j)
        return graph

# Initialize the GIRG network
n = 1000  # Number of nodes
d = 2  # Dimensionality
tau = 2.5  # Power-law exponent for long-range network
alpha = 2  # Distance dependence exponent
expected_weight = 1  # Expected weight

print("Generating the network...")
girg = GIRG(n, d, tau, alpha, expected_weight)
G = girg.graph
print("Network generated.")

# UPN-SEIRD model parameters
beta_u = 0.17
beta_p = 0.2 * beta_u
beta_n = 0.2 * beta_u
sigma = 1 / 5
gamma = 1 / 14
f = 0.01
kappa = 1
lambda_plus = 0.05 * kappa  # Reduced transition rate to positive spreader
lambda_minus = 0.05 * kappa  # Reduced transition rate to negative spreader
delta_plus = 1 / 300  # Increased forgetting rate for positive information
delta_minus = 1 / 300  # Increased forgetting rate for negative information

# Adjusted initial conditions
initial_infected = 50  # Increased initial infected
initial_aware = 10  # Increased initial aware
N = len(G)
i_0 = initial_infected / N
a_0 = initial_aware / N

# Initialize state variables
S = np.full(N, 1)
E = np.zeros(N)
I = np.zeros(N)
R = np.zeros(N)
D = np.zeros(N)
U = np.full(N, 1 - a_0)
P = np.zeros(N)
N_nodes = np.zeros(N)

infected_nodes = np.random.choice(np.arange(N), initial_infected, replace=False)
aware_nodes = np.random.choice(np.arange(N), initial_aware, replace=False)

I[infected_nodes] = 1
U[aware_nodes] = 0
P[aware_nodes] = 1 - a_0

# Ensure U + P + N = 1 initially
U[aware_nodes] = 0
P[aware_nodes] = 1
N_nodes[aware_nodes] = 0
total = U + P + N_nodes
U /= total
P /= total
N_nodes /= total

# Time parameters
t_max = 150  # Maximum time
dt = 0.01  # Time step
time_steps = np.arange(0, t_max, dt)

# Initialize lists to store results
S_list = []
E_list = []
I_list = []
R_list = []
D_list = []
U_list = []
P_list = []
N_list = []

print("Starting the simulation...")
# Stochastic simulation
for t_index, t in enumerate(time_steps):
    if t_index % 100 == 0:
        print(f"Simulation at time step {t_index}/{len(time_steps)} (time = {t:.1f})")
    
    # Store current state
    S_list.append(S.sum() / N)
    E_list.append(E.sum() / N)
    I_list.append(I.sum() / N)
    R_list.append(R.sum() / N)
    D_list.append(D.sum() / N)
    U_list.append(U.sum() / N)
    P_list.append(P.sum() / N)
    N_list.append(N_nodes.sum() / N)
    
    # Calculate transition probabilities
    adjacency_matrix = nx.to_numpy_array(G)
    new_E = (beta_u * S * adjacency_matrix.dot(I)) * dt
    new_I = (sigma * E) * dt
    new_R = ((1 - f) * gamma * I) * dt
    new_D = (f * gamma * I) * dt
    new_P = (lambda_plus * U * adjacency_matrix.dot(P) + kappa * I * U) * dt
    new_U_from_P = (delta_plus * P) * dt
    new_N = (lambda_minus * U * adjacency_matrix.dot(N_nodes) + kappa * I * U) * dt
    new_U_from_N = (delta_minus * N_nodes) * dt
    
    # Update states
    S = np.clip(S - new_E, 0, 1)
    E = np.clip(E + new_E - new_I, 0, 1)
    I = np.clip(I + new_I - new_R - new_D, 0, 1)
    R = np.clip(R + new_R, 0, 1)
    D = np.clip(D + new_D, 0, 1)
    U = np.clip(U - new_P - new_N + new_U_from_P + new_U_from_N, 0, 1)
    P = np.clip(P + new_P - new_U_from_P, 0, 1)
    N_nodes = np.clip(N_nodes + new_N - new_U_from_N, 0, 1)
    
    # Normalize to ensure U + P + N = 1
    UPN_total = U + P + N_nodes
    U /= UPN_total
    P /= UPN_total
    N_nodes /= UPN_total

print("Simulation complete. Plotting results...")

# Plotting results
fig, axs = plt.subplots(2, 1, figsize=(12, 16))

# Plot SEIRD components
axs[0].plot(time_steps * 30, S_list, label='Susceptible (S) (30*x-axis)')  # Expand x-axis by 50 for S
axs[0].plot(time_steps, E_list, label='Exposed (E)')
axs[0].plot(time_steps, I_list, label='Infected (I)')
axs[0].plot(time_steps, R_list, label='Recovered (R)')
axs[0].plot(time_steps, np.array(D_list) * 30, label='Deceased (D) (30*y-axis)')  # Expand y-axis by 50 for D
axs[0].set_xlabel('Time (days)')
axs[0].set_xlim([0, 80])  # Set x-axis range to 0-80 for SEIRD plot
axs[0].set_ylabel('Proportion')
axs[0].set_title('SEIRD Model Dynamics')
axs[0].grid(True)
axs[0].legend(loc='upper right')  # Move legend to the upper right

# Plot UPN components
axs[1].plot(time_steps, U_list, label='Unaware (U)')
axs[1].plot(time_steps, P_list, label='Positive Spreader (P)')
axs[1].plot(time_steps, N_list, label='Negative Spreader (N)')
axs[1].set_xlabel('Time (days)')
axs[1].set_xlim([0, 5])  # Set x-axis range to 0-5 for UPN plot
axs[1].set_ylabel('Proportion')
axs[1].set_title('UPN Model Dynamics')
axs[1].legend(loc='upper right')
axs[1].grid(True)

plt.tight_layout()
plt.subplots_adjust(hspace=0.4)  # Adjust the height space between subplots
plt.show()
print("Plotting complete.")
