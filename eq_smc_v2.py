import numpy as np
import torch
import torch.nn as nn
import pennylane as qml
import matplotlib.pyplot as plt
import multiprocessing as mp
from time import time
import logging
from scipy.ndimage import griddata

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set random seeds and device
np.random.seed(42)
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Initializing with device: {device}")

# --- Step 1: SAGIN Domain Modeling ---
def generate_sagin_domain(theta, N=64):
    """Generate a domain representing CubeSat/HALE-UAV positions."""
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    a1, a2, a3 = theta  # a1: amplitude, a2: frequency, a3: phase
    x = X + a1 * np.cos(a2 * np.pi + a3)
    y = Y + a1 * np.sin(a2 * np.pi + a3)
    return x, y

def reference_domain(N=64):
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    return np.stack([X, Y], axis=-1)

def subsampled_reference_domain(N=8):
    X, Y = np.meshgrid(np.linspace(0, 1, N), np.linspace(0, 1, N))
    return np.stack([X, Y], axis=-1)

def diffeomorphism(x, y, theta):
    a1, a2, a3 = theta
    X = x - a1 * np.cos(a2 * np.pi + a3)
    Y = y - a1 * np.sin(a2 * np.pi + a3)
    return X, Y

def solve_energy_pde(theta, bc, N=64):
    """Solve a PDE representing energy distribution."""
    x, y = generate_sagin_domain(theta, N)
    u = np.zeros((N, N))  # Energy field
    u[0, :] = bc[0]  # Sun side
    u[-1, :] = bc[1]  # Dark side
    u[:, 0] = bc[2]  # HALE-UAV boundary
    u[:, -1] = bc[3]  # Opposite boundary
    for _ in range(2000):
        u[1:-1, 1:-1] = 0.25 * (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2])
    return u

# --- Step 2: EnhancedMIONet with Quantum Layer ---
class EnhancedMIONet(nn.Module):
    def __init__(self, theta_dim=3, bc_dim=4, hidden_dim=512, num_quantum_weights=6):
        super(EnhancedMIONet, self).__init__()
        self.device = device
        
        self.branch_theta = nn.Sequential(
            nn.Linear(theta_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.branch_bc = nn.Sequential(
            nn.Linear(bc_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.trunk = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.final_layer = nn.Linear(hidden_dim, 1)
        
        self.num_quantum_weights = num_quantum_weights
        self.quantum_weights = nn.Parameter(torch.randn(num_quantum_weights, device=self.device) * 0.1)
        self.quantum_dev = qml.device("default.qubit.torch", wires=2, torch_device=device)
        
        @qml.qnode(self.quantum_dev, interface='torch')
        def quantum_circuit(inputs, weights):
            inputs = torch.pi * (inputs - inputs.min(dim=1, keepdim=True)[0]) / \
                     (inputs.max(dim=1, keepdim=True)[0] - inputs.min(dim=1, keepdim=True)[0] + 1e-8)
            for i in range(6):
                qml.RY(inputs[..., i], wires=i % 2)
                qml.RX(weights[i], wires=i % 2)
            qml.CNOT(wires=[0, 1])
            return qml.expval(qml.PauliZ(0)), qml.expval(qml.PauliZ(1))
        
        self.quantum_circuit = quantum_circuit
    
    def quantum_layer(self, inputs):
        z0, z1 = self.quantum_circuit(inputs, self.quantum_weights)
        return ((z0 + z1) / 2).to(dtype=torch.float32)
    
    def forward(self, theta, bc, X_ref):
        batch_size = theta.shape[0]
        n_points = X_ref.shape[-2]
        
        theta_out = self.branch_theta(theta)
        bc_out = self.branch_bc(bc)
        trunk_out = self.trunk(X_ref)
        
        if theta_out.dim() == 3:
            theta_out = theta_out.squeeze(1)
        if bc_out.dim() == 3:
            bc_out = bc_out.squeeze(1)
        
        theta_out = theta_out.unsqueeze(1).expand(-1, n_points, -1)
        bc_out = bc_out.unsqueeze(1).expand(-1, n_points, -1)
        if trunk_out.dim() == 2:
            trunk_out = trunk_out.unsqueeze(0).expand(batch_size, -1, -1)
        
        combined = theta_out * bc_out * trunk_out
        quantum_input = torch.cat((theta, bc[..., :3]), dim=-1)
        quantum_output = self.quantum_layer(quantum_input)
        final_output = self.final_layer(combined) * (1 + quantum_output.unsqueeze(-1).unsqueeze(-1))
        return final_output

    def compute_pde_residual(self, theta, bc, X_ref_sub):
        # Placeholder for PDE residual computation
        return torch.tensor(0.0, device=device)  # Replace with actual computation if needed

# --- Step 3: QMARLScheduler with SMC-based Bayesian Approach ---
class QMARLScheduler:
    def __init__(self, n_gs=5, n_cubesats=3, n_hale_uavs=2, n_actions=8, hidden_dim=256):
        self.n_gs = n_gs  # Number of ground stations (agents)
        self.n_cubesats = n_cubesats
        self.n_hale_uavs = n_hale_uavs
        self.n_ntn = n_cubesats + n_hale_uavs
        self.n_actions = n_actions  # Actions per ground station
        self.device = device
        
        # Quantum circuit setup
        self.qnn_dev = qml.device("default.qubit.torch", wires=self.n_gs, torch_device=device)
        
        # SMC parameters
        self.n_particles = 10  # Number of particles
        self.particles = [torch.randn(self.n_gs * 3, device=device) * 0.1 
                          for _ in range(self.n_particles)]  # Prior: N(0, 0.1^2)
        self.importance_weights = torch.ones(self.n_particles, device=device) / self.n_particles
        self.gamma = 0.1  # Learning rate for importance weight updates
        self.ess_threshold = 0.5 * self.n_particles  # Resampling threshold
        
        @qml.qnode(self.qnn_dev, interface='torch')
        def qmarl_circuit(states, weights):
            for i in range(self.n_gs):
                qml.RY(states[i, 0], wires=i)  # State encoding
                qml.RX(weights[i * 3], wires=i)
                qml.RZ(weights[i * 3 + 1], wires=i)
                if i < self.n_gs - 1:
                    qml.CNOT(wires=[i, i + 1])  # Entanglement
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_gs)]
        
        self.qmarl_circuit = qmarl_circuit
    
    def sample_particle(self):
        """Sample a particle index based on importance weights."""
        probs = self.importance_weights / self.importance_weights.sum()
        return np.random.choice(self.n_particles, p=probs.cpu().numpy())
    
    def get_actions(self, states, particle_idx):
        """Get actions using the specified particle's weights."""
        weights = self.particles[particle_idx]
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        q_values = torch.stack(self.qmarl_circuit(states_tensor, weights))  # [n_gs]
        actions = (q_values * self.n_actions / 2 + self.n_actions / 2).long() % self.n_actions
        return actions.cpu().numpy()
    
    def compute_reward(self, actions, energy_states, qos_demands):
        """Compute reward based on actions and states."""
        reward = 0.0
        for i, action in enumerate(actions):
            if action < self.n_cubesats:
                energy = energy_states[action]
                reward += energy * 0.5 + qos_demands[i] * 0.5
            elif action < self.n_ntn:
                energy = energy_states[action]
                reward += energy * 0.7 + qos_demands[i] * 0.3
        return reward / self.n_gs
    
    def update_importance_weight(self, particle_idx, reward):
        """Update the importance weight of the selected particle."""
        self.importance_weights[particle_idx] *= torch.exp(self.gamma * reward)
        self.importance_weights /= self.importance_weights.sum()  # Normalize
        
        # Check effective sample size (ESS) and resample if needed
        ess = 1 / (self.importance_weights ** 2).sum()
        if ess < self.ess_threshold:
            self.resample_particles()
    
    def resample_particles(self):
        """Resample particles based on importance weights and add noise."""
        probs = self.importance_weights / self.importance_weights.sum()
        indices = np.random.choice(self.n_particles, size=self.n_particles, 
                                   p=probs.cpu().numpy())
        new_particles = [self.particles[i].clone() for i in indices]
        for p in new_particles:
            p += torch.randn_like(p) * 0.01  # Add small noise for diversity
        self.particles = new_particles
        self.importance_weights = torch.ones(self.n_particles, device=device) / self.n_particles

# --- Step 4: Integrated eQ-DIMON + QMARL ---
class SAGINOptimizer:
    def __init__(self, batch_size=64, n_gs=5, n_cubesats=3, n_hale_uavs=2):
        self.eq_dimon = EnhancedMIONet().to(device)
        self.qmarl = QMARLScheduler(n_gs, n_cubesats, n_hale_uavs)
        self.optimizer = torch.optim.Adam(self.eq_dimon.parameters(), lr=0.001)
        self.batch_size = batch_size
        self.n_gs = n_gs
        self.n_ntn = n_cubesats + n_hale_uavs
    
    def train(self, data, epochs=5):
        X_ref_sub = torch.tensor(subsampled_reference_domain(N=8), dtype=torch.float32, device=device)
        X_ref_full = reference_domain()
        
        for epoch in range(epochs):
            total_loss = 0.0
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                theta_batch, bc_batch, _, u_batch = zip(*batch)
                theta_tensor = torch.tensor(np.stack(theta_batch), dtype=torch.float32, device=device)
                bc_tensor = torch.tensor(np.stack(bc_batch), dtype=torch.float32, device=device)
                
                # PDE solution for energy states
                X_ref_full_tensor = torch.tensor(X_ref_full.reshape(-1, 2), dtype=torch.float32, device=device)
                energy_pred = self.eq_dimon(theta_tensor, bc_tensor, X_ref_full_tensor).squeeze(-1)
                
                # Simulate states and QoS demands
                states = np.random.rand(self.n_gs, 1)  # Simplified energy levels
                qos_demands = np.random.rand(self.n_gs)
                
                # QMARL scheduling with SMC
                particle_idx = self.qmarl.sample_particle()
                actions = self.qmarl.get_actions(states, particle_idx)
                energy_states = energy_pred.mean(dim=1).cpu().numpy()[:self.n_ntn]
                reward = self.qmarl.compute_reward(actions, energy_states, qos_demands)
                
                # Train EnhancedMIONet
                self.optimizer.zero_grad()
                pde_loss = self.eq_dimon.compute_pde_residual(theta_tensor, bc_tensor, X_ref_sub)
                loss = pde_loss - 0.1 * reward  # Encourage high reward
                loss.backward()
                self.optimizer.step()
                
                # Update QMARL particles
                self.qmarl.update_importance_weight(particle_idx, reward)
                total_loss += loss.item()
            
            logging.info(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / (len(data) // self.batch_size):.6f}")
    
    def predict(self, theta, bc, X_ref):
        theta_tensor = torch.tensor(theta, dtype=torch.float32, device=device).unsqueeze(0)
        bc_tensor = torch.tensor(bc, dtype=torch.float32, device=device).unsqueeze(0)
        X_ref_tensor = torch.tensor(X_ref.reshape(-1, 2), dtype=torch.float32, device=device)
        with torch.no_grad():
            u_pred = self.eq_dimon(theta_tensor, bc_tensor, X_ref_tensor).squeeze(0).cpu().numpy()
        return u_pred

# --- Step 5: Data Generation ---
def generate_sagin_data_worker(args):
    theta, bc = args
    X_ref = reference_domain()
    u = solve_energy_pde(theta, bc)
    X_mapped, Y_mapped = diffeomorphism(*generate_sagin_domain(theta), theta)
    u_ref = griddata((X_mapped.flatten(), Y_mapped.flatten()), u.flatten(), (X_ref[..., 0], X_ref[..., 1]), method='cubic')
    return (theta, bc, X_ref, u_ref)

def generate_sagin_data(n_samples=300):
    pool = mp.Pool(mp.cpu_count())
    thetas = np.random.uniform([-0.5, 0, -0.5], [0.5, 0.75, 0.5], (n_samples, 3))
    bcs = np.random.uniform(0, 1, (n_samples, 4))
    data = pool.map(generate_sagin_data_worker, zip(thetas, bcs))
    pool.close()
    return data

# --- Main Execution ---
if __name__ == "__main__":
    start_time = time()
    logging.info("Generating SAGIN training data...")
    data = generate_sagin_data(n_samples=300)
    
    logging.info("Training SAGIN Optimizer...")
    sagin_opt = SAGINOptimizer(batch_size=64, n_gs=5, n_cubesats=3, n_hale_uavs=2)
    sagin_opt.train(data, epochs=5)
    
    logging.info("Testing on a sample...")
    theta_test, bc_test, X_ref, u_true = data[0]
    u_pred = sagin_opt.predict(theta_test, bc_test, X_ref)
    x_test, y_test = generate_sagin_domain(theta_test)
    
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.contourf(x_test, y_test, u_true, levels=20)
    plt.title("True Energy Distribution")
    plt.colorbar()
    plt.subplot(1, 2, 2)
    plt.contourf(x_test, y_test, griddata((X_ref[..., 0].flatten(), X_ref[..., 1].flatten()), 
                                          u_pred.flatten(), (x_test, y_test), method='cubic'), levels=20)
    plt.title("Predicted Energy Distribution")
    plt.colorbar()
    plt.show()
    
    logging.info(f"Total runtime: {time() - start_time:.2f} seconds")