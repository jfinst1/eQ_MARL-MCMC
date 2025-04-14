import os
import numpy as np
import torch
import pennylane as qml
import matplotlib.pyplot as plt
import json
import datetime
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import torch.nn as nn

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Parameters
n_stores = 2  # Number of agents (stores)
n_params_per_store = 2  # RX, RZ gates per store
n_params = n_stores * n_params_per_store
sigma_proposal = 0.5  # Initial proposal std
beta = 10.0  # Initial temperature
K = 15  # States for proposal evaluation
M = 200  # States for policy evaluation
n_mcmc_steps = 5000  # Total steps
n_products = 1  # Single product
hidden_dim = 8  # For BitLinear

# File paths
MODEL_FILE = 'best_model_weights.pt'
HISTORY_FILE = 'training_history.json'

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🛒 Quantum Store with Matrix-Free Twist! Running on: {device}")
print(f"🚀 Starting {n_mcmc_steps} MCMC steps, may take a few minutes...")

# BitLinear Layer (from matrix-free code)
class BitLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BitLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randint(-1, 2, (in_features, out_features)).float())

    def forward(self, x):
        return torch.matmul(x, self.weight)

# Custom Activation (from matrix-free code)
class CustomActivation(nn.Module):
    def __init__(self):
        super(CustomActivation, self).__init__()

    def forward(self, x):
        return torch.relu(x) * torch.sigmoid(x)

# Quantum Device
qnn_dev = qml.device("default.qubit", wires=n_stores)  # Fixed: no torch_device

# Quantum Circuit
@qml.qnode(qnn_dev, interface='torch')
def qmarl_circuit(states, weights):
    for i in range(n_stores):
        qml.RY(states[i] * np.pi, wires=i)  # Encode embedded sales
        qml.RX(weights[i * 2], wires=i)
        qml.RZ(weights[i * 2 + 1], wires=i)
    qml.CNOT(wires=[0, 1])  # Entangle stores
    qml.RY(weights[0] * weights[1], wires=1)
    qml.CNOT(wires=[1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_stores)]

# Action selection
def get_actions(states, weights, bitlinear, activation):
    # Convert NumPy array to tensor safely
    if isinstance(states, np.ndarray):
        states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    else:
        states_tensor = states.clone().detach().to(device=device, dtype=torch.float32)
    embedded = bitlinear(states_tensor)
    embedded = activation(embedded)
    embedded = torch.clamp(embedded, 0, 1).cpu()  # Move to CPU for QNode
    q_values = torch.stack(qmarl_circuit(embedded, weights))
    return (q_values >= 0).long().cpu().numpy()  # 0: don't restock, 1: restock

# Simulated retail environment with preprocessing
def simulate_sales_data(state, n_tickets=1000, store_id=0):
    """Simulate daily sales incidence and features."""
    if state == 0:  # OOS
        p = 0.0001
    elif state == 1:  # Low demand
        p = 0.009
    else:  # High demand
        p = 0.014
    incidence = np.random.binomial(n_tickets, p) / n_tickets
    return {
        'sales_incidence': incidence,
        'store_id': store_id,
        'transaction_type': np.random.choice(['online', 'in-store']),
        'timestamp': datetime.datetime.now().isoformat()
    }

def preprocess_sales_data(sales_records):
    """Preprocess sales data using matrix-free pipeline."""
    df = pd.DataFrame(sales_records)
    categorical_features = ['store_id', 'transaction_type']
    numeric_features = ['sales_incidence']

    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    features = preprocessor.fit_transform(df)  # Fixed: no .toarray()
    # print(f"Preprocessed features shape: {features.shape}")  # Debug
    return features

def simulate_latent_state(prev_state, transition_matrix):
    """Simulate next latent state with validation."""
    if prev_state not in [0, 1, 2]:
        raise ValueError(f"Invalid state: {prev_state}")
    probs = transition_matrix[prev_state]
    if not np.isclose(probs.sum(), 1.0, rtol=1e-5):
        raise ValueError(f"Probabilities for state {prev_state} sum to {probs.sum()}, not 1")
    if not all(probs >= 0):
        raise ValueError(f"Negative probabilities for state {prev_state}: {probs}")
    return np.random.choice([0, 1, 2], p=probs)

# Reward calculation
def compute_reward(actions, true_state):
    should_restock = 1 if true_state == 0 else 0
    return 1.0 if np.all(actions == should_restock) else 0.0

# Load previous model
def load_previous_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(HISTORY_FILE):
        weights = torch.load(MODEL_FILE, map_location=device)
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            previous_reward = history[-1]['final_eval_reward']
        print(f"📦 Loaded previous weights. Previous reward: {previous_reward:.3f}")
    else:
        weights = torch.randn(n_params, device=device) * 1.0
        previous_reward = 0.5
        torch.save(weights, MODEL_FILE)
        print("🏪 No previous model found. Initialized new weights.")
    return weights, previous_reward

# Save model weights
def save_model(weights):
    torch.save(weights, MODEL_FILE)
    print("📦 Saved best model weights.")

# Save training history
def save_history(history_entry):
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    else:
        history = []
    history.append(history_entry)
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history, f, indent=4)
    print("📜 Training history updated.")

# Reward-weighted distillation
def distill_models(weights_old, reward_old, weights_new, reward_new):
    total = reward_old + reward_new + 1e-6
    w_old = reward_old / total
    w_new = reward_new / total
    distilled_weights = w_old * weights_old + w_new * weights_new
    print("🔬 Performed reward-weighted distillation.")
    return distilled_weights

# Main training routine
def main():
    global sigma_proposal, beta
    # Transition matrix (fixed)
    transition_matrix = np.array([
        [0.74, 0.16, 0.10],  # OOS -> OOS, Low, High
        [0.02, 0.53, 0.45],  # Low -> OOS, Low, High
        [0.01, 0.42, 0.57]   # High -> OOS, Low, High
    ])

    weights_old, previous_reward = load_previous_model()
    weights = weights_old.clone()
    best_weights = weights.clone()
    best_reward = previous_reward

    # Initialize BitLinear and CustomActivation
    bitlinear = BitLinear(in_features=hidden_dim, out_features=n_stores).to(device)
    activation = CustomActivation().to(device)

    weights_trace, eval_rewards, accepted_list, state_trace = [], [], [], []

    # Simulate initial latent state
    current_state = np.random.choice([0, 1, 2])
    # print(f"Initial state: {current_state}")  # Debug

    for step in range(n_mcmc_steps):
        sigma_proposal = max(0.05, sigma_proposal * 0.999)
        beta = min(30, beta * 1.001)

        # Simulate sales data
        sales_records = [simulate_sales_data(current_state, store_id=i) for i in range(n_stores)]
        sales_features = preprocess_sales_data(sales_records)
        next_state = simulate_latent_state(current_state, transition_matrix)

        # Pad or truncate features to match hidden_dim
        sales_tensor = torch.tensor(sales_features, dtype=torch.float32, device=device)
        (f"Sales tensor shape before padding: {sales_tensor.shape}")  # Debug
        if sales_tensor.shape[1] < hidden_dim:
            sales_tensor = torch.nn.functional.pad(sales_tensor, (0, hidden_dim - sales_tensor.shape[1]))
        elif sales_tensor.shape[1] > hidden_dim:
            sales_tensor = sales_tensor[:, :hidden_dim]
        # print(f"Sales tensor shape after padding: {sales_tensor.shape}")  # Debug

        # Propose new weights
        proposed_weights = weights + torch.normal(0, sigma_proposal, size=weights.size(), device=device)

        # Evaluate rewards over K samples
        reward_current_sum = 0
        reward_proposed_sum = 0
        for _ in range(K):
            actions_current = get_actions(sales_tensor, weights, bitlinear, activation)
            actions_proposed = get_actions(sales_tensor, proposed_weights, bitlinear, activation)
            reward_current_sum += compute_reward(actions_current, current_state)
            reward_proposed_sum += compute_reward(actions_proposed, current_state)
        reward_current = reward_current_sum / K
        reward_proposed = reward_proposed_sum / K

        # Metropolis-Hastings
        alpha = min(1, np.exp(beta * (reward_proposed - reward_current)))
        accepted = np.random.rand() < alpha
        if accepted:
            weights = proposed_weights.clone()
        accepted_list.append(accepted)

        # Evaluate policy on M samples
        eval_reward_sum = 0
        for _ in range(M):
            actions = get_actions(sales_tensor, weights, bitlinear, activation)
            eval_reward_sum += compute_reward(actions, current_state)
        avg_eval_reward = eval_reward_sum / M

        # Update best weights
        if avg_eval_reward > best_reward:
            best_reward = avg_eval_reward
            best_weights = weights.clone()
            save_model(best_weights)
            print(f"🎉 New best reward: {best_reward:.3f} at step {step}")

        # Store traces
        weights_trace.append(weights.cpu().numpy().copy())
        eval_rewards.append(avg_eval_reward)
        state_trace.append(current_state)
        current_state = next_state

        # Progress update
        if step % 100 == 0:
            acceptance_rate = np.mean(accepted_list[-100:]) if len(accepted_list) >= 100 else np.mean(accepted_list)
            print(f"🏬 Step {step}, Eval Reward: {avg_eval_reward:.3f}, Acceptance Rate: {acceptance_rate:.3f}, State: {current_state}")

    final_eval_reward = eval_rewards[-1]
    print(f"🎯 Final Eval Reward: {final_eval_reward:.3f}")

    # Distill and save
    distilled_weights = distill_models(weights_old, previous_reward, best_weights, best_reward)
    save_model(distilled_weights)

    # Save history
    history_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "final_eval_reward": final_eval_reward,
        "acceptance_rate": np.mean(accepted_list),
        "steps": n_mcmc_steps,
        "best_reward": max(eval_rewards),
        "worst_reward": min(eval_rewards)
    }
    save_history(history_entry)

    # Visualizations
    sns.set_style("whitegrid")
    weights_trace = np.array(weights_trace)

    # Plot 1: Reward Evolution
    plt.figure(figsize=(12, 6))
    plt.plot(eval_rewards, label='Eval Reward', color='blue')
    plt.axhline(0.5, color='red', linestyle='--', label='Random Baseline (0.5)')
    plt.axhline(1.0, color='green', linestyle='--', label='Optimal (1.0)')
    plt.xlabel('MCMC Steps')
    plt.ylabel('Avg Reward')
    plt.title('🛍️ Quantum Stores with Matrix-Free Learning')
    plt.legend()
    plt.show()

    # Plot 2: Latent State Transitions
    plt.figure(figsize=(12, 6))
    plt.plot(state_trace, label='Latent State', color='purple', marker='o', linestyle='--')
    plt.yticks([0, 1, 2], ['OOS', 'Low', 'High'])
    plt.xlabel('MCMC Steps')
    plt.ylabel('State')
    plt.title('📊 Simulated Demand States Over Time')
    plt.legend()
    plt.show()

    # Plot 3: Sales vs. State
    sales_trace = [simulate_sales_data(s)['sales_incidence'] for s in state_trace]
    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(sales_trace)), sales_trace, c=state_trace, cmap='viridis', alpha=0.6)
    plt.colorbar(label='State (0=OOS, 1=Low, 2=High)')
    plt.xlabel('MCMC Steps')
    plt.ylabel('Sales Incidence')
    plt.title('💰 Sales Patterns by Latent State')
    plt.show()

    # Plot 4: Weights Evolution
    plt.figure(figsize=(12, 8))
    for i in range(n_params):
        plt.subplot(2, 2, i + 1)
        plt.plot(weights_trace[:, i], color='teal')
        plt.title(f"Weight {i + 1} (Store {i//2 + 1})")
        plt.xlabel('MCMC Step')
        plt.ylabel('Value')
    plt.suptitle("⚙️ Quantum Weights Evolution", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    main()