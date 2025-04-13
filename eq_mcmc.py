import os
import numpy as np
import torch
import pennylane as qml
import matplotlib.pyplot as plt
import json
import datetime

# Parameters
n_gs = 2
n_params_per_agent = 2
n_params = n_gs * n_params_per_agent

sigma_proposal = max(0.05, sigma_proposal * 0.999)
beta = min(30, beta * 1.001)

K = 15
M = 200
n_mcmc_steps = 5000

# File paths
MODEL_FILE = 'best_model_weights.pt'
HISTORY_FILE = 'training_history.json'

# Device configuration (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Quantum Device
qnn_dev = qml.device("default.qubit", wires=n_gs)

# Quantum Circuit
@qml.qnode(qnn_dev, interface='torch')
def qmarl_circuit(states, weights):
    for i in range(n_gs):
        qml.RY(states[i] * np.pi, wires=i)
        qml.RX(weights[i * 2], wires=i)
        qml.RZ(weights[i * 2 + 1], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.RY(weights[0] * weights[1], wires=1)
    qml.CNOT(wires=[1, 0])
    return [qml.expval(qml.PauliZ(i)) for i in range(n_gs)]

# Action selection
def get_actions(states, weights):
    states_tensor = torch.tensor(states, dtype=torch.float32, device=device)
    q_values = torch.stack(qmarl_circuit(states_tensor, weights))
    return (q_values >= 0).long().cpu().numpy()

# Reward calculation
def compute_reward(actions, s):
    optimal_action = 0 if s < 0.5 else 1
    return 1.0 if np.all(actions == optimal_action) else 0.0

# Load previous model
def load_previous_model():
    if os.path.exists(MODEL_FILE) and os.path.exists(HISTORY_FILE):
        weights = torch.load(MODEL_FILE, map_location=device)
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
            previous_reward = history[-1]['final_eval_reward']
        print(f"Loaded previous weights. Previous reward: {previous_reward:.3f}")
    else:
        weights = torch.randn(n_params, device=device) * 1.0
        previous_reward = 0.5
        torch.save(weights, MODEL_FILE)
        print("No previous model found. Initialized new weights.")
    return weights, previous_reward

# Save model weights
def save_model(weights):
    torch.save(weights, MODEL_FILE)
    print("Saved best model weights.")

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
    print("Training history updated.")

# Reward-weighted distillation
def distill_models(weights_old, reward_old, weights_new, reward_new):
    total = reward_old + reward_new
    w_old = reward_old / total
    w_new = reward_new / total
    distilled_weights = w_old * weights_old + w_new * weights_new
    print("Performed reward-weighted distillation.")
    return distilled_weights

# Main training routine
def main():
    global sigma_proposal, beta
    weights_old, previous_reward = load_previous_model()
    weights = weights_old.clone()

    weights_trace, eval_rewards, accepted_list = [], [], []

    for step in range(n_mcmc_steps):
        sigma_proposal = max(0.05, sigma_proposal * 0.995)
        beta = min(30, beta * 1.005)

        proposed_weights = weights + torch.normal(0, sigma_proposal, size=weights.size(), device=device)

        s_samples = np.random.rand(K)
        reward_current = np.mean([compute_reward(get_actions([s, s], weights), s) for s in s_samples])
        reward_proposed = np.mean([compute_reward(get_actions([s, s], proposed_weights), s) for s in s_samples])

        alpha = min(1, np.exp(beta * (reward_proposed - reward_current)))
        if np.random.rand() < alpha:
            weights = proposed_weights.clone()
            accepted = True
        else:
            accepted = False
        accepted_list.append(accepted)

        s_eval = np.random.rand(M)
        avg_eval_reward = np.mean([compute_reward(get_actions([s, s], weights), s) for s in s_eval])

        weights_trace.append(weights.cpu().numpy().copy())
        eval_rewards.append(avg_eval_reward)

        if step % 100 == 0:
            acceptance_rate = np.mean(accepted_list[-100:]) if len(accepted_list) >= 100 else np.mean(accepted_list)
            print(f"Step {step}, Eval Reward: {avg_eval_reward:.3f}, Acceptance Rate: {acceptance_rate:.3f}")

    final_eval_reward = eval_rewards[-1]
    print(f"Final Eval Reward: {final_eval_reward:.3f}")

    distilled_weights = distill_models(weights_old, previous_reward, weights, final_eval_reward)
    save_model(distilled_weights)

    history_entry = {
        "timestamp": datetime.datetime.now().isoformat(),
        "final_eval_reward": final_eval_reward,
        "acceptance_rate": np.mean(accepted_list),
        "steps": n_mcmc_steps,
        "best_reward": max(eval_rewards),
        "worst_reward": min(eval_rewards)
    }
    save_history(history_entry)

    plt.figure(figsize=(10, 6))
    plt.plot(eval_rewards, label='Eval Reward')
    plt.axhline(0.5, color='r', linestyle='--', label='Random Baseline (0.5)')
    plt.axhline(1.0, color='g', linestyle='--', label='Optimal (1.0)')
    plt.xlabel('MCMC Steps')
    plt.ylabel('Avg Reward')
    plt.title('Eval Reward over MCMC Steps')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    main()
