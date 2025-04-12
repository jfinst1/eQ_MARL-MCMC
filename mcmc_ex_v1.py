import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

# Placeholder Data (Replace with Real POS Data)
# Format: n_ijt (product incidence), N_jt (total transactions), for 1 product, 10 stores, 450 days
n_stores = 10
n_days = 450
np.random.seed(42)
data = {
    'store': np.repeat(range(n_stores), n_days),
    'day': np.tile(range(n_days), n_stores),
    'n_ijt': np.random.binomial(n=5000, p=0.01, size=n_stores * n_days),  # Placeholder incidence
    'N_jt': np.random.randint(4000, 6000, size=n_stores * n_days)  # Placeholder total transactions
}
df = pd.DataFrame(data)

# HMM Model Parameters
n_states = 3  # OOS, Low, High
epsilon = 1e-5  # Near-zero probability for OOS state (p. 8)

# Hierarchical Bayesian HMM
with pm.Model() as model:
    # 1. Hierarchical Parameters for Store-Level Heterogeneity (p. 7)
    # Transition matrix intercepts (tau_ijs for ordered logit, p. 6)
    tau_mean = pm.Normal('tau_mean', mu=0, sd=1, shape=(n_states, n_states-1))  # Mean across stores
    tau_sd = pm.HalfNormal('tau_sd', sd=1, shape=(n_states, n_states-1))  # SD across stores
    tau_ijs = pm.Normal('tau_ijs', mu=tau_mean, sd=tau_sd, 
                        shape=(n_stores, n_states, n_states-1))  # Store-specific

    # State-specific incidence probabilities (alpha_ijs, p. 7)
    alpha_mean = pm.Normal('alpha_mean', mu=0, sd=1, shape=n_states-1)  # Mean for Low, High
    alpha_sd = pm.HalfNormal('alpha_sd', sd=1, shape=n_states-1)  # SD for Low, High
    alpha_ijs_raw = pm.Normal('alpha_ijs_raw', mu=alpha_mean, sd=alpha_sd, 
                              shape=(n_stores, n_states-1))  # Store-specific raw

    # Ensure non-decreasing alphas to avoid label-switching (p. 7)
    alpha_ijs = pm.Deterministic('alpha_ijs', tt.sort(alpha_ijs_raw, axis=-1))

    # 2. Transition Matrix (Ordered Logit, p. 6)
    def compute_transition_probs(tau):
        """Convert tau to transition probabilities using ordered logit."""
        probs = []
        for s in range(n_states):
            logits = [tau[s, 0]]
            for k in range(1, n_states-1):
                logits.append(tau[s, k] + tt.exp(tau[s, k-1]))  # Ensure ordered thresholds
            logits = tt.stack(logits)
            exp_logits = tt.exp(logits)
            p_s1 = exp_logits[0] / (1 + exp_logits[0])
            p_s2 = exp_logits[1] / (1 + exp_logits[1]) - p_s1
            p_s3 = 1 - exp_logits[1] / (1 + exp_logits[1])
            probs.append([p_s1, p_s2, p_s3])
        return tt.stack(probs)

    transition_probs = pm.Deterministic('transition_probs', 
                                      tt.stack([compute_transition_probs(tau_ijs[j]) 
                                                for j in range(n_stores)]))

    # 3. State-Specific Incidence Probabilities (p. 7)
    def compute_incidence_probs(alpha, store_idx):
        """Convert alpha to incidence probabilities, with OOS constrained."""
        p_0 = epsilon  # Fixed OOS probability (p. 7)
        p_1 = 1 / (1 + tt.exp(-alpha[store_idx, 0]))  # Low demand
        p_2 = 1 / (1 + tt.exp(-alpha[store_idx, 1]))  # High demand
        return tt.stack([p_0, p_1, p_2])

    # 4. HMM States (Latent Variables)
    states = []
    for j in range(n_stores):
        # Initial state probabilities (uniform, p. 6)
        init_probs = pm.Dirichlet(f'init_probs_{j}', a=np.ones(n_states))
        
        # State sequence per store
        state_seq = pm.HiddenMarkov(f'states_{j}', 
                                   P=transition_probs[j], 
                                   pi=init_probs, 
                                   shape=n_days)
        states.append(state_seq)

    # 5. Observation Model (Binomial, p. 7)
    for j in range(n_stores):
        store_data = df[df['store'] == j]
        n_ijt = store_data['n_ijt'].values
        N_jt = store_data['N_jt'].values
        
        # Incidence probabilities for store j
        p_ijt = compute_incidence_probs(alpha_ijs, j)
        
        # Binomial likelihood
        pm.Binomial(f'obs_{j}', 
                   n=N_jt, 
                   p=p_ijt[states[j]], 
                   observed=n_ijt)

    # 6. MCMC Sampling
    trace = pm.sample(1000, tune=1000, target_accept=0.9, return_inferencedata=False)

# Summarize Results
print("MCMC Summary:")
print(pm.summary(trace, var_names=['tau_mean', 'alpha_mean']))

# Plot Trace for Diagnostics
pm.traceplot(trace, var_names=['tau_mean', 'alpha_mean'])
plt.show()

# Predict States (Example for Store 0)
with model:
    post_states = pm.sample_posterior_predictive(trace, var_names=['states_0'])
predicted_states = np.argmax(np.mean(post_states['states_0'], axis=0), axis=1)

# Visualize Predicted States
plt.figure(figsize=(12, 6))
plt.plot(predicted_states, label='Predicted States')
plt.title('Predicted HMM States for Store 0')
plt.xlabel('Day')
plt.ylabel('State (0: OOS, 1: Low, 2: High)')
plt.legend()
plt.show()