# Quantum Multi-Agent Reinforcement Learning (QMARL) with MCMC

This project provides a robust Python implementation of a Quantum Multi-Agent Reinforcement Learning (QMARL) framework utilizing Markov Chain Monte Carlo (MCMC) optimization, quantum circuits via PennyLane, and PyTorch for GPU acceleration.

---

## Overview

The program leverages quantum-enhanced reinforcement learning to optimize policy decisions of multiple agents. It employs an MCMC-based optimization strategy to iteratively improve the quantum model parameters, achieving high policy performance.

Key features:

- **Quantum Circuits:** Uses PennyLane for creating and executing quantum circuits, enhancing classical reinforcement learning.
- **MCMC Optimization:** Implements Metropolis-Hastings algorithm for robust optimization.
- **Model Distillation:** Combines previous and newly trained models using reward-weighted distillation.
- **Adaptive Training:** Dynamically adjusts exploration parameters for efficient convergence.
- **Persistent Memory:** Maintains training history and model checkpoints, enabling incremental improvements.
- **GPU Acceleration:** Automatically utilizes CUDA-enabled GPUs if available for faster computations.

---

## Dependencies

Ensure the following Python libraries are installed:

```bash
pip install pennylane torch matplotlib numpy
```

- Python >= 3.8
- PyTorch (CUDA-enabled installation recommended for GPU use)
- PennyLane
- Matplotlib
- NumPy

---

## Quick Start

1. Clone or download the repository:

```bash
git clone https://github.com/yourusername/qmarl-mcmc.git
cd qmarl-mcmc
```

2. Run the training script:

```bash
python qmarl_mcmc_gpu.py
```

The program automatically detects and uses GPU hardware if present.

---

## Usage and Customization

Adjust hyperparameters and training settings at the top of the `qmarl_mcmc_gpu.py` script:

- **Agents (`n_gs`)**: Number of agents involved.
- **MCMC steps (`n_mcmc_steps`)**: Adjust for more thorough or quicker training.
- **Exploration parameters (`sigma_proposal`, `beta`)**: Adjust initial exploration and selection pressures.

---

## Outputs

- **Model Weights:** Best model parameters saved to `best_model_weights.pt`.
- **Training History:** JSON file (`training_history.json`) detailing training metrics.
- **Plots:** Visual outputs showing reward evolution over MCMC steps.

---

## License

This project is licensed under my craziness. See the `LICENSE` file for details.

---

## Contributing

Contributions are welcome! Please submit pull requests or raise issues for discussion.

---

## Acknowledgements

- Built using [PennyLane](https://pennylane.ai/) and [PyTorch](https://pytorch.org/).
- Inspired by recent advancements in quantum reinforcement learning and MCMC methodologies.
