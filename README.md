# BB84 Quantum Key Distribution Simulation 

## Overview
This repository contains a comprehensive computational simulation of the **BB84 Quantum Key Distribution (QKD) protocol**. Developed as part of a Bachelor's Thesis in Physics at the Universitat de Barcelona, this project models the secure generation and exchange of cryptographic keys using quantum mechanics principles.

The simulation goes beyond ideal scenarios by introducing realistic constraints: **environmental channel noise** (bit-flip and phase-flip errors) and the presence of an **eavesdropper (Eve)**. By leveraging statistical modeling, the system dynamically calculates error thresholds to detect interceptions with high mathematical confidence.

## Key Features
* **End-to-End BB84 Lifecycle:** Simulates quantum state preparation (Alice), interception (Eve), quantum channel degradation, measurement (Bob), sifting, and error checking using the `qiskit_aer` simulator.
* **Dynamic Statistical Thresholding:** Uses inverse binomial distributions (`scipy.stats.binom.ppf`) to dynamically calculate the tolerated error threshold (T) based on the sample size (s), environmental noise (p0), and a strictly defined false-positive tolerance (alpha).
* **Robust Evaluation Metrics:** Evaluates the security of the protocol using Confusion Matrices, True Positive/False Positive rates, and ROC curves to measure detection capabilities under varying configurations.
* **Configurable Experimental Orchestrator:** Allows automated execution of large-scale statistical experiments varying key parameters such as the number of qubits (n), iterations (R), and Eve's interception aggressiveness (p).

## Mathematical Foundation
To determine if Eve is eavesdropping, Alice and Bob compare a fraction of their sifted key. The simulation establishes a detection threshold T such that the probability of rejecting a secure key due to natural noise is strictly bounded by alpha.

If the observed errors exceed T, the protocol assumes Eve's presence and aborts the key generation.

## Project Structure
* `bb84_simulator.py`: The core quantum simulation logic. Handles the Qiskit circuit creation, gate applications (X, H, Z), measurements, and the mathematical sifting/checking phases.
* `grafics.py`: A dedicated visualization module using Matplotlib to render Confusion Matrices, ROC curves, threshold evolutions, and error probability density functions.
* `experiments.py` *(Orchestrator)*: The main entry point to run large-scale Monte Carlo experiments. 

## Installation

1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/bb84-quantum-simulation.git](https://github.com/yourusername/bb84-quantum-simulation.git)
   cd bb84-quantum-simulation

2. Install the required dependencies. It is recommended to use a virtual environment:
   pip install numpy scipy matplotlib qiskit qiskit-aer

## Usage

You can run different experimental scenarios by modifying the CASO_A_ESTUDIAR variable inside the orchestrator file.

Available Experimental Cases:

- CASE A (Ideal Channel, 100% Interception): Tests the theoretical detection limits without environmental noise. Evaluates how the number of qubits (n) and protocol repetitions (R) affect the probability of catching Eve.

- CASE B (Noisy Channel, 100% Interception): Introduces a 2% baseline environmental noise. Generates statistical threshold graphs, error distributions, and ROC curves to evaluate the model's ability to distinguish between natural noise and an active attack.

- CASE C (Variable Eavesdropping): Analyzes the system's sensitivity by varying Eve's interception probability (p) from 0% to 100% across different qubit block sizes.
