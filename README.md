# tfg

Workflow regarding my Bachelor's Thesis on the No-Cloning Theorem and its implications in quantum cryptography 

# several Python-based quantum programming frameworks that are standard for creating simulations like the one planned for this project:

• Qiskit (IBM): Mentioned as a primary tool for quantum programming.
• Cirq (Google): Another key Python framework for quantum circuits.
• libquantum: A library specifically for quantum simulation.

# Logic for  Alice, Bob, and Eve Code

# Operational logic which is needed to turn into Python code:

• The BB84 Protocol: This is the standard method for secure communication between Alice and Bob. It involves Alice choosing a random basis (rectilinear or diagonal) to send photons, and Bob choosing a random basis to measure them.

• Simulating Eve (The No-Cloning Theorem): The code can represent Eve’s presence based on the fact that she cannot copy quantum data with perfect fidelity due to the no-cloning theorem. If Eve attempts to read the data, the quantum state will change due to wave function collapse, which Alice and Bob can detect as discrepancies or noise.

• Checking for Noise: The sources explain that Alice and Bob can prove a message was not wiretapped by checking for these discrepancies. In your code, the difference between the "spying" and "not spying" scenarios would be the error rate Bob sees in his measurement table when compared to Alice's original sequence.

# Simulation Parameters (Noise and Detection)

# To make the simulation more realistic, we can incorporate these "noise" factors mentioned in the source were we are getting inspired by:

• Detector Efficiency Mismatch: Noise could be simulated by adding a parameter for differences in photodetector efficiency, which can sometimes allow an eavesdropper to hide their presence.

• Multi-photon Sources: Real-world systems often use faint lasers rather than single photons, which allows for "photon splitting attacks" that could be simulated as a specific scenario.

• Information-Theoretic Security: We can use the code to show that while QKD is secure against a spy with unlimited computing power, practical "noise" in the hardware (like holes in the measurement table from lost qubits) can affect the ability to verify if a sequence was intercepted.