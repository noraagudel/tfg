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

prompt per ia
estoy trabajando en un proyecto sobre encriptacion cuantica y protocolo BB84 he estado calculando las probabilidades de acierto y de fallo que hay entre un emisor Alice y un receptor Bob cuando Alice envia un mensaje a Bob y como cambian esas probabilidades si hay un eavesdropper Eve intentando espiar. Debo crear una especie de pequeño programa en python que simule varias veces y con varias cosas cambiadas esta situacion de enviar y recibir en encriptacion cuantica siguiendo el protocolo BB84. Pongamos que todos eligen las bases con las que envian o miden de manera random y que Alice envia 8 bits separados en bases random que haya elegido a Bob que tambien mide en bases random. Por otro lado, tenemos a Eve que puede que este interceptando algunso, todos o ningun bit. Por ultimo, puede ser que se de una situacion donde al comparar las bases que han usado para medir; cuando Bob hace publicas dichas bases para que Alice le diga cuales debe descartar y cuales no, se de el caso que la fraccion de informacion que hagan publica no sea suficiente para determinar realmente si hay un error normal o un error debido a un intento de espionaje y que entonces, descarten que estan siendo espiados cuando en realidad lo estan siendo. Puede ser tambien que se produzan errores en el propio sistema, es decir, que la fibra optica o otro gire los bits de up a down por ejemplo. Hay varios numeros que entran en juego: Numero de qbits totales enviados de Alice a Bob, Numero de errores (sistematicos como el que hemos explicado), Numero de qbits usados como test para comprovar si Alice y Bob estan seguros, Numero de qbits que ha interceptado Eve. Quiero calcular distintas probabilidades dentro de esta misma situacion, por eso el programa debe servir para simular varias veces una misma situacion cambiando solo ciertos parametros o detalles:
- Probabilidad de que la comunicacion sea realmente segura
- Probabilidad que el calculo que se ha hecho para intentar determinar si habia una Eve o no sea correcto
- Probabilidad de detectar a Eve
- Probabilidad de saber distinguir entre ruido o fallos vs Eve
- Probabilidad de que Eve pase desapercibida y pueda robar informacion

El programa debe ser claro, correcto y explicado paso a paso. Tambien debe calcular los graficos de las 3 primeras probabilidades, es decir, que el programa prepare estos graficos y los exporte. Debe detectar la probabilidad (Y) en funcion del numero de qbits que se comparan (x), es decir, empezamos teniendo solo 8 qbits a enviar y de estos 8 puede que ni los 8 sean los totales que se comparan ya que pueden ser descartados, y luego la simulacion se debe hacer con 20 qbits, 50, 80 y 100. Para poder comparar bien asi que las funciones que se creen deben permitir adaptar el parametro de qbits. Se puede usar la libreria Qiskit en python usada para cosas de quantum.

Aqui hay un par de repositorios en github que pueden tener relacion:
https://github.com/Kairos-T/BB84-Simulator
https://github.com/yuvalbloom/Quantum-Cryptography-BB84-protocol