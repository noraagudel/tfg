import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from scipy.stats import binom

# Inicializamos el simulador cuántico local. 
# Esto crea un "ordenador cuántico virtual" en tu PC para ejecutar el circuito.
simulator = AerSimulator()


def compute_threshold(s_simulacion, 
                      noise_rate, 
                      alpha):
    
    """
    Calcula el umbral T de errores tolerados para un nivel de falsos positivos (alpha).
    """
    if noise_rate == 0.0 or alpha == 0.0:
        return 0
    # ppf devuelve el valor máximo T tal que la probabilidad acumulada sea <= 1 - alpha
    return int(binom.ppf(1 - alpha, s_simulacion, noise_rate))


def compute_detection_metrics(eve_present,
                              eve_detected):
    """
    Calcula las métricas de evaluación (Confusion Matrix)
    """
    return {
        'TP': int(eve_present and eve_detected),  # Verdadero Positivo
        'FP': int(not eve_present and eve_detected),  # Falso Positivo
        'TN': int(not eve_present and not eve_detected),  # Verdadero Negativo
        'FN': int(eve_present and not eve_detected)  # Falso Negativo
    }


def simulate_bb84_iteration(
        num_qubits,
        intercept_prob,
        noise_rate,
        check_fraction,
        alpha):
    """
    Simula una iteración completa del protocolo BB84.
    """
    
    # QuantumCircuit es el lienzo donde dibujamos nuestro algoritmo cuántico.
    # El primer número (num_qubits) es la cantidad de Qubits (cables cuánticos).
    # El segundo número (num_qubits * 2) es la cantidad de Bits clásicos (cables normales) 
    # donde guardaremos los resultados de las mediciones (la mitad para Bob y la mitad para Eve).
    qc = QuantumCircuit(num_qubits, num_qubits * 2)
    
    # ---------------------------------------------------------
    # 1. ALICE PREPARA LOS QUBITS
    # ---------------------------------------------------------
    alice_bits = np.random.randint(2, size=num_qubits)  # Mensaje secreto (0s y 1s)
    alice_bases = np.random.randint(2, size=num_qubits) # Bases: 0 es base Rectilínea (Z), 1 es Diagonal (X)
    
    for i in range(num_qubits):
        # Todos los qubits en Qiskit nacen por defecto en el estado |0>.
        
        # Si Alice quiere enviar un 1, aplica la puerta X.
        # qc.x(i) es el equivalente cuántico a la puerta clásica NOT. Cambia |0> a |1>.
        if alice_bits[i] == 1:
            qc.x(i)
            
        # Si Alice elige la base diagonal (1), aplica la puerta Hadamard (H).
        # qc.h(i) pone el qubit en superposición. 
        # Físicamente, esto gira el qubit para que se lea correctamente solo en la base diagonal.
        if alice_bases[i] == 1:
            qc.h(i)

    # qc.barrier() es una herramienta visual y de estructura. 
    # Evita que Qiskit mezcle o simplifique operaciones antes y después de esta línea.
    # Funciona como un muro que separa las distintas fases del protocolo.
    qc.barrier()
    
    # ---------------------------------------------------------
    # 2. EVE INTERCEPTA Y REENVÍA
    # ---------------------------------------------------------
    eve_bases = np.random.randint(2, size=num_qubits)
    eve_intercepted = np.random.rand(num_qubits) < intercept_prob 
    eve_present = np.any(eve_intercepted) # Para métricas de evaluación
    for i in range(num_qubits):
        if eve_intercepted[i]:
            # Si Eve adivina que es base diagonal, aplica H para "alinear" su equipo de medición.
            if eve_bases[i] == 1:
                qc.h(i)
                
            # qc.measure(qubit, bit_clasico) es la operación de MEDIR.
            # Esto destruye la superposición (colapsa el qubit a un 0 o 1 definitivo).
            # Aquí le decimos: "Mide el qubit 'i' y guarda el resultado clásico en el bit 'num_qubits + i'".
            qc.measure(i, num_qubits + i) 
            
            # Eve tiene que reenviar el qubit a Bob. Si ella midió en la base diagonal,
            # vuelve a aplicar H para dejar el qubit en el estado que ella cree que es correcto.
            if eve_bases[i] == 1:
                qc.h(i)
            
    # ---------------------------------------------------------
    # 3. RUIDO DEL CANAL CUÁNTICO
    # ---------------------------------------------------------
    for i in range(num_qubits):
        # Simulamos que el ambiente o el cable de fibra óptica alteran los qubits al azar.
        if np.random.rand() < noise_rate:
            qc.x(i)  # Error de Bit-Flip: cambia 0 por 1 o viceversa (afecta la base rectilínea).
        if np.random.rand() < noise_rate:
            qc.z(i)  # Error de Phase-Flip: altera la fase (afecta la base diagonal).

    qc.barrier()  # Fin del viaje por la fibra óptica.

    # ---------------------------------------------------------
    # 4. BOB RECIBE Y MIDE
    # ---------------------------------------------------------
    bob_bases = np.random.randint(2, size=num_qubits)
    
    for i in range(num_qubits):
        # Si Bob elige medir en base diagonal, aplica H primero.
        if bob_bases[i] == 1: 
            qc.h(i)
            
        # Bob mide y guarda sus resultados en la primera mitad de los bits clásicos (índices del 0 al num_qubits-1).
        qc.measure(i, i)
        
    # ---------------------------------------------------------
    # EJECUCIÓN DEL CIRCUITO EN EL SIMULADOR
    # ---------------------------------------------------------
    # simulator.run() compila y lanza el circuito.
    # shots=1 significa que ejecutamos el experimento una sola vez (un solo envío de la cadena de fotones).
    result = simulator.run(qc, shots=1).result()
    
    # get_counts() nos da un diccionario con los resultados, por ejemplo: {'10100 01101': 1}
    # La clave del diccionario es una cadena de texto con los ceros y unos que se midieron.
    counts = result.get_counts()
    measured_bits_str = list(counts.keys())[0]
    
    # IMPORTANTE: Qiskit ordena los bits de derecha a izquierda (Little-Endian).
    # Es decir, el bit 0 está a la derecha del todo.
    # Con [::-1] invertimos la cadena para leerla de izquierda a derecha de forma natural.
    # Como los bits de Bob los guardamos en los índices 0 a num_qubits-1, al invertir la cadena 
    # quedan al principio. Con [:num_qubits] cortamos exactamente la parte de Bob.
    bob_bits_str = measured_bits_str[::-1][:num_qubits]
    
    # Convertimos la cadena de texto de Bob en un array de números [0, 1, 1, 0...]
    bob_bits = np.array([int(b) for b in bob_bits_str])
    
    # ---------------------------------------------------------
    # 5. POST-PROCESADO (SIFTING)
    # ---------------------------------------------------------
    # Alice y Bob se llaman por teléfono y comparten qué bases usaron (pero NO los resultados).
    matching_bases = (alice_bases == bob_bases) 
    
    # Se quedan únicamente con los bits donde sus bases coincidieron.
    sifted_alice = alice_bits[matching_bases]
    sifted_bob = bob_bits[matching_bases]
    
    # Si por mala suerte no coincidieron en ninguna base, abortan.
    if len(sifted_alice) == 0:
        return None
        
    # ---------------------------------------------------------
    # 6. COMPROBACIÓN DE ESPIONAJE
    # ---------------------------------------------------------
    # Toman una fracción de la clave útil para compararla en público y ver si hay errores.
    s_simulacion = max(1, int(len(sifted_alice) * check_fraction))
    check_indices = np.random.choice(len(sifted_alice), s_simulacion, replace=False)

    check_alice = sifted_alice[check_indices]
    check_bob = sifted_bob[check_indices]
    
    # Cuentan cuántos bits son diferentes entre Alice y Bob en esa muestra.
    errors = np.sum(check_alice != check_bob)
    error_rate = errors / s_simulacion
    
    # Cálculo del Umbral Estadístico (T) y métricas
    T = compute_threshold(s_simulacion, noise_rate, alpha)
    eve_detected = errors > T
    metrics = compute_detection_metrics(eve_present, eve_detected)
    
    return {
        'eve_present': np.any(eve_intercepted),
        'eve_detected': eve_detected,
        'metrics': metrics,
        'sifted_length': len(sifted_alice),
        's_simulacion': s_simulacion,
        'threshold_T': T,
        'errors_count': errors
    }