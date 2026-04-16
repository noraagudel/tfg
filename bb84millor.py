import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

# Inicializamos el simulador cuántico local
simulator = AerSimulator()


def simulate_bb84_iteration(num_qubits, intercept_prob, noise_rate, check_fraction, threshold):
    """
    Simula una iteración completa del protocolo BB84.
    """
    # 1 qubit y 1 bit clásico por cada fotón enviado
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # 1. ALICE PREPARA EL MENSAJE
    alice_bits = np.random.randint(2, size=num_qubits)
    alice_bases = np.random.randint(2, size=num_qubits) # 0 = Rectilínea (Z), 1 = Diagonal (X)
    
    # qc.x(i): 
      #  La puerta X es como un “flip” clásico:
      #  transforma |0⟩ en |1⟩
      #  transforma |1⟩ en |0⟩

    for i in range(num_qubits):
        if alice_bits[i] == 1:
            qc.x(i) # Flip a |1>

    # qc.h(i):
      #  La puerta Hadamard H crea superposición y cambia entre bases.
      #  H|0⟩ = |+⟩
      #  H|1⟩ = |−⟩
    
        if alice_bases[i] == 1:
            qc.h(i) # Cambia a base diagonal |+> o |->
            
    qc.barrier()
    
    # 2. EVE INTERCEPTA (Teorema de No Clonación)
    # Eve también elige bases aleatorias y decide qué qubits interceptar.
    eve_bases = np.random.randint(2, size=num_qubits)  # Base que Eve usará para medir cada qubit interceptado.
    eve_intercepted = np.random.rand(num_qubits) < intercept_prob  # Decide aleatoriamente si intercepta cada qubit según la probabilidad dada.
    
    # Este bloque intenta representar la acción de Eve:
    for i in range(num_qubits):
        if eve_intercepted[i]:
            # Si Eve usa base distinta → introduce error con probabilidad 0.5
            if eve_bases[i] != alice_bases[i]:
                if np.random.rand() < 0.5:
                    qc.x(i)
            
    qc.barrier()
    
    # 3. RUIDO DEL SISTEMA (Ej: Errores en fibra óptica)
    for i in range(num_qubits):
        if np.random.rand() < noise_rate:
            qc.x(i) # Bit flip aleatorio simulando ruido
            
    qc.barrier()
    
    # 4. BOB RECIBE Y MIDE
    bob_bases = np.random.randint(2, size=num_qubits)
    # Si Bob elige la base diagonal (X), aplica H antes de medir para cambiar a esa base.
    for i in range(num_qubits):
        if bob_bases[i] == 1: 
            qc.h(i)
        qc.measure(i, i)
        
    # Ejecutamos el circuito
    result = simulator.run(qc, shots=1).result()
    counts = result.get_counts()
    
    # Qiskit devuelve los bits en orden inverso, los ordenamos
    measured_bits_str = list(counts.keys())[0]
    bob_bits = np.array([int(b) for b in measured_bits_str[::-1]])  # Bits medidos por Bob, ordenados correctamente
    
    # 5. POST-PROCESADO (Alice y Bob hablan por un canal público)
    # Solo se quedan con los bits donde eligieron la misma base
    matching_bases = (alice_bases == bob_bases)  # Boolean array indicando dónde coinciden las bases (True si coinciden, False si no)
    # Solo los bits donde coinciden las bases son útiles para la clave final:
    # Sifted_name son los bits filtrados
    sifted_alice = alice_bits[matching_bases]
    sifted_bob = bob_bits[matching_bases]
    
    if len(sifted_alice) == 0:
        return None  # Caso raro en pocos qubits: no coinciden bases
        
    # 6. COMPROBACIÓN DE ESPIONAJE (Sacrifican una fracción de bits)
    # num_check se asegura de que al menos 1 bit se use para la comprobación
    num_check = max(1, int(len(sifted_alice) * check_fraction))  # si check_fraction es 0.5 y sifted_alice tiene 10 bits, num_check será 5
    # Elegimos aleatoriamente qué bits usar para la comprobación
    check_indices = np.random.choice(len(sifted_alice), num_check, replace=False)
    
    # check_alice se calcula tomando los bits de Alice en las posiciones seleccionadas para la comprobación, y lo mismo para Bob.
    check_alice = sifted_alice[check_indices]
    check_bob = sifted_bob[check_indices]
    
    # check_alice != check_bob es un array booleano que indica dónde difieren los bits de Alice y Bob en la muestra de comprobación.
    # Calculan el error: np.sum(check_alice != check_bob) cuenta cuántos bits difieren, y luego se divide por num_check para obtener la tasa de error.
    errors = np.sum(check_alice != check_bob)
    error_rate = errors / num_check
    
    # Deciden si Eve está espiando: si la tasa de error es significativamente mayor que el ruido esperado, asumen que hay espionaje.
    eve_detected = error_rate > threshold
    
    return {
        'eve_present': np.any(eve_intercepted),  # eve_present: True si Eve interceptó al menos un qubit.
        'eve_detected': eve_detected,  # eve_detected: True si la tasa de error supera el umbral, indicando posible espionaje.
        'error_rate': error_rate  # error_rate: La tasa de error calculada a partir de la muestra de comprobación.
    }


# Esta función repite la simulación para varios números de qubits y luego grafica el resultado.
def run_experiment():
    """
    Corre experimentos variando el número de qubits para calcular y graficar probabilidades.
    """
    qubit_counts = [8, 20, 50, 80, 100]
    iterations = 2000  # Simulamos 2000 veces cada caso para sacar porcentajes fiables
    
    # Parámetros fijos para este experimento
    noise_rate = 0.05       # 5% de error por hardware
    intercept_prob = 0.5    # Eve intercepta el 100% de los qubits (para ver detección máxima)
    check_fraction = 0.5    # Se usa la mitad de la clave sifted para comprobar espionaje.
    threshold = 0.12        # Si el error pasa del 12% (superior al ruido del 5%), asumen espionaje
    
    detection_probabilities = []  # Aquí guardaremos la probabilidad de detección para cada número de qubits
    undetection_probabilities = []  # Aquí guardaremos la probabilidad que no se detecte a Eve
    print("Iniciando simulación...")
    print("-" * 50)
    
    for n in qubit_counts:
        valid_runs = 0
        eve_present_runs = 0
        eve_detected_runs = 0
        eve_undetected_runs = 0

        for _ in range(iterations):
            result = simulate_bb84_iteration(
                num_qubits=n,
                intercept_prob=intercept_prob,
                noise_rate=noise_rate,
                check_fraction=check_fraction,
                threshold=threshold
            )

            if result is not None:
                valid_runs += 1

                if result['eve_present']:
                    eve_present_runs += 1
                    if result['eve_detected']:
                        eve_detected_runs += 1
                    else:
                        eve_undetected_runs += 1

        prob_detect = (eve_detected_runs / eve_present_runs) * 100 if eve_present_runs > 0 else 0
        prob_undetect = (eve_undetected_runs / eve_present_runs) * 100 if eve_present_runs > 0 else 0

        detection_probabilities.append(prob_detect)
        undetection_probabilities.append(prob_undetect)

        print(f"Qubits enviados: {n:3} , P(detectada|Eve present): {prob_detect:6.2f}%")
        print(f"Qubits enviados: {n:3} , P(NO detectada|Eve present): {prob_undetect:6.2f}%")
    
    # Generación y exportación del gráfico
    plt.figure(figsize=(10, 6))
    plt.plot(qubit_counts, detection_probabilities, marker='o', linestyle='-', label='Detectada')
    plt.plot(qubit_counts, undetection_probabilities, marker='s', linestyle='--', label='No detectada')
    plt.title('Detecció vs No detecció de Eve')
    plt.xlabel('Qubits inicials enviats')
    plt.ylabel('Probabilitat estant Eve present (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 105)
    plt.legend()
    plt.savefig('probabilitat_Eve_perd_guanya_Eveintercepta50%qubits.png', dpi=300, bbox_inches='tight')
    print("-" * 50)
    print("Simulación terminada. Gráfico exportado como 'probabilitat_Eve_perd_guanya_Eveintercepta50%qubits.png'")
    plt.show()


if __name__ == "__main__":
    run_experiment()