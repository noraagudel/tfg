import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# Inicializamos el simulador cuántico local
simulator = AerSimulator()


def simulate_bb84_iteration(
        num_qubits,
        intercept_prob,
        noise_rate,
        check_fraction,
        threshold):
    """
    Simula una iteración completa del protocolo BB84.
    """
    qc = QuantumCircuit(num_qubits, num_qubits)
    
    # 1. ALICE PREPARA
    alice_bits = np.random.randint(2, size=num_qubits)
    alice_bases = np.random.randint(2, size=num_qubits)
    # qc.x(i): 
      #  La puerta X es como un “flip” clásico:
      #  transforma |0⟩ en |1⟩
      #  transforma |1⟩ en |0⟩
        for i in range(num_qubits):
            if alice_bits[i] == 1:
                qc.x(i)
    # qc.h(i):
      #  La puerta Hadamard H crea superposición y cambia entre bases.
      #  H|0⟩ = |+⟩
      #  H|1⟩ = |−⟩
        if alice_bases[i] == 1:
            qc.h(i)

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
        'sifted_length': len(sifted_alice),
        'num_check': num_check,
        'error_rate': error_rate  # error_rate: La tasa de error calculada a partir de la muestra de comprobación.
    }


def run_experiment(fixed_qubits_mode=True):
    """
    Ejecuta el experimento dependiendo del modo seleccionado.
    """
    # Parámetros fijos de entorno (Escenario Ideal)
    noise_rate = 0.00
    intercept_prob = 1.0
    check_fraction = 0.5
    # En un entorno ideal (0 ruido), cualquier error > 0 significa que Eve está ahí.
    threshold = 0.00
    
    print("-" * 50)
    
    if fixed_qubits_mode:
        print("MODO 1: Qubits fijos, variando el número de iteraciones R")
        print("Evaluando la probabilidad ACUMULADA de detectar a Eve en múltiples intentos.")
        
        n_fixed = 16  # Número de qubits por iteración (bajo para que Eve pueda escapar 1 vez y veamos la curva)
        R_values = [1, 2, 3, 5, 8, 12, 20]  # Diferentes cantidades de iteraciones acumuladas
        meta_experiments = 1000  # Cuántas veces repetimos el bloque de R iteraciones para sacar la estadística

        emp_probs = []
        teo_probs = []
        
        for R in R_values:
            detected_count = 0
            
            # Simulamos el bloque entero de R iteraciones varias veces
            for _ in range(meta_experiments):
                caught_in_this_block = False
                for _ in range(R):
                    res = simulate_bb84_iteration(n_fixed,
                                                  intercept_prob,
                                                  noise_rate,
                                                  check_fraction,
                                                  threshold)
                    if res and res['eve_detected']:
                        caught_in_this_block = True
                        break  # Si la pillamos una vez en las R iteraciones, ya está detectada
                
                if caught_in_this_block:
                    detected_count += 1
            
            # Cálculo empírico
            p_emp = detected_count / meta_experiments
            emp_probs.append(p_emp * 100)
            
            # Cálculo teórico acumulado: P = 1 - (3/4)^(R * s_promedio)
            s_promedio = n_fixed * 0.5 * check_fraction
            p_teo = 1 - (0.75)**(R * s_promedio)
            teo_probs.append(p_teo * 100)
            
            print(f"Iteraciones (R): {R:2} | Empírica: {p_emp*100:6.2f}% | Teórica: {p_teo*100:6.2f}%")
            
        # Gráfica Modo 1
        plt.figure(figsize=(10, 6))
        plt.plot(R_values, emp_probs, marker='o', label='Probabilidad Empírica Acumulada')
        plt.plot(R_values, teo_probs, marker='s', linestyle='--', label='Probabilidad Teórica ($1 - (3/4)^{Rs}$)')
        plt.title(f'Detección de Eve acumulada tras R iteraciones (n={n_fixed} qubits/iter)')
        plt.xlabel('Número de Iteraciones acumuladas (R)')
        plt.ylabel('Probabilidad de Detección (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig('prob_R_variable.png', dpi=300, bbox_inches='tight')
        plt.show()

    else:
        print("MODO 2: Iteraciones R fijas (como tamaño de muestra), variando los Qubits n")
        print("Evaluando la probabilidad de detectar a Eve en una sola ejecución.")
        
        qubit_counts = [4, 8, 16, 32, 64, 128]
        iterations = 2000  # Tamaño de la muestra estadística
        
        emp_probs = []
        teo_probs = []
        
        for n in qubit_counts:
            detected_count = 0
            valid_runs = 0
            total_s_in_valid_runs = 0
            
            for _ in range(iterations):
                res = simulate_bb84_iteration(n,
                                              intercept_prob,
                                              noise_rate,
                                              check_fraction,
                                              threshold)
                if res:
                    valid_runs += 1
                    total_s_in_valid_runs += res['num_check']
                    if res['eve_detected']:
                        detected_count += 1
            
            p_emp = detected_count / valid_runs if valid_runs > 0 else 0
            emp_probs.append(p_emp * 100)
            
            # Cálculo teórico por iteración: P = 1 - (3/4)^s
            # Usamos el promedio real de 's' que se generó para ser más exactos
            s_promedio_real = total_s_in_valid_runs / valid_runs if valid_runs > 0 else (n * 0.5 * check_fraction)
            p_teo = 1 - (0.75)**(s_promedio_real)
            teo_probs.append(p_teo * 100)
            
            print(f"Qubits (n): {n:3} | Empírica: {p_emp*100:6.2f}% | Teórica: {p_teo*100:6.2f}%")

        # Gráfica Modo 2
        plt.figure(figsize=(10, 6))
        plt.plot(qubit_counts, emp_probs, marker='o', label='Probabilidad Empírica (1 iteración)')
        plt.plot(qubit_counts, teo_probs, marker='s', linestyle='--', label='Probabilidad Teórica ($1 - (3/4)^s$)')
        plt.title(f'Detección de Eve en una ejecución vs Número de Qubits (Evaluado sobre {iterations} muestras)')
        plt.xlabel('Número de Qubits enviados (n)')
        plt.ylabel('Probabilidad de Detección (%)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.savefig('prob_n_variable.png', dpi=300, bbox_inches='tight')
        plt.show()


if __name__ == "__main__":
    # Cambiar esto a False para ver el comportamiento variando los Qubits
    FIXED_QUBITS_MODE = False
    run_experiment(fixed_qubits_mode=FIXED_QUBITS_MODE)