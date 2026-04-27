import numpy as np
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from bb84_simulator import simulate_bb84_iteration
simulator = AerSimulator()

# =====================================================================
# 2. LAS FUNCIONES DE EXPERIMENTO
# =====================================================================


def experiment_variable_R(n_fixed,
                          R_values,
                          numero_ensayos,
                          intercept_prob,
                          noise_rate,
                          threshold):
    
    """
    MODO 1
    Aquí se simula el caso donde el número de qubits es fijo (n_fixed) y lo que varía es el número de iteraciones R.
     - n_fixed: Número fijo de qubits a usar en cada iteración
     - R_values: Lista de valores de R (número de iteraciones) a probar
     - numero_ensayos: Número de veces que se repite el experimento para cada R (para hacer estadística)
     - intercept_prob: Porcentaje de intercepción de Eve (p)
     - noise_rate: Tasa de ruido ambiental
     - threshold: Umbral para decidir si detectamos a Eve o no
     El experimento calcula tanto la probabilidad empírica (basada en las simulaciones) como la teórica
     Finalmente, se grafica la probabilidad de detección contra R para ambos casos.
    """
    print(f"--- Exp: R Variable | n={n_fixed}, Ruido={noise_rate}, Eve={intercept_prob} ---")
    check_fraction = 0.5
    emp_probs = []
    teo_probs = []
    
    for R in R_values:
        detected_count = 0

        # Simulamos el experimento numero_ensayos veces para este valor de R
        for _ in range(numero_ensayos):
            caught = False
            for _ in range(R):
                res = simulate_bb84_iteration(n_fixed,
                                              intercept_prob,
                                              noise_rate,
                                              check_fraction,
                                              threshold)
                if res and res['eve_detected']:
                    caught = True
                    break  # Si detectamos a Eve en alguna iteración, ya contamos como detectada para este ensayo
            if caught:
                detected_count += 1
            
        p_emp = detected_count / numero_ensayos
        emp_probs.append(p_emp * 100)
        
        # CÁLCULO TEÓRICO (Ojo: Ajustado por si p < 1)
        s_promedio = (n_fixed * 0.5) * check_fraction
        prob_error_por_bit = intercept_prob * 0.25  # Si Eve intercepta menos, hay menos error
        p_teo = 1 - (1 - prob_error_por_bit)**(R * s_promedio)
        teo_probs.append(p_teo * 100)
        
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


def experiment_variable_n(qubit_counts,
                          numero_ensayos,
                          intercept_prob,
                          noise_rate,
                          threshold):
    """
    MODO 2
    Aquí se simula el caso donde el número de iteraciones es fijo (iterations) y lo que varía es el número de qubits.
     - qubit_counts: Lista de diferentes números de qubits a probar
     - iterations: Número fijo de iteraciones a realizar para cada cantidad de qubits
     - intercept_prob: Porcentaje de intercepción de Eve (p)
     - noise_rate: Tasa de ruido ambiental
     - threshold: Umbral para decidir si detectamos a Eve o no
     El experimento calcula tanto la probabilidad empírica (basada en las simulaciones) como la teórica
     Finalmente, se grafica la probabilidad de detección contra el número de qubits para ambos casos.
    """
    print(f"--- Exp: n Variable | Iter={numero_ensayos}, Ruido={noise_rate}, Eve={intercept_prob} ---")
    
    check_fraction = 0.5
    emp_probs = []
    teo_probs = [0.0] * len(qubit_counts)

    for n in qubit_counts:
        detected_count = 0
        valid_runs = 0
        
        # Suma las probabilidades teóricas exactas de cada ronda
        suma_p_teo_exacta = 0.0 
        
        for _ in range(numero_ensayos):
            res = simulate_bb84_iteration(n,
                                          intercept_prob,
                                          noise_rate,
                                          check_fraction,
                                          threshold)
            if res:
                valid_runs += 1
                if res['eve_detected']:
                    detected_count += 1
                    
                # Calculamos la probabilidad teórica para el 's' EXACTO de esta ronda
                s_de_esta_ronda = res['s_simulacion']
                print(f"Ronda con n={n} qubits, s={s_de_esta_ronda} bits para comprobación.")
                prob_error_por_bit = intercept_prob * 0.25
                suma_p_teo_exacta += 1 - (1 - prob_error_por_bit)**(s_de_esta_ronda)
                
        if valid_runs > 0:
            p_emp = detected_count / valid_runs
        else:
            p_emp = 0.0
        emp_probs.append(p_emp * 100)
        
        # El valor teórico ahora es el promedio de las probabilidades, no la probabilidad del promedio.
        if valid_runs > 0:
            p_teo = suma_p_teo_exacta / valid_runs
        else:
            p_teo = 0.0
        teo_probs.append(p_teo * 100)
        
        print(f"Qubits (n): {n:3} | Empírica: {p_emp*100:6.2f}% | Teórica: {p_teo*100:6.2f}%")

    # Gráfica Modo 2
    plt.figure(figsize=(10, 6))
    plt.plot(qubit_counts, emp_probs, marker='o', label='Probabilidad Empírica (1 iteración)')
    plt.plot(qubit_counts, teo_probs, marker='s', linestyle='--', label='Probabilidad Teórica ($1 - (3/4)^s$)')
    plt.title(f'Detección de Eve en una ejecución vs Número de Qubits (Evaluado sobre {numero_ensayos} muestras)')
    plt.xlabel('Número de Qubits enviados (n)')
    plt.ylabel('Probabilidad de Detección (%)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.savefig('prob_n_variable.png', dpi=300, bbox_inches='tight')
    plt.show()


def experiment_variable_p_and_n(qubit_counts,
                                numero_ensayos,
                                p_values,
                                noise_rate,
                                threshold):
    """
    CASO C (0% Ruido, Eve variable (p))
     - qubit_counts: Lista de diferentes números de qubits a probar
     - numero_ensayos: Número de veces que se repite el experimento para cada combinación de p y n (para hacer estadística)
     - p_values: Lista de diferentes valores de p (fracción de qubits interceptados por Eve) a probar
     - noise_rate: Tasa de ruido ambiental (en este caso, 0.0)
     - threshold: Umbral para decidir si detectamos a Eve o no (en este caso, 0.0)
        Este experimento es una combinación de los anteriores, pero con la particularidad de que ahora
        se evalúa la probabilidad de detección variando la agresividad de Eve (p) para distintos tamaños de bloques de qubits (n).
     El experimento calcula tanto la probabilidad empírica (basada en las simulaciones) como la teórica para cada combinación de p y n.
     Finalmente, se grafica la probabilidad de detección contra p para cada valor de n, mostrando tanto la curva empírica como la teórica 
     en la misma gráfica para facilitar la comparación.
   
    """
    print(f"--- Exp: Eve Variable (p) y Qubits Variables (n) | Iter={numero_ensayos}, Ruido={noise_rate} ---")
    check_fraction = 0.5
    
    # Preparamos la figura de Matplotlib
    plt.figure(figsize=(12, 8))
    
    # Paleta de colores para diferenciar cada valor de 'n'
    colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']
    
    for idx, n in enumerate(qubit_counts):
        print(f"Simulando para n = {n}...")
        emp_probs = []
        teo_probs = []
        
        # Color asignado a este tamaño de n
        color_actual = colores[idx % len(colores)]
        
        for p in p_values:
            detected_count = 0
            valid_runs = 0
            suma_p_teo_exacta = 0.0 
            
            prob_error_por_bit = p * 0.25
            
            for _ in range(numero_ensayos):
                res = simulate_bb84_iteration(n,
                                              p,
                                              noise_rate,
                                              check_fraction,
                                              threshold)
                
                if res:
                    valid_runs += 1
                    if res['eve_detected']:
                        detected_count += 1
                        
                    # CÁLCULO TEÓRICO EXACTO (Evaluado por ronda)
                    s_de_esta_ronda = res['s_simulacion']
                    suma_p_teo_exacta += 1 - (1 - prob_error_por_bit)**(s_de_esta_ronda)
                        
            # Promedio empírico
            if valid_runs > 0:
                p_emp = detected_count / valid_runs
            else:
                p_emp = 0.0
            emp_probs.append(p_emp * 100)
            
            # Promedio teórico (promedio de las probabilidades)
            if valid_runs > 0:
                p_teo = suma_p_teo_exacta / valid_runs
            else:
                p_teo = 0.0
            teo_probs.append(p_teo * 100)
            
        # Dibujamos las líneas en la gráfica para el 'n' actual
        # Línea empírica (sólida con marcadores)
        plt.plot(p_values, emp_probs, marker='o', linestyle='-', color=color_actual, 
                 label=f'Empírica (n={n})')
        # Línea teórica (punteada, un poco transparente para no saturar)
        plt.plot(p_values, teo_probs, marker='', linestyle='--', color=color_actual, alpha=0.6, 
                 label=f'Teórica (n={n})')

    # Configuración estética de la gráfica
    plt.title('Probabilidad de Detección vs Tasa de Intercepción de Eve (p)\n(0% Ruido Ambiental)', fontsize=14)
    plt.xlabel('Fracción de qubits interceptados por Eve (p)', fontsize=12)
    plt.ylabel('Probabilidad de detectar a Eve (%)', fontsize=12)
    
    # Eje X en formato porcentaje visual
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 105, 10))
    
    plt.grid(True, linestyle=':', alpha=0.7)
    
    # Movemos la leyenda fuera de la gráfica para que no tape las líneas
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0.)
    
    plt.tight_layout()
    plt.savefig('prob_deteccion_p_y_n.png', dpi=300, bbox_inches='tight')
    plt.show()


# =====================================================================
# 3. EL ORQUESTADOR (Donde decides qué estudiar hoy)
# =====================================================================


if __name__ == "__main__":
    
    CASO_A_ESTUDIAR = "A"  # Cambia esto a "B" o "C"
    
    if CASO_A_ESTUDIAR == "A":
        # CASO A: 0% Ruido, 100% Intercepción
        noise = 0.0
        p_eve = 1.0
        umbral = 0.0  # Cualquier error es Eve
        
        # A.1: R variable
        experiment_variable_R(n_fixed=16,
                              R_values=[1, 2, 3, 5, 8, 12, 20],
                              numero_ensayos=1000,
                              intercept_prob=p_eve,
                              noise_rate=noise,
                              threshold=umbral)
        
        # A.2: n variable
        experiment_variable_n(qubit_counts=[1, 4, 8, 16, 32, 64, 128],
                              numero_ensayos=1000,
                              intercept_prob=p_eve,
                              noise_rate=noise,
                              threshold=umbral)

    elif CASO_A_ESTUDIAR == "B":
        # CASO B: 2% Ruido, 100% Intercepción
        noise = 0.02
        p_eve = 1.0
        # ¡IMPORTANTE! El umbral ya no puede ser 0.0
        umbral = 0.05
        
        experiment_variable_R(n_fixed=16,
                              R_values=[1, 2, 3, 5, 8, 12, 20],
                              numero_ensayos=1000,
                              intercept_prob=p_eve,
                              noise_rate=noise,
                              threshold=umbral)
        experiment_variable_n(qubit_counts=[1, 4, 8, 16, 32, 64, 128],
                              numero_ensayos=1000,
                              intercept_prob=p_eve,
                              noise_rate=noise,
                              threshold=umbral)

    elif CASO_A_ESTUDIAR == "C":
        # CASO C: 0% Ruido, Eve variable (p)
        noise = 0.0
        umbral = 0.0
        
        qubit_counts_lista = [1, 6, 8, 12, 20]
        p_valores_lista = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    
        experiment_variable_p_and_n(
            qubit_counts=qubit_counts_lista,
            numero_ensayos=1000,
            p_values=p_valores_lista,
            noise_rate=0.0,
            threshold=0.0
        )