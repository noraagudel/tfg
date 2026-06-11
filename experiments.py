import numpy as np
import matplotlib.pyplot as plt
from qiskit_aer import AerSimulator
from scipy.stats import binom
from bb84_simulator import simulate_bb84_iteration
from bb84_simulator import compute_threshold
from grafics import plot_confusion_matrix, plot_static_threshold, plot_error_distributions, plot_multiple_roc_curves


# Inicializamos el simulador cuántico local.

simulator = AerSimulator()


# =====================================================================
# FUNCIONES AUXILIARES TEÓRICAS 
# =====================================================================


def theoretical_detection_prob(s,
                               p_eve,
                               noise_rate,
                               alpha):
    """
    Calcula la probabilidad teórica de detectar a Eve en UNA iteración.
    Separa limpiamente la lógica del caso ideal vs ruido.
    """
    if noise_rate == 0.0:
        # Caso ideal
        return 1.0 - (1.0 - 0.25 * p_eve)**s
    
    # Caso con ruido: Cálculo combinado de probabilidad de error
    p_err_eve = 0.25 * p_eve  # p_eve es la fracción de qubits interceptados, y cada uno tiene 25% de chance de error
    # XOR probabilístico: Error solo por ruido + Error solo por Eve
    p_err_total = noise_rate * (1 - p_err_eve) + (1 - noise_rate) * p_err_eve
    
    # Recalculamos T igual que lo haría el simulador
    T = compute_threshold(s, noise_rate, alpha)
    
    # La probabilidad de detectarla es la prob de que los errores superen T
    # Usamos 1 - CDF (Función de distribución acumulada)
    return 1.0 - binom.cdf(T, s, p_err_total)

# =====================================================================
# 2. LAS FUNCIONES DE EXPERIMENTO
# =====================================================================


def experiment_variable_R(n_fixed,
                          R_values,
                          numero_ensayos,
                          intercept_prob,
                          noise_rate,
                          alpha):
    
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
    all_metrics = []
    
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
                                              alpha)
                if res:
                    all_metrics.append(res['metrics'])
               
                    if res['eve_detected']:
                        caught = True
                        break  # Si detectamos a Eve en alguna iteración, ya contamos como detectada para este ensayo
            if caught:
                detected_count += 1
            
        p_emp = detected_count / numero_ensayos
        emp_probs.append(p_emp * 100)
        
        # CÁLCULO TEÓRICO 
        s_promedio = int(n_fixed * 0.5) * check_fraction
        if s_promedio == 0:
            s_promedio = 1  # Evitamos división por cero en casos extremos
        
        # Probabilidad de detectar a Eve en al menos una de las R iteraciones
        p_detect_single = theoretical_detection_prob(s_promedio,
                                                     intercept_prob,
                                                     noise_rate,
                                                     alpha)
        p_teo = 1 - (1 - p_detect_single)**R
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

    plot_confusion_matrix(all_metrics, title=f"Matriz de Confusión (R variable, n={n_fixed})")


def experiment_variable_n(qubit_counts,
                          numero_ensayos,
                          intercept_prob,
                          noise_rate,
                          alpha):
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
    teo_probs = []
    all_metrics = []

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
                                          alpha)
            if res:
                valid_runs += 1
                if res['eve_detected']:
                    detected_count += 1
                    
                # Calculamos la probabilidad teórica para el 's' EXACTO de esta ronda
                s_de_esta_ronda = res['s_simulacion']
                suma_p_teo_exacta += theoretical_detection_prob(s_de_esta_ronda,
                                                                intercept_prob,
                                                                noise_rate,
                                                                alpha)
                
                all_metrics.append(res['metrics'])
        
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
    
    plot_confusion_matrix(all_metrics, title=f"Matriz de Confusión (n variable, 1 unica ejecución, iteraciones = {numero_ensayos})")


def experiment_roc_variable_n(qubit_counts, numero_ensayos, intercept_prob, noise_rate):
    """
    Evalúa mediante Curvas ROC cómo mejora la detección al aumentar los Qubits (n).
    """
    print(f"--- Exp: ROC con n Variable | Ruido={noise_rate} ---")
    
    check_fraction = 0.5
    resultados_roc = {}  
    
    for n in qubit_counts:
        # y_true guarda la "Verdad Absoluta": ¿Estaba Eve de verdad? (True/False)
        y_true = []
        # y_scores guarda nuestra "Pista" o "Nivel de sospecha": La tasa de errores observada.
        y_scores = []
        
        for _ in range(numero_ensayos):
            # Simulamos el mundo real: a veces Eve ataca, a veces el canal está tranquilo.
            ataque_activo = np.random.rand() < 0.5
            p_actual = intercept_prob if ataque_activo else 0.0
            
            # Ejecutamos la simulación
            res = simulate_bb84_iteration(n, p_actual, noise_rate, check_fraction, alpha=0.10)
            
            if res:
                # -------------------------------------------------------------
                # LA CLAVE DEL ROC: Guardamos la pista y la verdad, ¡pero no tomamos la decisión final!
                # -------------------------------------------------------------
                # Calculamos el porcentaje de error observado en esta ronda
                tasa_error = res['errors_count'] / res['s_simulacion']
                
                # Anotamos si Eve estaba realmente en la fibra óptica
                y_true.append(res['eve_present'])
                
                # Anotamos la tasa de error (el "score" de sospecha)
                y_scores.append(tasa_error)
                
        # Guardamos todas las verdades y todos los scores para este valor de 'n'
        resultados_roc[f"n={n}"] = (y_true, y_scores)
        
    # La función externa plot_multiple_roc_curves tomará estas dos listas.
    # Ordenará los scores de mayor a menor e irá moviendo el umbral imaginario 
    # paso a paso para dibujar la curva.
    plot_multiple_roc_curves(resultados_roc, title=f"Evolución de la Detección según n (Ruido={noise_rate*100}%)")
    plt.savefig('curvas_roc.png', dpi=300, bbox_inches='tight')
    plt.show()


def experiment_variable_p_and_n(qubit_counts, numero_ensayos, p_values, noise_rate, alpha):
    """
    CASO C (x% Ruido, Eve variable (p))
    """
    print(f"--- Exp: Eve Variable (p) y Qubits Variables (n) | Iter={numero_ensayos}, Ruido={noise_rate} ---")
    check_fraction = 0.5
    
    datos_lineas_emp = {}
    datos_lineas_teo = {}
    datos_matrices = {}
    
    for n in qubit_counts:
        print(f"Simulando para n = {n}...")
        emp_probs = []
        teo_probs = []
        all_metrics = []
        
        for p in p_values:
            detected_count = 0
            valid_runs = 0
            suma_p_teo_exacta = 0.0
            
            for _ in range(numero_ensayos):
                res = simulate_bb84_iteration(n, p, noise_rate, check_fraction, alpha)
                
                if res:
                    all_metrics.append(res['metrics'])
                    valid_runs += 1
                    if res['eve_detected']:
                        detected_count += 1
                        
                    s_de_esta_ronda = res['s_simulacion']
                    suma_p_teo_exacta += theoretical_detection_prob(s_de_esta_ronda, p, noise_rate, alpha)
                        
            if valid_runs > 0:
                emp_probs.append((detected_count / valid_runs) * 100)
                teo_probs.append((suma_p_teo_exacta / valid_runs) * 100)
            else:
                emp_probs.append(0.0)
                teo_probs.append(0.0)
                
        datos_lineas_emp[n] = emp_probs
        datos_lineas_teo[n] = teo_probs
        datos_matrices[n] = all_metrics

    plt.figure(figsize=(9, 6))
    colores = plt.cm.tab10.colors
    
    for idx, n in enumerate(qubit_counts):
        color_actual = colores[idx % len(colores)]
        plt.plot(p_values, datos_lineas_emp[n], marker='o', linestyle='-', color=color_actual, label=f'E (n={n})')
        plt.plot(p_values, datos_lineas_teo[n], marker='', linestyle='--', color=color_actual, alpha=0.6, label=f'T (n={n})')

    plt.xlabel('Fraction of Qubits Intercepted by Eve (p)')
    plt.ylabel('Probability of Detecting Eve (%)')
    plt.xticks(np.arange(0, 1.1, 0.1))
    plt.yticks(np.arange(0, 105, 10))
    # Title removed
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0., frameon=True, edgecolor='black')
    
    plt.savefig('prob_detection_p_and_n.pdf')
    plt.show() 

    for n in qubit_counts:
        plot_confusion_matrix(datos_matrices[n], title=f"Confusion Matrix (n={n})")


def experiment_realistic_scenario(qubit_counts, numero_ensayos, noise_rate, alpha):
    """
    MODO REALISTA 
    """
    print(f"--- Exp: Escenario Realista | Ruido={noise_rate*100}%, Alpha={alpha} ---")
    check_fraction = 0.5
    
    datos_probabilidad = {}
    datos_matrices = {}  
    p_utilizados = {}   
    tasas_error = {}    
    
    rangos_p = np.arange(0.1, 1.1, 0.1) 
    colores = plt.cm.tab10.colors

    for n in qubit_counts:
        datos_probabilidad[n] = {
            'intentos': np.zeros(len(rangos_p)),
            'detecciones': np.zeros(len(rangos_p))
        }
        datos_matrices[n] = []  
        p_utilizados[n] = []    
        tasas_error[n] = []     
        
        for _ in range(numero_ensayos):
            ataque_activo = np.random.rand() < 0.5
            
            if ataque_activo:
                p_actual = np.random.uniform(0.1, 1.0)
            else:
                p_actual = 0.0
                
            res = simulate_bb84_iteration(n, p_actual, noise_rate, check_fraction, alpha)
            
            if res:
                datos_matrices[n].append(res['metrics'])
                
                if ataque_activo:
                    if res['s_simulacion'] > 0:
                        p_utilizados[n].append(p_actual)
                        tasas_error[n].append(res['errors_count'] / res['s_simulacion'])
                    
                    indice_caja = int(round(p_actual * 10)) - 1
                    indice_caja = min(max(indice_caja, 0), len(rangos_p) - 1)
                    
                    datos_probabilidad[n]['intentos'][indice_caja] += 1
                    if res['eve_detected']:
                        datos_probabilidad[n]['detecciones'][indice_caja] += 1

    # Gráfica 1: Dispersión Diferenciada por n
    plt.figure(figsize=(8, 5))
    for idx, n in enumerate(qubit_counts):
        color_actual = colores[idx % len(colores)]
        plt.scatter(p_utilizados[n], tasas_error[n], 
                    alpha=0.6, color=color_actual, s=15, 
                    edgecolor='white', linewidth=0.5, label=f'Qubits (n={n})')

    plt.xlabel('Fraction of Qubits Intercepted by Eve (p)')
    plt.ylabel('Error Rate in Verification (Empirical QBER)')
    # Title removed
    plt.legend(frameon=True, edgecolor='black')
    plt.savefig('realistic_scenario_error_by_n.pdf')
    plt.show()

    # Gráfica 2: Probabilidad de Detección vs p Agrupada
    plt.figure(figsize=(8, 5))
    for idx, n in enumerate(qubit_counts):
        intentos = datos_probabilidad[n]['intentos']
        detecciones = datos_probabilidad[n]['detecciones']
        
        probabilidades = []
        for i in range(len(rangos_p)):
            if intentos[i] > 0:
                prob = (detecciones[i] / intentos[i]) * 100
            else:
                prob = np.nan
            probabilidades.append(prob)
            
        color_actual = colores[idx % len(colores)]
        plt.plot(rangos_p, probabilidades, marker='s', linestyle='-', 
                 color=color_actual, label=f'Grouped Empirical Detection (n={n})')

    plt.xlabel('Fraction of Qubits Intercepted by Eve (p) - Grouped')
    plt.ylabel('Probability of Detection (%)')
    plt.xticks(rangos_p)
    plt.yticks(np.arange(0, 105, 10))
    # Title removed
    plt.legend(frameon=True, edgecolor='black')
    plt.savefig('prob_agrupada_realista.pdf')
    plt.show()

    # Gráfica 3: Matrices de Confusión Individuales por cada 'n'
    for n in qubit_counts:
        plot_confusion_matrix(datos_matrices[n], title=f"Confusion Matrix (n={n})")


def experiment_roc_variable_p(n_fixed, numero_ensayos, p_values, noise_rate):
    """
    Evalúa mediante Curvas ROC cómo se degrada la capacidad de detección.
    """
    print(f"--- Exp: ROC con p Variable | n={n_fixed}, Ruido={noise_rate*100:.0f}% ---")
    
    check_fraction = 0.5
    alpha_dummy = 0.10 
    
    resultados_roc = {}  
    
    for p in p_values:
        if p == 0.0:
            continue
            
        y_true = []
        y_scores = []
        
        for _ in range(numero_ensayos):
            ataque_activo = np.random.rand() < 0.5
            p_actual = p if ataque_activo else 0.0
            
            res = simulate_bb84_iteration(n_fixed, p_actual, noise_rate, check_fraction, alpha_dummy)
            
            if res:
                tasa_error = res['errors_count'] / res['s_simulacion']
                y_true.append(res['eve_present'])
                y_scores.append(tasa_error)
                
        resultados_roc[f"Eve p={p*100:.0f}%"] = (y_true, y_scores)
        
    plot_multiple_roc_curves(resultados_roc, title=f"ROC n={n_fixed}")


def experiment_compare_noise_profiles(n_fixed, numero_ensayos, p_values, noise_rate, alpha):
    """
    MODO E: Compara cómo de fácil es detectar a Eve según el tipo de ruido en el canal.
    Evalúa: Bit-Flip, Phase-Flip, y Los 2 combinados.
    """
    print(f"--- Exp: Comparativa de Ruidos | n={n_fixed}, Ruido={noise_rate*100}% ---")
    check_fraction = 0.5
    
    perfiles = ['bit_flip', 'phase_flip', 'all']
    nombres = {
        'bit_flip': 'Bit-Flip',
        'phase_flip': 'Phase-Flip',
        'all': 'All 2 types of error'
    }
    colores = {'bit_flip': 'blue', 'phase_flip': 'orange', 'all': 'red'}
    
    # Diccionario para almacenar las probabilidades de detección de cada perfil
    resultados = {prof: [] for prof in perfiles}
    
    for prof in perfiles:
        print(f"Simulando ruido: {nombres[prof]}...")
        for p in p_values:
            detected_count = 0
            valid_runs = 0
            
            for _ in range(numero_ensayos):
                res = simulate_bb84_iteration(
                    num_qubits=n_fixed, 
                    intercept_prob=p, 
                    noise_rate=noise_rate, 
                    check_fraction=check_fraction, 
                    alpha=alpha, 
                    noise_profile=prof  # Inyectamos el perfil específico
                )
                
                if res:
                    valid_runs += 1
                    if res['eve_detected']:
                        detected_count += 1
                        
            prob_emp = (detected_count / valid_runs * 100) if valid_runs > 0 else 0.0
            resultados[prof].append(prob_emp)
            
    # ---------------------------------------------------------
    # FASE DE DIBUJO: SUPERPOSICIÓN DE LAS 4 CURVAS
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for prof in perfiles:
        plt.plot(p_values, resultados[prof], marker='o', lw=2, 
                 color=colores[prof], label=nombres[prof])

    plt.title(f'Eve Detection Capability vs Environmental Noise Type\n(n={n_fixed}, Noise Rate={noise_rate*100}%, Alpha={alpha})', fontsize=14)
    plt.xlabel('Fraction of Qubits Intercepted by Eve (p)', fontsize=12)
    plt.ylabel('Probability of Detecting Eve (%)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('comparativa_ruidos.png', dpi=300, bbox_inches='tight')
    plt.show()
# =====================================================================
# 3. EL ORQUESTADOR (Donde decides qué estudiar hoy)
# =====================================================================


if __name__ == "__main__":
    
    CASO_A_ESTUDIAR = "D"
    
    if CASO_A_ESTUDIAR == "A":
        # CASO A: 0% Ruido, 100% Intercepción
        noise = 0.0
        p_eve = 1.0
        alpha_fp = 0.0  # Cualquier error es Eve
        
        # A.1: R variable

        experiment_variable_R(n_fixed=16,
                              R_values=[1, 2, 3, 5, 8, 12, 20],
                              numero_ensayos=1000,
                              intercept_prob=p_eve,
                              noise_rate=noise,
                              alpha=alpha_fp)

        # A.2: n variable

        experiment_variable_n(qubit_counts=[1, 4, 8, 16, 32, 64, 128],
                              numero_ensayos=1000,
                              intercept_prob=p_eve,
                              noise_rate=noise,
                              alpha=alpha_fp)

    elif CASO_A_ESTUDIAR == "B":
        # CASO B: 5% Ruido, 100% Intercepción
        noise = 0.05
        p_eve = 1.0
        # Aceptamos un 10% de Falsos Positivos
        alpha_fp = 0.10
        
        print("--- Generando gráficas teóricas previas ---")
        # 1. Mostramos cómo se comporta el umbral en general

        # plot_static_threshold(s_simulacion=50)

        # 2. Mostramos la distribución de errores para este escenario concreto (asumiendo s=50 para la foto)
        plot_error_distributions(s_simulacion=50, 
                                 noise_rate=noise, 
                                 intercept_prob=p_eve, 
                                 alpha=alpha_fp)
        """
        experiment_variable_R(n_fixed=16,
                              R_values=[1, 2, 3, 5, 8, 12, 20],
                              numero_ensayos=1000,
                              intercept_prob=p_eve,
                              noise_rate=noise,
                              alpha=alpha_fp)
                              
        experiment_variable_n(qubit_counts=[1, 4, 8, 16, 32, 64, 128],
                              numero_ensayos=1000,
                              intercept_prob=p_eve,
                              noise_rate=noise,
                              alpha=alpha_fp) 

        experiment_roc_variable_n(qubit_counts=[16, 32, 64, 128],
                                  numero_ensayos=1000,
                                  intercept_prob=p_eve,
                                  noise_rate=noise) """
            
    elif CASO_A_ESTUDIAR == "C":
        # CASO C: 5% Ruido, Eve variable (p)

        noise = 0.05
        alpha_fp = 0.10
        
        qubit_counts_lista = [16, 32, 64, 128, 256]
        p_valores_lista = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    
        #experiment_variable_p_and_n(
        #    qubit_counts=qubit_counts_lista,
        #    numero_ensayos=1000,
        #    p_values=p_valores_lista,
        #    noise_rate=noise,
        #    alpha=alpha_fp
        #)

        # ---------------------------------------------------------
        # Generamos el Análisis ROC de Sensibilidad
        # ---------------------------------------------------------
        # Seleccionamos un 'n' fuerte (ej. 128) para demostrar su talón de Aquiles
        n_optimo = 128
        
        # Seleccionamos unos pocos valores de p para que la gráfica quede limpia y clara:
        # 10% (Débil), 30% (Medio), 50% (Peligroso), 100% (Total)
        p_valores_roc = [0.1, 0.3, 0.5, 1.0] 
        
        experiment_roc_variable_p(
            n_fixed=n_optimo,
            numero_ensayos=1000,
            p_values=p_valores_roc,
            noise_rate=noise
        )
    
    elif CASO_A_ESTUDIAR == "D":
        # CASO D:(Ruido, Eve Impredecible y ajuste de Alpha)
        noise = 0.05       # 5% de ruido ambiental
        alpha_fp = 0.10    # Subimos alpha al 10% para ser más sensibles y reducir Falsos Negativos
        
        # Usamos tamaños de qubits más grandes para dar significancia estadística
        # y que Eve no se escape simplemente por "suerte" en muestras pequeñas.
        qubit_counts_lista = [32, 256] 
        
        experiment_realistic_scenario(
            qubit_counts=qubit_counts_lista,
            numero_ensayos=1000,
            noise_rate=noise,
            alpha=alpha_fp
        )
    
    elif CASO_A_ESTUDIAR == "E":
        # CASO E: Comparativa entre tipos de Ruido
        noise = 0.05
        alpha_fp = 0.10
        n_optimo = 128
        
        # Saltos del 20% para que la simulación no tarde mucho
        p_valores_lista = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0] 
        
        experiment_compare_noise_profiles(
            n_fixed=n_optimo,
            numero_ensayos=2000,  
            p_values=p_valores_lista,
            noise_rate=noise,
            alpha=alpha_fp
        )