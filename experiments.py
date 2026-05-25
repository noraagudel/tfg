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


def experiment_variable_p_and_n(qubit_counts,
                                numero_ensayos,
                                p_values,
                                noise_rate,
                                alpha):
    """
    CASO C (x% Ruido, Eve variable (p))
     - qubit_counts: Lista de diferentes números de qubits a probar
     - numero_ensayos: Número de veces que se repite el experimento para cada combinación de p y n (para hacer estadística)
     - p_values: Lista de diferentes valores de p (fracción de qubits interceptados por Eve) a probar
     - noise_rate: Tasa de ruido ambiental 
     - threshold: Umbral para decidir si detectamos a Eve o no 
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
        all_metrics = []
        
        # Color asignado a este tamaño de n
        color_actual = colores[idx % len(colores)]
        
        for p in p_values:
            detected_count = 0
            valid_runs = 0
            suma_p_teo_exacta = 0.0
            
            for _ in range(numero_ensayos):
                res = simulate_bb84_iteration(n,
                                              p,
                                              noise_rate,
                                              check_fraction,
                                              alpha)
                
                if res:
                    all_metrics.append(res['metrics'])
                    valid_runs += 1
                    if res['eve_detected']:
                        detected_count += 1
                        
                    # CÁLCULO TEÓRICO EXACTO (Evaluado por ronda)
                    s_de_esta_ronda = res['s_simulacion']
                    suma_p_teo_exacta += theoretical_detection_prob(s_de_esta_ronda,
                                                                    p,
                                                                    noise_rate,
                                                                    alpha)
                        
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
    plt.title('Probabilidad de Detección vs Tasa de Intercepción de Eve (p)\n(2% Ruido Ambiental)', fontsize=14)
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

    plot_confusion_matrix(all_metrics, title="Matriz de Confusión Global (Eve y n variables)")


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
            res = simulate_bb84_iteration(n, p_actual, noise_rate, check_fraction, alpha=0.05)
            
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


def experiment_realistic_scenario(qubit_counts,
                                  numero_ensayos,
                                  noise_rate,
                                  alpha):
    """
    MODO REALISTA 
    Simula un canal real con ataques aleatorios y además calcula 
    la probabilidad de detección agrupando los ataques por rangos.
    """
    print(f"--- Exp: Escenario Realista | Ruido={noise_rate*100}%, Alpha={alpha} ---")
    
    check_fraction = 0.5
    all_metrics = []
    p_utilizados = []
    tasas_error = []
    
    # Diccionario para almacenar los datos agrupados (binning) por cada tamaño de n
    # Estructura: { n: { 'rangos': [0.1, 0.2...], 'intentos': [0, 0...], 'detecciones': [0, 0...] } }
    datos_probabilidad = {}
    rangos_p = np.arange(0.1, 1.1, 0.1) # Crea cajas: 0.1, 0.2, 0.3 ... 1.0

    for n in qubit_counts:
        # Inicializamos los contadores para este tamaño de qubits
        datos_probabilidad[n] = {
            'intentos': np.zeros(len(rangos_p)),
            'detecciones': np.zeros(len(rangos_p))
        }
        
        for _ in range(numero_ensayos):
            # 1. Decisión de ataque
            ataque_activo = np.random.rand() < 0.5
            
            # 2. Asignación de p
            if ataque_activo:
                p_actual = np.random.uniform(0.1, 1.0)
            else:
                p_actual = 0.0
                
            # 3. Simulación
            res = simulate_bb84_iteration(n,
                                          p_actual,
                                          noise_rate,
                                          check_fraction,
                                          alpha)
            
            if res:
                # Guardamos para la matriz de confusión global
                all_metrics.append(res['metrics'])
                
                if ataque_activo:
                    # Guardamos para la gráfica de dispersión
                    if res['s_simulacion'] > 0:
                        p_utilizados.append(p_actual)
                        tasas_error.append(res['errors_count'] / res['s_simulacion'])
                    
                    # NUEVO: Lógica de agrupación para la probabilidad
                    # np.digitize nos dice en qué "caja" (índice) cae el p_actual
                    # Restamos 1 porque los índices de arrays empiezan en 0
                    indice_caja = np.digitize(p_actual, rangos_p) - 1
                    
                    # Aseguramos que el índice no se salga de los límites
                    indice_caja = min(indice_caja, len(rangos_p) - 1)
                    indice_caja = max(indice_caja, 0)
                    
                    # Sumamos 1 intento a esta caja
                    datos_probabilidad[n]['intentos'][indice_caja] += 1
                    
                    # Si detectamos a Eve, sumamos 1 detección a esta caja
                    if res['eve_detected']:
                        datos_probabilidad[n]['detecciones'][indice_caja] += 1

    # ---------------------------------------------------------
    # GRÁFICAS Y VISUALIZACIÓN
    # ---------------------------------------------------------

    # Gráfica 1: Dispersión
    plt.figure(figsize=(10, 6))
    plt.scatter(p_utilizados, tasas_error, alpha=0.5, color='purple', s=10) # s=10 hace los puntos más pequeños
    plt.title('Tasa de Errores Observada vs Agresividad Aleatoria de Eve (p)')
    plt.xlabel('Fracción de Intercepción de Eve (p)')
    plt.ylabel('Tasa de Error en Comprobación')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('dispersion_realista.png', dpi=300, bbox_inches='tight')
    plt.show()

    # NUEVA Gráfica 2: Probabilidad de Detección vs p (Estilo Caso C)
    plt.figure(figsize=(10, 6))
    colores = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']
    
    for idx, n in enumerate(qubit_counts):
        intentos = datos_probabilidad[n]['intentos']
        detecciones = datos_probabilidad[n]['detecciones']
        
        probabilidades = []
        for i in range(len(rangos_p)):
            if intentos[i] > 0:
                # Calculamos el porcentaje
                prob = (detecciones[i] / intentos[i]) * 100
            else:
                prob = 0.0
            probabilidades.append(prob)
            
        color_actual = colores[idx % len(colores)]
        plt.plot(rangos_p, probabilidades, marker='o', linestyle='-', 
                 color=color_actual, label=f'Empírica Agrupada (n={n})')

    plt.title(f'Probabilidad de Detección vs Tasa de Intercepción (p) \nEscenario Realista (Ruido={noise_rate*100}%, Alpha={alpha})')
    plt.xlabel('Fracción de Intercepción de Eve (p) - Agrupada')
    plt.ylabel('Probabilidad de Detección (%)')
    plt.xticks(rangos_p)
    plt.yticks(np.arange(0, 105, 10))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.savefig('prob_agrupada_realista.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Gráfica 3: Matriz de Confusión
    plot_confusion_matrix(all_metrics, title=f"Matriz de Confusión Realista\n(Ruido={noise_rate*100}%, Alpha={alpha})")

# =====================================================================
# 3. EL ORQUESTADOR (Donde decides qué estudiar hoy)
# =====================================================================


if __name__ == "__main__":
    
    CASO_A_ESTUDIAR = "C" 
    
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
        # CASO B: 2% Ruido, 100% Intercepción
        noise = 0.02
        p_eve = 1.0
        # Aceptamos un 5% de Falsos Positivos
        alpha_fp = 0.05

        print("--- Generando gráficas teóricas previas ---")
        # 1. Mostramos cómo se comporta el umbral en general
        plot_static_threshold(s_simulacion=50)
        
        # 2. Mostramos la distribución de errores para este escenario concreto (asumiendo s=50 para la foto)
        plot_error_distributions(s_simulacion=50, 
                                 noise_rate=noise, 
                                 intercept_prob=p_eve, 
                                 alpha=alpha_fp)
        
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

        experiment_roc_variable_n(qubit_counts=[8, 16, 32, 64],
                                  numero_ensayos=1000,
                                  intercept_prob=p_eve,
                                  noise_rate=noise)
            
    elif CASO_A_ESTUDIAR == "C":
        # CASO C: 2% Ruido, Eve variable (p)
        noise = 0.02
        alpha_fp = 0.10
        
        qubit_counts_lista = [16, 32, 64, 128, 256]
        p_valores_lista = np.linspace(0.0, 1.0, 11)  # [0.0, 0.1, 0.2, ..., 1.0]
    
        experiment_variable_p_and_n(
            qubit_counts=qubit_counts_lista,
            numero_ensayos=1000,
            p_values=p_valores_lista,
            noise_rate=noise,
            alpha=alpha_fp
        )
    
    elif CASO_A_ESTUDIAR == "D":
        # CASO D:(Ruido, Eve Impredecible y ajuste de Alpha)
        noise = 0.05       # 5% de ruido ambiental
        alpha_fp = 0.10    # Subimos alpha al 10% para ser más sensibles y reducir Falsos Negativos
        
        # Usamos tamaños de qubits más grandes para dar significancia estadística
        # y que Eve no se escape simplemente por "suerte" en muestras pequeñas.
        qubit_counts_lista = [32, 64, 128, 256] 
        
        experiment_realistic_scenario(
            qubit_counts=qubit_counts_lista,
            numero_ensayos=1000,
            noise_rate=noise,
            alpha=alpha_fp
        )