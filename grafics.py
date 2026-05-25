import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binom
from bb84_simulator import compute_threshold


def plot_confusion_matrix(metrics_list, title="Matriz de Confusión del Escenario"):
    """
    Agrega una lista de diccionarios de métricas, calcula porcentajes de éxito
    e imprime un reporte estadístico detallado en la consola antes de graficar.
    """
    # 1. Acumulamos los totales de cada métrica en variables numéricas
    total_TP = sum(m['TP'] for m in metrics_list)
    total_FP = sum(m['FP'] for m in metrics_list)
    total_TN = sum(m['TN'] for m in metrics_list)
    total_FN = sum(m['FN'] for m in metrics_list)
    
    total_casos = total_TP + total_FP + total_TN + total_FN

    # Comprobación de seguridad para evitar pantallas vacías
    if total_TN == 0 and total_FP == 0:
        print(f"\n[Aviso] Se omite '{title}':")
        print("-> Eve atacó en el 100% de los casos. No hay métricas negativas que graficar.\n")
        return 
    
    # ---------------------------------------------------------
    # NUEVO: CÁLCULO DE PORCENTAJES ESTADÍSTICOS DEL CÓDIGO
    # ---------------------------------------------------------
    casos_con_eve = total_TP + total_FN  # Total de veces que hubo ataque
    casos_solo_ruido = total_FP + total_TN  # Total de veces que el canal estuvo limpio
    
    # Evitamos divisiones por cero con condicionales en línea
    porcentaje_deteccion = (total_TP / casos_con_eve * 100) if casos_con_eve > 0 else 0.0
    porcentaje_falsas_alarmas = (total_FP / casos_solo_ruido * 100) if casos_solo_ruido > 0 else 0.0
    porcentaje_precision_global = ((total_TP + total_TN) / total_casos * 100) if total_casos > 0 else 0.0

    # Imprimimos el reporte analítico directamente en la consola de Python
    print("\n" + "="*50)
    print(f" REPORTES ESTADÍSTICOS: {title}")
    print("="*50)
    print(f"Total de iteraciones evaluadas en el código : {total_casos}")
    print(f"-> Eficacia de Detección (Atrapamos a Eve)  : {porcentaje_deteccion:.2f}% ({total_TP}/{casos_con_eve})")
    print(f"-> Tasa de Falsas Alarmas (Error de Ruido) : {porcentaje_falsas_alarmas:.2f}% ({total_FP}/{casos_solo_ruido})")
    print(f"-> Precisión Total del Umbral T            : {porcentaje_precision_global:.2f}%")
    print("="*50 + "\n")

    # 2. Estructuramos la matriz 2x2 para Matplotlib
    matrix = np.array([[total_TP, total_FN],
                       [total_FP, total_TN]])
    
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(matrix, cmap='Blues')
    
    # Colocamos los números dentro de las celdas de la gráfica
    for (i, j), val in np.ndenumerate(matrix):
        ax.text(j, i, f'{val}', ha='center', va='center', 
                color='white' if val > (np.max(matrix)/2) else 'black',
                fontsize=14, fontweight='bold')
    
    plt.title(title, pad=20)
    plt.colorbar(cax)
    
    # Configuramos las etiquetas de los ejes
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Alarma (Detectada)', 'Seguro (No Detectada)'])
    ax.set_yticklabels(['Eve Presente', 'Solo Ruido'])
    
    plt.xlabel('Predicción del Sistema (Umbral T)', fontsize=12)
    plt.ylabel('Realidad (Simulación)', fontsize=12)
    
    # Reajuste automático de márgenes para evitar textos cortados
    plt.tight_layout() 
    plt.show()


def plot_multiple_roc_curves(dict_of_results, title="Comparativa de Curvas ROC"):
    """
    Dibuja varias curvas ROC en la misma gráfica para comparar escenarios.
    Calcula y marca el umbral óptimo usando el Índice de Youden (J = TPR - FPR).
    """
    plt.figure(figsize=(9, 7))
    colores = plt.cm.tab10.colors  
    
    for idx, (label_name, (y_true, y_scores)) in enumerate(dict_of_results.items()):
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        
        P = np.sum(y_true)
        N = len(y_true) - P
        
        if P == 0 or N == 0:
            continue 
            
        umbrales = np.sort(np.unique(y_scores))[::-1]
        umbrales = np.concatenate(([max(umbrales) + 0.1], umbrales, [min(umbrales) - 0.1]))
        
        tpr_list, fpr_list = [], []
        for T in umbrales:
            prediccion_ataque = y_scores >= T
            TP = np.sum(prediccion_ataque & y_true)
            FP = np.sum(prediccion_ataque & ~y_true)
            tpr_list.append(TP / P)
            fpr_list.append(FP / N)
            
        # --- NUEVO: CÁLCULO DEL PUNTO ÓPTIMO (YOUDEN'S J) ---
        tpr_array = np.array(tpr_list)
        fpr_array = np.array(fpr_list)
        
        # J = Sensibilidad (TPR) + Especificidad (1 - FPR) - 1  --> Equivalente a TPR - FPR
        j_scores = tpr_array - fpr_array
        
        # Obtenemos el índice donde J es máximo
        best_idx = np.argmax(j_scores)
        best_threshold = umbrales[best_idx]
        best_tpr = tpr_array[best_idx]
        best_fpr = fpr_array[best_idx]
        # ----------------------------------------------------
        
        # Ordenamos para dibujar la curva suavemente
        indices = np.argsort(fpr_list)
        fpr_sorted = np.array(fpr_list)[indices]
        tpr_sorted = np.array(tpr_list)[indices]
        
        try:
            auc_value = np.trapezoid(tpr_sorted, fpr_sorted)
        except AttributeError:
            auc_value = np.trapz(tpr_sorted, fpr_sorted)
            
        color = colores[idx % len(colores)]
        
        # Dibujamos la curva principal
        plt.plot(fpr_sorted, tpr_sorted, lw=2, color=color,
                 label=f'{label_name} (AUC = {auc_value:.3f})')
        
        # --- NUEVO: DIBUJAMOS LA ESTRELLA EN EL PUNTO ÓPTIMO ---
        plt.plot(best_fpr, best_tpr, marker='*', markersize=12, color=color, 
                 markeredgecolor='black', linestyle='None')
        
        # Anotamos el valor del umbral (T) como porcentaje de error al lado de la estrella
        plt.annotate(f'T={best_threshold*100:.1f}%', (best_fpr, best_tpr), 
                     textcoords="offset points", xytext=(12,-5), ha='left', fontsize=10, color=color)

    plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--', label='Azar (AUC = 0.5)')
    
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)', fontsize=12)
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    

def plot_static_threshold(s_simulacion=50):
    """
    Alternativa estática: Muestra el umbral T frente a alpha para diferentes niveles de ruido.
    """
    alphas = np.linspace(0.001, 0.20, 200)
    noise_rates = [0.0, 0.02, 0.05, 0.10] # Diferentes escenarios de ruido
    colores = ['green', 'blue', 'orange', 'red']
    
    plt.figure(figsize=(10, 6))
    
    for noise, color in zip(noise_rates, colores):
        Ts = [int(binom.ppf(1 - a, s_simulacion, noise)) for a in alphas]
        # Dibujamos las líneas en formato escalón
        plt.step(alphas, Ts, where='post', color=color, linewidth=2, label=f'Ruido: {noise*100}%')
        
    plt.title(f'Evolución del Umbral T vs Tolerancia a Falsos Positivos $\\alpha$\n(Longitud de comprobación s={s_simulacion})')
    plt.xlabel(r'Tolerancia a Falsos Positivos ($\alpha$)')
    plt.ylabel('Umbral Tolerado de Errores (T)')
    
    plt.xticks(np.arange(0, 0.22, 0.02), [f"{x*100:.0f}%" for x in np.arange(0, 0.22, 0.02)])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_error_distributions(s_simulacion, noise_rate, intercept_prob, alpha):
    """
    Compara las distribuciones de errores esperadas con y sin la presencia de Eve.
    """
    # Cálculos de probabilidad (reutilizando tu excelente lógica)
    p_err_eve = 0.25 * intercept_prob
    p_err_total = noise_rate * (1 - p_err_eve) + (1 - noise_rate) * p_err_eve
    
    # Rango de posibles errores (de 0 hasta s_simulacion)
    k_values = np.arange(0, s_simulacion + 1)
    
    # Calculamos la PMF para ambos escenarios
    pmf_solo_ruido = binom.pmf(k_values, s_simulacion, noise_rate)
    pmf_con_eve = binom.pmf(k_values, s_simulacion, p_err_total)
    
    # Calculamos el umbral T
    T = compute_threshold(s_simulacion, noise_rate, alpha)
    
    plt.figure(figsize=(10, 6))
    
    # Rellenamos las áreas bajo las curvas para mayor claridad
    plt.fill_between(k_values, pmf_solo_ruido, color='blue', alpha=0.3, label='Solo Ruido (Canal Seguro)')
    plt.fill_between(k_values, pmf_con_eve, color='red', alpha=0.3, label='Ruido + Eve (Ataque)')
    
    # Dibujamos las líneas de las curvas
    plt.plot(k_values, pmf_solo_ruido, color='blue', lw=2)
    plt.plot(k_values, pmf_con_eve, color='red', lw=2)
    
    # Línea vertical para el Umbral T
    plt.axvline(x=T, color='black', linestyle='--', lw=2, label=f'Umbral T ({T} errores)')
    
    plt.title(f'Distribución de Errores (s={s_simulacion}, Ruido={noise_rate:.2f}, p_Eve={intercept_prob:.2f})')
    plt.xlabel('Número de Errores Observados (k)')
    plt.ylabel('Probabilidad $P(X = k)$')
    plt.xlim(0, max(20, T * 2.5)) # Ajustamos el zoom dinámicamente
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.show()

# Puedes llamarlo así desde tu código principal:
# plot_error_distributions(s_simulacion=100, noise_rate=0.02, intercept_prob=1.0, alpha=0.05)
