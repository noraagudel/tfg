import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from scipy.stats import binom
from bb84_simulator import compute_threshold


# ---------------------------------------------------------
# GLOBAL SETTINGS FOR ACADEMIC GRAPHICS
# ---------------------------------------------------------
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['xtick.labelsize'] = 11
mpl.rcParams['ytick.labelsize'] = 11
mpl.rcParams['legend.fontsize'] = 11
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['axes.grid'] = True
mpl.rcParams['grid.alpha'] = 0.5
mpl.rcParams['grid.linestyle'] = '--'


def plot_confusion_matrix(metrics_list, title="Matriz de Confusión del Escenario"):
    """
    Agrega una lista de diccionarios de métricas, calcula porcentajes de éxito
    e imprime un reporte estadístico detallado. Dibuja una matriz NORMALIZADA.
    """
    # 1. Acumulamos los totales de cada métrica en variables numéricas
    total_TP = sum(m['TP'] for m in metrics_list)
    total_FP = sum(m['FP'] for m in metrics_list)
    total_TN = sum(m['TN'] for m in metrics_list)
    total_FN = sum(m['FN'] for m in metrics_list)
    
    total_casos = total_TP + total_FP + total_TN + total_FN

    if total_TN == 0 and total_FP == 0:
        print(f"\n[Aviso] Se omite '{title}':")
        print("-> Eve atacó en el 100% de los casos. No hay métricas negativas que graficar.\n")
        return 
    
    # ---------------------------------------------------------
    # CÁLCULO DE PORCENTAJES ESTADÍSTICOS
    # ---------------------------------------------------------
    casos_con_eve = total_TP + total_FN  
    casos_solo_ruido = total_FP + total_TN  
    
    porcentaje_deteccion = (total_TP / casos_con_eve * 100) if casos_con_eve > 0 else 0.0
    porcentaje_falsas_alarmas = (total_FP / casos_solo_ruido * 100) if casos_solo_ruido > 0 else 0.0
    porcentaje_precision_global = ((total_TP + total_TN) / total_casos * 100) if total_casos > 0 else 0.0

    print("\n" + "="*50)
    print(f" REPORTES ESTADÍSTICOS: {title}")
    print("="*50)
    print(f"Total de iteraciones evaluadas en el código : {total_casos}")
    print(f"-> Eficacia de Detección (Atrapamos a Eve)  : {porcentaje_deteccion:.2f}% ({total_TP}/{casos_con_eve})")
    print(f"-> Tasa de Falsas Alarmas (Error de Ruido) : {porcentaje_falsas_alarmas:.2f}% ({total_FP}/{casos_solo_ruido})")
    print(f"-> Precisión Total del Umbral T            : {porcentaje_precision_global:.2f}%")
    print("="*50 + "\n")

    # 2. Estructuramos las matrices para Matplotlib
    matrix_abs = np.array([[total_TP, total_FN],
                           [total_FP, total_TN]])
    
    matrix_norm = np.zeros((2, 2))
    if casos_con_eve > 0:
        matrix_norm[0, 0] = total_TP / casos_con_eve
        matrix_norm[0, 1] = total_FN / casos_con_eve
    if casos_solo_ruido > 0:
        matrix_norm[1, 0] = total_FP / casos_solo_ruido
        matrix_norm[1, 1] = total_TN / casos_solo_ruido
    
    fig, ax = plt.subplots(figsize=(6, 5))
    
    cax = ax.matshow(matrix_norm, cmap='Blues', vmin=0, vmax=1)
    
    for (i, j), val_norm in np.ndenumerate(matrix_norm):
        val_abs = matrix_abs[i, j]
        texto_celda = f"{val_norm * 100:.1f}%\n({val_abs})"
        
        ax.text(j, i, texto_celda, ha='center', va='center', 
                color='white' if val_norm > 0.5 else 'black',
                fontsize=20, fontweight='bold')
    
    cbar = plt.colorbar(cax)
    ticks = cbar.get_ticks()
    cbar.ax.set_yticks(ticks) 
    cbar.ax.set_yticklabels([f"{x*100:.0f}%" for x in ticks]) 
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Predicted\nPositive', 'Predicted\nNegative'], fontsize=20)
    ax.set_yticklabels(['Actual\nPositive', 'Actual\nNegative'], fontsize=20)

    # Titles removed for thesis formatting
    plt.savefig(f'confusion_matrix_{title.replace(" ", "_")}.pdf')
    plt.show()


def plot_multiple_roc_curves(dict_of_results, title="Comparativa de Curvas ROC"):
    """
    Dibuja varias curvas ROC en la misma gráfica para comparar escenarios.
    Calcula y marca el umbral óptimo usando el Índice de Youden (J = TPR - FPR).
    """
    plt.figure(figsize=(8, 6))
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
            
        tpr_array = np.array(tpr_list)
        fpr_array = np.array(fpr_list)
        
        j_scores = tpr_array - fpr_array
        best_idx = np.argmax(j_scores)
        best_threshold = umbrales[best_idx]
        best_tpr = tpr_array[best_idx]
        best_fpr = fpr_array[best_idx]
        
        if best_fpr == 0:
            ratio_str = "N/A" if best_tpr == 0 else "∞"
        else:
            ratio_str = f"{(best_tpr / best_fpr):.2f}"
        
        indices = np.argsort(fpr_list)
        fpr_sorted = np.array(fpr_list)[indices]
        tpr_sorted = np.array(tpr_list)[indices]
        
        try:
            auc_value = np.trapezoid(tpr_sorted, fpr_sorted)
        except AttributeError:
            auc_value = np.trapz(tpr_sorted, fpr_sorted)
            
        color = colores[idx % len(colores)]
        
        plt.plot(fpr_sorted, tpr_sorted, lw=2, color=color,
                 label=f'{label_name} (AUC = {auc_value:.3f})')
        
        plt.plot(best_fpr, best_tpr, marker='s', markersize=8, color=color, 
                 markeredgecolor='black', linestyle='None', zorder=5)
        
        annot_text = f'T={best_threshold*100:.1f}%\nTPR/FPR={ratio_str}'
        
        annot_text = f'T={best_threshold*100:.1f}%\nTPR/FPR={ratio_str}'
        # Create a dynamic offset that fans out the boxes down and to the right
        base_offset_x = 10 
        base_offset_y = -10 
        # Add a staggered offset based on the loop index (idx) to avoid overlapping among the boxes
        stagger_x = idx * 15
        stagger_y = idx * -10
        
        # Calculate the final offset pair
        xytext_offset = (base_offset_x + stagger_x, base_offset_y + stagger_y)
        
        annot_text = f'T={best_threshold*100:.1f}%\nTPR/FPR={ratio_str}'
        plt.annotate(annot_text, (best_fpr, best_tpr), 
                     textcoords="offset points", 
                     xytext=xytext_offset, ha='left', va='top', 
                     fontsize=10, color='black', 
                     # Make the box more opaque for better legibility against the grid and lines
                     bbox=dict(boxstyle="square,pad=0.3", fc="white", ec=color, alpha=0.9))

    plt.plot([0, 1], [0, 1], color='black', lw=1.5, linestyle='--', label='Random (AUC = 0.5)')
    
    plt.xlim([-0.02, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    # Title removed
    plt.legend(loc="lower right", frameon=True, edgecolor='black')
    
    plt.savefig('comparativa_roc.pdf')
    plt.show()


def plot_error_distributions(s_simulacion, noise_rate, intercept_prob, alpha):
    """
    Compara las distribuciones de errores esperadas con y sin la presencia de Eve.
    """
    p_err_eve = 0.25 * intercept_prob
    p_err_total = noise_rate * (1 - p_err_eve) + (1 - noise_rate) * p_err_eve
    
    k_values = np.arange(0, s_simulacion + 1)
    
    pmf_solo_ruido = binom.pmf(k_values, s_simulacion, noise_rate)
    pmf_con_eve = binom.pmf(k_values, s_simulacion, p_err_total)
    
    T = compute_threshold(s_simulacion, noise_rate, alpha)
    
    plt.figure(figsize=(8, 5))
    
    plt.fill_between(k_values, pmf_solo_ruido, color='#1f77b4', alpha=0.3, label='Only external noise')
    plt.fill_between(k_values, pmf_con_eve, color='#d62728', alpha=0.3, label='External noise + Eve')

    plt.plot(k_values, pmf_solo_ruido, color='#1f77b4', lw=2)
    plt.plot(k_values, pmf_con_eve, color='#d62728', lw=2)
    
    plt.axvline(x=T, color='black', linestyle='--', lw=1.5, label=f'Threshold T ({T} errors)')
    
    plt.xlabel('Number of observed errors (k)')
    plt.ylabel('Probability $P(X = k)$')
    plt.xlim(0, max(20, T * 2.5)) 
    # Title removed
    plt.legend(frameon=True, edgecolor='black')
    
    plt.savefig('error_distributions.pdf')
    plt.show()


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
        plt.step(alphas, Ts, where='post', color=color, linewidth=2, label=f'External Noise: {noise*100}%')

    plt.title(f'Evolution of threshold T vs False Positive Tolerance $\\alpha$\n(Check length s={s_simulacion})')
    plt.xlabel(f'False Positive Tolerance ($\alpha$)')
    plt.ylabel('Tolerated Error Threshold (T)')

    plt.xticks(np.arange(0, 0.22, 0.02), [f"{x*100:.0f}%" for x in np.arange(0, 0.22, 0.02)])
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()