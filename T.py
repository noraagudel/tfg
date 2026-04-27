import math
from scipy.stats import binom

def calcular_T_manual(s, p0, alpha):
    """
    Calcula el umbral T sumando probabilidades manualmente en un bucle.
    s: tamaño de la muestra (bits sacrificados)
    p0: probabilidad de error por ruido ambiental
    alpha: tolerancia a falsas alarmas (ej. 0.05)
    """
    objetivo_acumulado = 1.0 - alpha
    probabilidad_acumulada = 0.0
    T = 0
    
    # Bucle: Mientras no alcancemos nuestro objetivo del 95% (1 - alpha)
    while T <= s:
        # 1. Calculamos la probabilidad EXACTA de tener 'T' errores
        # Fórmula binomial: Combinaciones * (prob. éxito) * (prob. fracaso)
        combinaciones = math.comb(s, T)
        prob_exacta = combinaciones * (p0**T) * ((1 - p0)**(s - T))
        
        # 2. Acumulamos esta probabilidad
        probabilidad_acumulada += prob_exacta
        
        # 3. Comprobamos si ya superamos la barrera
        if probabilidad_acumulada >= objetivo_acumulado:
            return T # ¡Hemos encontrado el umbral!
            
        # Si no, probamos con el siguiente número de errores
        T += 1
        
    return s

# --- Probemos el código ---
N_qubits = 200
s = N_qubits // 4   # 50 bits de muestra
p0 = 0.05           # 5% de ruido ambiental
alpha = 0.05        # 5% de tolerancia a falsas alarmas

# Cálculo manual paso a paso
T_manual = calcular_T_manual(s, p0, alpha)

# Cálculo usando la "caja negra" de scipy (Percent Point Function)
T_scipy = int(binom.ppf(1 - alpha, s, p0))

print(f"Buscando cubrir el {(1-alpha)*100}% de los casos sin Eve:")
print(f"Umbral T calculado con bucle manual: {T_manual} errores")
print(f"Umbral T calculado con scipy:        {T_scipy} errores")