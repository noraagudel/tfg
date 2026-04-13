
"""
Simulador BB84 (Monte Carlo)

Modelo:
- Alice envía n qubits, bits y bases aleatorias.
- Bob mide en bases aleatorias.
- Eve puede interceptar una fracción o un número esperado de qubits con intercept-resend.
- Se añade ruido de canal con probabilidad p_noise.
- Alice y Bob comparan un subconjunto de los bits con bases compatibles (sifting + test).

Salidas:
- Probabilidad de comunicación realmente segura.
- Probabilidad de que el cálculo/diagnóstico coincida con el "oráculo" que vería todo el sifted key.
- Probabilidad de detectar a Eve.
- Probabilidad de distinguir ruido/fallos frente a Eve.
- Probabilidad de que Eve pase desapercibida y robe información.

Además:
- Genera 3 gráficas (las tres primeras probabilidades) en función del número de qubits comparados.
- Exporta PNG y CSV con los resultados.

Requisitos:
    numpy, scipy, matplotlib, pandas
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import binom


@dataclass
class BB84Config:
    total_qubits_list: Tuple[int, ...] = (8, 20, 50, 80, 100)
    max_test_bits: int = 100
    p_noise: float = 0.02
    eve_intercepts: int | None = None  # si es None, se usa eve_fraction * n
    eve_fraction: float = 0.25         # fracción esperada de qubits interceptados por Eve
    alpha: float = 0.05                # nivel de significancia para el test binomial
    n_trials: int = 5000
    seed: int = 12345
    output_dir: str = "bb84_outputs"


def critical_value(m: int, p0: float, alpha: float) -> int:
    """
    Devuelve el menor k tal que P(X >= k | Bin(m, p0)) <= alpha.
    Si p0=0, cualquier error es evidencia de algo anómalo.
    """
    if m <= 0:
        return 1
    if p0 <= 0:
        return 1
    if p0 >= 1:
        return m + 1

    for k in range(m + 1):
        tail = binom.sf(k - 1, m, p0)  # P(X >= k)
        if tail <= alpha:
            return k
    return m + 1


def precompute_thresholds(max_m: int, p0: float, alpha: float) -> np.ndarray:
    return np.array([critical_value(m, p0, alpha) for m in range(max_m + 1)], dtype=int)


def simulate_trial(
    n: int,
    eve_intercepts: int,
    p_noise: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, int, bool]:
    """
    Simula una ronda BB84:
    - devuelve errores del sifted key (0/1 por bit)
    - cuántos bits sifted conoce Eve exactamente
    - si Eve ha interceptado al menos un qubit
    """
    alice_bits = rng.integers(0, 2, size=n, dtype=np.int8)
    alice_bases = rng.integers(0, 2, size=n, dtype=np.int8)
    bob_bases = rng.integers(0, 2, size=n, dtype=np.int8)

    eve_mask = np.zeros(n, dtype=bool)
    if eve_intercepts > 0:
        k = min(eve_intercepts, n)
        idx = rng.choice(n, size=k, replace=False)
        eve_mask[idx] = True

    eve_bases = rng.integers(0, 2, size=n, dtype=np.int8)

    # Eve mide y reenvía
    eve_bits = np.empty(n, dtype=np.int8)
    eve_same_as_alice = eve_bases == alice_bases
    eve_bits[eve_same_as_alice] = alice_bits[eve_same_as_alice]
    if np.any(~eve_same_as_alice):
        eve_bits[~eve_same_as_alice] = rng.integers(
            0, 2, size=np.count_nonzero(~eve_same_as_alice), dtype=np.int8
        )

    # Bob mide
    bob_bits = np.empty(n, dtype=np.int8)

    no_eve = ~eve_mask
    bob_same_alice = bob_bases == alice_bases
    if np.any(no_eve & bob_same_alice):
        bob_bits[no_eve & bob_same_alice] = alice_bits[no_eve & bob_same_alice]
    if np.any(no_eve & ~bob_same_alice):
        bob_bits[no_eve & ~bob_same_alice] = rng.integers(
            0, 2, size=np.count_nonzero(no_eve & ~bob_same_alice), dtype=np.int8
        )

    bob_same_eve = bob_bases == eve_bases
    if np.any(eve_mask & bob_same_eve):
        bob_bits[eve_mask & bob_same_eve] = eve_bits[eve_mask & bob_same_eve]
    if np.any(eve_mask & ~bob_same_eve):
        bob_bits[eve_mask & ~bob_same_eve] = rng.integers(
            0, 2, size=np.count_nonzero(eve_mask & ~bob_same_eve), dtype=np.int8
        )

    # Ruido del canal / del sistema
    if p_noise > 0:
        noise = rng.random(n) < p_noise
        bob_bits[noise] ^= 1

    # Sifting
    sift_mask = alice_bases == bob_bases
    sifted_errors = (alice_bits[sift_mask] != bob_bits[sift_mask]).astype(np.int8)

    # Información que conoce Eve sobre bits sifted
    eve_knows_sifted = int(np.sum(
        eve_mask[sift_mask] & (eve_bases[sift_mask] == alice_bases[sift_mask])
    ))

    return sifted_errors, eve_knows_sifted, bool(np.any(eve_mask))


def run_experiment_for_n(
    n: int,
    cfg: BB84Config,
    thresholds: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con probabilidades para m = 1..min(n, max_test_bits).
    """
    max_m = min(n, cfg.max_test_bits)
    counts = {
        "secure_communication": np.zeros(max_m, dtype=np.int64),
        "calculation_correct": np.zeros(max_m, dtype=np.int64),
        "detect_eve": np.zeros(max_m, dtype=np.int64),
        "distinguish_noise_vs_eve": np.zeros(max_m, dtype=np.int64),
        "eve_undetected_and_steals": np.zeros(max_m, dtype=np.int64),
        "runs_with_eve": 0,
    }

    for _ in range(cfg.n_trials):
        eve_intercepts = cfg.eve_intercepts
        if eve_intercepts is None:
            eve_intercepts = int(round(cfg.eve_fraction * n))
        eve_intercepts = max(0, min(eve_intercepts, n))

        sifted_errors, eve_known, eve_present = simulate_trial(
            n=n,
            eve_intercepts=eve_intercepts,
            p_noise=cfg.p_noise,
            rng=rng,
        )

        L = len(sifted_errors)
        if eve_present:
            counts["runs_with_eve"] += 1

        if L == 0:
            # No hay bases compatibles -> no se puede evaluar nada de forma útil.
            continue

        perm = rng.permutation(L)
        errs = sifted_errors[perm]
        cum_errs = np.cumsum(errs)

        oracle_crit = thresholds[L] if L < len(thresholds) else critical_value(L, cfg.p_noise, cfg.alpha)
        oracle_decision = cum_errs[-1] >= oracle_crit

        for m in range(1, max_m + 1):
            mm = min(m, L)
            sample_errs = int(cum_errs[mm - 1])
            sample_decision = sample_errs >= thresholds[mm]

            # 1) Comunicación realmente segura:
            #    el protocolo acepta, el error observado no supera la tolerancia
            #    y Eve no conoce ningún bit sifted.
            if not sample_decision: # el test de errores no detecta problema.
                counts["secure_communication"][m - 1] += 1

            # 2) El cálculo/diagnóstico coincide con el oráculo que ve todo el sifted key.
            if sample_decision == oracle_decision:
                counts["calculation_correct"][m - 1] += 1

            # 3) Detectar Eve: cuando Eve está presente y el test la acusa.
            if eve_present and sample_decision:
                counts["detect_eve"][m - 1] += 1

            # 4) Distinguir ruido/fallos vs Eve: acierta si su etiqueta binaria coincide
            #    con la presencia real de Eve.
            if sample_decision == eve_present:
                counts["distinguish_noise_vs_eve"][m - 1] += 1

            # 5) Eve no detectada y además conoce al menos un bit sifted.
            if eve_present and (not sample_decision) and (eve_known > 0):
                counts["eve_undetected_and_steals"][m - 1] += 1

    denom = cfg.n_trials
    df = pd.DataFrame({
        "compared_bits": np.arange(1, max_m + 1),
        "total_qubits": n,
        "secure_communication": counts["secure_communication"] / denom,
        "calculation_correct": counts["calculation_correct"] / denom,
        "detect_eve": counts["detect_eve"] / denom,
        "distinguish_noise_vs_eve": counts["distinguish_noise_vs_eve"] / denom,
        "eve_undetected_and_steals": counts["eve_undetected_and_steals"] / denom,
    })
    return df


def run_all(cfg: BB84Config) -> pd.DataFrame:
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    thresholds = precompute_thresholds(cfg.max_test_bits, cfg.p_noise, cfg.alpha)

    frames = []
    for n in cfg.total_qubits_list:
        frames.append(run_experiment_for_n(n, cfg, thresholds, rng))

    results = pd.concat(frames, ignore_index=True)
    results.to_csv(os.path.join(cfg.output_dir, "bb84_results.csv"), index=False)
    return results


def plot_metric(results: pd.DataFrame, metric: str, title: str, filename: str, output_dir: str) -> str:
    plt.figure(figsize=(10, 6))
    for n in sorted(results["total_qubits"].unique()):
        sub = results[results["total_qubits"] == n]
        plt.plot(sub["compared_bits"], sub[metric], marker="o", linewidth=1.6, markersize=3, label=f"n={n}")
    plt.xlabel("Número de qubits comparados (x)")
    plt.ylabel("Probabilidad")
    plt.title(title)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()
    return path


def main() -> None:
    cfg = BB84Config(
        total_qubits_list=(8, 20, 50, 80, 100),
        max_test_bits=100,
        p_noise=0.02,
        eve_intercepts=None,  # usa eve_fraction * n; pon un entero si quieres un valor fijo
        eve_fraction=0.25,    # 0.0 = sin Eve, 1.0 = intercepta todos los qubits
        alpha=0.05,
        n_trials=5000,
        seed=12345,
        output_dir="bb84_outputs",
    )
    results = run_all(cfg)

    os.makedirs(cfg.output_dir, exist_ok=True)
    p1 = plot_metric(
        results,
        "secure_communication",
        "Probabilidad de comunicación realmente segura",
        "01_prob_segura.png",
        cfg.output_dir,
    )
    p2 = plot_metric(
        results,
        "calculation_correct",
        "Probabilidad de que el cálculo diagnóstico sea correcto",
        "02_prob_calculo_correcto.png",
        cfg.output_dir,
    )
    p3 = plot_metric(
        results,
        "detect_eve",
        "Probabilidad de detectar a Eve",
        "03_prob_detectar_eve.png",
        cfg.output_dir,
    )

    # Guarda también las dos probabilidades extra.
    plot_metric(
        results,
        "distinguish_noise_vs_eve",
        "Probabilidad de distinguir ruido/fallos vs Eve",
        "04_prob_distinguir_ruido_vs_eve.png",
        cfg.output_dir,
    )
    plot_metric(
        results,
        "eve_undetected_and_steals",
        "Probabilidad de que Eve pase desapercibida y robe info",
        "05_prob_eve_indetectada.png",
        cfg.output_dir,
    )

    print("Resultados guardados en:", cfg.output_dir)
    print("CSV:", os.path.join(cfg.output_dir, "bb84_results.csv"))
    print("Gráficas:")
    print(" -", p1)
    print(" -", p2)
    print(" -", p3)


if __name__ == "__main__":
    main()
