
"""
BB84 Monte Carlo simulator

Este script modela BB84 con:
- elección aleatoria de bases por Alice y Bob,
- Eve con ataque intercept-resend sobre una fracción de qubits,
- ruido del canal como bit-flip con probabilidad configurable,
- test público sobre una muestra aleatoria de los bits sifted,
- cálculo de cinco probabilidades y exportación de tres gráficas.

Idea del modelo:
1) Alice prepara bits y bases al azar.
2) Eve, si interviene, mide en una base aleatoria y reenvía el qubit.
3) Bob mide en una base aleatoria.
4) Alice y Bob conservan solo los qubits donde sus bases coinciden.
5) De esos sifted bits, revelan una muestra pública de tamaño x.
6) A partir del error observado en esa muestra, deciden si hay Eve.

Notas:
- El núcleo es una simulación clásica de Monte Carlo porque, para BB84,
  las probabilidades de acierto/fallo tras elegir bases se pueden modelar
  directamente.
- Qiskit puede usarse para simular circuitos individuales, pero no es
  necesario para estudiar estas probabilidades y sería bastante más lento.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Configuración del experimento
# -----------------------------

TEST_QUBITS = [8, 20, 50, 80, 100]  # total de qubits enviados en cada experimento
TRIALS_PER_POINT = 5000             # súbelo a 10000 si quieres curvas más suaves
NOISE_RATE = 0.03                  # ruido sistemático del canal
DETECTION_THRESHOLD = 0.10         # umbral sobre el error observado para declarar Eve

# Eve presente en qué proporción de las pruebas.
# 0.0 = nunca, 1.0 = siempre, 0.5 = mitad de las veces, etc.
P_EVE_PRESENT = 0.50

# Si Eve está presente, intercepta cada qubit con esta probabilidad.
# 0.0 = no intercepta, 0.25 = algunos, 1.0 = todos.
EVE_INTERCEPT_PROB = 0.50

# Semilla para reproducibilidad.
SEED = 2026


@dataclass
class TrialData:
    """Datos generados una sola vez para un número fijo de qubits totales."""

    alice_bits: np.ndarray          # (trials, n)
    alice_bases: np.ndarray         # (trials, n)
    bob_bases: np.ndarray           # (trials, n)
    bob_bits: np.ndarray            # (trials, n)
    sifted_mask: np.ndarray          # (trials, n)
    mismatch_mask: np.ndarray        # (trials, n)
    sifted_count: np.ndarray        # (trials,)
    total_sifted_errors: np.ndarray  # (trials,)
    eve_present: np.ndarray         # (trials,)
    true_sifted_error_rate: np.ndarray  # (trials,)


def _simulate_channel(
    n_qubits: int,
    trials: int,
    noise_rate: float,
    p_eve_present: float,
    eve_intercept_prob: float,
    rng: np.random.Generator,
) -> TrialData:
    """
    Genera un conjunto de pruebas BB84 completo para un número fijo de qubits.

    Devuelve las bases y bits de Alice/Bob ya simulados, de forma vectorizada.
    """

    # Bits y bases aleatorias.
    alice_bits = rng.integers(0, 2, size=(trials, n_qubits), dtype=np.int8)
    alice_bases = rng.integers(0, 2, size=(trials, n_qubits), dtype=np.int8)
    bob_bases = rng.integers(0, 2, size=(trials, n_qubits), dtype=np.int8)

    # Eve presente o no en cada prueba.
    eve_present = rng.random(trials) < p_eve_present

    # Eve intercepta cada qubit con probabilidad independiente.
    intercepted = (rng.random((trials, n_qubits)) < eve_intercept_prob) & eve_present[:, None]

    # Estado que llega a Bob tras la posible acción de Eve.
    forwarded_bits = alice_bits.copy()
    forwarded_bases = alice_bases.copy()

    if np.any(intercepted):
        eve_bases = rng.integers(0, 2, size=(trials, n_qubits), dtype=np.int8)

        # Si Eve usa la misma base que Alice, reenvía el bit correcto.
        same_basis = intercepted & (eve_bases == alice_bases)

        # Si Eve usa base distinta, su resultado es aleatorio.
        diff_basis = intercepted & (eve_bases != alice_bases)

        forwarded_bits[same_basis] = alice_bits[same_basis]
        forwarded_bits[diff_basis] = rng.integers(0, 2, size=np.count_nonzero(diff_basis), dtype=np.int8)

        forwarded_bases[intercepted] = eve_bases[intercepted]

    # Bob mide:
    # - si su base coincide con la que llega, obtiene el bit enviado;
    # - si no coincide, el resultado es aleatorio.
    bob_bits = np.empty((trials, n_qubits), dtype=np.int8)
    same_as_forwarded = bob_bases == forwarded_bases
    bob_bits[same_as_forwarded] = forwarded_bits[same_as_forwarded]
    bob_bits[~same_as_forwarded] = rng.integers(0, 2, size=np.count_nonzero(~same_as_forwarded), dtype=np.int8)

    # Ruido del canal: bit-flip sobre el resultado de Bob.
    noise_mask = rng.random((trials, n_qubits)) < noise_rate
    bob_bits = np.bitwise_xor(bob_bits, noise_mask.astype(np.int8))

    # Sifting: solo se conservan los qubits con bases coincidentes.
    sifted_mask = alice_bases == bob_bases
    mismatch_mask = sifted_mask & (alice_bits != bob_bits)

    sifted_count = sifted_mask.sum(axis=1)
    total_sifted_errors = mismatch_mask.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        true_sifted_error_rate = np.where(
            sifted_count > 0,
            total_sifted_errors / sifted_count,
            0.0,
        )

    return TrialData(
        alice_bits=alice_bits,
        alice_bases=alice_bases,
        bob_bases=bob_bases,
        bob_bits=bob_bits,
        sifted_mask=sifted_mask,
        mismatch_mask=mismatch_mask,
        sifted_count=sifted_count,
        total_sifted_errors=total_sifted_errors,
        eve_present=eve_present,
        true_sifted_error_rate=true_sifted_error_rate,
    )


def _sample_prefix_errors_for_x(
    shuffled_sifted_errors: np.ndarray,
    x: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Devuelve el número de errores observados en la muestra pública de tamaño x
    para cada prueba. La muestra se elige al azar sin reemplazo.

    Parámetro:
        shuffled_sifted_errors:
            Array (trials, n) con errores 0/1 ordenados de forma aleatoria
            dentro de los qubits sifted, y 0 en el resto.
    """
    # Este helper no se usa en la versión final, pero se deja aquí por claridad.
    raise NotImplementedError


def evaluate_for_n(
    n_qubits: int,
    test_sizes: Iterable[int],
    trials: int,
    noise_rate: float,
    p_eve_present: float,
    eve_intercept_prob: float,
    detection_threshold: float,
    rng: np.random.Generator,
) -> Dict[int, Dict[str, float]]:
    """
    Simula BB84 para un número fijo de qubits totales y devuelve las métricas
    para distintos tamaños de muestra pública x.
    """

    data = _simulate_channel(
        n_qubits=n_qubits,
        trials=trials,
        noise_rate=noise_rate,
        p_eve_present=p_eve_present,
        eve_intercept_prob=eve_intercept_prob,
        rng=rng,
    )

    # Orden aleatorio por prueba para muestrear los sifted bits sin reemplazo.
    # Las posiciones no sifted se envían al final usando +inf.
    random_priority = rng.random((trials, n_qubits))
    priority = np.where(data.sifted_mask, random_priority, np.inf)
    order = np.argsort(priority, axis=1)

    ordered_sifted = np.take_along_axis(data.sifted_mask.astype(np.int8), order, axis=1)
    ordered_errors = np.take_along_axis(data.mismatch_mask.astype(np.int8), order, axis=1)

    # Solo nos interesan los sifted bits; los no sifted quedan con 0 en errores.
    ordered_errors = ordered_errors * ordered_sifted

    cum_sifted = np.cumsum(ordered_sifted, axis=1)
    cum_errors = np.cumsum(ordered_errors, axis=1)

    results: Dict[int, Dict[str, float]] = {}

    for x in test_sizes:
        valid = data.sifted_count >= x
        valid_count = int(np.count_nonzero(valid))
        if valid_count == 0:
            results[x] = {
                "real_secure": np.nan,
                "decision_correct": np.nan,
                "detect_eve": np.nan,
                "distinguish_noise_vs_eve": np.nan,
                "eve_undetected": np.nan,
                "mean_true_error": np.nan,
                "mean_test_error": np.nan,
                "valid_trials": 0,
            }
            continue

        # Índice donde aparece el x-ésimo sifted bit en el orden aleatorio.
        # cum_sifted >= x es True desde ese punto en adelante.
        idx = np.argmax(cum_sifted >= x, axis=1)
        test_errors = cum_errors[np.arange(trials), idx]
        observed_error_rate = test_errors / x

        decision_eve = observed_error_rate > detection_threshold
        actual_eve = data.eve_present

        protocol_accepts = ~decision_eve
        true_secure = (~actual_eve) & (data.true_sifted_error_rate <= detection_threshold)

        # Métricas condicionadas a tener suficiente material sifted para testear.
        sel = valid

        real_secure = np.mean((true_secure & protocol_accepts)[sel])
        decision_correct = np.mean((decision_eve == actual_eve)[sel])
        detect_eve = np.mean((actual_eve & decision_eve)[sel])
        distinguish_noise_vs_eve = np.mean(((~actual_eve) & protocol_accepts)[sel])
        eve_undetected = np.mean((actual_eve & protocol_accepts)[sel])

        results[x] = {
            "real_secure": float(real_secure),
            "decision_correct": float(decision_correct),
            "detect_eve": float(detect_eve),
            "distinguish_noise_vs_eve": float(distinguish_noise_vs_eve),
            "eve_undetected": float(eve_undetected),
            "mean_true_error": float(np.mean(data.true_sifted_error_rate[sel])),
            "mean_test_error": float(np.mean(observed_error_rate[sel])),
            "valid_trials": valid_count,
        }

    return results


def plot_metric(
    all_results: Dict[int, Dict[int, Dict[str, float]]],
    metric: str,
    title: str,
    ylabel: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(10, 6))

    for n_qubits, per_x in all_results.items():
        xs = sorted(per_x.keys())
        ys = [per_x[x][metric] for x in xs]
        plt.plot(xs, ys, marker="o", linewidth=2, label=f"{n_qubits} qubits totales")

    plt.title(title)
    plt.xlabel("Qubits públicos comparados (x)")
    plt.ylabel(ylabel)
    plt.ylim(0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_results_csv(all_results: Dict[int, Dict[int, Dict[str, float]]], output_path: Path) -> None:
    lines = [
        "n_total,x,real_secure,decision_correct,detect_eve,distinguish_noise_vs_eve,eve_undetected,mean_true_error,mean_test_error,valid_trials"
    ]
    for n_qubits, per_x in all_results.items():
        for x, values in per_x.items():
            lines.append(
                f"{n_qubits},{x},"
                f"{values['real_secure']:.6f},"
                f"{values['decision_correct']:.6f},"
                f"{values['detect_eve']:.6f},"
                f"{values['distinguish_noise_vs_eve']:.6f},"
                f"{values['eve_undetected']:.6f},"
                f"{values['mean_true_error']:.6f},"
                f"{values['mean_test_error']:.6f},"
                f"{values['valid_trials']}"
            )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rng = np.random.default_rng(SEED)

    all_results: Dict[int, Dict[int, Dict[str, float]]] = {}

    for n_qubits in TEST_QUBITS:
        # En BB84 solo se comparan los qubits con bases coincidentes.
        # En media eso es ~n/2, así que limitamos x a ese rango para que tenga sentido.
        max_public_tests = max(1, n_qubits // 2)
        test_sizes = range(1, max_public_tests + 1)
        all_results[n_qubits] = evaluate_for_n(
            n_qubits=n_qubits,
            test_sizes=test_sizes,
            trials=TRIALS_PER_POINT,
            noise_rate=NOISE_RATE,
            p_eve_present=P_EVE_PRESENT,
            eve_intercept_prob=EVE_INTERCEPT_PROB,
            detection_threshold=DETECTION_THRESHOLD,
            rng=rng,
        )

    out_dir = Path(__file__).resolve().parent
    save_results_csv(all_results, out_dir / "bb84_results.csv")

    plot_metric(
        all_results,
        metric="real_secure",
        title="BB84: probabilidad de comunicación realmente segura",
        ylabel="Probabilidad",
        output_path=out_dir / "bb84_prob_comunicacion_realmente_segura.png",
    )
    plot_metric(
        all_results,
        metric="decision_correct",
        title="BB84: probabilidad de que la decisión sobre Eve sea correcta",
        ylabel="Probabilidad",
        output_path=out_dir / "bb84_prob_decision_correcta.png",
    )
    plot_metric(
        all_results,
        metric="detect_eve",
        title="BB84: probabilidad de detectar a Eve",
        ylabel="Probabilidad",
        output_path=out_dir / "bb84_prob_detectar_eve.png",
    )

    # Resumen en consola para el caso de 100 qubits y 100 pruebas públicas.
    ref_n = max(TEST_QUBITS)
    ref_x = max(1, ref_n // 2)
    ref = all_results[ref_n][ref_x]
    print(f"\nResumen final (n_total={ref_n}, x={ref_x}):")
    for k, v in ref.items():
        if k == "valid_trials":
            print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v:.4f}")

    print("\nArchivos generados:")
    print(f"  - {out_dir / 'bb84_results.csv'}")
    print(f"  - {out_dir / 'bb84_prob_comunicacion_realmente_segura.png'}")
    print(f"  - {out_dir / 'bb84_prob_decision_correcta.png'}")
    print(f"  - {out_dir / 'bb84_prob_detectar_eve.png'}")


if __name__ == "__main__":
    main()
