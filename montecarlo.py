import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Params:
    n_bases: int = 3
    p_noise: float = 0.02
    eve_fraction: float = 0.25   # fracción de qubits interceptados por Eve
    n_test: int = 8              # qubits sifted usados para el test
    z_threshold: float = 3.0
    seed: int = 123


def basis_probs(m: int):
    return {
        "P(A=B)": 1 / m,
        "P(A=E)": 1 / m,
        "P(B=E)": 1 / m,
        "P(A=B=E)": 1 / (m * m),
        "P(exactamente dos)": 3 * (m - 1) / (m * m),
        "P(todas distintas)": ((m - 1) * (m - 2)) / (m * m) if m >= 3 else 0.0,
    }


def expected_qber_due_to_eve(m: int, eve_fraction: float) -> float:
    return eve_fraction * (m - 1) / (2 * m)


def simulate_round(n_qubits: int, params: Params, rng: np.random.Generator):
    m = params.n_bases

    alice_bits = rng.integers(0, 2, size=n_qubits)
    alice_bases = rng.integers(0, m, size=n_qubits)
    bob_bases = rng.integers(0, m, size=n_qubits)

    eve_mask = np.zeros(n_qubits, dtype=bool)
    k = int(round(params.eve_fraction * n_qubits))
    k = max(0, min(k, n_qubits))
    if k > 0:
        eve_mask[rng.choice(n_qubits, size=k, replace=False)] = True

    bob_bits = np.empty(n_qubits, dtype=int)

    for i in range(n_qubits):
        a_bit = alice_bits[i]
        a_base = alice_bases[i]
        b_base = bob_bases[i]

        if eve_mask[i]:
            eve_base = rng.integers(0, m)
            eve_bit = a_bit if eve_base == a_base else rng.integers(0, 2)
            bob_bit = eve_bit if b_base == eve_base else rng.integers(0, 2)
        else:
            bob_bit = a_bit if b_base == a_base else rng.integers(0, 2)

        # ruido del canal / sistema
        if rng.random() < params.p_noise:
            bob_bit ^= 1

        bob_bits[i] = bob_bit

    sift_mask = (alice_bases == bob_bases)
    sift_idx = np.where(sift_mask)[0]
    eve_touched_sift = bool(np.any(eve_mask[sift_idx])) if len(sift_idx) else False

    if len(sift_idx) == 0:
        return {
            "n_sift": 0,
            "observed_error_rate": 1.0,
            "threshold": 0.0,
            "eve_touched_sift": eve_touched_sift,
            "eve_detected": False,
            "accepted": False,
        }

    test_size = min(params.n_test, len(sift_idx))
    test_idx = rng.choice(sift_idx, size=test_size, replace=False)
    observed_error_rate = float(np.mean(bob_bits[test_idx] != alice_bits[test_idx]))

    baseline = params.p_noise
    sigma = np.sqrt(max(baseline * (1 - baseline), 1e-12) / test_size)
    threshold = min(1.0, baseline + params.z_threshold * sigma)
    eve_detected = observed_error_rate > threshold

    return {
        "n_sift": len(sift_idx),
        "observed_error_rate": observed_error_rate,
        "threshold": threshold,
        "eve_touched_sift": eve_touched_sift,
        "eve_detected": eve_detected,
        "accepted": not eve_detected,
    }


def run_mc(n_qubits: int, params: Params, trials: int = 5000):
    rng = np.random.default_rng(params.seed + n_qubits + int(1000 * params.eve_fraction))
    accepted = detected = miss = false_pos = 0
    total_sift = 0
    qbers = []

    for _ in range(trials):
        r = simulate_round(n_qubits, params, rng)
        total_sift += r["n_sift"]
        qbers.append(r["observed_error_rate"])

        if r["accepted"]:
            accepted += 1

        if r["eve_touched_sift"]:
            if r["eve_detected"]:
                detected += 1
            else:
                miss += 1
        else:
            if r["eve_detected"]:
                false_pos += 1

    eve_cases = detected + miss
    no_eve_cases = trials - eve_cases

    return {
        "P(acepta)": accepted / trials,
        "P(detec | Eve en clave)": detected / eve_cases if eve_cases else 0.0,
        "P(falla | Eve en clave)": miss / eve_cases if eve_cases else 0.0,
        "P(falso positivo | sin Eve)": false_pos / no_eve_cases if no_eve_cases else 0.0,
        "sift medio": total_sift / trials,
        "QBER medio observado": float(np.mean(qbers)),
    }


def main():
    outdir = Path("bb84_graphs")
    outdir.mkdir(exist_ok=True)

    n_qubits_values = np.array([8, 16, 24, 32, 48, 64, 96, 128, 192, 256])
    eve_fractions = [0.0, 0.125, 0.25, 0.5, 1.0]

    results = {}
    for f in eve_fractions:
        params = Params(n_bases=3, p_noise=0.02, eve_fraction=f, n_test=8, z_threshold=3.0, seed=123)
        rows = []
        for n in n_qubits_values:
            r = run_mc(n, params, trials=4000)
            r["n_qubits"] = n
            rows.append(r)
        results[f] = rows

    # 1) Detección de Eve
    plt.figure(figsize=(9, 6))
    for f, rows in results.items():
        xs = [r["n_qubits"] for r in rows]
        ys = [r["P(detec | Eve en clave)"] for r in rows]
        plt.plot(xs, ys, marker="o", label=f"Eve intercepta {int(f*100)}%")
    plt.title("Probabilidad de detectar a Eve vs número de qubits")
    plt.xlabel("Número de qubits enviados")
    plt.ylabel("P(detectar a Eve)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "deteccion_vs_qubits.png", dpi=160)
    plt.show()

    # 2) Eve pasa desapercibida
    plt.figure(figsize=(9, 6))
    for f, rows in results.items():
        xs = [r["n_qubits"] for r in rows]
        ys = [r["P(falla | Eve en clave)"] for r in rows]
        plt.plot(xs, ys, marker="o", label=f"Eve intercepta {int(f*100)}%")
    plt.title("Probabilidad de que Eve pase desapercibida vs número de qubits")
    plt.xlabel("Número de qubits enviados")
    plt.ylabel("P(Eve no detectada | Eve en clave)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "evasión_vs_qubits.png", dpi=160)
    plt.show()

    # 3) Falsos positivos
    plt.figure(figsize=(9, 6))
    rows = results[0.0]
    xs = [r["n_qubits"] for r in rows]
    ys = [r["P(falso positivo | sin Eve)"] for r in rows]
    plt.plot(xs, ys, marker="o")
    plt.title("Falsos positivos del test vs número de qubits (sin Eve)")
    plt.xlabel("Número de qubits enviados")
    plt.ylabel("P(falso positivo)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(outdir / "falsos_positivos_vs_qubits.png", dpi=160)
    plt.show()

    # 4) QBER
    plt.figure(figsize=(9, 6))
    for f, rows in results.items():
        xs = [r["n_qubits"] for r in rows]
        ys = [r["QBER medio observado"] for r in rows]
        plt.plot(xs, ys, marker="o", label=f"Eve intercepta {int(f*100)}%")
    plt.title("Tasa media de error observada (QBER) vs número de qubits")
    plt.xlabel("Número de qubits enviados")
    plt.ylabel("QBER medio observado")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "qber_vs_qubits.png", dpi=160)
    plt.show()

    # 5) Aceptación del protocolo
    plt.figure(figsize=(9, 6))
    for f, rows in results.items():
        xs = [r["n_qubits"] for r in rows]
        ys = [r["P(acepta)"] for r in rows]
        plt.plot(xs, ys, marker="o", label=f"Eve intercepta {int(f*100)}%")
    plt.title("Probabilidad de aceptar el protocolo vs número de qubits")
    plt.xlabel("Número de qubits enviados")
    plt.ylabel("P(aceptar)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "aceptacion_vs_qubits.png", dpi=160)
    plt.show()

    # 6) Coincidencia de bases
    m2 = basis_probs(2)
    m3 = basis_probs(3)

    labels = list(m3.keys())
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(9, 6))
    vals2 = [m2[k] for k in labels]
    vals3 = [m3[k] for k in labels]

    plt.bar(x - width/2, vals2, width, label="2 bases (BB84)")
    plt.bar(x + width/2, vals3, width, label="3 bases")
    plt.title("Probabilidades teóricas de coincidencia de bases")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel("Probabilidad")
    plt.grid(True, axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "coincidencia_bases.png", dpi=160)
    plt.show()

    print(f"Gráficos guardados en: {outdir.resolve()}")


if __name__ == "__main__":
    main()