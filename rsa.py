
from typing import List, Tuple

def egcd(a: int, b: int) -> Tuple[int,int,int]:
    """Extended GCD: devuelve (g, x, y) con g = gcd(a,b) y ax + by = g."""
    if b == 0:
        return (a, 1, 0)
    else:
        g, x1, y1 = egcd(b, a % b)
        return (g, y1, x1 - (a // b) * y1)

def modinv(a: int, m: int) -> int:
    """Inverse modular de a mod m (si existe)."""
    g, x, _ = egcd(a, m)
    if g != 1:
        raise ValueError(f"No existe inverso modular de {a} mod {m} (gcd={g})")
    return x % m

def to_ascii(message: str) -> List[int]:
    """
    1) Convierte el mensaje a una lista de códigos ASCII.
    Imprime y devuelve la lista de enteros.
    """
    ascii_codes = [ord(ch) for ch in message]
    print("Paso 1 — Mensaje a ASCII:")
    print("Mensaje:", message)
    print("ASCII codes:", ascii_codes)
    return ascii_codes

def encrypt_ascii(ascii_codes: List[int], p: int, q: int, e: int) -> Tuple[List[int], int, int]:
    """
    2) Calcula N = p*q, phi = (p-1)(q-1), comprueba/obtiene d tal que e*d ≡ 1 (mod phi),
       cifra cada código ASCII con C = M^e mod N.
    Imprime e, d, N y la lista de cifrados. Devuelve (cipher_list, d, N).
    """
    N = p * q
    phi = (p - 1) * (q - 1)
    # Calculamos d (inverso modular de e mod phi)
    d = modinv(e, phi)
    # Comprobación (opcional)
    if (e * d) % phi != 1:
        raise RuntimeError("e * d mod phi no es 1 — parámetros inválidos")
    # Ciframos por-byte
    cipher = [pow(m, e, N) for m in ascii_codes]
    print("\nPaso 2 — Encriptación RSA:")
    print(f"p = {p}, q = {q}")
    print(f"N = p * q = {N}")
    print(f"phi = (p-1)*(q-1) = {phi}")
    print(f"e = {e}")
    print(f"d (inverso de e mod phi) = {d}")
    print("Cipher (lista de enteros):", cipher)
    return cipher, d, N

def decrypt_cipher(cipher: List[int], d: int, N: int) -> str:
    """
    3) Descifra cada bloque con M = C^d mod N y convierte los códigos a caracteres.
    Imprime la lista de códigos recuperados y el mensaje final.
    """
    decrypted_codes = [pow(c, d, N) for c in cipher]
    # Convertir a string (si los códigos resultantes están en rango válido)
    try:
        message = ''.join(chr(m) for m in decrypted_codes)
    except ValueError:
        # Si algún m está fuera del rango, devolvemos una representación
        message = None
    print("\nPaso 3 — Descifrado RSA:")
    print("Códigos descifrados:", decrypted_codes)
    if message is not None:
        print("Mensaje recuperado:", message)
        return message
    else:
        raise ValueError("Algunos códigos descifrados no son caracteres válidos Unicode.")


if __name__ == "__main__":
    # Ejemplo: p=61, q=53, e=17 => d=2753 (phi=3120)
    p = 61
    q = 53
    e = 17

    mensaje = "Hello World"   # mensaje de ejemplo
    ascii_codes = to_ascii(mensaje)
    cipher, d, N = encrypt_ascii(ascii_codes, p, q, e)
    recovered = decrypt_cipher(cipher, d, N)