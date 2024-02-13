import numpy as np

def _get_quartic_solution(W, pm1=+1, pm2=+1):
    def isclose(a, b, rel_tol=1e-09, abs_tol=1e-8):
        return np.abs(a - b) <= np.maximum(
            rel_tol * np.maximum(np.abs(a), np.abs(b)), np.array(abs_tol)
        )

    def isnonzero(a):
        return not isclose(a, np.array(0)).item()

    W_conj = np.conj(W)

    A = (1 - W * W_conj) ** 3 / 27 + (W**2 - W_conj**2) ** 2 / 16
    C = (2 * (W_conj**2 - W**2) + 8 * np.sqrt(A)) ** (1 / 3)
    if isnonzero(C):
        u1 = C - 4 * (1 - W * W_conj) / (3 * C)
    else:
        u1 = (4 * (W_conj**2 - W**2)) ** (1 / 3)

    term1 = W + pm1 * np.sqrt(u1 + W**2)
    term2 = u1 + pm1 * (
        ((2 * W_conj + W * u1) / (np.sqrt(u1 + W**2)))
        if isnonzero(u1 + W**2)
        else np.sqrt(u1**2 + 4)
    )

    z = (term1 + pm2 * np.sqrt(term1**2 - 2 * term2)) / 2

    # if magnitude of z is not close to 1 return None
    if not isclose(np.abs(z), np.array(1), rel_tol=1e-09, abs_tol=1e-8):
        return None

    return z


def get_ACLE_angular_solns(W):
    solns = set()
    
    for i in range(4):
        # TODO can be sped by restructuring loop
        pm1 = [+1, -1][i // 2]
        pm2 = [+1, -1][i % 2]
        s = _get_quartic_solution(W, pm1, pm2)
        if s is not None:
            solns.add(s)
    return solns


if __name__ == "__main__":
    W = complex(0.03, 0.1)
    solutions = get_ACLE_angular_solns(W)

    for i, sol in enumerate(solutions):
        print(f"Solution {i+1}: {sol:.3f}")