import torch

def _get_quartic_solution(W, pm1=+1, pm2=+1):
    def isclose(a, b, rel_tol=1e-09, abs_tol=1e-8):
        return torch.abs(a - b) <= torch.maximum(
            rel_tol * torch.maximum(torch.abs(a), torch.abs(b)), torch.tensor(abs_tol)
        )

    def isnonzero(a):
        return not isclose(a, torch.tensor(0)).item()

    W_conj = W.conj()

    A = (1 - W * W_conj) ** 3 / 27 + (W**2 - W_conj**2) ** 2 / 16
    C = (2 * (W_conj**2 - W**2) + 8 * torch.sqrt(A)) ** (1 / 3)
    if isnonzero(C):
        u1 = C - 4 * (1 - W * W_conj) / (3 * C)
    else:
        u1 = (4 * (W_conj**2 - W**2)) ** (1 / 3)

    term1 = W + pm1 * torch.sqrt(u1 + W**2)
    term2 = u1 + pm1 * (
        ((2 * W_conj + W * u1) / (torch.sqrt(u1 + W**2)))
        if isnonzero(u1 + W**2)
        else torch.sqrt(u1**2 + 4)
    )

    z = (term1 + pm2 * torch.sqrt(term1**2 - 2 * term2)) / 2

    # if magnitude of z is not close to 1 return None
    if not isclose(torch.abs(z), torch.tensor(1), rel_tol=1e-09, abs_tol=1e-8):
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

def solve_quartic(W):
    # convert to torch
    W = torch.tensor(W, dtype=torch.complex128)
    return [_get_quartic_solution(W, pm1, pm2).item() for pm1 in [+1, -1] for pm2 in [+1, -1]]

if __name__ == "__main__":
    W = complex(0.000001, 0.000001)
    solutions = solve_quartic(W)

    for i, sol in enumerate(solutions):
        print(f"Solution {i+1}: {sol:.3f}")