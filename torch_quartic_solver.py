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
    if not isclose(torch.abs(z), torch.tensor(1)):
        return None
    
    return z

def ACLE(W) -> list[torch.Tensor]:
    solns = []
    for i in range(4):
        pm1 = [+1, -1][i // 2]
        pm2 = [+1, -1][i % 2]
        s = _get_quartic_solution(W, pm1, pm2)
        if s is not None:
            solns.append(s)
    return solns

def ACLE_dWreal(W, z):
    return z * (z*z - 1.0) / (2.0 * z*z*z - 3.0 * z*z * W + W.conj())

def ACLE_dWimag(W, z):
    i = complex(0.0, 1.0)
    return i * z * (z*z + 1.0) / (2.0 * z*z*z - 3.0 * z*z * W +  W.conj())

def get_solutions_and_derivatives(W):
    solutions = ACLE(W)
    return solutions, [ACLE_dWreal(W, z) for z in solutions], [ACLE_dWimag(W, z) for z in solutions]


def compare_derivatives(x, y, print_values=False):
    x = torch.tensor(x, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(y, dtype=torch.float64, requires_grad=True)
    W = x + y * 1j
    solutions = ACLE(W)
    
    for i, z in enumerate(solutions):
        # Assuming z is complex and ACLE returns separate real and imag parts
        z_real = z.real
        z_imag = z.imag

        # Compute gradients wrt real and imaginary parts
        grad_real_x = torch.autograd.grad(z_real, x, retain_graph=True, create_graph=True)
        grad_imag_x = torch.autograd.grad(z_imag, x, retain_graph=True, create_graph=True)
        grad_real_y = torch.autograd.grad(z_real, y, retain_graph=True, create_graph=True)
        grad_imag_y = torch.autograd.grad(z_imag, y, retain_graph=True, create_graph=True)

        auto_derivative_real = grad_real_x[0].item() + 1j * grad_imag_x[0].item()  # Gradient of z wrt x
        auto_derivative_imag = grad_real_y[0].item() + 1j * grad_imag_y[0].item()  # Gradient of z wrt y


        manual_derivative_real = ACLE_dWreal(W, z).item()
        manual_derivative_imag = ACLE_dWimag(W, z).item()

        ACLE_val = z**4 - 2*W*z**3 + 2*W.conj()*z - 1

        if print_values:
            print(f"Solution {i+1}: {z.item():.3f}")
            print(f"Auto Derivative Real: {auto_derivative_real}")
            print(f"Auto Derivative Imaginary: {auto_derivative_imag}")
            print(f"Manual Derivative Real: {manual_derivative_real}")
            print(f"Manual Derivative Imaginary: {manual_derivative_imag}")
        else:
            assert torch.isclose(ACLE_val, torch.tensor(0.0, dtype=torch.complex128), atol=1e-8)
            assert torch.isclose(torch.tensor(auto_derivative_real), torch.tensor(manual_derivative_real), atol=1e-8)
            assert torch.isclose(torch.tensor(auto_derivative_imag), torch.tensor(manual_derivative_imag), atol=1e-8)


if __name__ == "__main__":
    x = 0.1
    y = 0.2
    # W = torch.tensor(complex(x, y), dtype=torch.complex128, requires_grad=True)
    # solutions = ACLE(W)

    # for i, sol in enumerate(solutions):
    #     print(f"Solution {i+1}: {sol:.3f}")

    
    compare_derivatives(x, y)

    solns, dWreal, dWimag = get_solutions_and_derivatives(torch.tensor(complex(x, y), dtype=torch.complex128))
    for i, sol in enumerate(solns):
        print(f"Solution {i+1}: {sol:.3f}")
        print(f"dWreal: {dWreal[i]}")
        print(f"dWimag: {dWimag[i]}")
        print()