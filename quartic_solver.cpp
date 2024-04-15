#include <iostream>
#include <complex>
#include <cmath>
#include <vector>
#include <iomanip> // Include the necessary header for std::setprecision
#include "quartic_solver.h" // Include the header file for the function _get_quartic_solution

using namespace std;

std::complex<double> _get_quartic_solution(std::complex<double> W, int pm1 = +1, int pm2 = +1) {
    auto isclose = [](std::complex<double> a, std::complex<double> b, double rel_tol = 1e-09, double abs_tol = 1e-8) {
        return std::abs(a - b) <= std::max(rel_tol * std::max(std::abs(a), std::abs(b)), abs_tol);
    };

    std::complex<double> x = 0;

    std::complex<double> W_conj = std::conj(W);

    std::complex<double> A = std::pow((1.0 - W * W_conj), 3) / 27.0 + std::pow((W * W - W_conj * W_conj), 2) / 16.0;
    std::complex<double> C = std::pow((2.0 * (W_conj * W_conj - W * W) + 8.0 * std::sqrt(A)), 1.0 / 3.0);

    std::complex<double> u1;
    if (!isclose(C, std::complex<double>(0.0, 0.0))) {
        u1 = C - 4.0 * (1.0 - W * W_conj) / (3.0 * C);
    } else {
        u1 = std::pow((4.0 * (W_conj * W_conj - W * W)), 1.0 / 3.0);
    }

    std::complex<double> term1 = W + static_cast<double>(pm1) * std::sqrt(u1 + W * W);
    std::complex<double> term2 = u1 + static_cast<double>(pm1) * (((2.0 * W_conj + W * u1) / std::sqrt(u1 + W * W)) != 0.0 ? ((2.0 * W_conj + W * u1) / std::sqrt(u1 + W * W)) : std::sqrt(u1 * u1 + 4.0));

    std::complex<double> z = (term1 + static_cast<double>(pm2) * std::sqrt(term1 * term1 - 2.0 * term2)) / 2.0;

    // if magnitude of z is not close to 1 return None
    if (!isclose(std::abs(z), 1.0, 1e-09, 1e-8)) {
        return std::complex<double>(0.0, 0.0);
    }

    return z;
}

std::vector<std::complex<double>> get_ACLE_angular_solns(std::complex<double> W) {
    std::vector<std::complex<double>> solns;

    for (int i = 0; i < 4; i++) {
        int pm1 = (i / 2 == 0) ? +1 : -1;
        int pm2 = (i % 2 == 0) ? +1 : -1;
        std::complex<double> s = _get_quartic_solution(W, pm1, pm2);
        if (s != std::complex<double>(0.0, 0.0)) {
            solns.push_back(s);
        }
    }
    return solns;
}


int main() {
    std::complex<double> W(0.03, 0.1);
    std::vector<std::complex<double>> solutions = get_ACLE_angular_solns(W);

    for (int i = 0; i < solutions.size(); i++) {
        std::cout << "Solution " << i + 1 << ": " << std::fixed << std::setprecision(3) << solutions[i] << std::endl;
    }

    return 0;
}
