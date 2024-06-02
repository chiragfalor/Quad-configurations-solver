#include "QuarticSolver.hpp"

complex<double> _get_quartic_solution(complex<double> W, int pm1, int pm2) {
    auto isclose = [](complex<double> a, complex<double> b, double rel_tol = 1e-09, double abs_tol = 1e-8) {
        return abs(a - b) <= std::max(rel_tol * std::max(abs(a), abs(b)), abs_tol);
    };

    complex<double> W_conj = conj(W);

    complex<double> A = pow((1.0 - W * W_conj), 3) / 27.0 + pow((W * W - W_conj * W_conj), 2) / 16.0;
    complex<double> C = pow((2.0 * (W_conj * W_conj - W * W) + 8.0 * sqrt(A)), 1.0 / 3.0);

    complex<double> u1;
    if (!isclose(C, complex<double>(0.0, 0.0))) {
        u1 = C - 4.0 * (1.0 - W * W_conj) / (3.0 * C);
    } else {
        u1 = pow((4.0 * (W_conj * W_conj - W * W)), 1.0 / 3.0);
    }

    complex<double> term1 = W + static_cast<double>(pm1) * sqrt(u1 + W * W);
    complex<double> term2 = u1 + static_cast<double>(pm1) * (((2.0 * W_conj + W * u1) / sqrt(u1 + W * W)) != 0.0 ? ((2.0 * W_conj + W * u1) / sqrt(u1 + W * W)) : sqrt(u1 * u1 + 4.0));

    complex<double> z = (term1 + static_cast<double>(pm2) * sqrt(term1 * term1 - 2.0 * term2)) / 2.0;

    // if magnitude of z is not close to 1 return None
    if (!isclose(abs(z), 1.0, 1e-09, 1e-8)) {
        return complex<double>(0.0, 0.0);
    }

    return z;
}

vector<complex<double>> ACLE(complex<double> W) {
    vector<complex<double>> solns;
    for (int i = 0; i < 4; i++) {
        int pm1 = (i / 2 == 0) ? +1 : -1;
        int pm2 = (i % 2 == 0) ? +1 : -1;
        complex<double> s = _get_quartic_solution(W, pm1, pm2);
        if (s != complex<double>(0.0, 0.0)) {
            solns.push_back(s);
        }
    }
    return solns;
}

complex<double> ACLE_dWreal(complex<double> W, complex<double> z) {
    return z*(z*z - 1.0) / (2.0*z*z*z - 3.0*z*z*W + conj(W));
}

complex<double> ACLE_dWimag(complex<double> W, complex<double> z) {
    complex<double> i = complex<double>(0.0, 1.0);
    return i*z*(z*z + 1.0) / (2.0*z*z*z - 3.0*z*z*W + conj(W));
}

vector<Point> ACLE_dual(Point W) {
    vector<Point> solns;
    real Wx = W.x, Wy = W.y;
    complex<double> W_complex = complex<double>(Wx.val(), Wy.val());
    vector<complex<double>> solns_complex = ACLE(W_complex);

    for (int i = 0; i < solns_complex.size(); i++) {
        complex<double> z = solns_complex[i];
        Point soln = Point(z.real(), z.imag());

        if(Wx[1] != 0.0) {
            soln.x[1] = Wx[1] * ACLE_dWreal(W_complex, z).real(); 
            soln.y[1] = Wx[1] * ACLE_dWreal(W_complex, z).imag();

        }

        if(Wy[1] != 0.0) {
            soln.x[1] += Wy[1] * ACLE_dWimag(W_complex, z).real();
            soln.y[1] += Wy[1] * ACLE_dWimag(W_complex, z).imag();
        }

        solns.push_back(soln);
    }

    return solns;
}

