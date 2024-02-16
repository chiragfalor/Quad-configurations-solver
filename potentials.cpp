#include <iostream>
#include <cmath>
#include <complex>
#include <vector>
#include <unordered_map>
#include <corecrt_math_defines.h>

// #include "quartic_solver.h"

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

class SIEP_plus_XS {
private:
    double b, eps, gamma, x_g, y_g, eps_theta, gamma_theta;

public:
    SIEP_plus_XS(double b = 1.0, double eps = 0.0, double gamma = 0.0, double x_g = 0.0, double y_g = 0.0, double eps_theta = 0.0, double gamma_theta = 0.0) {
        this->b = b;
        this->eps = eps;
        this->gamma = gamma;
        this->x_g = x_g;
        this->y_g = y_g;
        this->eps_theta = eps_theta;
        this->gamma_theta = (gamma_theta == 0.0) ? eps_theta : gamma_theta;
        
        // patch for circular case. TODO: better patch using some direct approximate solution when W is so big
        if (std::abs(this->eps) + std::abs(this->gamma) < 1e-8) {
            this->gamma += 1e-8;
        }
    }

    std::complex<double> rotate(double x, double y, double theta, std::complex<double> center = {0, 0}) {
        theta = theta * M_PI / 180.0;
        std::complex<double> point(x - center.real(), y - center.imag());
        std::complex<double> rotated_point = point * std::polar(1.0, theta);
        return rotated_point + center;
    }

    std::complex<double> standardize(double x, double y) {
        std::complex<double> point(x - x_g, y - y_g);
        return rotate(point.real(), point.imag(), -eps_theta);
    }

    std::complex<double> destandardize(double x, double y) {
        std::complex<double> point = rotate(x, y, eps_theta);
        return point + std::complex<double>(x_g, y_g);
    }

    std::complex<double> scronch(std::complex<double> soln, double x_s, double y_s) {
        double costheta = soln.real();
        double sintheta = soln.imag();
        // std::complex<double> point = standardize(x_s, y_s);
        std::complex<double> ellipse_center = get_Wynne_ellipse_center(x_s, y_s);
        double x_e = ellipse_center.real();
        double y_e = ellipse_center.imag();
        std::complex<double> ellipse_semiaxes = get_Wynne_ellipse_semiaxes();
        double x_a = ellipse_semiaxes.real();
        double y_a = ellipse_semiaxes.imag();
        return std::complex<double>(x_e + x_a * costheta, y_e + y_a * sintheta);
    }

    std::vector<std::complex<double>> get_image_configurations(double x_s, double y_s) {
        std::vector<std::complex<double>> image_configurations;
        std::vector<std::complex<double>> solns = get_angular_solns(x_s, y_s);

    for (std::complex<double> soln : solns) {
            image_configurations.push_back(scronch(soln, x_s, y_s));
        }

        return image_configurations;
    }

    std::unordered_map<std::string, double> get_image_and_mag_configuration(double x_s, double y_s) {
        std::unordered_map<std::string, double> image_conf;
        image_conf["b"] = b;
        image_conf["eps"] = eps;
        image_conf["gamma"] = gamma;
        image_conf["x_g"] = x_g;
        image_conf["y_g"] = y_g;
        image_conf["eps_theta"] = eps_theta;
        image_conf["gamma_theta"] = gamma_theta;
        image_conf["x_s"] = x_s;
        image_conf["y_s"] = y_s;

        std::vector<std::complex<double>> scronched_solns = get_image_configurations(x_s, y_s);

        std::vector<double> mags(scronched_solns.size(), 1.0); // TODO: edit this to get magnifications
        std::vector<std::complex<double>> images;
        for (const auto& s : scronched_solns) {
            std::complex<double> image = destandardize(s.real(), s.imag());
            images.push_back(image);
        }

        for (int i = 0; i < images.size(); i++) {
            image_conf["x_" + std::to_string(i+1)] = images[i].real();
            image_conf["y_" + std::to_string(i+1)] = images[i].imag();
            image_conf["mu_" + std::to_string(i+1)] = mags[i];
        }

        return image_conf;
    }

private:
    std::complex<double> get_Wynne_ellipse_center(double x_s, double y_s) {
        std::complex<double> point = standardize(x_s, y_s);
        double x_e = point.real() / (1 + gamma);
        double y_e = point.imag() / (1 - gamma);
        return std::complex<double>(x_e, y_e);
    }

    std::complex<double> get_Wynne_ellipse_semiaxes() {
        double x_a = b / (1 + gamma);
        double y_a = b / ((1 - gamma) * (1 - eps));
        return std::complex<double>(x_a, y_a);
    }

    std::complex<double> get_W(double x_s, double y_s) {
        using namespace std::complex_literals;
        std::complex<double> point = standardize(x_s, y_s);
        double f = (1 - eps) / (b * (1 - std::pow(1 - eps, 2) * (1 - gamma) / (1 + gamma)));
        std::complex<double> W = f * ((1 - eps) * (1 - gamma) / (1 + gamma) * point.real() - 1i * point.imag());
        return W;
    }

    std::vector<std::complex<double>> get_angular_solns(double x_s, double y_s) {
        std::complex<double> W = get_W(x_s, y_s);
        std::vector<std::complex<double>> solns = get_ACLE_angular_solns(W);
        return solns;
    }
};


std::vector<std::complex<double>> get_quad_configuration(double x_s, double y_s, double b, double eps, double gamma, double x_g, double y_g, double eps_theta, double gamma_theta=0.0) {
    SIEP_plus_XS pot(b, eps, gamma, x_g, y_g, eps_theta, gamma_theta);
    std::vector<std::complex<double>> image_conf = pot.get_image_configurations(x_s, y_s);
    return image_conf;
}

int main() {
    double b = 1.133513e00;
    double x_g = -3.830000e-01;
    double y_g = -1.345000e00;
    double eps = 3.520036e-01;
    double eps_theta = 6.638131e01 + 90;
    double gamma = 0.000000e00;

    double x_s = -3.899678e-01;
    double y_s = -1.179477e00;

    SIEP_plus_XS pot(b, eps, gamma, x_g, y_g, eps_theta);
    std::unordered_map<std::string, double> image_conf = pot.get_image_and_mag_configuration(x_s, y_s);

    for (const auto& pair : image_conf) {
        std::cout << pair.first << ": " << pair.second << std::endl;
    }

    // std::vector<std::complex<double>> image_conf = get_quad_configuration(x_s, y_s, b, eps, gamma, x_g, y_g, eps_theta);
    // for (const auto& pair : image_conf) {
    //     std::cout << pair.real() << ", " << pair.imag() << std::endl;
    // }


    return 0;
}
