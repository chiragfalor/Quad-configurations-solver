// #include <bits/stdc++.h>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>
#include <unordered_map>
#include <iomanip>


using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


complex<double> _get_quartic_solution(complex<double> W, int pm1 = +1, int pm2 = +1) {
    auto isclose = [](complex<double> a, complex<double> b, double rel_tol = 1e-09, double abs_tol = 1e-8) {
        return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol);
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

vector<complex<double>> get_ACLE_angular_solns(complex<double> W) {
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
        if (abs(this->eps) + abs(this->gamma) < 1e-8) {
            this->gamma += 1e-8;
        }
    }

    complex<double> rotate(double x, double y, double theta, complex<double> center = {0, 0}) {
        theta = theta * M_PI / 180.0;
        complex<double> point(x - center.real(), y - center.imag());
        complex<double> rotated_point = point * polar(1.0, theta);
        return rotated_point + center;
    }

    complex<double> standardize(double x, double y) {
        complex<double> point(x - x_g, y - y_g);
        return rotate(point.real(), point.imag(), -eps_theta);
    }

    complex<double> destandardize(double x, double y) {
        complex<double> point = rotate(x, y, eps_theta);
        return point + complex<double>(x_g, y_g);
    }

    complex<double> scronch(complex<double> soln, double x_s, double y_s) {
        double costheta = soln.real();
        double sintheta = soln.imag();
        // complex<double> point = standardize(x_s, y_s);
        complex<double> ellipse_center = get_Wynne_ellipse_center(x_s, y_s);
        double x_e = ellipse_center.real();
        double y_e = ellipse_center.imag();
        complex<double> ellipse_semiaxes = get_Wynne_ellipse_semiaxes();
        double x_a = ellipse_semiaxes.real();
        double y_a = ellipse_semiaxes.imag();
        return complex<double>(x_e + x_a * costheta, y_e + y_a * sintheta);
    }

    vector<complex<double>> get_image_configurations(double x_s, double y_s) {
        vector<complex<double>> image_configurations;
        vector<complex<double>> solns = get_angular_solns(x_s, y_s);

    for (complex<double> soln : solns) {
            image_configurations.push_back(scronch(soln, x_s, y_s));
        }

        return image_configurations;
    }

    tuple<double, double, double> double_grad_pot(double x, double y) {
        double t = pow((x*x + y*y/(pow(1-eps, 2))), 0.5);
        double f = b / (pow(t, 3) * pow(1-eps, 2));
        double D_xx = f*y*y - gamma;
        double D_yy = f*x*x + gamma;
        double D_xy = -f*x*y;
        return make_tuple(D_xx, D_yy, D_xy);
    }

    double soln_to_magnification(complex<double> scronched_soln) {
        double x = scronched_soln.real();
        double y = scronched_soln.imag();
        double D_xx, D_yy, D_xy;
        tie(D_xx, D_yy, D_xy) = double_grad_pot(x, y);
        double mu_inv = (1 - D_xx) * (1 - D_yy) - pow(D_xy, 2);
        return 1 / mu_inv;
    }

    tuple<vector<complex<double>>, vector<double>> get_image_and_mags(double x_s, double y_s) {
        vector<complex<double>> scronched_solns = get_image_configurations(x_s, y_s);

        vector<complex<double>> images;
        vector<double> mags;
        for (const auto& s : scronched_solns) {
            complex<double> image = destandardize(s.real(), s.imag());
            images.push_back(image);
            mags.push_back(soln_to_magnification(s));
        }

        return make_tuple(images, mags);
    }

private:
    complex<double> get_Wynne_ellipse_center(double x_s, double y_s) {
        complex<double> point = standardize(x_s, y_s);
        double x_e = point.real() / (1 + gamma);
        double y_e = point.imag() / (1 - gamma);
        return complex<double>(x_e, y_e);
    }

    complex<double> get_Wynne_ellipse_semiaxes() {
        double x_a = b / (1 + gamma);
        double y_a = b / ((1 - gamma) * (1 - eps));
        return complex<double>(x_a, y_a);
    }

    complex<double> get_W(double x_s, double y_s) {
        using namespace complex_literals;
        complex<double> point = standardize(x_s, y_s);
        double f = (1 - eps) / (b * (1 - pow(1 - eps, 2) * (1 - gamma) / (1 + gamma)));
        complex<double> W = f * ((1 - eps) * (1 - gamma) / (1 + gamma) * point.real() - 1i * point.imag());
        return W;
    }

    vector<complex<double>> get_angular_solns(double x_s, double y_s) {
        complex<double> W = get_W(x_s, y_s);
        vector<complex<double>> solns = get_ACLE_angular_solns(W);
        return solns;
    }
};


tuple<vector<complex<double>>, vector<double>> get_images_and_mags(double x_s, double y_s, double b, double eps, double gamma, double x_g, double y_g, double eps_theta, double gamma_theta=0.0) {
    SIEP_plus_XS pot(b, eps, gamma, x_g, y_g, eps_theta, gamma_theta);
    return pot.get_image_and_mags(x_s, y_s);
    // vector<complex<double>> image_conf = pot.get_image_configurations(x_s, y_s);
    // return image_conf;
}

int main() {
    clock_t start, end;
    double b = 1.133513e00;
    double x_g = -3.830000e-01;
    double y_g = -1.345000e00;
    double eps = 3.520036e-01;
    double eps_theta = 6.638131e01 + 90;
    double gamma = 0.000000e00;

    double x_s = -3.899678e-01;
    double y_s = -1.179477e00;

    // SIEP_plus_XS pot(b, eps, gamma, x_g, y_g, eps_theta);
    vector<complex<double>> images;
    vector<double> mags;
    start = clock();
    tie(images, mags) = get_images_and_mags(x_s, y_s, b, eps, gamma, x_g, y_g, eps_theta);
    end = clock();
    cout << "id " << "x " << "y " << "mag " << endl;

    for (size_t i = 0; i < images.size(); i++) {
            cout << i+1 << " " << images[i].real() << " " << images[i].imag() << " " << mags[i] << endl;
        }

    cout << "Execution time: " << fixed << setprecision(6) << ( 1e3 * (end - start) / CLOCKS_PER_SEC ) << " ms" <<endl;

    // vector<complex<double>> image_conf = get_quad_configuration(x_s, y_s, b, eps, gamma, x_g, y_g, eps_theta);
    // for (const auto& pair : image_conf) {
    //     cout << pair.real() << ", " << pair.imag() << endl;
    // }


    return 0;
}
