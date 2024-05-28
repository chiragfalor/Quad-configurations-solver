// #include <bits/stdc++.h>
#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <iomanip>
#include <cassert>

#include <autodiff/forward/dual.hpp>
#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>
using autodiff::dual;
using autodiff::real;
using autodiff::VectorXreal;

using Eigen::MatrixXd;

using std::complex;
using std::cout;
using std::cerr;
using std::endl;
using std::ifstream;
using std::ofstream;
using std::stringstream;
using std::vector;
using std::string;
using std::ostream;
using std::setw;
using std::tuple;
using std::tie;
using std::make_tuple;
using std::scientific;
using std::showpos;
using std::setprecision;
using std::streambuf;
using std::noshowpos;
using std::fixed;
using std::max;

#define COUT_PRECISION 16

// Convert degrees to radians if needed
real toRadians(real degrees) 
{
    return degrees * 3.14159265358979323846 / 180.0;
}

// // Structure to hold the coordinates
class Point {
public:
    real x;
    real y;
    Point(real x = 0.0, real y = 0.0) : x(x), y(y) {}

    Point operator+(const Point& other) const {
        return Point(x + other.x, y + other.y);
    }

    Point operator-(const Point& other) const {
        return Point(x - other.x, y - other.y);
    }

    // Overload the * operator to handle scalar multiplication
    Point operator*(real scalar) const {
        return Point(x * scalar, y * scalar);
    }

    // Optionally, to allow scalar multiplication in the reverse order (scalar * Point)
    friend Point operator*(real scalar, const Point& p) {
        return Point(p.x * scalar, p.y * scalar);
    }
};

ostream& operator<<(ostream& os, const Point& point) {
    os << point.x << "," << point.y;
    return os;
}

// Function to rotate a point around a center by an angle theta (in degrees)
Point rotate(Point point, real theta, Point center = {0.0, 0.0}) {
    // Translate the point to the origin relative to the center
    point = point - center;

    // Convert the angle to radians
    theta = toRadians(theta);

    // Apply rotation matrix
    real cosTheta = cos(theta);
    real sinTheta = sin(theta);
    Point rotatedPoint = {point.x * cosTheta - point.y * sinTheta, point.x * sinTheta + point.y * cosTheta};

    // Translate back to the original center
    return rotatedPoint + center;
}


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

// vector<Point> get_ACLE_angular_solns(Point W) {
//     vector<Point> solns;
//     complex<double> W_complex = complex<double>(W.x.val(), W.y.val());  
//     for (int i = 0; i < 4; i++) {
//         int pm1 = (i / 2 == 0) ? +1 : -1;
//         int pm2 = (i % 2 == 0) ? +1 : -1;
//         complex<double> s = _get_quartic_solution(W_complex, pm1, pm2);
//         if (s != complex<double>(0.0, 0.0)) {
//             solns.push_back(Point(s.real(), s.imag()));
//         }
//     }
//     return solns;
// }

struct PotentialParams {
    real b, eps, gamma, x_g, y_g, eps_theta, gamma_theta, x_s, y_s;
    PotentialParams(real b = 1.0, real eps = 0.0, real gamma = 0.0, real x_g = 0.0, real y_g = 0.0, real eps_theta = 0.0, real gamma_theta = 0.0, real x_s = 0.0, real y_s = 0.0) {
        this->b = b;
        this->eps = eps;
        this->gamma = gamma;
        this->x_g = x_g;
        this->y_g = y_g;
        this->eps_theta = eps_theta;
        this->gamma_theta = gamma_theta;
        this->x_s = x_s;
        this->y_s = y_s;
    }
};


ostream& operator<<(ostream& os, const PotentialParams& params) {
    os << "b=" << params.b << " eps=" << params.eps << " gamma=" << params.gamma << " x_g=" << params.x_g << " y_g=" << params.y_g << " eps_theta=" << params.eps_theta << " gamma_theta=" << params.gamma_theta << " x_s=" << params.x_s << " y_s=" << params.y_s;
    return os;
}

class SIEP_plus_XS {
private:
    real b, eps, gamma, x_g, y_g, eps_theta, gamma_theta, x_s, y_s;
    Point galaxy_center, source;

public:
    SIEP_plus_XS(const PotentialParams& params) {
        this->b = params.b;
        this->eps = params.eps;
        this->gamma = params.gamma;
        this->x_g = params.x_g;
        this->y_g = params.y_g;
        this->eps_theta = params.eps_theta;
        this->gamma_theta = (params.gamma_theta == 0.0) ? params.eps_theta : params.gamma_theta;
        this->x_s = params.x_s;
        this->y_s = params.y_s;

        this->galaxy_center = Point(x_g, y_g);
        this->source = Point(x_s, y_s);
        
        // patch for circular case. TODO: better patch using some direct approximate solution when W is so big
        if (abs(this->eps) + abs(this->gamma) < 1e-8) {
            this->gamma += 1e-8;
        }
    }


    Point standardize(Point point) {
        return rotate(point - galaxy_center, -eps_theta);
    }

    Point destandardize(Point point) {
        return rotate(point, eps_theta) + galaxy_center;
    }

    Point scronch(Point soln) {
        real costheta = soln.x;
        real sintheta = soln.y;
        
        // stretch the point by the semiaxes of the ellipse
        Point ellipse_semiaxes = get_Wynne_ellipse_semiaxes();
        real x_a = ellipse_semiaxes.x;
        real y_a = ellipse_semiaxes.y;
        Point ellipse_point = Point(x_a * costheta, y_a * sintheta);

        // displace the point by the center of the ellipse
        return ellipse_point + get_Wynne_ellipse_center();

    }
    // is image lost
    // # get hyperbola center
    //     W = self.get_W(**kwargs)
    //     (x_e, y_e), (x_a, y_a) = self._get_Wynne_ellipse_params(**kwargs)
    //     x_h, y_h = W.real * x_a + x_e, W.imag * y_a + y_e

    //     x_g, y_g = 0, 0 # in standardized coordinates, galaxy is at origin

    //     g_angle = torch.atan2(y_g-y_h, x_g-x_h)
    //     e_angle = torch.atan2(y_e-y_h, x_e-x_h)


    //     image_angle = torch.atan2(im.imag-y_h, im.real-x_h)

    //     # an image is lost if it is between the galaxy and center of ellipse on the hyperbola0
    //     return (g_angle < angle < e_angle) or (g_angle > angle > e_angle)0

    bool isImageLost(Point scronchedImage) {
        Point W = get_W();
        Point ellipse_center = get_Wynne_ellipse_center();
        Point ellipse_semiaxes = get_Wynne_ellipse_semiaxes();
        Point hyperbola_center = Point(W.x * ellipse_semiaxes.x + ellipse_center.x, W.y * ellipse_semiaxes.y + ellipse_center.y);

        real x_g = 0.0, y_g = 0.0;
        real g_angle = atan2(y_g - hyperbola_center.y, x_g - hyperbola_center.x);
        real e_angle = atan2(ellipse_center.y - hyperbola_center.y, ellipse_center.x - hyperbola_center.x);
        real image_angle = atan2(scronchedImage.y - hyperbola_center.y, scronchedImage.x - hyperbola_center.x);

        return (g_angle < image_angle && image_angle < e_angle) || (g_angle > image_angle && image_angle > e_angle);

    }

    vector<Point> get_image_configurations() {
        vector<Point> image_configurations;
        vector<Point> solns = ACLE_dual(get_W());

    for (Point soln : solns) {
            Point scronchedSoln = scronch(soln);
            if (!isImageLost(scronchedSoln)) {
                image_configurations.push_back(scronchedSoln);
            }
        }

        return image_configurations;
    }

    tuple<real, real, real> double_grad_pot(real x, real y) {
        real t = pow((x*x + y*y/((1-eps)*(1-eps))), 0.5);
        real f = b / (t*t*t * ((1-eps)*(1-eps)));
        real D_xx = f*y*y - gamma;
        real D_yy = f*x*x + gamma;
        real D_xy = -f*x*y;
        return make_tuple(D_xx, D_yy, D_xy);
    }

    real soln_to_magnification(Point scronched_soln) {
        real D_xx, D_yy, D_xy;
        tie(D_xx, D_yy, D_xy) = double_grad_pot(scronched_soln.x, scronched_soln.y);
        real mu_inv = (1 - D_xx) * (1 - D_yy) - D_xy*D_xy;
        return 1 / mu_inv;
    }

    tuple<vector<Point>, vector<real>> get_image_and_mags(bool computeMagnification=true) {
        vector<Point> scronched_solns = get_image_configurations();

        vector<Point> images;
        vector<real> mags;
        for (const Point& s : scronched_solns) {
            Point image = destandardize(s);
            images.push_back(image);
            mags.push_back(computeMagnification ? soln_to_magnification(s) : 0.0);
        }

        return make_tuple(images, mags);
    }

private:
    Point get_Wynne_ellipse_center() {
        Point point = standardize(source);
        real x_e = point.x / (1 + gamma);
        real y_e = point.y / (1 - gamma);
        return Point(x_e, y_e);
    }

    Point get_Wynne_ellipse_semiaxes() {
        real x_a = b / (1 + gamma);
        real y_a = b / ((1 - gamma) * (1 - eps));
        return Point(x_a, y_a);
    }

    Point get_W() {
        Point point = standardize(source);
        real f = (1 - eps) / (b * (1 - ((1-eps)*(1-eps))* (1 - gamma) / (1 + gamma)));
        Point W = f * Point((1 - eps) * (1 - gamma) / (1 + gamma) * point.x, -point.y);
        return W;
    }

};


tuple<vector<Point>, vector<real>> get_images_and_mags(const PotentialParams& params, bool computeMagnification=true) {
    SIEP_plus_XS pot(params);
    return pot.get_image_and_mags(computeMagnification);
}

MatrixXd get_derivatives(const PotentialParams& params, bool computeMagnification=true) {
    auto f = [&](const VectorXreal& p) -> VectorXreal {
        PotentialParams params(p[0], p[3], p[4], p[1], p[2], p[5], 0.0, p[6], p[7]);
        SIEP_plus_XS pot(params);
        vector<Point> images;
        vector<real> mags;
        tie(images, mags) = pot.get_image_and_mags(computeMagnification);

        VectorXreal result(3 * images.size());
        for (size_t i = 0; i < images.size(); i++) {
            result[3 * i] = images[i].x;
            result[3 * i + 1] = images[i].y;
            result[3 * i + 2] = mags[i];
        }

        return result;
    };
    
    // the jacobian matrix will be a 3n * 8 matrix where n is the number of images
    // the parameters are in the order: "b,x_g,y_g,eps,gamma,theta,x_s,y_s", which is same as expected in the header of a CSV file

    VectorXreal p(8);
    p << params.b, params.x_g, params.y_g, params.eps, params.gamma, params.eps_theta, params.x_s, params.y_s;


    return autodiff::jacobian(f, autodiff::wrt(p), autodiff::at(p));

}


void generate_image_configurations_from_CSV(const string& inputFile, ostream& output, bool computeMagnification=true, bool computeDerivatives=false) {
    ifstream inFile(inputFile);
    // ofstream outFile(outputFile);

    if (!inFile.is_open()) {
        cerr << "Error opening file: " << inputFile << endl;
        return;
    }

    vector<vector<double>> input_configurations;

    
    string header;
    getline(inFile, header);
    assert(header == "b,x_g,y_g,eps,gamma,theta,x_s,y_s");

    string line;
    while (getline(inFile, line)) {
        stringstream ss(line);
        double value;
        vector<double> conf;
        while (ss >> value) {
            conf.push_back(value);
            if (ss.peek() == ',')
                ss.ignore();
        }
        assert(conf.size() == 8);
        input_configurations.push_back(conf);
    }   
    inFile.close();
    // output to the output file
    vector<Point> images;
    vector<real> mags;

    vector<string> paramNames = {"b", "x_g", "y_g", "eps", "gamma", "theta", "x_s", "y_s"};
    vector<string> outputNames = {"x_1", "y_1", "mu_1", "x_2", "y_2", "mu_2", "x_3", "y_3", "mu_3", "x_4", "y_4", "mu_4"};

    for (const string& name : paramNames) {
        output << name << ",";
    }
    for (const string& name : outputNames) {
        output << name;
        if (&name != &outputNames.back())
            output << ",";
    }

    // Append derivative headers if necessary
    if (computeDerivatives) {
        for (const auto& outputName : outputNames) {
            for (const auto& paramName : paramNames) {
                output << ",d(" << outputName << ")/d(" << paramName << ")";
            }
        }
    }

    output << endl;
    output << scientific << showpos << setprecision(16);
    for (const vector<double>& conf : input_configurations) {
        // double x_s = conf[6], y_s = conf[7], b = conf[0], eps = conf[3], gamma = conf[4], x_g = conf[1], y_g = conf[2], theta = conf[5];
        PotentialParams params(conf[0], conf[3], conf[4], conf[1], conf[2], conf[5], 0.0, conf[6], conf[7]);

        tie(images, mags) = get_images_and_mags(params, computeMagnification);
        // Write the results to the output file
        output << params.b << "," << params.x_g << "," << params.y_g << "," << params.eps << "," << params.gamma << "," << params.eps_theta << "," << params.x_s << "," << params.y_s << ",";
        // size_t num_images = images.size();
        int maxImages = 4;  // Assume there are at most 4 images
        for (int i = 0; i < maxImages; i++) {
            if (i < images.size()) {
                output << images[i].x << "," << images[i].y << "," << mags[i];
            } else {
                output << ",,";  // Fill missing image data with empty fields
            }
            if (i < maxImages - 1) output << ",";
        }

        // Compute and write derivatives if required
        if (computeDerivatives) {
            MatrixXd derivatives = get_derivatives(params, computeMagnification);
            for (int i = 0; i < 3*maxImages; ++i) {
                for (int j = 0; j < derivatives.cols(); ++j) {
                    output << ",";
                    if (i < derivatives.rows()) {
                        output << derivatives(i, j);
                    }
                }
            }
        }

        output << endl;
    }
    return;
}


namespace run_options {

    void parse(int argc, char* argv[]);
    void print_help();

    string input_file;
    PotentialParams params;

    string output_file;


    bool one_conf;
    bool computeMagnification = true;
    bool computeDerivatives = false;
    bool out_to_file;

}

void run_options::parse(int argc, char* argv[]) {
    if (argc < 2) {
        print_help();
        exit(1);
    }

    for (int i = 1; i < argc; i++) {
        string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_help();
            exit(0);
        } else if (arg == "-c" || arg == "--conf") {
            if (i + 8 < argc) {
                one_conf = true;
                params.b = atof(argv[++i]);
                params.x_g = atof(argv[++i]);
                params.y_g = atof(argv[++i]);
                params.eps = atof(argv[++i]);
                params.gamma = atof(argv[++i]);
                params.eps_theta = atof(argv[++i]);
                params.x_s = atof(argv[++i]);
                params.y_s = atof(argv[++i]);
            } else {
                cerr << "Error: -c option requires 8 floating point arguments: b x_g y_g eps gamma theta x_s y_s." << endl;
                exit(1);
            }
        } else if (arg == "-o" || arg == "--output") {
            if (i + 1 < argc) {
                out_to_file = true;
                output_file = argv[i + 1];
                i++;
            } else {
                cerr << "Error: -o option requires one argument." << endl;
                exit(1);
            }
        } else if (arg == "-n" || arg == "--nomag") {
            computeMagnification = false;
        } // else check if the file name has been provided
        else if (arg == "-d" || arg == "--deriv") {
            computeDerivatives = true;
        }
        else if (i == argc - 1) {
            input_file = argv[i];
        } else {
            cerr << "Error: unknown option " << arg << endl;
            exit(1);
        }
    }
}

void run_options::print_help() {
    cerr << "Usage: " << "potentials" << " [-n] [-d] [-o output.csv] [-v] input.csv" << endl;
    cerr << "Usage:./potentials -c b x_g y_g eps gamma theta x_s y_s [-o output_file] [-v] [-n] [-d]" << endl;
    cerr << "Options:" << endl;
    cerr << "  -h, --help: print this help message" << endl;
    cerr << "  -c, --conf: calculate for a single configuration" << endl;
    cerr << "  -o, --output FILE: write the output to the specified FILE" << endl;
    cerr << "  -n, --nomag: don't calculate magnifications" << endl;
    cerr << "  -d, --deriv: calculate derivatives" << endl;
    cerr << endl;
    cerr << "Input CSV format:" << endl;
    cerr << "b,x_g,y_g,eps,gamma,theta,x_s,y_s" << endl;
    exit(1);
    return;
}

streambuf *original_cout_buffer;

void restore_cout_buffer() {
    cout.rdbuf(original_cout_buffer);
}

int main(int argc, char* argv[]) {
    run_options::parse(argc, argv);
    
    clock_t start, end;

    start = clock();

    ostream& output = cout;
    ofstream file_output;

    if (run_options::out_to_file) {
    if (!run_options::output_file.empty()) {
        file_output.open(run_options::output_file);
        if (file_output.is_open()) {
            original_cout_buffer = cout.rdbuf();
            output.rdbuf(file_output.rdbuf()); // Redirect output to the file
        } else {
            cerr << "Error opening output file " << run_options::output_file << endl;
            exit(1);
        }
    } else {
        cerr << "Error: No output file specified." << endl;
        exit(1);
    }
}

    if (run_options::one_conf) {
        cout << "Running:";
        cout << run_options::params << endl;

        vector<Point> images;
        vector<real> mags;
        tie(images, mags) = get_images_and_mags(run_options::params, run_options::computeMagnification);
        
        output << "id" << setw(COUT_PRECISION+8) <<  "x" << setw(COUT_PRECISION+8) << "y" << setw(COUT_PRECISION+8) << "mu" << endl;

        output << scientific << showpos << setprecision(16);
        for (size_t i = 0; i < images.size(); i++) {
            output << setw(2) << (i+1);
            output << setw(COUT_PRECISION+8) << images[i].x;
            output << setw(COUT_PRECISION+8) << images[i].y;
            output << setw(COUT_PRECISION+8) << mags[i] << endl;
        }
        // output derivative matrix if required
        if (run_options::computeDerivatives) {
            MatrixXd derivatives = get_derivatives(run_options::params, run_options::computeMagnification);
            output << "Derivatives:" << endl;
            output << derivatives << endl;
        }
    } else {
        generate_image_configurations_from_CSV(run_options::input_file, output, run_options::computeMagnification, run_options::computeDerivatives);
    }

            
    if (file_output.is_open()) {
        // redirect the output back to stdout
        restore_cout_buffer();
        file_output.close();
    }


    end = clock();
    cout << "Execution completed successfully." << endl;
    cout << "Execution time: " << noshowpos << fixed << setprecision(6) << ( 1e3 * (end - start) / CLOCKS_PER_SEC ) << " ms" <<endl;

}
