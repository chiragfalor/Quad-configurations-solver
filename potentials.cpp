// #include <bits/stdc++.h>
#include <cmath>
#include <complex>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <cassert>


using namespace std;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define COUT_PRECISION 16

// Convert degrees to radians if needed
double toRadians(double degrees) {
    return degrees * M_PI / 180.0;
}

// // Structure to hold the coordinates
class Point {
public:
    double x;
    double y;
    Point(double x = 0.0, double y = 0.0) : x(x), y(y) {}

    Point operator+(const Point& other) const {
        return Point(x + other.x, y + other.y);
    }

    Point operator-(const Point& other) const {
        return Point(x - other.x, y - other.y);
    }

    // Overload the * operator to handle scalar multiplication
    Point operator*(double scalar) const {
        return Point(x * scalar, y * scalar);
    }

    // Optionally, to allow scalar multiplication in the reverse order (scalar * Point)
    friend Point operator*(double scalar, const Point& p) {
        return Point(p.x * scalar, p.y * scalar);
    }
};

// Function to rotate a point around a center by an angle theta (in degrees)
Point rotate(Point point, double theta, Point center = {0.0, 0.0}) {
    // Translate the point to the origin relative to the center
    point = point - center;

    // Convert the angle to radians
    theta = toRadians(theta);

    // Apply rotation matrix
    double cosTheta = cos(theta);
    double sinTheta = sin(theta);
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

vector<Point> get_ACLE_angular_solns(Point W) {
    vector<Point> solns;
    complex<double> W_complex = complex<double>(W.x, W.y);
    for (int i = 0; i < 4; i++) {
        int pm1 = (i / 2 == 0) ? +1 : -1;
        int pm2 = (i % 2 == 0) ? +1 : -1;
        complex<double> s = _get_quartic_solution(W_complex, pm1, pm2);
        if (s != complex<double>(0.0, 0.0)) {
            solns.push_back(Point(s.real(), s.imag()));
        }
    }
    return solns;
}

class SIEP_plus_XS {
private:
    double b, eps, gamma, x_g, y_g, eps_theta, gamma_theta, x_s, y_s;
    Point galaxy_center, source;

public:
    SIEP_plus_XS(double b = 1.0, double eps = 0.0, double gamma = 0.0, double x_g = 0.0, double y_g = 0.0, double eps_theta = 0.0, double gamma_theta = 0.0, double x_s = 0.0, double y_s = 0.0) {
        this->b = b;
        this->eps = eps;
        this->gamma = gamma;
        this->x_g = x_g;
        this->y_g = y_g;
        this->eps_theta = eps_theta;
        this->gamma_theta = (gamma_theta == 0.0) ? eps_theta : gamma_theta;
        this->x_s = x_s;
        this->y_s = y_s;

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
        double costheta = soln.x;
        double sintheta = soln.y;
        
        // stretch the point by the semiaxes of the ellipse
        Point ellipse_semiaxes = get_Wynne_ellipse_semiaxes();
        double x_a = ellipse_semiaxes.x;
        double y_a = ellipse_semiaxes.y;
        Point ellipse_point = Point(x_a * costheta, y_a * sintheta);

        // displace the point by the center of the ellipse
        return ellipse_point + get_Wynne_ellipse_center();

    }

    vector<Point> get_image_configurations() {
        vector<Point> image_configurations;
        vector<Point> solns = get_ACLE_angular_solns(get_W());

    for (Point soln : solns) {
            image_configurations.push_back(scronch(soln));
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

    double soln_to_magnification(Point scronched_soln) {
        double x = scronched_soln.x;
        double y = scronched_soln.y;
        double D_xx, D_yy, D_xy;
        tie(D_xx, D_yy, D_xy) = double_grad_pot(x, y);
        double mu_inv = (1 - D_xx) * (1 - D_yy) - pow(D_xy, 2);
        return 1 / mu_inv;
    }

    tuple<vector<Point>, vector<double>> get_image_and_mags(bool computeMagnification=true) {
        vector<Point> scronched_solns = get_image_configurations();

        vector<Point> images;
        vector<double> mags;
        for (const auto& s : scronched_solns) {
            Point image = destandardize(s);
            images.push_back(image);
            mags.push_back(computeMagnification ? soln_to_magnification(s) : 0.0);
        }

        return make_tuple(images, mags);
    }

private:
    Point get_Wynne_ellipse_center() {
        Point point = standardize(source);
        double x_e = point.x / (1 + gamma);
        double y_e = point.y / (1 - gamma);
        return Point(x_e, y_e);
    }

    Point get_Wynne_ellipse_semiaxes() {
        double x_a = b / (1 + gamma);
        double y_a = b / ((1 - gamma) * (1 - eps));
        return Point(x_a, y_a);
    }

    Point get_W() {
        Point point = standardize(source);
        double f = (1 - eps) / (b * (1 - pow(1 - eps, 2) * (1 - gamma) / (1 + gamma)));
        Point W = f * Point((1 - eps) * (1 - gamma) / (1 + gamma) * point.x, -point.y);
        return W;
    }

};


tuple<vector<Point>, vector<double>> get_images_and_mags(double b, double eps, double gamma, double x_g, double y_g, double theta,double x_s, double y_s, bool computeMagnification=true) {
    SIEP_plus_XS pot(b, eps, gamma, x_g, y_g, theta, theta, x_s, y_s);
    return pot.get_image_and_mags(computeMagnification);
}


void generate_image_configurations_from_CSV(const string& inputFile, ostream& output, bool computeMagnification=true) {
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
    vector<double> mags;
    output << "b,x_g,y_g,eps,gamma,theta,x_s,y_s,";
    output << "x_1,y_1,mu_1,x_2,y_2,mu_2,x_3,y_3,mu_3,x_4,y_4,mu_4" << endl;
    output << scientific << showpos << setprecision(16);
    for (const auto& conf : input_configurations) {
        double x_s = conf[6], y_s = conf[7], b = conf[0], eps = conf[3], gamma = conf[4], x_g = conf[1], y_g = conf[2], theta = conf[5];
        tie(images, mags) = get_images_and_mags(b, eps, gamma, x_g, y_g, theta, x_s, y_s, computeMagnification);
        // Write the results to the output file
        output << b << "," << x_g << "," << y_g << "," << eps << "," << gamma << "," << theta << "," << x_s << "," << y_s << ",";
        size_t num_images = images.size();
        for (size_t i = 0; i < 4; ++i) {
            if (i < num_images) {
                output << images[i].x << "," << images[i].y << "," << mags[i];
            } else {
                output << ",,"; // Extra commas for fewer images
            }
            if (i < 3)
                output << ",";
        }
        output << endl;
    }
    return;
}


namespace run_options {

    void parse(int argc, char* argv[]);
    void print_help();

    string input_file;
    double b, x_g, y_g, eps, gamma, theta, x_s, y_s;

    string output_file;


    bool one_conf;
    bool computeMagnification = true;
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
                b = atof(argv[++i]);
                x_g = atof(argv[++i]);
                y_g = atof(argv[++i]);
                eps = atof(argv[++i]);
                gamma = atof(argv[++i]);
                theta = atof(argv[++i]);
                x_s = atof(argv[++i]);
                y_s = atof(argv[++i]);
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
        else if (i == argc - 1) {
            input_file = argv[i];
        } else {
            cerr << "Error: unknown option " << arg << endl;
            exit(1);
        }
    }
}

void run_options::print_help() {
    cerr << "Usage: " << "potentials" << " [-n] [-o output.csv] [-v] input.csv" << endl;
    cerr << "Usage:./potentials -c b x_g y_g eps gamma theta x_s y_s [-o output_file] [-v] [-n]" << endl;
    cerr << "Options:" << endl;
    cerr << "  -h, --help: print this help message" << endl;
    cerr << "  -c, --conf: calculate for a single configuration" << endl;
    cerr << "  -o, --output FILE: write the output to the specified FILE" << endl;
    cerr << "  -n, --nomag: don't calculate magnifications" << endl;
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
        cout << " b=" << run_options::b << " x_g=" << run_options::x_g << " y_g=" << run_options::y_g << " eps=" << run_options::eps << " gamma=" << run_options::gamma << " theta=" << run_options::theta << " x_s=" << run_options::x_s << " y_s=" << run_options::y_s << endl;

        vector<Point> images;
        vector<double> mags;
        tie(images, mags) = get_images_and_mags(run_options::b, run_options::eps, run_options::gamma, run_options::x_g, run_options::y_g, run_options::theta, run_options::x_s, run_options::y_s, run_options::computeMagnification);
        
        output << "id" << setw(COUT_PRECISION+8) <<  "x" << setw(COUT_PRECISION+8) << "y" << setw(COUT_PRECISION+8) << "mu" << endl;

        output << scientific << showpos << setprecision(16);
        for (size_t i = 0; i < images.size(); i++) {
            output << setw(2) << (i+1);
            output << setw(COUT_PRECISION+8) << images[i].x;
            output << setw(COUT_PRECISION+8) << images[i].y;
            output << setw(COUT_PRECISION+8) << mags[i] << endl;
        }
    } else {
        generate_image_configurations_from_CSV(run_options::input_file, output, run_options::computeMagnification);


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
