#include "SIEP_plus_XS.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <iomanip>

using std::cout;
using std::cerr;
using std::endl;
using std::string;
using std::setw;

constexpr int COUT_PRECISION = 16;

void generate_image_configurations_from_CSV(const string& inputFile, ostream& output, bool computeMagnification=true, bool computeDerivatives=false) {
    std::ifstream inFile(inputFile);

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
        std::stringstream ss(line);
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
    output << std::scientific << std::showpos << std::setprecision(COUT_PRECISION);
    for (const vector<double>& conf : input_configurations) {
        // double x_s = conf[6], y_s = conf[7], b = conf[0], eps = conf[3], gamma = conf[4], x_g = conf[1], y_g = conf[2], theta = conf[5];
        PotentialParams params(conf[0], conf[3], conf[4], conf[1], conf[2], conf[5], 0.0, conf[6], conf[7]);

        tie(images, mags) = get_images_and_mags(params, computeMagnification);
        // Write the results to the output file
        output << params.b << "," << params.x_g << "," << params.y_g << "," << params.eps << "," << params.gamma << "," << params.eps_theta << "," << params.x_s << "," << params.y_s << ",";

        int maxImages = 4;  // There are at most 4 images
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

std::streambuf *original_cout_buffer;

void restore_cout_buffer() {
    cout.rdbuf(original_cout_buffer);
}

int main(int argc, char* argv[]) {
    run_options::parse(argc, argv);
    
    clock_t start, end;

    start = clock();

    ostream& output = cout;
    std::ofstream file_output;

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

        output << std::scientific << std::showpos << std::setprecision(COUT_PRECISION);
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
    cout << "Execution time: " << std::noshowpos << std::fixed << std::setprecision(6) << ( 1e3 * (end - start) / CLOCKS_PER_SEC ) << " ms" <<endl;

}
