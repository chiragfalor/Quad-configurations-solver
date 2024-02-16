# pragma once

#include <complex>

using namespace std;

extern std::complex<double> _get_quartic_solution(std::complex<double>, std::complex<double>, int, int);

extern std::vector<std::complex<double>> get_ACLE_angular_solns(std::complex<double>);