#pragma once
#include "Point.hpp"

#include <complex>
#include <vector>

using std::complex;
using std::vector;

complex<double> _get_quartic_solution(complex<double> W, int pm1 = +1, int pm2 = +1);
vector<complex<double>> ACLE(complex<double> W);
complex<double> ACLE_dWreal(complex<double> W, complex<double> z);
complex<double> ACLE_dWimag(complex<double> W, complex<double> z);
vector<Point> ACLE_dual(Point W);