#pragma once
#include "QuarticSolver.hpp"

#include <tuple>
#include <autodiff/forward/real/eigen.hpp>

using std::tuple;
using Eigen::MatrixXd;

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

ostream& operator<<(ostream& os, const PotentialParams& params);


class SIEP_plus_XS {
private:
    real b, eps, gamma, x_g, y_g, eps_theta, gamma_theta, x_s, y_s;
    Point galaxy_center, source;

    Point standardize(Point point);
    Point destandardize(Point point);
    Point scronch(Point soln);
    bool isImageLost(Point scronchedImage);
    Point get_Wynne_ellipse_center();
    Point get_Wynne_ellipse_semiaxes();
    Point get_W();
    tuple<real, real, real> double_grad_pot(real x, real y);
    real soln_to_magnification(Point scronched_soln);

public:
    SIEP_plus_XS(const PotentialParams& params);
    vector<Point> get_image_configurations();
    tuple<vector<Point>, vector<real>> get_image_and_mags(bool computeMagnification = true);
};

tuple<vector<Point>, vector<real>> get_images_and_mags(const PotentialParams& params, bool computeMagnification = true);
MatrixXd get_derivatives(const PotentialParams& params, bool computeMagnification = true);