#include "SIEP_plus_XS.hpp"

using autodiff::VectorXreal;

ostream& operator<<(ostream& os, const PotentialParams& params) {
    os << "b=" << params.b << " eps=" << params.eps << " gamma=" << params.gamma << " x_g=" << params.x_g << " y_g=" << params.y_g << " eps_theta=" << params.eps_theta << " gamma_theta=" << params.gamma_theta << " x_s=" << params.x_s << " y_s=" << params.y_s;
    return os;
}

SIEP_plus_XS::SIEP_plus_XS(const PotentialParams& params) {
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


    Point SIEP_plus_XS::standardize(Point point) {
        return rotate(point - galaxy_center, -eps_theta);
    }

    Point SIEP_plus_XS::destandardize(Point point) {
        return rotate(point, eps_theta) + galaxy_center;
    }

    Point SIEP_plus_XS::scronch(Point soln) {
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

    bool SIEP_plus_XS::isImageLost(Point scronchedImage) {
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

    vector<Point> SIEP_plus_XS::get_image_configurations() {
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

    tuple<real, real, real> SIEP_plus_XS::double_grad_pot(real x, real y) {
        real t = pow((x*x + y*y/((1-eps)*(1-eps))), 0.5);
        real f = b / (t*t*t * ((1-eps)*(1-eps)));
        real D_xx = f*y*y - gamma;
        real D_yy = f*x*x + gamma;
        real D_xy = -f*x*y;
        return std::make_tuple(D_xx, D_yy, D_xy);
    }

    real SIEP_plus_XS::soln_to_magnification(Point scronched_soln) {
        real D_xx, D_yy, D_xy;
        std::tie(D_xx, D_yy, D_xy) = double_grad_pot(scronched_soln.x, scronched_soln.y);
        real mu_inv = (1 - D_xx) * (1 - D_yy) - D_xy*D_xy;
        return 1 / mu_inv;
    }

    tuple<vector<Point>, vector<real>> SIEP_plus_XS::get_image_and_mags(bool computeMagnification) {
        vector<Point> scronched_solns = get_image_configurations();

        vector<Point> images;
        vector<real> mags;
        for (const Point& s : scronched_solns) {
            Point image = destandardize(s);
            images.push_back(image);
            mags.push_back(computeMagnification ? soln_to_magnification(s) : 0.0);
        }

        return std::make_tuple(images, mags);
    }


    Point SIEP_plus_XS::get_Wynne_ellipse_center() {
        Point point = standardize(source);
        real x_e = point.x / (1 + gamma);
        real y_e = point.y / (1 - gamma);
        return Point(x_e, y_e);
    }

    Point SIEP_plus_XS::get_Wynne_ellipse_semiaxes() {
        real x_a = b / (1 + gamma);
        real y_a = b / ((1 - gamma) * (1 - eps));
        return Point(x_a, y_a);
    }

    Point SIEP_plus_XS::get_W() {
        Point point = standardize(source);
        real f = (1 - eps) / (b * (1 - ((1-eps)*(1-eps))* (1 - gamma) / (1 + gamma)));
        Point W = f * Point((1 - eps) * (1 - gamma) / (1 + gamma) * point.x, -point.y);
        return W;
    }


tuple<vector<Point>, vector<real>> get_images_and_mags(const PotentialParams& params, bool computeMagnification) {
    SIEP_plus_XS pot(params);
    return pot.get_image_and_mags(computeMagnification);
}

MatrixXd get_derivatives(const PotentialParams& params, bool computeMagnification) {
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


