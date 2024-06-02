#pragma once
#include <iostream>
#include <autodiff/forward/real.hpp>

using autodiff::real;
using std::ostream;

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

    Point operator*(real scalar) const {
        return Point(x * scalar, y * scalar);
    }

    friend Point operator*(real scalar, const Point& p) {
        return Point(p.x * scalar, p.y * scalar);
    }

    friend ostream& operator<<(ostream& os, const Point& point) {
        os << point.x << "," << point.y;
        return os;
    }
};

real toRadians(real degrees);
Point rotate(Point point, real theta, Point center = {0.0, 0.0});
