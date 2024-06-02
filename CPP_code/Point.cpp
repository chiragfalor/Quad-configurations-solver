#include "Point.hpp"

real toRadians(real degrees) {
    return degrees * 3.14159265358979323846 / 180.0;
}

Point rotate(Point point, real theta, Point center) {
    point = point - center;
    theta = toRadians(theta);
    real cosTheta = cos(theta);
    real sinTheta = sin(theta);
    Point rotatedPoint = {point.x * cosTheta - point.y * sinTheta, point.x * sinTheta + point.y * cosTheta};
    return rotatedPoint + center;
}