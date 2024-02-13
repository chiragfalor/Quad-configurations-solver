from quartic_solver import _get_quartic_solution, get_ACLE_angular_solns
import numpy as np
from typing import Tuple, List, Set, Dict, Any


def rotate(
    x: np.float64,
    y: np.float64,
    theta: np.float64,
    center: np.ndarray = np.array([0, 0]),
) -> Tuple[np.float64, np.float64]:
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in degrees.

    Parameters
    ----------
    x : float
        The x-coordinate of the point to be rotated.
    y : float
        The y-coordinate of the point to be rotated.
    theta : float
        The angle of rotation in degrees.
    center : np.ndarray, optional
        The center of rotation, by default np.array([0, 0])

    Returns
    -------
    Tuple[float, float]
        The rotated coordinates (x_new, y_new).
    """
    theta = np.deg2rad(theta)
    x, y = x - center[0], y - center[1]
    x_new = x * np.cos(theta) - y * np.sin(theta)
    y_new = x * np.sin(theta) + y * np.cos(theta)
    x_new, y_new = x_new + center[0], y_new + center[1]
    return x_new, y_new


class SIEP_plus_XS:
    def __init__(
        self,
        b=1.0,
        eps=0.0,
        gamma=0.0,
        x_g=0.0,
        y_g=0.0,
        eps_theta=0.0,
        gamma_theta=None,
        **kwargs,
    ):
        # phi = b\sqrt{x^2 + y^2/(1-eps)^2} - gamma/2*(x^2 - y^2)

        self.gamma_theta = eps_theta if gamma_theta is None else gamma_theta

        # assert theta are close, we don't handle unparallel cases yet
        assert abs(self.gamma_theta - eps_theta) < 1e-3 or abs(gamma * eps) < 1e-10

        self.pot_params = {
            "b": np.array(b, dtype=np.float64).item(),
            "eps": np.array(eps, dtype=np.float64).item(),
            "gamma": np.array(gamma, dtype=np.float64).item(),
            "x_g": np.array(x_g, dtype=np.float64).item(),
            "y_g": np.array(y_g, dtype=np.float64).item(),
            "theta": np.array(eps_theta, dtype=np.float64).item(),
        }
        self.b = self.pot_params["b"]
        self.eps = self.pot_params["eps"]
        self.gamma = self.pot_params["gamma"]
        self.x_g, self.y_g = self.pot_params["x_g"], self.pot_params["y_g"]
        self.eps_theta, self.gamma_theta = (
            self.pot_params["theta"],
            self.pot_params["theta"],
        )

        # patch for circular case. TODO: better patch using some direct approximate solution when W is so big
        if np.abs(self.eps) + np.abs(self.gamma) < 1e-8:
            with np.errstate(divide="ignore"):
                self.gamma += 1e-8

        # self.potential = lambda x, y: b*torch.sqrt((x-x_g)**2 + (y-y_g)**2/(1-eps)**2) - gamma/2*((x-x_g)**2 - (y-y_g)**2)
        self.potential = lambda x, y: b * np.sqrt(
            x**2 + y**2 / (1 - eps) ** 2
        ) - gamma / 2 * (x**2 - y**2)

    def standardize(
        self, x: np.float64, y: np.float64
    ) -> Tuple[np.float64, np.float64]:
        x, y = x - self.x_g, y - self.y_g
        return rotate(x, y, -self.eps_theta)

    def destandardize(
        self, x: np.float64, y: np.float64
    ) -> Tuple[np.float64, np.float64]:
        x, y = rotate(x, y, self.eps_theta)
        x, y = x + self.x_g, y + self.y_g
        return x, y

    def _get_Wynne_ellipse_params(
        self, x_s: np.float64, y_s: np.float64
    ) -> Tuple[Tuple[np.float64, np.float64], Tuple[np.float64, np.float64]]:
        x_s, y_s = self.standardize(x_s, y_s)
        x_e, y_e = (x_s) / (1 + self.gamma), (y_s) / (1 - self.gamma)
        x_a, y_a = self.b / (1 + self.gamma), self.b / (
            (1 - self.gamma) * (1 - self.eps)
        )
        return (x_e, y_e), (x_a, y_a)

    def get_W(self, x_s: np.float64, y_s: np.float64) -> np.complex128:
        x_s, y_s = self.standardize(x_s, y_s)
        b, eps, g = self.b, self.eps, self.gamma
        f = (1 - eps) / (b * (1 - (1 - eps) ** 2 * (1 - g) / (1 + g)))
        W = f * ((1 - eps) * (1 - g) / (1 + g) * (x_s) - 1j * (y_s))

        return W

    def scronch(
        self, soln: np.complex128, x_s: np.float64, y_s: np.float64
    ) -> np.complex128:
        costheta, sintheta = soln.real, soln.imag
        (x_e, y_e), (x_a, y_a) = self._get_Wynne_ellipse_params(x_s, y_s)
        return x_e + x_a * costheta + 1j * (y_e + y_a * sintheta)

    def get_angular_solns(self, x_s: np.float64, y_s: np.float64) -> Set[np.complex128]:
        W = self.get_W(x_s, y_s)
        assert W.dtype == np.complex128

        return get_ACLE_angular_solns(W)

    def images_and_mags(self, x_s: np.float64, y_s: np.float64):

        soln = self.get_angular_solns(x_s, y_s)
        scronched_solns = [self.scronch(s, x_s, y_s) for s in soln]

        # mags = [self.soln_to_magnification(soln) for soln in scronched_solns]
        mags = [np.array(1.0) for soln in scronched_solns]  # no magnification for numpy

        images = [self.destandardize(s.real, s.imag) for s in scronched_solns]
        images = [x[0] + 1j * x[1] for x in images]

        return images, mags

    def get_image_configuration(
        self, x_s: np.float64, y_s: np.float64
    ) -> Dict[str, np.float64]:
        image_conf = self.pot_params.copy()
        image_conf["x_s"], image_conf["y_s"] = x_s, y_s

        images, mags = self.images_and_mags(x_s=x_s, y_s=y_s)

        for i, (im, mag) in enumerate(zip(images, mags)):
            image_conf[f"x_{i+1}"] = im.real
            image_conf[f"y_{i+1}"] = im.imag
            image_conf[f"mu_{i+1}"] = mag
        return image_conf


if __name__ == "__main__":

    test1115 = {
        "b": 1.133513e00,
        "x_g": -3.830000e-01,
        "y_g": -1.345000e00,
        "eps": 3.520036e-01,
        "eps_theta": 6.638131e01 + 90,
        "gamma": 0.000000e00,
    }

    source = {
        "x_s": -3.899678e-01,
        "y_s": -1.179477e00,
    }

    pot = SIEP_plus_XS(**test1115)
    print(pot.get_image_configuration(**source))
