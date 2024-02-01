## Quad Quasar Configuration Solver
This is a simple program that solves the quadruple image configurations of source lensed by Singular Isothermal Elliptical Potential with parallel eXternal Shear (SIEP+XS $_{||}$). It outputs the image positions, magnifications, and all the derivatives with respect to the input parameters. 

## Requirements
- Python 3.6 or later
- pytorch 2.0 or later

## Usage

To utilize the code, one needs to import the `SIEP_plus_XS` class from `potentials.py`. Then, one can create an object of this class by passing the lens parameters: `b, x_g, y_g, eps, eps_theta, gamma, gamma_theta`. Here, `eps_theta` and `gamma_theta` are in degrees and they should be equal to each other.

Then, from this object, one can call the function `get_image_configuration` by passing the source position: `x_s, y_s`. This will return the image positions and magnifications. One can also call the function `get_all_derivations` with source position to get the derivatives of the image positions and magnifications with respect to the lens parameters and source position. The output of these functions will be dictionary objects. The keys of the dictionary are the names of the parameters and the values are the corresponding values of the parameters.

The `main.py` script gives the skeleton code to run the program. The potential parameters and source position can be modified in the `main.py` script. The output is a dictionary containing the image positions, magnifications, and all the derivatives with respect to the input parameters.

## SIEP+XS $_{||}$ potential
The SIEP+XS $_{||}$ potential is given by the following equation:
$$ \psi(x,y) = b\sqrt{(x-x_g)^2 + \frac{(y-y_g)^2}{(1-\epsilon)^2}} - \frac{\gamma}{2}\left((x-x_g)^2 - (y-y_g)^2\right) $$
where $b$ is the Einstein radius, $(x_g, y_g)$ is the lens center, $\epsilon$ is the ellipticity, and $\gamma$ is the external shear which is parallel to the ellipticity and aligned along the x-axis.


## Code structure
The code is organized as follows:
- `quartic_solver.py`: This file contains the function to solve the ACLE quartic equation. The function `_get_quartic_solution` takes in a complex parameter $W$ and returns the angular image configuration.
- `potentials.py`: This file contains the base class `Potential`. It contains the main class `SIEP_plus_XS` which inherits from `Potential` and is specific to the SIEP+XS potential. It has the functions `get_image_configuration` and `get_all_derivations`.
- `main.py`: This script uses the class `SIEP_plus_XS` from `potentials.py` to compute the image positions and magnifications from the lens parameters and source position.

## TODO
- Known issues:
    - Currently, the code always outputs four images even when the potential might just produce two images. This will be fixed by checking if the solutions of the ACLE quartic lie on the unit circle.
    - The code needs non-zero ellipticity or shear and crashes for asymptotically spherical potential. This can be fixed by interpolation for small eccentricity.