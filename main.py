from potentials import SIEP_plus_XS

pot_params = {
    "b": 1.00,
    "x_g": 0.00,
    "y_g": 0.00,
    "eps": 0.3,
    "eps_theta": -90.0,
    "gamma": 0.15,
    "gamma_theta": -90.0,
}

x_s, y_s = 0.04, 0.1

# initialize potential
potential = SIEP_plus_XS(**pot_params)

image_configurations = potential.get_image_configuration(x_s=x_s, y_s=y_s)
print(image_configurations)

derivatives = potential.get_all_derivatives(x_s=x_s, y_s=y_s)
print(derivatives)



