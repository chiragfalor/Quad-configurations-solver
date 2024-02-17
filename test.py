from quartic_solver import get_ACLE_angular_solns
from SIEP_XS import SIEP_plus_XS
# from potentials import SIEP_plus_XS
import pytest
import os

@pytest.mark.parametrize("W, expected", [
    (complex(0, 0), [1, -1, 1j, -1j]),
    (complex(1, 0), [1, 1, 1, -1]),
    (complex(0, 1), [1j, 1j, 1j, -1j]),
    (complex(-1, 0), [1, -1, -1, -1]),
    (complex(0, -1), [-1j, -1j, -1j, 1j]),
])
def test_solve_quartic(W, expected):
    solutions = get_ACLE_angular_solns(W)
    # Convert solutions to a set for order-independent comparison
    solutions_set = {complex(round(sol.real, 3), round(sol.imag, 3)) for sol in solutions}
    expected_set = {complex(round(sol.real, 3), round(sol.imag, 3)) for sol in expected}
    assert solutions_set == expected_set

def config_to_set(config):
    """ Convert image configuration dictionary to a set of tuples for comparison. """
    return {
        (config[f'x_{i}'], config[f'y_{i}'], abs(config[f'mu_{i}'])) for i in range(1, 5) if f'x_{i}' in config
    }

def config_to_nomag_set(config):
    """ Convert image configuration dictionary to a set of tuples for comparison. """
    return {
        (config[f'x_{i}'], config[f'y_{i}']) for i in range(1, 5) if f'x_{i}' in config
    }

def eq_config(config1, config2, check_mag=True):
    """ Compare two image configurations. 
    The first configuration should be a subset of the second.
    """
    if check_mag:
        config1_set = config_to_set(config1)
        config2_set = config_to_set(config2)
    else:
        config1_set = config_to_nomag_set(config1)
        config2_set = config_to_nomag_set(config2)
    return all(any(pytest.approx(res, rel=1e-3, abs=1e-5) == exp for exp in config2_set) for res in config1_set)

def get_image_configuration(params):
    ks = SIEP_plus_XS(**params)
    x_s, y_s = params["x_s"], params["y_s"]
    data = ks.get_image_configuration(x_s=x_s, y_s=y_s)
    return data



@pytest.mark.parametrize("b, eps, x_s, y_s, expected", [
    (1, 0.1, 0.01, 0, {
        'x_1': 5.263158e-02, 'y_1': 1.110101e+00, 'mu_1': 5.272741e+00,
        'x_2': 5.263158e-02, 'y_2': -1.110101e+00, 'mu_2': 5.272741e+00,
        'x_3': 1.010000e+00, 'y_3': 0, 'mu_3': -4.497526e+00,
        'x_4': -9.900000e-01, 'y_4': 0, 'mu_4': -4.047956e+00
    }),
    (1, 0.1, 0.03, 0.08, {
    'x_1': 1.226381e-01, 'y_1': 1.186333e+00, 'mu_1': 4.113508e+00,
    'x_2': 2.388038e-01, 'y_2': -1.006619e+00, 'mu_2': 8.570159e+00,
    'x_3': 9.270349e-01, 'y_3': -4.110664e-01, 'mu_3': 6.654215e+00,
    'x_4': -9.126874e-01, 'y_4': -2.907525e-01, 'mu_4': 4.029452e+00
    }), # cf2.outA
    (1, 0.2, 0.03, 0.02, {
    'x_1': 8.106099e-02, 'y_1': 1.268369e+00, 'mu_1': 2.708795e+00,
    'x_2': 8.581801e-02, 'y_2': -1.228051e+00, 'mu_2': 2.869959e+00,
    'x_3': 1.028897e+00, 'y_3': -3.868910e-02, 'mu_3': 1.938970e+00,
    'x_4': -9.691095e-01, 'y_4': -3.274024e-02, 'mu_4': 1.639783e+00
    }), # cf3.outA
    (1, 0.3, 0.04, 0.1, {
    'x_1': 7.379077e-02, 'y_1': 1.527756e+00, 'mu_1': 1.846623e+00,
    'x_2': 8.455250e-02, 'y_2': -1.327153e+00, 'mu_2': 2.118688e+00,
    'x_3': 1.029752e+00, 'y_3': -1.039996e-01, 'mu_3': 1.062538e+00,
    'x_4': -9.512321e-01, 'y_4': -8.875996e-02, 'mu_4': 9.027727e-01
    }), # cf4.outA
])
def test_SIEP(b, eps, x_s, y_s, expected):
    result = get_image_configuration({
        "x_s": x_s, "y_s": y_s, "b": b, "eps": eps
    })

    assert eq_config(result, expected)


def load_configuration(file_path):
    
    config = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip().split() for line in lines]
    
    image_start = None
    for i, line in enumerate(lines):
        if 'findimg' in line:
            source_line = lines[i+1]
        elif 'images:' in line:
            image_start = i+1
        elif image_start is not None and '>' in line:
            image_end = i
            break
    image_lines = lines[image_start:image_end]
    x_s, y_s = float(source_line[0]), float(source_line[1])
    config['x_s'], config['y_s'] = x_s, y_s
    for i, image_line in enumerate(image_lines):
        config[f'x_{i+1}'] = float(image_line[0])
        config[f'y_{i+1}'] = float(image_line[1])
        config[f'mu_{i+1}'] = float(image_line[2])

    
    config_file_path = file_path.replace('.out', '.mod')
    with open(config_file_path, 'r') as f:
        lines = f.readlines()
    pot_params = lines[1].split()[1:]
    b, x_g, y_g, eps, eps_theta, gamma, gamma_theta, _, core_r, slope = [float(param) for param in pot_params]
    config['b'] = b
    config['x_g'] = x_g
    config['y_g'] = y_g
    config['eps'] = eps
    config['eps_theta'] = (eps_theta - 90)
    config['gamma'] = gamma
    config['gamma_theta'] = gamma_theta - 90
    
    return config


@pytest.mark.parametrize("file", ['data/minimal_params/' + file 
                                  for file in os.listdir('data/minimal_params/') 
                                  if ('out' in file) and 
                                  not 'cf6' in file]) # cf6 has unaligned shear and elliptcity 
def test_trivial_param_files(file):
    cf = load_configuration(file)
    expected_config = get_image_configuration(cf)
    assert eq_config(cf, expected_config), f"Test failed for file: {file}"

@pytest.mark.parametrize("file", ['data/keeton_tests/' + file for file in os.listdir('data/keeton_tests/') if 'out' in file])
def test_extensive_tests_files(file):
    cf = load_configuration(file)
    expected_config = get_image_configuration(cf)
    assert eq_config(cf, expected_config), f"Test failed for file: {file}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])