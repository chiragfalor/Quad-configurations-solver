from quartic_solver import get_ACLE_angular_solns
# from np_potentials import SIEP_plus_XS
from potentials import SIEP_plus_XS

import pandas as pd
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
    # check all values are numbers
    config = {k: v for k, v in config.items() if isinstance(v, (int, float)) and v == v}
    return {
        (config[f'x_{i}'], config[f'y_{i}'], abs(config[f'mu_{i}'])) for i in range(1, 5) if f'x_{i}' in config
    }

def config_to_nomag_set(config):
    """ Convert image configuration dictionary to a set of tuples for comparison. """
    config = {k: v for k, v in config.items() if isinstance(v, (int, float)) and v == v}
    return {
        (config[f'x_{i}'], config[f'y_{i}']) for i in range(1, 5) if f'x_{i}' in config
    }

def eq_config(config1, config2, check_mag=True):
    """ Compare two image configurations. 
    The first configuration should be a subset of the second.
    """
    # convert to sets for order-independent comparison
    if check_mag:
        config1_set = config_to_set(config1)
        config2_set = config_to_set(config2)
    else:
        config1_set = config_to_nomag_set(config1)
        config2_set = config_to_nomag_set(config2)
    return all(any(pytest.approx(res, rel=1e-5, abs=1e-8) == exp for exp in config2_set) for res in config1_set)

def get_image_configuration(params):
    ks = SIEP_plus_XS(**params)
    x_s, y_s = params["x_s"], params["y_s"]
    data = ks.get_image_configuration(x_s=x_s, y_s=y_s)
    return data

def get_derivatives(params):
    ks = SIEP_plus_XS(**params)
    x_s, y_s = params["x_s"], params["y_s"]
    data = ks.get_all_derivatives(x_s=x_s, y_s=y_s)
    return data

def compare_images_derivs(images_1, derivs_1, images_2, derivs_2):

    def clean_up(config):
        return {k: v for k, v in config.items() if isinstance(v, (int, float)) and v == v}
    
    images_1, images_2 = clean_up(images_1), clean_up(images_2)
    derivs_1, derivs_2 = clean_up(derivs_1), clean_up(derivs_2)

    # compare images
    assert len(images_1) == len(images_2)
    # assert len(derivs_1) == len(derivs_2) # TODO change this back to assert

    assert eq_config(images_1, images_2, check_mag=True)

    map_btwn_imgs = {}
    for i in range(1, 5):
        if f'x_{i}' in images_1:
            for j in range(1, 5):
                if f'x_{j}' in images_2:
                    if pytest.approx(images_1[f'x_{i}'], rel=1e-3, abs=1e-5) == images_2[f'x_{j}'] and pytest.approx(images_1[f'y_{i}'], rel=1e-3, abs=1e-5) == images_2[f'y_{j}'] and pytest.approx(images_1[f'mu_{i}'], rel=1e-3, abs=1e-5) == images_2[f'mu_{j}']:
                        map_btwn_imgs[i] = j
                        break
    pass
    
    images_2 = {k.replace(str(j), str(i)): v for k, v in images_2.items() for i, j in map_btwn_imgs.items() if str(j) in k}
    derivs_2 = {k.replace(str(j), str(i)): v for k, v in derivs_2.items() for i, j in map_btwn_imgs.items() if str(j) in k}

    return all(pytest.approx(images_1[k], rel=1e-3, abs=1e-5) == images_2[k] for k in images_1) and all(pytest.approx(derivs_1[k], rel=1e-3, abs=1e-5) == derivs_2[k] for k in derivs_1 if 'mu' not in k) # TODO check the derivatives of mu


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
    '''
    Give the path to a Keeton output .out file, this function will read the configuration and return a dictionary of the parameters and image configuration.
    '''
    
    config = {}

    with open(file_path, 'r') as f:
        lines = f.readlines()

    lines = [line.strip().split() for line in lines]
    
    image_start = None
    source_line = None
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
def test_trivial_param_files_python(file):
    cf = load_configuration(file)
    expected_config = get_image_configuration(cf)
    assert eq_config(cf, expected_config), f"Test failed for file: {file}"

@pytest.mark.parametrize("file", ['data/keeton_tests/' + file for file in os.listdir('data/keeton_tests/') if 'out' in file])
def test_extensive_tests_files_python(file):
    cf = load_configuration(file)
    expected_config = get_image_configuration(cf)
    assert eq_config(cf, expected_config), f"Test failed for file: {file}"

def test_cpp_model_on_all_files():
    files = (['data/keeton_tests/' + file 
            for file in os.listdir('data/keeton_tests/') 
            if 'out' in file] + 
            ['data/minimal_params/' + file 
            for file in os.listdir('data/minimal_params/') 
            if ('out' in file) and 
            not 'cf6' in file])
    configs = [load_configuration(file) for file in files]
    df = pd.DataFrame(configs).rename(columns={"eps_theta":"theta"})
    params_list = ["b", "x_g", "y_g", "eps", "gamma", "theta", "x_s", "y_s"]
    params_df = df[params_list]

    params_df.to_csv("test.csv", index=False)
    os.system("./potentials.exe -o test_output.csv test.csv")
    cpp_output_df = pd.read_csv("test_output.csv")
    os.remove("test.csv")
    os.remove("test_output.csv")

    image_col_names = [col for col in cpp_output_df.columns if col not in params_list] # ['x_1', 'y_1', 'mu_1', 'x_2', 'y_2', 'mu_2', 'x_3', 'y_3', 'mu_3', 'x_4', 'y_4', 'mu_4']

    cpp_output_df = cpp_output_df[image_col_names]
    expected_output_df = df[cpp_output_df.columns]
    for i, cpp_row in cpp_output_df.iterrows():
        expected = expected_output_df.iloc[i].to_dict()
        cpp = cpp_row.to_dict()
        assert eq_config(expected, cpp), f"File {files[i]}, Expected {expected} but got {cpp}"


def test_cpp_derivs_with_python_on_all_files():
    files = (['data/keeton_tests/' + file 
            for file in os.listdir('data/keeton_tests/') 
            if 'out' in file] + 
            ['data/minimal_params/' + file 
            for file in os.listdir('data/minimal_params/') 
            if ('out' in file) and 
            not 'cf6' in file])
    configs = [load_configuration(file) for file in files]
    df = pd.DataFrame(configs).rename(columns={"eps_theta":"theta"})
    params_list = ["b", "x_g", "y_g", "eps", "gamma", "theta", "x_s", "y_s"]
    params_df = df[params_list]

    params_df.to_csv("test.csv", index=False)
    os.system("./potentials.exe -o test_output.csv -d test.csv")
    cpp_output_df = pd.read_csv("test_output.csv")
    os.remove("test.csv")
    os.remove("test_output.csv")

    image_col_names = [col for col in cpp_output_df.columns if (col not in params_list) and ('d' not in col)] # ['x_1', 'y_1', 'mu_1', 'x_2', 'y_2', 'mu_2', 'x_3', 'y_3', 'mu_3', 'x_4', 'y_4', 'mu_4']
    deriv_col_names = [col for col in cpp_output_df.columns if 'd' in col]

    cpp_images_df = cpp_output_df[image_col_names]
    params_df = cpp_output_df[params_list]
    cpp_derivs_df = cpp_output_df[deriv_col_names]

    expected_images = df[image_col_names]

    for i, cpp_row in cpp_images_df.iterrows():
        params = params_df.iloc[i].to_dict()
        cpp_images = cpp_row.to_dict()
        cpp_derivs = cpp_derivs_df.iloc[i].to_dict()
        python_images = {k: v for k, v in get_image_configuration(params).items() if k in image_col_names}
        python_derivs = get_derivatives(params)
        
        assert compare_images_derivs(python_images, python_derivs, cpp_images, cpp_derivs), f"File {files[i]}, Expected {python_images}, {python_derivs} but got {cpp_images}, {cpp_derivs}"




    



if __name__ == "__main__":
    pytest.main([__file__, "-v"])
    # just test the cpp model
    # pytest.main([__file__, "-k test_extensive_tests_files_python"])
    # pytest.main([__file__, "-k test_cpp_derivs_with_python_on_all_files"])