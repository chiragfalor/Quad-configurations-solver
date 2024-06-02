import subprocess

class SIEP_plus_XS_CPP():
    def __init__(self, b=0, eps=0, gamma=0, x_g=0, y_g=0, theta=0, x_s=0, y_s=0, **kwargs):
        # phi = b\sqrt{x^2 + y^2/(1-eps)^2} - gamma/2*(x^2 - y^2)
        self.executable_path = "./CPP_code/Main.exe"
        self.params = {
            'b': b,
            'eps': eps,
            'gamma': gamma,
            'x_g': x_g,
            'y_g': y_g,
            'theta': theta,
            'x_s': x_s,
            'y_s': y_s,
        }

        
        self.param_order = ["b", "x_g", "y_g", "eps", "gamma", "theta", "x_s", "y_s"]

    def get_image_configuration(self, **kwargs):
        command = [self.executable_path, "-c"] + [str(self.params[key]) for key in self.param_order]

        
        # Execute the command
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            raise Exception("Command failed:", process.stderr)
        
        # Process the output
        output_lines = process.stdout.splitlines()
        image_configurations = {}
        for line in output_lines:
           if line.strip() and line.split()[0].isdigit():
                parts = line.split()
                if len(parts) == 4:
                    id, x, y, mu = map(float, parts)
                    id = int(id)
                    image_configurations.update({
                        f'x_{id}': x,
                        f'y_{id}': y,
                        f'mu_{id}': mu,
                    })             
        # Update dictionary with initial parameters
        image_configurations.update({k: v for k, v in self.params.items()})
        return image_configurations

    def get_all_derivatives(self, **kwargs):
        command = [self.executable_path, "-c"] + [str(self.params[key]) for key in self.param_order] + ["-d"]
        
        # Execute the command
        process = subprocess.run(command, capture_output=True, text=True)
        if process.returncode != 0:
            raise Exception("Command failed:", process.stderr)
        
        derivatives = {}
        parsing_derivatives = False
        line_num = 0
        vars = ['x', 'y', 'mu']
        for line in process.stdout.splitlines():
            if 'Derivatives:' in line:
                parsing_derivatives = True
                continue
            if parsing_derivatives and line.strip():
                parts = line.split()
                if len(parts) == len(self.param_order):
                    parts = list(map(float, parts))
                    for i, param in enumerate(self.param_order):
                        var = vars[line_num % 3]
                        derivatives[f'd({var}_{line_num//3+1})/d({param})'] = parts[i]
                    line_num += 1

        return derivatives

if __name__=="__main__":
    params = {
        'b': 1.0,
        'x_g': 0.3,
        'y_g': 0.5,
        'eps': 0.3,
        'gamma': 0.1,
        'theta': -80.0,
        'x_s': 0.04,
        'y_s': 0.1
    }
    siep_plus_xs = SIEP_plus_XS_CPP(**params)
    image_configurations = siep_plus_xs.get_image_configuration()
    derivatives = siep_plus_xs.get_all_derivatives()
    print(image_configurations)
    print(derivatives)
    print("Successfully ran the C++ code")