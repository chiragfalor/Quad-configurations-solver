from torch_quartic_solver import ACLE
import torch

def rotate(x, y, theta, center=torch.tensor((0, 0))):
    """
    Rotate a point counterclockwise by a given angle around a given origin.
    The angle should be given in degrees.
    """
    theta = torch.deg2rad(theta)
    x, y = x-center[0], y-center[1]
    x_new = x * torch.cos(theta) - y * torch.sin(theta)
    y_new = x * torch.sin(theta) + y * torch.cos(theta)
    x_new, y_new = x_new+center[0], y_new+center[1]
    return x_new, y_new


def tensorize_dict(d):
    return {k: v if isinstance(v, torch.Tensor) else torch.tensor(v, dtype=torch.float64, requires_grad=True)
                  for k, v in d.items()}



class Potential:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.params = tensorize_dict(kwargs)

    def get_W(self, x_s, y_s, **kwargs):
        '''
        returns the W parameter of the ACLE
        '''
        raise NotImplementedError
    
    def _get_Wynne_ellipse_params(self, **kwargs):
        """
        This function computes the center and axes lengths of the Wynne ellipse in standardized coordinates. It accepts optional keyword arguments for calculations that depend on the potential.

        Parameters
        ----------
        **kwargs : dict, optional
            Arbitrary keyword arguments.

        Returns
        -------
        ((float, float), (float, float))
            A tuple of two tuples:
            - First tuple: the center of the ellipse (x_e, y_e).
            - Second tuple: the horizontal and vertical axes of the ellipse (x_a, y_a).
        """
        raise NotImplementedError
    
    def standardize(self, x, y):
        return x, y
    
    def destandardize(self, x, y):
        return x, y
    
    def scronch(self, soln, **kwargs) -> torch.Tensor:
        costheta, sintheta = soln.real, soln.imag
        (x_e, y_e), (x_a, y_a) = self._get_Wynne_ellipse_params(**kwargs)
        return x_e + x_a*costheta + 1j*(y_e + y_a*sintheta)
    

    # first derivatives
    def grad_pot(self, x, y):
        '''
        returns a tuple of the first derivatives of the potential with respect to x and y
        '''
        return torch.autograd.grad(self.potential(x, y), [x, y], create_graph=True, retain_graph=True)
    
    # second derivatives
    def double_grad_pot_auto(self, x, y):
        '''
        This doesn't quite give the right result when taking the derivative of this. Using analytical derivatives
        '''
        grads = self.grad_pot(x, y)
        D_xx, D_xy = torch.autograd.grad(grads[0], [x, y], retain_graph=True, create_graph=True)
        D_yx, D_yy = torch.autograd.grad(grads[1], [x, y],retain_graph=True, create_graph=True)
        assert torch.allclose(D_xy, D_yx)
        return D_xx, D_yy, D_xy
    
    def soln_to_magnification(self, scronched_soln):
        '''
        returns the magnification of the image corresponding to the solution soln
        '''
        x, y = scronched_soln.real, scronched_soln.imag
        D_xx, D_yy, D_xy = self.double_grad_pot(x, y)
        mu_inv = (1 - D_xx)*(1 - D_yy) - D_xy**2
        return 1/mu_inv
    
    def _images_and_mags(self, **kwargs):
        kwargs = tensorize_dict(kwargs)
        soln = ACLE(self.get_W(**kwargs))
        scronched_solns = [self.scronch(s, **kwargs) for s in soln]
        scronched_solns = self.remove_the_lost_image(scronched_solns, **kwargs)
        
        mags = [self.soln_to_magnification(soln) for soln in scronched_solns]

        images = [self.destandardize(s.real, s.imag) for s in scronched_solns]
        images = [x[0]+1j*x[1] for x in images]

        return images, mags
    
    def images_and_mags(self, **kwargs):
        images, mags = self._images_and_mags(**kwargs)
        return [image.detach().numpy().item() for image in images], [mag.detach().numpy().item() for mag in mags]
    
    
    def get_derivative(self, qty, param_name, **kwargs):
        kwargs = tensorize_dict(kwargs)
        images, mags = self._images_and_mags(**kwargs)
        # qty is x_i, y_i, or mu_i
        qty, i = qty.split('_')
        i = int(i)
        assert qty in ['x', 'y', 'mu']
        param = self.params[param_name] if param_name in self.params else kwargs[param_name] if param_name in kwargs else KeyError(f"Parameter {param_name} not found in potential parameters or kwargs")
        if qty == 'x':
            return self.d(images[i-1].real, param).detach().numpy().item()
        elif qty == 'y':
            return self.d(images[i-1].imag, param).detach().numpy().item()
        elif qty == 'mu':
            return self.d(mags[i-1], param).detach().numpy().item()
        
    def get_all_derivatives(self, **kwargs):
        kwargs = tensorize_dict(kwargs)
        images, mags = self._images_and_mags(**kwargs)
        kwargs.update(self.params)
        derivatives = {}
        for param_name, param in self.params.items():
            for i, mag in enumerate(mags):
                dW = self.d(mag, param)
                derivatives[f'd(mu_{i+1})/d({param_name})'] = dW.detach().numpy().item()
            for i, image in enumerate(images):
                dW = self.d(image.real, param)+1j*self.d(image.imag, param)
                derivatives[f'd(x_{i+1})/d({param_name})'] = dW.real.detach().numpy().item()
                derivatives[f'd(y_{i+1})/d({param_name})'] = dW.imag.detach().numpy().item()
        return derivatives
    
    def get_image_configuration(self, raw=False, **kwargs):
        image_conf = tensorize_dict(kwargs) if raw else kwargs.copy()
        images, mags = self.images_and_mags(**image_conf) if not raw else self._images_and_mags(**image_conf)

        image_conf.update(self.params if raw else {k: v.detach().numpy().item() for k, v in self.params.items()})
        for i, (im, mag) in enumerate(zip(images, mags)):
            image_conf[f'x_{i+1}'] = im.real
            image_conf[f'y_{i+1}'] = im.imag
            image_conf[f'mu_{i+1}'] = mag
        return image_conf
    
    def d(self, y, x):
        # take derivative of tensor y with respect to tensor x
        return torch.autograd.grad(y, [x], create_graph=True, retain_graph=True)[0]
    
    def remove_the_lost_image(self, scronched_images, **kwargs):
        # get hyperbola center
        W = self.get_W(**kwargs)
        (x_e, y_e), (x_a, y_a) = self._get_Wynne_ellipse_params(**kwargs)
        x_h, y_h = W.real * x_a + x_e, W.imag * y_a + y_e

        x_g, y_g = 0, 0 # in standardized coordinates, galaxy is at origin

        g_angle = torch.atan2(y_g-y_h, x_g-x_h)
        e_angle = torch.atan2(y_e-y_h, x_e-x_h)

        
        # assert that ellipse and galaxy angle should be in the same quadrant, as the hyperbola is right angle with asymptotes parallel to axes
        assert g_angle // (torch.pi/2) == e_angle // (torch.pi/2)

        images_angles = [torch.atan2(im.imag-y_h, im.real-x_h) for im in scronched_images]

        # an image is lost if it is between the galaxy and center of ellipse on the hyperbola
        lost_image = None
        for i, angle in enumerate(images_angles):
            if g_angle < angle < e_angle or g_angle > angle > e_angle:
                lost_image = i
                break

        if lost_image is not None:
            scronched_images.pop(lost_image)

        return scronched_images

    


class SIEP_plus_XS(Potential):
    def __init__(self, b=0, eps=0, gamma=0, x_g=0, y_g=0, eps_theta=0, gamma_theta=None, theta=None, x_s=0, y_s=0, **kwargs):
        # phi = b\sqrt{x^2 + y^2/(1-eps)^2} - gamma/2*(x^2 - y^2)
        if theta is not None:
            eps_theta = theta
        self.gamma_theta = eps_theta if gamma_theta is None else gamma_theta
        if abs(eps) < 1e-5:
            eps_theta = self.gamma_theta
        # assert theta are close, we don't handle unparallel cases yet
        assert abs(self.gamma_theta - eps_theta) < 1e-3 or abs(gamma*eps) < 1e-10
        super().__init__(b=b, eps=eps, gamma=gamma, x_g=x_g, y_g=y_g, theta=eps_theta, x_s=x_s, y_s=y_s)
        self.b = self.params['b']
        self.eps = self.params['eps']
        self.gamma = self.params['gamma']
        self.x_g, self.y_g = self.params['x_g'], self.params['y_g']
        self.eps_theta, self.gamma_theta = self.params['theta'], self.params['theta']
        self.x_s, self.y_s = self.params['x_s'], self.params['y_s']
        # self.potential = lambda x, y: b*torch.sqrt((x-x_g)**2 + (y-y_g)**2/(1-eps)**2) - gamma/2*((x-x_g)**2 - (y-y_g)**2)
        self.potential = lambda x, y: b*torch.sqrt(x**2 + y**2/(1-eps)**2) - gamma/2*(x**2 - y**2)

    def standardize(self, x, y):
        x, y = x-self.x_g, y-self.y_g
        return rotate(x, y, -self.eps_theta)
    
    def destandardize(self, x, y):
        x, y = rotate(x, y, self.eps_theta)
        x, y = x+self.x_g, y+self.y_g
        return x, y

    def _get_Wynne_ellipse_params(self, **kwargs):
        x_s, y_s = self.standardize(self.x_s, self.y_s)
        x_e, y_e = (x_s)/(1+self.gamma), (y_s)/(1-self.gamma)
        x_a, y_a = self.b/(1+self.gamma), self.b/((1-self.gamma)*(1-self.eps))
        return (x_e, y_e), (x_a, y_a)
    
    def get_W(self, **kwargs):
        x_s, y_s = self.standardize(self.x_s, self.y_s)
        b, eps, g = self.b, self.eps, self.gamma

        f = (1-eps) / (b * (1 - (1-eps)**2 * (1-g)/(1+g)))
        W = f*((1-eps)*(1-g)/(1+g) * (x_s) - 1j*(y_s))

        return W
    
        
    def double_grad_pot(self, x, y):
        '''
        returns a tuple of the second derivatives of the potential.
        (d^2 psi / dx^2, d^2 psi / dy^2, d^2 psi / dx dy)
        '''
        t = torch.sqrt(x**2 + (y/(1-self.eps))**2)
        f = self.b / (t**3*(1-self.eps)**2)
        D_xx = f*y**2 - self.gamma
        D_yy = f*x**2 + self.gamma
        D_xy = -f*x*y
        return D_xx, D_yy, D_xy

if __name__ == '__main__':

    test1115 = {
        'b': 1.133513e+00,
        'x_g': -3.830000e-01,
        'y_g': -1.345000e+00,
        'eps': 3.520036e-01,
        'eps_theta': 6.638131e+01 + 90,
        'gamma': 0.000000e+00,
    }

    source = {
        'x_s': -3.899678e-01,
        'y_s': -1.179477e+00,
    }

    pot = SIEP_plus_XS(**test1115)
    print(pot.get_image_configuration(**source))