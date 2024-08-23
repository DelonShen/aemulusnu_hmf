import numpy as np
from scipy.spatial.distance import cdist

key_ordering = ['10^9 As', 'ns', 'H0', 'w0', 'ombh2', 'omch2', 'nu_mass_ev']

class LinearMean():
    def __init__(self, weights, bias):
        """
        assumes weights is n dimensional
        assumes bias is is a scalar
        """
        self.weights = weights
        self.bias   = bias

    def __call__(self, x):
        """
        assumes x is n-dimensional 
        """
        return x @ self.weights + self.bias

class MultitaskMean:
    def __init__(self, base_means):
        self.base_means = base_means
    
    def __call__(self, input):
        return np.vstack([base_mean(input) for base_mean in self.base_means]).T



class MaternKernel():
    def __init__(self, nu, lengthscale):
        self.nu = nu
        self.lengthscale = lengthscale 
    def __call__(self, x1, x2 = None):
        _x1 = x1 / self.lengthscale
        
        if(x2 is None):
            x2 = x1
        _x2 = x2 / self.lengthscale
        

        distance = cdist(_x1, _x2, 'minkowski', p=2)
        exp_component = np.exp(-np.sqrt(self.nu * 2.) * distance)

        if self.nu == 0.5:
            constant_component = 1
        elif self.nu == 1.5:
            constant_component = (np.sqrt(3) * distance) + 1.
        elif self.nu == 2.5:
            constant_component = (np.sqrt(5) * distance)+ 1. + (5/3 * distance**2)
        
        return constant_component * exp_component


class ScaleKernel():
    def __init__(self, base_kernel, scale):
        self.base_kernel = base_kernel
        self.scale = scale
    
    def __call__(self, x1, x2 = None):
        if(x2 is None):
            x2 = x1
        return self.base_kernel(x1, x2) * self.scale


class ConstantKernel():
    def __init__(self, const):
        self.const = const
    
    def __call__(self, x1, x2 = None):
        return self.const * np.eye(len(x1))


class AdditiveKernel():
    def __init__(self, kernel_A, kernel_B):
        self.kernel_A = kernel_A
        self.kernel_B = kernel_B
        
    def __call__(self, x1, x2 = None):
        if(x2 is None):
            x2 = x1
        
        return self.kernel_A(x1, x2) + self.kernel_B(x1, x2)



class IndexKernel():
    def __init__(self, covar_factor, var):
        self.covar_factor = covar_factor
        self.var = var
        self.covar_matrix = self.covar_factor@self.covar_factor.T + np.diag(self.var)
        
    def __call__(self, i1, i2):
        return self.covar_matrix[i1, i2]


class MultitaskKernel():
    def __init__(self,
                 task_covar_module,
                data_covar_module):
        self.task_covar_module = task_covar_module
        self.data_covar_module = data_covar_module
    
    def __call__(self, x1, x2 = None):
        if(x2 is None):
            x2 = x1
        covar_i = self.task_covar_module.covar_matrix
        covar_x = self.data_covar_module(x1, x2)
        res = np.kron(covar_x, covar_i)
        return res
