import torch

class LevelsetComponent:
    def __init__(self, func, eps_in, eps_out, tolerance = 0.):
        '''
        func: callable (x, y) -> f(x, y)
            if x and y are both torch.Tensor, then f(x, y) should return a torch.Tensor
            if f(x, y) > 0, then the point (x, y) is inside the component
            if f(x, y) < 0, then the point (x, y) is outside the component
            if f(x, y) = 0, then the point (x, y) is on the boundary of the component
        eps_in: permittivity inside the component, can be a torch.Tensor
        eps_out: permittivity outside the component, can be a torch.Tensor
        tolerance: float for relaxing the boundary
        '''
        self.func = func
        self.eps_in = eps_in
        self.eps_out = eps_out
        self.tolerance = tolerance
        self.device = eps_in.device
        self.dtype = torch.float64 if eps_in.dtype == torch.complex128 else torch.float32
    
    def width(self, y = 0.):
        pass
    
    def height(self, x = 0.):
        pass

    def ratio(self, a0, a1, b0, b1, type, h = None):
        alpha = 1 / (self.tolerance + 1e-12)
        f00 = self.func(a0, b0)
        f10 = self.func(a1, b0)
        f01 = self.func(a0, b1)
        f11 = self.func(a1, b1)
        r = torch.sigmoid(alpha * 0.25 * (f00 + f10 + f01 + f11))
        return r
    
    def smoothen(self, a0, a1, b0, b1, type, h = None):
        '''
        Compute the smoothened permittivity over the grid provided by a0, a1, b0, b1
        This is not differentiable currently

        a0, a1, b0, b1: torch.Tensor
            The gridlines of the grid
        
        type: str
            The type of smoothening.
        '''
        r = self.ratio(a0, a1, b0, b1, type, h)
        return self.eps_in * r + self.eps_out * (1 - r)