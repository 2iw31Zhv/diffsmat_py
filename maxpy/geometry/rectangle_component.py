import torch
from .utils import smooth_max, smooth_min

class RectangleComponent:
    def __init__(self, x0, x1, y0, y1, eps_in, eps_out, tolerance = 0.):
        '''
        x0, x1, y0, y1: specify the boundary of the rectangle
            x0: the left boundary
            x1: the right boundary
            y0: the bottom boundary
            y1: the up boundary

        eps_in: permittivity inside the component, can be a torch.Tensor
        eps_out: permittivity outside the component, can be a torch.Tensor
        tolerance: float for relaxing the boundary
        '''        
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.eps_in = eps_in
        self.eps_out = eps_out
        self.tolerance = tolerance
        self.device = x0.device
        self.dtype = x0.dtype
    
    def width(self, y = 0.):
        return self.x1 - self.x0 if y >= self.y0 and y <= self.y1 else torch.tensor(0., device = self.device, dtype = self.dtype)
    
    def height(self, x = 0.):
        return self.y1 - self.y0 if x >= self.x0 and x <= self.x1 else torch.tensor(0., device = self.device, dtype = self.dtype)
    
    def ratio(self, a0, a1, b0, b1, type, h = None):
        # it can be not differentiable if the device collocates with the gridline
        # so we need to smoothen it
        with torch.no_grad():
            if h is None:
                h = [torch.min(a1 - a0), torch.min(b1 - b0)]
            relax = h * self.tolerance
    
        xl = smooth_max(self.x0, a0, relax[0])
        xr = smooth_min(self.x1, a1, relax[0])
        yb = smooth_max(self.y0, b0, relax[1])
        yu = smooth_min(self.y1, b1, relax[1])

        rx = torch.clamp((xr - xl) / h[0], min = 0., max = 1.)
        ry = torch.clamp((yu - yb) / h[1], min = 0., max = 1.)
        r = rx * ry
        
        # it looks hard to do subpixel smoothening if boolean operator is applied later,
        # here we just do a simple smoothening based on volume average
        return r

    def smoothen(self, a0, a1, b0, b1, type, h = None):
        '''
        Compute the smoothened permittivity over the grid provided by a0, a1, b0, b1

        a0, a1, b0, b1: torch.Tensor
            The gridlines of the grid
        
        type: str
            The type of smoothening.
        '''
        # it can be not differentiable if the device collocates with the gridline
        # so we need to smoothen it
        with torch.no_grad():
            if h is None:
                h = [torch.min(a1 - a0), torch.min(b1 - b0)]
            relax = h * self.tolerance
    
        xl = smooth_max(self.x0, a0, relax[0])
        xr = smooth_min(self.x1, a1, relax[0])
        yb = smooth_max(self.y0, b0, relax[1])
        yu = smooth_min(self.y1, b1, relax[1])

        rx = torch.clamp((xr - xl) / h[0], min = 0., max = 1.)
        ry = torch.clamp((yu - yb) / h[1], min = 0., max = 1.)
        full_avg = self.eps_out + (self.eps_in - self.eps_out) * rx * ry
        
        # subpixel smoothening
        # https://opg.optica.org/ol/fulltext.cfm?uri=ol-34-18-2778&id=185824
        if type == "ex":
            is_fully_contained = torch.logical_and(b1 >= self.y1, b0 <= self.y0)
            ex_avg = 1. / (1. / self.eps_out + (1. / self.eps_in - 1. / self.eps_out) * rx)
            return full_avg + is_fully_contained * (ex_avg - full_avg)
        elif type == "ey":
            is_fully_contained = torch.logical_and(a1 >= self.x1, a0 <= self.x0)
            ey_avg = 1. / (1. / self.eps_out + (1. / self.eps_in - 1. / self.eps_out) * ry)
            return full_avg + is_fully_contained * (ey_avg - full_avg)
        else:
            return full_avg