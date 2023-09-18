import torch

class OrComponent:
    '''
        A component that is the union of two components

        If the region is in either of the two components, then it is in the union

        c1: Component
        c2: Component
    '''
    def __init__(self, c1, c2):
        assert c1.eps_in == c2.eps_in, 'The permittivity inside the two components must be the same'
        assert c1.eps_out == c2.eps_out, 'The permittivity outside the two components must be the same'
        
        self.c1 = c1
        self.c2 = c2
        self.eps_in = c1.eps_in
        self.eps_out = c1.eps_out

    def smoothen(self, a0, a1, b0, b1, type, h = None):
        c1_r = self.c1.ratio(a0, a1, b0, b1, type, h)
        c2_r = self.c2.ratio(a0, a1, b0, b1, type, h)
        r = torch.max(c1_r, c2_r)
        return self.eps_in * r + self.eps_out * (1 - r)
    
class AndComponent:
    '''
        A component that is the intersection of two components

        If the region is in both the two components, then it is in the component

        c1: Component
        c2: Component
    '''
    def __init__(self, c1, c2):
        assert c1.eps_in == c2.eps_in, 'The permittivity inside the two components must be the same'
        assert c1.eps_out == c2.eps_out, 'The permittivity outside the two components must be the same'
        
        self.c1 = c1
        self.c2 = c2
        self.eps_in = c1.eps_in
        self.eps_out = c1.eps_out

    def smoothen(self, a0, a1, b0, b1, type, h = None):
        c1_r = self.c1.ratio(a0, a1, b0, b1, type, h)
        c2_r = self.c2.ratio(a0, a1, b0, b1, type, h)
        r = torch.min(c1_r, c2_r)
        return self.eps_in * r + self.eps_out * (1 - r)
    
class SubstractComponent:
    '''
        A component that is c1 - c2

        If the region is in both the two components, then it is in the component

        c1: Component
        c2: Component
    '''
    def __init__(self, c1, c2):
        assert c1.eps_in == c2.eps_in, 'The permittivity inside the two components must be the same'
        assert c1.eps_out == c2.eps_out, 'The permittivity outside the two components must be the same'
        
        self.c1 = c1
        self.c2 = c2
        self.eps_in = c1.eps_in
        self.eps_out = c1.eps_out

    def smoothen(self, a0, a1, b0, b1, type, h = None):
        c1_r = self.c1.ratio(a0, a1, b0, b1, type, h)
        c2_r = self.c2.ratio(a0, a1, b0, b1, type, h)
        r = torch.clamp(c1_r - c2_r, 0., 1.)
        return self.eps_in * r + self.eps_out * (1 - r)