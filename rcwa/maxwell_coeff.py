import numpy as np
import torch
from .fourier_2d import analytical_fourier_2d

class MaxwellCoeff:
    '''
    This class is used to compute the Maxwell coefficient matrix for RCWA.
    '''
    def __init__(self, nx, ny, Lx, Ly, pml_thickness = 0., device = "cpu"):
        '''
        nx: half of the harmonics along x direction
        ny: half of the harmonics along y direction
        Lx: period along x direction
        Ly: period along y direction
        pml_thickness: thickness of the pml layer
        '''
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly

        self.kx = 2. * np.pi / Lx * torch.arange(-nx, nx + 1, requires_grad = False, device = device)
        self.ky = 2. * np.pi / Ly * torch.arange(-ny, ny + 1, requires_grad = False, device = device)
        self.kx = self.kx.type(torch.complex128)
        self.ky = self.ky.type(torch.complex128)
        self.meshkx, self.meshky = torch.meshgrid(self.kx, self.ky, indexing = "xy")
        self.meshkx = self.meshkx.flatten()
        self.meshky = self.meshky.flatten()
    
        self.nx_harmonics = 2 * nx + 1
        self.ny_harmonics = 2 * ny + 1
        
        self.half_dim = self.nx_harmonics * self.ny_harmonics
        self.ndim = 2 * self.half_dim
        self.fff = True

        if pml_thickness > 0.:
            raise Exception("Sorry, PML for RCWA is not implemented yet, please refer to our Cpp Library VarRCWA.")
    
    def setfff(self, on):
        '''
        Whether to use fast Fourier factorization (default: True)
        
        on: True or False
        '''
        self.fff = on

    def kx(self):
        return self.kx
    
    def ky(self):
        return self.ky
    
    def matrix_pq(self):
        return self.PQ

    def compute(self, wavelength, ex, ey=None, inv_ez=None, kappa = 0., device = "cpu"):
        '''
        wavelength: wavelength of the incident light
        ex: permittivity along x direction
        ey: permittivity along y direction
        inv_ez: inverse of the permittivity along z direction
        kappa: curvature of the domain center

        In Fourier space, we don't need a staggered grid to improve the accuracy of the results.
        So we just assume ex, ey, and inv_ez are defined at the same location (the center) of a grid cell.
        '''
        if kappa != 0.:
            raise Exception("Sorry, curvilinear space for RCWA is not implemented yet.")
        if ey != None or inv_ez != None:
            raise Exception("Sorry, we don't support anisotropic permittivity currently.")
        
        self.k0 = 2. * np.pi / wavelength
        k02 = self.k0 ** 2
        
        # merge the fourier transform and the block toeplitz matrix into one operation
        self.fe = analytical_fourier_2d(ex, self.Lx, self.Ly, self.nx_harmonics, self.ny_harmonics)
        if self.fff:
            self.fex = analytical_fourier_2d(ex, self.Lx, self.Ly, self.nx_harmonics, self.ny_harmonics, "ex")
            self.fey = analytical_fourier_2d(ex, self.Lx, self.Ly, self.nx_harmonics, self.ny_harmonics, "ey")
        else:
            self.fex = self.fe
            self.fey = self.fe

        F1h = torch.linalg.solve(self.fe, torch.diag(self.meshkx))
        F2h = torch.linalg.solve(self.fe, torch.diag(self.meshky))
 
        P00 = - torch.diag(self.meshkx) @ F1h
        P00.diagonal().add_(k02)
        P01 = torch.diag(self.meshkx) @ F2h
        P10 = - torch.diag(self.meshky) @ F1h
        P11 = torch.diag(self.meshky) @ F2h
        P11.diagonal().add_(-k02)

        self.P = torch.cat((torch.cat((P00, P01), dim = 1), torch.cat((P10, P11), dim = 1)), dim = 0)

        Q00 = - torch.diag(self.meshky**2) + k02 * self.fex
        Q01 = torch.diag(self.meshky * self.meshkx)
        Q10 =  - torch.diag(self.meshkx * self.meshky)
        Q11 = torch.diag(self.meshkx**2) - k02 * self.fey

        self.Q = torch.cat((torch.cat((Q00, Q01), dim = 1), torch.cat((Q10, Q11), dim = 1)), dim = 0)

        self.PQ = self.P @ self.Q