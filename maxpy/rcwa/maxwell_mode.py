import numpy as np
import torch 

class MaxwellMode:
    def __init__(self):
        pass

    def __eigval_sqrt(self, vals):
        self.valsqrt = torch.sqrt(vals)
        self.valsqrt = (self.valsqrt.imag > 0) * self.valsqrt - (self.valsqrt.imag <= 0) * self.valsqrt
        self.valsqrt = torch.logical_or(torch.abs(self.valsqrt.imag) >= torch.abs(self.valsqrt.real), 
                                        self.valsqrt.real >= 0) * self.valsqrt - torch.logical_and(
                                        torch.abs(self.valsqrt.imag) < torch.abs(self.valsqrt.real),
                                        self.valsqrt.real < 0) * self.valsqrt
        self.valsqrt = (self.valsqrt.imag >= 0) * self.valsqrt + (self.valsqrt.imag < 0) * self.valsqrt.real

    def __evaluate_H(self, matrix_q, valsqrt, vecs):
        self.dual_vecs = (matrix_q @ vecs) / valsqrt
    
    def compute_in_vacuum(self, wavelength, coeff, device = "cpu"):
        '''
        compute the eigenmodes in vacuum

        wavelength: wavelength of the incident light
        coeff: MaxwellCoeff object, it does not need to call compute() method
        '''
        kx = coeff.kx
        ky = coeff.ky
        self.k0 = 2 * np.pi / wavelength
        self.nx = kx.shape[0]
        self.ny = ky.shape[0]
        self.half_dim = self.nx * self.ny
        self.ndim = 2 * self.half_dim

        self.meshkx, self.meshky = torch.meshgrid(kx, ky, indexing = "xy")
        self.meshkx = self.meshkx.flatten()
        self.meshky = self.meshky.flatten()

        self.vals = self.k0**2 * (self.k0**2 - self.meshkx**2 - self.meshky**2)
        self.vals = torch.cat((self.vals, self.vals))
        self.vecs = torch.eye(self.ndim, dtype = torch.complex128, device = device)
        self.left_vecs = torch.eye(self.ndim, dtype = torch.complex128, device = device)
        self.__eigval_sqrt(self.vals)

        Q00 = - torch.diag(self.meshky**2)
        Q00.diagonal().add_(self.k0**2)
        Q01 = torch.diag(self.meshky * self.meshkx)
        Q10 =  - torch.diag(self.meshkx * self.meshky)
        Q11 = torch.diag(self.meshkx**2)
        Q11.diagonal().add_(-self.k0**2)
        matrix_q = torch.cat((torch.cat((Q00, Q01), dim = 1), torch.cat((Q10, Q11), dim = 1)), dim = 0)
        self.dual_vecs = matrix_q / self.valsqrt

    def compute(self, coeff, device = "cpu"):
        '''
        compute the eigenmodes in a distribution specified by coeff

        coeff: MaxwellCoeff object, it need to call compute() method first
        '''
        self.k0 = coeff.k0
        self.half_dim = coeff.half_dim
        self.ndim = coeff.ndim
        self.nx = 2 * coeff.nx + 1
        self.ny = 2 * coeff.ny + 1

        pq = coeff.matrix_pq()
        self.vals, self.vecs = torch.linalg.eig(pq)
        
        self.vals = self.vals.detach()
        self.vecs = self.vecs.detach()
        self.left_vecs = torch.linalg.pinv(self.vecs).detach()
        self.__eigval_sqrt(self.vals)
        self.__evaluate_H(coeff.Q, self.valsqrt, self.vecs)

    def Ex_fourier(self, k):
        '''
        return the Ex field in Fourier space
        '''
        return self.vecs[:self.half_dim, k].reshape((self.ny, self.nx))
    
    def Ey_fourier(self, k):
        '''
        return the Ey field in Fourier space
        '''
        return self.vecs[self.half_dim:None, k].reshape((self.ny, self.nx))

    def Hx_fourier(self, k):
        '''
        return the Hx field in Fourier space
        '''
        return self.dual_vecs[self.half_dim:None, k].reshape((self.ny, self.nx))
       
    def Hy_fourier(self, k):
        '''
        return the Hy field in Fourier space
        '''
        return self.dual_vecs[:self.half_dim, k].reshape((self.ny, self.nx))
    
    def field_in_real(self, field_in_fourier, nx_grid, ny_grid, Lx, Ly):
        M = (self.nx - 1) // 2
        N = (self.ny - 1) // 2
        ms = torch.arange(-M, M+1, device = field_in_fourier.device)
        ns = torch.arange(-N, N+1, device = field_in_fourier.device)
        kx = 2 * np.pi / Lx * ms
        ky = 2 * np.pi / Ly * ns
        xs = torch.linspace(0, Lx, nx_grid, device = field_in_fourier.device, dtype = torch.float64)
        ys = torch.linspace(0, Ly, ny_grid, device = field_in_fourier.device, dtype = torch.float64) 
        basis_kx = torch.exp(1j * kx[:, None] * xs[None, :])
        basis_ky = torch.exp(1j * ky[None, :] * ys[:, None])
        return basis_ky @ field_in_fourier @ basis_kx
    
    def Ex(self, k, nx_grid = 100, ny_grid = 100, Lx = 1, Ly = 1):
        '''
        return the Ex field in real space

        k: the index of the eigenmode
        nx_grid: the number of grid points in x direction
        ny_grid: the number of grid points in y direction
        Lx: the length of the simulation box in x direction
        Ly: the length of the simulation box in y direction
        
        assume the region is [0, Lx] x [0, Ly]
        '''
        Ex_fourier = self.Ex_fourier(k)
        return self.field_in_real(Ex_fourier, nx_grid, ny_grid, Lx, Ly)
    
    def Ey(self, k, nx_grid = 100, ny_grid = 100, Lx = 1, Ly = 1):
        '''
        return the Ey field in real space
        k: the index of the eigenmode
        nx_grid: the number of grid points in x direction
        ny_grid: the number of grid points in y direction
        Lx: the length of the simulation box in x direction
        Ly: the length of the simulation box in y direction
        
        assume the region is [0, Lx] x [0, Ly]
        '''
        Ey_fourier = self.Ey_fourier(k)
        return self.field_in_real(Ey_fourier, nx_grid, ny_grid, Lx, Ly)
    
    def Hx(self, k, nx_grid = 100, ny_grid = 100, Lx = 1, Ly = 1):
        '''
        return the Hx field in real space

        k: the index of the eigenmode
        nx_grid: the number of grid points in x direction
        ny_grid: the number of grid points in y direction
        Lx: the length of the simulation box in x direction
        Ly: the length of the simulation box in y direction
        
        assume the region is [0, Lx] x [0, Ly]
        '''
        Hx_fourier = self.Hx_fourier(k)
        return self.field_in_real(Hx_fourier, nx_grid, ny_grid, Lx, Ly)
    
    def Hy(self, k, nx_grid = 100, ny_grid = 100, Lx = 1, Ly = 1):
        '''
        return the Hy field in real space

        k: the index of the eigenmode
        nx_grid: the number of grid points in x direction
        ny_grid: the number of grid points in y direction
        Lx: the length of the simulation box in x direction
        Ly: the length of the simulation box in y direction
        
        assume the region is [0, Lx] x [0, Ly]
        '''
        Hy_fourier = self.Hy_fourier(k)
        return self.field_in_real(Hy_fourier, nx_grid, ny_grid, Lx, Ly)
