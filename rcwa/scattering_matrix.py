import torch
from .diff_matfunc import sqrtm_ops_core, eig_diff

class ScatteringMatrix:
    def __init__(self):
        pass
    
    @staticmethod
    def allocate(half_dim, device = "cpu"):
        '''
        half_dim: the half dimension of the scattering matrix
        device: the device that the scattering matrix is allocated on

        This method allocates a scattering matrix with the given half dimension
        for a perfect transmission layer with thickness 0

        smat = [Tuu, Rud;
                Rdu, Tdd]
        where Tuu = Tdd = identity, Rud = Rdu = 0
        '''
        smat = ScatteringMatrix()
        smat.half_dim = half_dim
        smat.smat = torch.eye(2 * half_dim, device = device, dtype = torch.complex128)
        return smat

    def compute(self, mode, z):
        '''
        mode: the mode that the scattering matrix is computed for (the core)
        '''
        self.W = mode.vecs
        self.invW = mode.left_vecs
        self.valsqrt = mode.valsqrt
        self.V = mode.dual_vecs
        self.k0 = mode.k0
        self.z = z
        self.ndim = mode.vecs.shape[0]
        self.nmode = mode.valsqrt.shape[0]
        self.propogator = torch.exp(1j * z / self.k0 * mode.valsqrt)

    # construct the scattering matrix using 
    def port_project_v2(self, port_mode, coeff, tolerance = 1e-6):
        '''
        The version of relaxing the repeated eigenvalues
        '''
        PQ = coeff.matrix_pq()
        vals, W = eig_diff(PQ)
        valsqrt = torch.sqrt(vals)
        valsqrt = (valsqrt.imag > 0) * valsqrt - (valsqrt.imag <= 0) * valsqrt
        valsqrt = torch.logical_or(torch.abs(valsqrt.imag) >= torch.abs(valsqrt.real), 
                                        valsqrt.real >= 0) * valsqrt - torch.logical_and(
                                        torch.abs(valsqrt.imag) < torch.abs(valsqrt.real),
                                        valsqrt.real < 0) * valsqrt
        valsqrt = (valsqrt.imag >= 0) * valsqrt + (valsqrt.imag < 0) * valsqrt.real
        Omega = W @ torch.diag(valsqrt) @ torch.linalg.inv(W)
        
        # invW_PQ_W = self.invW @ coeff.matrix_pq() @ self.W
        # a hacky way to do differentiable matrix square root with the square root of diagonal alreay computed
        # Omega = sqrtm_ops_core(invW_PQ_W, self.valsqrt.detach())
        invQV0 = torch.linalg.solve(coeff.Q, port_mode.dual_vecs)
        # TV0 = (self.W @ Omega) @ (self.invW @ invQV0)
        TV0 = Omega @ invQV0
        # differentiable matrix exponential is available after PyTorch 1.7.0
        # we wrote differentiable Pade approximation manually before since it was not available,
        # you can check our cpp code for that implementation
        K = torch.linalg.matrix_exp(1j * self.z / self.k0 * Omega)

        M1 = port_mode.vecs + TV0
        N2 = - (port_mode.vecs - TV0)
        M2 = K @ N2
        N1 = K @ M1
        
        M = torch.cat((torch.cat((M1, M2), dim=1), torch.cat((M2, M1), dim=1)), dim=0) # (2n x 2m)
        N = torch.cat((N1, N2), dim = 0)
        
        # this saves more memory compared to solving a least square problem
        # slightly different than our paper, but equivalent
        # here we solve the linear system M @ X = N using QR decomposition
        # as it is more general and even work if the matrix is not square
        mQ, mR = torch.linalg.qr(M) 
        mQ_mH_N = mQ.mH @ N 
        TR = torch.linalg.solve_triangular(mR, mQ_mH_N, upper = True) 
        k = N.shape[1]
        T, R = torch.split(TR, k)
        self.smat = torch.cat((TR, torch.cat((R, T), dim = 0)), dim = 1)

        self.half_dim = self.smat.shape[0] // 2
        self.left_mode = port_mode
        self.right_mode = port_mode

    # construct the scattering matrix represented in the port modes
    def port_project(self, port_mode, coeff):
        '''
        port_mode: the mode that the scattering matrix is projected to (the two facesheets)
        coeff: the coefficient matrix derived from the central part (the core)

        You need to call compute(mode, z) method before calling this method
        where mode is derived from coeff using matrix_mode
        and z is the thickness of the core
        '''
        invW_PQ_W = self.invW @ coeff.matrix_pq() @ self.W
        # a hacky way to do differentiable matrix square root with the square root of diagonal alreay computed
        Omega = sqrtm_ops_core(invW_PQ_W, self.valsqrt.detach())
        invQV0 = torch.linalg.solve(coeff.Q, port_mode.dual_vecs)
        TV0 = (self.W @ Omega) @ (self.invW @ invQV0)

        # differentiable matrix exponential is available after PyTorch 1.7.0
        # we wrote differentiable Pade approximation manually before since it was not available,
        # you can check our cpp code for that implementation
        K = self.W @ torch.linalg.matrix_exp(1j * self.z / self.k0 * Omega)

        M1 = port_mode.vecs + TV0
        N2 = - (port_mode.vecs - TV0)
        M2 = K @ (self.invW @ N2)
        N1 = K @ (self.invW @ M1)
        
        M = torch.cat((torch.cat((M1, M2), dim=1), torch.cat((M2, M1), dim=1)), dim=0) # (2n x 2m)
        N = torch.cat((N1, N2), dim = 0)
        
        # this saves more memory compared to solving a least square problem
        # slightly different than our paper, but equivalent
        # here we solve the linear system M @ X = N using QR decomposition
        # as it is more general and even work if the matrix is not square
        mQ, mR = torch.linalg.qr(M) 
        mQ_mH_N = mQ.mH @ N 
        TR = torch.linalg.solve_triangular(mR, mQ_mH_N, upper = True) 
        k = N.shape[1]
        T, R = torch.split(TR, k)
        self.smat = torch.cat((TR, torch.cat((R, T), dim = 0)), dim = 1)

        self.half_dim = self.smat.shape[0] // 2
        self.left_mode = port_mode
        self.right_mode = port_mode

    def Tuu(self):
        return self.smat[:self.half_dim, :self.half_dim].clone()
    
    def Tdd(self):
        return self.smat[self.half_dim:, self.half_dim:].clone()
    
    def Rud(self):
        return self.smat[:self.half_dim, self.half_dim:].clone()
    
    def Rdu(self):
        return self.smat[self.half_dim:, :self.half_dim].clone()