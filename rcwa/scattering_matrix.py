import torch
from .diff_matfunc import sqrtm_ops_core

class ScatteringMatrix:
    def __init__(self):
        pass
    
    @staticmethod
    def allocate(half_dim, device = "cpu"):
        smat = ScatteringMatrix()
        smat.half_dim = half_dim
        smat.smat = torch.eye(2 * half_dim, device = device, dtype = torch.complex128)
        return smat

    def compute(self, mode, z):
        self.W = mode.vecs
        self.invW = mode.left_vecs
        self.valsqrt = mode.valsqrt
        self.V = mode.dual_vecs
        self.k0 = mode.k0
        self.z = z
        self.ndim = mode.vecs.shape[0]
        self.nmode = mode.valsqrt.shape[0]
        self.propogator = torch.exp(1j * z / self.k0 * mode.valsqrt)

    # construct the scattering matrix represented in the port modes
    def port_project(self, port_mode, coeff):
        invW_PQ_W = self.invW @ coeff.matrix_pq() @ self.W
        Omega = sqrtm_ops_core(invW_PQ_W, self.valsqrt.detach())
        invQV0 = torch.linalg.solve(coeff.Q, port_mode.dual_vecs)
        TV0 = (self.W @ Omega) @ (self.invW @ invQV0)
        K = self.W @ torch.linalg.matrix_exp(1j * self.z / self.k0 * Omega)

        M1 = port_mode.vecs + TV0
        N2 = - (port_mode.vecs - TV0)
        M2 = K @ (self.invW @ N2)
        N1 = K @ (self.invW @ M1)
        
        M = torch.cat((torch.cat((M1, M2), dim=1), torch.cat((M2, M1), dim=1)), dim=0) # (2n x 2m)
        N = torch.cat((N1, N2), dim = 0)
        
        # this saves more memory compared to solving a least square problem
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