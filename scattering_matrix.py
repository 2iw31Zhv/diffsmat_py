import torch
import matplotlib.pyplot as plt
from diff_matfunc import spsolve, sqrtm_core, sqrtm, sqrtm_ops_core
from linear_operator.operators import MatmulLinearOperator

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
    def port_project(self, port_mode, matrix_p, matrix_q, mumps_context = None):
        invW_PQ_W = MatmulLinearOperator(self.invW, matrix_p @ matrix_q @ self.W)
        Omega = sqrtm_ops_core(invW_PQ_W, self.valsqrt.detach())
        # QW = matrix_q @ self.W 
        # TV0 = (self.W @ Omega) @ torch.linalg.lstsq(QW, port_mode.dual_vecs).solution
        invQV0 = spsolve(matrix_q, port_mode.dual_vecs, mumps_context)
        TV0 = (self.W @ Omega) @ (self.invW @ invQV0) # (m x n) (n x 2) 
        
        K = self.W @ torch.linalg.matrix_exp(1j * self.z / self.k0 * Omega)

        M1 = port_mode.vecs + TV0
        N2 = - (port_mode.vecs - TV0)
        M2 = K @ (self.invW @ N2)
        N1 = K @ (self.invW @ M1)
        M = torch.cat((torch.cat((M1, M2), dim=1), torch.cat((M2, M1), dim=1)), dim=0) # (2n x 2m)
        N = torch.cat((N1, N2), dim = 0) # (2n x m)
        
        # M1 = port_mode.vecs + TV0
        # M2 = - K @ (self.invW @ (port_mode.vecs - TV0))
        # M = torch.cat((torch.cat((M1, M2), dim=1), torch.cat((M2, M1), dim=1)), dim=0) # (2n x 2m)

        # N1 = K @ (self.invW @ (port_mode.vecs + TV0))
        # N2 = - (port_mode.vecs - TV0)        
        # N = torch.cat((N1, N2), dim = 0) # (2n x m)

        # this saves more memory compared to solving a least square problem
        mQ, mR = torch.linalg.qr(M) # (2n x 2m) (2m x 2m)
        mQ_mH_N = mQ.mH @ N # (2m x m)
        TR = torch.linalg.solve_triangular(mR, mQ_mH_N, upper = True) # (2m x m)
        k = N.shape[1]
        T, R = torch.split(TR, k)
        self.smat = torch.cat((TR, torch.cat((R, T), dim = 0)), dim = 1)

        self.half_dim = self.smat.shape[0] // 2
        self.left_mode = port_mode
        self.right_mode = port_mode

    def Tuu(self):
        return self.smat[:self.half_dim, :self.half_dim]
    
    def Tdd(self):
        return self.smat[self.half_dim:, self.half_dim:]
    
    def Rud(self):
        return self.smat[:self.half_dim, self.half_dim:]
    
    def Rdu(self):
        return self.smat[self.half_dim:, :self.half_dim]
    
    def setTuu(self, mat):
        self.smat[:self.half_dim, :self.half_dim] = mat

    def setTdd(self, mat):
        self.smat[self.half_dim:, self.half_dim:] = mat

    def setRud(self, mat):
        self.smat[:self.half_dim, self.half_dim:] = mat

    def setRdu(self, mat):
        self.smat[self.half_dim:, :self.half_dim] = mat

    def redheffer_rh_product(self, rhs):
        invC1 = -rhs.Rdu() @ self.Rud()
        invC1.diagonal().add_(1.)
        
        C2 = self.Rud() @ torch.linalg.solve(invC1, rhs.Rdu())
        C2.diagonal().add_(1.)
        C2_tuu = C2 @ self.Tuu()
        C1_tdd = torch.linalg.solve(invC1, rhs.Tdd())

        self.setRdu(torch.addmm(self.Rdu(), self.Tdd() @ rhs.Rdu(), C2_tuu))
        # self.setRdu(self.Rdu() + self.Tdd() @ rhs.Rdu() @ C2 @ self.Tuu())
        self.setTdd(self.Tdd() @ C1_tdd)
        self.setRud(rhs.Rud() + rhs.Tuu() @ self.Rud() @ C1_tdd)
        self.setTuu(rhs.Tuu() @ C2_tuu)