import unittest
import torch

def redheffer_star_product(lhs, rhs):
    invC1 = -rhs.Rdu() @ lhs.Rud()
    invC1.diagonal().add_(1.)

    C1_rdu = torch.linalg.solve(invC1, rhs.Rdu())
    C1_tdd = torch.linalg.solve(invC1, rhs.Tdd())

    C2 = lhs.Rud() @ C1_rdu
    C2.diagonal().add_(1.)

    Rdu = lhs.Rdu() + lhs.Tdd() @ rhs.Rdu() @ C2 @ lhs.Tuu()
    Tdd = lhs.Tdd() @ C1_tdd
    Rud = rhs.Rud() + rhs.Tuu() @ lhs.Rud() @ C1_tdd
    Tuu = rhs.Tuu() @ C2 @ lhs.Tuu()

    smat = type(lhs)()
    smat.half_dim = lhs.half_dim
    smat.smat = torch.cat((torch.cat((Tuu, Rud), dim = 1), torch.cat((Rdu, Tdd), dim = 1)), dim = 0)
    
    return smat

from maxpy.rcwa.scattering_matrix import ScatteringMatrix

class TestRedheffer(unittest.TestCase):
    def test1(self):
        s1 = ScatteringMatrix.allocate(2)
        s1.smat[:2, :2] = torch.tensor([[0.4, 0.6], [0.6, 0.4]])
        s2 = ScatteringMatrix.allocate(2)
        print(s1.smat)
        print(s2.smat)
        s = redheffer_star_product(s1, s2)
        self.assertTrue(torch.allclose(s1.smat, s.smat))

if __name__ == '__main__':
    unittest.main()