import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from maxwell_coeff import *
from maxwell_mode import *
from scattering_matrix import *
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")


nx = ny = 3 # half of the harmonics along x and y directions
nx_grid = 4 # grid number along x direction, we use analytical Fourier transform, so nx_grid can be very small
n_opt = 2 # number of optimization grid along one direction
wavelength = 1.55
Lx = 1. # period
Ly = 1. # period
n_mode = 2 # number of modes to be optimized
ny_grid = int(nx_grid * Ly / Lx) # grid number along y direction
k0 = 2 * np.pi / wavelength # free space wavevector

eps_in = torch.tensor(3.48*3.48+0j, device = device, requires_grad = False, dtype = torch.complex128)
eps_out = torch.tensor(1.+0j, device = device, requires_grad = False, dtype = torch.complex128)

de = torch.rand(2*n_opt//2, 2*n_opt//2, device = device, dtype = torch.float64)
de.requires_grad_(True)

def compute_loss(de):
    ex = eps_out * torch.ones(nx_grid, ny_grid, device = device, dtype = torch.float64)
    ex[nx_grid//2 - n_opt//2 : nx_grid//2 + n_opt//2, ny_grid//2 - n_opt//2 : ny_grid//2 + n_opt//2] += (eps_in - eps_out) * de
    coeff = MaxwellCoeff(nx, ny, Lx, Ly, device = device)
    port_mode = MaxwellMode()
    port_mode.compute_in_vacuum(wavelength, coeff.kx, coeff.ky, device = device)
    neff_port = port_mode.valsqrt.real / k0 / k0
    neff_port, port_indices = torch.sort(neff_port, descending = True)
    coeff.compute(wavelength, ex, device = device)
    mode = MaxwellMode()
    mode.compute(coeff, device = device)
    smat = ScatteringMatrix()
    smat.compute(mode, 1.)
    smat.port_project(port_mode, coeff)
    smatsqr = torch.abs(smat.smat)**2
    smatsqr = smatsqr[port_indices[:n_mode], port_indices[:n_mode]]
    return -torch.sum(torch.diag(smatsqr))

# check if the gradient is correct
print(torch.autograd.gradcheck(compute_loss, (de)))

loss_history = []
niters = 10
optimizer = optim.Adam([de], lr=1e-3)
for i in range(niters):
    optimizer.zero_grad()
    loss = compute_loss(de)
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()
    de.data = torch.clamp(de.data, 0., 1.)
    # t5 = time.perf_counter()
    print("i, de, loss = ", i, de.flatten().detach().cpu().numpy(), loss.item())

plt.plot(loss_history)
plt.savefig("loss_history.png")
plt.close()