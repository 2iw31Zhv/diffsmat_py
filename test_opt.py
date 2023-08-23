import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from rcwa.maxwell_coeff import *
from rcwa.maxwell_mode import *
from rcwa.scattering_matrix import *
import matplotlib.pyplot as plt
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

nx = ny = 8 # half of the harmonics along x and y directions
# number of harmonics is (2 * nx + 1) x (2 * ny + 1)
nx_grid = 20 # grid number along x direction, we use analytical Fourier transform, so nx_grid can be very small
n_opt = 10 # number of optimization grid along one direction
wavelength = 1.55
Lx = 1. # period
Ly = 1. # period
n_mode = 2 # number of modes to be optimized
ny_grid = int(nx_grid * Ly / Lx) # grid number along y direction
k0 = 2 * np.pi / wavelength # free space wavevector

eps_in = torch.tensor(3.48*3.48+0j, device = device, requires_grad = False, dtype = torch.complex128)
eps_out = torch.tensor(1.+0j, device = device, requires_grad = False, dtype = torch.complex128)

de = 0.5 * torch.ones(2*n_opt//2, 2*n_opt//2, device = device, dtype = torch.float64)
de.requires_grad_(True)

coeff = MaxwellCoeff(nx, ny, Lx, Ly, device = device)
port_mode = MaxwellMode()
port_mode.compute_in_vacuum(wavelength, coeff, device = device)
neff_port = port_mode.valsqrt.real / k0 / k0
neff_port, port_indices = torch.sort(neff_port, descending = True)
select_indices = port_indices[:n_mode]

def get_permittivity(de):
    ex = eps_out * torch.ones(nx_grid, ny_grid, device = device, dtype = torch.float64)
    ex[nx_grid//2 - n_opt//2 : nx_grid//2 + n_opt//2, ny_grid//2 - n_opt//2 : ny_grid//2 + n_opt//2] += (eps_in - eps_out) * de
    return ex

def compute_loss(de):
    ex = get_permittivity(de)
    coeff.compute(wavelength, ex, device = device)
    mode = MaxwellMode()
    mode.compute(coeff, device = device)
    smat = ScatteringMatrix()
    smat.compute(mode, 1.)
    # smat.port_project_v2(port_mode, coeff)
    smat.port_project(port_mode, coeff)
    Tuu_2 = torch.abs(smat.Tuu())**2
    ind_0 = select_indices[0]
    ind_1 = select_indices[1]
    print(Tuu_2[ind_0, ind_0].item(), Tuu_2[ind_1, ind_1].item())
    # maximize the transmission of one polarization
    # minimize the transmission of the other polarization
    return -Tuu_2[ind_0, ind_0] + Tuu_2[ind_1, ind_1]

plt.imshow(get_permittivity(de).detach().cpu().numpy().real)
plt.savefig("de_init.png")
plt.close()

loss_history = []
niters = 20
optimizer = optim.Adam([de], lr=5e-2)
t1 = time.perf_counter()
for i in range(niters):
    optimizer.zero_grad()
    loss = compute_loss(de)
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()
    de.data = torch.clamp(de.data, 0., 1.)
    print("i, de, loss = ", i, loss.item())
t2 = time.perf_counter()
print("time = ", t2 - t1)

# np.save("loss_history_v2.npy", np.array(loss_history))
loss_v2 = np.load("loss_history_v2.npy")
plt.imshow(get_permittivity(de).detach().cpu().numpy().real)
plt.savefig("de_final.png")
plt.close()

plt.plot(loss_v2, label = "Lorentzian broadening")
plt.plot(loss_history, label = "our method")
plt.legend()
plt.xlabel("iteration")
plt.ylabel("loss")
plt.savefig("loss_history.png")
plt.close()