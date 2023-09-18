import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import maxpy.rcwa as rcwa
import matplotlib.pyplot as plt
import maxpy.geometry as geometry
from maxpy.geometry.utils import assign_permittivity_distribution
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

nx = ny = 8 # half of the harmonics along x and y directions
# number of harmonics is (2 * nx + 1) x (2 * ny + 1)
nx_grid = 100 # grid number along x direction, we use analytical Fourier transform, so nx_grid can be very small
n_opt = 10 # number of optimization grid along one direction
wavelength = 1.55
tolerance = 1e-2
Lx = 1. # period
Ly = 1. # period
n_mode = 2 # number of modes to be optimized
ny_grid = int(nx_grid * Ly / Lx) # grid number along y direction
k0 = 2 * np.pi / wavelength # free space wavevector

eps_in = torch.tensor(3.48*3.48+0j, device = device, requires_grad = False, dtype = torch.complex128)
eps_out = torch.tensor(1.+0j, device = device, requires_grad = False, dtype = torch.complex128)

coeff = rcwa.MaxwellCoeff(nx, ny, Lx, Ly, device = device)
port_mode = rcwa.MaxwellMode()
port_mode.compute_in_vacuum(wavelength, coeff, device = device)
neff_port = port_mode.valsqrt.real / k0 / k0
neff_port, port_indices = torch.sort(neff_port, descending = True)
select_indices = port_indices[:n_mode]

def compute_loss(parameters):
    rect1 = geometry.RectangleComponent(parameters[0], parameters[1], parameters[2], parameters[3], eps_in, eps_out, tolerance = tolerance)
    rect2 = geometry.RectangleComponent(parameters[4], parameters[5], parameters[6], parameters[7], eps_in, eps_out, tolerance = tolerance)
    component = geometry.OrComponent(rect1, rect2)
    ex, _, _ = assign_permittivity_distribution(nx_grid, ny_grid, Lx, Ly, component, device = device)
    coeff.compute(wavelength, ex, device = device)
    mode = rcwa.MaxwellMode()
    mode.compute(coeff, device = device)
    smat = rcwa.ScatteringMatrix()
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

parameters = torch.tensor([0.2, 0.7, 0.2, 0.7, 0.3, 0.8, 0.3, 0.8], device = device, requires_grad = True, dtype = torch.float64)
rect1 = geometry.RectangleComponent(parameters[0], parameters[1], parameters[2], parameters[3], eps_in, eps_out, tolerance = tolerance)
rect2 = geometry.RectangleComponent(parameters[4], parameters[5], parameters[6], parameters[7], eps_in, eps_out, tolerance = tolerance)
component = geometry.OrComponent(rect1, rect2)
ex, _, _ = assign_permittivity_distribution(nx_grid, ny_grid, Lx, Ly, component, device = device)
plt.imshow(ex.detach().cpu().numpy().real)
plt.savefig("de_initial_boolean.png")
plt.close()

loss_history = []
niters = 20
optimizer = optim.Adam([parameters], lr=1e-3)
t1 = time.perf_counter()
for i in range(niters):
    optimizer.zero_grad()
    loss = compute_loss(parameters)
    loss_history.append(loss.item())
    loss.backward()
    optimizer.step()
    parameters.data = torch.clamp(parameters.data, 0.1, 0.9)
    print("i, de, loss, parameters = ", i, loss.item(), parameters.data)
t2 = time.perf_counter()
print("time = ", t2 - t1)

rect1 = geometry.RectangleComponent(parameters[0], parameters[1], parameters[2], parameters[3], eps_in, eps_out, tolerance = tolerance)
rect2 = geometry.RectangleComponent(parameters[4], parameters[5], parameters[6], parameters[7], eps_in, eps_out, tolerance = tolerance)
component = geometry.OrComponent(rect1, rect2)
ex, _, _ = assign_permittivity_distribution(nx_grid, ny_grid, Lx, Ly, component, device = device)
plt.imshow(ex.detach().cpu().numpy().real)
plt.savefig("de_final_boolean.png")
plt.close()

plt.plot(loss_history, label = "our method")
plt.xlabel("iteration")
plt.ylabel("loss")
plt.savefig("loss_history_boolean.png")
plt.close()