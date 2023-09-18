import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import maxpy.rcwa as rcwa
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")
nh_min = 3
nh_max = 20
neff_list = []
for nh in range(nh_min, nh_max):
    nx = ny = nh # half of the harmonics along x and y directions
    nx_grid = 100 # grid number along x direction, we use analytical Fourier transform, so nx_grid can be very small
    n_opt = 20 # number of optimization grid along one direction
    wavelength = 1.55
    Lx = 1. # period
    Ly = 1. # period
    n_mode = 2 # number of modes to be optimized
    ny_grid = int(nx_grid * Ly / Lx) # grid number along y direction
    k0 = 2 * np.pi / wavelength # free space wavevector

    # the foreground permittivity, silicon
    eps_in = torch.tensor(3.48*3.48+0j, device = device, requires_grad = False, dtype = torch.complex128)
    # the background permittivity, air
    eps_out = torch.tensor(1.+0j, device = device, requires_grad = False, dtype = torch.complex128)

    # set the initial value of the optimized permittivity to be 0.5 * (eps_in + eps_out)
    de = 0.5 * torch.ones(2*n_opt//2, 2*n_opt//2, device = device, dtype = torch.float64)
    de.requires_grad_(True)
    ex = eps_out * torch.ones(nx_grid, ny_grid, device = device, dtype = torch.float64)
    ex[nx_grid//2 - n_opt//2 : nx_grid//2 + n_opt//2, ny_grid//2 - n_opt//2 : ny_grid//2 + n_opt//2] += (eps_in - eps_out) * de
    
    coeff_fff = rcwa.MaxwellCoeff(nx, ny, Lx, Ly, device = device)
    coeff_fff.compute(wavelength, ex, device = device)
    mode = rcwa.MaxwellMode()
    mode.compute(coeff_fff, device = device)
    neff_fff = mode.valsqrt.real / k0 / k0
    neff_fff, indices = torch.sort(neff_fff, descending = True)

    coeff = rcwa.MaxwellCoeff(nx, ny, Lx, Ly, device = device)
    coeff.setfff(False) # turn off fast Fourier factorization
    coeff.compute(wavelength, ex, device = device)
    mode.compute(coeff, device = device)
    neff = mode.valsqrt.real / k0 / k0
    neff, indices = torch.sort(neff, descending = True)
    neff_list.append([neff_fff[0].item(), neff[0].item()])

plt.plot(range(nh_min, nh_max), np.array(neff_list)[:, 0], label = "fff")
plt.plot(range(nh_min, nh_max), np.array(neff_list)[:, 1], label = "direct")
plt.xlabel("half number of harmonics")
plt.ylabel("effective index")
plt.legend()
plt.savefig("convergence.png")
plt.close()
