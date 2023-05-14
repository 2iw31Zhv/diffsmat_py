import time
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from maxwell_coeff import *
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    print("Using GPU")
else:
    print("Using CPU")

nx = 20
n_opt = 6
wavelength = 1.55
Lx = 1.
Ly = 1.
ny = int(nx * Ly / Lx)
k0 = 2 * np.pi / wavelength

x0 = torch.tensor(0.233, requires_grad = False, device = device, dtype = torch.float64)
x1 = torch.tensor(0.766, requires_grad = False, device = device, dtype = torch.float64)
y0 = torch.tensor(0.4, requires_grad = False, device = device, dtype = torch.float64)
y1 = torch.tensor(0.6, requires_grad = False, device = device, dtype = torch.float64)

eps_in = torch.tensor(3.48*3.48+0j, device = device, requires_grad = False, dtype = torch.complex128)
eps_out = torch.tensor(1.445*1.445+0j, device = device, requires_grad = False, dtype = torch.complex128)

de = 0.5 * torch.ones(2*n_opt//2, 2*n_opt//2, device = device, dtype = torch.float64)
de.requires_grad_(True)

ex = eps_out * torch.ones(nx, ny, device = device, dtype = torch.float64)
ex[nx//2 - n_opt//2 : nx//2 + n_opt//2, ny//2 - n_opt//2 : ny//2 + n_opt//2] += (eps_in - eps_out) * de

coeff = MaxwellCoeff(nx, ny, Lx, Ly, device = device)
coeff.compute(wavelength, ex, device = device)

plt.imshow(ex.detach().cpu().numpy().real)
plt.savefig("ex.png")
plt.close()

plt.imshow(coeff.fe2D.detach().cpu().numpy().real)
plt.savefig("fe2D.png")
plt.close()

plt.imshow(coeff.fe.detach().cpu().numpy().real)
plt.savefig("fe.png")
plt.close()
