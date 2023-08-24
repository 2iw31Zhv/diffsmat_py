import torch

def assign_permittivity_distribution(nx, ny, Lx, Ly, component, device):
    dx = Lx / nx
    dy = Ly / ny
    h = torch.tensor([dx, dy], device = device, dtype=torch.float64)
    xs = torch.linspace(0., Lx-dx, nx, requires_grad = False, device = device, dtype=torch.float64)
    ys = torch.linspace(0., Ly-dy, ny, requires_grad = False, device = device, dtype=torch.float64)
    meshxs, meshys = torch.meshgrid(xs, ys, indexing = "xy")
    ex = component.smoothen(meshxs, meshxs+dx, meshys-0.5*dy, meshys+0.5*dy, "ex", h = h)
    ey = component.smoothen(meshxs-0.5*dx, meshxs+0.5*dx, meshys, meshys+dy, "ey", h = h)
    xs = torch.linspace(0., Lx, nx+1, requires_grad = False, device = device, dtype=torch.float64)
    ys = torch.linspace(0., Ly, ny+1, requires_grad = False, device = device, dtype=torch.float64)
    meshxs, meshys = torch.meshgrid(xs, ys, indexing = "xy")
    inv_ez = 1. / component.smoothen(meshxs-0.5*dx, meshxs+0.5*dx, meshys-0.5*dy, meshys+0.5*dy, "ez", h = h)
    return ex, ey, inv_ez