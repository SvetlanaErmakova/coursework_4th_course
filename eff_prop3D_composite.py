import cupy as cp
import numpy as np
import time

def av4_xy(A):
    return 0.25 * (A[:-1, :-1, :] + A[:-1, 1:, :] + A[1:, :-1, :] + A[1:, 1:, :])
def av4_xz(A):
    return 0.25*(A[:-1, :, :-1] + A[:-1, :, 1:] + A[1:, :, :-1] + A[1:, :, 1:])
def av4_yz(A):
    return 0.25*(A[:, :-1, :-1] + A[:, :-1, 1:] + A[:, 1:, :-1] + A[:, 1:, 1:])
#
start_numpy = time.time()

# PHYSICS
Lx = 1.0
Ly = 1.0
Lz = 1.0

E0n = 10.0
nu0n = 0.25
alp0n = 1.3e-5

rho = 1.0
K0 = E0n / (3.0 * (1 - 2 * nu0n))
G0 = E0n / (2.0 + 2.0 * nu0n)

# NUMERICS
Nx = 200
Ny = 200
Nz = 200

Nt = 10000
CFL = 0.25

# PREPROCESSING
dX = Lx / (Nx - 1)
dY = Ly / (Ny - 1)
dZ = Lz / (Nz - 1)
x = cp.linspace(-Lx/2, Lx/2, Nx)
y = cp.linspace(-Ly/2, Ly/2, Ny)
z = cp.linspace(-Lz/2, Lz/2, Nz)
x, y, z = cp.meshgrid(x, y, z)

xUx, yUx, zUx = cp.meshgrid( cp.linspace(-(Lx + dX)/2, (Lx + dX)/2, Nx+1),
                             cp.linspace(-Ly/2, Ly/2, Ny), 
                             cp.linspace(-Lz/2, Lz/2, Nz), indexing='ij')

xUy, yUy, zUy = cp.meshgrid( cp.linspace(-Lx/2, Lx/2, Nx), 
                             cp.linspace(-(Ly + dY)/2, (Ly + dY)/2, Ny+1), 
                             cp.linspace(-Lz/2, Lz/2, Nz), indexing='ij')

xUz, yUz, zUz = cp.meshgrid(cp.linspace(-Lx/2, Lx/2, Nx), 
                             cp.linspace(-Ly/2, Ly/2, Ny), 
                             cp.linspace(-(Lz + dZ)/2, (Lz + dZ)/2, Nz+1), indexing='ij')


dt = CFL * min(dX, min(dY, dZ)) / cp.sqrt((K0 + 4*G0/3) / rho)
damp = 4 / dt / Nx

# MATERIALS
E0p = 1.0
nu0p = 0.4
alp0p = 7.7e-5

E = E0p * cp.ones((Nx, Ny, Nz))
nu = nu0p * cp.ones((Nx, Ny, Nz))
alp = alp0p * cp.ones((Nx, Ny, Nz))

radios = 0.228542449538
indices = cp.sqrt(x**2 + y**2 + z**2) < radios
E[indices] = E0n
nu[indices] = nu0n
alp[indices] = alp0n

K = E / (3.0 * (1 - 2 * nu))
G = E / (2.0 + 2.0 * nu)

# INITIAL CONDITIONS
P0 = cp.zeros((Nx, Ny, Nz))
Ux = cp.zeros((Nx + 1, Ny, Nz))
Uy = cp.zeros((Nx, Ny + 1, Nz))
Uz = cp.zeros((Nx, Ny, Nz + 1))

Vx = cp.zeros((Nx + 1, Ny, Nz))
Vy = cp.zeros((Nx, Ny + 1, Nz))
Vz = cp.zeros((Nx, Ny, Nz + 1))

tauxx = cp.zeros((Nx, Ny, Nz))
tauyy = cp.zeros((Nx, Ny, Nz))
tauzz = cp.zeros((Nx, Ny, Nz))

tauxy = cp.zeros((Nx - 1, Ny - 1, Nz))
tauxz = cp.zeros((Nx - 1, Ny, Nz - 1))
tauyz = cp.zeros((Nx, Ny - 1, Nz - 1))

# BOUNDARY CONDITIONS
loadValue = 0.002
loadType=[1, 0, 0, 0, 0, 0]

dUxdx = loadValue * loadType[0]
dUydy = loadValue * loadType[1]
dUzdz = loadValue * loadType[2]

dUxdy = loadValue * loadType[3]
dUxdz = loadValue * loadType[4]
dUydz = loadValue * loadType[5]

Ux += dUxdx * xUx + dUxdy * yUx
Uy += dUydy * yUy
Uz += dUzdz * zUz

# ACTION LOOP
for it in range(Nt):
    # displacement divergence
    divU = cp.diff(Ux, axis=0) / dX + cp.diff(Uy, axis=1) / dY + cp.diff(Uz, axis=2) / dZ

    # constitutive equation - Hooke's law
    P = P0 - K * divU

    tauxx = 2.0 * G * (cp.diff(Ux, axis=0) / dX - divU/3.0)
    tauyy = 2.0 * G * (cp.diff(Uy, axis=1) / dY - divU/3.0)
    tauzz = 2.0 * G * (cp.diff(Uz, axis=2) / dZ - divU/3.0)

    tauxy = av4_xy(G) * (cp.diff(Ux[1:-1,:,:], axis=1) / dY + cp.diff(Uy[:,1:-1,:], axis=0) / dX)
    tauxz = av4_xz(G) * (cp.diff(Ux[1:-1,:,:], axis=2) / dZ + cp.diff(Uz[:,:,1:-1], axis=0) / dX)
    tauyz = av4_yz(G) * (cp.diff(Uy[:,1:-1,:], axis=2) / dZ + cp.diff(Uz[:,:,1:-1], axis=1) / dY)

    # motion equation
    dVxdt = (cp.diff(-P[:,1:-1,1:-1] + tauxx[:,1:-1,1:-1], axis=0) / dX + 
             cp.diff(tauxy[:,:,1:-1], axis=1) / dY + 
             cp.diff(tauxz[:,1:-1,:], axis=2) / dZ) / rho
    Vx[1:-1,1:-1,1:-1] = Vx[1:-1,1:-1,1:-1] * (1 - dt * damp) + dVxdt * dt

    dVydt = (cp.diff(tauxy[:,:,1:-1], axis=0) / dX + 
             cp.diff(-P[1:-1,:,1:-1] + tauyy[1:-1,:,1:-1], axis=1) / dY + 
             cp.diff(tauyz[1:-1,:,:], axis=2) / dZ) / rho
    Vy[1:-1,1:-1,1:-1] = Vy[1:-1,1:-1,1:-1] * (1 - dt * damp) + dVydt * dt

    dVzdt = (cp.diff(tauxz[:,1:-1,:], axis=0) / dX + 
             cp.diff(tauyz[1:-1,:,:], axis=1) / dY + 
             cp.diff(-P[1:-1,1:-1,:] + tauzz[1:-1,1:-1,:], axis=2) / dZ) / rho
    Vz[1:-1,1:-1,1:-1] = Vz[1:-1,1:-1,1:-1] * (1 - dt * damp) + dVzdt * dt

    # displacements
    Ux = Ux + Vx * dt
    Uy = Uy + Vy * dt
    Uz = Uz + Vz * dt

Keff = cp.mean(-P) / (dUxdx + dUydy + dUzdz)
print(f'Keff={Keff}')

end_numpy = time.time()
time_numpy = end_numpy - start_numpy
print(f"Время вычисление Keff с cupy: {time_numpy} секунд")

#
start_numpy = time.time()
# INITIAL CONDITIONS
T = 1.0
P0 = K * alp * 2 * T

Ux = cp.zeros((Nx + 1, Ny, Nz))  # displacement
Uy = cp.zeros((Nx, Ny + 1, Nz))
Uz = cp.zeros((Nx, Ny, Nz + 1))

Vx = cp.zeros((Nx + 1, Ny, Nz))  # velocity
Vy = cp.zeros((Nx, Ny + 1, Nz))
Vz = cp.zeros((Nx, Ny, Nz + 1))

tauxx = cp.zeros((Nx, Ny, Nz))  # deviatoric stress
tauyy = cp.zeros((Nx, Ny, Nz))
tauzz = cp.zeros((Nx, Ny, Nz))

tauxy = cp.zeros((Nx - 1, Ny - 1, Nz))
tauxz = cp.zeros((Nx - 1, Ny, Nz - 1))
tauyz = cp.zeros((Nx, Ny - 1, Nz - 1))

# BOUNDARY CONDITIONS
loadValue = 0.0
loadType = [1, 0, 0, 0, 0, 0]

dUxdx = loadValue * loadType[0]
dUydy = loadValue * loadType[1]
dUzdz = loadValue * loadType[2]

dUxdy = loadValue * loadType[3]
dUxdz = loadValue * loadType[4]
dUydz = loadValue * loadType[5]

Ux = Ux + (dUxdx * xUx + dUxdy * yUx)
Uy = Uy + dUydy * yUy
Uz = Uz + dUzdz * zUz

# ACTION LOOP
for it in range(1, Nt+1):
    # displacement divergence
    divU = cp.diff(Ux, axis=0) / dX + cp.diff(Uy, axis=1) / dY + cp.diff(Uz, axis=2) / dZ

    # constitutive equation - Hooke's law
    P = P0 - K * divU

    tauxx = 2.0 * G * (cp.diff(Ux, axis=0) / dX - divU/3.0)
    tauyy = 2.0 * G * (cp.diff(Uy, axis=1) / dY - divU/3.0)
    tauzz = 2.0 * G * (cp.diff(Uz, axis=2) / dZ - divU/3.0)

    tauxy = av4_xy(G) * (cp.diff(Ux[1:-1,:,:], axis=1) / dY + cp.diff(Uy[:,1:-1,:], axis=0) / dX)
    tauxz = av4_xz(G) * (cp.diff(Ux[1:-1,:,:], axis=2) / dZ + cp.diff(Uz[:,:,1:-1], axis=0) / dX)
    tauyz = av4_yz(G) * (cp.diff(Uy[:,1:-1,:], axis=2) / dZ + cp.diff(Uz[:,:,1:-1], axis=1) / dY)

    # motion equation
    dVxdt = (cp.diff(-P[:,1:-1,1:-1] + tauxx[:,1:-1,1:-1], axis=0) / dX + 
             cp.diff(tauxy[:,:,1:-1], axis=1) / dY + 
             cp.diff(tauxz[:,1:-1,:], axis=2) / dZ) / rho
    Vx[1:-1,1:-1,1:-1] = Vx[1:-1,1:-1,1:-1] * (1 - dt * damp) + dVxdt * dt

    dVydt = (cp.diff(tauxy[:,:,1:-1], axis=0) / dX + 
             cp.diff(-P[1:-1,:,1:-1] + tauyy[1:-1,:,1:-1], axis=1) / dY + 
             cp.diff(tauyz[1:-1,:,:], axis=2) / dZ) / rho
    Vy[1:-1,1:-1,1:-1] = Vy[1:-1,1:-1,1:-1] * (1 - dt * damp) + dVydt * dt

    dVzdt = (cp.diff(tauxz[:,1:-1,:], axis=0) / dX + 
             cp.diff(tauyz[1:-1,:,:], axis=1) / dY + 
             cp.diff(-P[1:-1,1:-1,:] + tauzz[1:-1,1:-1,:], axis=2) / dZ) / rho
    Vz[1:-1,1:-1,1:-1] = Vz[1:-1,1:-1,1:-1] * (1 - dt * damp) + dVzdt * dt

    # displacements
    Ux = Ux + Vx * dt
    Uy = Uy + Vy * dt
    Uz = Uz + Vz * dt


alpha = cp.mean(P) / Keff / T / 2
print(f'alpha={alpha}')

end_numpy = time.time()
time_numpy = end_numpy - start_numpy
print(f"Время вычисление alpha с cupy: {time_numpy} секунд")
