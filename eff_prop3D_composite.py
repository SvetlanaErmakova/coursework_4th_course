import time

import cupy as cp
import numpy as np

def timer(func):
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        print(f"Метод {func.__name__} работал {end_time - start_time:.2f} секунд")
        return result
    return wrapper

class eff_prop3D():

  def __init__(self, size, grid_points, CFL, rho, E0p, nu0p, alp0p, E0n, nu0n, alp0n, radios,
                stress_initial_cond, U_initial_cond, V_initial_cond, T, strain_bound_cond):
      #size
      self.Lx=size[0]
      self.Ly=size[1]
      self.Lz=size[2]

      #grid
      self.Nx=grid_points[0]
      self.Ny=grid_points[1]
      self.Nz=grid_points[2]
      self.Nt=grid_points[3]
      #CFL
      self.CFL=CFL
      #preprocessing
      self.__fill_preprocessing(E0p, nu0p, E0n, nu0n, rho)
      #material
      self.__fill_material(rho, E0p, nu0p, alp0p, E0n, nu0n, alp0n, radios)
      #initial condition
      self.set_initial_cond( stress_initial_cond, U_initial_cond, V_initial_cond, T)
      #boundary condition
      self.set_boundary_cond(strain_bound_cond )

  def __fill_preprocessing(self, E0p, nu0p, E0n, nu0n, rho):
      #для dt
      if E0p>E0n:
          E00=E0p
          nu00=nu0p
      else:
          E00=E0n
          nu00=nu0n

      self.K0 = E00 / (3.0 * (1 - 2 * nu00))
      self.G0 = E00 / (2.0 + 2.0 * nu00)
      #
      dX = self.Lx / (self.Nx - 1)
      dY = self.Ly / (self.Ny - 1)
      dZ = self.Lz / (self.Nz - 1)

      x = cp.linspace(-self.Lx/2, self.Lx/2, self.Nx)
      y = cp.linspace(-self.Ly/2, self.Ly/2, self.Ny)
      z = cp.linspace(-self.Lz/2, self.Lz/2, self.Nz)
      self.x, self.y, self.z = cp.meshgrid(x, y, z)

      self.xUx, self.yUx, self.zUx = cp.meshgrid( cp.linspace(-(self.Lx + dX)/2, (self.Lx + dX)/2, self.Nx+1),
                                  cp.linspace(-self.Ly/2, self.Ly/2, self.Ny),
                                  cp.linspace(-self.Lz/2, self.Lz/2, self.Nz), indexing='ij')

      self.xUy, self.yUy, self.zUy = cp.meshgrid( cp.linspace(-self.Lx/2, self.Lx/2, self.Nx),
                                  cp.linspace(-(self.Ly + dY)/2, (self.Ly + dY)/2, self.Ny+1),
                                  cp.linspace(-self.Lz/2, self.Lz/2, self.Nz), indexing='ij')

      self.xUz, self.yUz, self.zUz = cp.meshgrid(cp.linspace(-self.Lx/2, self.Lx/2, self.Nx),
                                  cp.linspace(-self.Ly/2, self.Ly/2, self.Ny),
                                  cp.linspace(-(self.Lz + dZ)/2, (self.Lz + dZ)/2, self.Nz+1), indexing='ij')
      self.dt = self.CFL * min(dX, min(dY, dZ)) / cp.sqrt((self.K0 + 4*self.G0/3) / rho)
      self.damp = 4 / self.dt / self.Nx

      self.dX=dX
      self.dY=dY
      self.dZ=dZ

  def __fill_material(self, rho, E0p, nu0p, alp0p, E0n, nu0n, alp0n, radios):
      self.rho = rho
      #
      E = E0p * cp.ones((self.Nx, self.Ny, self.Nz))
      nu = nu0p * cp.ones((self.Nx, self.Ny, self.Nz))
      alp = alp0p * cp.ones((self.Nx, self.Ny, self.Nz))
      indices = cp.sqrt(self.x**2 + self.y**2 + self.z**2) < radios
      E[indices] = E0n
      nu[indices] = nu0n
      alp[indices] = alp0n
      K = E / (3.0 * (1 - 2 * nu))
      G = E / (2.0 + 2.0 * nu)

      self.E=E
      self.nu=nu
      self.alp=alp
      self.K=K
      self.G=G

  def set_initial_cond(self, stress_initial_cond, U_initial_cond, V_initial_cond, T):
      self.P0=stress_initial_cond[0]
      self.tauxx=stress_initial_cond[1]
      self.tauyy=stress_initial_cond[2]
      self.tauzz=stress_initial_cond[3]

      self.tauxy=stress_initial_cond[4]
      self.tauxz=stress_initial_cond[5]
      self.tauyz=stress_initial_cond[6]

      self.Ux=U_initial_cond[0]
      self.Uy=U_initial_cond[1]
      self.Uz=U_initial_cond[2]

      self.Vx=V_initial_cond[0]
      self.Vy=V_initial_cond[1]
      self.Vz=V_initial_cond[2]

      self.T=T

  def set_boundary_cond(self, strain_bound_cond ):
      self.dUxdx=strain_bound_cond[0]
      self.dUydy=strain_bound_cond[1]
      self.dUzdz=strain_bound_cond[2]

      self.dUxdy=strain_bound_cond[3]
      self.dUxdz=strain_bound_cond[4]
      self.dUydz=strain_bound_cond[5]
  @timer
  def find_Keff(self):
      self.__solving_equation()
      Keff = cp.mean(-self.P) / (self.dUxdx + self.dUydy + self.dUzdz)
      print(f'Keff={Keff}')
      return Keff

  @timer
  def find_alpha(self, Keff):
      self.__solving_equation()
      rr=cp.mean(self.P)
      alpha = rr / Keff / self.T / 2
      print(f'Alpha={alpha}')
      return alpha

  def __av4_xy(self,A):
      return 0.25 * (A[:-1, :-1, :] + A[:-1, 1:, :] + A[1:, :-1, :] + A[1:, 1:, :])
  def __av4_xz(self,A):
      return 0.25*(A[:-1, :, :-1] + A[:-1, :, 1:] + A[1:, :, :-1] + A[1:, :, 1:])
  def __av4_yz(self,A):
      return 0.25*(A[:, :-1, :-1] + A[:, :-1, 1:] + A[:, 1:, :-1] + A[:, 1:, 1:])

  def __solving_equation(self):
      self.Ux += self.dUxdx * self.xUx + self.dUxdy * self.yUx
      self.Uy += self.dUydy * self.yUy
      self.Uz += self.dUzdz * self.zUz

      for it in range(self.Nt):
          # displacement divergence
          divU = cp.diff(self.Ux, axis=0) / self.dX + cp.diff(self.Uy, axis=1) / self.dY + cp.diff(self.Uz, axis=2) / self.dZ

          # constitutive equation - Hooke's law
          self.P = self.P0 - self.K * divU

          self.tauxx = 2.0 * self.G * (cp.diff(self.Ux, axis=0) / self.dX - divU/3.0)
          self.tauyy = 2.0 * self.G * (cp.diff(self.Uy, axis=1) / self.dY - divU/3.0)
          self.tauzz = 2.0 * self.G * (cp.diff(self.Uz, axis=2) / self.dZ - divU/3.0)

          self.tauxy = self.__av4_xy(self.G) * (cp.diff(self.Ux[1:-1,:,:], axis=1) / self.dY + cp.diff(self.Uy[:,1:-1,:], axis=0) / self.dX)
          self.tauxz = self.__av4_xz(self.G) * (cp.diff(self.Ux[1:-1,:,:], axis=2) / self.dZ + cp.diff(self.Uz[:,:,1:-1], axis=0) / self.dX)
          self.tauyz = self.__av4_yz(self.G) * (cp.diff(self.Uy[:,1:-1,:], axis=2) / self.dZ + cp.diff(self.Uz[:,:,1:-1], axis=1) / self.dY)

          # motion equation
          dVxdt = (cp.diff(-self.P[:,1:-1,1:-1] + self.tauxx[:,1:-1,1:-1], axis=0) / self.dX +
                  cp.diff(self.tauxy[:,:,1:-1], axis=1) / self.dY +
                  cp.diff(self.tauxz[:,1:-1,:], axis=2) / self.dZ) / self.rho
          self.Vx[1:-1,1:-1,1:-1] = self.Vx[1:-1,1:-1,1:-1] * (1 - self.dt * self.damp) + dVxdt * self.dt

          dVydt = (cp.diff(self.tauxy[:,:,1:-1], axis=0) / self.dX +
                  cp.diff(-self.P[1:-1,:,1:-1] + self.tauyy[1:-1,:,1:-1], axis=1) / self.dY +
                  cp.diff(self.tauyz[1:-1,:,:], axis=2) / self.dZ) / self.rho
          self.Vy[1:-1,1:-1,1:-1] = self.Vy[1:-1,1:-1,1:-1] * (1 - self.dt * self.damp) + dVydt * self.dt

          dVzdt = (cp.diff(self.tauxz[:,1:-1,:], axis=0) / self.dX +
                  cp.diff(self.tauyz[1:-1,:,:], axis=1) / self.dY +
                  cp.diff(-self.P[1:-1,1:-1,:] + self.tauzz[1:-1,1:-1,:], axis=2) / self.dZ) / self.rho
          self.Vz[1:-1,1:-1,1:-1] = self.Vz[1:-1,1:-1,1:-1] * (1 - self.dt * self.damp) + dVzdt * self.dt

          # displacements
          self.Ux = self.Ux + self.Vx * self.dt
          self.Uy = self.Uy + self.Vy * self.dt
          self.Uz = self.Uz + self.Vz * self.dt

def main():
    Lx = 1.0
    Ly = 1.0
    Lz = 1.0
    size=[Lx,Ly,Lz]

    # NUMERICS
    Nx = 100
    Ny = 100
    Nz = 100
    Nt = 1000
    grid_points=[Nx, Ny, Nz, Nt]
    CFL = 0.25

    #Material
    rho=1.0
    E0p=1.0
    nu0p=0.4
    alp0p=7.7e-5
    #hole
    # E0n=0.0
    # nu0n=0.0
    # alp0n=0.0
    E0n = 10.0
    nu0n = 0.25
    alp0n = 1.3e-5

    radios= 0.228542449538

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

    stress_initial_cond = [P0, tauxx, tauyy, tauzz, tauxy, tauxz, tauyz]

    U_initial_cond = [Ux, Uy, Uz]
    V_initial_cond = [Vx, Vy, Vz]

    T = 1.0
    # BOUNDARY CONDITIONS
    loadValue = 0.002
    loadType=[1, 0, 0, 0, 0, 0]

    dUxdx = loadValue * loadType[0]
    dUydy = loadValue * loadType[1]
    dUzdz = loadValue * loadType[2]

    dUxdy = loadValue * loadType[3]
    dUxdz = loadValue * loadType[4]
    dUydz = loadValue * loadType[5]

    strain_bound_cond=[dUxdx, dUydy, dUzdz, dUxdy, dUxdz, dUydz]

    #
    ob=eff_prop3D(size, grid_points, CFL, rho, E0p, nu0p, alp0p, E0n, nu0n, alp0n, radios,
            stress_initial_cond, U_initial_cond, V_initial_cond, T, strain_bound_cond)

    Keff=ob.find_Keff()
    #
    #T=1.0
    P0 = ob.K * ob.alp * 2 * ob.T
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

    stress_initial_cond = [P0, tauxx, tauyy, tauzz, tauxy, tauxz, tauyz]

    U_initial_cond = [Ux, Uy, Uz]
    V_initial_cond = [Vx, Vy, Vz]
    #
    ob.set_initial_cond( stress_initial_cond, U_initial_cond, V_initial_cond, T)
    #
    loadValue = 0.0
    loadType=[1, 0, 0, 0, 0, 0]

    dUxdx = loadValue * loadType[0]
    dUydy = loadValue * loadType[1]
    dUzdz = loadValue * loadType[2]

    dUxdy = loadValue * loadType[3]
    dUxdz = loadValue * loadType[4]
    dUydz = loadValue * loadType[5]

    strain_bound_cond=[dUxdx, dUydy, dUzdz, dUxdy, dUxdz, dUydz]
    #
    ob.set_boundary_cond(strain_bound_cond)

    al=ob.find_alpha(Keff)

if __name__ == "__main__":
    main()