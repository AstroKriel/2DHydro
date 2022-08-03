import numpy as np


class FiniteVolumeGrid():
  def __init__(self, args, bounds=(0.0, 1.0)):
    ## store grid information
    self.x_min, self.x_max  = bounds
    self.y_min, self.y_max  = bounds
    self.num_cells_x        = args["num_cells"]
    self.num_cells_y        = args["num_cells"]
    self.width              = self.x_max - self.x_min
    self.height             = self.y_max - self.y_min
    self.dx                 = self.width  / self.num_cells_x
    self.dy                 = self.height / self.num_cells_y
    self.area               = self.dx * self.dy
    self.coords_y, self.coords_x = np.meshgrid(
      self.x_min + self.dx * (np.arange(self.num_cells_x) + 0.5),
      self.y_min + self.dy * (np.arange(self.num_cells_y) + 0.5)
    )


def prim2cons(rho, velx, vely, press, gamma, area):
  ## conversion of primitive variables to concervative variables
  mass   = rho  * area
  momx   = mass * velx
  momy   = mass * vely
  energy = area * (
    0.5 * rho * (velx*velx + vely*vely) + press / (gamma - 1)
  )
  return mass, momx, momy, energy


def cons2prim(mass, momx, momy, energy, gamma, area):
  ## conversion of concervative variables to primitive variables
  rho   = mass / area
  velx  = momx / mass
  vely  = momy / mass
  press = (gamma - 1) * (
    energy / area - 0.5 * rho * (velx*velx + vely*vely)
  )
  return rho, velx, vely, press


def calcFieldGradient(f_matrix, dx):
  ## compute gradients with central difference approximation
  df_dx = (np.roll(f_matrix, -1, axis=0) - np.roll(f_matrix, 1, axis=0)) / (2 * dx)
  df_dy = (np.roll(f_matrix, -1, axis=1) - np.roll(f_matrix, 1, axis=1)) / (2 * dx)
  return df_dx, df_dy


def slopeLimiter(f_matrix, df_dx, df_dy, dx):
  ## calculate the L/R or B/T field gradients
  df_dx_LB = ( f_matrix - np.roll(f_matrix,  1, axis=0)) / dx
  df_dy_LB = ( f_matrix - np.roll(f_matrix,  1, axis=1)) / dx
  df_dx_RT = (-f_matrix + np.roll(f_matrix, -1, axis=0)) / dx
  df_dy_RT = (-f_matrix + np.roll(f_matrix, -1, axis=1)) / dx
  ## rescale the gradient to L/R or B/T gradients if the latter is smaller with the same sign
  ## if the signs of the gradient in x vs y direction differ (i.e. discontinuity), set the gradient to 0
  df_dx *= np.maximum(0, np.minimum( 1, df_dx_LB / (df_dx + 1e-8 * (df_dx==0)) ))
  df_dx *= np.maximum(0, np.minimum( 1, df_dx_RT / (df_dx + 1e-8 * (df_dx==0)) ))
  df_dy *= np.maximum(0, np.minimum( 1, df_dy_LB / (df_dy + 1e-8 * (df_dy==0)) ))
  df_dy *= np.maximum(0, np.minimum( 1, df_dy_RT / (df_dy + 1e-8 * (df_dy==0)) ))
  return df_dx, df_dy


def calcFieldAtInterfaces(f_matrix, df_dx, df_dy, dx):
  ## second order Taylor expansion for field quantities either side of the interfaces
  f_xL = np.roll(f_matrix - 0.5 * df_dx * dx, -1, axis=0)
  f_xR =         f_matrix + 0.5 * df_dx * dx
  f_yB = np.roll(f_matrix - 0.5 * df_dy * dx, -1, axis=1)
  f_yT =         f_matrix + 0.5 * df_dy * dx
  return f_xL, f_xR, f_yB, f_yT


def calcRusanovFlux(rho_iLB, rho_iRT, veli_iLB, veli_iRT, velj_iLB, velj_iRT, press_iLB, press_iRT, gamma):
  ## energies at the left/bottom and right/top interfaces (missing scaling factors and units of area)
  energy_LB = 0.5 * rho_iLB * (veli_iLB*veli_iLB + velj_iLB*velj_iLB) + press_iLB / (gamma - 1)
  energy_RT = 0.5 * rho_iRT * (veli_iRT*veli_iRT + velj_iRT*velj_iRT) + press_iRT / (gamma - 1)
  ## compute cell averaged (in direction 'i') states (missing scaling factors and units of area / mass)
  rho_prime    = 0.5 * (rho_iLB + rho_iRT)                       # primitive
  momx_prime   = 0.5 * (rho_iLB * veli_iLB + rho_iRT * veli_iRT) # conservative
  momy_prime   = 0.5 * (rho_iLB * velj_iLB + rho_iRT * velj_iRT) # conservative
  energy_prime = 0.5 * (energy_LB + energy_RT)                   # conservative
  press_prime  = (gamma - 1) * (                                 # primitive
    energy_prime - 0.5 / rho_prime * (momx_prime*momx_prime + momy_prime*momy_prime)
  )
  ## ?
  ## compute fluxes (local Lax-Friedrichs / Rusanov)
  flux_mass   = momx_prime
  flux_momx   = momx_prime * momx_prime / rho_prime + press_prime
  flux_momy   = momx_prime * momy_prime / rho_prime
  flux_energy = (energy_prime + press_prime) * momx_prime / rho_prime
  ## calculate maximum local fluid soundspeed
  c_LB  = np.sqrt(gamma * press_iLB / rho_iLB) + np.abs(veli_iLB)
  c_RT  = np.sqrt(gamma * press_iRT / rho_iRT) + np.abs(veli_iRT)
  c_max = np.maximum(c_LB, c_RT)
  ## add stabilizing diffusive term
  flux_mass   -= c_max * 0.5 * (rho_iLB - rho_iRT)
  flux_momx   -= c_max * 0.5 * (rho_iLB * veli_iLB - rho_iRT * veli_iRT)
  flux_momy   -= c_max * 0.5 * (rho_iLB * velj_iLB - rho_iRT * velj_iRT)
  flux_energy -= c_max * 0.5 * (energy_LB - energy_RT)
  return flux_mass, flux_momx, flux_momy, flux_energy


def updateCells(f_matrix, flux_f_x, flux_f_y, dx, dt):
  ## update the L side of the R face along x
  f_matrix += -dt * dx * flux_f_x
  ## update the R side of the R face along x
  f_matrix +=  dt * dx * np.roll(flux_f_x, 1, axis=0)
  ## update the B side of the T face along y
  f_matrix += -dt * dx * flux_f_y
  ## update the T side of the T face along y
  f_matrix +=  dt * dx * np.roll(flux_f_y, 1, axis=1)
  return f_matrix


## END OF LIBRARY