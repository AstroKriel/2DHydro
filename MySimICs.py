import numpy as np
from MySimFuncs import *


def KelvinHelholtz(grid, gamma=5/3):
  ## initial band conifguration
  rho  =  1.0 + (np.abs(grid.coords_y - grid.height / 2) < grid.height / 4)
  velx = -0.5 + (np.abs(grid.coords_y - grid.height / 2) < grid.height / 4)
  ## alternate initial velocity perturbations along narrow Gaussian bands at the flow boundaries
  sigma = 0.03 # width of perturbation along y-direction
  vely  = 0.1 * np.sin(4 * np.pi * grid.coords_x) * (
    np.exp(-(grid.coords_y - grid.height/4)**2 / (2*sigma**2))
    + np.exp(-(grid.coords_y - 3*grid.height/4)**2 / (2*sigma**2))
  )
  ## uniform initial pressure
  press = 2.5 * np.ones(grid.coords_x.shape)
  ## convert primitive quantities to conserved
  mass, momx, momy, energy = prim2cons(rho, velx, vely, press, gamma, (grid.dx * grid.dy))
  return mass, momx, momy, energy


def Swirl(grid, gamma=5/3):
  ## get simulation grid information
  dx = grid.dx
  Y = grid.coords_y
  X = grid.coords_x
  box_size = grid.height
  N = grid.num_cells_x
  ## checkerboard patter
  rho = 1 + (
    2 + (np.abs(Y-box_size/2) < 0.2) + (np.abs(X-box_size/2) < 0.2)
  ) % 2
  vx = -10 * (
    (np.abs(Y-box_size/2) < 0.2) + (np.abs(X-box_size/2) > 0.2) - 1
  ) * (
    np.abs(Y-box_size/2) - box_size/2
  ) * (
    ((Y-box_size/2) > 0) - 0.5
  )
  vy = 10 * (
    (np.abs(X-box_size/2) < 0.2) + (np.abs(Y-box_size/2) > 0.2) - 1
  ) * (
    np.abs(X-box_size/2) - box_size/2
  ) * (
    ((X-box_size/2) > 0) - 0.5
  )
  ## add stochastic components to velocities
  vx += np.random.normal(loc=0, scale=0.5, size=(N,N))
  vy += np.random.normal(loc=0, scale=0.5, size=(N,N))
  press = np.random.uniform(1, 3, size=(N,N))
  ## convert primitive quantities to conserved
  m, px, py, E = prim2cons(rho, vx, vy, press, gamma, dx**2)
  return m, px, py, E


def Supernova(grid, gamma=5/3):
  ## get simulation grid information
  dx = grid.dx
  Y = grid.coords_y
  X = grid.coords_x
  box_size = grid.height
  N = grid.num_cells_x
  center_x = center_y = box_size/2
  R = np.sqrt((X-center_x)**2 + (Y-center_y)**2)
  rho = 2 * (0.4 * box_size) / np.sqrt(R) # circular config, densest at center
  vt0 = 3 # up to about 8 with slope limiting; up to about 4 without
  vr0 = 0.5
  v_tang_x = -vt0 * (Y - center_y) * R # CCW tangential motion
  v_tang_y =  vt0 * (X - center_x) * R
  v_rad_x  =  vr0 * (X - center_x) * (1 - 0.5 * box_size / R) # infalling velocity
  v_rad_y  =  vr0 * (Y - center_y) * (1 - 0.5 * box_size / R)
  ## add stochastic components to v
  vx = v_tang_x + v_rad_x + np.random.normal(loc=0, scale=0.5, size=(N,N))
  vy = v_tang_y + v_rad_y + np.random.normal(loc=0, scale=0.5, size=(N,N))
  press = 1 * rho # isothermal press~rho
  ## convert primitive quantities to conserved
  m, px, py, E = prim2cons(rho, vx, vy, press, gamma, dx**2)
  return m, px, py, E


dict_ICs = {
  "KH" : KelvinHelholtz,
  "SW" : Swirl,
  "SN" : Supernova
}

## END OF LIBRARY