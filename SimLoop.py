import numpy as np
from SimFuncs import *
from PlotFuncs import *
from HelperFuncs import *


def sim(ICs_cons, grid, args):
  ## initialise variables
  t_sim      = 0.0
  iter_count = 0
  ## extract initial conditions (in conservative form)
  mass, momx, momy, energy = ICs_cons
  ## extract simulation parameters
  folder_name = args["sim_name"]
  num_cells   = args["num_cells"]
  gamma       = args["gamma"]
  cfl         = args["cfl"]
  t_stop      = args["t_stop"]
  slope_limit = args["slope_limit"]
  save_plots  = args["save_plots"]
  save_data   = args["save_data"]
  output_freq = args["output_freq"]
  ## initialiase variables for quiver plot
  domain = np.linspace(0.0, 1.0, num_cells)
  posx, posy = np.meshgrid(domain, domain)
  sr = max(1, num_cells // 25)
  ## enter simulation loop
  while t_sim < t_stop:
    ## convert conserved quantities to primitive
    rho, velx, vely, press = cons2prim(mass, momx, momy, energy, gamma, grid.area)
    ## calculate the minimum timestep
    max_signal_speed = np.sqrt(gamma * press / rho) + np.sqrt(velx*velx + vely*vely)
    dt = cfl * np.min(grid.dx / max_signal_speed)
    ## calculate gradient
    drho_dx,   drho_dy   = calcFieldGradient(rho,   grid.dx)
    dvelx_dx,  dvelx_dy  = calcFieldGradient(velx,  grid.dx)
    dvely_dx,  dvely_dy  = calcFieldGradient(vely,  grid.dx)
    dpress_dx, dpress_dy = calcFieldGradient(press, grid.dx)
    ## apply slope limitting
    if slope_limit:
      drho_dx,   drho_dy   = slopeLimiter(rho,   drho_dx,   drho_dy,   grid.dx)
      dvelx_dx,  dvelx_dy  = slopeLimiter(velx,  dvelx_dx,  dvelx_dy,  grid.dx)
      dvely_dx,  dvely_dy  = slopeLimiter(vely,  dvely_dx,  dvely_dy,  grid.dx)
      dpress_dx, dpress_dy = slopeLimiter(press, dpress_dx, dpress_dy, grid.dx)
    ## extrapolate half a timestep (at the cell centers)
    rho_prime   = rho   - 0.5 * dt * ( velx * drho_dx   + vely * drho_dy   + rho * (dvelx_dx + dvely_dy) )
    velx_prime  = velx  - 0.5 * dt * ( velx * dvelx_dx  + vely * dvelx_dy  + dpress_dx / rho )
    vely_prime  = vely  - 0.5 * dt * ( velx * dvely_dx  + vely * dvely_dy  + dpress_dy / rho )
    press_prime = press - 0.5 * dt * ( velx * dpress_dx + vely * dpress_dy + gamma * press * (dvelx_dx + dvely_dy) )
    ## extropolate to the interfaces (face centers) between cells
    rho_xL,   rho_xR,   rho_yB,   rho_yT   = calcFieldAtInterfaces(rho_prime,   drho_dx,   drho_dy,   grid.dx)
    velx_xL,  velx_xR,  velx_yB,  velx_yT  = calcFieldAtInterfaces(velx_prime,  dvelx_dx,  dvelx_dy,  grid.dx)
    vely_xL,  vely_xR,  vely_yB,  vely_yT  = calcFieldAtInterfaces(vely_prime,  dvely_dx,  dvely_dy,  grid.dx)
    press_xL, press_xR, press_yB, press_yT = calcFieldAtInterfaces(press_prime, dpress_dx, dpress_dy, grid.dx)
    ## calculate the fluxes across the vertical interfaces (x-faces) using Rusanov method
    flux_mass_x, flux_momx_x, flux_momy_x, flux_energy_x = calcRusanovFlux(
      rho_iLB   = rho_xL,
      rho_iRT   = rho_xR,
      veli_iLB  = velx_xL,
      veli_iRT  = velx_xR,
      velj_iLB  = vely_xL,
      velj_iRT  = vely_xR,
      press_iLB = press_xL,
      press_iRT = press_xR,
      gamma     = gamma
    )
    ## calculate the fluxes across the horizontal interfaces (y-faces) using Rusanov method
    flux_mass_y, flux_momy_y, flux_momx_y, flux_energy_y = calcRusanovFlux(
      rho_iLB   = rho_yB,
      rho_iRT   = rho_yT,
      veli_iLB  = vely_yB,
      veli_iRT  = vely_yT,
      velj_iLB  = velx_yB,
      velj_iRT  = velx_yT,
      press_iLB = press_yB,
      press_iRT = press_yT,
      gamma     = gamma
    )
    ## update primitive variables
    mass   = updateCells(mass,   flux_mass_x,   flux_mass_y,   grid.dx, dt)
    momx   = updateCells(momx,   flux_momx_x,   flux_momx_y,   grid.dx, dt)
    momy   = updateCells(momy,   flux_momy_x,   flux_momy_y,   grid.dx, dt)
    energy = updateCells(energy, flux_energy_x, flux_energy_y, grid.dx, dt)
    ## update simulation time
    t_sim += dt
    ## plot live update
    if (iter_count % output_freq) == 0:
      str_iter_count = str(iter_count).zfill(5)
      if save_data or save_plots:
        print(f"\t> Saving snapshot at index = {str_iter_count}, t = {t_sim:.3f}, dt = {dt:.3e}")
      if save_data:
        ## save solution history
        saveMatrix(rho.T,   f"{folder_name}/rho/{str_iter_count}")
        saveMatrix(velx.T,  f"{folder_name}/velx/{str_iter_count}")
        saveMatrix(vely.T,  f"{folder_name}/vely/{str_iter_count}")
        saveMatrix(press.T, f"{folder_name}/press/{str_iter_count}")
      if save_plots:
        plotData(
          filepath_data = f"{folder_name}/{str_iter_count}",
          f_matrix      = rho.T,
          posx_matrix   = posx[::sr, ::sr],
          posy_matrix   = posy[::sr, ::sr],
          vecx_matrix   = velx.T[::sr, ::sr],
          vecy_matrix   = vely.T[::sr, ::sr],
          cbar_label    = r"density $(\rho$)",
          text_label    = rf"$t = {t_sim:.2f}$"
        )
    ## update simulation loop counter
    iter_count += 1


## END OF LIBRARY