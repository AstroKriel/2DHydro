## ############################################
## IMPORT MODULES AND FUNCTIONS
## ############################################
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import cmasher as cmr

from the_matplotlib_styler import *

## ############################################
## CONFIGURE WORKING ENVIRONMENT
## ############################################
# ## for debugging: handle warnings as errors, so they can be caught
# import warnings
# warnings.filterwarnings("error")

os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend

## ############################################
## HELPFUL FUNCTIONS: DEBUGGING
## ############################################
def checkMatrix(f_matrix):
    print(f_matrix.min(), f_matrix.max())

def saveMatrix(f_matrix, filepath_data):
    np.savetxt("{}.txt".format(filepath_data), f_matrix, fmt="%2.9f", delimiter=",")

## ############################################
## HELPFUL FUNCTIONS: GETTING USER INPUTS
## ############################################
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        sys.exit(1)

def str2bool(v):
    if isinstance(v, bool): return v
    if v.lower() in ("yes", "true", "t", "y", "1"): return True
    elif v.lower() in ("no", "false", "f", "n", "0"): return False
    else: raise argparse.ArgumentTypeError("Boolean value expected.")

## ############################################
## HELPFUL FUNCTIONS
## ############################################
def createFolder(sim_folder):
    if not(os.path.exists(sim_folder)):
        os.makedirs(sim_folder)
        print("\t> SUCCESS: Sub-folder created: '{}'".format(sim_folder))
    else: print("\t> WARNING: Sub-folder already exists (sub-folder not created): '{}'".format(sim_folder))

def aniEvolution(sim_folder):
    print("\t> Animating snapshots...")
    os.system(
        "ffmpeg -y -start_number 0 -i {}/%*.png".format(sim_folder) +
        " -loglevel quiet" +
        " -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 {}.mp4".format(sim_folder)
    )

def iniFigure():
    ## initialise figure
    fig, ax = plt.subplots()
    ## get axis dimensions
    ax_info   = ax.get_position()
    ax_x1     = ax_info.x1
    ax_y0     = ax_info.y0
    ax_height = ax_info.height
    ## define + add colorbar axis
    ax_cbar = fig.add_axes([
        0.85 * ax_x1, # left edge
        ax_y0,        # bottom edge
        0.035,        # width
        ax_height,    # height
    ])
    return fig, ax, ax_cbar

def funcAddColorbar(fig, ax, im, title=r""):
    ## get axis dimensions
    ax_info   = ax.get_position()
    ax_x1     = ax_info.x1
    ax_y0     = ax_info.y0
    ax_height = ax_info.height
    ## define + add colorbar axis
    ax_cbar = fig.add_axes([
        ax_x1 + 0.01, # left edge
        ax_y0,        # bottom edge
        0.035,        # width
        ax_height,    # height
    ])
    ## create colormap
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
    ## add colorbar label
    cbar.set_label(title, size=18, color="black")

def plotData(fig, ax, ax_cbar, f_matrix, sim_name, iter_count, cbar_label=r""):
    ## plot data
    im = ax.imshow(f_matrix, extent=[-1.0, 1.0, -1.0, 1.0], cmap=cmr.ember)
    ## add labels
    ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1])
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1])
    ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"], fontsize=14)
    ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"], fontsize=14)
    ## create colormap
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
    ## add colorbar label
    cbar.set_label(cbar_label, size=18, color="black")
    ## save figure
    plt.savefig(sim_name + "/{}.png".format( str(iter_count).zfill(5) ))
    ## clear figure and axis
    fig.artists.clear()
    ax.clear()
    ax_cbar.clear()

## ############################################
## CLASSES: SET UP SIMULATION GRID
## ############################################
class FiniteVolumeGrid():
    def __init__(
            self,
            x_bounds = (0.0, 1.0),
            y_bounds = (0.0, 1.0),
            num_cells_x = 128,
            num_cells_y = 128
        ):
        ## domain range
        self.x_min, self.x_max = x_bounds
        self.y_min, self.y_max = y_bounds
        ## number of discrete cells
        self.num_cells_x = num_cells_x
        self.num_cells_y = num_cells_y
        ## width of the cell interval
        self.width  = self.x_max - self.x_min
        self.height = self.y_max - self.y_min
        self.dx = self.width  / self.num_cells_x
        self.dy = self.height / self.num_cells_y
        ## create coordinates
        self.coords_y, self.coords_x = np.meshgrid(
            self.x_min + self.dx * (np.arange(self.num_cells_x) + 0.5),
            self.y_min + self.dy * (np.arange(self.num_cells_y) + 0.5)
        )
        ## total domain area
        self.area = self.dx * self.dy

class IniSimSetup():
    def KelvinHelholtz(sim_grid, gamma=5/3):
        ## initial band conifguration
        rho  =  1.0 + (np.abs(sim_grid.coords_y - sim_grid.height / 2) < sim_grid.height / 4)
        velx = -0.5 + (np.abs(sim_grid.coords_y - sim_grid.height / 2) < sim_grid.height / 4)
        ## alternate initial velocity perturbations along narrow Gaussian bands at the flow boundaries
        sigma = 0.03 # width of perturbation along y-direction
        vely  = 0.1 * np.sin(4 * np.pi * sim_grid.coords_x) * (
            np.exp(-(sim_grid.coords_y - sim_grid.height/4)**2 / (2*sigma**2))
            + np.exp(-(sim_grid.coords_y - 3*sim_grid.height/4)**2 / (2*sigma**2))
        )
        ## uniform initial pressure
        press = 2.5 * np.ones(sim_grid.coords_x.shape)
        ## convert primitive quantities to conserved
        mass, momx, momy, energy = prim2cons(rho, velx, vely, press, gamma, (sim_grid.dx * sim_grid.dy))
        return mass, momx, momy, energy, gamma
    def Swirl(sim_grid, gamma=5/3):
        ## get simulation grid information
        dx = sim_grid.dx
        Y = sim_grid.coords_y
        X = sim_grid.coords_x
        box_size = sim_grid.height
        N = sim_grid.num_cells_x
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
        return m, px, py, E, gamma
    def Explosion(sim_grid, gamma=5/3):
        ## get simulation grid information
        dx = sim_grid.dx
        Y = sim_grid.coords_y
        X = sim_grid.coords_x
        box_size = sim_grid.height
        N = sim_grid.num_cells_x
        center_x = center_y = box_size/2
        R = np.sqrt((X-center_x)**2 + (Y-center_y)**2)
        rho = 2 * (0.4 * box_size) / np.sqrt(R) # circular config, densest at center
        vt0 = 3 # up to ~8 with slope limiting; up to ~4 without
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
        return m, px, py, E, gamma

## ############################################
## FUNCTIONS FOR MAIN SIMULATION LOOP
## ############################################
def prim2cons(rho, velx, vely, press, gamma, area):
    ## conversion of primitive variables to concervative variables
    mass   = rho  * area
    momx   = mass * velx
    momy   = mass * vely
    energy = area * (
        0.5 * rho * (velx * velx + vely * vely) + press / (gamma - 1)
    )
    return mass, momx, momy, energy

def cons2prim(mass, momx, momy, energy, gamma, area):
    ## conversion of concervative variables to primitive variables
    rho   = mass / area
    velx  = momx / mass
    vely  = momy / mass
    press = (gamma - 1) * (
        energy / area - 0.5 * rho * (velx * velx + vely * vely)
    )
    return rho, velx, vely, press

def calcFieldGradient(f_matrix, dx):
    ## calculate gradients with central difference approximation
    df_dx = (np.roll(f_matrix, -1, axis=0) - np.roll(f_matrix, 1, axis=0)) / (2 * dx)
    df_dy = (np.roll(f_matrix, -1, axis=1) - np.roll(f_matrix, 1, axis=1)) / (2 * dx)
    return df_dx, df_dy

def slope_limiter(f_matrix, df_dx, df_dy, dx):
    ## calculate the L/R or B/T field gradients
    df_dx_LB = ( f_matrix - np.roll(f_matrix,  1, axis=0)) / dx
    df_dy_LB = ( f_matrix - np.roll(f_matrix,  1, axis=1)) / dx
    df_dx_RT = (-f_matrix + np.roll(f_matrix, -1, axis=0)) / dx
    df_dy_RT = (-f_matrix + np.roll(f_matrix, -1, axis=1)) / dx
    ## rescale the gradient to L/R or B/T gradients if the latter is smaller with same sign
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
    energy_LB = 0.5 * rho_iLB * (veli_iLB * veli_iLB + velj_iLB * velj_iLB) + press_iLB / (gamma - 1)
    energy_RT = 0.5 * rho_iRT * (veli_iRT * veli_iRT + velj_iRT * velj_iRT) + press_iRT / (gamma - 1)
    ## compute cell averaged (in direction 'i') states (missing scaling factors and units of area / mass)
    rho_prime    = 0.5 * (rho_iLB + rho_iRT)                       # primitive
    momx_prime   = 0.5 * (rho_iLB * veli_iLB + rho_iRT * veli_iRT) # conservative
    momy_prime   = 0.5 * (rho_iLB * velj_iLB + rho_iRT * velj_iRT) # conservative
    energy_prime = 0.5 * (energy_LB + energy_RT)                   # conservative
    press_prime  = (gamma - 1) * (                                 # primitive
        energy_prime - 0.5 / rho_prime * (momx_prime * momx_prime + momy_prime * momy_prime)
    )
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

## ############################################
## FUNCTION: GET USER INPUT
## ############################################
def getInputArgs():
    ## define input arguments
    parser = MyParser()
    ## ------------------- DEFINE OPTIONAL ARGUMENTS
    args_opt = parser.add_argument_group(description="Optional processing arguments:") # optional argument group
    optional_bool_args = {"required":False, "type":str2bool, "nargs":"?", "const":True}
    args_opt.add_argument("-slope_limit", default=False, **optional_bool_args)
    args_opt.add_argument("-live_plot",   default=False, **optional_bool_args)
    args_opt.add_argument("-save_data",   default=True,  **optional_bool_args)
    args_opt.add_argument("-num_x",       type=int,   default=128, required=False)
    # args_opt.add_argument("-num_y",       type=int,   default=128, required=False)
    args_opt.add_argument("-snap_freq",   type=int,   default=25,  required=False)
    args_opt.add_argument("-t_stop",      type=float, default=5.0, required=False)
    args_opt.add_argument("-CFL",         type=float, default=0.4, required=False)
    ## ------------------- DEFINE REQUIRED ARGUMENTS
    args_req = parser.add_argument_group(description="Required processing arguments:")
    args_req.add_argument("-sim_name", type=str, required=True)
    ## ------------------- OPEN ARGUMENTS
    args = vars(parser.parse_args())
    ## ------------------- SAVE PARAMETERS
    output_args = {
        "bool_live_plot":args["live_plot"],
        "bool_save_data":args["save_data"],
        "iter_snapshot_freq":args["snap_freq"]
    }
    sim_args = {
        "bool_slope_limit":args["slope_limit"],
        "sim_name":args["sim_name"],
        "t_stop":args["t_stop"],
        "cfl":args["CFL"],
        "num_cells_x":args["num_x"],
        "num_cells_y":args["num_x"]
    }
    ## ------------------- PRINT PARAMETERS
    print("\t> Simulation setup:")
    print("\t\t> slope limitter:", args["slope_limit"])
    print("\t\t> grid resolution: ({}, {})".format( args["num_x"], args["num_x"] ))
    print("\t\t> CFL:", args["CFL"])
    print("\t\t> stop time:", args["t_stop"])
    print("\t\t> live plotting:", args["live_plot"])
    print("\t\t> saving data:", args["save_data"])
    print(" ")
    ## return arguments
    return output_args, sim_args

## ############################################
## MAIN PROGRAM
## ############################################
def main():
    ## ------------------- GET INPUT ARGUMENTS
    output_args, sim_args = getInputArgs()
    ## check that something will done with the simulation output
    if not(output_args["bool_live_plot"]) and not(output_args["bool_save_data"]):
        Exception("ERROR: No output specified for simulation.")
    ## create plotting / data subfolders
    createFolder(sim_args["sim_name"])
    ## create folders where data is to be stored
    if output_args["bool_save_data"]:
        createFolder(sim_args["sim_name"] + "/rho")
        createFolder(sim_args["sim_name"] + "/velx")
        createFolder(sim_args["sim_name"] + "/vely")
        createFolder(sim_args["sim_name"] + "/press")
        print(" ")
    ## create figure
    if output_args["bool_live_plot"]:
        fig, ax, ax_cbar = iniFigure()
    ## ------------------- 
    ## ------------------- GENERATE THE SIMULATION GRID
    print("\t> Generating simulation grid...")
    sim_grid = FiniteVolumeGrid(
        num_cells_x = sim_args["num_cells_x"],
        num_cells_y = sim_args["num_cells_y"]
    )
    ## ------------------- 
    ## ------------------- GENERATE INITIAL CONDITIONS
    print("\t> Generating initial conditions...")
    if sim_args["sim_name"] == "KH":
        mass, momx, momy, energy, gamma = IniSimSetup.KelvinHelholtz(sim_grid, gamma=5/3)
    elif sim_args["sim_name"] == "SW":
        mass, momx, momy, energy, gamma = IniSimSetup.Swirl(sim_grid, gamma=5/3)
    elif sim_args["sim_name"] == "SN":
        mass, momx, momy, energy, gamma = IniSimSetup.Explosion(sim_grid, gamma=5/3)
    else: Exception("Initial conditions '{}' not implimented.".format( sim_args["sim_name"] ))
    ## ------------------- 
    ## ------------------- RUN SIMULATION
    t_sim      = 0.0
    iter_count = 0
    print("\t> Running simulation...")
    while t_sim < sim_args["t_stop"]:
        ## convert conserved quantities to primitive
        rho, velx, vely, press = cons2prim(mass, momx, momy, energy, gamma, sim_grid.area)
        ## calculate the minimum timestep
        max_signal_speed = np.sqrt(gamma * press / rho) + np.sqrt(velx * velx + vely * vely)
        dt = sim_args["cfl"] * np.min(sim_grid.dx / max_signal_speed)
        ## calculate gradient
        drho_dx,   drho_dy   = calcFieldGradient(rho,   sim_grid.dx)
        dvelx_dx,  dvelx_dy  = calcFieldGradient(velx,  sim_grid.dx)
        dvely_dx,  dvely_dy  = calcFieldGradient(vely,  sim_grid.dx)
        dpress_dx, dpress_dy = calcFieldGradient(press, sim_grid.dx)
        ## apply slope limitting
        if sim_args["bool_slope_limit"]:
            drho_dx,   drho_dy   = slope_limiter(rho,   drho_dx,   drho_dy,   sim_grid.dx)
            dvelx_dx,  dvelx_dy  = slope_limiter(velx,  dvelx_dx,  dvelx_dy,  sim_grid.dx)
            dvely_dx,  dvely_dy  = slope_limiter(vely,  dvely_dx,  dvely_dy,  sim_grid.dx)
            dpress_dx, dpress_dy = slope_limiter(press, dpress_dx, dpress_dy, sim_grid.dx)
        ## extrapolate half a timestep (at the cell centers)
        rho_prime   = rho   - 0.5 * dt * ( velx * drho_dx   + vely * drho_dy   + rho * (dvelx_dx + dvely_dy) )
        velx_prime  = velx  - 0.5 * dt * ( velx * dvelx_dx  + vely * dvelx_dy  + dpress_dx / rho )
        vely_prime  = vely  - 0.5 * dt * ( velx * dvely_dx  + vely * dvely_dy  + dpress_dy / rho )
        press_prime = press - 0.5 * dt * ( velx * dpress_dx + vely * dpress_dy + gamma * press * (dvelx_dx + dvely_dy) )
        ## extropolate to the interfaces between cells (face centers)
        rho_xL,   rho_xR,   rho_yB,   rho_yT   = calcFieldAtInterfaces(rho_prime,   drho_dx,   drho_dy,   sim_grid.dx)
        velx_xL,  velx_xR,  velx_yB,  velx_yT  = calcFieldAtInterfaces(velx_prime,  dvelx_dx,  dvelx_dy,  sim_grid.dx)
        vely_xL,  vely_xR,  vely_yB,  vely_yT  = calcFieldAtInterfaces(vely_prime,  dvely_dx,  dvely_dy,  sim_grid.dx)
        press_xL, press_xR, press_yB, press_yT = calcFieldAtInterfaces(press_prime, dpress_dx, dpress_dy, sim_grid.dx)
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
        ## update the solution for the primitive variables
        mass   = updateCells(mass,   flux_mass_x,   flux_mass_y,   sim_grid.dx, dt)
        momx   = updateCells(momx,   flux_momx_x,   flux_momx_y,   sim_grid.dx, dt)
        momy   = updateCells(momy,   flux_momy_x,   flux_momy_y,   sim_grid.dx, dt)
        energy = updateCells(energy, flux_energy_x, flux_energy_y, sim_grid.dx, dt)
        ## update the time
        t_sim += dt
        ## plot live update
        if (iter_count % output_args["iter_snapshot_freq"]) == 0:
            if output_args["bool_save_data"] or output_args["bool_live_plot"]:
                print("\t\t> Saving snapshot at index = {0}, t = {1:.3}".format(
                    str(iter_count).zfill(5),
                    t_sim
                ))
            if output_args["bool_save_data"]:
                ## save solution history
                str_iter_count = str(iter_count).zfill(5)
                saveMatrix(rho.T,   filepath_data="{0}/rho/rho_{1}".format( sim_args["sim_name"], str_iter_count ))
                saveMatrix(velx.T,  filepath_data="{0}/velx/velx_{1}".format( sim_args["sim_name"], str_iter_count ))
                saveMatrix(vely.T,  filepath_data="{0}/vely/vely_{1}".format( sim_args["sim_name"], str_iter_count ))
                saveMatrix(press.T, filepath_data="{0}/press/press_{1}".format( sim_args["sim_name"], str_iter_count ))
            if output_args["bool_live_plot"]:
                plotData(fig, ax, ax_cbar, rho.T, sim_args["sim_name"], iter_count)
        ## update simulation loop counter
        iter_count += 1
    ## ------------------- 
    ## ------------------- ANIMATE SNAPSHOTS
    if output_args["bool_live_plot"]:
        ## close figure
        plt.close(fig)
        ## animate snapshots
        aniEvolution(sim_args["sim_name"])

## ############################################
## RUN PROGRAM
## ############################################
if __name__ == "__main__":
    print("Starting program")
    main()
    print("Finished program")
    sys.exit()

## ############################################
## END OF PROGRAM
## ############################################