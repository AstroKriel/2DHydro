#!/usr/bin/env python3

import os, time
from MyHelperFuncs import *
from MySimICs import dict_ICs
from MySimLoop import sim
from MySimFuncs import FiniteVolumeGrid
from MyPlotFuncs import aniEvolution

# clear terminal window
os.system("clear")


## ############################################
## MAIN PROGRAM LOOP
## ############################################
def main():
  ## read in parameters
  args = readParams("params.txt")
  sim_name = args["sim_name"].upper()
  ## check that some simulation output is specified
  if not(args["save_plots"]) and not(args["save_data"]):
    raise Exception("ERROR: No simulation output was specified in the 'params.txt' file.")

  ## generate grids
  grid = FiniteVolumeGrid(args)

  ## initialise state quantities (in conservative form)
  print("Initialising simulation...")
  try: ICs_cons = dict_ICs[sim_name](grid, args["gamma"])
  except: print(f"Initial conditions '{sim_name}' not implimented.")
  print(" ")

  # ## the following achieves the same thing as above, but requires Python 3.10+
  # ## add this import statement above: "from SimICs import *"
  # match sim_name:
  #   case "KH" | "kh":
  #     ICs_cons = KelvinHelholtz(grid, args["gamma"])
  #   case "SW" | "sw":
  #     ICs_cons = Swirl(grid, args["gamma"])
  #   case "SN" | "sn":
  #     ICs_cons = Supernova(grid, args["gamma"])
  #   case _:
  #     raise Exception(f"Initial conditions '{sim_name}' not implimented.")
  # print(" ")

  ## create output folders
  print("Creating simulation output folders...")
  createFolder(sim_name)
  ## create folders where data will be stored
  if args["save_data"]:
    createFolder(f"{sim_name}/rho")
    createFolder(f"{sim_name}/velx")
    createFolder(f"{sim_name}/vely")
    createFolder(f"{sim_name}/press")
    print(" ")

  ## run simulation
  print("Running simulation...")
  start_time = time.time()
  sim(ICs_cons, grid, args)
  run_time = time.time() - start_time
  print(f"Finished in {run_time:.2f} seconds.")
  print(" ")

  ## animate simulation snapshots
  if args["save_plots"]:
    print("Animating snapshots...")
    aniEvolution(sim_name)
    print(f"Snapshots saved: ./{sim_name}/*.png")
    print(f"Animation saved: ./{sim_name}.mp4")


## ############################################
## RUN PROGRAM
## ############################################
if __name__ == "__main__":
  main()


## END OF PROGRAM