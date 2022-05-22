## ############################################
## IMPORT MODULES AND FUNCTIONS
## ############################################
# import subprocess
# subprocess.check_call(["latex"])

import os
import sys
import glob
import argparse
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cmasher as cmr

os.environ["PATH"] += os.pathsep + '/usr/bin'

from tqdm.auto import tqdm
# from the_matplotlib_styler import *
# mpl.rcParams["text.usetex"] = True

## ############################################
## CONFIGURE WORKING ENVIRONMENT
## ############################################
os.system("clear") # clear terminal window
plt.ioff()
plt.switch_backend("agg") # use a non-interactive plotting backend

## ############################################
## GETTING USER INPUTS
## ############################################
class MyParser(argparse.ArgumentParser):
    def error(self, message):
        self.print_help()
        sys.exit(1)

## ############################################
## HELPFUL FUNCTIONS
## ############################################
def loopListWithUpdates(list_elems):
    lst_len = len(list_elems)
    return zip(
        list_elems,
        tqdm(
            range(lst_len),
            total = lst_len - 1
        )
    )

def checkFileEmpty(filepath):
    if os.stat(filepath).st_size == 0:
        return True
    else: return False

## ############################################
## PLOTTING FUNCTIONS
## ############################################
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

def plotData(fig, ax, ax_cbar, f_matrix, filepath_plot, iter_count, cbar_lims=[None, None], cbar_label=""):
    ## plot data
    im = ax.imshow(
        f_matrix,
        extent = [-1.0, 1.0, -1.0, 1.0],
        cmap   = cmr.ember,
        vmin   = cbar_lims[0],
        vmax   = cbar_lims[1]
    )
    ax.set_xticks([ ])
    ax.set_yticks([ ])
    ax.set_xticklabels([ ])
    ax.set_yticklabels([ ])
    # ## add labels
    # ax.set_xticks([-1.0, -0.5, 0.0, 0.5, 1])
    # ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1])
    # ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"], fontsize=14)
    # ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"], fontsize=14)
    ## create colormap
    cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
    ## add colorbar label
    cbar.set_label(cbar_label, size=18, color="black")
    ## save figure
    plt.savefig("{}/{}.png".format( filepath_plot, iter_count ))
    ## clear figure and axis
    fig.artists.clear()
    ax.clear()
    ax_cbar.clear()

def aniEvolution(filepath_plot, filepath_ani, sim_name):
    os.system(
        "ffmpeg -y -start_number 0 -i {}/%*.png".format(filepath_plot) +
        " -loglevel quiet" +
        " -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 {}/{}.mp4".format(
            filepath_ani,
            sim_name
        )
    )

## ############################################
## MAIN PROGRAM
## ############################################
def main():
    ## define input arguments
    parser = MyParser()
    ## ------------------- DEFINE REQUIRED ARGUMENTS
    args_req = parser.add_argument_group(description="Required processing arguments:")
    args_req.add_argument("-sim_name", type=str, required=True)
    args_req.add_argument("-var_name", type=str, required=True)
    ## ------------------- 
    ## ------------------- OPEN ARGUMENTS
    args = vars(parser.parse_args())
    ## ------------------- 
    ## ------------------- SAVE PARAMETERS
    sim_folder = args["sim_name"]
    var_name   = args["var_name"]
    print("\t> Simulation folder: {}".format( sim_folder ))
    print("\t> Plotting: {}".format( var_name ))
    ## ------------------- 
    ## ------------------- CALCULATE IMSHOW COLORBAR LIMITS
    fig, ax, ax_cbar = iniFigure()
    print("\t> Calculating cbar limits...")
    ## get list of file dumps
    list_data_dumps = glob.glob("{}/{}/*.txt".format(
        sim_folder,
        var_name
    ))
    ## initialise the colorbar limits
    cbar_min =  np.inf
    cbar_max = -np.inf
    for filepath_data_dump, _ in loopListWithUpdates(
        list_data_dumps[ len(list_data_dumps)//2: ]
        ):
        ## check the file is not empty
        if checkFileEmpty(filepath_data_dump): continue
        ## load data dump
        data_matrix = np.loadtxt(filepath_data_dump, delimiter=",")
        ## check if the cbar limits need to expand
        cbar_min = np.min([cbar_min, data_matrix.min()])
        cbar_max = np.max([cbar_max, data_matrix.max()])
    ## ------------------- 
    ## ------------------- PLOT SNAPSHOTS
    print("\t> Plotting snapshots...")
    for filepath_data_dump, _ in loopListWithUpdates(list_data_dumps):
        ## check the file is not empty
        if checkFileEmpty(filepath_data_dump): continue
        ## load data dump
        data_matrix = np.loadtxt(filepath_data_dump, delimiter=",")
        ## plot data dump
        plotData(
            fig, ax, ax_cbar,
            data_matrix,
            filepath_plot = "{}/{}".format( sim_folder, var_name ),
            iter_count = filepath_data_dump.split("_")[-1].split(".")[0],
            cbar_lims  = [cbar_min, cbar_max],
            cbar_label = var_name if not(var_name == "rho") else "density"
        )
    ## ------------------- 
    ## ------------------- ANIMATE SNAPSHOTS
    print("\t> Animating snapshots...")
    aniEvolution(
        filepath_plot = "{}/{}".format( sim_folder, var_name ),
        filepath_ani = sim_folder,
        sim_name = "{}_{}".format( sim_folder, var_name )
    )

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