import os
import numpy as np
import cmasher as cmr
import matplotlib.pyplot as plt


def aniEvolution(sim_folder):
  os.system(f"ffmpeg -y -start_number 0 -i {sim_folder}/%*.png -loglevel quiet -vb 40M -framerate 40 -vf scale=1440:-1 -vcodec mpeg4 {sim_folder}.mp4")


def iniFigure():
  ## initialise figure
  fig, ax = plt.subplots()
  ## get axis dimensions
  ax_info   = ax.get_position()
  ax_x1     = ax_info.x1
  ax_y0     = ax_info.y0
  ax_height = ax_info.height
  ## add colorbar axis
  ax_cbar = fig.add_axes([
    0.9 * ax_x1, # left edge
    ax_y0,        # bottom edge
    0.035,        # width
    ax_height,    # height
  ])
  return fig, ax, ax_cbar


def plotData(
    filepath_data,
    f_matrix,
    posx_matrix, posy_matrix, vecx_matrix, vecy_matrix,
    cbar_label = r"",
    text_label = None
  ):
  fig, ax, ax_cbar = iniFigure()
  extent_x = [ np.min(posx_matrix), np.max(posx_matrix) ]
  extent_y = [ np.min(posy_matrix), np.max(posy_matrix) ]
  im = ax.imshow(f_matrix, extent=extent_x+extent_y, cmap=cmr.ember)
  ax.quiver(posx_matrix, posy_matrix, vecx_matrix, vecy_matrix, color="white", width=0.002)
  ax.set_xlim(extent_x) 
  ax.set_ylim(extent_y)
  ## add labels
  ax.text(0.05, 0.95, text_label, va="top", ha="left", transform=ax.transAxes, fontsize=16, color="white")
  ax.set_xticks(np.linspace( extent_x[0], extent_x[1], 5 ))
  ax.set_yticks(np.linspace( extent_y[0], extent_y[1], 5 ))
  ax.set_xticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"], fontsize=14)
  ax.set_yticklabels([r"$-L/2$", r"$-L/4$", r"$0$", r"$L/4$", r"$L/2$"], fontsize=14)
  ## create colormap
  cbar = fig.colorbar(im, cax=ax_cbar, orientation="vertical")
  cbar.set_label(cbar_label, size=18, color="black", rotation=90)
  ## save figure
  fig.savefig(f"{filepath_data}.png", dpi=200)
  plt.close(fig)


## END OF LIBRARY