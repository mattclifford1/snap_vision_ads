# https://stackoverflow.com/questions/46615554/how-to-display-multiple-images-in-one-figure-correctly
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import pathlib 
# pathlib.Path().resolve()
import glob
from PIL import Image

#mpl.rcParams[]
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.bottom'] = False
plt.rcParams['axes.spines.left'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['xtick.minor.visible'] = False
plt.rcParams['xtick.bottom'] = False
plt.rcParams['ytick.left'] = False
plt.rcParams['ytick.minor.visible'] = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['xtick.labelbottom'] = False

plt.rcParams["figure.figsize"] = (10,5)

FOLDERS = ["15777458", "11373824"]
#print(plt.rcParams.keys())
plt.close()
def get_figure(folder, colour_bar = False):
    images = []
    for f in glob.iglob(f"data/uob_image_set/{folder}/*"):
        print(f)
        images.append(np.asarray(Image.open(f)))   

    images = np.array(images)
    fig = plt.figure()
    
    for i in range(len(images)):
        ax = fig.add_subplot(1, len(images), i+1)
        plt.imshow(images[i])
        #ax.set_title("tings")
        #plt.colorbar()
    fig.suptitle(folder, fontsize='x-large', verticalalignment="top", y=0.85) #, y=0.8)
    #plt.tight_layout()
    plt.savefig(f'exploration/plots/{folder}.png')
    plt.show()
    

def get_figures(folders, colour_bar = False):
    for folder in folders:
        get_figure(folder)

get_figures(folders=FOLDERS)