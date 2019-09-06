import os
os.environ['CUDA_DEVICE'] = str(0) #Set CUDA device, starting at 0
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
# from matplotlib.patches import Rectangle, Ellipse, Circle
# from mpl_toolkits.axes_grid1.anchored_artists import AnchoredDrawingArea
import matplotlib.patches as patches
import mpl_toolkits
import mpl_toolkits.mplot3d


#Set a datatype to use for arrays
DTYPE = np.complex
vectorSize = 2

matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 15

#constantsvectorSize = 10

def calculateRho(QuantumField):
    rho = np.zeros(xSize, dtype = DTYPE)
    for i in xrange(vectorSize):
        rho[:] = rho[:] + (QuantumField[:, 0, 0, i] * (QuantumField[:, 0, 0, i]).conjugate())
    return rho

def calculateRhoComp(QuantumField, comp):
        rho = ((QuantumField[:, 0, 0, comp] * (QuantumField[:, 0, 0, comp]).conjugate())
                 +(QuantumField[:, 0, 0, comp+1] * (QuantumField[:, 0, 0, comp+1]).conjugate()))
        return rho

def calculateProbability(QuantumField):
    prob = 0
    for x in xrange(xSize):
        prob = prob + calculateRho(QuantumField,x) * A
    return prob

'''
------------------------------------------
-------- PLOT AND ANIMATE SECTION --------
------------------------------------------
'''
linThresh = .1
linScale = 1
colorMapVort = cm.bwr
colorMapRho = cm.copper
colorMapE = cm.inferno
colorMapPhase = cm.gist_rainbow
colorMapGrey = cm.gist_gray
colorMapBlue = cm.Blues
colorMapGreen = cm.Greens
colorMapRed = cm.Reds

# Get the colormap colors
blues_alpha = colorMapBlue(np.arange(colorMapBlue.N))
reds_alpha = colorMapRed(np.arange(colorMapRed.N))
greens_alpha = colorMapGreen(np.arange(colorMapGreen.N))

# Set alpha
blues_alpha[:,-1] = np.linspace(0, 1, colorMapBlue.N)
reds_alpha[:,-1] = np.linspace(0, 1, colorMapRed.N)
greens_alpha[:,-1] = np.linspace(0, 1, colorMapGreen.N)

blues_alpha = ListedColormap(blues_alpha)
reds_alpha = ListedColormap(reds_alpha)
greens_alpha = ListedColormap(greens_alpha)

xSize = 0
ySize = 0
xAxis = None
yAxis = None
rhoMin = 0.
rhoMax = 0.
E = 0.


def set_globals(global_vars, QuantumState):
    global xSize, ySize, zSize, xAxis, yAxis, vectorSize
    xSize = global_vars["xSize"]
    ySize = global_vars["ySize"]
    xAxis = np.arange(xSize)
    yAxis = np.arange(ySize)
    vectorSize = global_vars["vectorSize"]

def putLabels(fig, ax, im, x_label, y_label, colorbar_label):
    cbar = plt.colorbar(im, ax=ax) #fig.colorbar(im, shrink = .9)
    cbar.set_label(colorbar_label, fontsize=18)
    cbar.ax.tick_params(labelsize=10)
    ax.set_xlabel(x_label).set_fontsize(24)
    ax.set_ylabel(y_label).set_fontsize(24)
    ax.tick_params(axis="both",which="major",labelsize=16)
    ax.tick_params(axis="both",which="minor",labelsize=16)
    ax.set_aspect('auto')


def make_frame(frame_dir, frame, image_dir, frames, global_vars, add_ds_patches = False, ds_wall_params=[60,60,360], fig_size=(16,12), **kwargs):
    # print i
    global rhoMin, rhoMax
    frame_number = (frame.split("_")[1]).split(".")[0]
    QuantumState = np.load(frame_dir)
    if int(frame_number) == 0:
        set_globals(global_vars, QuantumState)
    RhoFields = []
    PhaseFields = []
    num_comps = int(len(QuantumState[0,0,0,:])/2)
    for comp in xrange(num_comps):
        RhoFields.append((QuantumState[:,:,0,2*comp]+QuantumState[:,:,0,2*comp+1])*(QuantumState[:,:,0,2*comp]+QuantumState[:,:,0,2*comp+1]).conjugate())
        PhaseFields.append(np.arctan2((QuantumState[:,:,0,2*comp]+QuantumState[:,:,0,2*comp+1]).imag,(QuantumState[:,:,0,2*comp]+QuantumState[:,:,0,2*comp+1]).real))
    if int(frame_number) == 0:
        for comp in xrange(num_comps):
            if np.amax(RhoFields[comp].real)>rhoMax:
                rhoMax = np.amax(RhoFields[comp].real)
            if np.amin(RhoFields[comp].real)<rhoMin:
                rhoMin = np.amin(RhoFields[comp].real)

    fig, axs = plt.subplots(num_comps, 2, figsize=fig_size, constrained_layout=True)
    plt.rc('text', usetex=True)
    #Ref below    
    time = int(frame_number)
    time_text = plt.suptitle(r'$\tau = $' + str(time), fontsize=14, horizontalalignment='center',verticalalignment='top')


    cases = np.arange(2*num_comps)
    for ax, case in zip(axs, cases):
        if case%2==0:
            im = ax.imshow(RhoFields[int(case/2)].real, extent=(np.amin(yAxis), np.amax(yAxis), np.amin(xAxis), np.amax(xAxis)), origin = 'lower',
                alpha = 1.0, cmap=blues_alpha, norm=colors.SymLogNorm(linthresh=linThresh*rhoMax,linscale=linScale,vmin=0.,vmax=rhoMax))
            putLabels(fig,ax,im,r'$y\ \ (\ell)$', r'$x\ \ (\ell)$', r'$\rho \ \ (\frac{1}{\ell^2})$')
            ax.set_aspect('auto')
        else:
            # # Phase
            im = ax.imshow((PhaseFields[int(case/2)].real), extent=(np.amin(yAxis), np.amax(yAxis), np.amin(xAxis), np.amax(xAxis)), origin = 'lower',
                cmap=colorMapPhase, norm=colors.Normalize(vmin=0.,vmax=2.*np.pi))
            # Any value whose absolute value is > .0001 will have zero transparency
            rhoMaxCase = np.amax(RhoFields[int(case/2)].real)
            alphas = Normalize(0, rhoMaxCase, clip=True)((rhoMaxCase-RhoFields[int(case/2)].real))
            # # alphas = colors.SymLogNorm(linthresh=linThresh*rhoMax,linscale=linScale,vmin=0.,vmax=10., clip=True)((rhoMax-RhoField.real).T)
            alphas = np.clip(alphas**4, 0.0, 1)  # alpha value clipped at the bottom at .4
            # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
            cmap = plt.cm.gist_gray.reversed()
            colorsMap = colors.SymLogNorm(linthresh=linThresh*rhoMaxCase,linscale=linScale,vmin=0.,vmax=rhoMaxCase)((0.*RhoFields[int(case/2)].real))
            colorsMap = cmap(colorsMap)

            # Now set the alpha channel to the one we created above
            colorsMap[..., -1] = alphas
            ax.imshow(colorsMap, origin = 'lower')
            putLabels(fig,ax,im,r'$y\ \ (\ell)$', r'$x\ \ (\ell)$', r'$\theta \ \ (Radians)$')

        if add_ds_patches:
            #Add walls and screen
            slit_width = ds_wall_params[0]
            wall_width = ds_wall_params[1]
            spacing = ds_wall_params[2]
            #Double slit
            ax.add_patch(patches.Rectangle(
                (0.,2.*(xSize-1)/5.-wall_width/2.), #xLoc,yLoc
                ySize/2. - spacing/2. - slit_width/2.,  #Width
                wall_width,  #Height
                color = 'black'
                )
              )
            ax.add_patch(patches.Rectangle(
                ((ySize-1.)/2. - spacing/2. + slit_width/2.,2.*(xSize-1)/5.-wall_width/2.), #xLoc,yLoc
                spacing - slit_width,  #Width
                wall_width,  #Height
                color = 'black'
                )
              )
            ax.add_patch(patches.Rectangle(
                ((ySize-1.)/2. + spacing/2. + slit_width/2.,2.*(xSize-1)/5.-wall_width/2.), #xLoc,yLoc
                ySize/2. - spacing/2. - slit_width/2.,  #Width
                wall_width,  #Height
                color = 'black'
                )
              )

            #Screen
            ax.add_patch(patches.Rectangle(
                (0.,2.*(xSize)/3.), #xLoc,yLoc
                ySize,  #Width
                wall_width/8.,  #Height
                color = 'red'
                )
              )
            #Boundary walls
            ax.add_patch(patches.Rectangle(
                (0.,0.), #xLoc,yLoc
                ySize,  #Width
                wall_width,  #Height
                color = 'black'
                )
              )
            ax.add_patch(patches.Rectangle(
                (0.,xSize-wall_width), #xLoc,yLoc
                ySize,  #Width
                wall_width,  #Height
                color = 'black'
                )
              )


    # fig.tight_layout(rect=[0, 0.03, 1.2, 1.15])
    if not os.path.exists(image_dir + '/'):
        os.makedirs(image_dir + '/')
    fig.savefig(image_dir + '/Frame_' + frame_number +".png")
    print "Saved " + image_dir + '/Frame_' + frame_number +".png"
    plt.close(fig)


