import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as colors
import mpl_toolkits
import mpl_toolkits.mplot3d
import pycuda.autoinit
import pycuda.driver as drv
import math
import VIS_GPU_2D_SOURCE as gpuVis

gpuMagic = gpuVis.gpuSource
getPlotDetails = gpuMagic.get_function("getPlotDetailsPhaseAndDensity")
getPlotDetailsForMF= gpuMagic.get_function("getPlotDetailsForMF")
aveVorticity = gpuMagic.get_function("aveVorticity")
get_total_rho = gpuMagic.get_function("getTotalRho")
get_total_rho_comp = gpuMagic.get_function("getTotalRhoComp")
test = gpuMagic.get_function("test")

#Set a datatype to use for arrays
DTYPE = np.complex
vectorSize = 10
spinComps = 5

matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 15

#constantsvectorSize = 10


def find_nan_mf_levels(quantum_field):
    non_empty_mfs = []
    QField = quantum_field.real
    for mf in xrange(5):
        if (np.any(QField[:, :, :, 2*mf] == float('nan'))): #or np.all(quantum_field[:, :, :, 2*mf + 1]) == 0.):
            non_empty_mfs.append(mf)
    print non_empty_mfs

def find_non_empty_mf_levels(quantum_field):
    non_empty_mfs = []
    QField = quantum_field.real
    for mf in xrange(spinComps):
        if not (np.all(QField[:, :, :, 2*mf] == 0.)): #or np.all(quantum_field[:, :, :, 2*mf + 1]) == 0.):
            non_empty_mfs.append(mf)
    return non_empty_mfs


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


blockX = 16
blockY = 16
blockZ = 1
gridX = 1
gridY = 1
xSize = 0
ySize = 0
zSize = 0
mfs = []
xAxis = None
yAxis = None
A = 0.
E = 0.
scaling = 0.
tau = 0.
rhoMin, rhoMax = 0, 0

def set_globals(file_name, QuantumState):
    global xSize, ySize, zSize, A, E, scaling, tau, blockX, blockY, blockZ, gridX, gridY, xAxis, yAxis, mfs
    mfs = find_non_empty_mf_levels(QuantumState)
    xSize = np.size(QuantumState,0)
    ySize = np.size(QuantumState,1)
    zSize = np.size(QuantumState,2)
    block_x = 16
    block_y = 16
    while xSize%block_x != 0:
      block_x /= 2
    while ySize%block_y != 0:
      block_y /= 2
    blockX = block_x
    blockY = block_y
    gridX = xSize/blockX
    gridY = ySize/blockY
    xAxis = np.arange(xSize)
    yAxis = np.arange(ySize)
    scaling = float(file_name.split("Scaling_")[1].split("_")[0])
    A = scaling/(xSize)
    tau = A*A

def putLabels(x_label, y_label, colorbar_label, ax):
  cbar = plt.colorbar(shrink = .9)
  cbar.set_label(colorbar_label, fontsize=14)
  cbar.ax.tick_params(labelsize=10)
  ax.set_xlabel(x_label).set_fontsize(24)
  ax.set_ylabel(y_label).set_fontsize(24)
  plt.tick_params(axis="both",which="major",labelsize=16)
  plt.tick_params(axis="both",which="minor",labelsize=16)


def animate(i, fig, file_name):
    global rhoMin, rhoMax
    gs = gridspec.GridSpec(2, 2)
    print(file_name)
    frame_number = int(file_name.split("Frame_")[1].split(".")[0])
    QuantumState = np.load(file_name)
    oldQuantumState = None
    if i == 0:
        set_globals(file_name, QuantumState)
        oldQuantumState = QuantumState.copy()
    else:
        step_size = int(file_name.split("Step_")[1].split("/")[0])
        #f_num = frame_number - step_size
        f = file_name.split("Frame_")[0] + "Frame_" + str(frame_number).zfill(8) + ".npy"
        oldQuantumState = np.load(f)
    VField = np.zeros((xSize, ySize), dtype = DTYPE)
    VxField = np.zeros((xSize, ySize), dtype = DTYPE)
    VyField = np.zeros((xSize, ySize), dtype = DTYPE)
    PhaseField = np.zeros((xSize, ySize), dtype = np.float64)
    VFieldAverage = np.zeros((xSize, ySize), dtype = DTYPE)
    RhoField = np.zeros((xSize, ySize), dtype = DTYPE)
    Lattice = np.zeros(4, dtype = np.int_)
    Lattice[0],  Lattice[1], Lattice[2], Lattice[3]= xSize, ySize, zSize, int(mfs[0])
    gpuQField = drv.to_device(QuantumState)
    gpuPhaseField = drv.to_device(PhaseField)
    gpuRhoField = drv.to_device(RhoField)
    gpuLattice = drv.to_device(Lattice)
    getPlotDetailsForMF(gpuQField, gpuRhoField, gpuPhaseField,gpuLattice, 
                block=(blockX,blockY,blockZ), grid=(gridX,gridY))
    RhoField = drv.from_device(gpuRhoField, RhoField.shape, DTYPE)
    PhaseField = drv.from_device(gpuPhaseField, PhaseField.shape, np.float64)
    if i == 0:
        rhoMin, rhoMax = np.amin(RhoField).real, np.amax(RhoField).real 
        print rhoMin, rhoMax
    Prob = np.sum(RhoField)
    #find_nan_mf_levels(QuantumState)
    time = frame_number
    time_text = plt.suptitle(r'$\tau = $' + str(time)
                       + "              " + '$P = $' + str(Prob.real) 
                       + "              ", fontsize=14,horizontalalignment='center',verticalalignment='top')
    # Density
    ax = fig.add_subplot(gs[0, 0])
    plt.imshow((RhoField.real), extent=(np.amin(xAxis), np.amax(xAxis), np.amin(yAxis), np.amax(yAxis)), origin = 'lower',
        cmap=colorMapRho, norm=colors.SymLogNorm(linthresh=linThresh*rhoMax,linscale=linScale,vmin=0.,vmax=rhoMax))
    putLabels('', r'$y\ \ (\ell)$', r'$\rho \ \ (\frac{1}{\ell^2})$', ax)
    ax = plt.gca()
    ax.set_aspect('equal')

    #Phase
    ax = fig.add_subplot(gs[0, 1])
    plt.imshow((PhaseField.real), extent=(np.amin(xAxis), np.amax(xAxis), np.amin(yAxis), np.amax(yAxis)), origin = 'lower',
        cmap=colorMapPhase, norm=colors.Normalize(vmin=0.,vmax=2.*np.pi))
    putLabels('', '', r'$\theta \ \ (Radians)$', ax)
    ax = plt.gca()
    ax.set_aspect('equal')
    #Trying to make a second plot
    gpuLattice.free()
    Lattice[0],  Lattice[1], Lattice[2], Lattice[3]= xSize, ySize, zSize, int(mfs[1])
    gpuLattice = drv.to_device(Lattice)
    getPlotDetailsForMF(gpuQField, gpuRhoField, gpuPhaseField, gpuLattice, 
                block=(blockX,blockY,blockZ), grid=(gridX,gridY))
    RhoField2 = drv.from_device(gpuRhoField, RhoField.shape, DTYPE)
    PhaseField2 = drv.from_device(gpuPhaseField, PhaseField.shape, np.float64)
    
    ax = fig.add_subplot(gs[1, 0])
    plt.imshow((RhoField2.real), extent=(np.amin(xAxis), np.amax(xAxis), np.amin(yAxis), np.amax(yAxis)), origin = 'lower',
        cmap=colorMapRho, norm=colors.SymLogNorm(linthresh=linThresh*rhoMax,linscale=linScale,vmin=0.,vmax=rhoMax))
    putLabels(r'$x\ \ (\ell)$', r'$y\ \ (\ell)$', r'$\rho \ \ (\frac{1}{\ell^2})$', ax)
    ax = plt.gca()
    ax.set_aspect('equal')

    #Phase
    ax = fig.add_subplot(gs[1, 1])
    plt.imshow((PhaseField2.real), extent=(np.amin(xAxis), np.amax(xAxis), np.amin(yAxis), np.amax(yAxis)), origin = 'lower',
        cmap=colorMapPhase, norm=colors.Normalize(vmin=0.,vmax=2.*np.pi))
    putLabels(r'$x\ \ (\ell)$', '', r'$\theta \ \ (Radians)$', ax)
    ax = plt.gca()
    ax.set_aspect('equal')

    # Screen density


    fig.tight_layout(pad=0.4, w_pad=5.0, h_pad=1.0, rect = [.05, .05, .95, .95])

    #Free GPU memory
    gpuQField.free()
    gpuPhaseField.free()
    gpuRhoField.free()
    gpuLattice.free()
    return fig


