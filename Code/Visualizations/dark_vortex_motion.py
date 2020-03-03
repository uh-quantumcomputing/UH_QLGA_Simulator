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
getPlotDetailsVorticity = gpuMagic.get_function("getPlotDetailsVorticity")
getPlotDetailsForComponent = gpuMagic.get_function("getPlotDetailsForComponent")
aveVorticity = gpuMagic.get_function("aveVorticity")
get_total_rho = gpuMagic.get_function("getTotalRho")
get_total_rho_comp = gpuMagic.get_function("getTotalRhoComp")
test = gpuMagic.get_function("test")

#Set a datatype to use for arrays
DTYPE = np.complex
vectorSize = 10
spinComps = 5


color_code = {0 : 'b', 1: 'r', 2 : 'c', 3 : 'y', 4: 'g'}

matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 15

#constantsvectorSize = 10


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
xAxis = None
yAxis = None
A = 0.
E = 0.
scaling = 0.
tau = 0.
rhoMin, rhoMax = 0, 0

def set_globals(file_name, QuantumState):
    global xSize, ySize, zSize, A, E, scaling, tau, blockX, blockY, blockZ, gridX, gridY, xAxis, yAxis
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


# A = np.ones((9, 9, 1, 10), dtype = DTYPE)
# A[3, 4, 0, 2] = 0.
# A[3, 4, 0, 3] = 0.
# A[6, 6, 0, 2] = 0.
# A[6, 6, 0, 3] = 0.
# A[3, 4, 0, 4] = 0.
# A[3, 4, 0, 5] = 0.
# A[6, 6, 0, 0] = 0.
# A[6, 6, 0, 1] = 0.
# A[3, 4, 0, 6] = 0.
# A[3, 4, 0, 7] = 0.
# A[6, 6, 0, 0] = 0.
# A[6, 6, 0, 1] = 0.



def make_boson_state(quantum_state):
    shape = quantum_state.shape
    boson_state = np.zeros((shape[0], shape[1], shape[2], spinComps), dtype = DTYPE)
    for i in xrange(spinComps):
        boson_state[:, :, :, i] = quantum_state[:, :, :, 2*i]*(quantum_state[:, :, :, 2*i + 1].conjugate())
    return boson_state

def find_dark_vortex(quantum_state):
    boson_state = make_boson_state(quantum_state)
    vortex_centers = []
    for i in xrange(spinComps):
        x, y = np.where((boson_state[:, :, 0, i] < .005*np.amax(boson_state[:, :, 0, i])) & (boson_state[:, :, 0, i] < .0001) & (np.amax(boson_state[:, :, 0, i]) > .1))
        if(x.shape[0] > 0):           
            centers = zip(x, y)
            for center in centers:
                vortex_centers.append([center[0], center[1], i])
    return np.asarray(vortex_centers)

def find_dark_vortex_from_boson(boson_state):
    vortex_centers = []
    for i in xrange(spinComps):
        x, y = np.where((boson_state[:, :, i] < .005*np.amax(boson_state[:, :, i])) & (boson_state[:, :, i] < .0001) & (np.amax(boson_state[:, :, i]) > .1))
        if(x.shape[0] > 0):           
            centers = zip(x, y)
            for center in centers:
                vortex_centers.append([center[0], center[1], i])
    return np.asarray(vortex_centers)

# def find_dark_vortex_from_boson(boson_state):
#     vortex_centers = []
#     for i in xrange(spinComps):
#         x, y = np.where((boson_state[:, :, i] < .0000025) & (np.amax(boson_state[:, :, i]) > .1))
#         if(x.shape[0] > 0):           
#             centers = zip(x, y)
#             for center in centers:
#                 vortex_centers.append([center[0], center[1], i])
#     return np.asarray(vortex_centers)


def find_dark_vortex_from_vorticity(vorticity_field):
    vortex_centers = []
    for i in xrange(spinComps):
        x, y = np.where((np.abs(vorticity_field[:, :, i]) > .2))
        if(x.shape[0] > 0):           
            centers = zip(x, y)
            for center in centers:
                    vortex_centers.append([center[0], center[1], i])
    return np.asarray(vortex_centers)




def get_color_array(vortex_centers):
    color_array = []
    for center in vortex_centers:
        color_array.append(color_code[center[2]])
    return np.asarray(color_array)



# vortex_centers = find_dark_vortex(A)
# print (vortex_centers)
# #colors = color_code[vortex_centers[:, 2]]

# #colors = get_color_array(vortex_centers)
# print colors
# quit()

def putLabels(x_label, y_label, colorbar_label):
  cbar = plt.colorbar(shrink = .9)
  cbar.set_label(colorbar_label, fontsize=14)
  cbar.ax.tick_params(labelsize=10)
  plt.xlabel(x_label).set_fontsize(24)
  plt.ylabel(y_label).set_fontsize(24)
  plt.tick_params(axis="both",which="major",labelsize=16)
  plt.tick_params(axis="both",which="minor",labelsize=16)


def animate(i, fig, file_name):
    global rhoMin, rhoMax
    frame_number = int(file_name.split("Frame_")[1].split(".")[0])
    print("loading file " + file_name)
    quantum_state = np.load(file_name)
    if i == 0:
        set_globals(file_name, quantum_state) 
    VortField = np.zeros((xSize, ySize, spinComps), dtype = DTYPE)
    VxField = np.zeros((xSize, ySize), dtype = DTYPE)
    VyField = np.zeros((xSize, ySize), dtype = DTYPE)
    PhaseField = np.zeros((xSize, ySize), dtype = np.float64)
    VFieldAverage = np.zeros((xSize, ySize), dtype = DTYPE)
    RhoField = np.zeros((xSize, ySize), dtype = DTYPE)
    boson_field = np.zeros((xSize, ySize, spinComps), dtype = DTYPE)
    Lattice = np.zeros(4, dtype = np.int_)
    Lattice[0],  Lattice[1], Lattice[2]= xSize, ySize, zSize
    gpuQField = drv.to_device(quantum_state)
    quantum_state = drv.from_device(gpuQField,quantum_state.shape,DTYPE)
    gpuVField = drv.to_device(VortField)
    gpuVxField = drv.to_device(VxField)
    gpuVyField = drv.to_device(VyField)
    gpuPhaseField = drv.to_device(PhaseField)
    gpuVFieldAverage = drv.to_device(VFieldAverage)
    gpuRhoField = drv.to_device(RhoField)
    gpuBosonField = drv.to_device(boson_field)
    gpuLattice = drv.to_device(Lattice)
    getPlotDetailsVorticity(gpuQField, gpuVField, gpuRhoField,  gpuBosonField, gpuPhaseField, gpuVxField, gpuVyField, gpuLattice, 
                block=(blockX,blockY,blockZ), grid=(gridX,gridY))
    VortField = drv.from_device(gpuVField, VortField.shape, DTYPE)
    boson_field = drv.from_device(gpuBosonField, boson_field.shape, DTYPE)
    #vortex_centers = find_dark_vortex_from_boson(boson_field) 
    vortex_centers = find_dark_vortex_from_vorticity(VortField) 
    x = vortex_centers[: , 0]
    y = vortex_centers[: , 1]
    colors = [color_code[c] for c in vortex_centers[:, 2]]
    plt.subplot(111)
    plt.scatter(x, y, c = colors, alpha = .25)
    #putLabels('', r'$y\ \ (\ell)$', r'$\rho \ \ (\frac{1}{\ell^2})$')
    ax = plt.gca()
    ax.set_aspect('equal')
    ax.set_ylim(0, ySize)
    ax.set_xlim(0, xSize)
    # Screen density    fig.tight_layout(pad=0.4, w_pad=5.0, h_pad=1.0, rect = [.05, .05, .95, .95])

    #Free GPU memory
    gpuQField.free()
    gpuVField.free()
    gpuVxField.free()
    gpuVyField.free()
    gpuPhaseField.free()
    gpuVFieldAverage.free()
    gpuRhoField.free()
    gpuBosonField.free()
    gpuLattice.free()
    return fig


