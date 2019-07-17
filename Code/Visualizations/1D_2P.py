import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import mpl_toolkits
import mpl_toolkits.mplot3d
import pycuda.autoinit
import pycuda.driver as drv
import VIS_GPU_1D_2P_SOURCE as gpuVis

gpuMagic = gpuVis.gpuSource
get_QS = gpuMagic.get_function("getQS_x1_x2")
get_Rho_projected = gpuMagic.get_function("getRho_projected")
get_Analytic = gpuMagic.get_function("getAnalyticField")

#Set a datatype to use for arrays
DTYPE = np.complex
spinComps = 1

matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 15


yMax = 0.

'''
------------------------------------------
-------- PLOT AND ANIMATE SECTION --------
------------------------------------------
'''
def setXaxis(xLen):
    xAxis = np.zeros((xLen,), dtype=np.int)
    for x in xrange(xLen):
        xAxis[x] = 0 + x
    return xAxis

def rho_projected(x,psi):
    val = 0.
    for l in xrange(xSize):
        val += psi[x,l]*(psi[x,l].conjugate())
    return val

def psi_projected(x,psi):
    val = 0.
    for l in xrange(xSize):
        val += psi[x,l]
    return val


def make_frame(frame_dir, frame, image_dir, frames, global_vars, find_total_max = False, **kwargs):
    global yMax
    frame_number = int((frame.split("_")[1]).split(".")[0])
    print "Plotting", frame_dir
    QuantumState = np.load(frame_dir)
    fig = plt.figure(figsize = (15,9))
    oldQuantumState = None
    oldQuantumState = QuantumState.copy()
    xSize = global_vars["Qx"]/2
    ySize = 1
    zSize = 1
    blockX = 256
    x_nSize = xSize*(2*xSize-1)
    while x_nSize%blockX != 0:
      blockX /= 2
    gridX = x_nSize/blockX
    blockY = 1
    blockZ = 1
    gridY = 1
    gridZ = 1
    blockX_x1_x2 = 16
    blockY_x1_x2 = 16
    blockZ_x1_x2 = 1
    while xSize%blockX_x1_x2 != 0:
      blockX_x1_x2 /= 2
      blockY_x1_x2 /= 2
    gridX_x1_x2 = xSize/blockX_x1_x2
    gridY_x1_x2 = xSize/blockX_x1_x2
    gridZ_x1_x2 = 1

    QuantumField_x1_x2_real = np.zeros((xSize, xSize), dtype = np.float64)
    QuantumField_x1_x2_imag = np.zeros((xSize, xSize), dtype = np.float64)
    Rho_projected_real = np.zeros((xSize), dtype = np.float64)
    AnalyticFieldReal = np.zeros((xSize, xSize), dtype = np.float64)
    AnalyticFieldImag = np.zeros((xSize, xSize), dtype = np.float64)
    Lattice = np.zeros(3, dtype = np.int_)
    Lattice[0],  Lattice[1], Lattice[2]= xSize, ySize, zSize
    Time = np.zeros(1, dtype = np.float64)
    Time[0] = frame_number

    gpuLattice = drv.to_device(Lattice)
    gpuTime = drv.to_device(Time)
    gpuQField = drv.to_device(QuantumState)
    gpuQF_x1_x2_real = drv.to_device(QuantumField_x1_x2_real)
    gpuQF_x1_x2_imag = drv.to_device(QuantumField_x1_x2_imag)
    gpuRho_projected_real = drv.to_device(Rho_projected_real)
    gpuAnalyticFieldReal = drv.to_device(AnalyticFieldReal)
    gpuAnalyticFieldImag = drv.to_device(AnalyticFieldImag)
    get_QS(gpuQField, gpuQF_x1_x2_real, gpuQF_x1_x2_imag, gpuLattice, block=(blockX,blockY,blockZ), grid=(gridX,gridY,gridZ))
    get_Rho_projected(gpuQF_x1_x2_real, gpuQF_x1_x2_imag, gpuRho_projected_real, gpuLattice, block=(blockX_x1_x2,blockY_x1_x2,blockZ_x1_x2), grid=(gridX_x1_x2,gridY_x1_x2,gridZ_x1_x2))
    # get_Analytic(gpuAnalyticFieldReal, gpuAnalyticFieldImag, gpuAn, gpuBn, gpuAn2, gpuBn2, gpuLattice, gpuTime, block=(blockX,blockY,blockZ), grid=(gridX,gridY,gridZ))
    QuantumField_x1_x2_real = drv.from_device(gpuQF_x1_x2_real, QuantumField_x1_x2_real.shape, np.float64)
    QuantumField_x1_x2_imag = drv.from_device(gpuQF_x1_x2_imag, QuantumField_x1_x2_imag.shape, np.float64)
    Rho_projected_real = drv.from_device(gpuRho_projected_real, Rho_projected_real.shape, np.float64)
    AnalyticFieldReal = drv.from_device(gpuAnalyticFieldReal, AnalyticFieldReal.shape, np.float64)
    AnalyticFieldImag = drv.from_device(gpuAnalyticFieldImag, AnalyticFieldImag.shape, np.float64)
    RhoFieldProjected = np.zeros((xSize), dtype = DTYPE)
    RhoFieldProjectedAnalytic = np.zeros((xSize), dtype = DTYPE)

    rho_total = np.sum(Rho_projected_real)
    # print rho_total
    for l in xrange(xSize):
        RhoFieldProjected[l] = Rho_projected_real[l]/rho_total

    if frame_number == 0:
        yMax = np.amax(RhoFieldProjected.real)

    time = frame_number
    # prob = np.sum(RhoFieldProjected.real)
    # probP = np.sum((QuantumState*QuantumState.conjugate()).real)
    time_text = plt.suptitle(r'$\tau = $' + str(time) ,fontsize=14,horizontalalignment='center',verticalalignment='top')
    # full_text = plt.suptitle(r'$\tau = $' + str(time) + '    ' + r'$P_{f} = $' + str('{:1.15f}'.format(prob)) + '     ' + r'$P_{p} = $' + str('{:1.15f}'.format(probP)),fontsize=14,horizontalalignment='center',verticalalignment='top')
    gs = gridspec.GridSpec(1,1)
    ax = fig.add_subplot(gs[0,0], xlim=(0,xSize), xlabel=r'$x(\ell)$', ylim=(-0.0000001, 1.1*yMax), ylabel=r'${| \Psi |}^{2}$')


    ############################### PLOTTING ####################################################
    xdata = np.arange(xSize)
    ax.plot(RhoFieldProjected.real,zorder=2)


    #Free memory
    gpuLattice.free()
    gpuQField.free()
    gpuQF_x1_x2_real.free()
    gpuQF_x1_x2_imag.free()
    gpuRho_projected_real.free()

    gpuTime.free()
    gpuAnalyticFieldReal.free()
    gpuAnalyticFieldImag.free()

    if not os.path.exists(image_dir + '/'):
        os.makedirs(image_dir + '/')
    fig.savefig(image_dir + '/Frame_' + str(frame_number) +".png")
    plt.close(fig)


