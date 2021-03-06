import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
import mpl_toolkits
import mpl_toolkits.mplot3d
import os
os.environ['CUDA_DEVICE'] = str(1) #Set CUDA device, starting at 0
import pycuda.autoinit
import pycuda.driver as drv
import VIS_GPU_1D_2P_SOURCE as gpuVis
from scipy.integrate import simps

gpuMagic = gpuVis.gpuSource
get_QS = gpuMagic.get_function("getQS_x1_x2")
get_Rho_projected = gpuMagic.get_function("getRho_projected")
get_Analytic = gpuMagic.get_function("getAnalyticField")
calcPurity = gpuMagic.get_function("calcPurity")

#Set a datatype to use for arrays
DTYPE = np.complex
spinComps = 1

matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 15


yMax = 0.
rhoMin = 0.
rhoMax = 0.
linThresh = .1
linScale = 1
colorMapVort = cm.bwr
colorMapRho = cm.copper
colorMapE = cm.inferno
colorMapPhase = cm.gist_rainbow

def arrayFunc2D(array, x, y):
    val = 0.
    xHigh = np.ceil(x).astype(np.int32)
    xLow = np.floor(x).astype(np.int32)
    yHigh = np.ceil(y).astype(np.int32)
    yLow = np.floor(y).astype(np.int32)
    # xVal, yVal
    return (1.-(xHigh-x))*array[xHigh]+(1.-(x-xLow))*array[xLow], (1.-(yHigh-y))*array[yHigh]+(1.-(y-yLow))*array[yLow]

# arr = np.arange(16).reshape(4,4)
# for idx in np.ndindex(4):
#     for idy in np.ndindex(4):
#         print(idx, arr[idx][idy])
# quit()

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

def setXaxis(xLen):
    xAxis = np.zeros((xLen,), dtype=np.int)
    for x in xrange(xLen):
        xAxis[x] = 0 + x
    return xAxis

def putLabels(x_label, y_label, colorbar_label):
  cbar = plt.colorbar(shrink = .5)
  cbar.set_label(colorbar_label, fontsize=14)
  cbar.ax.tick_params(labelsize=10)
  plt.xlabel(x_label).set_fontsize(24)
  plt.ylabel(y_label).set_fontsize(24)
  plt.tick_params(axis="both",which="major",labelsize=16)
  plt.tick_params(axis="both",which="minor",labelsize=16)

def make_frame(frame_dir, frame, image_dir, frames, global_vars, find_total_max = False, save_density = False, save_entanglement = False, save_expectation_value = False, exp_range = [0,0], **kwargs):
    global yMax, rhoMax, rhoMin
    frame_number = (frame.split("_")[1]).split(".")[0]
    print "Plotting", frame_dir
    QuantumState = np.load(frame_dir)
    fig = plt.figure(figsize = (15,12))
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
    Time[0] = int(frame_number)
    purity = np.zeros(1, dtype = np.float64)

    gpuPurity = drv.to_device(purity)
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
    if save_entanglement:
        calcPurity(gpuQF_x1_x2_real, gpuQF_x1_x2_imag, gpuPurity, gpuLattice, block=(blockX_x1_x2,blockY_x1_x2,blockZ_x1_x2), grid=(gridX_x1_x2,gridY_x1_x2,gridZ_x1_x2))
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

    #Get Rho and Phase from QField
    RhoField = (QuantumField_x1_x2_real+1.j*QuantumField_x1_x2_imag)*(QuantumField_x1_x2_real+1.j*QuantumField_x1_x2_imag).conjugate()
    PhaseField = np.arctan2(QuantumField_x1_x2_imag,QuantumField_x1_x2_real)+np.pi

    if int(frame_number) == 0:
        rhoMin, rhoMax = np.amin(RhoField).real, np.amax(RhoField).real
        yMax = np.amax(RhoFieldProjected.real)


    time = int(frame_number)
    purity = drv.from_device(gpuPurity, purity.shape, np.float64)
    entanglement = 1-2.*purity[0]/(rho_total*rho_total)
    xAxis = setXaxis(xSize)

    time_text = plt.suptitle(r'$\tau = $' + str(time) ,fontsize=14,horizontalalignment='center',verticalalignment='top')
    if save_entanglement:
        time_text = plt.suptitle(r'$\tau = $' + str(time) + '    ' + r'$\mathcal{E} = $' + str(entanglement) ,fontsize=14,horizontalalignment='center',verticalalignment='top')
    # full_text = plt.suptitle(r'$\tau = $' + str(time) + '    ' + r'$P_{f} = $' + str('{:1.15f}'.format(prob)) + '     ' + r'$P_{p} = $' + str('{:1.15f}'.format(probP)),fontsize=14,horizontalalignment='center',verticalalignment='top')
    # gs = gridspec.GridSpec(1,1)
    # ax = fig.add_subplot(gs[0,0], xlim=(0,xSize), xlabel=r'$x(\ell)$', ylim=(-0.0000001, 1.1*yMax), ylabel=r'${| \Psi |}^{2}$')


    ############################### PLOTTING ####################################################
    # Density
    plt.subplot(121)
    plt.imshow((RhoField.real), extent=(np.amin(xAxis), np.amax(xAxis), np.amin(xAxis), np.amax(xAxis)), origin = 'lower',
        cmap=colorMapRho, norm=colors.SymLogNorm(linthresh=linThresh*rhoMax,linscale=linScale,vmin=0.,vmax=rhoMax))
    putLabels(r'$x_{1}\ \ (\ell)$', r'$x_{2}\ \ (\ell)$', r'$\rho \ \ (\frac{1}{\ell^2})$')
    ax = plt.gca()
    ax.set_aspect('equal')

    #Phase
    plt.subplot(122)
    plt.imshow((PhaseField.real), extent=(np.amin(xAxis), np.amax(xAxis), np.amin(xAxis), np.amax(xAxis)), origin = 'lower',
        cmap=colorMapPhase, norm=colors.Normalize(vmin=0.,vmax=2.*np.pi))
    putLabels(r'$x_{1}\ \ (\ell)$', r'$x_{2}\ \ (\ell)$', r'$\theta \ \ (Radians)$')
    # Any value whose absolute value is > .0001 will have zero transparency
    alphas = Normalize(0, rhoMax, clip=True)((rhoMax-RhoField.real))
    # alphas = colors.SymLogNorm(linthresh=linThresh*rhoMax,linscale=linScale,vmin=0.,vmax=10., clip=True)((rhoMax-RhoField.real).T)
    alphas = np.clip(alphas**4, 0.0, 1)  # alpha value clipped at the bottom at .4
    # Normalize the colors b/w 0 and 1, we'll then pass an MxNx4 array to imshow
    cmap = plt.cm.gist_gray
    colorsMap = colors.SymLogNorm(linthresh=linThresh*rhoMax,linscale=linScale,vmin=0.,vmax=rhoMax)((0.*RhoField.real))
    colorsMap = cmap(colorsMap)

    # Now set the alpha channel to the one we created above
    colorsMap[..., -1] = alphas
    plt.imshow(colorsMap, extent=(np.amin(xAxis), np.amax(xAxis), np.amin(xAxis), np.amax(xAxis)), origin = 'lower')
    ax = plt.gca()
    ax.set_aspect('equal')

    # Screen density


    fig.tight_layout(pad=0.4, w_pad=5.0, h_pad=1.0, rect = [.05, .05, .95, .95])

    if save_density:
        rho_dir = frame_dir.split("Data")[0] + "Density"
        if not os.path.exists(rho_dir+ '/'):
            os.makedirs(rho_dir + '/')
        print "Saving density to", rho_dir + '/Frame_' + frame_number +'.npy'
        np.save(rho_dir + '/Frame_' + frame_number , RhoFieldProjected.real)

	if save_entanglement:
		ent_dir = frame_dir.split("Data")[0] + "Entanglement"
		if not os.path.exists(ent_dir+ '/'):
			os.makedirs(ent_dir + '/')
		print "Saving entanglement to", ent_dir + '/entanglement.npy'
		if time==0:
			entanglementArray = np.asarray([[entanglement, time]])
		else:
			entArrayOld = np.load(ent_dir + '/entanglement.npy'  )
			entArrayNew = np.asarray([[entanglement, time]])
			entanglementArray = np.append(entArrayOld, entArrayNew, axis=0)
		np.save(ent_dir + '/entanglement' , entanglementArray)

    if save_expectation_value:
        exp_dir = frame_dir.split("Data")[0] + "expectation"
        if not os.path.exists(exp_dir+ '/'):
            os.makedirs(exp_dir + '/')
        print "Saving expectation to", exp_dir + '/expectation.npy'
        exp_val = 0.
        if exp_range==[0,0]:
            for x in xrange(xSize):
                exp_val += x*RhoFieldProjected[x].real
        else:
            for x in xrange(int(exp_range[0]*xSize),int(exp_range[1]*xSize),1):
                exp_val += x*RhoFieldProjected[x].real
        if time==0:
            expectationArray = np.asarray([exp_val])
        else:
            expArrayOld = np.load(exp_dir + '/expectation.npy'  )
            expArrayNew = np.asarray([exp_val])
            expectationArray = np.append(expArrayOld, expArrayNew, axis=0)
        np.save(exp_dir + '/expectation' , expectationArray)

    #Free memory
    gpuLattice.free()
    gpuQField.free()
    gpuQF_x1_x2_real.free()
    gpuQF_x1_x2_imag.free()
    gpuRho_projected_real.free()

    gpuTime.free()
    gpuPurity.free()
    gpuAnalyticFieldReal.free()
    gpuAnalyticFieldImag.free()

    if not os.path.exists(image_dir + '/'):
        os.makedirs(image_dir + '/')
    fig.savefig(image_dir + '/Frame_' + frame_number +".png")
    plt.close(fig)


