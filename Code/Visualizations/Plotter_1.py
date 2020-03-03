import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.gridspec as gridspec
import mpl_toolkits
import mpl_toolkits.mplot3d



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
        prob = prob + calculateRho(QuantumField,x) 
    return prob

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


yMax = 0.
xSize = 0
A = 0.
E = 0.
scaling = 0.
tau = 0.
xAxis = None
fieldMax = 0.

def set_globals(file_name, QuantumState):
    global yMax, xSize, A, E, scaling, tau, xAxis, fieldMax
    xSize = np.size(QuantumState,0)
    xAxis = setXaxis(xSize)
    scaling = float(file_name.split("Scaling_")[1].split("_")[0])
    A = scaling/(xSize)
    tau = A*A
    yMax = np.amax(calculateRho(QuantumState))
    for n in xrange(spinComps):
        if np.amax((QuantumState[:, 0, 0, 2*n]+QuantumState[:, 0, 0, 2*n+1]).real) > fieldMax:
            fieldMax = np.amax((QuantumState[:, 0, 0, 2*n]+QuantumState[:, 0, 0, 2*n+1]).real)
        if np.amax((QuantumState[:, 0, 0, 2*n]+QuantumState[:, 0, 0, 2*n+1]).imag) > fieldMax:
            fieldMax = np.amax((QuantumState[:, 0, 0, 2*n]+QuantumState[:, 0, 0, 2*n+1]).imag)
        if np.abs(np.amin((QuantumState[:, 0, 0, 2*n]+QuantumState[:, 0, 0, 2*n+1]).imag)) > fieldMax:
            fieldMax = np.abs(np.amin((QuantumState[:, 0, 0, 2*n]+QuantumState[:, 0, 0, 2*n+1]).imag))
        if np.abs(np.amin((QuantumState[:, 0, 0, 2*n]+QuantumState[:, 0, 0, 2*n+1]).real)) > fieldMax:
            fieldMax = np.abs(np.amin((QuantumState[:, 0, 0, 2*n]+QuantumState[:, 0, 0, 2*n+1]).real))


def calculate_energy(QuantumState, oldQuantumState, Prob, file_name):
    innerPsiPrimePsi = np.zeros(xSize, dtype = DTYPE)
    step_size = int(file_name.split("Step_")[1].split("/")[0])
    for n in xrange(vectorSize):
        innerPsiPrimePsi[:] += QuantumState[:,0,0,n].conjugate() * oldQuantumState[:, 0, 0, n]
    return 1j*(Prob-np.sum(innerPsiPrimePsi))/(tau*step_size*Prob)

def animate(i, fig, file_name):
    # print i
    frame_number = int(file_name.split("Frame_")[1].split(".")[0])
    QuantumState = np.load(file_name)
    oldQuantumState = None
    if i == 0:
        set_globals(file_name, QuantumState)
        oldQuantumState = QuantumState.copy()
    else:
        step_size = int(file_name.split("Step_")[1].split("/")[0])
        f_num = frame_number - step_size
        f = file_name.split("Frame_")[0] + "Frame_" + str(f_num).zfill(8) + ".npy"
        oldQuantumState = np.load(f)
    rho = calculateRho(QuantumState)
    Prob = np.sum(rho)
    E = calculate_energy(QuantumState, oldQuantumState, Prob, file_name)  
    time = frame_number
    time_text = plt.suptitle(r'$\tau = $' + str(time) + "              " + '$E_{0} = $' + str(E.real)
                       + "              " + '$P = $' + str(Prob.real) ,fontsize=14,horizontalalignment='center',verticalalignment='top')
    gs = gridspec.GridSpec(3,5)
    ax = fig.add_subplot(gs[0,0:5], xlim=(0,xSize-1), xlabel=r'$x(\ell)$', ylim=(-0.001, 1.1*yMax), ylabel=r'${| \Psi |}^{2}$')
    ax1 = fig.add_subplot(gs[2,0],projection='3d', xlim=(-fieldMax, fieldMax), ylim=(-fieldMax, fieldMax), xlabel=r'$\Re({\Psi}_{2})$', ylabel=r'$\Im({\Psi}_{2})$', xticks=[], yticks=[])
    ax1.xaxis.set_label_coords(0.5, -0.01)
    ax1.yaxis.set_label_coords(0.5, -0.01)
    ax1A = fig.add_subplot(gs[1,0], xlim=(0,xSize-1), xlabel=r'$x(\ell)$', ylim=(-0.001, 1.1*yMax), ylabel=r'${| \Psi_{2} |}^{2}$')
    ax2 = fig.add_subplot(gs[2,1],projection='3d',xlim=(-fieldMax, fieldMax), ylim=(-fieldMax, fieldMax), xlabel=r'$\Re({\Psi}_{1})$', ylabel=r'$\Im({\Psi}_{1})$', xticks=[], yticks=[])
    ax2.xaxis.set_label_coords(0.5, 0)
    ax2.yaxis.set_label_coords(0.5, 0)
    ax2A = fig.add_subplot(gs[1,1], xlim=(0,xSize-1), xlabel=r'$x(\ell)$', ylim=(-0.001, 1.1*yMax), yticks=[], ylabel=r'${| \Psi_{1} |}^{2}$')
    ax3 = fig.add_subplot(gs[2,2],projection='3d',xlim=(-fieldMax, fieldMax), ylim=(-fieldMax, fieldMax), xlabel=r'$\Re({\Psi}_{0})$', ylabel=r'$\Im({\Psi}_{0})$', xticks=[], yticks=[])
    ax3.xaxis.set_label_coords(0.5, 0)
    ax3.yaxis.set_label_coords(0.5, 0)
    ax3A = fig.add_subplot(gs[1,2], xlim=(0,xSize-1), xlabel=r'$x(\ell)$', ylim=(-0.001, 1.1*yMax), yticks=[], ylabel=r'${| \Psi_{0} |}^{2}$')
    ax4 = fig.add_subplot(gs[2,3],projection='3d',xlim=(-fieldMax, fieldMax), ylim=(-fieldMax, fieldMax), xlabel=r'$\Re({\Psi}_{-1})$', ylabel=r'$\Im({\Psi}_{-1})$', xticks=[], yticks=[])
    ax4.xaxis.set_label_coords(0.5, 0)
    ax4.yaxis.set_label_coords(0.5, 0)
    ax4A = fig.add_subplot(gs[1,3], xlim=(0,xSize-1), xlabel=r'$x(\ell)$', ylim=(-0.001, 1.1*yMax), yticks=[], ylabel=r'${| \Psi_{-1} |}^{2}$')
    ax5 = fig.add_subplot(gs[2,4],projection='3d',xlim=(-fieldMax, fieldMax), ylim=(-fieldMax, fieldMax), xlabel=r'$\Re({\Psi}_{-2})$', ylabel=r'$\Im({\Psi}_{-2})$', xticks=[], yticks=[])
    ax5.xaxis.set_label_coords(0.5, 0)
    ax5.yaxis.set_label_coords(0.5, 0)
    ax5A = fig.add_subplot(gs[1,4], xlim=(0,xSize-1), xlabel=r'$x(\ell)$', ylim=(-0.001, 1.1*yMax), yticks=[], ylabel=r'${| \Psi_{-2} |}^{2}$')
    ############################### PLOTTING ####################################################
    ax.plot(xAxis, calculateRho(QuantumState).real)
    ### Potential barrier
    # tophat = np.zeros(len(xAxis))
    # wall_width = 60.
    # for x in xrange(xSize):
    #     if ((x>(2.*xSize/5.-wall_width/2.)) and (x<(2.*xSize/5.+wall_width/2.))):
    #         tophat[x] = np.pi/16.
    #     elif (x>(xSize-wall_width)):
    #         tophat[x] = np.pi/16.
    #     elif (x<(wall_width)):
    #         tophat[x] = np.pi/16.
    # ax.plot(xAxis, tophat, "r-")
    ax1A.plot(xAxis, calculateRhoComp(QuantumState,0).real) 
    ax2A.plot(xAxis, calculateRhoComp(QuantumState,2).real) 
    ax3A.plot(xAxis, calculateRhoComp(QuantumState,4).real) 
    ax4A.plot(xAxis, calculateRhoComp(QuantumState,6).real) 
    ax5A.plot(xAxis, calculateRhoComp(QuantumState,8).real) 
    ax1.plot((QuantumState[:, 0, 0, 0]+QuantumState[:, 0, 0, 1]).real, (QuantumState[:, 0, 0, 0]+QuantumState[:, 0, 0, 1]).imag, xAxis, color = 'red')
    ax2.plot((QuantumState[:, 0, 0, 2]+QuantumState[:, 0, 0, 3]).real, (QuantumState[:, 0, 0, 2]+QuantumState[:, 0, 0, 3]).imag, xAxis, color = 'orange')
    ax3.plot((QuantumState[:, 0, 0, 4]+QuantumState[:, 0, 0, 5]).real, (QuantumState[:, 0, 0, 4]+QuantumState[:, 0, 0, 5]).imag, xAxis, color = 'yellow')
    ax4.plot((QuantumState[:, 0, 0, 6]+QuantumState[:, 0, 0, 7]).real, (QuantumState[:, 0, 0, 6]+QuantumState[:, 0, 0, 7]).imag, xAxis, color = 'green')
    ax5.plot((QuantumState[:, 0, 0, 8]+QuantumState[:, 0, 0, 9]).real, (QuantumState[:, 0, 0, 8]+QuantumState[:, 0, 0, 9]).imag, xAxis, color = 'blue')
    gs.tight_layout(fig, rect = (.05, .05, .95, .95))
    return fig



