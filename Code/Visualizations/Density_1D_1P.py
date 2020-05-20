






import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import mpl_toolkits
import mpl_toolkits.mplot3d
import matplotlib.gridspec as gridspec



plt.rcParams.update({'font.size': 20})

DTYPE = np.complex
vectorSize = 2

def calculateRho(QuantumField):
    rho = np.zeros(xSize, dtype = DTYPE)
    psi = QuantumField[:, 0, 0, 0] + QuantumField[:, 0, 0, 1]
    for i in xrange(vectorSize):
        rho[:] = rho[:] + (psi[:] * (psi[:]).conjugate())
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
xAxis = None

def set_globals(global_vars, QuantumState):
    global yMax, xSize, xAxis, fieldMax, vectorSize
    xSize = global_vars["xSize"]
    vectorSize = global_vars["vectorSize"]
    xAxis = setXaxis(xSize)
    yMax = np.amax(calculateRho(QuantumState))


def make_frame(frame_dir, frame, image_dir, frames, global_vars, find_total_max = False, **kwargs):
    # print i
    frame_number = (frame.split("_")[1]).split(".")[0]
    fig = plt.figure(figsize = (15,12))
    plt.rc('text', usetex=True)
    QuantumState = np.load(frame_dir)
    if int(frame_number) == 0:
        set_globals(global_vars, QuantumState)
    rho = calculateRho(QuantumState)
    Prob = np.sum(rho)
    time = int(frame_number)
    time_text = plt.suptitle(r'$\tau = $' + str(time) + "              "
                       + "              " + '$P = $' + str(Prob.real) ,fontsize=14,horizontalalignment='center',verticalalignment='top')
    gs = gridspec.GridSpec(1,1)
    ax = fig.add_subplot(gs[0,0], xlim=(0,xSize-1), xlabel=r'$x(\ell)$', ylim=(-0.000001, 1.1*yMax), ylabel=r'${| \Psi |}^{2}$')
    ax.plot(rho.real,zorder=2)

    if not os.path.exists(image_dir + '/'):
        os.makedirs(image_dir + '/')
    fig.savefig(image_dir + '/Frame_' + frame_number +".png")
    print "Saved " + image_dir + '/Frame_' + frame_number +".png"
    plt.close(fig)

