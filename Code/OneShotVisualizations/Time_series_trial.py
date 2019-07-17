
import os
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
import numpy as np 
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d import Axes3D





DTYPE = np.complex
vectorSize = 10
spinComps = 5

matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 15


# x1 = np.arange(9.0).reshape((3, 3))
# x2 = np.arange(9.0).reshape((3, 3))
# a = np.multiply(x1, x2)
# print(a)
# quit()


class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))



def calc_F(path, file):
    quantum_field = np.load(path + file) 
    plus_two = quantum_field[:, :, :, 0] + quantum_field[:, :, :, 1]
    plus_one = quantum_field[:, :, :, 2] + quantum_field[:, :, :, 3]
    zero = quantum_field[:, :, :, 4] + quantum_field[:, :, :, 5]
    minus_one = quantum_field[:, :, :, 6] + quantum_field[:, :, :, 7]
    minus_two = quantum_field[:, :, :, 8] + quantum_field[:, :, :, 9]
    Fz = 2.*(plus_two*plus_two.conjugate() - minus_two*minus_two.conjugate()) + plus_one*plus_one.conjugate() - minus_one*minus_one.conjugate()
    Fplus = 2.*(plus_two.conjugate()*plus_one + minus_one.conjugate()*minus_two) + np.sqrt(6.)*(plus_one.conjugate()*zero +zero.conjugate()*minus_one) 
    Fminus = Fplus.conjugate()
    Fx = (1./2.)*(Fplus + Fminus)
    Fy = (1./2.j)*(Fplus - Fminus)
    Ftrans = np.sqrt(Fx*Fx.conjugate() + Fy*Fy.conjugate())
    Fz = np.sqrt(Fz*Fz.conjugate())
    rho = plus_two*plus_two.conjugate() + plus_one*plus_one.conjugate() + minus_two*minus_two.conjugate() + minus_one*minus_one.conjugate() + zero*zero.conjugate()
    singlet_amplitude = np.abs((1./np.sqrt(5.))*(2.*plus_two*minus_two - 2.*plus_one*minus_one + zero*zero))
    max_sing = np.amax([.1 + 0.j, np.amax(singlet_amplitude)])
    max_z = np.amax([.1 + 0.j, np.amax(Fz)])
    max_trans = np.amax([.1 + 0.j, np.amax(Ftrans)])
    max_rho = np.amax([.1 + 0.j, np.amax(rho)])
    return max_rho, max_z, max_trans, max_sing, rho, Fz, Ftrans, singlet_amplitude


# dcmplx Fp_first_part = Mul(conj(phi[0]), phi[1]) + Mul(conj(phi[3]), phi[4]);
# dcmplx Fp_second_part = Mul(conj(phi[1]), phi[2]) + Mul(conj(phi[2]), phi[3]);
# dcmplx Fp_combined = Mul(two, Fp_first_part) + Mul(root6, Fp_second_part); 





def get_color(x, max_rho, max_z, max_trans, max_sing, rho, Fz, Ftrans, singlet_amplitude):
    VectorMag = np.sqrt(Fz[x, 0, 0].real * Fz[x, 0, 0].real + Ftrans[x, 0, 0].real * Ftrans[x, 0, 0].real + singlet_amplitude[x, 0, 0].real * singlet_amplitude[x, 0, 0].real)
    r = Fz[x, 0, 0].real/VectorMag
    g = Ftrans[x, 0, 0].real/VectorMag
    b = singlet_amplitude[x, 0, 0].real/VectorMag
    a = rho[x, 0, 0].real/max_rho.real
    if VectorMag < 0.0001:
        r, g, b = 0., 0., 0.
    return (r, g, b, a)




def find_non_empty_mf_levels(quantum_field):
    non_empty_mfs = []
    for mf in xrange(spinComps):
        if not (np.all(quantum_field[:, :, :, 2*mf]) == 0 or np.all(quantum_field[:, :, :, 2*mf + 1]) == 0):
            non_empty_mfs.append(mf)
    return non_empty_mfs

def waterfall_plot(fig,ax,X,Y,Z, path, files):
    print ("making waterfall plot for ", path)
    n,m = Z.shape
    total_steps = float(n*m) 
    for j in range(n):
        color_index_magic = n - j - 1
        max_rho, max_z, max_trans, max_sing, rho, Fz, Ftrans, singlet_amplitude = calc_F(path, files[color_index_magic])
        for k in range(m - 1):
        # reshape the X,Z into pairs
            x = np.asarray([X[j, k], X[j, k + 1]]) 
            y = np.asarray([Y[j, k], Y[j, k + 1]]) 
            z = np.asarray([Z[j, k], Z[j, k + 1]]) 
            c = get_color(k, max_rho, max_z, max_trans, max_sing, rho, Fz, Ftrans, singlet_amplitude)
            ax.plot(x, y, z, color = c, linewidth = 2)
            if k == 0 :
                print "%.2f" % (100.*(k + j*m)/total_steps) +  "%" 


def get_data_files(path):
    files = os.listdir(path)
    files.sort()
    return files


def get_nearest_divisible_N(N, files):
    if len(files) % (N - 1) == 0:
        return N
    else :
        try_higher = N + 1
        try_lower = N - 1
        while(True):  
            if len(files) % (try_higher - 1) == 0:
                print ("here")
                return try_higher
            if len(files) % (try_lower - 1) == 0:
                print("there")
                return try_lower
            try_higher += 1
            try_lower -= 1

def get_N_files(N, files, path):
    step_size = get_step_size(path)
    if N <= 1:
        print ("ERROR: N must be greater than 1.")
        quit() 
    new_files = []
    time_stamps = []
    # N = get_nearest_divisible_N(N, files)
    mod = int(len(files) / (N-1)) 
    for i, f in enumerate(files):
        if i % mod == 0:
            new_files.append(f)
            time_stamps.append(i)
    new_files = new_files[1:]
    time_stamps = time_stamps[1:]
    return np.asarray(time_stamps), new_files


def get_x_size(path):
    return int(path.split("XSize_")[1].split("_")[0])

def get_step_size(path):
    return int(path.split("Step_")[1].split("/")[0])


def get_total_bosonic_probabilities_for_mf(path, files, mf):
    print("print acquiring data for mf = ", mf)
    x_size = get_x_size(path)
    arr = np.zeros((len(files), x_size), dtype = DTYPE)
    for i, f in enumerate(reversed(files)):
        fermion_data = np.load(path + f)
        boson_data = np.zeros((x_size, spinComps), dtype = DTYPE)
        boson_data[:, mf] = np.add(fermion_data[:, 0, 0 , 2*mf], fermion_data[:, 0, 0, 2*mf + 1])
        boson_probability = np.multiply(boson_data, boson_data.conjugate()) 
        arr[i, :] = np.sum(boson_probability, axis = 1)
    return arr.real


def get_total_bosonic_probabilities(path, files):
    print("acquiring data .... ")
    x_size = get_x_size(path)
    arr = np.zeros((len(files), x_size), dtype = DTYPE)
    for i, f in enumerate(reversed(files)):
        fermion_data = np.load(path + f)
        boson_data = np.zeros((x_size, spinComps), dtype = DTYPE)
        for mf in xrange(spinComps):
            boson_data[:, mf] = np.add(fermion_data[:, 0, 0 , 2*mf], fermion_data[:, 0, 0, 2*mf + 1])
        boson_probability = np.multiply(boson_data, boson_data.conjugate()) 
        arr[i, :] = np.sum(boson_probability, axis = 1)
    return arr.real


def make_waterfall_plot(N, path, save = False, save_name = 'sample_waterfall'):
    N = N + 1
    x_size = get_x_size(path)
    step_size = get_step_size(path)
    files = get_data_files(path)
    time_stamps, new_files = get_N_files(N, files, path)
    waterfall_data = get_total_bosonic_probabilities(path, new_files)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x_axis = np.flip(np.linspace(0, x_size, x_size), 0)
    X,Y = np.meshgrid(x_axis, time_stamps)
    Z = waterfall_data
    waterfall_plot(fig, ax, X, Y, Z, path, new_files)
    ax.set_xlabel('x' + r'$(\ell)$') ;  ax.set_xlim3d(np.amin(x_axis),np.amax(x_axis))
    ax.set_ylim3d(np.amin(time_stamps),np.amax(time_stamps))
    ax.set_zlabel(r'$|\psi|^2$') ; ax.set_zlim3d(np.amin(Z),1.1*np.amax(Z))
    ax.zaxis.set_rotate_label(False) 
    ax.zaxis.labelpad = 10 
    ax.yaxis.set_rotate_label(False)
    fig.text(0.25, 0.25, r'$t ($' + str(step_size) + r'$\tau$)', fontsize=12, rotation = 315)  
    ax.xaxis.set_rotate_label(False)
    ax.yaxis.set_ticks([0, np.amax(time_stamps)/2, np.amax(time_stamps)])
    ax.set_yticklabels([np.amax(time_stamps), np.amax(time_stamps)/2, 0])
    ax.xaxis.set_ticks([0, np.amax(x_axis)/2, np.amax(x_axis)])
    ax.set_xticklabels([int(np.amax(x_axis)), int(np.amax(x_axis)/2), 0])
    ax.xaxis.labelpad = 10
    ax.grid(False)
    ax.view_init(60, 45)
    if save:
        print("supposedly saving somewhere")
        plt.savefig("WaterfallPlots/" + save_name + ".png")
    else:
        plt.show()
    plt.clf()
    plt.close()

def make_waterfall_plot_for_all_mf_levels(N, path, save = False, save_name = 'sample_waterfall'):
    N = N + 1
    x_size = get_x_size(path)
    step_size = get_step_size(path)
    files = get_data_files(path)
    mfs = find_non_empty_mf_levels(np.load(path + files[len(files) - 1]))
    for mf in mfs:
        time_stamps, new_files = get_N_files(N, files, path)
        waterfall_data = get_total_bosonic_probabilities_for_mf(path, new_files, mf)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        x_axis = np.flip(np.linspace(0, x_size, x_size), 0)
        X,Y = np.meshgrid(x_axis, time_stamps)
        Z = waterfall_data
        waterfall_plot(fig, ax, X, Y, Z)
        ax.set_xlabel('x' + r'$(\ell)$') ;  ax.set_xlim3d(np.amin(x_axis),np.amax(x_axis))
        ax.set_ylim3d(np.amin(time_stamps),np.amax(time_stamps))
        ax.set_zlabel(r'$|\psi|^2$') ; ax.set_zlim3d(np.amin(Z),1.1*np.amax(Z))
        ax.zaxis.set_rotate_label(False) 
        ax.zaxis.labelpad = 20 
        ax.yaxis.set_rotate_label(False)
        fig.text(0.15, 0.85, r'$t($' + str(step_size) + r'$\tau$)', fontsize=12)  
        ax.xaxis.set_rotate_label(False)
        ax.yaxis.set_ticks([0, np.amax(time_stamps)/2, np.amax(time_stamps)])
        ax.set_yticklabels([np.amax(time_stamps), np.amax(time_stamps)/2, 0])
        ax.xaxis.set_ticks([0, np.amax(x_axis)/2, np.amax(x_axis)])
        ax.set_xticklabels([int(np.amax(x_axis)), int(np.amax(x_axis)/2), 0])
        ax.grid(False)
        ax.view_init(30, 90)
        if save:
            plt.savefig(save_name + "_mf_" + str(mf) + ".pdf")
        else:
            plt.show()


def get_state1(s):
    return s.split("state1_")[1].split("_")[0]

def get_state2(s):
    if "state2_" in s:
        return s.split("state2_")[1].split("_")[0]
    else:
        return "0"

# pre_path = "/Volumes/Simulations/QuantumLatticeGas/Experiments/charmonium/SimVersion_2.0/spin_2_BEC/default/stationary_states/"
# post_path = "/XSize_4096_YSize_1_ZSize_1/G0_-1.0_G1_-1.0_G2_-1.0_MU_-1.0_Scaling_30.0_Step_1000/Data/"


pre_path = '/Volumes/Simulations/QuantumLatticeGas/Experiments/charmonium/SimVersion_2.0/spin_2_BEC_1D_test/default/kink_states/'
post_path = '/XSize_2048_YSize_1_ZSize_1/G0_1.0_G1_1.0_G2_1.0_MU_1.0_Scaling_50.0_Step_1000/Data/'


def waterfalls_for_big_file(N, pre_path, post_path):
    files = get_data_files(path = pre_path)
    for f in files:
        s1 = get_state1(f)
        s2 = get_state2(f)
        print (s1, s2)
        p = pre_path + f + post_path
        save_name = "s1_" + s1 + "_s2_"+ s2
        if not 'c' in f:
            make_waterfall_plot(N, p, save = True, save_name = save_name)

#waterfalls_for_big_file(25, pre_path, post_path)

#calc_F()
path = "/media/qcgroup/HD1/2019/QLGA_Simulator/Experiments/quasar/SimVersion_2.0/spin2_BEC_1D/default/kink_states/p_10_state1_13/XSize_4096_YSize_1_ZSize_1/G0_1.0_G1_1.0_G2_1.0_MU_0.0_Scaling_30.0_Step_2000/Data/"

make_waterfall_plot(25, path, save = True, save_name = 'dark_kicked_13')
#make_waterfall_plot_for_all_mf_levels(25, path, save = True, save_name = 's1_11_s2_7')

