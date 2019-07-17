import numpy as np
import matplotlib 
import  matplotlib.pyplot as plt


matplotlib.rcParams['legend.fontsize'] = 10
matplotlib.rcParams['text.usetex'] = False
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 15






# sizes should be sizes that L2 norm is calculated at for example 
#sizes =  [256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
sizes =  [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
#File name should be named at appropriate timestep that is compared to t = 0  
file_name1 = "/media/qcgroup/HD1/2019/SynchedV4/Experiments/quasar/SimVersion_2.0/spin2_BEC_1D/default/kink_states/state1_14/XSize_256_YSize_1_ZSize_1/G0_1.0_G1_1.0_G2_1.0_MU_1.0_Scaling_30.0_Step_1000/Data/Frame_00001000.npy"
file_name2 = "/media/qcgroup/HD1/2019/SynchedV4/Experiments/quasar/SimVersion_2.0/spin2_BEC_1D_no_trot/default/kink_states/state1_14/XSize_256_YSize_1_ZSize_1/G0_1.0_G1_1.0_G2_1.0_MU_1.0_Scaling_30.0_Step_1000/Data/Frame_00001000.npy"
save_name = 'bright_states_L2'

def file_list_from_sizes(file_name, sizes):
	files = []
	prefix = file_name.split('XSize_')[0]
	suffix = file_name.split('_YSize')[1]
	for s in sizes:
		files.append(prefix+'XSize_' + str(s) +'_YSize' +  suffix)
	return files


def L2norm(analytic_field,sim_field):
	rhoAnalytic = np.real(np.conj(analytic_field)*analytic_field)
	rhoSim = np.real(np.conj(sim_field)*sim_field)
	return np.sum((rhoAnalytic-rhoSim)**2)


def set_y_data(files, sizes):
	y_data = []
	for f in files:
		prefix = f.split('Frame_')[0]
		suffix = '.npy'
		analytic_field = np.load(prefix+ 'Frame_'+'00000000' + suffix)
		sim_field = np.load(f)
		y_data.append(L2norm(analytic_field, sim_field))
	return np.log(y_data)



def L2_norms(y_data, sizes):
	x_data = np.log(sizes)
	print (np.polyfit(x_data, y_data, 1))
	plt.scatter(x_data, y_data)
	plt.plot(np.unique(x_data), np.poly1d(np.polyfit(x_data, y_data, 1))(np.unique(x_data)))

def plot_one_L2(file_name, sizes, show = True):
	print(file_name)
	files = file_list_from_sizes(file_name, sizes)
	y_data = set_y_data(files, sizes)
	L2_norms(y_data, sizes)
	if show:
		plt.gca().set(xlabel='Log(L)', ylabel='Log('+ r'$\epsilon$' +')')
		plt.savefig("L2_norm_plots/" + save_name + ".png")
		plt.show()

def plot_two_L2(file_name1, file_name2, sizes):
	plot_one_L2(file_name1, sizes, show = False)
	plot_one_L2(file_name2, sizes)


plot_two_L2(file_name1, file_name2, sizes)