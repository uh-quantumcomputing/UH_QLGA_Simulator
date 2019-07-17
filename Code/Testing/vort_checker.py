import numpy as np 
import os
import matplotlib.pyplot as plt

directory = '/media/qcgroup/HD1/2019/SynchedV4/Experiments/quasar/SimVersion_2.0/spin2_BEC_2D/default/double_quadrupole/square_size_7_mf_1_2_mf_2_1/XSize_512_YSize_512_ZSize_1/G0_0.005_G1_0.0004_G2_0.005_MU_0.005_Scaling_600.0_Step_10/Extra/'

dir2 = '/media/qcgroup/HD1/2019/SynchedV4/Experiments/quasar/SimVersion_2.0/spin2_BEC_2D/default/double_quadrupole/square_size_7_mf_1_2_mf_2_1/XSize_512_YSize_512_ZSize_1/G0_0.005_G1_0.0004_G2_0.0_MU_0.005_Scaling_600.1_Step_10/Data/Frame_00000800.npy'


# files = os.listdir(directory)

# for f in files:
# 	if '.npy' in f:
# 		print(f, np.amax(np.load(directory + f)))



# a = np.load(directory + 'vorticity_mf23.npy')
# print a[:, :, 0, 0].shape
# b = np.asarray([[0, 1, 1], 
# 				[1, 1, 1]])
# x, y = np.where(a[:, :, 0, 0] > 20)
# points = zip(x,y)
# print points



for i in xrange(50):
	plt.plot(xrange(512), (np.load(dir2)[:, 10*i , 0, 2]+ np.load(dir2)[:, 10*i, 0, 3])*((np.load(dir2)[:, 10*i , 0, 2]+ np.load(dir2)[:, 10*i , 0, 3]).conj()))
	print(i*10)
	plt.ylim(0,2.)
	plt.show()
