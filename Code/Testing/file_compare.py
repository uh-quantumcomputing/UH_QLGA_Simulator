import numpy as np
import os


directory = '/Volumes/Simulations/QuantumLatticeGas/Experiments/charmonium/SimVersion_2.0/spin_2_BEC_1D_test/default/kink_states/state1_12/XSize_2048_YSize_1_ZSize_1/G0_1.0_G1_1.0_G2_1.0_MU_1.0_Scaling_50.0_Step_1000/Data/Frame_00002000.npy'
directory2 = '/Volumes/Simulations/QuantumLatticeGas/Experiments/charmonium/SimVersion_2.0/spin_2_BEC_1D_test/default/kink_states/state1_12/XSize_2048_YSize_1_ZSize_1/G0_1.0_G1_1.0_G2_1.0_MU_1.0_Scaling_50_Step_1000/Data/Frame_00002000.npy'

def compare (directory, directory2):
	state1 = np.load(directory)
	state2 = np.load(directory2)
	max_diff_real = 0.
	max_diff_imag = 0.
	min_diff_real = 0.
	min_diff_imag = 0.
	for x in xrange(2048):
		#print("state 1:", x, state1[x, 0, 0, :])
		#print("state 2:", x, state2[x, 0, 0, :])
		if np.amax(state1[x, 0, 0, :].real- state2[x, 0, 0, :].real) > max_diff_real:
			max_diff_real = np.amax(state1[x, 0, 0, :].real- state2[x, 0, 0, :].real)
		if np.amax(state1[x, 0, 0, :].imag- state2[x, 0, 0, :].imag) > max_diff_imag:
			max_diff_imag = np.amax(state1[x, 0, 0, :].imag- state2[x, 0, 0, :].imag)
		if np.amin(state1[x, 0, 0, :].real- state2[x, 0, 0, :].real) < min_diff_real:
			min_diff_real = np.amin(state1[x, 0, 0, :].real- state2[x, 0, 0, :].real)
		if np.amin(state1[x, 0, 0, :].imag- state2[x, 0, 0, :].imag) < min_diff_imag:
			min_diff_imag = np.amin(state1[x, 0, 0, :].imag- state2[x, 0, 0, :].imag)
		print("difference real: = ", np.amax(state1[x, 0, 0, :].real- state2[x, 0, 0, :].real))
		print("difference imag: = ", np.amax(state1[x, 0, 0, :].imag- state2[x, 0, 0, :].imag))
	print ('max_diff_real  = ', max_diff_real)
	print ("max_diff_imag  = ", max_diff_imag)
	print ("min_diff_real  = ", min_diff_real)
	print ("min_diff_imag  = ", min_diff_imag)

compare(directory, directory2)

