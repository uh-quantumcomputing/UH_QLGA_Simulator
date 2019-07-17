import numpy as np
import matplotlib
import matplotlib.pyplot as plt

xSize = 2048
ySize = 2048
qs_ds = np.load('/media/qcg/HDD1/Synched_v1/Synched/Experiments/QSimSLI/SimVersion_2.0/double_slit_spin2/slit_width_60_wall_width_60_spacing_360/stationary_states/state1_12_x1_0.2_p1_40.0/XSize_2048_YSize_2048_ZSize_1/G0_-1.0_G1_-1.0_G2_2.0_MU_-1.0_Scaling_30_Step_250/Data/Frame_00004250.npy')
# qs_ss = np.load('/media/qcg/HDD1/Synched_v1/Synched/Experiments/QSimSLI/SimVersion_2.0/double_slit_spin2/slit_width_60_wall_width_60_spacing_0/stationary_states/state1_1_x1_0.2_p1_20.0/XSize_2048_YSize_2048_ZSize_1/G0_-1.0_G1_-1.0_G2_-1.0_MU_-1.0_Scaling_30_Step_250/Data/Frame_00008500.npy')


def plotMF(mf):
	psi_squared_ds = qs_ds[int(2.*xSize/3.),:,:,2*mf]*qs_ds[int(2.*xSize/3.),:,:,2*mf].conjugate()
	# psi_squared_ss = qs_ss[int(2.*xSize/3.),:,:,2*mf]*qs_ss[int(2.*xSize/3.),:,:,2*mf].conjugate()
	# psi_amplitudeAdd = np.roll(qs_ss[int(2.*xSize/3.),:,:,2*mf],180)+np.roll(qs_ss[int(2.*xSize/3.),:,:,2*mf],-180)
	# psi_amplitudeAdd_squared = psi_amplitudeAdd*(psi_amplitudeAdd.conjugate())
	# psi_densityAdd_r = np.roll(psi_squared_ss,180)#+np.roll(psi_squared_ss,-180)
	# psi_densityAdd_l = np.roll(psi_squared_ss,-180)
	ran = xrange(ySize/2+1)
	fig = plt.figure(figsize=(12,8))
	## Amplitude adding
	plt.plot(xrange(ySize), psi_squared_ds[:, 0],'k', linewidth=2.0)
	# plt.plot(xrange(ySize), psi_amplitudeAdd_squared[:, 0],"--", color="#c6770d", dashes=(8, 2),linewidth=2.0)
	# plt.plot(xrange(ySize), psi_amplitudeAdd_squared[:, 0], color="#c6770d",linewidth=2.0)
	#Density adding
	# plt.plot(xrange(ySize), psi_densityAdd_r[:, 0], linewidth=2.0)
	# plt.plot(xrange(ySize), psi_densityAdd_l[:, 0], linewidth=2.0)
	# plt.plot(xrange(ySize), psi_densityAdd_r[:, 0]+psi_densityAdd_l[:, 0], linewidth=2.0)
	# plt.plot(xrange(ySize), psi_amplitudeAdd_squared[:, 0], linewidth=2.0)
	# plt.plot(xrange(ySize), 0.021+0.02*np.cos(2.*np.pi*(np.arange(ySize)-ySize/2.)/237.))
	plt.xlabel(r'$y(\ell)$',fontsize=24)
	plt.ylabel(r'${| \Psi |}^{2}$',fontsize=24)
	# plt.gcf().gca().set_xticklabels([0,0,500,1000,1500,2000],fontsize=16)
	# plt.gcf().gca().set_yticklabels([0.00,0.00,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08],fontsize=16)
	# plt.gcf().gca().set_yticklabels([0.00,0.004,0.005,0.010,0.015,0.020,0.025,0.030,0.035,0.040],fontsize=16)
	# plt.gcf().gca().set_yticklabels([0.000,0.000,0.005,0.010,0.015,0.020],fontsize=16)
	# plt.legend(fontsize=16)
	# plt.savefig('/home/qcg/Desktop/ss_density_add_v_ds_spin2.png')
	plt.show()
	# return psi_squared_other

c = plotMF(3)

print np.roll(np.arange(10),-2)

