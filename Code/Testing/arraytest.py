import numpy as np 

a = np.load('/media/qcgroup/HD3/PycharmProjects/QLGA-Simulator/Experiments/quasar/SimVersion_2.0/spin2_BEC_3D/three_D/pade_quadrupole/default/XSize_64_YSize_64_ZSize_64/G0_1.0_G1_0.1_G2_1.0_MU_1.0_Scaling_50.0_Step_5/Data/Frame_00000000.npy')

print(a[10, 12, 0, 0])
print(a[10, 12, 50, 0])