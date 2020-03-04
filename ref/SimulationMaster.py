from __future__ import print_function 
import setup as QLGA
import os
from types import ModuleType
import subprocess


######################################################################################
############################# SIMULATION PARAMS ######################################
######################################################################################
#### LATTICE INFO ####
# For particles >=2, (x,y,z) represents indices of number basis kets along each direction, ie. if xDim=L then XBLOCK*XGRID = L(2*L-1)
PARTICLES = 1
KINETIC_OPERATOR = 'S'   # S for Schroedinger equation, D for Dirac ... 
FRAME_SIZE = 20
NUM_FRAMES = 30


Lx = 512 # Parameter to scale lattice size along x
Ly = 512 # Parameter to scale lattice size along y
Lz = 1 # Parameter to scale lattice size along z


BATCH = 'HPC_Testing_Phase_Initial_First_Time_Test_RUN'
RUN ='charlie_alpha_beta_test'




DEVICES = [0] # References to devices to use


######################################################################################
########################## SIMULATION PARAMS END #####################################
######################################################################################


############################# INITIAL CONDITIONS #####################################


# INIT = "gaussians_2P_1D"
# INIT = 'double_quadrupole_3d'
INIT = 'double_quadrupole_3d'
# INIT = 'double_quadrupole'
# INIT = 'pade_quadrupole'


############################ SIMULATION PHYSICS MODEL ################################

MODEL = 'spin2_BEC'
# MODEL = "No_Self_Interaction"
# MODEL = "Internal_Function"



############################ EXPERIMENT KEYWORDS #####################################


#EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 5, "solution1" : 1} 
EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 5, "solution1" : 2}
# EXP_KWARGS = {'momentums': [4.0, 0.0, 4.0, 0.0], 'shifts': [0.3, 0.7, 0.4, 0.8], 'sigmas': [0.025, 0.025, 0.25, 0.25]}
### 'cond_list': ['X1==X2', 'true'], 'func_list': ['0.', '(4.*pi)*(1./abs(X1-X2))'], coulomb

###########################  EXTERNAL POTENTIAL #####################################


POTENTIAL = "No_Potential"
POTENTIAL_KWARGS = {}
# POTENTIAL = "External_Function"
# POTENTIAL_KWARGS = {"cond_list":["X1<Lx/10. && X2<Lx/10.", "(X1<Lx/10. && X2>Lx/10.) || (X1>Lx/10. && X2<Lx/10.)", "true"],"func_list" : ["(pi/50.)","(pi/100.)","0."]}
# double trap POTENTIAL_KWARGS = {"cond_list":["X1<Lx/2. && X2<Lx/2.","X1>=Lx/2. && X2>=Lx/2.","X1<Lx/2. && X2>=Lx/2.", "X1>=Lx/2. && X2<Lx/2."],"func_list" : ["((pi/12.)/(Lx*Lx))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))","((pi/12.)/(Lx*Lx))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))","((pi/12.)/(Lx*Lx))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))","((pi/12.)/(Lx*Lx))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))"]} 
# single trap POTENTIAL_KWARGS = {"cond_list":["true"],"func_list" : ["((pi/40.)/(Lx*Lx))*((X1-Lx/2.)*(X1-Lx/2.)+(X2-Lx/2.)*(X2-Lx/2.))"]}

###########################  MEASUREMENT #####################################


MEASUREMENT = "No_Measurement"
MEASUREMENT_KWARGS = {}
# MEASUREMENT = "Measurement_1D"
# MEASUREMENT_KWARGS = {'timestep' : 1150, 'position' : 90, 'width' : 10, 'smooth' : True, 'Measured' : True}



########################   VISUALIZATION TECHNIQUE #################################

# VISUALIZATION = '1D_1P'
# VISUALIZATION = '1D_2P'
# VISUALIZATION = 'mayavi_2d_surface'
# VISUALIZATION = 'mayavi_3d_isosurface' 
VISUALIZATION = 'total_density_isosurface'
# VISUALIZATION = 'colored_mf_isosurface'


VIS_KWARGS = {"fps":6, "contour_percent" : [.001]}

RUN_TYPE = "runNoVis" # 'run' or 'vis'


meta_data = QLGA.setup(PARTICLES, KINETIC_OPERATOR, FRAME_SIZE, NUM_FRAMES, Lx,  Ly, Lz, 
								BATCH, RUN, DEVICES, INIT, MODEL, EXP_KWARGS, POTENTIAL, POTENTIAL_KWARGS,
								VISUALIZATION, VIS_KWARGS, MEASUREMENT, MEASUREMENT_KWARGS,
								RUN_TYPE, OVERWRITE = True)


def execute(cmd):
    popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1)
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line 
    popen.stdout.close()
    return_code = popen.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, cmd)

for path in execute(["python", "AbstractionLayer.py", RUN_TYPE, meta_data]):
    print(path, end="")

# #subprocess.call("python AbstractionLayer.py " + RUN_TYPE + " " + meta_data, shell = True, stdout=subprocess.PIPE)
# #os.system("python AbstractionLayer.py " + RUN_TYPE + " " + meta_data)

# for i in xrange(1, 5):
# 	RUN ='test' + str(i)
# 	EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 5 * (i), "solution1" : 1,"solution2" : 1, "orientation2" : "x", "y_shift2" : 1./8., "y_shift1" :-1./8.} 
# 	meta_data = QLGA.setup(PARTICLES, KINETIC_OPERATOR, FRAME_SIZE, NUM_FRAMES, Lx,  Ly, Lz, 
# 								BATCH, RUN, DEVICES, INIT, MODEL, EXP_KWARGS, POTENTIAL, POTENTIAL_KWARGS,
# 								VISUALIZATION, VIS_KWARGS, MEASUREMENT, MEASUREMENT_KWARGS,
# 								RUN_TYPE, OVERWRITE = True)
# 	os.system("python AbstractionLayer.py " + RUN_TYPE + " " + meta_data)

