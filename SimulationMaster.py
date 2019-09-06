import setup as QLGA
import os
from types import ModuleType


######################################################################################
############################# SIMULATION PARAMS ######################################
######################################################################################
#### LATTICE INFO ####
# For particles >=2, (x,y,z) represents indices of number basis kets along each direction, ie. if xDim=L then XBLOCK*XGRID = L(2*L-1)
PARTICLES = 1
KINETIC_OPERATOR = 'S'   # S for Schroedinger equation, D for Dirac ... 
FRAME_SIZE = 100
NUM_FRAMES = 10


Lx = 512 # Parameter to scale lattice size along x
Ly = 512 # Parameter to scale lattice size along y
Lz = 1 # Parameter to scale lattice size along z


BATCH = 'implementation_paper_set'
RUN ='double_slit_init_test'


DEVICES = [1,4] # References to devices to use


######################################################################################
########################## SIMULATION PARAMS END #####################################
######################################################################################


############################# INITIAL CONDITIONS #####################################

INIT = 'function_1P'
# INIT = 'gaussians_1P_1D'
# INIT = "gaussians_2P_1D"
# INIT = 'double_quadrupole_3d'

#INIT = 'double_quadrupole'
#INIT = 'pade_quadrupole'


############################ SIMULATION PHYSICS MODEL ################################

# MODEL = 'spin2_BEC'
MODEL = "No_Self_Interaction"
# MODEL = "Internal_Function"



############################ EXPERIMENT KEYWORDS #####################################


# EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 5, "solution1" : 5, "orientation1" : "x"} 
# EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 25, "solution2" : 2, "orientation2" : "x", "y_shift2" : 1./4.} 
# GS harmonic 
# c = 500.*Lx
# sigma = (((c/(3.141592653589793238462643383279502884197)))**(1./4.))/float(Lx)
# print sigma
# quit()
EXP_KWARGS = {'func': 'exp(-(X-Lx/4.)*(X-Lx/4.)*(X-Lx/4.)*(X-Lx/4.)/((Lx/20.)*(Lx/20.)*(Lx/20.)*(Lx/20.)))'}
# EXP_KWARGS = {'momentums': [32.0, 0.0, 32.0, 0.0], 'shifts': [0.28, 0.68, 0.42, 0.82], 'sigmas': [0.015, 0.015, 0.015, 0.015]}

###########################  EXTERNAL POTENTIAL #####################################

POTENTIAL = "No_Potential"
POTENTIAL_KWARGS = {}
# POTENTIAL = "External_Function"
# POTENTIAL_KWARGS = {"cond_list":["X1<Lx/7. && X2<Lx/7.", "(X1<Lx/7. && X2>Lx/7.) || (X1>Lx/7. && X2<Lx/7.)", "true"],"func_list" : ["(pi/10.)","(pi/20.)","0."]}
# double trap 
# POTENTIAL_KWARGS = {"cond_list":["X1<Lx/2. && X2<Lx/2.","X1>=Lx/2. && X2>=Lx/2.","X1<Lx/2. && X2>=Lx/2.", "X1>=Lx/2. && X2<Lx/2."],"func_list" : ["((pi/"+ str(c)+"))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))","((pi/"+ str(c)+"))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))"]} 
# single trap 
# POTENTIAL_KWARGS = {"cond_list":["true"],"func_list" : ["((pi/"+ str(c)+"))*((X1-Lx/2.)*(X1-Lx/2.)+(X2-Lx/2.)*(X2-Lx/2.))"]}
# coulomb
# POTENTIAL_KWARGS = {'cond_list': ['X1==X2', 'true'], 'func_list': ['0.', '(4.*pi)*(1./abs(X1-X2))']}
# double trap plus quadratic particle interaction
# POTENTIAL_KWARGS = {"cond_list":["X1<Lx/2. && X2<Lx/2.","X1>=Lx/2. && X2>=Lx/2.","X1<Lx/2. && X2>=Lx/2.", "X1>=Lx/2. && X2<Lx/2."],"func_list" : ["((pi/"+ str(c)+"))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))+((pi/"+ str(10.*c)+"))*(X1-X2)*(X1-X2)","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))+((pi/"+ str(10.*c)+"))*(X1-X2)*(X1-X2)","((pi/"+ str(c)+"))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))+((pi/"+ str(10.*c)+"))*(X1-X2)*(X1-X2)","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))+((pi/"+ str(10.*c)+"))*(X1-X2)*(X1-X2)"]} 
# barrier 1p
# POTENTIAL_KWARGS = {'cond_list': ['(X>=50.*Lx/100.) && (X<=51.*Lx/100.)'], 'func_list': ['pi/400.']}

# EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 25, "solution1" : 1,"solution2" : 3, "orientation2" : "x", "y_shift2" : 1./4.} 
# EXP_KWARGS = {'momentums': [4.0, 0.0, 4.0, 0.0], 'shifts': [0.3, 0.7, 0.4, 0.8], 'sigmas': [0.025, 0.025, 0.25, 0.25]}
### 'cond_list': ['X1==X2', 'true'], 'func_list': ['0.', '(4.*pi)*(1./abs(X1-X2))'], coulomb

###########################  MEASUREMENT #####################################


MEASUREMENT = "No_Measurement"
MEASUREMENT_KWARGS = {}
# MEASUREMENT = "Measurement_1D"
# MEASUREMENT_KWARGS = {'timestep' : 1150, 'position' : 90, 'width' : 10, 'smooth' : True, 'Measured' : True}



########################   VISUALIZATION TECHNIQUE #################################

# VISUALIZATION = '1D_1P'
# VISUALIZATION = '1D_2P'
VISUALIZATION = '2D_Density_Phase_1P'
# VISUALIZATION = '1D_2P_density'
# VISUALIZATION = 'mayavi_2d_surface'
# VISUALIZATION = 'mayavi_3d_isosurface' 
# VISUALIZATION = 'total_density_isosurface'
# VISUALIZATION = 'colored_mf_isosurface'



RUN_TYPE = 'vis' # 'run' or 'vis'


VIS_KWARGS = {"fps":6, 'vid_fmt':'mp4'}


#### LOOP #####
# for i in xrange(1, 2):
# 	for j in xrange(i, 2):
# 		RUN ='sols ' + str(i) + ', ' + str(j)
# 		EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 25, "solution1" : i,"solution2" : j, "orientation2" : "x", "y_shift2" : 1./4.} 
# 		meta_data = QLGA.setup(PARTICLES, KINETIC_OPERATOR, FRAME_SIZE, NUM_FRAMES, Lx,  Ly, Lz, 
# 								BATCH, RUN, DEVICES, INIT, MODEL, EXP_KWARGS, POTENTIAL, POTENTIAL_KWARGS,
# 								VISUALIZATION, VIS_KWARGS, MEASUREMENT, MEASUREMENT_KWARGS,
# 								RUN_TYPE, OVERWRITE = True)
# 		os.system("python AbstractionLayer.py " + RUN_TYPE + " " + meta_data)


################################# EXPERIMENT  ########################################
meta_data = QLGA.setup(PARTICLES, KINETIC_OPERATOR, FRAME_SIZE, NUM_FRAMES, Lx,  Ly, Lz, 
						BATCH, RUN, DEVICES, INIT, MODEL, EXP_KWARGS, POTENTIAL, POTENTIAL_KWARGS,
						VISUALIZATION, VIS_KWARGS, MEASUREMENT, MEASUREMENT_KWARGS,
						RUN_TYPE, OVERWRITE = True)
os.system("python AbstractionLayer.py " + RUN_TYPE + " " + meta_data)
