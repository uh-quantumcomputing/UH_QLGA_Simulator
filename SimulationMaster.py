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

FRAME_SIZE = 100
NUM_FRAMES = 10


Lx = 4096 # Parameter to scale lattice size along x
Ly = 16 # Parameter to scale lattice size along y
Lz = 1 # Parameter to scale lattice size along z


BATCH = 'fermi_condensate_set'
RUN ='1d_soliton_type1'





DEVICES = [0,1] # References to devices to use


######################################################################################
########################## SIMULATION PARAMS END #####################################
######################################################################################


############################# INITIAL CONDITIONS #####################################


# INIT = 'function_1P'
# INIT = 'gaussians_1P_1D'
# INIT = "gaussians_2P_1D"
# INIT = 'double_quadrupole_3d'
INIT = 'spinHalf_cross_solitons'


############################ SIMULATION PHYSICS MODEL ################################


# MODEL = 'spin2_BEC'
# MODEL = "No_Self_Interaction"
# MODEL = "Internal_Function"
# MODEL = 'scalar_BEC'
MODEL = 'spinHalf_FC'



############################ EXPERIMENT KEYWORDS #####################################


EXP_KWARGS = {'dim2':0., 'scaling':3., 'coeffs':[0.,2.j]}
# EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 5, "solution1" : 5, "orientation1" : "x"} 
# EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 25, "solution2" : 2, "orientation2" : "x", "y_shift2" : 1./4.} 
# GS harmonic 
c = 500.*Lx
sigma = (((c/(3.141592653589793238462643383279502884197)))**(1./4.))/float(Lx) 
# print sigma
# quit()
# EXP_KWARGS = {'func': '100.*exp(-(X-Lx/5.)*(X-Lx/5.)*(X-Lx/5.)*(X-Lx/5.)/((Lx/10.)*(Lx/10.)*(Lx/10.)*(Lx/10.)))', 'px':'-20.'}
# EXP_KWARGS = {'momentums': [4.0, -4.0], 'shifts': [0.25, 0.75], 'sigmas': [sigma,sigma]}

###########################  EXTERNAL POTENTIAL #####################################

POTENTIAL = "No_Potential"
POTENTIAL_KWARGS = {}
# POTENTIAL = "External_Function"
# POTENTIAL_KWARGS = {"cond_list":["X1<Lx/7. && X2<Lx/7.", "(X1<Lx/7. && X2>Lx/7.) || (X1>Lx/7. && X2<Lx/7.)", "true"],"func_list" : ["(pi/10.)","(pi/20.)","0."]}
# double trap 
# POTENTIAL_KWARGS = {'func_list': ['((pi/'+ str(c)+'))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))', '((pi/'+ str(c)+'))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))', '((pi/'+ str(c)+'))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))', '((pi/'+ str(c)+'))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))'], 'cond_list': ['X1<Lx/2. && X2<Lx/2.', 'X1>=Lx/2. && X2>=Lx/2.', 'X1<Lx/2. && X2>=Lx/2.', 'X1>=Lx/2. && X2<Lx/2.']}
# double trap oscillation
# POTENTIAL_KWARGS = {"cond_list":["X1<Lx/2. && X2<Lx/2.","X1>=Lx/2. && X2>=Lx/2.","X1<Lx/2. && X2>=Lx/2.", "X1>=Lx/2. && X2<Lx/2."],"func_list" : ["((pi/"+ str(c)+")*(1.-0.2*cos(T/325.)))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))","((pi/"+ str(c)+"))*((X1-Lx/4.)*(X1-Lx/4.)*(1.-0.2*cos(T/325.))+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.)*(1.-0.2*cos(T/325.)))"]} 
# single trap 
# POTENTIAL_KWARGS = {"cond_list":["true"],"func_list" : ["((pi/"+ str(c)+"))*((X1-Lx/2.)*(X1-Lx/2.)+(X2-Lx/2.)*(X2-Lx/2.))"]}
# coulomb
# POTENTIAL_KWARGS = {'cond_list': ['X1==X2', 'true'], 'func_list': ['0.', '(4.*pi)*(1./abs(X1-X2))']}
# double trap plus quadratic particle interaction
# POTENTIAL_KWARGS = {"cond_list":["X1<Lx/2. && X2<Lx/2.","X1>=Lx/2. && X2>=Lx/2.","X1<Lx/2. && X2>=Lx/2.", "X1>=Lx/2. && X2<Lx/2."],"func_list" : ["((pi/"+ str(c)+"))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))+((pi/"+ str(10.*c)+"))*(X1-X2)*(X1-X2)","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))+((pi/"+ str(10.*c)+"))*(X1-X2)*(X1-X2)","((pi/"+ str(c)+"))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))+((pi/"+ str(10.*c)+"))*(X1-X2)*(X1-X2)","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))+((pi/"+ str(10.*c)+"))*(X1-X2)*(X1-X2)"]} 
# barrier 1p
# POTENTIAL_KWARGS = {'cond_list': ['(X>=50.*Lx/100.) && (X<=51.*Lx/100.)'], 'func_list': ['pi/400.']}
# single slit 1p
# POTENTIAL_KWARGS = {'func_list': ['pi/700.'], 'cond_list': ['((X>=2.*Lx/5.-Lx/70.) && (X<=2.*Lx/5.+Lx/70.) && (Y>=((Ly-1)/2.+Ly/70.+Ly/12.) || Y<=((Ly-1)/2.-Ly/70.+Ly/12.)) ) || (X<=Lx/50. || X>49.*Lx/50.)']}
# double slit 1p
# POTENTIAL_KWARGS = {'func_list': ['pi/700.'], 'cond_list': ['( (X>=2.*Lx/5.-Lx/70.) && (X<=2.*Lx/5.+Lx/70.) && ( Y<=((Ly-1)/2.-Ly/70.-Ly/12.) || ( Y>=((Ly-1)/2.+Ly/70.-Ly/12.) && Y<=((Ly-1)/2.-Ly/70.+Ly/12.) ) || Y>=((Ly-1)/2.+Ly/70.+Ly/12.) ) ) || (X<=Lx/50. || X>49.*Lx/50.)']}


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
# VISUALIZATION = 'mayavi_2d_surface_1comp'
# VISUALIZATION = 'mayavi_2d_surface'
# VISUALIZATION = 'mayavi_3d_isosurface' 
# VISUALIZATION = 'total_density_isosurface'
# VISUALIZATION = 'colored_mf_isosurface'


VIS_KWARGS = {"fps":6, 'vid_fmt':'mp4'}

RUN_TYPE = "vis" # 'run' or 'vis'


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

