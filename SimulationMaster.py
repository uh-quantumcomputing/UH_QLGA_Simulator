#import Code.Simulations.SimulationHelper as sim
import AbstractionLayer as QLGA

######################################################################################
############################# SIMULATION PARAMS ######################################
######################################################################################

#### LATTICE INFO ####
# For particles >=2, (x,y,z) represents indices of number basis kets along each direction, ie. if xDim=L then XBLOCK*XGRID = L(2*L-1)
PARTICLES = 2
KINETIC_OPERATOR = 'S'   # S for Schroedinger equation, D for Dirac ... 
FRAME_SIZE = 25
NUM_FRAMES = 12


Lx = 1024 # Parameter to scale lattice size along x
Ly = 1 # Parameter to scale lattice size along y
Lz = 1 # Parameter to scale lattice size along z


BATCH = '2p_paper_set'
RUN ='Emax_free_for_barrier_collision_compare'




DEVICES = [1,4] # References to devices to use


######################################################################################
########################## SIMULATION PARAMS END #####################################
######################################################################################


############################# INITIAL CONDITIONS #####################################


INIT = "gaussians_2P_1D"
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
# # quit()
# EXP_KWARGS = {'momentums': [0.0, 0.0, 0.0, 0.0], 'shifts': [0.18, 0.68, 0.32, 0.82], 'sigmas': [sigma, sigma, sigma, sigma]}
EXP_KWARGS = {'momentums': [32.0, 0.0, 32.0, 0.0], 'shifts': [0.28, 0.68, 0.42, 0.82], 'sigmas': [0.015, 0.015, 0.015, 0.015]}

###########################  EXTERNAL POTENTIAL #####################################


POTENTIAL = "No_Potential"
POTENTIAL_KWARGS = {}
# POTENTIAL = "External_Function"
# POTENTIAL_KWARGS = {"cond_list":["X1<Lx/7. && X2<Lx/7.", "(X1<Lx/7. && X2>Lx/7.) || (X1>Lx/7. && X2<Lx/7.)", "true"],"func_list" : ["(pi/50.)","(pi/100.)","0."]}
#double trap 
# POTENTIAL_KWARGS = {"cond_list":["X1<Lx/2. && X2<Lx/2.","X1>=Lx/2. && X2>=Lx/2.","X1<Lx/2. && X2>=Lx/2.", "X1>=Lx/2. && X2<Lx/2."],"func_list" : ["((pi/"+ str(c)+"))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))","((pi/"+ str(c)+"))*((X1-Lx/4.)*(X1-Lx/4.)+(X2-3.*Lx/4.)*(X2-3.*Lx/4.))","((pi/"+ str(c)+"))*((X1-3.*Lx/4.)*(X1-3.*Lx/4.)+(X2-Lx/4.)*(X2-Lx/4.))"]} 
# single trap 
# POTENTIAL_KWARGS = {"cond_list":["true"],"func_list" : ["((pi/"+ str(c)+"))*((X1-Lx/2.)*(X1-Lx/2.)+(X2-Lx/2.)*(X2-Lx/2.))"]}
# coulomb
# POTENTIAL_KWARGS = {'cond_list': ['X1==X2', 'true'], 'func_list': ['0.', '(4.*pi)*(1./abs(X1-X2))']}

###########################  MEASUREMENT #####################################


MEASUREMENT = "No_Measurement"
MEASUREMENT_KWARGS = {}
# MEASUREMENT = "Measurement_1D"
# MEASUREMENT_KWARGS = {'timestep' : 50, 'position' : 100, 'width' : 10, 'smooth' : True, 'Measured' : True}



########################   VISUALIZATION TECHNIQUE #################################

# VISUALIZATION = '1D_1P'
VISUALIZATION = '1D_2P'
# VISUALIZATION = 'mayavi_2d_surface'
# VISUALIZATION = 'mayavi_3d_isosurface_full' 
# VISUALIZATION = 'total_density_isosurface'


VIS_KWARGS = {"fps":6, 'vid_fmt':'mp4'}

#####################################SETUP#######################################
# ALWAYS RUN THIS
QLGA.setup(PARTICLES, KINETIC_OPERATOR, FRAME_SIZE, NUM_FRAMES, Lx,  Ly, Lz, 
			BATCH, RUN, DEVICES, INIT, MODEL, EXP_KWARGS, POTENTIAL, POTENTIAL_KWARGS,
			VISUALIZATION, VIS_KWARGS, MEASUREMENT, MEASUREMENT_KWARGS)

################################# EXPERIMENT  ########################################
QLGA.run()
# QLGA.run_no_vis()
# QLGA.vis()
# QLGA.ani()
# QLGA.vis_ani()
# QLGA.resume()
