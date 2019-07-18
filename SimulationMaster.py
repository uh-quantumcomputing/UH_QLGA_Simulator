#import Code.Simulations.SimulationHelper as sim
import AbstractionLayer as QLGA

######################################################################################
############################# SIMULATION PARAMS ######################################
######################################################################################

#### LATTICE INFO ####
# For particles >=2, (x,y,z) represents indices of number basis kets along each direction, ie. if xDim=L then XBLOCK*XGRID = L(2*L-1)
PARTICLES = 1
KINETIC_OPERATOR = 'S'   # S for schroedinger equation, D for dirac ... 
FRAME_SIZE = 10
NUM_FRAMES = 10


Lx = 256 # Parameter to scale lattice size along x
Ly = 256 # Parameter to scale lattice size along y
Lz = 256 # Parameter to scale lattice size along z

BATCH = 'z_dir_test'
RUN ='run2'


DEVICES = [0,1] # References to devices to use

######################################################################################
########################## SIMULATION PARAMS END #####################################
######################################################################################


############################# INITIAL CONDITIONS #####################################


#INIT = "entangled_gaussian_2P_1D"
INIT = 'double_quadrupole_3d'
#INIT = 'double_quadrupole'
#INIT = 'pade_quadrupole'


############################ SIMULATION PHYSICS MODEL ################################

MODEL = 'spin2_BEC'
#MODEL = "No_Self_Interaction"



############################ EXPERIMENT KEYWORDS #####################################


# EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 5, "solution1" : 5, "orientation1" : "x"} 
EXP_KWARGS = {'G0' : 1., 'G1' : .1, 'G2': 1., 'MU':1.,  'scaling' : 25, "solution2" : 2, "orientation2" : "x", "y_shift2" : 1./4.} 
# EXP_KWARGS = {'sigma1': 0.1, 'sigma2': 0.1, 'shift1': 0.35, 'shift2': 0.65,'sigma3': 0.1, 'sigma4': 0.1, 'shift3': 0.35, 'shift4': 0.65}


###########################  EXTERNAL POTENTIAL #####################################


POTENTIAL = "No_Potential"
POTENTIAL_KWARGS = {}

###########################  MEASUREMENT #####################################


MEASUREMENT = "No_Measurement"
MEASUREMENT_KWARGS = {}
# MEASUREMENT = "Measurement"
# MEASUREMENT_KWARGS = {'timestep' : 10, 'position' : 120, 'width' : 0, 'func' : "1./cosh", 'smooth' : False, 'Measured' : True}



########################   VISUALIZATION TECHNIQUE #################################

# VISUALIZATION = '1D_2P'
# VISUALIZATION = 'mayavi_2d_surface'
VISUALIZATION = 'mayavi_3d_isosurface' 


VIS_KWARGS = {"fps":3}

#####################################SETUP#######################################
# ALWAYS RUN THIS
QLGA.setup(PARTICLES, KINETIC_OPERATOR, FRAME_SIZE, NUM_FRAMES, Lx,  Ly, Lz, 
			BATCH, RUN, DEVICES, INIT, MODEL, EXP_KWARGS, POTENTIAL, POTENTIAL_KWARGS,
			VISUALIZATION, VIS_KWARGS, MEASUREMENT, MEASUREMENT_KWARGS)

################################# EXPERIMENT  ########################################
# QLGA.run()
# QLGA.run_no_vis()
# QLGA.vis()
QLGA.ani()
# QLGA.vis_ani()
# QLGA.resume()