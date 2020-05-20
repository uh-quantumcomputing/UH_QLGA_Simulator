from __future__ import print_function
import setup as QLGA
import os
from types import ModuleType
import subprocess


######################################################################################
############################# SIMULATION PARAMS ######################################
######################################################################################
#### LATTICE INFO ####
# For particles >=2, (x,y,z) represents indices	 of number basis kets along each direction, ie. if xDim=L then XBLOCK*XGRID = L(2*L-1)
PARTICLES = 1
KINETIC_OPERATOR = 'S'   # S for Schroedinger equation, D for Dirac ...
FRAME_SIZE = 10
NUM_FRAMES = 10


Lx = 512# Parameter to scale lattice size along x
Ly = 512# Parameter to scale lattice size along y
Lz = 1# Parameter to scale lattice size along z


BATCH = 'Test'
RUN ='RUN'




DEVICES = [0,1] # References to devices to use


######################################################################################
########################## SIMULATION PARAMS END #####################################
######################################################################################


############################# INITIAL CONDITIONS #####################################


INIT = "Gaussians_1D_1P"

############################ SIMULATION PHYSICS MODEL ################################


MODEL = "Free_1P"




############################ EXPERIMENT KEYWORDS #####################################
EXP_KWARGS = {}

###########################  EXTERNAL POTENTIAL #####################################


POTENTIAL = "No_Potential"
POTENTIAL_KWARGS = {}


###########################  MEASUREMENT #####################################


MEASUREMENT = "No_Measurement"
MEASUREMENT_KWARGS = {}


########################   VISUALIZATION TECHNIQUE #################################


VISUALIZATION = 'Density_Phase_2D_1P'



VIS_KWARGS = {"fps":6}



RUN_TYPE = 'run' # 'run' or 'vis'


meta_data = QLGA.setup(PARTICLES, KINETIC_OPERATOR, FRAME_SIZE, NUM_FRAMES, Lx,  Ly, Lz,
BATCH, RUN, DEVICES, INIT, MODEL, EXP_KWARGS, POTENTIAL, POTENTIAL_KWARGS,
VISUALIZATION, VIS_KWARGS, MEASUREMENT, MEASUREMENT_KWARGS,
RUN_TYPE, OVERWRITE = False, TIME_STEP = 0)



os.system("python AbstractionLayer.py " + RUN_TYPE + " " + meta_data)
