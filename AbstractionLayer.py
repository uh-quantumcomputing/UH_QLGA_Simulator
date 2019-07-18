import Code.Simulations.SimulationHelper as sim

PARTICLES = None
KINETIC_OPERATOR = None
FRAME_SIZE = None
NUM_FRAMES = None
Lx = None  
Ly = None 
Lz = None 
BATCH = None
RUN = None
DEVICES = None
INIT = None
MODEL = None
EXP_KWARGS = None
POTENTIAL = None
POTENTIAL_KWARGS = None
VISUALIZATION = None 
VIS_KWARGS = None
MEASUREMENT = None
MEASUREMENT_KWARGS = None

def setup(particles, kinetic_operator, frame_size, num_frames, lx,  ly, lz, 
			batch, run, devices, init, model, exp_kwargs, potential, potential_kwargs,
			visualization, vis_kwargs, measurement, measurement_kwargs):
	global PARTICLES, KINETIC_OPERATOR, FRAME_SIZE, NUM_FRAMES, Lx,  Ly, Lz 
	global BATCH, RUN, DEVICES, INIT, MODEL, EXP_KWARGS, POTENTIAL, POTENTIAL_KWARGS
	global VISUALIZATION, VIS_KWARGS, MEASUREMENT, MEASUREMENT_KWARGS
	PARTICLES, KINETIC_OPERATOR, FRAME_SIZE, NUM_FRAMES, Lx,  Ly, Lz = particles, kinetic_operator, frame_size, num_frames, lx,  ly, lz 
	BATCH, RUN, DEVICES, INIT, MODEL, EXP_KWARGS, POTENTIAL, POTENTIAL_KWARGS = batch, run, devices, init, model, exp_kwargs, potential, potential_kwargs
	VISUALIZATION, VIS_KWARGS, MEASUREMENT, MEASUREMENT_KWARGS = visualization, vis_kwargs, measurement, measurement_kwargs
############################################################################

def run(**kwargs):
	sim.set_size(Lx, Ly, Lz, PARTICLES, DEVICES) 
	sim.set_experiment(MODEL, KINETIC_OPERATOR, INIT, POTENTIAL, MEASUREMENT, EXP_KWARGS, POTENTIAL_KWARGS, MEASUREMENT_KWARGS)
	sim.set_runtime(FRAME_SIZE, NUM_FRAMES)
	sim.set_directory(BATCH, RUN, VISUALIZATION, **kwargs)
	sim.init_GPU()
	sim.simulate()
	sim.visualize(VISUALIZATION, **VIS_KWARGS)
	sim.animate(VISUALIZATION, **VIS_KWARGS)

def run_no_vis(**kwargs):
	sim.set_size(Lx, Ly, Lz, PARTICLES, DEVICES) 
	sim.set_experiment(MODEL, KINETIC_OPERATOR, INIT, POTENTIAL, MEASUREMENT, EXP_KWARGS, POTENTIAL_KWARGS, MEASUREMENT_KWARGS)
	sim.set_runtime(FRAME_SIZE, NUM_FRAMES)
	sim.set_directory(BATCH, RUN, VISUALIZATION, **kwargs)
	sim.init_GPU()
	sim.simulate()


def vis_ani(file_name = '', **kwargs):
	sim.set_size(Lx, Ly, Lz, PARTICLES, DEVICES) 
	sim.set_experiment(MODEL, KINETIC_OPERATOR, INIT, POTENTIAL, MEASUREMENT, EXP_KWARGS, POTENTIAL_KWARGS, MEASUREMENT_KWARGS)
	sim.set_runtime(FRAME_SIZE, NUM_FRAMES)
	sim.set_directory(BATCH, RUN, VISUALIZATION, VIS_ONLY = True, **kwargs)
	sim.visualize(VISUALIZATION, **VIS_KWARGS)
	sim.animate(VISUALIZATION, **VIS_KWARGS)

def vis(file_name = '', **kwargs):
	sim.set_size(Lx, Ly, Lz, PARTICLES, DEVICES) 
	sim.set_experiment(MODEL, KINETIC_OPERATOR, INIT, POTENTIAL, MEASUREMENT, EXP_KWARGS, POTENTIAL_KWARGS, MEASUREMENT_KWARGS)
	sim.set_runtime(FRAME_SIZE, NUM_FRAMES)
	sim.set_directory(BATCH, RUN, VISUALIZATION, VIS_ONLY = True, **kwargs)
	sim.visualize(VISUALIZATION, **VIS_KWARGS)


def ani(file_name = '', **kwargs):
	sim.set_size(Lx, Ly, Lz, PARTICLES, DEVICES) 
	sim.set_experiment(MODEL, KINETIC_OPERATOR, INIT, POTENTIAL, MEASUREMENT, EXP_KWARGS, POTENTIAL_KWARGS, MEASUREMENT_KWARGS)
	sim.set_runtime(FRAME_SIZE, NUM_FRAMES)
	sim.set_directory(BATCH, RUN, VISUALIZATION, VIS_ONLY = True, ANI_ONLY = True, **kwargs)
	sim.animate(VISUALIZATION, **VIS_KWARGS)

### Not working yet
def resume(file_name = '', **kwargs):
	sim.set_size(Lx, Ly, Lz, PARTICLES, DEVICES) 
	sim.set_experiment(MODEL, KINETIC_OPERATOR, INIT, POTENTIAL, MEASUREMENT, EXP_KWARGS, POTENTIAL_KWARGS, MEASUREMENT_KWARGS)
	sim.set_runtime(FRAME_SIZE, NUM_FRAMES)
	sim.set_directory(BATCH, RUN, VISUALIZATION, **kwargs)
	sim.init_GPU()
	sim.simulate()
	sim.visualize(VISUALIZATION, **VIS_KWARGS)
	sim.animate(VISUALIZATION, **VIS_KWARGS)

