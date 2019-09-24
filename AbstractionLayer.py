import Code.Simulations.SimulationHelper as sim
import sys
run_type = sys.argv[1]
meta_data = ''
for i in xrange(2, len(sys.argv)):
	meta_data = meta_data + sys.argv[i] + ' '
meta_data = meta_data[:-1]
############################################################################
sim.load_global_vars(meta_data)

def run():
	sim.init_GPU()
	sim.simulate()
	sim.visualize()
	sim.animate()

def vis():
	sim.visualize()
	sim.animate()


locals()[run_type]()

# ### Not working yet
# def resume(file_name = '', **kwargs):
# 	sim.init_GPU()
# 	sim.simulate()
# 	sim.visualize(VISUALIZATION, **VIS_KWARGS)
# 	sim.animate(VISUALIZATION, **VIS_KWARGS)

