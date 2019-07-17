import numpy as np 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap
vortex_file_root_dir = '/media/qcgroup/HD3/PycharmProjects/QLGA-Simulator/Experiments/quasar/SimVersion_2.0/spin2_BEC_2D/for_vortex_tracks/double_quadrupole/p1x_10_solution_number_2_2_solution_number_1_1/XSize_512_YSize_512_ZSize_1/G0_1.0_G1_0.1_G2_1.0_MU_1.0_Scaling_50.0_Step_700/Extra/'
vortex_file_stub = 'vorticity'
vortex_file = vortex_file_root_dir + vortex_file_stub + '.npy'



cdict1 = {'red':   ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 0.0, 0.0))}



cdict2 = {'red':   ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 1.0, 1.0))}

cdict3 = {'red':   ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 0.0, 0.0))}

cdict4 = {'red':   ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 1.0, 1.0))}
         
cdict5 = {'red':   ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0,  1.0, 1.0),
                   (0.5,  0.0, 0.0),
                   (1.0, 1.0, 1.0))}



cdict1a = cdict1.copy()
cdict2a = cdict2.copy()
cdict3a = cdict3.copy()
cdict4a = cdict4.copy()
cdict5a = cdict5.copy()
cdict1a['alpha'] = ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0))

cdict2a['alpha'] = ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0))

cdict3a['alpha'] = ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0))

cdict4a['alpha'] = ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0))

cdict5a['alpha'] = ((0.0, 0.0, 0.0),
                   (0.5, 0.0, 0.0),
                   (0.75, 1.0, 1.0),
                   (1.0, 1.0, 1.0))






trial1 = LinearSegmentedColormap('trial', cdict1a)
trial2 = LinearSegmentedColormap('trial', cdict2a)
trial3 = LinearSegmentedColormap('trial', cdict3a)
trial4 = LinearSegmentedColormap('trial', cdict4a)
trial5 = LinearSegmentedColormap('trial', cdict5a)

def make_vortex_tracks(file, mf):
  vortices = np.load(file)
  if np.amax(vortices[:, : , 0, 4-2*mf].T) > .5:
    plt.imshow(vortices[:, : , 0, 4-2*mf].T, cmap = trial1, vmin = 0, vmax = .5)
    plt.imshow(vortices[:, : , 0, 5-2*mf].T, cmap = trial2, vmin = 0, vmax = .5)
    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False, 
        left=False,
        right = False,     # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False,
        labelleft = False)
        # labels along the bottom edge are off
    plt.gca().set_facecolor((1,1,1))
    plt.gcf().savefig(vortex_file_root_dir + vortex_file_stub + file.split('Step_')[1].split('/')[0] + str(mf) + ".png")
    plt.clf()



# timesteps = [20]
# files = []
# for t in timesteps:
#   a = vortex_file.split('Step_')
#   new_file = a[0] + 'Step_' + str(t) + '/' + a[1].split('/')[1] + '/' + a[1].split('/')[2]
#   files.append(new_file)
# print(files)

# for f in files:
#   for m in xrange(-2, 3):
#     make_vortex_tracks(f, m)



for m in xrange(-2, 3):
    make_vortex_tracks(vortex_file, m)