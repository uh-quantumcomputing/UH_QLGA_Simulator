#######################################################
######## Mayavi plotting with PyCUDA speedup ##########
#######################################################
# from traits.etsconfig.api import ETSConfig
# ETSConfig.toolkit = 'qt4'
###### Import Libraries ######
import os
os.environ['CUDA_DEVICE'] = str(1) #Set CUDA device, starting at 0
import pycuda.autoinit
import pycuda.driver as drv
import pylab as pl
import numpy as np
from mayavi import mlab
import moviepy.editor as mpy
import os
import time
import VIS_GPU_2D_SOURCE as gpuVis
from tvtk.api import tvtk

mlab.options.offscreen = True

###### Constants and globals ######
#Set a datatype to use for arrays
DTYPE = np.complex
vectorSize = 10
spinComps = 5

blockX = 16
blockY = 16
blockZ = 1
gridX = 1
gridY = 1
xSize = 0
ySize = 0
zSize = 0
xAxis = None
yAxis = None
A = 0.
E = 0.
scaling = 0.
tau = 0.

QuantumState = None
PhaseField = None
RhoField = None
Lattice = np.zeros(4, dtype = np.int_)
rhoMax = 3.
rhoFrames = 6 #Total frames with rho<Cutoff
mf_index = 0

data_dir = None
image_directory = None
animation_directory = None
frames_list = None

def set_globals(file_name, QuantumState):
	global xSize, ySize, zSize, A, E, scaling, tau, blockX, blockY, blockZ
	global gridX, gridY, xAxis, yAxis, PhaseField, RhoField
	xSize = np.size(QuantumState,0)
	ySize = np.size(QuantumState,1)
	zSize = np.size(QuantumState,2)
	PhaseField = np.zeros((xSize, ySize), dtype = np.float64)
	RhoField = np.zeros((xSize, ySize), dtype = DTYPE)
	block_x = 16
	block_y = 16
	while xSize%block_x != 0:
	  block_x /= 2
	while ySize%block_y != 0:
	  block_y /= 2
	blockX = block_x
	blockY = block_y
	gridX = xSize/blockX
	gridY = ySize/blockY
	xAxis = np.arange(xSize)
	yAxis = np.arange(ySize)
	scaling = float(file_name.split("Scaling_")[1].split("_")[0])
	A = scaling/(xSize)
	tau = A*A

#Mayavi
duration = 11 # duration of the animation in seconds (it will loop)
fps = 6 #frames per second
fig_myv = mlab.figure(size=(1200,1200),bgcolor=(1,1,1),fgcolor=(0.,0.,0.))

######## CUDA Setup ##########
gpuMagic = gpuVis.gpuSource
getPlotDetails = gpuMagic.get_function("getPlotDetailsMayavi")


######## Functions ##########
def setComponent(directory, frame, component = 0):
	global QuantumState, RhoField, PhaseField, Lattice, frames_list
	QuantumState = np.load(directory+frames_list[frame])
	Lattice[0],  Lattice[1], Lattice[2], Lattice[3]= xSize, ySize, zSize, component
	gpuQuantumState = drv.to_device(QuantumState)
	gpuPhaseField = drv.to_device(PhaseField)
	gpuRhoField = drv.to_device(RhoField)
	gpuLattice = drv.to_device(Lattice)
	getPlotDetails(gpuQuantumState, gpuRhoField, gpuPhaseField, gpuLattice, 
                block=(blockX,blockY,blockZ), grid=(gridX,gridY))
	RhoField = drv.from_device(gpuRhoField, RhoField.shape, DTYPE)
	PhaseField = drv.from_device(gpuPhaseField, PhaseField.shape, np.float64)
	gpuQuantumState.free()
	gpuPhaseField.free()
	gpuRhoField.free()
	gpuLattice.free()

# ANIMATE THE FIGURE WITH MOVIEPY, WRITE AN ANIMATED GIF
i=-1
def make_frame(t):
	global RhoField,PhaseField,data_dir,i,mf_index,frames_to_plot,rhoMax
	setComponent(data_dir,i,mf_index) #Set field values for frame i
	i=(i+1)%(frames_to_plot)
	####Remake plot####
	# Create the data source
	RhoField = (1./rhoMax)*RhoField
	src = mlab.pipeline.array2d_source(np.log(RhoField.real +.00001))

	# Add the additional scalar information 'w', for color map data
	phaseData = src.image_data.point_data.add_array((PhaseField.real).T.ravel())
	src.image_data.point_data.get_array(phaseData).name = 'color'
	mlab.clf() # clear the figure (to reset the colors)
	src.update()

	#Build surface plot pipeline
	warp = mlab.pipeline.warp_scalar(src, warp_scale = 50)
	normals = mlab.pipeline.poly_data_normals(warp)
	active_attr = mlab.pipeline.set_active_attribute(normals,point_scalars='color')
	surf = mlab.pipeline.surface(active_attr,colormap='hsv',vmin=0.,vmax=2.*np.pi)


	#Add labels, outlines, and legends
	zMax = 10000.*rhoMax
	print("zMax = ", zMax)
	print("Field Max ", np.amax(RhoField.real))
	phi_viewangle = 55
	theta = 120

	mlab.view(azimuth=theta, elevation=phi_viewangle, distance=5.*xSize, focalpoint=(0,0, -.5*xSize))
	# mlab.move(up=270.,right=-140.)
	#mlab.outline(surf,extent=[-xSize/2,xSize/2,-ySize/2,ySize/2,0,zMax])
	# ax = mlab.axes(surf,extent=[-xSize/2,xSize/2,-ySize/2,ySize/2,0,zMax],
	# 			ranges=[xSize-1,0,0,ySize-1,0,rhoMax],xlabel='',ylabel='',
	# 			zlabel='',nb_labels=0)
	# mlab.colorbar(title='Phase(Radians)',orientation='vertical', nb_labels=3)
	mlab.savefig(image_directory + 'mf_'+str(2-mf_index) + '_' + frames_list[i].split(".npy")[0] +".png", figure=mlab.gcf())
	ss = mlab.screenshot(antialiased=False)
	return ss

def plotComponent(directory, frames, component = 0):
  global QuantumState, RhoField, PhaseField, Lattice, duration, rhoMax
  Lattice[0],  Lattice[1], Lattice[2], Lattice[3]= xSize, ySize, zSize, component
  gpuQuantumState = drv.to_device(QuantumState)
  gpuPhaseField = drv.to_device(PhaseField)
  gpuRhoField = drv.to_device(RhoField)
  gpuLattice = drv.to_device(Lattice)
  getPlotDetails(gpuQuantumState, gpuRhoField, gpuPhaseField, gpuLattice, 
                block=(blockX,blockY,blockZ), grid=(gridX,gridY))
  RhoField = drv.from_device(gpuRhoField, RhoField.shape, DTYPE)
  PhaseField = drv.from_device(gpuPhaseField, PhaseField.shape, np.float64)
  QuantumState = drv.from_device(gpuQuantumState, QuantumState.shape, DTYPE)
  rhoMax = get_max(QuantumState, component, frames, directory)# get_max(QuantumState, component)
  print("Rho max = ", rhoMax)
  animation = mpy.VideoClip(make_frame, duration=duration)
  animation.write_videofile(animation_directory+'mf_'+str(2-mf_index)+".mp4", fps=fps)
  gpuQuantumState.free()
  gpuPhaseField.free()
  gpuRhoField.free()
  gpuLattice.free()

def get_max(QuantumState, component, frames, directory):
	print("Calculating maximum rho")
	comp_max = 0.
	for i in xrange(len(frames)): 
		QS = np.load(directory + frames[i])
		fermi_probability = QuantumState[:, :, :, 2*component]*QuantumState[:, :, :, 2*component].conjugate() + QuantumState[:, :, :, 1 + 2*component]*QuantumState[:, :, :, 1 + 2*component].conjugate()
		if np.amax(fermi_probability) > comp_max:
			comp_max = np.amax(fermi_probability) 
	print "Done."
	return comp_max

# def frameMax(directories,component = 1):
# 	global rhoMax, fps, frames_to_plot
# 	global startFrame, QuantumStatePhaseField, RhoField, PhaseField, Lattice
# 	setDirectory(directories, comp = component)
# 	QuantumState = np.load(directories[-1] + '.npy')
# 	PhaseField = np.zeros((xSize, ySize), dtype = DTYPE)
# 	RhoField = np.zeros((xSize, ySize), dtype = DTYPE)
# 	Lattice = np.zeros(4, dtype = np.int_)
# 	Lattice[0],  Lattice[1], Lattice[2], Lattice[3]= xSize, ySize, zSize, component
# 	gpuQuantumState = drv.to_device(QuantumState)
# 	gpuPhaseField = drv.to_device(PhaseField)
# 	gpuRhoField = drv.to_device(RhoField)
# 	gpuLattice = drv.to_device(Lattice)
# 	getPlotDetails(gpuQuantumState, gpuRhoField, gpuPhaseField, gpuLattice, 
# 	            block=(blockX,blockY,blockZ), grid=(gridX,gridY))
# 	RhoField = drv.from_device(gpuRhoField, RhoField.shape, DTYPE)
# 	PhaseField = drv.from_device(gpuPhaseField, PhaseField.shape, np.float64)
# 	for d in (directories):
# 		QuantumStateOld = QuantumState.copy()
# 		QuantumState = np.load(d + '.npy')
# 		gpuQuantumState.free()
# 		gpuQuantumState = drv.to_device(QuantumState)
# 		getPlotDetails(gpuQuantumState, gpuRhoField, gpuPhaseField, gpuLattice, 
# 	            	block=(blockX,blockY,blockZ), grid=(gridX,gridY))
# 		RhoField = drv.from_device(gpuRhoField, RhoField.shape, DTYPE)
# 		frameRhoMax = np.amax(RhoField.real)
# 		if frameRhoMax > rhoMax:
# 			if directories.index(d) < frames_to_plot:
# 				frames_to_plot = directories.index(d)
# 			break
# 	gpuQuantumState.free()
# 	gpuPhaseField.free()
# 	gpuRhoField.free()
# 	gpuLattice.free()


def get_mf_levels(directory, frames):
  mf_levels = []
  QuantumState = np.load(directory + frames[-1])
  for mf in xrange(vectorSize/2):
    if np.any(QuantumState[:,:,:,mf*2]):
      print(mf)
      mf_levels.append(mf)
  return mf_levels

def plot_directory(directory, ani_dir, image_dir, frames, **kwargs):
	global data_dir, animation_directory ,image_directory, mf_index
	global QuantumState, frames_to_plot, duration, fps, i, frames_list
	QuantumState = np.load(directory + frames[0])
	set_globals(directory + frames[0], QuantumState)
	data_dir = directory
	image_directory = image_dir
	animation_directory = ani_dir
	frames_list = frames
	mf_levels = get_mf_levels(directory, frames)
	if 'fps' in kwargs:
		fps = kwargs['fps']
	frames_to_plot = len(frames)
	# for mf in mf_levels:
	#  	frameMax(directory, mf) #Finds frame with rho>rhoMax
  	duration = ((frames_to_plot - .5)/float(fps))
  	if 'duration' in kwargs:
		duration = kwargs['duration']
  	for mf in mf_levels:
  		print("Plotting Component ", mf)
  		mf_index = mf
  		i = -1
  		plotComponent(directory, frames, mf)



################################################################
################## Directory to plot ###########################
################################################################

# root, experimentDirs, files = os.walk('SimVersion_1.0/cross_stationary_states/').next()
# print(experimentDirs)
# for d in experimentDirs:
#   root, size, files  = os.walk('SimVersion_1.0/cross_stationary_states/' + d).next()
#   for d1 in size:
#     root, kwarg, files =  os.walk('SimVersion_1.0/cross_stationary_states/' + d + '/' + d1).next()
#     for d2 in kwarg:
# 		print 'Plotting....     SimVersion_1.0/cross_stationary_states/' + d + '/' + d1 +'/' + d2 + '/Data/'
# 		plotDir = 'SimVersion_1.0/cross_stationary_states/' + d + '/' + d1 +'/' + d2 + '/Data/'
# 		visDir = plotDir.replace("Data", "Visualization")
# 		if not os.path.exists(visDir):
# 			os.makedirs(visDir)
# 		plotDirectory(plotDir)
# 		frames_to_plot=80 


# plotDirectory(plotDir)