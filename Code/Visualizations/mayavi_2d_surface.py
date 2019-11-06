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
DTYPE = np.complex
Lattice = np.zeros(4, dtype = np.int_)
RhoField = None
PhaseField = None
xSize = None
rhoMax = .5
fig_myv = mlab.figure(size=(1200,1200),bgcolor=(1,1,1),fgcolor=(0.,0.,0.))
mf_levels = []
######## CUDA Setup ##########
gpuMagic = gpuVis.gpuSource
getPlotDetails = gpuMagic.get_function("getPlotDetailsMayavi")
getTotalDensity = gpuMagic.get_function("getPlotDetailsMayaviTotal")


######## Functions ##########
def setComponent(data_dir, mf, global_vars):
	global RhoField, PhaseField, xSize
	blockX, blockY, blockZ = global_vars["blockX"], global_vars["blockY"], global_vars["blockZ"]
	gridX, gridY, gridZ = global_vars["gridX"] * global_vars["num_GPUs"], global_vars["gridY"], global_vars["gridZ"]
	QuantumState = np.load(data_dir)
	xSize, ySize, zSize = QuantumState.shape[0], QuantumState.shape[1], QuantumState.shape[2]
	Lattice[0],  Lattice[1], Lattice[2], Lattice[3]= xSize, ySize, zSize, mf
	RhoField =   np.zeros((xSize, ySize), dtype = DTYPE)
	PhaseField =   np.zeros((xSize, ySize), dtype = np.float64)
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

def setTotalDensity(data_dir, mf, global_vars):
	global RhoField, PhaseField, xSize
	blockX, blockY, blockZ = global_vars["blockX"], global_vars["blockY"], global_vars["blockZ"]
	gridX, gridY, gridZ = global_vars["gridX"] * global_vars["num_GPUs"], global_vars["gridY"], global_vars["gridZ"]
	QuantumState = np.load(data_dir)
	xSize, ySize, zSize = QuantumState.shape[0], QuantumState.shape[1], QuantumState.shape[2]
	Lattice[0],  Lattice[1], Lattice[2], Lattice[3]= xSize, ySize, zSize, mf
	RhoField =   np.zeros((xSize, ySize), dtype = DTYPE)
	PhaseField =   np.zeros((xSize, ySize), dtype = np.float64)
	gpuQuantumState = drv.to_device(QuantumState)
	gpuPhaseField = drv.to_device(PhaseField)
	gpuRhoField = drv.to_device(RhoField)
	gpuLattice = drv.to_device(Lattice)
	getTotalDensity(gpuQuantumState, gpuRhoField, gpuPhaseField, gpuLattice, 
                block=(blockX,blockY,blockZ), grid=(gridX,gridY))
	RhoField = drv.from_device(gpuRhoField, RhoField.shape, DTYPE)
	PhaseField = drv.from_device(gpuPhaseField, PhaseField.shape, np.float64)
	gpuQuantumState.free()
	gpuPhaseField.free()
	gpuRhoField.free()
	gpuLattice.free()

# ANIMATE THE FIGURE WITH MOVIEPY, WRITE AN ANIMATED GIF
def plotComponent(data_dir, frame, image_dir, frames, mf, global_vars, total_density):
	if not total_density:
		setComponent(data_dir, mf, global_vars)
	else:
		print("YANG GANG")
		setTotalDensity(data_dir, mf, global_vars)
	src = mlab.pipeline.array2d_source(RhoField.real)

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
	zMax = 5000.*rhoMax
	phi_viewangle = 55
	theta = 120

	mlab.view(azimuth=theta, elevation=phi_viewangle, distance=3.*xSize, focalpoint=(0,0, .2*xSize))
	if not os.path.exists(image_dir + 'mf_'+str(2-mf) + '/'):
		os.makedirs(image_dir + 'mf_'+str(2-mf) + '/')
	mlab.savefig(image_dir + 'mf_'+str(2-mf) + '/' + frame.split(".")[0] +".png", figure=mlab.gcf())

  


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


def get_mf_levels(directory, frames):
  global mf_levels, vectorSize
  QuantumState = np.load(directory.split("Frame")[0] + frames[-1])
  vectorSize = QuantumState.shape[3]
  for mf in xrange(vectorSize/2):
    if np.any(QuantumState[:,:,:,mf*2]):
      print("Found a non-zero field in mf = " + str(2 - mf))
      mf_levels.append(mf)

def make_frame(frame_dir, frame, image_dir, frames, global_vars, find_total_max = False, total_density = False, **kwargs):
	global mf_levels
	if frame == "Frame_00000000.npy":
		get_mf_levels(frame_dir, frames)
	if find_total_max == True:
		for mf in mf_levels:
		 	frameMax(directory, mf) 
	for mf in mf_levels:
		plotComponent(frame_dir, frame, image_dir, frames, mf, global_vars, total_density)

