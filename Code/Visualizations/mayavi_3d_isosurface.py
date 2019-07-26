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
import custom_color_map as ccm
from tvtk.api import tvtk

mlab.options.offscreen = True

###### Constants and globals ######
DTYPE = np.complex
Lattice = np.zeros(4, dtype = np.int_)
PhaseField = None
xSize = None
rhoMax = .5
fig_myv = mlab.figure(size=(1200,1200),bgcolor=(1,1,1),fgcolor=(0.,0.,0.))
mf_levels = []
######## CUDA Setup ##########
gpuMagic = gpuVis.gpuSource
getPlotDetails = gpuMagic.get_function("getPlotDetailsMayavi_three_d")


######## Functions ##########
# def setComponent(data_dir, mf, global_vars):
# 	global RhoField, PhaseField, xSize
# 	blockX, blockY, blockZ = global_vars["blockX"], global_vars["blockY"], global_vars["blockZ"]
# 	gridX, gridY, gridZ = global_vars["gridX"] * global_vars["num_GPUs"], global_vars["gridY"], global_vars["gridZ"]
# 	QuantumState = np.load(data_dir)
# 	xSize, ySize, zSize = QuantumState.shape[0], QuantumState.shape[1], QuantumState.shape[2]
# 	QuantumStateNew = QuantumState[:xSize/2, :ySize/2, :zSize/2, :].copy(order='C')
# 	Lattice[0],  Lattice[1], Lattice[2], Lattice[3]= xSize, ySize, zSize, mf
# 	RhoField =   np.zeros((xSize/2, ySize/2, zSize/2), dtype = DTYPE)
# 	PhaseField =   np.zeros((xSize/2, ySize/2, zSize/2), dtype = np.float64)
# 	gpuQuantumState = drv.to_device(QuantumStateNew)
# 	gpuPhaseField = drv.to_device(PhaseField)
# 	gpuRhoField = drv.to_device(RhoField)
# 	gpuLattice = drv.to_device(Lattice)
# 	getPlotDetails(gpuQuantumState, gpuRhoField, gpuPhaseField, gpuLattice, 
#                 block=(blockX,blockY,blockZ), grid=(gridX/2,gridY/2, gridZ/2))
# 	RhoField = drv.from_device(gpuRhoField, RhoField.shape, DTYPE)
# 	PhaseField = drv.from_device(gpuPhaseField, PhaseField.shape, np.float64)
# 	gpuQuantumState.free()
# 	gpuPhaseField.free()
# 	gpuRhoField.free()
# 	gpuLattice.free()

def set_rho(data_dir, mf, global_vars):
	xSize, ySize, zSize, vectorSize = global_vars["xSize"], global_vars["ySize"], global_vars["zSize"], global_vars["vectorSize"]
	QuantumState = np.load(data_dir)
	QuantumStateNew = QuantumState[:xSize/2, :ySize/2, :zSize/2, :]
	RhoField = QuantumStateNew[:, :, :, 2*mf] * QuantumStateNew[:, :, :, 2*mf].conjugate() + QuantumStateNew[:, :, :, 2*mf + 1] * QuantumStateNew[:, :, :, 2*mf + 1].conjugate()
	color_field = set_color_data(QuantumStateNew, global_vars)
	return RhoField, color_field

def set_color_data(Q, global_vars):
	xSize, ySize, zSize, vectorSize = global_vars["xSize"], global_vars["ySize"], global_vars["zSize"], global_vars["vectorSize"]
	total_size = xSize*ySize*zSize/8
	temp = np.arange(total_size, dtype = DTYPE)/total_size
	return temp.reshape((xSize/2, ySize/2, zSize/2)) 

# ANIMATE THE FIGURE WITH MOVIEPY, WRITE AN ANIMATED GIF
def plotComponent(data_dir, frame, image_dir, frames, mf, global_vars):
	xSize, ySize, zSize = global_vars["xSize"], global_vars["ySize"], global_vars["zSize"]
	#setComponent(data_dir, mf, global_vars)
	RhoField, color_field = set_rho(data_dir, mf, global_vars)
	src = mlab.pipeline.scalar_field(RhoField.real)   #The 3D whose isosurface we want to compute
	c= src.image_data.point_data.add_array((color_field.real).T.ravel()) 
	src.image_data.point_data.get_array(c).name = 'angle'
	mlab.clf() # clear the figure (to reset the colors) 
	src.update()                # thecustomized colormap
	src2 = mlab.pipeline.set_active_attribute(src, point_scalars='scalar') #
	contour = mlab.pipeline.contour(src2)
	contour2 = mlab.pipeline.set_active_attribute(contour,point_scalars='angle') #few more
	surf = mlab.pipeline.surface(contour2, colormap='hsv',vmin= 0., vmax = 2.) #display the surface
	print surf.module_manager.scalar_lut_manager.lut.number_of_colors
	surf.module_manager.scalar_lut_manager.lut.number_of_colors = 1000
	print surf.module_manager.scalar_lut_manager.lut.number_of_colors
	lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()
	print lut.shape
	lut[:, -1] = np.linspace(0, 255, 256)
	lut2 = .5*np.ones((256*256*256, 4))
	surf.module_manager.scalar_lut_manager.lut.table = lut
	mlab.view()
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
  global mf_levels
  mf_levels.append(1)
  # QuantumState = np.load(directory.split("Frame")[0] + frames[-1])
  # vectorSize = QuantumState.shape[3]
  # for mf in xrange(vectorSize/2):
  #   if np.any(QuantumState[:,:,:,mf*2]):
  #     print("Found a non-zero field in mf = " + str(2 - mf))
  #     mf_levels.append(mf)
  #   else:
  #   	print("We found the mf " + str(2 - mf) + " to be empty")

def make_frame(frame_dir, frame, image_dir, frames, global_vars, find_total_max = False, **kwargs):
	global mf_levels
	if frame == "Frame_00000000.npy":
		get_mf_levels(frame_dir, frames)
	if find_total_max == True:
		for mf in mf_levels:
		 	frameMax(directory, mf) 
  	for mf in mf_levels:
  		print('Plotting Component ', mf, ' for frame ', frame)
  		plotComponent(frame_dir, frame, image_dir, frames, mf, global_vars)
