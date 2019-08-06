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
fig_myv = mlab.figure(size=(1200,1200),bgcolor=(1,1,1),fgcolor=(0.,0.,0.))
mf_levels = []
max_value = 0.
min_value = 0.


alpha = 255*np.power(np.linspace(0, 1, 256), .7)
z = np.zeros(256, dtype = np.int32)
color = 255*np.ones(256, dtype = np.int32)

red_lut = np.append(np.append(color, z), np.append(z, alpha)).reshape((4, 256)).T
blue_lut = np.append(np.append(z, color), np.append(z, alpha)).reshape((4, 256)).T
green_lut = np.append(np.append(z, z), np.append(color, alpha)).reshape((4, 256)).T
white_lut = np.append(np.append(z, z), np.append(color, color)).reshape((4, 256)).T


######## CUDA Setup ##########
gpuMagic = gpuVis.gpuSource
getPlotDetails = gpuMagic.get_function("getPlotDetailsMayavi_three_d")


def calc_F(quantum_field):
    plus_two = quantum_field[:, :, :, 0] + quantum_field[:, :, :, 1]
    plus_one = quantum_field[:, :, :, 2] + quantum_field[:, :, :, 3]
    zero = quantum_field[:, :, :, 4] + quantum_field[:, :, :, 5]
    minus_one = quantum_field[:, :, :, 6] + quantum_field[:, :, :, 7]
    minus_two = quantum_field[:, :, :, 8] + quantum_field[:, :, :, 9]
    Fz = 2.*(plus_two*plus_two.conjugate() - minus_two*minus_two.conjugate()) + plus_one*plus_one.conjugate() - minus_one*minus_one.conjugate()
    Fplus = 2.*(plus_two.conjugate()*plus_one + minus_one.conjugate()*minus_two) + np.sqrt(6.)*(plus_one.conjugate()*zero +zero.conjugate()*minus_one) 
    Fminus = Fplus.conjugate()
    Fx = (1./2.)*(Fplus + Fminus)
    Fy = (1./2.j)*(Fplus - Fminus)
    Ftrans = np.sqrt(Fx*Fx.conjugate() + Fy*Fy.conjugate())
    Fz = np.sqrt(Fz*Fz.conjugate())
    rho = plus_two*plus_two.conjugate() + plus_one*plus_one.conjugate() + minus_two*minus_two.conjugate() + minus_one*minus_one.conjugate() + zero*zero.conjugate()
    singlet_amplitude = np.abs((1./np.sqrt(5.))*(2.*plus_two*minus_two - 2.*plus_one*minus_one + zero*zero))
    max_sing = np.amax([.1 + 0.j, np.amax(singlet_amplitude)])
    max_z = np.amax([.1 + 0.j, np.amax(Fz)])
    max_trans = np.amax([.1 + 0.j, np.amax(Ftrans)])
    max_rho = np.amax([.1 + 0.j, np.amax(rho)])
    return max_rho, max_z, max_trans, max_sing, rho, Fz, Ftrans, singlet_amplitude



def colorize(max_rho, max_z, max_trans, max_sing, rho, Fz, Ftrans, singlet_amplitude):
    VectorMag = np.sqrt(Fz.real * Fz.real + Ftrans.real * Ftrans.real + singlet_amplitude.real * singlet_amplitude.real)
    epsilon = 0.0000000000000000000000001
    r = (255 * np.power(np.abs(Fz.real)/(VectorMag + epsilon*np.ones(VectorMag.shape)),1)).astype('int32')
    g = (255 * np.power(np.abs(Ftrans.real)/(VectorMag + epsilon*np.ones(VectorMag.shape)),1)).astype('int32')
    b = (255 * np.power(np.abs(singlet_amplitude.real)/(VectorMag + epsilon*np.ones(VectorMag.shape)),1)).astype('int32')
    return r, g, b

def set_color_data(quantum_field):
	max_rho, max_z, max_trans, max_sing, rho, Fz, Ftrans, singlet_amplitude = calc_F(quantum_field)
	return colorize(max_rho, max_z, max_trans, max_sing, rho, Fz, Ftrans, singlet_amplitude)





def set_rho(data_dir, global_vars, full):
	scale = 2
	if full:
		scale = 1
	xSize, ySize, zSize, vectorSize = global_vars["xSize"], global_vars["ySize"], global_vars["zSize"], global_vars["vectorSize"]
	QuantumState = np.load(data_dir)
	QuantumStateNew = QuantumState[:xSize/scale, :ySize/scale, :zSize/scale, :]
	RhoField = np.zeros((xSize/scale, ySize/scale, zSize/scale), dtype = DTYPE)
	for mf in mf_levels:
		RhoField = RhoField + (QuantumStateNew[:, :, :, 2*mf] * QuantumStateNew[:, :, :, 2*mf].conjugate() + QuantumStateNew[:, :, :, 2*mf + 1] * QuantumStateNew[:, :, :, 2*mf + 1].conjugate())				
	r, g, b = set_color_data(QuantumStateNew)
	return RhoField, r, g, b


# ANIMATE THE FIGURE WITH MOVIEPY, WRITE AN ANIMATED GIF
def plotComponent(data_dir, frame, image_dir, frames, global_vars, full = False, contour_percent = [0.25, 0.5, .75], **kwargs):
	global max_value, min_value
	RhoField, r, g, b = set_rho(data_dir, global_vars, full)
	if frame == "Frame_00000000.npy":
		max_value = np.amax(RhoField.real)
		min_value = np.amin(RhoField.real)
	i = [min_value + (max_value - min_value)*cp for cp in contour_percent]
	mlab.clf()
	make_surface(RhoField, np.ones(RhoField.shape), white_lut, i)
	make_surface(RhoField, r, red_lut, i)
	make_surface(RhoField, g, green_lut, i)
	make_surface(RhoField, b, blue_lut, i)
	mlab.view()
	mlab.savefig(image_dir +  frame.split(".")[0] +".png", figure=mlab.gcf())


def make_surface(rho, color, lut, i):
	src = mlab.pipeline.scalar_field(rho.real)  
	c = src.image_data.point_data.add_array((color).T.ravel()) 
	src.image_data.point_data.get_array(c).name = 'characteristic'
	src.update() 
	src2 = mlab.pipeline.set_active_attribute(src, point_scalars='scalar') #
	contour = mlab.pipeline.contour(src2)
	contour2 = mlab.pipeline.set_active_attribute(contour,point_scalars='characteristic') 
	contour.filter.contours = i
	src.update() 
	surf = mlab.pipeline.surface(contour2, representation = 'surface') #display the surface
	surf.module_manager.scalar_lut_manager.lut.table = lut




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
  QuantumState = np.load(directory.split("Frame")[0] + frames[-1])
  vectorSize = QuantumState.shape[3]
  for mf in xrange(vectorSize/2):
    if np.any(QuantumState[:,:,:,mf*2]):
      print("Found a non-zero field in mf = " + str(2 - mf))
      mf_levels.append(mf)
    else:
    	print("We found the mf " + str(2 - mf) + " to be empty")

def make_frame(frame_dir, frame, image_dir, frames, global_vars, **kwargs):
	global mf_levels
	if frame == "Frame_00000000.npy":
		get_mf_levels(frame_dir, frames)
  	plotComponent(frame_dir, frame, image_dir, frames, global_vars, **kwargs)
