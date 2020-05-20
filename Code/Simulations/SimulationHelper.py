#Importing the good stuff
import os
import pycuda.autoinit
import pycuda.driver as drv
import time
import math
import numpy as np
import physics_models as models
import model_sizes as modelSizes
import lattice_maker as latticeMaker
from ..Initialization import initializations as init
from ..Visualizations import visualizer as vis
import socket
import deepdish as dd
import Tkinter as tk
import tkMessageBox
import shutil
import getpass
username = getpass.getuser()


VERSION = "SimVersion_3.0"             
DTYPE = np.complex

#######################################################################################################
          
#############################  GLOBAL VARIABLES #######################################################

#######################################################################################################
gpu = []

global_vars = {

"num_GPUs" : None,
"devices" : None,

"blockX" : None,   
"blockY" : None,    
"blockZ" : None,     
"gridX" : None,    
"gridY" : None, 
"gridZ" : None,    

"xSize" : None,
"ySize" : None,
"zSize" : None,
"Qx" : None,
"Qy" : None,
"Qz" : None,
"vectorSize" : None,


"dimensions" : None,
"particle_number" : None,

"frame_size" : None,
"num_frames" : None,
"time_step" : None,

"kinetic_operator" : None,
"model" : None,
"init" : None,
"potential" : None,
"exp_kwargs" : None,
"measurement" : None,
"potential_kwargs" : None,
"measurement_kwargs" : None,
"visualization" : None,
"vis_kwargs" : None,
"base_directory_name" : None,

"save_vort" : None
}

'''
------------------------------------------------------------
-----------------  END GLOBAL VARIABLES --------------------
------------------------------------------------------------
'''

#-------------------  SETUP FUNCTIONS ---------------------#


def set_size(Lx, Ly, Lz, PARTICLES, DEVICES):
  global global_vars 
  global_vars["particle_number"] = PARTICLES
  global_vars["num_GPUs"] = len(DEVICES)
  global_vars["devices"] = DEVICES
  get_dimensions(Lx, Ly, Lz)
  set_lattice(Lx, Ly, Lz)

def set_experiment(MODEL, KINETIC_OPERATOR, INIT, POTENTIAL, MEASUREMENT, SAVE_VORT, EXP_KWARGS, POTENTIAL_KWARGS, MEASUREMENT_KWARGS, VIS_KWARGS):
  global global_vars
  global_vars["model"], global_vars["kinetic_operator"], global_vars["init"], global_vars["potential"], global_vars['measurement'] = MODEL, KINETIC_OPERATOR, INIT, POTENTIAL, MEASUREMENT
  global_vars["exp_kwargs"], global_vars["potential_kwargs"], global_vars["measurement_kwargs"], global_vars["vis_kwargs"] = EXP_KWARGS, POTENTIAL_KWARGS, MEASUREMENT_KWARGS, VIS_KWARGS
  global_vars["vectorSize"] = modelSizes.sizes[MODEL]
  global_vars["save_vort"] = SAVE_VORT

def set_runtime(FRAME_SIZE, NUM_FRAMES, TIME_STEP = 0):
  global global_vars
  global_vars["frame_size"], global_vars["num_frames"], global_vars["time_step"] = FRAME_SIZE, NUM_FRAMES, TIME_STEP

def get_dimensions(Lx, Ly, Lz):
  global global_vars
  global_vars["dimensions"] = 1
  if Ly>1 and Lz == 1:
    global_vars["dimensions"] = 2
  elif Ly>1 and Lz>1:
    global_vars["dimensions"] = 3

def set_global_sizes((xSize, ySize, zSize, blockX, blockY, blockZ, gridX, gridY, gridZ, Qx, Qy, Qz)):
  global global_vars
  global_vars["blockX"], global_vars["blockY"], global_vars["blockZ"] = int(blockX), int(blockY), int(blockZ)
  global_vars["gridX"], global_vars["gridY"], global_vars["gridZ"] = int(gridX), int(gridY), int(gridZ)
  global_vars["xSize"], global_vars["ySize"], global_vars["zSize"] = int(xSize*global_vars["num_GPUs"]), int(ySize), int(zSize)
  global_vars["Qx"], global_vars["Qy"], global_vars["Qz"] = int(Qx), int(Qy), int(Qz)

def set_lattice(Lx, Ly, Lz):
  dimensions, particle_number, num_GPUs = global_vars["dimensions"], global_vars["particle_number"], global_vars["num_GPUs"]
  if dimensions == 1 and particle_number == 1:
    Lx = np.floor(Lx/global_vars["num_GPUs"])
    set_global_sizes(latticeMaker.getSimulationParams1P1D(Lx, Ly, Lz))
  elif dimensions == 2 and particle_number == 1:
    Lx = np.floor(Lx/global_vars["num_GPUs"])
    set_global_sizes(latticeMaker.getSimulationParams1P2D(Lx, Ly, Lz))
  elif dimensions == 3 and particle_number == 1:
    Lx = np.floor(Lx/global_vars["num_GPUs"])
    set_global_sizes(latticeMaker.getSimulationParams1P3D(Lx, Ly, Lz))
  elif dimensions == 1 and particle_number == 2:
    set_global_sizes(latticeMaker.getSimulationParams2P1D(Lx, Ly, Lz, num_GPUs))
  elif dimensions == 2 and particle_number == 2:
    set_global_sizes(latticeMaker.getSimulationParams2P2D(Lx, Ly, Lz, num_GPUs))
  elif dimensions == 3 and particle_number == 2:
    set_global_sizes(latticeMaker.getSimulationParams2P3D(Lx, Ly, Lz, num_GPUs))
  else:
    print("Unsupported particle number and dimensionality combination.  Try again.")
    quit()

def load_global_vars(path):
  global global_vars
  global_vars = dd.io.load(path)

def set_directory(BATCH, RUN, VISUALIZATION, VIS_ONLY, OVERWRITE, root_save_file = '/home/' + username + '/Desktop/Experiments'): 
  global global_vars
  first_time = False
  global_vars["visualization"] = VISUALIZATION
  global_vars["base_directory_name"] = (root_save_file + "/" + BATCH + "/" + RUN + "/" )
  base_directory_name = global_vars["base_directory_name"]
  continue_sim = ''
  im_dir = base_directory_name + "Images/" + VISUALIZATION
  ani_dir = base_directory_name + "Animation/" + VISUALIZATION
  if not os.path.exists(base_directory_name):
      os.makedirs(base_directory_name)
  if not os.path.exists(base_directory_name + "Data/"):
    os.makedirs(base_directory_name + "Data/")
    data_files = [x[2] for x in os.walk(im_dir)]
    if len(data_files) == 0:
      first_time = True
  elif not OVERWRITE:
      print("LOOK FOR DIALOGUE BOX")
      print("You may be overwriting previous experiments")
      continue_sim = give_overwrite_permission(base_directory_name)
  if continue_sim == "yes" or OVERWRITE:
    if not (VIS_ONLY or global_vars["time_step"] > 0):
      shutil.rmtree(base_directory_name + "Data/")
      os.makedirs(base_directory_name + "Data/")
    if os.path.exists(im_dir):
      shutil.rmtree(im_dir)
    if os.path.exists(ani_dir):
      shutil.rmtree(ani_dir)
  else:
    if not first_time:
      quit()
  meta_data = base_directory_name + 'meta_data.h5'
  dd.io.save(meta_data, global_vars)
  f= open(base_directory_name + "experiment_details.txt","w+")
  f.write(stringify(global_vars))
  f.close()
  print("Running an experiment in " + base_directory_name + "with:" )
  print(stringify(global_vars))
  return meta_data

def give_overwrite_permission(base_directory_name):
  root = tk.Tk()
  root.withdraw()
  MsgBox = tkMessageBox.askquestion ('Pre-exisiting data','You are about to overwrite the data in ' + base_directory_name + " do you wish to continue?" , icon = 'warning')
  root.destroy()
  return MsgBox



########################## initialization logic ################################################################
def init_GPU():
  global gpu
  xSize, ySize, zSize, vectorSize, num_GPUs = global_vars["xSize"], global_vars["ySize"], global_vars["zSize"], global_vars["vectorSize"], global_vars["num_GPUs"] 
  if xSize%num_GPUs != 0:
    print "xSize & blockX must be divisible by the number of GPUs... quiting"
    quit()
  QuantumState, gpu = [], []
  for i in xrange(num_GPUs):
    QuantumState.append(np.zeros((xSize/num_GPUs, ySize, zSize, vectorSize), dtype=DTYPE))
  for i in xrange(num_GPUs):
    gpu.append(init.gpuObject(QuantumState, global_vars["devices"][i], i, global_vars))
    print i
    gpu[i].initialize()
  if num_GPUs > 1 :
    for i in xrange(num_GPUs):
      for j in xrange(i, num_GPUs - 1):
        gpu[i].enableBoundaryAccess(gpu[(j + 1)].context)
        gpu[j + 1].enableBoundaryAccess(gpu[(i)].context)
  save_and_print(global_vars["time_step"])

def resume():
  global gpu
  xSize, ySize, zSize, vectorSize, num_GPUs = global_vars["xSize"], global_vars["ySize"], global_vars["zSize"], global_vars["vectorSize"], global_vars["num_GPUs"] 
  if xSize%num_GPUs != 0:
    print "xSize & blockX must be divisible by the number of GPUs... quiting"
    quit()
  QuantumState, gpu = [], []
  loaded_file = np.load(global_vars["base_directory_name"] + "Data/" + "Frame_" + str('{:08d}'.format(global_vars["time_step"])) + ".npy")
  for i in xrange(num_GPUs):
    QuantumState.append(loaded_file[(xSize/num_GPUs*i):(xSize/num_GPUs*(i+1)), :, :, :])
    print QuantumState[i].shape
  for i in xrange(num_GPUs):
    gpu.append(init.gpuObject(QuantumState, global_vars["devices"][i], i, global_vars))
  if num_GPUs > 1 :
    for i in xrange(num_GPUs):
      for j in xrange(i, num_GPUs - 1):
        gpu[i].enableBoundaryAccess(gpu[(j + 1)].context)
        gpu[j + 1].enableBoundaryAccess(gpu[(i)].context)

############################# Simulation  ###############################################################

def simulate(PRINTING_SUR = True):
  model, num_GPUs, dimensions, frame_size, num_frames = global_vars["model"], global_vars["num_GPUs"], global_vars["dimensions"], global_vars["frame_size"], global_vars["num_frames"]
  xSize, ySize, zSize = global_vars["xSize"], global_vars["ySize"], global_vars["zSize"]
  numParticles = global_vars["particle_number"]
  models.set_model(gpu, num_GPUs, model, dimensions, numParticles) 
  for i in xrange(num_frames):
    frame_number = global_vars["time_step"] + frame_size*(i+1)
    start_time = time.time()
    models.evolve(gpu, num_GPUs, frame_size)
    save_and_print(frame_number)
    end_time = time.time()
    if (PRINTING_SUR):
      SUR = (xSize*ySize*zSize*frame_size/(end_time - start_time))
      print "Site update rate = " + str(SUR)
      seconds_remaining = (num_frames - i-1)*(end_time - start_time)
      print "Time remaining is approximately",
      print_time(seconds_remaining)
  # zeroFieldsGPU()
  if global_vars["save_vort"]:
    saveVorticityGPU()
  clearFiles()
  clearGPU()
  print("Simulation Complete")


##########################    Visualization   #####################################
def saveVorticityGPU():
  num_GPUs, directory_name = global_vars["num_GPUs"], global_vars["base_directory_name"]
  vorticity = drv.from_device(gpu[0].gpuVortField, gpu[0].vortField.shape, dtype = np.int_)
  for i in xrange(1,global_vars["num_GPUs"]):
    vortI = gpu[i].pullVort()
    vorticity = np.concatenate((vorticity,vortI))
  direct = directory_name + "/Extra/"
  if not os.path.exists(direct):
      os.makedirs(direct)
  np.save(direct+"vorticity.npy", vorticity)

def visualize():
  visualization, vis_kwargs = global_vars["visualization"], global_vars["vis_kwargs"]
  visualizer = vis.visualizer(visualization, global_vars, **vis_kwargs)  
  visualizer.visualize(**vis_kwargs)

def animate():
  visualization, vis_kwargs = global_vars["visualization"], global_vars["vis_kwargs"]
  visualizer = vis.visualizer(visualization, global_vars, **vis_kwargs)  
  visualizer.animate(**vis_kwargs)
##############   Methods that make strings ########################################


def save_and_print(frame_number):
  num_GPUs, directory_name = global_vars["num_GPUs"], global_vars["base_directory_name"] + "Data/" 
  QuantumState = []
  for i in xrange(num_GPUs):
    QuantumState.append(gpu[i].pullField())
  QFieldOut = QuantumState[0]
  if num_GPUs>1:
    for i in xrange(1,num_GPUs):
      QFieldOut = np.append(QFieldOut, QuantumState[i], axis = 0)
  frame_name = directory_name + "Frame_" + str('{:08d}'.format(frame_number))
  np.save(directory_name + "Frame_" + str('{:08d}'.format(frame_number)) , QFieldOut)
  print ("Wrote out " + frame_name)


def print_time(seconds):
  if seconds >= 24.*3600.:
      days = math.floor(seconds/(24*3600.))
      print str(int(days)) + " day(s),", 
      seconds -= 24.*3600.*days
  if seconds >= 3600.:
    hours = math.floor(seconds/3600.)
    print str(int(hours)) + " hour(s),", 
    seconds -= 3600.*hours
  if seconds >= 60.:
    minutes = math.floor(seconds/60.)
    print str(int(minutes)) + " minute(s) and", 
    seconds -= 60*minutes
  print str(int(seconds)) + " second(s)"



def stringify(kwargs):
  S = ""
  for key in sorted(global_vars):
    S = S + key + " = " + str(global_vars[key]) + "\n"
  return S


############################################### CLEANUP FUNCTIONS ##################################################################

def clearGPU():
  global gpu
  for i in xrange(global_vars["num_GPUs"]):
      gpu[i].freeMem()

def zeroFieldsGPU():
  global gpu
  for i in xrange(global_vars["num_GPUs"]):
      gpu[i].zeroFields()

def SYNC():
  global gpu
  for i in xrange(global_vars["num_GPUs"]):
      gpu[i].synchronizeDevice()

def clearFiles():
  global gpu
  for i in xrange(global_vars["num_GPUs"]):
      gpu[i].removeFiles()

