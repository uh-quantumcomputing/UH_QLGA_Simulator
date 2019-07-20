"Importing initialization files..."
import pycuda.driver as drv 
import numpy as np
from importlib import import_module
from shutil import copyfile
import os

drv.init() 
DTYPE = np.complex


class gpuObject:
	def __init__(self, QuantumState, deviceID, deviceNum, global_vars):
		print "Initializing GPU " + str(deviceNum) + " ..." 
		self.gridX, self.gridY, self.gridZ = global_vars["gridX"], global_vars["gridY"], global_vars["gridZ"]
		self.blockX, self.blockY, self.blockZ = global_vars["blockX"], global_vars["blockY"], global_vars["blockZ"]
		self.xSize, self.ySize, self.zSize = self.gridX*self.blockX, self.gridY*self.blockY, self.gridZ*self.blockZ
		self.root_dir = 'Code/Initialization/'
		self.deviceNum = deviceNum
		self.device = drv.Device(deviceID) 
		self.context = self.device.make_context(drv.ctx_flags.SCHED_YIELD)
		self.compile_init_source(global_vars)
		self.compile_model_source(global_vars)
		init_source = 'Code/Initialization/compiled_init_cuda_code_' + str(deviceNum) + '.py'
		model_source = 'Code/Initialization/compiled_model_cuda_code_' + str(deviceNum) + '.py'
		copyfile('Code/Initialization/compiled_init_cuda_code.py', init_source)
		copyfile('Code/Initialization/compiled_model_cuda_code.py', model_source)
		init_source =  __import__('Code.Initialization.compiled_init_cuda_code_' + str(deviceNum), globals(), locals(), ['gpuSource'], -1)
		model_source = __import__('Code.Initialization.compiled_model_cuda_code_' + str(deviceNum), globals(), locals(), ['gpuSource'], -1)
		self.gpuInitMagic = init_source.gpuSource
		self.gpuModelMagic = model_source.gpuSource
		gpu_param_array = np.asarray([deviceNum, global_vars["num_GPUs"]], dtype = np.int_)
		lattice_array = np.asarray([self.xSize, self.ySize, self.zSize, global_vars["Qx"], global_vars["Qy"], global_vars["Qz"], global_vars["vectorSize"], global_vars["time_step"]], dtype = np.int_)
		self.qShape = QuantumState[deviceNum].shape
		self.GPU_Lattice = drv.to_device(lattice_array)
		self.QField = drv.to_device(QuantumState[deviceNum])
		self.QFieldCopy = drv.to_device(QuantumState[deviceNum])
		self.GPU_params = drv.to_device(gpu_param_array)
		self.context.pop()
		print "GPU " + str(deviceNum) + " initialized."

                                                                                    
	def compile_init_source(self, global_vars):
		print ("Compiling initialization CUDA code ...")
		init = import_module("Code.Initialization.CUDA_Initializations.Default." + global_vars["init"])  
		f1= open(self.root_dir + "base_CUDA.py","r")
		base_contents = f1.read()
		f2 = open(self.root_dir + "compiled_init_cuda_code.py", "w+")
		f2.write(base_contents + "\n")
		initiailization_contents = init.get_CUDA(global_vars["vectorSize"], **global_vars["exp_kwargs"])
		f2.write(initiailization_contents + "\n")
		f2.write(r'''""")''')
		f1.close()
		f2.close()
		print ("Finished compiling initialization CUDA code.")


	def compile_model_source(self, global_vars):
		print ("Compiling physics model CUDA code ...")
		# Making local variables
		particle_number, dimensions, vectorSize = global_vars["particle_number"], global_vars["dimensions"], global_vars["vectorSize"]
		kinetic_operator, model, potential, measurement = global_vars["kinetic_operator"], global_vars["model"], global_vars["potential"], global_vars["measurement"]
		exp_kwargs, potential_kwargs, measurement_kwargs = global_vars["exp_kwargs"], global_vars["potential_kwargs"], global_vars["measurement_kwargs"]
		# Importing files  
		kinetic = import_module("Code.Initialization.CUDA_physics_models.kinetic." + kinetic_operator +"_" + str(particle_number) + "P")
		internal_interaction = import_module("Code.Initialization.CUDA_physics_models.internal_interaction." + model + "_" + str(particle_number) + "P")
		external_interaction = import_module("Code.Initialization.CUDA_physics_models.external_interaction." + potential + "_" + str(particle_number) + "P")
		measurement_interaction = import_module("Code.Initialization.CUDA_physics_models.measurement_interaction." + measurement + "_" + str(particle_number) + "P")
		# Writing CUDA file 
		f1= open(self.root_dir + "base_CUDA.py","r")
		base_contents = f1.read()
		f2 = open(self.root_dir + "compiled_model_cuda_code.py", "w+")
		f2.write(base_contents + "\n")
		kinetic_contents = kinetic.get_CUDA(vectorSize, global_vars["num_GPUs"])
		f2.write(kinetic_contents + "\n")
		internal_interaction_contents = internal_interaction.get_CUDA(dimensions, vectorSize, **exp_kwargs)
		f2.write(internal_interaction_contents + "\n")	
		external_interaction_contents = external_interaction.get_CUDA(dimensions, vectorSize, **potential_kwargs)
		f2.write(external_interaction_contents + "\n")
		measurement_interaction_contents = measurement_interaction.get_CUDA(dimensions, vectorSize, **measurement_kwargs)
		f2.write(measurement_interaction_contents + "\n")	
		f2.write(r'''""")''')
		f1.close()
		f2.close()
		print ("Finished compiling physics model CUDA code.")

	def initialize(self):
		self.context.push()
		initialize_field = self.gpuInitMagic.get_function("initialize")
		initialize_field(self.QField, self.GPU_Lattice, self.GPU_params, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX,self.gridY, self.gridZ))
		initialize_field(self.QFieldCopy, self.GPU_Lattice, self.GPU_params, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX,self.gridY, self.gridZ))
		self.context.pop()

	def freeMem(self):
		self.context.push()
		self.GPU_Lattice.free()
		self.GPU_params.free()
		self.QField.free()
		self.QFieldCopy.free()
		self.context.pop()
		self.context.detach()
		print "Memory Freed for device ", self.deviceNum

	def removeFiles(self):
		if self.deviceNum == 0:
			# os.remove('Code/Initialization/compiled_init_cuda_code.py')
			os.remove('Code/Initialization/compiled_model_cuda_code.py')	
		os.remove('Code/Initialization/compiled_init_cuda_code_' + str(self.deviceNum) + '.py')
		os.remove('Code/Initialization/compiled_init_cuda_code_' + str(self.deviceNum) + '.pyc')
		os.remove('Code/Initialization/compiled_model_cuda_code_' + str(self.deviceNum) + '.py')
		os.remove('Code/Initialization/compiled_model_cuda_code_' + str(self.deviceNum) + '.pyc')
		print "Files removed for device ", self.deviceNum

	def stream(self, direction, dim, comp, num_GPUs, field_copy_pointers):
		self.context.push()	
		func_name = "stream" + dim + direction + str(int(comp))
		stream_field = self.gpuModelMagic.get_function(func_name)
		if dim=="X":
			exec(generate_shared_call(num_GPUs, stream=True), {"stream_field" : stream_field, "field_copy_pointers" : field_copy_pointers, "self" : self})
		else:	
			stream_field(self.QField, self.QFieldCopy, self.GPU_Lattice, self.GPU_params, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX,self.gridY, self.gridZ))
		self.context.pop()

	def collide_multi(self, dim, num_GPUs, field_copy_pointers):
		self.context.push()	
		func_name = "collide" + dim
		collide_field = self.gpuModelMagic.get_function(func_name)
		if dim=="X":
			exec(generate_shared_call(num_GPUs, stream=False), {"collide_field" : collide_field, "field_copy_pointers" : field_copy_pointers, "self" : self})
		else:	
			collide_field(self.QField, self.QFieldCopy, self.GPU_Lattice, self.GPU_params, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX,self.gridY, self.gridZ))
		self.context.pop()

	def collide(self):
		self.context.push()	
		collide_field = self.gpuModelMagic.get_function("collide")	
		collide_field(self.QField, self.QFieldCopy, self.GPU_Lattice, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX,self.gridY, self.gridZ))
		self.context.pop()

	def internal_interaction(self):
		self.context.push()	
		internal_collision = self.gpuModelMagic.get_function("internal")	
		internal_collision(self.QField, self.QFieldCopy, self.GPU_Lattice, self.GPU_params, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX,self.gridY, self.gridZ))
		self.context.pop()

	def external_interaction(self):
		self.context.push()	
		external_collision = self.gpuModelMagic.get_function("external")	
		external_collision(self.QField, self.QFieldCopy, self.GPU_Lattice, self.GPU_params, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX,self.gridY, self.gridZ))
		self.context.pop()

	def measurement_interaction(self):
		self.context.push()	
		measurement = self.gpuModelMagic.get_function("measurement")	
		measurement(self.QField, self.QFieldCopy, self.GPU_Lattice, self.GPU_params, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX,self.gridY, self.gridZ))
		self.context.pop()

	def setCopy(self):
		self.context.push()
		copy_field = self.gpuModelMagic.get_function("copy")
		copy_field(self.QField, self.QFieldCopy, self.GPU_Lattice, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX, self.gridY, self.gridZ))
		self.context.pop()

	def incrementTime(self):
		self.context.push()
		inc_time = self.gpuModelMagic.get_function("incrementTime")
		inc_time(self.GPU_Lattice, block=(1,1,1), grid=(1, 1, 1))
		self.context.pop()

	def pullField(self):
		self.context.push()
		QuantumState = drv.from_device(self.QField, self.qShape, DTYPE)
		self.context.pop()
		return QuantumState

	def zeroFields(self):
		self.context.push()
		self.zeroField_GPU(self.QField, self.QFieldCopy, self.gpuVortField, self.GPU_Lattice, block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX, self.gridY, self.gridZ))
		self.context.pop()

	def enableBoundaryAccess(self, bound_context):
		self.context.push()
		self.context.enable_peer_access(bound_context)
		self.context.pop()

	def synchronizeDevice(self):
		self.context.push()
		self.context.synchronize()
		self.context.pop()

# Stream/collide function call generator for shared GPU memory
def generate_shared_call(num_GPUs, stream=True):
		func_call = "stream_field"
		if stream!=True:
			func_call = "collide_field"
		call_pref = func_call + "(self.QField, self.QFieldCopy, self.GPU_Lattice, self.GPU_params, "
		call_suf = "block=(self.blockX,self.blockY,self.blockZ), grid=(self.gridX,self.gridY, self.gridZ))"
		for i in xrange(num_GPUs):
			call_pref += "field_copy_pointers[" + str(int(i)) + "], "
		return call_pref + call_suf