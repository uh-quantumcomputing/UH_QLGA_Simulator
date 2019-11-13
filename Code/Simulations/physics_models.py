import numpy as np

evolution_func = None
gpu_field_copy_pointers = []

def set_field_pointers(gpu, num_GPUs):
	global gpu_field_copy_pointers
	for i in xrange(num_GPUs):
		gpu_field_copy_pointers.append(gpu[i].QFieldCopy)

def set_model(gpu, num_GPUs, model, dimensions, num_particles):
	global evolution_func, gpu_field_copy_pointers
	if num_particles==1:
		evolution_func = eval(get_dim_string(dimensions) + "_D_EVOLUTION")
	else:
		evolution_func = eval(get_dim_string(dimensions) + "_D_EVOLUTION_MULTI")
	set_field_pointers(gpu, num_GPUs)

def get_dim_string(dimensions):
	if dimensions == 1:
		return "ONE" 
	if dimensions == 2:
		return "TWO" 
	if dimensions == 3:
		return "THREE" 

def evolve(gpu, num_GPUs, steps):
    for i in xrange(steps):
        evolution_func(gpu, num_GPUs)
        INCREMENT_TIME_STEP(gpu, num_GPUs)

def INCREMENT_TIME_STEP(gpu, num_GPUs):
    for i in xrange(num_GPUs):
        gpu[i].incrementTime()

def SYNC(gpu, num_GPUs):
    for i in xrange(num_GPUs):
        gpu[i].synchronizeDevice()

def COLLIDE(gpu, num_GPUs):
    for i in xrange(num_GPUs):
    	SYNC(gpu, num_GPUs)
        gpu[i].collide()

def SET_COPY(gpu, num_GPUs):
    for i in xrange(num_GPUs):
    	SYNC(gpu, num_GPUs)
        gpu[i].setCopy()

def STREAM_COLLIDE(gpu, num_GPUs, dimension, component):
	COLLIDE(gpu, num_GPUs)
	if dimension == "X":
		for i in xrange(num_GPUs):
			SYNC(gpu, num_GPUs)
			# gpu[i].stream("Pos", dimension, component, NeighborField = gpu[(i-1+num_GPUs)%num_GPUs].QFieldCopy)
			gpu[i].stream("Pos", dimension, component, num_GPUs, gpu_field_copy_pointers)
	else:
		for i in xrange(num_GPUs):
			gpu[i].stream("Pos", dimension, component, num_GPUs, gpu_field_copy_pointers)
	COLLIDE(gpu, num_GPUs)
	if dimension == "X":
		for i in xrange(num_GPUs):
			SYNC(gpu, num_GPUs)
			# gpu[i].stream("Neg", dimension, component, NeighborField = gpu[(i+1)%num_GPUs].QFieldCopy)
			gpu[i].stream("Neg", dimension, component, num_GPUs, gpu_field_copy_pointers)
	else:
		for i in xrange(num_GPUs):
			gpu[i].stream("Neg", dimension, component, num_GPUs, gpu_field_copy_pointers)

def STREAM_COLLIDE_MULTI(gpu, num_GPUs, dimension, component):
	# SET_COPY(gpu, num_GPUs)
	# #Collide
	# for i in xrange(num_GPUs):
	# 	SYNC(gpu, num_GPUs)
	# 	gpu[i].collide_multi(dimension, num_GPUs, gpu_field_copy_pointers)
	# SET_COPY(gpu, num_GPUs)
	# Stream
	for i in xrange(num_GPUs):
		SYNC(gpu, num_GPUs)
		gpu[i].stream("Pos", dimension, component, num_GPUs, gpu_field_copy_pointers)
	SET_COPY(gpu, num_GPUs)
	# # Collide
	# for i in xrange(num_GPUs):
	# 	SYNC(gpu, num_GPUs)
	# 	gpu[i].collide_multi(dimension, num_GPUs, gpu_field_copy_pointers)
	# SET_COPY(gpu, num_GPUs)
	# #Stream
	# for i in xrange(num_GPUs):
	# 	SYNC(gpu, num_GPUs)
	# 	gpu[i].stream("Neg", dimension, component, num_GPUs, gpu_field_copy_pointers)

def INTERNAL(gpu, num_GPUs):
    for i in xrange(num_GPUs):
    	SYNC(gpu, num_GPUs)
        gpu[i].internal_interaction()

def EXTERNAL(gpu, num_GPUs):
    for i in xrange(num_GPUs):
    	SYNC(gpu, num_GPUs)
        gpu[i].external_interaction()

def MEASUREMENT(gpu, num_GPUs):
    for i in xrange(num_GPUs):
    	SYNC(gpu, num_GPUs)
        gpu[i].measurement_interaction()
    
def ONE_D_EVOLUTION(gpu, num_GPUs):
	STREAM_COLLIDE(gpu, num_GPUs, "X", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "X", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "X", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "X", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	MEASUREMENT(gpu, num_GPUs)


def TWO_D_EVOLUTION(gpu, num_GPUs):
	STREAM_COLLIDE(gpu, num_GPUs, "X", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Y", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "X", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Y", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	######### Halfway ##########
	STREAM_COLLIDE(gpu, num_GPUs, "Y", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "X", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Y", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "X", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	MEASUREMENT(gpu, num_GPUs)


def THREE_D_EVOLUTION(gpu, num_GPUs):
	STREAM_COLLIDE(gpu, num_GPUs, "X", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "X", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Y", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Y", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Z", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Z", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	######### Halfway ##########
	STREAM_COLLIDE(gpu, num_GPUs, "X", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "X", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Y", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Y", 0)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Z", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE(gpu, num_GPUs, "Z", 1)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	MEASUREMENT(gpu, num_GPUs)

def ONE_D_EVOLUTION_MULTI(gpu, num_GPUs):
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 0)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 0)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 1)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 1)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	MEASUREMENT(gpu, num_GPUs)


def TWO_D_EVOLUTION_MULTI(gpu, num_GPUs):
	# STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 0)
	# SET_COPY(gpu, num_GPUs)
	# INTERNAL(gpu, num_GPUs)
	# EXTERNAL(gpu, num_GPUs)
	# STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 0)
	# SET_COPY(gpu, num_GPUs)
	# INTERNAL(gpu, num_GPUs)
	# EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Y", 0)
	SET_COPY(gpu, num_GPUs)
	# INTERNAL(gpu, num_GPUs)
	# EXTERNAL(gpu, num_GPUs)
	# STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Y", 0)
	# SET_COPY(gpu, num_GPUs)
	# INTERNAL(gpu, num_GPUs)
	# EXTERNAL(gpu, num_GPUs)
	######## Halfway ##########
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Y", 1)
	SET_COPY(gpu, num_GPUs)
	# INTERNAL(gpu, num_GPUs)
	# EXTERNAL(gpu, num_GPUs)
	# STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Y", 1)
	# SET_COPY(gpu, num_GPUs)
	# INTERNAL(gpu, num_GPUs)
	# EXTERNAL(gpu, num_GPUs)
	# STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 1)
	# SET_COPY(gpu, num_GPUs)
	# INTERNAL(gpu, num_GPUs)
	# EXTERNAL(gpu, num_GPUs)
	# STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 1)
	# SET_COPY(gpu, num_GPUs)
	# INTERNAL(gpu, num_GPUs)
	# EXTERNAL(gpu, num_GPUs)
	# MEASUREMENT(gpu, num_GPUs)


def THREE_D_EVOLUTION_MULTI(gpu, num_GPUs):
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 0)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 0)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Y", 1)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Y", 1)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Z", 0)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Z", 0)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	######### Halfway ##########
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 1)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "X", 1)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Y", 0)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Y", 0)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Z", 1)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	STREAM_COLLIDE_MULTI(gpu, num_GPUs, "Z", 1)
	SET_COPY(gpu, num_GPUs)
	INTERNAL(gpu, num_GPUs)
	EXTERNAL(gpu, num_GPUs)
	MEASUREMENT(gpu, num_GPUs)