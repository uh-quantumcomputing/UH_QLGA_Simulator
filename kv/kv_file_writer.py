import os
import kv_text_files as kvt



init_dir = './Code/Initialization/CUDA_Initializations/Default/'
model_dir = './Code/Initialization/CUDA_physics_models/internal_interaction/'
vis_dir = './Code/Visualizations/'

def find_files(d):
	file_arr = os.listdir(d)
	files = []
	for file in file_arr:
		if '.py' in file and '_init_' not in file and 'GPU' not in file:
			if 'visualizer' not in file and '.pyc' not in file:
				files.append(file.split('.py')[0]) 
	return files

def write_run():
	run_file_name = "kv/physicsmodeling.kv"
	if os.path.isfile(run_file_name):
		os.remove(run_file_name)
	f = open(run_file_name,"w+")
	found_files = [find_files(model_dir), find_files(init_dir), find_files(vis_dir)] 
	print(found_files)
	for i in range(len(found_files)):
		f.write(kvt.run[i] + str(found_files[i]) + '\n            ' )
	f.write(kvt.run[-1])
	f.close() 


write_run()