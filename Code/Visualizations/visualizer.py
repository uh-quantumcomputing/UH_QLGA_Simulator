"Importing visualization files..."
import numpy as np
import os
import cv2
os.environ['CUDA_DEVICE'] = str(0) #Set CUDA device, starting at 0
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from importlib import import_module


class visualizer:
	def __init__(self, technique, global_vars, **kwargs):
		print("here")
		self.global_vars = global_vars
		self.technique = technique
		self.directory_name = global_vars["base_directory_name"]
		self.already_visualized = []
		self.frames = self.get_frames()
		self.image_dir = self.make_image_dir()
		self.ani_dir = self.make_ani_dir()
		self.frame_maker = None
		self.animation_maker = None
		self.fig = None		
		self.set_vis_technique()

	def make_image_dir(self):
		dn = self.directory_name + "Images/" +self.technique +"/"
		if not os.path.exists(dn):
			os.makedirs(dn)
		return dn

	def make_ani_dir(self):
		dn = self.directory_name + "Animation/" +self.technique +"/"
		if not os.path.exists(dn):
			os.makedirs(dn)
		return dn

	def get_frames(self):
		arr = os.listdir(self.directory_name + "Data/")
		arr_new = [x for x in arr if x not in self.already_visualized]
		arr_new.sort()
		return arr_new


	def set_vis_technique(self):
		vis_file = import_module("Code.Visualizations." + self.technique)
		self.frame_maker = vis_file.make_frame 


	def visualize(self, **kwargs):
		for frame in self.frames:
			print (frame)
			self.frame_maker(self.directory_name  + "Data/" + frame, frame, self.image_dir, self.frames, self.global_vars, **kwargs)

	def animate(self, fps = 2, **kwargs):
		name = ''
		dirs = [x[0] for x in os.walk(self.image_dir)]
		files = [x[2] for x in os.walk(self.image_dir)]
		for idx, d in enumerate(dirs):
			if len(files[idx]) > 0:
				images = sorted(files[idx])
				if (d != self.image_dir):
					name = d.split("/")[-1]
				frame = cv2.imread(os.path.join(d, images[0]))
				height, width, layers = frame.shape
				save_name = self.get_animation_name(self.ani_dir, name, **kwargs)
				print(save_name)
				fourcc = cv2.VideoWriter_fourcc(*'XVID')
				video = cv2.VideoWriter(save_name, fourcc, fps, (width,height))
				
				for image in images:
					print (os.path.join(d, image))
					video.write(cv2.imread(os.path.join(d, image)))
				cv2.destroyAllWindows()
				video.release()

	def get_animation_name(self, ani_dir, name, vid_fmt = "avi" , **kwargs):
		if name == '':
			return ani_dir + "animation." + vid_fmt
		else:
			return self.ani_dir + name + "_animation." + vid_fmt

