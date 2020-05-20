from __future__ import print_function 
import setup as QLGA
import os
from types import ModuleType
import subprocess
from kv import kv_file_writer as kvw
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

kvw.write_run()

run = None

UI = {
"PARTICLES" : None,
"KINETIC_OPERATOR": 'S',
"FRAME_SIZE": None,
"NUM_FRAMES": None,
"Lx": None,
"Ly": None,
"Lz": None,
"BATCH": None,
"RUN": None,
"DEVICES": None,
"INIT": None,
"MODEL": None,
"EXP_KWARGS": None,
"POTENTIAL": None,
"POTENTIAL_KWARGS" : None,
"VISUALIZATION": None,
"VIS_KWARGS": None,
"MEASUREMENT": None,
"MEASUREMENT_KWARGS": None,
"RUN_TYPE": None,
"OVERWRITE" : False,
"TIME_STEP" : 0,
"SAVE_VORT" : False
}

class SimulationParameters(Screen):

    # Create variables at top of .kv file as blank 'Objects'
    # When the code runs through the .kv file it will assign
    # variables in .py file to matching id's in .kv file
    numParticles = ObjectProperty(None)
    run_type = ObjectProperty(None)
    numOfFrames = ObjectProperty(None)
    frameSize = ObjectProperty(None)
    lx = ObjectProperty(None)
    ly = ObjectProperty(None)
    lz = ObjectProperty(None)

    def particle_spinner_clicked(self, value):
        UI["PARTICLES"] = int(self.numParticles.text)
        # print("Number of partices Selected is " + value)

    def kinetic_spinner_clicked(self, value):
        UI["RUN_TYPE"] = str(self.run_type.text)


    def next_button(self):
        if self.frameSize.text.isdigit():
            UI["FRAME_SIZE"] = int(self.frameSize.text)
        else:
            UI["FRAME_SIZE"] = 0
        if self.frameSize.text.isdigit():
            UI["NUM_FRAMES"] = int(self.numOfFrames.text)
        else:
            UI["NUM_FRAMES"] = 0
        if self.lx.text.isdigit():
            UI["Lx"] = int(self.lx.text)
        else:
            UI["Lx"] = 1
        if self.ly.text.isdigit():
            UI["Ly"] = int(self.ly.text)
        else:
            UI["Ly"] = 1
        if self.lz.text.isdigit():
            UI["Lz"] = int(self.lz.text)
        else:
            UI["Lz"] = 1

    def go_back(self):
         print("GO BACK button pressed!!")




class RunOptions(Screen):
    batch = ObjectProperty(None)
    numGPUs = ObjectProperty(None)
    overwrite = ObjectProperty(None)
    runSolutions = ObjectProperty(None)

    def oneGPU_click(self, instance, value):
        if value is True:
            RunOptions.numGPUs = [0]

    def twoGPU_click(self, instance, value):
        if value is True:
            RunOptions.numGPUs = [0,1]


    def overwrite_click(self, instance, value):
        if value is True:
            RunOptions.overwrite = True
        else:
            RunOptions.overwrite = False

    def next_button(self):
        UI["BATCH"] = self.batch.text
        UI["RUN"] = self.runSolutions.text
        UI["DEVICES"] = self.numGPUs
        UI["OVERWRITE"] = self.overwrite


class PhysicsModeling(Screen):
	physicsModel = ObjectProperty(None)
	visTechnique = ObjectProperty(None)
	initialConditions = ObjectProperty(None)
	inits = ObjectProperty(None)

	def physicsmodel_spinner_clicked(self, value):
	    UI["MODEL"] = self.physicsModel.text

	def initialconditions_spinner_clicked(self, value):
	    UI["INIT"] = self.initialConditions.text

	def visualization_spinner_clicked(self, value):
	    UI["VISUALIZATION"] = self.visTechnique.text

	def next_button(self):
	     print("NEXT button pressed!!")

class KwargsOptions(Screen):
    expDefault = True
    expKwargs = ObjectProperty(None)
    potentialDefault = True
    potentialKwargs = ObjectProperty(None)
    measurementDefault = True
    measurementKwargs = ObjectProperty(None)
    visualizationDefault = True
    visualizationKwargs = ObjectProperty(None)


    def exp_default_click(self, instance, value):
        if value is True:
            KwargsOptions.expDefault = True
        else:
            KwargsOptions.expDefault = False

    def exp_custom_click(self, instance, value):
        if value is True:
            KwargsOptions.expDefault = False
        else:
            KwargsOptions.expDefault = True

    def potential_default_click(self, instance, value):
        if value is True:
            KwargsOptions.potentialDefault = True
        else:
            KwargsOptions.potentialDefault = False

    def potential_custom_click(self, instance, value):
        if value is True:
            KwargsOptions.potentialDefault = False
        else:
            KwargsOptions.potentialDefault = True

    def measurement_default_click(self, instance, value):
        if value is True:
            KwargsOptions.measurementDefault = True
        else:
            KwargsOptions.measurementDefault = False

    def measurement_custom_click(self, instance, value):
        if value is True:
            KwargsOptions.measurementDefault = False
        else:
            KwargsOptions.measurementDefault = True

    def visualization_default_click(self, instance, value):
        if value is True:
            KwargsOptions.visualizationDefault = True
        else:
            KwargsOptions.visualizationDefault = False

    def visualization_custom_click(self, instance, value):
        if value is True:
            KwargsOptions.visualizationDefault = False
        else:
            KwargsOptions.visualizationDefault = True

    def next_button(self):

        if self.expDefault:
            UI["EXP_KWARGS"] = {}
        else:
            UI["EXP_KWARGS"] = self.expKwargs.text

        if self.potentialDefault:
            UI["POTENTIAL"] = "No_Potential"
            UI["POTENTIAL_KWARGS"] = {}
        else:
            UI["POTENTIAL"] = "External_Function"
            UI["POTENTIAL_KWARGS"] = self.potentialKwargs.text

        if self.measurementDefault:
            UI["MEASUREMENT"] = "No_Measurement"
            UI["MEASUREMENT_KWARGS"] = {}
        else:
            UI["MEASUREMENT"] = "Measurement_1D"
            UI["MEASUREMENT_KWARGS"] = self.measurementKwargs.text

        if self.visualizationDefault:
            UI["VIS_KWARGS"] = {}
        else:
            UI["VIS_KWARGS"] = self.visualizationKwargs.text

class WindowManager(ScreenManager):
    pass

Builder.load_file("kv/runoptions.kv")
Builder.load_file("kv/physicsmodeling.kv")
Builder.load_file("kv/kwargsoptions.kv")
kv = Builder.load_file("kv/qlgasimulator.kv")


class QLGASimulator(App):
    def build(self):
        self.icon = 'ref/atom.png'
        return kv


if __name__ == "__main__":
    QLGASimulator().run()
    print(UI)
    meta_data = QLGA.setup(UI["PARTICLES"], UI["KINETIC_OPERATOR"], UI["FRAME_SIZE"], UI["NUM_FRAMES"], UI["Lx"], UI["Ly"], UI["Lz"], 
        UI["BATCH"], UI["RUN"], UI["DEVICES"], UI["INIT"], UI["MODEL"], UI["EXP_KWARGS"], UI["POTENTIAL"], UI["POTENTIAL_KWARGS"],
        UI["VISUALIZATION"], UI["VIS_KWARGS"], UI["MEASUREMENT"], UI["MEASUREMENT_KWARGS"],
        UI["RUN_TYPE"], UI["OVERWRITE"], UI["TIME_STEP"], UI["SAVE_VORT"])

    os.system("python AbstractionLayer.py " + UI["RUN_TYPE"] + " " + meta_data)
    print("Run Complete")





