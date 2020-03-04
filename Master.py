# from __future__ import print_function
# import setup as QLGA
# import os
# from types import ModuleType
# import subprocess
from kivy.app import App
from kivy.properties import ObjectProperty
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen

#todo rename output array to global variables with specific names
outputArray = [0] * 20

#todo delete debug print statements once outputs proven working
#todo add tab functionality between input boxes
#todo add export of previous run to text file and add option to reload last run
#todo ADD error message/popup
#todo delete ANI only and reformat to SIMULATIONMASTER.PY


class SimulationParameters(Screen):

    # Create variables at top of .kv file as blank 'Objects'
    # When the code runs through the .kv file it will assign
    # variables in .py file to matching id's in .kv file
    numParticles = ObjectProperty(None)
    kinOperator = ObjectProperty(None)
    numOfFrames = ObjectProperty(None)
    frameSize = ObjectProperty(None)
    lx = ObjectProperty(None)
    ly = ObjectProperty(None)
    lz = ObjectProperty(None)

    def particle_spinner_clicked(self, value):
        outputArray[0] = int(self.numParticles.text)
        # print("Number of partices Selected is " + value)

    def kinetic_spinner_clicked(self, value):
        if self.kinOperator.text == 'Shrodinger':
            outputArray[1] = 'S'
        else:
            print("Dirac not supported yet, Defaulting value to Shrodinger")
            outputArray[1] = 'S'
        # print("Kinetic Operator Selected is " + value)

    def next_button(self):
        if self.frameSize.text.isdigit():
            outputArray[2] = int(self.frameSize.text)
        else:
            outputArray[2] = 0
        if self.frameSize.text.isdigit():
            outputArray[3] = int(self.numOfFrames.text)
        else:
            outputArray[3] = 0
        if self.lx.text.isdigit():
            outputArray[4] = int(self.lx.text)
        else:
            outputArray[4] = 0
        if self.ly.text.isdigit():
            outputArray[5] = int(self.ly.text)
        else:
            outputArray[5] = 0
        if self.lz.text.isdigit():
            outputArray[6] = int(self.lz.text)
        else:
            outputArray[6] = 0

        # print("NEXT button pressed!!")
        # print("Number of Particles: ", outputArray[0])
        # print("Kinetic Operator: ", outputArray[1])
        # print("Frame Size: ", outputArray[2])
        # print("Number of Frames: ", outputArray[3])
        # print("Lx: ", outputArray[4])
        # print("Ly: ", outputArray[5])
        # print("Lz: ", outputArray[6])

    def go_back(self):
         print("GO BACK button pressed!!")
        # print("Number of Particles: ", self.numParticles.text)
        # print("Number of Frames: ", self.numOfFrames.text)
        # print("Frame Size: ", self.frameSize.text)
        # print("Lx: ", self.lx.text)
        # print("Ly: ", self.ly.text)
        # print("Lz: ", self.lz.text)


class RunOptions(Screen):
    batch = ObjectProperty(None)
    runSolutions = ObjectProperty(None)
    numGPUs = ObjectProperty(None)
    ani = ObjectProperty(None)
    visOnly = ObjectProperty(None)
    overwrite = ObjectProperty(None)

    #set checkbox values to default as False (Unchecked)
    outputArray[19] = False



#todo modfify GPU buttons to pull from device poll
    def oneGPU_click(self, instance, value):
        if value is True:
            RunOptions.numGPUs = [0]
            # print("Checkbox Checked")
        # else:
        #     print("Checkbox Unchecked")

    def twoGPU_click(self, instance, value):
        if value is True:
            RunOptions.numGPUs = [0,1]
            # print("Checkbox Checked")
        # else:
        #     print("Checkbox Unchecked")

#todo fix visOnly parameters to available options
    def visOnly_click(self, instance, value):
        if value is True:
            RunOptions.visOnly = 'run'
            # print("Checkbox Checked")
        else:
            RunOptions.visOnly = False
            # print("Checkbox Unchecked")

    def overwrite_click(self, instance, value):
        if value is True:
            RunOptions.overwrite = True
            # print("Checkbox Checked")
        else:
            RunOptions.overwrite = False
            # print("Checkbox Unchecked")

    def next_button(self):
        # print("NEXT button pressed!!")
        # print("Batches Selected ", self.batch.text)
        # print("Run Solutions ", self.runSolutions.text)
        # print("Number of GPUS: ", self.numGPUs)
        outputArray[7] = self.batch.text
        outputArray[8] = self.runSolutions.text
        outputArray[9] = self.numGPUs
        outputArray[19] = self.visOnly
        # if self.ani:
        #     print("ANI is Selected")
        # else:
        #     print("ANI IS NOT Selected")
        # if self.visOnly:
        #     print("Visual only Selected")
        # else:
        #     print("Visual only NOT Selected")
        # if self.overwrite:
        #     print("OVERWRITE Selected")
        # else:
        #     print("OVERWRITE NOT Selected")

class PhysicsModeling(Screen):
    physicsModel = ObjectProperty(None)
    visTechnique = ObjectProperty(None)
    initialConditions = ObjectProperty(None)


    def physicsmodel_spinner_clicked(self, value):
        outputArray[11] = self.physicsModel.text
        # print("Physics Model selected is " + value)

    def initialconditions_spinner_clicked(self, value):
        outputArray[10] = self.initialConditions.text
        # print("Intial Conditions " + value + " Selected")

    def visualization_spinner_clicked(self, value):
        outputArray[15] = self.visTechnique.text
        # print("Visualization" + value + " Selected")

    def next_button(self):
         print("NEXT button pressed!!")
        # print("Physics Model ", outputArray[11])
        # print("Initial Conditions ", outputArray[10])
        # print("Visualization Technique ", outputArray[15])

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
            # print("expDefault is " + str(KwargsOptions.expDefault))
        else:
            KwargsOptions.expDefault = False
            # print("expDefault is " + str(KwargsOptions.expDefault))

    def exp_custom_click(self, instance, value):
        if value is True:
            KwargsOptions.expDefault = False
            # print("expDefault is " + str(KwargsOptions.expDefault))
        else:
            KwargsOptions.expDefault = True

    def potential_default_click(self, instance, value):
        if value is True:
            KwargsOptions.potentialDefault = True
            # print("potential Default is " + str(KwargsOptions.potentialDefault))
        else:
            KwargsOptions.potentialDefault = False
            # print("potential Default is " + str(KwargsOptions.potentialDefault))

    def potential_custom_click(self, instance, value):
        if value is True:
            KwargsOptions.potentialDefault = False
            # print("potential Default is " + str(KwargsOptions.potentialDefault))
        else:
            KwargsOptions.potentialDefault = True
            # print("potential Default is " + str(KwargsOptions.potentialDefault))

    def measurement_default_click(self, instance, value):
        if value is True:
            KwargsOptions.measurementDefault = True
            # print("measurementDefault is on")
        else:
            KwargsOptions.measurementDefault = False
            # print("measurementDefault is off")

    def measurement_custom_click(self, instance, value):
        if value is True:
            KwargsOptions.measurementDefault = False
            # print("measurementDefault is on")
        else:
            KwargsOptions.measurementDefault = True
            # print("measurementDefault is off")

    def visualization_default_click(self, instance, value):
        if value is True:
            KwargsOptions.visualizationDefault = True
            # print("visualizationDefault is on")
        else:
            KwargsOptions.visualizationDefault = False
            # print("visualizationDefault is off")

    def visualization_custom_click(self, instance, value):
        if value is True:
            KwargsOptions.visualizationDefault = False
            # print("visualizationDefault is on")
        else:
            KwargsOptions.visualizationDefault = True
            # print("visualizationDefault is off")

    def next_button(self):
        # print("expDefault is " + str(KwargsOptions.expDefault))
        # print("potential Default is " + str(KwargsOptions.potentialDefault))
        # print("measurement Default is " + str(KwargsOptions.measurementDefault))
        # print("visualization Default is " + str(KwargsOptions.visualizationDefault))

        if self.expDefault:
            outputArray[12] = {}
        else:
            outputArray[12] = self.expKwargs.text

        if self.potentialDefault:
            outputArray[13] = "No_Potential"
            outputArray[14] = {}
        else:
            outputArray[13] = "External_Function"
            outputArray[14] = self.potentialKwargs.text

        if self.measurementDefault:
            outputArray[17] = "No_Measurement"
            outputArray[18] = {}
        else:
            outputArray[17] = "Measurement_1D"
            outputArray[18] = self.measurementKwargs.text

        if self.visualizationDefault:
            outputArray[16] = {}
        else:
            outputArray[16] = self.visualizationKwargs.text

class WindowManager(ScreenManager):
    pass

Builder.load_file("screens/runoptions.kv")
Builder.load_file("screens/physicsmodeling.kv")
Builder.load_file("screens/kwargsoptions.kv")
kv = Builder.load_file("screens/qlgasimulator.kv")


class QLGASimulator(App):
    def build(self):
        self.icon = 'ref/atom.png'
        return kv


if __name__ == "__main__":

    # def execute(cmd):
    #     popen = subprocess.Popen(cmd, stdout=subprocess.PIPE, universal_newlines=True, bufsize=1)
    #     for stdout_line in iter(popen.stdout.readline, ""):
    #         yield stdout_line
    #     popen.stdout.close()
    #     return_code = popen.wait()
    #     if return_code:
    #         raise subprocess.CalledProcessError(return_code, cmd)

    QLGASimulator().run()
    print(outputArray)
    # meta_data = QLGA.setup(outputArray[0],outputArray[1], outputArray[2], outputArray[3], outputArray[4], outputArray[5],  outputArray[6], outputArray[7],
	# 		outputArray[8], outputArray[9], outputArray[10], outputArray[11], outputArray[12], outputArray[13], outputArray[14], outputArray[15],
	# 		outputArray[16], outputArray[17], outputArray[18], outputArray[19], OVERWRITE = True)
    # for path in execute(["python", "AbstractionLayer.py", outputArray[19], meta_data]):
    #     print(path)

    print("Run Complete")





