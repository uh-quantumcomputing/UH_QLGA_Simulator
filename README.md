# Quantum Lattice Gas Algorithm Simulator 

NOTE: This project is still in a work-in-progress state. 

This software aims to simulate quantum lattice gas algorithms using graphics processing units. 

We have included copies of PhD dissertations from Steven Smith and Jasper Taylor in the References folder. These can be used as references for both the modeling methodology and inplementation. 

## Prerequisites
Hardware:  
* [CUDA capable GPU](https://developer.nvidia.com/cuda-zone) <br/>

Main software:  <br/>

* [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
* [Python](https://www.python.org/)
* [ffmpeg](https://ffmpeg.org/) <br/>

Python Packages: <br/>

* [NumPy](https://numpy.org/)
* [PyCUDA](https://documen.tician.de/pycuda/)
* [Mayavi](https://docs.enthought.com/mayavi/mayavi/)
* [moviePy](https://pypi.org/project/moviepy/)
* [Matplotlib](https://matplotlib.org/)
* [Python-tk](https://wiki.python.org/moin/TkInter)
* [PyQT](https://wiki.python.org/moin/PyQt)
* [KIVY](https://kivy.org/)

## Getting started

After downloading all the packages, look in the base directory you will find two files "Master.py" and "SimulationMaster.py". These are the main files that will need to be used to run simulations. "SimulationMaster.py" offers more flexibility, such as setting use of specific GPUs or setting up loops to run multiple simulations in succession. "Master.py" is an early version of a simple GUI, to facilitate use by inexperienced coders. Be aware that you may have to use "SimulationMaster.py" first, to generate .pyc files, before using "Master.py" GUI. 

## Built-in Models

Here is a list of some of the built-in models included in the current version of the software. The code is modular, so that a user can also add their own models.

Make sure that the particle number indicator (1P,2P) at the end of the model name matches the number of particles you are simulating. Options described using the format: 
<br/> *Function*(Keyword options) <br/> 

Model and intial condition keywords can both be input into the "Experiment Keywords" field using python dictionary syntax, i.e. {'key':value}. Function based methods, such as *Function_1P* take lists created by the user (see Additional info). Some methods may works across different models, but no guarentee.

### Schrödinger equation

**Models** <br/>  
*Free_1P*(None) <br/>
*Free_2P*(None) <br/>

**Initial Conditions** <br/> 

*Function_1P*(func, px, py, pz) <br/>
* func is a user specified function written in CUDA/C++ syntax using variables vectorSize, X, Y, Z, Lx, Ly, and Lz.
* px, py, & pz are integers that give momentum kicks in the x, y, & z directions respectively. <br/>

*Gaussians_1D_1P*(momentums, shifts, sigmas) <br/>
* momentums are a list of N momentum kicks for N gaussians.
* shifts are a list of N discribution centers for N gaussians. 
* sigmas are a list of N widths for N gaussians. <br/>

*Gaussians_1D_2P*(momentums, shifts, sigmas) <br/>
* momentums are a list of N momentum kicks for N gaussians.
* shifts are a list of N discribution centers for N gaussians. 
* sigmas are a list of N widths for N gaussians. <br/>

*Gaussians_2D_2P*(momentums_x, momentums_y, shifts_x, shifts_y, sigmas_x, sigmas_y) <br/>
* momentums_x/y are a list of N momentum kicks along x/y for N gaussians.
* shifts_x/y are a list of N discribution centers along _x/y for N gaussians. 
* sigmas_x/y are a list of N widths along x/y for N gaussians. <br/>

**Visualization** <br/> 

*Density_1D_1P*(None) <br/>
*Density_1D_2P*(save_density, save_entanglement, save_expectation_value, exp_range) 
* save_density saves the density if set to True.
* save_entanglement saves the entanglement if set to True. 
* save_expectation value saves the expectation value over the range exp_range=[lower bound, upper bound]if set to True. <br/>

*Density_Phase_2D_1P*None) <br/>

*Projection_1D_2P*(save_density, save_entanglement, save_expectation_value, exp_range) 
* save_density saves the density if set to True.
* save_entanglement saves the entanglement if set to True. 
* save_expectation value saves the expectation value over the range exp_range=[lower bound, upper bound]if set to True. <br/>

*Projection_2D_2P*(save_density, save_entanglement, save_expectation_value, exp_range) 
* save_density saves the density if set to True.
* save_entanglement saves the entanglement if set to True. 
* save_expectation value saves the expectation value over the range exp_range=[lower bound, upper bound]if set to True. <br/>

### Fermi condensate

**Models** <br/>  
*SpinHalf_FC_1P*(G0,G1,scaling) <br/>
* G0 & G1 are the effective coupling constants in the GP-equation
* scaling rescales the dimensions of the lattice <br/>

**Initial Conditions** <br/> 

*Function_1P*(func, px, py, pz) <br/>
* func is a user specified function written in CUDA/C++ syntax using variables vectorSize, X, Y, Z, Lx, Ly, and Lz.
* px, py, & pz are integers that give momentum kicks in the x, y, & z directions respectively. <br/>

*Skyrmion_Quandrupole*(G0, G1, K, scaling, p1x, p1y) <br/>
* G0 & G1 are the effective coupling constants in the GP-equation.
* K/scaling scale the size of the Skyrmion/lattice respectively. 
* p1x & p1y give momentum kicks in the x & y directions respectively. <br/>

*Skyrmion_Octopole*(G0, G1, K, scaling, p1x, p1y) <br/>
* G0 & G1 are the effective coupling constants in the GP-equation.
* K/scaling scale the size of the Skyrmion/lattice respectively. 
* p1x & p1y give momentum kicks in the x & y directions respectively. <br/>


**Visualization** <br/> 

*3D_FC*(None) <br/>


### Spin-2 BEC

**Models** <br/>  
*Spin2_BEC_1P*(G0, G1, G2, scaling) <br/>
* G0, G1, & G2 are the effective coupling constants in the GP-equation.
* scaling rescales the dimensions of the lattice. <br/>

**Initial Conditions** <br/> 

*Function_1P*(func, px, py, pz) <br/>
* func is a user specified function written in CUDA/C++ syntax using variables vectorSize, X, Y, Z, Lx, Ly, and Lz.
* px, py, & pz are integers that give momentum kicks in the x, y, & z directions respectively. <br/>

*Bright_Solitons*(center1, center2, G0, G1, G2, scaling, state_1, state_2, p1x, p2x) <br/>
* center1 & center2 give locations of centers for soliton 1/2 respectively.
* G0, G1, & G2 are the effective coupling constants in the GP-equation
* p1x & p2x give momentum kicks for soliton 1/2 respectively. 
* state_1, state_2 can be integers between 0-16, where 1-16 are different bright soliton solutions and 0 is no soliton.
* scaling rescales the dimensions of the lattice. <br/>

*Dark_Soliton*(center1, center2, G0, G1, G2, scaling, state_1, p1x) <br/>
* center1 & center2 give locations of centers for soliton kinks respectively.
* G0, G1, & G2 are the effective coupling constants in the GP-equation
* p1x give momentum kick for the dark soliton. 
* state_1 can be integers between 0-16, where 1-16 are different dark soliton solutions and 0 is no soliton.
* scaling rescales the dimensions of the lattice. <br/>

*Quadrupoles*(G0, G1, G2, scaling, solution1, solution2, p1x, p1y, p1z, p2x, p2y, p2z, lr_shift1, lr_shift2, ud_shift1, ud_shift2, separation, orientation1, orientation2) <br/>
* G0, G1, & G2 are the effective coupling constants in the GP-equation.
* p1x & p2x (or y,z) give momentum kicks for quadrupole 1/2 respectively. 
* lr_shift and ud_shift shifts the quadrupoles left/right and up/down.
* solution1/2 can be integers between 0-8, where 1-16 are different Pade approximates and 0 is no quadrupole.
* orientation can be either 'x','y', or 'z' and speficies the orientation.
* scaling rescales the dimensions of the lattice. <br/>

**Visualization** <br/> 

*1D_BEC*(None) <br/>
*2D_BEC*(None) <br/>
*3D_BEC*(None) <br/>

### User generated

**Models** <br/>  
*Function_1P*(cond_list,func_list) <br/>
*Function_2P*(cond_list,func_list) <br/>
* cond_list is a list of conditions that determine which function to use in func_list (like a piecewise function).
* func_list is a list of CUDA/C++ syntax functions using variables vectorSize, X, Y, Z, T, Lx, Ly, and Lz.

## Additional info

External potentials and measurements are also utilized when running "SimulationMaster.py". To do this you change the "POTENTIAL" & "MEASUREMENT" parameters to match the name of the file of the desired method located in "/Code/Initializations/CUDA_physics_models/". Then change the kwargs to desired values, found in the method file.


## Authors

* **Steven Smith**
* **Jasper Taylor**
* **Anthony Gasbarro** 

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* This work could not have been done without the guidance of [Jeffrey Yepez](https://www.phys.hawaii.edu/~yepez/).
