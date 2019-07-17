import stationary_states.pade as pade
import numpy as np

def get_end_q_pole(y_shift, z_shift):
      return r'''
        x_shift = 0;
        y_shift = __double2int_rn(__dmul_rn(''' + str(y_shift) + r''', ySize));
        z_shift = __double2int_rn(__dmul_rn(''' + str(z_shift) + r''', zSize));
        double arg1 = __dmul_rn(__dmul_rn(2., pi), __dmul_rn(px, this_x));
        double arg2 = __dmul_rn(__dmul_rn(2., pi), __dmul_rn(py, this_y));
        double arg3 = __dmul_rn(__dmul_rn(2., pi), __dmul_rn(pz, this_z));
        dcmplx phase_velocity1 = exp(dcmplx(0., arg1/(double(xSize))));
        dcmplx phase_velocity2 = exp(dcmplx(0., arg2/(double(ySize))));
        dcmplx phase_velocity3 = exp(dcmplx(0., arg2/(double(zSize))));
        dcmplx phase_velocity = Mul3(phase_velocity1, phase_velocity2, phase_velocity3);
        result = Mul(phase_velocity, result);
        QField[2*mf+((z + z_shift)%zSize)*vectorSize+((y + y_shift)%ySize)*vectorSize*zSize+((x + x_shift)%xSize)*zSize*ySize*vectorSize] = Mul3(half, (dcmplx) is_mf_empty[mf], result);
        QField[2*mf+1+((z + z_shift)%zSize)*vectorSize+((y + y_shift)%ySize)*vectorSize*zSize+((x + x_shift)%xSize)*zSize*ySize*vectorSize] = Mul3(half, (dcmplx) is_mf_empty[mf], result);
      }
    }'''


def get_preamble(scaling):
  return r''' 
  __global__ void initialize(dcmplx *QField, int* lattice, int* gpu_params){
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int deviceNum = gpu_params[0];
  int numGPUs = gpu_params[2];
  int xSize = lattice[0]*numGPUs;
  int ySize = lattice[2];
  int zSize = lattice[4];
  int spinComps = 5;
  int vectorSize = 10;
  int x_shift = 0;
  int y_shift = 0;
  int z_shift = 0;
  double x_center = __dmul_rn(double(xSize), 1./2.);
  double y_center = __dmul_rn(double(ySize), 1./2.);
  double z_center = __dmul_rn(double(zSize), 1./2.);
  double this_x = double(x+deviceNum*lattice[0]) - x_center;
  double this_y = double(y) - y_center;
  double this_z = double(z) - z_center;
  dcmplx scaling = ''' + str(scaling) + r''';
  double A = scaling.real()/256.;
  double AA = __dmul_rn(A, A); 
  int square_size = 9;
  x_center = 0.;
  y_center = 0.;
  z_center = 0.;
  dcmplx half(.5, 0.);
  double separation = 0.;
  double px = 0.;
  double py = 0.;
  double pz = 0.;
  double a1 = 0.;
  double a2 = 0.;
  double b1 = 0.; 
  int is_mf_empty[5] = {0, 0, 0, 0, 0};'''


def make_CUDA_mf_aray(mf_array):
  string =''''''
  for i in xrange(len(mf_array)):
    string = string + r'''is_mf_empty[''' + str(i) + r'''] = ''' + str(int(mf_array[i])) + r''';'''
  return string



def get_quadrupoles_in_z_orientation(a1, a2, b1, mf_array,  separation, px, py, pz):
  return r''' 
  separation = __dmul_rn(''' + str(separation) + r''', xSize);
  px = ''' + str(px) + r''';
  py = ''' + str(py) + r''';
  pz = ''' + str(pz) + r''';
  a1 = ''' + str(a1) + r''';
  a2 = ''' + str(a2) + r''';
  b1 = ''' + str(b1) + r'''; 
  ''' + make_CUDA_mf_aray(mf_array) + r'''
  for (int mf = 0; mf < spinComps; mf ++){
    dcmplx result(1., 0.);
    if (is_mf_empty[mf] > 0) {
      for (int i = 0; i < square_size; i ++){
        for (int j = 0; j < square_size; j ++){
          int adjusted_i = i - square_size/2;
          int adjusted_j = j - square_size/2;
          double x_center_new = x_center + double(adjusted_i * (xSize));
          double y_center_new = y_center + double(adjusted_j * (ySize));
          for (int q = -1; q <2 ; q +=2){
            for (int p = -1; p <2 ; p +=2){
            double x_center_pole = x_center_new + double(q)*separation;
            double y_center_pole = y_center_new + double(p)*separation;
            double r_squared = get_r_squared(this_x, this_y, x_center_pole, y_center_pole, AA); 
            double mag = sqrt((__dmul_rn(a1 , r_squared) + __dmul_rn(a2 , __dmul_rn(r_squared, r_squared)))
                            /(1. + __dmul_rn(b1 , r_squared) + __dmul_rn(a2 , __dmul_rn(r_squared, r_squared))));
            double local_phase = __dmul_rn(1.0, phase(dcmplx(this_x -  x_center_pole, this_y - y_center_pole)));
            double phase_sign = double(q*p);
            result =  Mul3(result, dcmplx(mag, 0.), exp(dcmplx(0., __dmul_rn(phase_sign, local_phase))));
          }
        }
      }
      }'''


def rotate_poles(string, orientation):
  if orientation == 'y':
    string = string.replace("this_y", "this_z")
    string = string.replace("y_center", "z_center")
    string = string.replace("ySize", "zSize")
  if orientation == 'x':
    string = string.replace("this_x", "this_z")
    string = string.replace("x_center", "z_center")
    string = string.replace("xSize", "zSize")
  return string

def get_q_poles(G0, G1, G2, MU, solution, orientation, separation, px, py, pz, y_shift, z_shift):
  a1, a2, b1 = pade.get_pade_quad(G0, G1, G2, MU, solution)
  mf_array = np.asarray(pade.get_mfs(solution), dtype = np.int_)
  z_poles = get_quadrupoles_in_z_orientation(a1, a2, b1, mf_array, separation, px, py, pz)
  rotated_poles = rotate_poles(z_poles, orientation)
  q_poles = rotated_poles + get_end_q_pole(y_shift, z_shift)
  return q_poles

def get_CUDA(vectorSize = 10, G0 = 1., G1 = .1, G2 = 1., MU = 1., scaling = 50., 
            solution1 = 1, solution2 = 0, p1x = 0., p1y = 0., p1z = 0., p2x = 0., p2y = 0., p2z = 0.,
            y_shift1 = 0., z_shift1 = 0., y_shift2 = 0., z_shift2 = 0.,
            separation1 = 1./4., separation2 = 1./4., orientation1 = 'z', orientation2 = 'z', **kwargs):
  preamble = get_preamble(scaling)
  first_q_poles = get_q_poles(G0, G1, G2, MU, solution1, orientation1, separation1, p1x, p1y, p1z, y_shift1, z_shift1)
  second_q_poles = get_q_poles(G0, G1, G2, MU, solution2, orientation2, separation2, p2x, p2y, p2z, y_shift2, z_shift2)
  return preamble+first_q_poles+second_q_poles+r'''}'''

