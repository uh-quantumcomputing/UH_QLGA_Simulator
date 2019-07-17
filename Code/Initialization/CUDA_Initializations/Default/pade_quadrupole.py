import stationary_states.pade as pade

def get_CUDA(vectorSize = 10, separation = 1./4., G0 = -.7, G1 = .5, G2 = .6, MU = 1., scaling = 30, solution_number = 1, p1x = 0., p1y = 0.):
  a1, a2, b1 = pade.get_pade_quad(G0, G1, G2, MU, solution_number)
  mf_array = pade.get_mfs(solution_number)
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
  int is_mf_empty[5] = {''' + str(int(mf_array[0])) + r''',''' + str(int(mf_array[1])) + r''', ''' + str(int(mf_array[2])) + r''', ''' + str(int(mf_array[3])) + r''', ''' + str(int(mf_array[4])) + r'''}; 
  double x_center = __dmul_rn(double(xSize), 1./2.);
  double y_center = __dmul_rn(double(ySize), 1./2.);
  double this_x = double(x+deviceNum*lattice[0]) - x_center;
  double this_y = double(y) - y_center;
  dcmplx scaling = ''' + str(scaling) + r''';
  double A = scaling.real()/double(xSize);
  double AA = __dmul_rn(A, A); 
  int square_size = 9;
  double a1 = ''' + str(a1) + r''';
  double a2 = ''' + str(a2) + r''';
  double b1 = ''' + str(b1) + r''';
  double separation = __dmul_rn(''' + str(separation) + r''', xSize);
  double p1x = ''' + str(p1x) + r''';
  double p1y = ''' + str(p1y) + r''';
  x_center = 0.;
  y_center = 0.;
  dcmplx half(.5, 0.); 
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
      }
        //printf("mf = %d", mf);
        double arg1 = __dmul_rn(__dmul_rn(2., pi), __dmul_rn(p1x, this_x));
        double arg2 = __dmul_rn(__dmul_rn(2., pi), __dmul_rn(p1y, this_y));
        dcmplx phase_velocity1 = exp(dcmplx(0., arg1/(double(xSize))));
        dcmplx phase_velocity2 = exp(dcmplx(0., arg2/(double(ySize))));
        dcmplx phase_velocity = Mul(phase_velocity1, phase_velocity2);
        result = Mul(phase_velocity, result);
        QField[2*mf+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul3(half, (dcmplx) is_mf_empty[mf], result);
        QField[2*mf+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul3(half, (dcmplx) is_mf_empty[mf], result);
      }
    } 
}'''