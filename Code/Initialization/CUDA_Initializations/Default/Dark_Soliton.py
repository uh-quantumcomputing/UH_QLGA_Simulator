import stationary_states.dark_solitons as ds
# k = sqrt(-params[3].real());
#def get_states(g0, g1, g2, state_num, c1 = 1./2., c2 = 1./2., **kwargs):

def get_CUDA(vector_size = 10, center1 = .25, center2 = .75, G0 = -.7, G1 = .5, G2 = .6, MU = 1., scaling = 30, state_1 = 1, state_2 = 0, p1x = 0., p2x = 0.):
  s1 = ds.get_states(G0, G1, G2, state_1, c1 = 1./2., c2 = 1./2.)
  s2 = ds.get_states(G0, G1, G2, state_2, c1 = 1./2., c2 = 1./2.)
  print(s1, s2)
  print(int(s1[0]))
  print str(center1)
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
  double state1[10] = {''' + str((s1[0])) + r''',''' + str((s1[1])) + r''', ''' + str((s1[2])) + r''', ''' + str((s1[3])) + r''', ''' + str((s1[4])) + r''', ''' +str((s1[5])) + r''',''' + str((s1[6])) + r''', ''' + str((s1[7])) + r''', ''' + str((s1[8])) + r''', ''' + str((s1[9])) + r'''};
  dcmplx scaling = ''' + str(scaling) + r''';
  double p1 = ''' + str(p1x) + r''';
  double p2 = ''' + str(p2x) + r''';
  double x1 = ''' + str(center1) + r''' * double(xSize);
  double x2 = ''' + str(center2) + r''' * double(xSize);
  double this_x = double(x+deviceNum*lattice[0]);
  double k = 1.; 
  double A = scaling.real()/double(xSize); 
  double arg = __dmul_rn(__dmul_rn(2., pi), __dmul_rn(p1, this_x));
  dcmplx phase = expC(dcmplx(0., arg/double(xSize)));
  for (int i = 0; i < spinComps; i++){
      if(this_x < double(xSize)/2.){
        double targ1 = __dmul_rn(__dmul_rn(k, A), this_x - x1);
        dcmplx tanh1 (tanh(targ1), 0);
        double c = __dmul_rn(__dmul_rn(0.5, k), __dmul_rn(state1[i + spinComps], sqrt(state1[i])));
        dcmplx constant (c, 0);
        QField[2*i+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(phase, Mul(constant, tanh1));
        QField[2*i+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(phase, Mul(constant, tanh1));
      }
      else {
        double targ2 = __dmul_rn(__dmul_rn(k, A), this_x - x2);
        dcmplx tanh2  (-tanh(targ2), 0.); 
        double c = __dmul_rn(__dmul_rn(0.5, k), __dmul_rn(state1[i + spinComps], sqrt(state1[i])));
        dcmplx constant (c, 0);
        QField[2*i+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(phase, Mul(constant, tanh2));
        QField[2*i+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(phase, Mul(constant, tanh2));
      }
  }
}
'''