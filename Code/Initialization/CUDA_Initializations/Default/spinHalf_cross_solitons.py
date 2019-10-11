def get_CUDA(vectorSize = 4, G0 = 1., G1 = 0.5, MU = -1., scaling = 1., momentums = [0.,0.], positions = [0.5,0.5], dim2=0, coeffs=[1.,1.]):
  coeff = -2./G0
  pos_mom_string = '''dcmplx Cm =  (''' + str(coeffs[0].real) + r'''+','+''' + str(coeffs[0].imag) + r''');'''+'''
      dcmplx Cp =  (''' + str(coeffs[1].real) + r'''+','+''' + str(coeffs[1].imag) + r''');'''
  for i in xrange(len(positions)):
    pos_mom_string += ''' double p''' + str(i) + r''' = ''' + str(momentums[i]) + r''';
    double cen''' + str(i) + r''' = ''' + str(positions[i]) + r'''*double(xSize);
    '''
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
  int vectorSize = ''' + str(vectorSize) + r''';
  double this_x = double(x+deviceNum*lattice[0]);
  double this_y = double(y);
  dcmplx scaling = ''' + str(scaling) + r''';
  double k = sqrt(-1.*(''' + str(MU) + r'''));
  double A = scaling.real()/256.;
  double AA = __dmul_rn(A, A); 
  double numSols = ''' + str(float(1+dim2)) + ''';
  double norm = (1./numSols);
  double coeff = ''' + str(coeff) + r''';
  ''' + pos_mom_string + r''';
  dcmplx fieldM = norm*(0.5*k*Cm/cosh(k*A*(this_x-cen0))+0.5*''' + str(dim2) + r'''*k*Cm/cosh(k*A*(this_y-cen1)));
  dcmplx fieldP = norm*(0.5*k*Cp/cosh(k*A*(this_x-cen0))+0.5*''' + str(dim2) + r'''*k*Cp/cosh(k*A*(this_y-cen1)));
  dcmplx phase1 = expC(dcmplx(0., 2.*pi*p0*this_x/(double)xSize));
  dcmplx phase2 = expC(dcmplx(0., 2.*pi*p1*this_y/(double)ySize));
  QField[z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = fieldM;
  QField[1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = fieldM;
  QField[2+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = fieldP;
  QField[3+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = fieldP;

}'''