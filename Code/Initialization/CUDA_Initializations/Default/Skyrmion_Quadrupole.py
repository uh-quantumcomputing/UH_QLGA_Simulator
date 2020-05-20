def get_CUDA(vectorSize = 4, separation = 1./4., G0 = -.01, G1 = .01, K = 1., scaling = 1., p1x = 0., p1y = 0.):
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
  int spinComps = 2;
  int vectorSize = 4;
  double x_center = __dmul_rn(double(xSize), 1./2.);
  double y_center = __dmul_rn(double(ySize), 1./2.);
  double z_center = __dmul_rn(double(zSize), 1./2.);
  double this_x = double(x+deviceNum*lattice[0]) - x_center;
  double this_y = double(y) - y_center;
  double this_z = double(z) - z_center;
  dcmplx scaling = ''' + str(scaling) + r''';
  double K = ''' + str(K) + r''';
  double A = scaling.real();
  double AA = __dmul_rn(A, A); 
  dcmplx ii(0.,1.);
  int square_size = 3;
  double separation = __dmul_rn(''' + str(separation) + r''', xSize);
  double p1x = ''' + str(p1x) + r''';
  double p1y = ''' + str(p1y) + r''';
  x_center = 0.;
  y_center = 0.;
  z_center = 0.;
  dcmplx half(.5, 0.); 
  for (int mf = 0; mf < spinComps; mf ++){
    dcmplx result(1., 0.);
    if (mf==1) {
      for (int i = 0; i < square_size; i ++){
        for (int j = 0; j < square_size; j ++){
          for (int k = 0; k < square_size; k ++){
            int adjusted_i = i - square_size/2;
            int adjusted_j = j - square_size/2;
            int adjusted_k = k - square_size/2;
            double x_center_new = x_center + double(adjusted_i * (xSize));
            double y_center_new = y_center + double(adjusted_j * (ySize));
            double z_center_new = z_center + double(adjusted_k * (zSize));
            for (int q = -1; q <2 ; q +=2){
              for (int p = -1; p <2 ; p +=2){
              double x_center_skyrmion = x_center_new + double(q)*separation;
              double y_center_skyrmion = y_center_new + double(p)*separation;
              double z_center_skyrmion = z_center_new;
              double r_squared = get_r_squared_3d(this_x, this_y, this_z, x_center_skyrmion, y_center_skyrmion, z_center_skyrmion, AA); 
              double r = sqrt(r_squared);
              dcmplx field = 1.;
              if (r>0){
                field = cos(__dmul_rn(pi,tanh(__dmul_rn(K,r))))-Mul3(ii,sin(__dmul_rn(pi,tanh(__dmul_rn(K,r)))),__ddiv_rn(__dmul_rn(A,(this_z-z_center_skyrmion)),r));
              }
              result = Mul(result,field);
              }
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
        QField[2*mf+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(half,  result);
        QField[2*mf+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(half, result);
    } else { 
        result = 0.;
        for (int i = 0; i < square_size; i ++){
          for (int j = 0; j < square_size; j ++){
            for (int k = 0; k < square_size; k ++){
              int adjusted_i = i - square_size/2;
              int adjusted_j = j - square_size/2;
              int adjusted_k = k - square_size/2;
              double x_center_new = x_center + double(adjusted_i * (xSize));
              double y_center_new = y_center + double(adjusted_j * (ySize));
              double z_center_new = z_center + double(adjusted_k * (zSize));
              for (int q = -1; q <2 ; q +=2){
                for (int p = -1; p <2 ; p +=2){
                double x_center_skyrmion = x_center_new + double(q)*separation;
                double y_center_skyrmion = y_center_new + double(p)*separation;
                double z_center_skyrmion = z_center_new;
                double r_squared = get_r_squared_3d(this_x, this_y, this_z, x_center_skyrmion, y_center_skyrmion, z_center_skyrmion, AA);
                double r = sqrt(r_squared);
                dcmplx field = 0.;
                if (r>0){
                  field = Mul3(-ii,sin(__dmul_rn(pi,tanh(__dmul_rn(K,r)))),sqrt(1.-__dmul_rn(__dmul_rn(AA,(this_z-z_center_skyrmion)),__ddiv_rn((this_z-z_center_skyrmion),(r_squared)))));
                }
                double local_phase = __dmul_rn(1.0, phase(dcmplx(this_x -  x_center_skyrmion, this_y - y_center_skyrmion)));
                double phase_sign = double(q*p);
                result +=  Mul(field, exp(dcmplx(0., __dmul_rn(phase_sign, local_phase))));
                }
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
        QField[2*mf+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(half, result);
        QField[2*mf+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(half, result);

      }

    }
}'''