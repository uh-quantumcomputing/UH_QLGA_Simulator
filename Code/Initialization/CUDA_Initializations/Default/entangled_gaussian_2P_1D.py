def get_CUDA(vectorSize = 10, sigma1 = 0.1, sigma2 = 0.1, shift1 = 0.25, shift2 = 0.75, sigma3 = 0.1, sigma4 = 0.1, shift3 = 0.25, shift4 = 0.75, p1 = 0., p2 = 0., p3 = 0., p4 = 0.):
  return r'''
    __device__ double gaussian(double x, double sigma, double shift){
        double arg = ((x-shift)/(sqrt(2.)*sigma));
        return exp(-arg*arg)/sqrt(sigma*sqrt(pi));
    }
    
    __device__ dcmplx make_odd_gaussian(double x1, double x2, double sigma1, double sigma2,double shift1, double shift2, double p1, double p2, double L){
        dcmplx i(0.,1.);
        dcmplx kick1 = exp(i*(2.*pi*p1)*(-x1/L));
        dcmplx kick2 = exp(i*(2.*pi*p2)*(-x1/L));
        dcmplx kick3 = exp(i*(2.*pi*p1)*(-x2/L));
        dcmplx kick4 = exp(i*(2.*pi*p2)*(-x2/L));
        return (1./sqrt(2.))*(gaussian(x1,sigma1,shift1)*kick1*gaussian(x2,sigma2,shift2)*kick4-gaussian(x2,sigma1,shift1)*kick3*gaussian(x1,sigma2,shift2)*kick2);
    }

    __global__ void initialize(dcmplx *QField, int* lattice, int* gpu_params){
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        int z = blockIdx.z * blockDim.z + threadIdx.z;
        int deviceNum = gpu_params[0];
        int numGPUs = gpu_params[2];
        int local_xSize = lattice[0];
        int xSize = lattice[0]*numGPUs;
        int ySize = lattice[2];
        int zSize = lattice[4];
        int vectorSize = ''' + str(int(vectorSize)) + r''';
        int Qx = lattice[6];
        double sigma1 = ''' + str(sigma1) + r'''*Qx/2.;
        double sigma2 = ''' + str(sigma2) + r'''*Qx/2.;
        double shift1 = ''' + str(shift1) + r'''*Qx/2.;
        double shift2 = ''' + str(shift2) + r'''*Qx/2.;
        double sigma3 = ''' + str(sigma3) + r'''*Qx/2.;
        double sigma4 = ''' + str(sigma4) + r'''*Qx/2.;
        double shift3 = ''' + str(shift3) + r'''*Qx/2.;
        double shift4 = ''' + str(shift4) + r'''*Qx/2.;
        double p1 = ''' + str(p1) + r''';
        double p2 = ''' + str(p2) + r''';
        double p3 = ''' + str(p3) + r''';
        double p4 = ''' + str(p4) + r''';
        int n;
        index_pair iToNum = index_to_number(Qx, x + deviceNum*local_xSize);
        int alpha = iToNum.index0;
        int beta = iToNum.index1;
        dcmplx i(0.,1.);

        for(n=0; n < vectorSize; n++){
          QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = (1./sqrt(2.))*0.5*(make_odd_gaussian(floor((double)alpha/2.),floor((double)beta/2.), sigma1, sigma2, shift1, shift2, p1, p2, (double)Qx/2.)+make_odd_gaussian(floor((double)alpha/2.),floor((double)beta/2.), sigma3, sigma4, shift3, shift4, p3, p4, (double)Qx/2.));
        }
    }
'''