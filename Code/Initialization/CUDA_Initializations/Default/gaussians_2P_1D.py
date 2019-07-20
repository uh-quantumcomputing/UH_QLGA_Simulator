def get_CUDA(vectorSize = 10, momentums = [0.,0.], shifts = [0.35,0.65], sigmas = [0.05, 0.05] , **kwargs):
    if len(shifts)%2==1:
        print "Must have even number of wavefunctions (pairs). Quitting..."
        quit()
    if (len(momentums)!= len(shifts)) or (len(momentums)!= len(sigmas)) or (len(shifts)!= len(sigmas)):
        print "Input arrays must have the same length. Quitting..."
        quit()
    variableString = """"""
    for i in xrange(len(shifts)):
        variableString += """double sigma""" + str(i) + """ = """ + str(sigmas[i]) + r"""*Qx/2.;
        double shift""" + str(i) + """ = """ + str(shifts[i]) + r"""*Qx/2.;
        double p""" + str(i) + """ = """ + str(momentums[i]) + """;"""
    funcString = "norm*("
    for i in xrange(0,len(shifts)-1,2):
        if i>0:
            funcString += "+"
        funcString += """make_odd_gaussian(floor((double)alpha/2.),floor((double)beta/2.), sigma""" + str(i) + """, sigma""" + str(i+1) + """, shift""" + str(i) + """, shift""" + str(i+1) + """, p""" + str(i) + """, p""" + str(i+1) + """, (double)Qx/2.)"""

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
        double numGaussPairs = ''' + str(float(len(shifts))) + r''';
        double norm = (1./(sqrt(2.)*numGaussPairs));
        ''' + variableString + r'''
        int n;
        index_pair iToNum = index_to_number(Qx, x + deviceNum*local_xSize);
        int alpha = iToNum.index0;
        int beta = iToNum.index1;
        dcmplx i(0.,1.);

        for(n=0; n < vectorSize; n++){
          QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] =''' + funcString + r''');
        }
    }
'''