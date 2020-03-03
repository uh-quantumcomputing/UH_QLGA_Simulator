def get_CUDA(vectorSize = 10, momentums = [0.,0.], shifts = [0.35,0.65], sigmas = [0.05, 0.05] , **kwargs):
    if (len(momentums)!= len(shifts)) or (len(momentums)!= len(sigmas)) or (len(shifts)!= len(sigmas)):
        print "Input arrays must have the same length. Quitting..."
        quit()
    variableString = """"""
    for i in xrange(len(shifts)):
        variableString += """
        double sigma""" + str(i) + """ = """ + str(sigmas[i]) + r"""*(double)xSize;
        double shift""" + str(i) + """ = """ + str(shifts[i]) + r"""*(double)xSize;
        double p""" + str(i) + """ = """ + str(momentums[i]) + """;
        dcmplx phaseKick""" + str(i) + """ = exp(i*(-p""" + str(i) + """*2.*pi)*(this_x)/(double)(xSize));"""
    fieldString = "dcmplx field = norm*(Mul(phaseKick0, gaussian(this_x,sigma0,shift0))"
    for i in xrange(1,len(shifts)):
        fieldString += """+Mul(phaseKick""" + str(i) + """, gaussian(this_x,sigma""" + str(i) + """,shift""" + str(i) + """))"""
    return r'''
    __device__ double gaussian(double x, double sigma, double shift){
        double arg = ((x-shift)/(sqrt(2.)*sigma));
        return exp(-arg*arg)/sqrt(sigma*sqrt(pi));
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
        double this_x = (double)(x+deviceNum*local_xSize);
        dcmplx i(0.,1.);
        int vectorSize = ''' + str(int(vectorSize)) + r''';
        double numGaussPairs = ''' + str(float(len(shifts))) + ''';
        double norm = (1./numGaussPairs);
        ''' + variableString + '''
        int n;
        ''' + fieldString + ''');

        for(n=0; n < vectorSize; n++){
          QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field/2.;
        }
    }
'''