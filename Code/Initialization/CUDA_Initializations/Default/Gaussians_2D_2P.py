def get_CUDA(vectorSize = 2, momentums_x = [0.,0.], shifts_x = [0.35,0.65], sigmas_x = [0.05, 0.05], momentums_y = [0.,0.], shifts_y = [0.35,0.65], sigmas_y = [0.05, 0.05], **kwargs):
    if len(shifts_x)%4==1:
        print "Must have even number of wavefunctions (pairs). Quitting..."
        quit()
    if (len(momentums_x)!= len(shifts_x)) or (len(momentums_x)!= len(sigmas_x)) or (len(shifts_x)!= len(sigmas_x)):
        print "Input arrays must have the same length. Quitting..."
        quit()
    variableString = """"""
    for i in xrange(len(shifts_x)):
        variableString += """double sigmaX""" + str(i) + """ = """ + str(sigmas_x[i]) + r"""*Qx/2.;
        double shiftX""" + str(i) + """ = """ + str(shifts_x[i]) + r"""*Qx/2.;
        double pX""" + str(i) + """ = """ + str(momentums_x[i]) + """;"""
    for i in xrange(len(shifts_y)):
        variableString += """double sigmaY""" + str(i) + """ = """ + str(sigmas_y[i]) + r"""*Ly;
        double shiftY""" + str(i) + """ = """ + str(shifts_y[i]) + r"""*Ly;
        double pY""" + str(i) + """ = """ + str(momentums_y[i]) + """;"""
    funcString = "norm*("
    for i in xrange(0,len(shifts_x)-1,2):
        if i>0:
            funcString += "+"
        funcString += """make_odd_gaussian_2d(floor((double)alpha_x/2.),floor((double)beta_x/2.), sigmaX""" + str(i) + """, sigmaX""" + str(i+1) + """, shiftX""" + str(i) + """, shiftX""" + str(i+1) + """, pX""" + str(i) + """, pX""" + str(i+1) + """, (double)Qx/2., floor((double)alpha_y),floor((double)beta_y), sigmaY""" + str(i) + """, sigmaY""" + str(i+1) + """, shiftY""" + str(i) + """, shiftY""" + str(i+1) + """, pY""" + str(i) + """, pY""" + str(i+1) + """, (double)Ly)"""

    return r'''
    __device__ double gaussian(double x, double sigma, double shift){
        double arg = ((x-shift)/(sqrt(2.)*sigma));
        return exp(-arg*arg)/sqrt(sigma*sqrt(pi));
    }
    
    __device__ dcmplx make_odd_gaussian_2d(double x1, double x2, double sigmaX1, double sigmaX2, double shiftX1, double shiftX2, double pX1, double pX2, double Lx, double y1, double y2, double sigmaY1, double sigmaY2, double shiftY1, double shiftY2, double pY1, double pY2, double Ly){
        dcmplx i(0.,1.);
        dcmplx kick1X = exp(i*(2.*pi*pX1)*(-x1/Lx));
        dcmplx kick2X = exp(i*(2.*pi*pX2)*(-x1/Lx));
        dcmplx kick3X = exp(i*(2.*pi*pX1)*(-x2/Lx));
        dcmplx kick4X = exp(i*(2.*pi*pX2)*(-x2/Lx));
        dcmplx kick1Y = exp(i*(2.*pi*pY1)*(-y1/Ly));
        dcmplx kick2Y = exp(i*(2.*pi*pY2)*(-y1/Ly));
        dcmplx kick3Y = exp(i*(2.*pi*pY1)*(-y2/Ly));
        dcmplx kick4Y = exp(i*(2.*pi*pY2)*(-y2/Ly));
        return (1./sqrt(2.))*(gaussian(x1,sigmaX1,shiftX1)*kick1X*gaussian(x2,sigmaX2,shiftX2)*kick4X*gaussian(y1,sigmaY1,shiftY1)*kick1Y*gaussian(y2,sigmaY2,shiftY2)*kick4Y-gaussian(x2,sigmaX1,shiftX1)*kick3X*gaussian(x1,sigmaX2,shiftX2)*kick2X*gaussian(y2,sigmaY1,shiftY1)*kick3Y*gaussian(y1,sigmaY2,shiftY2)*kick2Y);
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
        int Ly = lattice[8]/2;
        double numGaussPairs = ''' + str(float(len(shifts_x))) + r''';
        double norm = (1./(sqrt(2.)*numGaussPairs));
        ''' + variableString + r'''
        int n;
        index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
        int alpha = iToNum.index0;
        int beta = iToNum.index1;
        index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
        int alpha_x = alphaBeta2d.index0;
        int alpha_y = alphaBeta2d.index1;
        int beta_x = alphaBeta2d.index2;
        int beta_y = alphaBeta2d.index3;
        dcmplx i(0.,1.);

        double num = 0.;
        //if (alpha_x==0 && alpha_y==0 && beta_y==0 && beta_x==2){num=1.;}
        //if (alpha_x==0 && alpha_y==1 && beta_y==1 && beta_x==3){num=1.;}
        for(n=0; n < vectorSize; n++){
          QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = ''' + funcString + r''');
        }
    }
'''