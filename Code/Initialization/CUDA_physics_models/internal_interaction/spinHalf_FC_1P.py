def get_CUDA(dimensions, vectorSize, G0 = 1., G1 = 0.5, MU = -1., scaling = 1., **kwargs):
	return r'''
		__global__ void internal(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{   
	    int xSize = lattice[0]*gpu_params[2];
	    int ySize = lattice[2];
	    int zSize = lattice[4];
	    int vectorSize = ''' + str(int(vectorSize)) + r''';
	  	int spinComps = ''' + str(int(vectorSize/2)) + r''';
	    dcmplx g0 = ''' + str(G0) + r''';
	    dcmplx g1 = ''' + str(G1) + r''';
	    dcmplx mu2 =''' + str(MU) + r''';
	    dcmplx scaling = ''' + str(scaling) + r''';
	    int x = blockIdx.x * blockDim.x + threadIdx.x;
	    int y = blockIdx.y * blockDim.y + threadIdx.y;
	    int z = blockIdx.z * blockDim.z + threadIdx.z;
	    int n, j, nn;
	    dcmplx k(1.,0.);
	    dcmplx A(scaling.real()/256., 0.); 
	    dcmplx mu = Mul(mu2, Mul(k, k));
	    dcmplx tau = Mul(A, A);

	    dcmplx NLPot = 0.;
	    dcmplx epsilon(1./sqrt(2.), 0.);
	    double delta = 0.0000000000000001;
	    dcmplx root2(sqrt(2.), 0);
	    dcmplx root3(sqrt(3.), 0);
	    dcmplx root6(sqrt(6.), 0.);
	    dcmplx i(0.,1./''' + str(int(4*dimensions)) + r'''.); // trotterization with N = 4*dims
	    dcmplx psi[''' + str(int(vectorSize)) + r'''];
	    dcmplx phi[''' + str(int(vectorSize/2)) + r'''];  

	    for (n=0; n<vectorSize; n=n+1){
        psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
        }
	    for(nn = 0; nn < spinComps; nn++){
	        phi[nn] = psi[2*nn] + psi[2*nn+1];
	        }
	    dcmplx rhoMinus = Mul(phi[0], conj(phi[0]));
	    dcmplx rhoPlus = Mul(phi[1], conj(phi[1]));
	    dcmplx NLPotPlus = exp(Mul3(-i, real(Mul(rhoPlus,g1)-Mul(rhoMinus,g0) + mu), tau));
	    dcmplx NLPotMinus = exp(Mul3(-i, real(Mul(rhoMinus,g1)-Mul(rhoPlus,g0) + mu), tau));

	    /*Apply operator*/

	    //Minus field
	    QField[0+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=Mul(NLPotMinus,QField[0+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
	    QField[1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=Mul(NLPotMinus,QField[1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	    //Plus field
	    QField[2+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=Mul(NLPotPlus,QField[2+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
	    QField[3+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=Mul(NLPotPlus,QField[3+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
  
    }
	'''