def get_CUDA(dimensions, vectorSize, cond_list = ["true"], func_list = ["sin(X1/Lx)+sin(X2/Lx)"], **kwargs):
	# Add in scale and shift along X1
	potentialString = """dcmplx potentialU = 1.;
		if (""" + cond_list[0] + """){potentialU = exp(-Mul(i/4.,(dcmplx)(""" + func_list[0] + """)));}"""
	if len(cond_list)>1:
		for i in xrange(1,len(cond_list)):
			potentialString += """
			else if(""" + cond_list[i] + """){potentialU = exp(-Mul(i/4.,(dcmplx)(""" + func_list[i] + """)));}"""
	return '''__global__ void external(dcmplx *QField, dcmplx *QFieldCopy, int* lattice, int* gpu_params){
	int time_step = lattice[14];
	int deviceNum = gpu_params[0];
	int numGPUs = gpu_params[2];
	int xSize = lattice[0]*numGPUs;
	int local_xSize = lattice[0];
	int ySize = lattice[2];
	int zSize = lattice[4];
	int Qx = lattice[6];
	int Qy = lattice[8];
	int Qz = lattice[10];
	int vectorSize = ''' + str(vectorSize) + ''';
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int n;
	dcmplx i(0.,1.);
	dcmplx field(0.,0.);
	index_pair iToNumX = index_to_number(Qx*Qy/2, x + deviceNum*local_xSize);
	int alpha = iToNumX.index0;
	int beta = iToNumX.index1;
	index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	int alpha_x = alphaBeta2d.index0;
	int alpha_y = alphaBeta2d.index1;
	int beta_x = alphaBeta2d.index2;
	int beta_y = alphaBeta2d.index3;
	double X1 = (double)(alpha_x/2);
	double X2 = (double)(beta_x/2);
	double Y1 = (double)(alpha_y);
	double Y2 = (double)(beta_y);
	double T = (double)time_step;
	double Lx = (double) Qx/2;
	double Ly = (double) Qy/2;
	double Lz = (double) Qz/2;
	''' + potentialString + '''
	
	for (n=0; n<vectorSize; n=n+1){
		field = Mul(potentialU,QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
		QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
		QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
	}  
}'''
