def get_CUDA(dimensions, vectorSize, func_string = "sin(X)"):
	# Add in scale and shift along X1
	potentialString = "dcmplx potentialU = exp(-Mul(i,(dcmplx)(" + func_string + ")));"
	return '''__global__ void internal(dcmplx *QField, dcmplx *QFieldCopy, int* lattice, int* gpu_params){
	int time_step = lattice[14];
	int deviceNum = gpu_params[0];
	int numGPUs = gpu_params[2];
	int xSize = lattice[0]*numGPUs;
	int ySize = lattice[2];
	int zSize = lattice[4];
	int vectorSize = ''' + str(vectorSize) + ''';
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int n;
	dcmplx i(0.,1.);
	dcmplx field(0.,0.);
	double X = (double)(x);
	double Y = (double)(y);
	double Z = (double)(z);
	double T = (double)time_step;
	double Lx = (double) xSize;
	double Ly = (double) ySize;
	double Lz = (double) zSize;
	''' + potentialString + '''
	
	for (n=0; n<vectorSize; n=n+1){
		field = Mul(potentialU,QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
		QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
		QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
	}  
}'''
