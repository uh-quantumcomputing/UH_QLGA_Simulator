def get_CUDA(vectorSize = 10, func = "sin(X)", px = "0.", py = "0.", pz = "0.",  **kwargs):
	return '''__global__ void initialize(dcmplx *QField, int* lattice, int* gpu_params){
	int deviceNum = gpu_params[0];
	int numGPUs = gpu_params[2];
	int xSize = lattice[0]*numGPUs;
	int ySize = lattice[2];
	int zSize = lattice[4];
	int vectorSize = ''' + str(vectorSize) + ''';
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	double this_x = double(x+deviceNum*lattice[0]);
	int n;
	dcmplx i(0.,1.);
	double X = (double)(this_x);
	double Y = (double)(y);
	double Z = (double)(z);
	double Lx = (double) xSize;
	double Ly = (double) ySize;
	double Lz = (double) zSize;
	dcmplx phaseKickX = exp(i*(-'''+px+'''*2.*pi)*(X)/(double)(xSize));
	dcmplx phaseKickY = exp(i*(-'''+py+'''*2.*pi)*(Y)/(double)(ySize));
	dcmplx phaseKickZ = exp(i*(-'''+pz+'''*2.*pi)*(Z)/(double)(zSize));
	dcmplx field = ''' + func + '''*phaseKickX*phaseKickY*phaseKickZ;
	
	for (n=0; n<vectorSize; n=n+1){
		QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
	}  
}'''
