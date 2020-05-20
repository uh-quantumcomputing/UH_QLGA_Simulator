# User inputs cond_list and func_list for desired potential
def get_CUDA(dimensions, vectorSize, cond_list=['true'] ,func_list = ['sin(2*pi*X/Lx)'], **kwargs):
	# Default to identity unless otherwise specified and set unitary operator
	potentialString = '''dcmplx potentialU = 1.;
		if ('''+ cond_list[0] +'''){potentialU = exp(-Mul(i,(dcmplx)('''+ func_list[0] +''')));}'''
	if len(cond_list)>1:
		for i in xrange(1,len(cond_list)):
			# Check the remaining conditionals and convert input V-->e^(-iV)
			potentialString += '''
			else if('''+ cond_list[i] +'''){potentialU = exp(-Mul(i,(dcmplx)('''+ func_list[i] +''')));}'''
	return '''__global__ void external(dcmplx *QField, dcmplx *QFieldCopy, int* lattice, int* gpu_params){
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
	double this_x = double(x+deviceNum*lattice[0]);
	int n;
	dcmplx i(0.,1.);
	dcmplx field(0.,0.);
	double X = (double)(this_x);
	double Y = (double)(y);
	double Z = (double)(z);
	double T = (double)time_step;
	double Lx = (double) xSize;
	double Ly = (double) ySize;
	double Lz = (double) zSize;
	
	''' + potentialString + '''

	for (n=0; n<vectorSize; n=n+1){
		field = Mul(potentialU,QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
		QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
		QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
	}  
}'''
