def get_CUDA(dimensions, vectorSize, timesteps = [1], positions = [1], widths = [1.], func = "1./cosh", smooth = False, Measured = True, **kwargs):
	potentialString = """potentialU = 0.;
	  if ( ((this_x)<=(x_measured+delta_x) && (this_x)>=(x_measured-delta_x))){
	    potentialU = 1.;
	  }"""
	if Measured:
		if smooth:
			potentialString = """if (delta_x == 0){
			delta_x = 1;
		}
		dcmplx potentialU = """ + func + """(((double)this_x-(double)x_measured)/((double)delta_x));"""
	else:
		potentialString = """potentialU = 1.;
		  if ( ((this_x)<=(x_measured+delta_x) && (this_x)>=(x_measured-delta_x)) ){
		    potentialU = 0.;
		  }"""
		if smooth:
			potentialString = """if (delta_x == 0){
			delta_x = 1;
		}
		potentialU = 1 - """ + func + """(((double)this_x-(double)x_measured)/((double)delta_x));"""
	measureString = ""
	for measurement in xrange(len(timesteps)):
		measureString += ''' if (time_step == ''' + str(int(timesteps[measurement])) + '''){
			x_measured = ''' + str(int(positions[measurement])) + ''';
			delta_x = ''' + str(int(widths[measurement])) + ''';
			''' + potentialString + '''
			
			for (n=0; n<vectorSize; n=n+1){
				field = Mul(potentialU,QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
				QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
				QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
			} 
		} 
		'''

	return '''__global__ void measurement(dcmplx *QField, dcmplx *QFieldCopy, int* lattice, int* gpu_params){
	int time_step = lattice[14];
	int deviceNum = gpu_params[0];
	int numGPUs = gpu_params[2];
	int xSize = lattice[0]*numGPUs;
	int local_xSize = lattice[0];
	int ySize = lattice[2];
	int zSize = lattice[4];
	int Qx = lattice[6];
	int vectorSize = ''' + str(vectorSize) + ''';
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int n;
	dcmplx i(0.,1.);
	dcmplx field(0.,0.);
	int x_measured = 0;
	int delta_x = 0;
	double this_x = (double)(x + deviceNum*local_xSize);
	dcmplx potentialU = 0.;
	'''+ measureString +'''
}'''
