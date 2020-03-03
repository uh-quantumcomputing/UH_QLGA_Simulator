def get_CUDA(dimensions, vectorSize, timesteps = [1], positions = [0], widths = [1.], func = "1./cosh", smooth = False, Measured = True, **kwargs):
	potentialString = """measurementAmp = 0.;
	  if ( ((alpha/2)<=(x_measured+delta_x) && (alpha/2)>=(x_measured-delta_x)) || ((beta/2)<=(x_measured+delta_x) && (beta/2)>=(x_measured-delta_x))){
	    measurementAmp = 1.;
	  }"""
	if Measured:
		if smooth:
			potentialString = """if (delta_x == 0){
				delta_x = 1;
			}
			measurementAmp = 0.5*""" + func + """(((double)alpha/2.-x_measured)/(delta_x)) + 0.5*""" + func + """(((double)beta/2.-x_measured)/(delta_x));"""
	else:
		potentialString = """measurementAmp = 1.;
		  if ( ((alpha/2)<=(x_measured+delta_x) && (alpha/2)>=(x_measured-delta_x)) || ((beta/2)<=(x_measured+delta_x) && (beta/2)>=(x_measured-delta_x))){
		    measurementAmp = 0.;
		  }"""
		if smooth:
			potentialString = """if (delta_x == 0){
			delta_x = 1;
			}
			measurementAmp = 1 - 0.5*""" + func + """(((double)alpha/2.-x_measured)/(delta_x)) - 0.5*""" + func + """(((double)beta/2.-x_measured)/(delta_x));"""
	measureString = ""
	for measurement in xrange(len(timesteps)):
		measureString += ''' if (time_step == ''' + str(int(timesteps[measurement])) + '''){
			x_measured = ''' + str(positions[measurement]) + ''';
			delta_x = ''' + str(widths[measurement]) + ''';
			''' + potentialString + '''
			
			for (n=0; n<vectorSize; n=n+1){
				field = Mul(measurementAmp,QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
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
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int z = blockIdx.z * blockDim.z + threadIdx.z;
	int n;
	dcmplx i(0.,1.);
	dcmplx field(0.,0.);
	int vectorSize = ''' + str(vectorSize) + ''';
	index_pair iToNum = index_to_number(Qx, x + deviceNum*local_xSize);
	int alpha = iToNum.index0;
	int beta = iToNum.index1;
	double x_measured = 0;
	double delta_x = 0;
	double this_x = (double)(x + deviceNum*local_xSize);
	dcmplx measurementAmp = 0.;
	'''+ measureString +'''
}'''
