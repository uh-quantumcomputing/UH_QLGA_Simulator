def get_CUDA(dimensions, vectorSize, timestep = 1, position = 0, width = 1., func = "1./cosh", smooth = False, Measured = True):
	potentialString = """dcmplx potentialU = 0.;
	  if ( ((alpha/2)<=(x_measured+delta_x) && (alpha/2)>=(x_measured-delta_x)) || ((beta/2)<=(x_measured+delta_x) && (beta/2)>=(x_measured-delta_x))){
	    potentialU = 1.;
	  }"""
	if Measured:
		if smooth:
			potentialString = """if (delta_x == 0){
			delta_x = 1;
		}
		dcmplx potentialU = """ + func + """(((double)alpha/2.-(double)x_measured)/((double)delta_x*2.)) + """ + func + """(((double)beta/2.-(double)x_measured)/((double)delta_x*2.));"""
	else:
		potentialString = """dcmplx potentialU = 1.;
		  if ( ((alpha/2)<=(x_measured+delta_x) && (alpha/2)>=(x_measured-delta_x)) || ((beta/2)<=(x_measured+delta_x) && (beta/2)>=(x_measured-delta_x))){
		    potentialU = 0.;
		  }"""
		if smooth:
			potentialString = """if (delta_x == 0){
			delta_x = 1;
		}
		dcmplx potentialU = 1 - """ + func + """(((double)alpha/2.-(double)x_measured)/((double)delta_x*2.)) - """ + func + """(((double)beta/2.-(double)x_measured)/((double)delta_x*2.));"""

	return '''__global__ void measurement(dcmplx *QField, dcmplx *QFieldCopy, int* lattice, int* gpu_params){
	int time_step = lattice[14];
	if (time_step == ''' + str(int(timestep)) + '''){
		int deviceNum = gpu_params[0];
		int numGPUs = gpu_params[2];
		int xSize = lattice[0]*numGPUs;
		int local_xSize = lattice[0];
		int ySize = lattice[2];
		int zSize = lattice[4];
		int Qx = lattice[6];
		int vectorSize = ''' + str(vectorSize) + ''';
		int x_measured = ''' + str(int(position)) + ''';
		int delta_x = ''' + str(int(width)) + ''';
		int x = blockIdx.x * blockDim.x + threadIdx.x;
		int y = blockIdx.y * blockDim.y + threadIdx.y;
		int z = blockIdx.z * blockDim.z + threadIdx.z;
		int n;
		dcmplx i(0.,1.);
		dcmplx field(0.,0.);
		index_pair iToNum = index_to_number(Qx, x + deviceNum*local_xSize);
		int alpha = iToNum.index0;
		int beta = iToNum.index1;
		''' + potentialString + '''
		
		for (n=0; n<vectorSize; n=n+1){
			field = Mul(potentialU,QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
			QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
			QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = field;
		} 
	} 
}'''
