def generate_stream_x_string(numGPUs):
	string = "if(GPU_to_reference==0){QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = sign*Neighbor0[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize];}"
	for i in xrange(1,numGPUs):
		string += "\n else if(" + str(int(i)) + "==GPU_to_reference){QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = sign*Neighbor" + str(int(i)) + "[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize];}"
	return string

def generate_collide_x_string(numGPUs, coeff):
	string = "if(GPU_to_reference==0){new_qfield += Mul("+coeff+",Neighbor0[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize]);}"
	for i in xrange(1,numGPUs):
		string += "\n else if(" + str(int(i)) + "==GPU_to_reference){new_qfield += Mul("+coeff+",Neighbor" + str(int(i)) + "[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize]);}"
	return string


def get_CUDA(vectorSize, numGPUs):
	input_string = "dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params,"
	for i in xrange(numGPUs):
		input_string += "dcmplx *Neighbor" + str(i)
		if i != numGPUs-1:
			input_string += ", "
	return r'''
	// Apply sqrt(SWAP) and then copy field for x axis
	__global__ void collideX(''' + input_string + ''') {
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  dcmplx new_qfield;
	  int new_ind;
	  int GPU_to_reference;
	  dcmplx AA(0.,0.5); // i/2
	  dcmplx AcAc(0.,-0.5); // -i/2
	  dcmplx AcA(0.5, 0.); // 1/2
	  dcmplx i(0.,1.);
	  index_pair iToNum = index_to_number(Qx, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  if ((alpha%2 == 0) && (beta == alpha+1)){
	    for (n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(i,QField2[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
	    }
	  } else {
	    for (n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      // START WITH OWN BASIS
	      new_qfield = Mul(AcAc,QField2[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // DETERMINE WHICH GPU THE INDEX IS ON AND ADD NEW FIELD
	      new_ind = number_to_index(Qx,alpha+(1-2*(alpha%2)),beta);
	      GPU_to_reference = index_on_GPU(new_ind, xSize, numGPUs); // GPU to A*Ac term

	      // CHANGE GLOBAL INDEX TO LOCAL GPU INDEX
	      new_ind = new_ind - GPU_to_reference*local_xSize;
	      
	      if (deviceNum == GPU_to_reference){
	        new_qfield += Mul(AcA,QField2[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize]);
	      } else {
	        ''' + generate_collide_x_string(numGPUs, "AcA") + '''
	      }

	      
	      // DETERMINE WHICH GPU THE INDEX IS ON AND ADD NEW FIELD
	      new_ind = number_to_index(Qx,alpha,beta+(1-2*(beta%2)));
	      GPU_to_reference = index_on_GPU(new_ind, xSize, numGPUs); // GPU to Ac*A term

	      // CHANGE GLOBAL INDEX TO LOCAL GPU INDEX
	      new_ind = new_ind - GPU_to_reference*local_xSize;
	      if (deviceNum == GPU_to_reference){
	        new_qfield += Mul(AcA,QField2[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize]);
	      } else {
	        ''' + generate_collide_x_string(numGPUs, "AcA") + '''
	      }
	      

	      // DETERMINE WHICH GPU THE INDEX IS ON AND ADD NEW FIELD
	      new_ind = number_to_index(Qx,alpha+(1-2*(alpha%2)),beta+(1-2*(beta%2)));
	      GPU_to_reference = index_on_GPU(new_ind, xSize, numGPUs); // GPU to Ac*Ac term

	      // CHANGE GLOBAL INDEX TO LOCAL GPU INDEX
	      new_ind = new_ind - GPU_to_reference*local_xSize;
	      if (deviceNum == GPU_to_reference){
	        new_qfield += Mul(AA,QField2[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize]);
	      } else {
	        ''' + generate_collide_x_string(numGPUs, "AA") + '''
	      }

	      // SET NEW FIELD
	      QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = new_qfield;
	    } 
	  }
	} 

	// Apply sqrt(SWAP) and then copy field for y axis
	__global__ void collideY(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params) {
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qy = lattice[8];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  dcmplx AA(0.,0.5); // i/2
	  dcmplx AcAc(0.,-0.5); // -i/2
	  dcmplx AcA(0.5, 0.); // 1/2
	  index_pair iToNum = index_to_number(Qy, y);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  if ((alpha%2 == 0) && (beta == alpha+1)){
	    for (n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = -QField2[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	    }
	  } else {
	    for (n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      // START WITH OWN BASIS
	      dcmplx new_qfield = Mul(AcAc,QField2[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // Index for A*Ac term
	      int new_ind = number_to_index(Qy,alpha+(1-2*(alpha%2)),beta);
	      new_qfield += Mul(AcA,QField2[n+z*vectorSize+((new_ind)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // Index for A*Ac term
	      new_ind = number_to_index(Qy,alpha,beta+(1-2*(beta%2)));
	      new_qfield += Mul(AcA,QField2[n+z*vectorSize+((new_ind)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // Index for A*A term
	      new_ind = number_to_index(Qy,alpha+(1-2*(alpha%2)),beta+(1-2*(beta%2)));
	      new_qfield += Mul(AA,QField2[n+z*vectorSize+((new_ind)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // SET NEW FIELD
	      QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = new_qfield;
	    }
	  }
	}

	// Apply sqrt(SWAP) and then copy field for z axis
	__global__ void collideZ(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params) {
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qz = lattice[10];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  dcmplx AA(0.,0.5); // i/2
	  dcmplx AcAc(0.,-0.5); // -i/2
	  dcmplx AcA(0.5, 0.); // 1/2
	  index_pair iToNum = index_to_number(Qz, z);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  if ((alpha%2 == 0) && (beta == alpha+1)){
	    for (n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = -QField2[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	    }
	  } else {
	    for (n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      // START WITH OWN BASIS
	      dcmplx new_qfield = Mul(AcAc,QField2[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // Index for A*Ac term
	      int new_ind = number_to_index(Qz,alpha+(1-2*(alpha%2)),beta);
	      new_qfield += Mul(AcA,QField2[n+((new_ind)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // Index for A*Ac term
	      new_ind = number_to_index(Qz,alpha,beta+(1-2*(beta%2)));
	      new_qfield += Mul(AcA,QField2[n+((new_ind)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // Index for A*A term
	      new_ind = number_to_index(Qz,alpha+(1-2*(alpha%2)),beta+(1-2*(beta%2)));
	      new_qfield += Mul(AA,QField2[n+((new_ind)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // SET NEW FIELD
	      QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = new_qfield;
	    }
	  }
	}

	//Stream -l along x for 'left' comp in 2 qbit basis
	__global__ void streamXNeg0(''' + input_string + ''')
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;
	  double sign = signFromStreamEpsilon(1, alpha, beta, Qx, 1);

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(beta == 1) {
	    new_ind = number_to_index(Qx, 0, Qx-1);
	  } else if(alpha == 1) {
	    new_ind = number_to_index(Qx, beta-2*(beta%2), Qx-1);
	  } else if((beta == alpha+1) && (beta%2 != 0)) {
	    new_ind = number_to_index(Qx, beta-2, alpha);
	  } else {
	    new_ind = number_to_index(Qx, alpha-2*(alpha%2), beta-2*(beta%2));
	  }


	  // DETERMINE WHICH GPU THE INDEX IS ON AND SET NEW FIELD
	  int GPU_to_reference = index_on_GPU(new_ind, xSize, numGPUs);
	  // CHANGE GLOBAL INDEX TO LOCAL GPU INDEX
	  new_ind = new_ind - GPU_to_reference*local_xSize;
	  if (deviceNum == GPU_to_reference){
	    for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = sign*QField2[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize];
	    }
	  } else {
	    for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      ''' + generate_stream_x_string(numGPUs) + '''
	    }
	  }
	}

	//Stream +l along x for 'left' comp in 2 qbit basis
	__global__ void streamXPos0(''' + input_string + ''')
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;
	  double sign = signFromStreamEpsilon(1, alpha, beta, Qx, -1);

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(beta == Qx-1) {
	    if(alpha ==0){
	      new_ind = number_to_index(Qx, alpha, 1);
	    } else {
	      new_ind = number_to_index(Qx, 1, alpha+2*(alpha%2));
	    }
	  } else if((beta == alpha+1) && (beta%2 == 0)) {
	    new_ind = number_to_index(Qx, beta, alpha+2);
	  } else {
	    new_ind = number_to_index(Qx, alpha+2*(alpha%2), beta+2*(beta%2));
	  }


	  // DETERMINE WHICH GPU THE INDEX IS ON AND SET NEW FIELD
	  int GPU_to_reference = index_on_GPU(new_ind, xSize, numGPUs);
	  // CHANGE GLOBAL INDEX TO LOCAL GPU INDEX
	  new_ind = new_ind - GPU_to_reference*local_xSize;
	  if (deviceNum == GPU_to_reference){
	    for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = sign*QField2[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize];
	    }
	  } else {
	    for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      ''' + generate_stream_x_string(numGPUs) + '''
	    }
	  }
	}


	//Stream -l along x for 'right' comp in 2 qbit basis
	__global__ void streamXNeg1(''' + input_string + ''')
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;
	  double sign = signFromStreamEpsilon(0, alpha, beta, Qx, 1);

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(alpha == 0) {
	    if(beta == Qx-1){
	      new_ind = number_to_index(Qx, Qx-2, Qx-1);
	    } else {
	      new_ind = number_to_index(Qx, beta-2*((beta+1)%2), Qx-2);
	    }
	  } else if((beta == alpha+1) && (beta%2 == 0)) {
	    new_ind = number_to_index(Qx, beta-2, alpha);
	  } else {
	    new_ind = number_to_index(Qx, alpha-2*((alpha+1)%2), beta-2*((beta+1)%2));
	  }


	  // DETERMINE WHICH GPU THE INDEX IS ON AND SET NEW FIELD
	  int GPU_to_reference = index_on_GPU(new_ind, xSize, numGPUs);
	  // CHANGE GLOBAL INDEX TO LOCAL GPU INDEX
	  new_ind = new_ind - GPU_to_reference*local_xSize;
	  if (deviceNum == GPU_to_reference){
	    for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = sign*QField2[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize];
	    }
	  } else {
	    for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      ''' + generate_stream_x_string(numGPUs) + '''
	    }
	  }
	}

	//Stream +l along x for 'right' comp in 2 qbit basis
	__global__ void streamXPos1(''' + input_string + ''')
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;
	  double sign = signFromStreamEpsilon(0, alpha, beta, Qx, -1);

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(alpha == Qx-2) {
	    new_ind = number_to_index(Qx, 0, Qx-1);
	  } else if(beta == Qx-2) {
	    new_ind = number_to_index(Qx, 0, alpha+2*((alpha+1)%2));
	  } else if((beta == alpha+1) && (beta%2 != 0)) {
	    new_ind = number_to_index(Qx, beta, alpha+2);
	  } else {
	    new_ind = number_to_index(Qx, alpha+2*((alpha+1)%2), beta+2*((beta+1)%2));
	  }


	  // DETERMINE WHICH GPU THE INDEX IS ON AND SET NEW FIELD
	  int GPU_to_reference = index_on_GPU(new_ind, xSize, numGPUs);
	  // CHANGE GLOBAL INDEX TO LOCAL GPU INDEX
	  new_ind = new_ind - GPU_to_reference*local_xSize;
	  if (deviceNum == GPU_to_reference){
	    for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = sign*QField2[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize];
	    }
	  } else {
	    for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      ''' + generate_stream_x_string(numGPUs) + '''
	    }
	  }
	}

	//Stream +l along y for 'left' comp in 2 qbit basis
	__global__ void streamYPos0(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qy = lattice[8];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qy, y);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(beta == Qy-1) {
	    if(alpha ==0){
	      new_ind = number_to_index(Qy, alpha, 1);
	    } else {
	      new_ind = number_to_index(Qy, 1, alpha+2*(alpha%2));
	    }
	  } else if((beta == alpha+1) && (beta%2 == 0)) {
	    new_ind = number_to_index(Qy, beta, alpha+2);
	  } else {
	    new_ind = number_to_index(Qy, alpha+2*(alpha%2), beta+2*(beta%2));
	  }

	  for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	    QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+((new_ind)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize];
	  }

	}

	//Stream -l along x for 'left' comp in 2 qbit basis
	__global__ void streamYNeg0(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qy = lattice[8];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qy, y);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(beta == 1) {
	    new_ind = number_to_index(Qy, 0, Qy-1);
	  } else if(alpha == 1) {
	    new_ind = number_to_index(Qy, beta-2*(beta%2), Qy-1);
	  } else if((beta == alpha+1) && (beta%2 != 0)) {
	    new_ind = number_to_index(Qy, beta-2, alpha);
	  } else {
	    new_ind = number_to_index(Qy, alpha-2*(alpha%2), beta-2*(beta%2));
	  }

	  for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	    QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+((new_ind)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize];
	  }

	}

	//Stream +l along y for 'right' comp in 2 qbit basis
	__global__ void streamYPos1(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qy = lattice[8];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qy, y);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(alpha == Qy-2) {
	    new_ind = number_to_index(Qy, 0, Qy-1);
	  } else if(beta == Qy-2) {
	    new_ind = number_to_index(Qy, 0, alpha+2*((alpha+1)%2));
	  } else if((beta == alpha+1) && (beta%2 != 0)) {
	    new_ind = number_to_index(Qy, beta, alpha+2);
	  } else {
	    new_ind = number_to_index(Qy, alpha+2*((alpha+1)%2), beta+2*((beta+1)%2));
	  }

	  for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	    QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+((new_ind)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize];
	  }

	}

	//Stream -l along y for 'right' comp in 2 qbit basis
	__global__ void streamYNeg1(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qy = lattice[8];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qy, y);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(alpha == 0) {
	    if(beta == Qy-1){
	      new_ind = number_to_index(Qy, Qy-2, Qy-1);
	    } else {
	      new_ind = number_to_index(Qy, beta-2*((beta+1)%2), Qy-2);
	    }
	  } else if((beta == alpha+1) && (beta%2 == 0)) {
	    new_ind = number_to_index(Qy, beta-2, alpha);
	  } else {
	    new_ind = number_to_index(Qy, alpha-2*((alpha+1)%2), beta-2*((beta+1)%2));
	  }

	  for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	    QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+((new_ind)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize];
	  }

	}

	//Stream +l along y for 'left' comp in 2 qbit basis
	__global__ void streamZPlusL(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qz = lattice[10];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qz, z);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(beta == Qz-1) {
	    if(alpha ==0){
	      new_ind = number_to_index(Qz, alpha, 1);
	    } else {
	      new_ind = number_to_index(Qz, 1, alpha+2*(alpha%2));
	    }
	  } else if((beta == alpha+1) && (beta%2 == 0)) {
	    new_ind = number_to_index(Qz, beta, alpha+2);
	  } else {
	    new_ind = number_to_index(Qz, alpha+2*(alpha%2), beta+2*(beta%2));
	  }

	  for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	    QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+((new_ind)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	  }

	}

	//Stream -l along x for 'left' comp in 2 qbit basis
	__global__ void streamZNeg0(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qz = lattice[10];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qz, z);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(beta == 1) {
	    new_ind = number_to_index(Qz, 0, Qz-1);
	  } else if(alpha == 1) {
	    new_ind = number_to_index(Qz, beta-2*(beta%2), Qz-1);
	  } else if((beta == alpha+1) && (beta%2 != 0)) {
	    new_ind = number_to_index(Qz, beta-2, alpha);
	  } else {
	    new_ind = number_to_index(Qz, alpha-2*(alpha%2), beta-2*(beta%2));
	  }

	  for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	    QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+((new_ind)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	  }

	}

	//Stream +l along y for 'right' comp in 2 qbit basis
	__global__ void streamZPos1(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qz = lattice[10];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qz, z);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(alpha == Qz-2) {
	    new_ind = number_to_index(Qz, 0, Qz-1);
	  } else if(beta == Qz-2) {
	    new_ind = number_to_index(Qz, 0, alpha+2*((alpha+1)%2));
	  } else if((beta == alpha+1) && (beta%2 != 0)) {
	    new_ind = number_to_index(Qz, beta, alpha+2);
	  } else {
	    new_ind = number_to_index(Qz, alpha+2*((alpha+1)%2), beta+2*((beta+1)%2));
	  }

	  for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	    QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+((new_ind)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	  }

	}

	//Stream -l along y for 'right' comp in 2 qbit basis
	__global__ void streamZNeg1(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qz = lattice[10];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qz, z);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  int new_ind = 0;

	  // DETERMINE INDEX FIELD WAS STREAMED FROM
	  if(alpha == 0) {
	    if(beta == Qz-1){
	      new_ind = number_to_index(Qz, Qz-2, Qz-1);
	    } else {
	      new_ind = number_to_index(Qz, beta-2*((beta+1)%2), Qz-2);
	    }
	  } else if((beta == alpha+1) && (beta%2 == 0)) {
	    new_ind = number_to_index(Qz, beta-2, alpha);
	  } else {
	    new_ind = number_to_index(Qz, alpha-2*((alpha+1)%2), beta-2*((beta+1)%2));
	  }

	  for (int n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	    QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+((new_ind)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	  }

	}

	
	'''
