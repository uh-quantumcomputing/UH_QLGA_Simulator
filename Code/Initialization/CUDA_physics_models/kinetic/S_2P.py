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
	__global__ void collide(''' + input_string + ''') {
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int Ly = lattice[8]/2;
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
	  index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	  int alpha_x = alphaBeta2d.index0;
	  int alpha_y = alphaBeta2d.index1;
	  int beta_x = alphaBeta2d.index2;
	  int beta_y = alphaBeta2d.index3;
	  if ((alpha_x/2==beta_x/2) && beta_y==alpha_y){
	    for (n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(i,QField2[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
	    }
	  } else {
	    for (n=0; n<''' + str(int(vectorSize)) + '''; n=n+1){
	      // START WITH OWN BASIS
	      new_qfield = Mul(AcAc,QField2[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);

	      // DETERMINE WHICH GPU THE INDEX IS ON AND ADD NEW FIELD
	      new_ind = index_from2d_num(Qx, Ly, alpha_x+(1-2*(alpha_x%2)), beta_x, alpha_y, beta_y);
	      GPU_to_reference = index_on_GPU(new_ind, xSize, numGPUs); // GPU to A*Ac term

	      // CHANGE GLOBAL INDEX TO LOCAL GPU INDEX
	      new_ind = new_ind - GPU_to_reference*local_xSize;
	      
	      if (deviceNum == GPU_to_reference){
	        new_qfield += Mul(AcA,QField2[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize]);
	      } else {
	        ''' + generate_collide_x_string(numGPUs, "AcA") + '''
	      }

	      
	      // DETERMINE WHICH GPU THE INDEX IS ON AND ADD NEW FIELD
	      new_ind = index_from2d_num(Qx, Ly, alpha_x, beta_x+(1-2*(beta_x%2)), alpha_y, beta_y);
	      GPU_to_reference = index_on_GPU(new_ind, xSize, numGPUs); // GPU to Ac*A term

	      // CHANGE GLOBAL INDEX TO LOCAL GPU INDEX
	      new_ind = new_ind - GPU_to_reference*local_xSize;
	      if (deviceNum == GPU_to_reference){
	        new_qfield += Mul(AcA,QField2[n+z*vectorSize+y*vectorSize*zSize+((new_ind)%(xSize))*zSize*ySize*vectorSize]);
	      } else {
	        ''' + generate_collide_x_string(numGPUs, "AcA") + '''
	      }
	      

	      // DETERMINE WHICH GPU THE INDEX IS ON AND ADD NEW FIELD
	      new_ind = index_from2d_num(Qx, Ly, alpha_x+(1-2*(alpha_x%2)), beta_x+(1-2*(beta_x%2)), alpha_y, beta_y);
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
	  int Ly = lattice[8]/2;
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	  int alpha_x = alphaBeta2d.index0;
	  int alpha_y = alphaBeta2d.index1;
	  int beta_x = alphaBeta2d.index2;
	  int beta_y = alphaBeta2d.index3;
	  double sign = 1.;
	  int new_alpha_x = alpha_x;
	  int new_beta_x = beta_x;

	  // Determine new alpha_x, beta_x
	  if(beta_x%2==0){
	  	new_beta_x = (Qx+beta_x-2)%Qx;
	  }
	  if(alpha_x%2==0){
	  	new_alpha_x = (Qx+alpha_x-2)%Qx;
	  } 

	  int new_ind = index_from2d_num(Qx, Ly, new_alpha_x, new_beta_x, alpha_y, beta_y);

	  // Case where alpha and beta switch meanings, to keep alpha<beta
	  if (alpha_y*Qx+new_alpha_x>beta_y*Qx+new_beta_x){
	  	sign = -1.;
	  	new_ind = index_from2d_num(Qx, Ly, new_beta_x, new_alpha_x, beta_y, alpha_y);
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
	  int Ly = lattice[8]/2;
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	  int alpha_x = alphaBeta2d.index0;
	  int alpha_y = alphaBeta2d.index1;
	  int beta_x = alphaBeta2d.index2;
	  int beta_y = alphaBeta2d.index3;
	  double sign = 1;
	  int new_alpha_x = alpha_x;
	  int new_beta_x = beta_x;

	  // Determine new alpha_x, beta_x
	  if(beta_x%2==0){
	  	new_beta_x = (Qx+beta_x+2)%Qx;
	  }
	  if(alpha_x%2==0){
	  	new_alpha_x = (Qx+alpha_x+2)%Qx;
	  }

	  int new_ind = index_from2d_num(Qx, Ly, new_alpha_x, new_beta_x, alpha_y, beta_y);

	  // Case where alpha and beta switch meanings, to keep alpha<beta
	  if (alpha_y*Qx+new_alpha_x>beta_y*Qx+new_beta_x){
	  	sign = -1.;
	  	new_ind = index_from2d_num(Qx, Ly, new_beta_x, new_alpha_x, beta_y, alpha_y);
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
	  int Ly = lattice[8]/2;
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	  int alpha_x = alphaBeta2d.index0;
	  int alpha_y = alphaBeta2d.index1;
	  int beta_x = alphaBeta2d.index2;
	  int beta_y = alphaBeta2d.index3;
	  double sign = 1;
	  int new_alpha_x = alpha_x;
	  int new_beta_x = beta_x;

	  // Determine new alpha_x, beta_x
	  if(beta_x%2==1){
	  	new_beta_x = (Qx+beta_x-2)%Qx;
	  }
	  if(alpha_x%2==1){
	  	new_alpha_x = (Qx+alpha_x-2)%Qx;
	  }

	  int new_ind = index_from2d_num(Qx, Ly, new_alpha_x, new_beta_x, alpha_y, beta_y);

	  // Case where alpha and beta switch meanings, to keep alpha<beta
	  if (alpha_y*Qx+new_alpha_x>beta_y*Qx+new_beta_x){
	  	sign = -1.;
	  	new_ind = index_from2d_num(Qx, Ly, new_beta_x, new_alpha_x, beta_y, alpha_y);
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
	  int Ly = lattice[8]/2;
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	  int alpha_x = alphaBeta2d.index0;
	  int alpha_y = alphaBeta2d.index1;
	  int beta_x = alphaBeta2d.index2;
	  int beta_y = alphaBeta2d.index3;
	  double sign = 1;
	  int new_alpha_x = alpha_x;
	  int new_beta_x = beta_x;

	  // Determine new alpha_x, beta_x
	  if(beta_x%2==1){
	  	new_beta_x = (Qx+beta_x+2)%Qx;
	  }
	  if(alpha_x%2==1){
	  	new_alpha_x = (Qx+alpha_x+2)%Qx;
	  }

	  int new_ind = index_from2d_num(Qx, Ly, new_alpha_x, new_beta_x, alpha_y, beta_y);

	  // Case where alpha and beta switch meanings, to keep alpha<beta
	  if (alpha_y*Qx+new_alpha_x>beta_y*Qx+new_beta_x){
	  	sign = -1.;
	  	new_ind = index_from2d_num(Qx, Ly, new_beta_x, new_alpha_x, beta_y, alpha_y);
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
	__global__ void streamYPos0(''' + input_string + ''')
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int Ly = lattice[8]/2;
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	  int alpha_x = alphaBeta2d.index0;
	  int alpha_y = alphaBeta2d.index1;
	  int beta_x = alphaBeta2d.index2;
	  int beta_y = alphaBeta2d.index3;
	  double sign = 1.;
	  int new_alpha_y = alpha_y;
	  int new_beta_y = beta_y;

	  // Determine new alpha_y, beta_y
	  if(beta_x%2==0){
	  	new_beta_y = (Ly+beta_y+1)%Ly;
	  }
	  if(alpha_x%2==0){
	  	new_alpha_y = (Ly+alpha_y+1)%Ly;
	  }

	  int new_ind = index_from2d_num(Qx, Ly, alpha_x, beta_x, new_alpha_y, new_beta_y);

	  // Case where alpha and beta switch meanings, to keep alpha<beta
	  if (new_alpha_y*Qx+alpha_x>new_beta_y*Qx+beta_x){
	  	sign = -1.;
	  	new_ind = index_from2d_num(Qx, Ly, beta_x, alpha_x, new_beta_y, new_alpha_y);
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

	//Stream -l along y for 'left' comp in 2 qbit basis
	__global__ void streamYNeg0(''' + input_string + ''')
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int Ly = lattice[8]/2;
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	  int alpha_x = alphaBeta2d.index0;
	  int alpha_y = alphaBeta2d.index1;
	  int beta_x = alphaBeta2d.index2;
	  int beta_y = alphaBeta2d.index3;
	  double sign = 1;
	  int new_alpha_y = alpha_y;
	  int new_beta_y = beta_y;

	  // Determine new alpha_y, beta_y
	  if(beta_x%2==0){
	  	new_beta_y = (Ly+beta_y-1)%Ly;
	  }
	  if(alpha_x%2==0){
	  	new_alpha_y = (Ly+alpha_y-1)%Ly;
	  }
	
		int new_ind = index_from2d_num(Qx, Ly, alpha_x, beta_x, new_alpha_y, new_beta_y);

	  // Case where alpha and beta switch meanings, to keep alpha<beta
	  if (new_alpha_y*Qx+alpha_x>new_beta_y*Qx+beta_x){
	  	sign = -1.;
	  	new_ind = index_from2d_num(Qx, Ly, beta_x, alpha_x, new_beta_y, new_alpha_y);
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

	//Stream +l along y for 'right' comp in 2 qbit basis
	__global__ void streamYPos1(''' + input_string + ''')
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int Ly = lattice[8]/2;
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	  int alpha_x = alphaBeta2d.index0;
	  int alpha_y = alphaBeta2d.index1;
	  int beta_x = alphaBeta2d.index2;
	  int beta_y = alphaBeta2d.index3;
	  double sign = 1;
	  int new_alpha_y = alpha_y;
	  int new_beta_y = beta_y;

	  // Determine new alpha_y, beta_y
	  if(beta_x%2==1){
	  	new_beta_y = (Ly+beta_y+1)%Ly;
	  }
	  if(alpha_x%2==1){
	  	new_alpha_y = (Ly+alpha_y+1)%Ly;
	  }
	
		int new_ind = index_from2d_num(Qx, Ly, alpha_x, beta_x, new_alpha_y, new_beta_y);

	  // Case where alpha and beta switch meanings, to keep alpha<beta
	  if (new_alpha_y*Qx+alpha_x>new_beta_y*Qx+beta_x){
	  	sign = -1.;
	  	new_ind = index_from2d_num(Qx, Ly, beta_x, alpha_x, new_beta_y, new_alpha_y);
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

	//Stream -l along y for 'right' comp in 2 qbit basis
	__global__ void streamYNeg1(''' + input_string + ''')
	{
	  int deviceNum = gpu_params[0];
	  int numGPUs = gpu_params[2];
	  int xSize = lattice[0]*numGPUs;
	  int local_xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int Qx = lattice[6];
	  int Ly = lattice[8]/2;
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  index_pair iToNum = index_to_number(Qx*Ly, x + deviceNum*local_xSize);
	  int alpha = iToNum.index0;
	  int beta = iToNum.index1;
	  index_quad alphaBeta2d = num_to2d_num(Qx, alpha, beta);
	  int alpha_x = alphaBeta2d.index0;
	  int alpha_y = alphaBeta2d.index1;
	  int beta_x = alphaBeta2d.index2;
	  int beta_y = alphaBeta2d.index3;
	  double sign = 1;
	  int new_alpha_y = alpha_y;
	  int new_beta_y = beta_y;

	  // Determine new alpha_y, beta_y
	  if(beta_x%2==1){
	  	new_beta_y = (Ly+beta_y-1)%Ly;
	  }
	  if(alpha_x%2==1){
	  	new_alpha_y = (Ly+alpha_y-1)%Ly;
	  }
	
		int new_ind = index_from2d_num(Qx, Ly, alpha_x, beta_x, new_alpha_y, new_beta_y);

	  // Case where alpha and beta switch meanings, to keep alpha<beta
	  if (new_alpha_y*Qx+alpha_x>new_beta_y*Qx+beta_x){
	  	sign = -1.;
	  	new_ind = index_from2d_num(Qx, Ly, beta_x, alpha_x, new_beta_y, new_alpha_y);
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



	
	'''
