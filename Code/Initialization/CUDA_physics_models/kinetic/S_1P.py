def generate_stream_x_neg_string(numGPUs):
	string = "if(deviceNum==" + str(int(numGPUs-1)) + "){QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = Neighbor0[n+z*vectorSize+y*vectorSize*zSize+((0)%(xSize))*zSize*ySize*vectorSize];}"
	for i in xrange(numGPUs-1):
		string += "\n else if(" + str(int(i)) + "==deviceNum){QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = Neighbor" + str(int(i+1)) + "[n+z*vectorSize+y*vectorSize*zSize+((0)%(xSize))*zSize*ySize*vectorSize];}"
	return string

def generate_stream_x_pos_string(numGPUs):
	string = "if(deviceNum==0){QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = Neighbor" + str(int(numGPUs-1)) + "[n+z*vectorSize+y*vectorSize*zSize+((xSize-1)%(xSize))*zSize*ySize*vectorSize];}"
	for i in xrange(1,numGPUs):
		string += "\n else if(" + str(int(i)) + "==deviceNum){QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = Neighbor" + str(int(i-1)) + "[n+z*vectorSize+y*vectorSize*zSize+((xSize-1)%(xSize))*zSize*ySize*vectorSize];}"
	return string

def get_CUDA(vectorSize, numGPUs):
	stream_x_input_string = "dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params,"
	for i in xrange(numGPUs):
		stream_x_input_string += "dcmplx *Neighbor" + str(i)
		if i != numGPUs-1:
			stream_x_input_string += ", "
	return r'''
	__global__ void collide(dcmplx *QField, dcmplx *QFieldCopy, int* lattice) {
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  dcmplx plus(0.5,0.5); // 1/2 + i/2
	  dcmplx minus(0.5,-0.5); // 1/2 - i/2
	  //Apply sqrt(SWAP) pairwise
	  for (n=0; n<''' + str(int(vectorSize/2)) + '''; n=n+1){
	      dcmplx psiL = QField[2*n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	      dcmplx psiR = QField[2*n+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	      QField[2*n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(minus,psiL)+Mul(plus,psiR);
	      QField[2*n+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = Mul(plus,psiL)+Mul(minus,psiR);
	    }
	  for (n=0; n<''' + str(int(vectorSize)) + '''; n++){
	    QFieldCopy[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	  }

	}
		//Stream -l along x for 'left' comp in 2 qbit basis
	__global__ void streamXNeg0(''' + stream_x_input_string + ''')
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int deviceNum = gpu_params[0];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  for (int n = 0; n<''' + str(int(vectorSize)) + '''; n+=2){
	    if(x != xSize-1){
	        QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+y*vectorSize*zSize+((x+1)%(xSize))*zSize*ySize*vectorSize];
	    } else {
	        ''' + generate_stream_x_neg_string(numGPUs) + '''
	    }
	  }
	}

	//Stream +l along x for 'left' comp in 2 qbit basis
	__global__ void streamXPos0(''' + stream_x_input_string + ''')
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int deviceNum = gpu_params[0];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  for (int n = 0; n<''' + str(int(vectorSize)) + '''; n+=2){
	    if(x != 0){
	        QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+y*vectorSize*zSize+((x-1+xSize)%(xSize))*zSize*ySize*vectorSize];
	    } else {
	         ''' + generate_stream_x_pos_string(numGPUs) + '''
	    }
	  }
	}

	//Stream -l along x for 'right' comp in 2 qbit basis
	__global__ void streamXNeg1(''' + stream_x_input_string + ''')
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int deviceNum = gpu_params[0];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  for (int n = 1; n<''' + str(int(vectorSize)) + '''; n+=2){
	    if(x != xSize-1){
	        QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+y*vectorSize*zSize+((x+1)%(xSize))*zSize*ySize*vectorSize];
	    } else {
	        ''' + generate_stream_x_neg_string(numGPUs) + '''
	    }
	  }
	}

	//Stream +l along x for 'right' comp in 2 qbit basis
	__global__ void streamXPos1(''' + stream_x_input_string + ''')
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int deviceNum = gpu_params[0];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  for (int n = 1; n<''' + str(int(vectorSize)) + '''; n+=2){
	    if(x != 0){
	        QField[n+z*vectorSize+y*vectorSize*zSize+((x)%(xSize))*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+y*vectorSize*zSize+((x-1+xSize)%(xSize))*zSize*ySize*vectorSize];
	    } else {
	         ''' + generate_stream_x_pos_string(numGPUs) + '''
	    }
	  }
	}

	__global__ void streamYPos0(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  for (n=0; n<''' + str(int(vectorSize)) + '''; n+=2){
	      QField[n+z*vectorSize+((y)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+((y-1+ySize)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize];
	    }

	}

	__global__ void streamYNeg0(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  for (n=0; n<''' + str(int(vectorSize)) + '''; n+=2){
	      QField[n+z*vectorSize+((y)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+((y+1)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize]; 
	    }

	}

	__global__ void streamYPos1(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  for (n=1; n<''' + str(int(vectorSize)) + '''; n+=2){
	      QField[n+z*vectorSize+((y)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+((y-1+ySize)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize]; 
	    }

	}

	__global__ void streamYNeg1(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  for (n=1; n<''' + str(int(vectorSize)) + '''; n+=2){
	      QField[n+z*vectorSize+((y)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+z*vectorSize+((y+1)%(ySize))*vectorSize*zSize+x*zSize*ySize*vectorSize]; 
	    }

	}

	__global__ void streamZPos0(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  for (n=0; n<''' + str(int(vectorSize)) + '''; n+=2){
	      QField[n+((z)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+((z-1+zSize)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	    }
	}

	__global__ void streamZNeg0(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  for (n=0; n<''' + str(int(vectorSize)) + '''; n+=2){
	      QField[n+((z)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+((z+1)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	    }

	}

	__global__ void streamZPos1(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  for (n=1; n<''' + str(int(vectorSize)) + '''; n+=2){
	      QField[n+((z)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+((z-1+zSize)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	    }

	}

	__global__ void streamZNeg1(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{
	  int xSize = lattice[0];
	  int ySize = lattice[2];
	  int zSize = lattice[4];
	  int vectorSize = lattice[12];
	  int x = blockIdx.x * blockDim.x + threadIdx.x;
	  int y = blockIdx.y * blockDim.y + threadIdx.y;
	  int z = blockIdx.z * blockDim.z + threadIdx.z;
	  int n;
	  for (n=1; n<''' + str(int(vectorSize)) + '''; n+=2){
	      QField[n+((z)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField2[n+((z+1)%(zSize))*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	    }

	}
	'''
