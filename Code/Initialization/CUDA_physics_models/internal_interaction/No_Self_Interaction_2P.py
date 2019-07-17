def get_CUDA(dimensions, vectorSize, **model_kwargs):
	return r'''
		__global__ void internal(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{   
	}
	'''