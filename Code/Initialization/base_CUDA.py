from pycuda.compiler import SourceModule
gpuSource = SourceModule("""
	#include <pycuda-complex.hpp>
  #include <stdio.h>
  #define CUDART_NAN __longlong_as_double(0xfff8000000000000ULL)
	typedef pycuda::complex<double> dcmplx;

	const double pi = 3.14159265358979323846264338328;

struct index_pair {
  int index0;
  int index1;
};

struct index_quad {
  int index0;
  int index1;
  int index2;
  int index3;
};

__device__  dcmplx Mul (dcmplx u, dcmplx z)
{ 
    return dcmplx(__dmul_rn(u.real(), z.real()) - __dmul_rn(u.imag(), z.imag()), __dmul_rn(u.real(), z.imag()) + __dmul_rn(u.imag(), z.real()));    
}


__device__ dcmplx Mul3 (dcmplx z1, dcmplx z2, dcmplx z3)
{ 
    dcmplx res1 = Mul(z1, z2);
    dcmplx res = Mul(res1, z3);   
    return res; 
    }

__device__  dcmplx Mul4 (dcmplx z1, dcmplx z2, dcmplx z3, dcmplx z4)
{ 
    dcmplx res1 = Mul(z1, z2);
    dcmplx res2 = Mul(z4, z3); 
    return Mul(res1, res2); 
    }

__device__  double phase (dcmplx z)
{   
    if (z.imag() == 0. && z.real()==0.){
    return 0.;
    }
    double R = atan2(z.imag(), z.real());
    return R;

}

__device__  double magnitude (dcmplx z)
{   
    return sqrt(Mul(z, conj(z)).real());
}

__device__ index_pair index_to_number(int Q, int i){
  int n = Q - 1;
  while (i>=0){
    i -= n;
    n -= 1;
  }
  // alpha = Q-2-n
  // beta = Q+i
  return {Q-2-n,Q+i};
}

__device__ int number_to_index(int Q, int alpha, int beta){
  int i = 0;
  int n = Q - 1;
  for(int j = 0; j < alpha; j++ ){
    i += n;
    n -= 1;
  }
  return i+beta-(alpha+1); //index
}

__device__ index_quad num_to2d_num(int Qx, int alpha, int beta){
  return {alpha%Qx,alpha/(Qx),beta%Qx,beta/(Qx)};
}

__device__ int index_from2d_num(int Qx, int Ly, int alpha_x, int beta_x,int alpha_y, int beta_y){
  return number_to_index(Qx*Ly, alpha_y*Qx+alpha_x, beta_y*Qx+beta_x);
}

__device__ dcmplx switch_phase_1 (dcmplx z){
  double new_phase = -phase(z) + pi/2.;
  double mag = sqrt(__dmul_rn(z.real(), z.real()) + __dmul_rn(z.imag(), z.imag()));
  return Mul(dcmplx(mag, 0), exp(dcmplx(0 , new_phase))); 
}

__device__ dcmplx switch_phase_2 (dcmplx z){
  double new_phase = phase(z) + pi;
  double mag = sqrt(__dmul_rn(z.real(), z.real()) + __dmul_rn(z.imag(), z.imag()));
  return Mul(dcmplx(mag, 0), exp(dcmplx(0 , new_phase))); 
}

__device__ dcmplx switch_phase_3 (dcmplx z){
  double new_phase = -phase(z) + 3.*pi/2.;
  double mag = sqrt(__dmul_rn(z.real(), z.real()) + __dmul_rn(z.imag(), z.imag()));
  return Mul(dcmplx(mag, 0), exp(dcmplx(0 , new_phase))); 
}


__device__ int nearest_pole_center(int this_x, int xSize, int x_center){
  int distance = abs(this_x - x_center);
  if(distance < xSize/2){
    return double(x_center);
  } else if(x_center < xSize/2){
    return double(x_center + xSize);
  } else{
    return double(x_center - xSize);
  }
}

__device__ double gauss(double arg){
    return exp(-arg*arg);
}

__device__  dcmplx expC (dcmplx z)
{
    int cos_last_changed = 0;
    double R = 1.;
    double c = (cos(z.imag()));
    int sgn_s = signbit(sin(z.imag()));
    int sgn_c = signbit(c);
    double s = sqrt(R-(c*c));
    if (c*c + s*s == R){
      if (sgn_s == 0){ 
      return dcmplx(c, s);
      }
      else {
      return dcmplx(c, -s);
      }
    }
    for(int i =0; i < 10; i++){
      if (c*c + s*s == R){
        break;
      }
      if (i == 9){
        printf("ERROR: CUDA expC critical failure");
        return CUDART_NAN;
      }
      if (cos_last_changed == 0){
        c =sqrt(R-(s*s));
        cos_last_changed = 1;
      } else {
        s = sqrt(R-(c*c));
        cos_last_changed = 0;
      }
    }
    if(sgn_c == 0 && sgn_s == 0){
      return dcmplx (c,s);
    } else if(sgn_c == 1 && sgn_s == 0){
      return dcmplx (-c,s);
    }
    else if(sgn_c == 0 && sgn_s == 1){
      return dcmplx (c,-s);
    } else {
    return dcmplx (-c,-s);
    }

}

__device__ double cosC (double x){
    dcmplx expX = expC(dcmplx(0.,x));
    return expX.real();
}

__device__ double sinC (double x){
    dcmplx expX = expC(dcmplx(0.,x));
    return expX.imag();
}



__device__ double get_r_squared(double x, double y, double x_center, double y_center, double AA){   
  double xx = __dmul_rn(__dmul_rn(x - x_center, x - x_center), AA);   
  double yy = __dmul_rn(__dmul_rn(y - y_center, y - y_center), AA);   
  return (xx + yy);     
}

__device__ double get_r_squared_3d(double x, double y, double z, double x_center, double y_center, double z_center, double AA){   
  double xx = __dmul_rn(__dmul_rn(x - x_center, x - x_center), AA);   
  double yy = __dmul_rn(__dmul_rn(y - y_center, y - y_center), AA); 
  double zz = __dmul_rn(__dmul_rn(z - z_center, z - z_center), AA);    
  return (xx + yy + zz);     
}

__device__ double sign(double x){
  if (x>=0.){
    return 1.;
  } else {
    return -1.;
  }
}

__device__ double jumpsToNeighbor(int dim, int alpha, int beta){
  if (dim==0){
    if (beta==(alpha+1) && alpha%2==0){return -1.;} else {return 1.;}
  } else {
    if (beta==alpha){return -1.;} else {return 1.;}
  }
}

__device__ double jumpsFromNeighbor(int dim, int alpha, int beta){
  if (dim==0){
    if (beta==(alpha+1) && alpha%2==0){return -1.;} else {return 1.;}
  } else {
    if (beta==alpha){return -1.;} else {return 1.;}
  }
}


__device__ double jumpsOverBoundary(int dim, int alpha, int beta, int alpha_new, int beta_new){
  double sign = 1.;
  if (dim==0){
    if (abs((double)alpha-(double)alpha_new)>2.){sign = -sign;}
    if (abs((double)beta-(double)beta_new)>2.){sign = -sign;}
  } else {
    if (abs((double)alpha-(double)alpha_new)>1.){sign = -sign;}
    if (abs((double)beta-(double)beta_new)>1.){sign = -sign;}
  }
  return sign;
} 


__device__ double signFromStreamEpsilon(int n, int alpha, int beta, int Q, int direction){
  if (direction==1){ //Streaming 1s right
    if (alpha%2==n and beta%2==n){ //Both 1s streaming
      if (alpha==n){
        return -1.;
      } else {
        return 1.;
      }
    } else if (beta%2==n and beta==alpha+1 and beta!=1){ //Jumping
      return -1.;
    } else if (alpha==n and beta!=Q-1){ //Boundary jump
      return -1.;
    } else {
      return 1.;
    }
  } else { //Streaming 1s left
    if (alpha%2==n and beta%2==n){ //Both 1s streaming
      if (beta==Q-2+n){
        return -1.;
      } else {
        return 1.;
      }
    } else if (alpha%2==n and beta==alpha+1 and beta!=Q-1){ //Jumping
      return -1.;
    } else if (beta==Q-2+n and alpha!=0){ //Boundary jump
      return -1.;
    } else {
      return 1.;
    }
  }
}

__device__ double signFromStreamEpsilonY(int n, int alpha_y, int beta_y, int alpha_x, int beta_x, int L, int direction){
  if (direction==1){ //Streaming 1s right
    if (alpha_x%2==n and beta_x%2==n){ //Both 1s streaming
      if (alpha_y==0){
        return -1.;
      } else {
        return 1.;
      }
    } else if (beta_x%2==n and beta_y==alpha_y+1 and beta_y!=1){ //Jumping
      return -1.;
    } else if (alpha_y==n and beta_y!=L-1){ //Boundary jump
      return -1.;
    } else {
      return 1.;
    }
  } else { //Streaming 1s left
    if (alpha_x%2==n and beta_x%2==n){ //Both 1s streaming
      if (beta_y==L-1){
        return -1.;
      } else {
        return 1.;
      }
    } else if (alpha_x%2==n and beta_y==alpha_y+1 and beta_y!=L-1){ //Jumping
      return -1.;
    } else if (beta_y==L-1 and alpha_y!=0){ //Boundary jump
      return -1.;
    } else {
      return 1.;
    }
  }
}

__device__ int index_on_GPU(int i, int simSize, int numGPUs){
  int GPU = -1;
  while (i>=0){
    i -= simSize/numGPUs;
    GPU += 1;
  }
  return GPU;
}

__global__ void copy(dcmplx *QField, dcmplx *QField2, int* lattice){
    int xSize = lattice[0];
    int ySize = lattice[2];
    int zSize = lattice[4];
    int vectorSize = lattice[12];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    for (int i = 0; i < vectorSize; i ++){
      QField2[i+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = QField[i+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
    }
}

// Zero Fields
__global__ void zeroFields(dcmplx *QField, dcmplx *QField2, int* lattice) {
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int vectorSize = lattice[12];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;

  // SET NEW FIELD
  for (int i = 0; i < vectorSize; i ++){
    QField[i+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = 0.;
    QField2[i+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] = 0.;
  }
    
} 



__global__ void incrementTime(int* lattice){
    lattice[14] += 1;
}

__device__ double step(double x, double a)
{
  return x >= a;
}