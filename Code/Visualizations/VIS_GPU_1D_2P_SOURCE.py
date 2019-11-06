#CUDA Code
from pycuda.compiler import SourceModule
gpuSource = SourceModule("""
#include <pycuda-complex.hpp>
#include <stdio.h>

typedef pycuda::complex<double> dcmplx;

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

const int spinComps = 1;
const double pi = 3.14159265358979323846264338328;

/* QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] */

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

__device__ index_pair numbers_to_positions(int Q, int alpha, int beta){
  int x1 = alpha/2;
  int x2 = beta/2;
  return {x1,x2}; //position
}

__device__ index_quad num_to2d_num(int Qx, int alpha, int beta){
  return {alpha%Qx,alpha/(Qx),beta%Qx,beta/(Qx)};
}


__device__ __forceinline__ double phase (dcmplx z)
{   
    double R = atan2(z.imag(), z.real());
    return R + pi;

}

__device__ __forceinline__ double phaseSubtract (double phase1, double phase2)
{   
    bool edgeCase1 = (phase1 < pi/2. && phase2 > 3.*pi/2.);
    bool edgeCase2 = (phase2 < pi/2. && phase1 > 3.*pi/2.);
    double phaseDiff = phase1 - phase2;
    if (edgeCase1){
      return 2.*pi + phaseDiff; 
    } else if (edgeCase2){
      return phaseDiff - 2.*pi;
    } else {
      return phaseDiff;
    }
    
}


__global__ void getQS_x1_x2(dcmplx *QField, double *QF_x1_x2_real, double *QF_x1_x2_imag, int *lattice){
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int x_nSize = xSize*(2*xSize-1);
  int y_nSize = ySize*(2*ySize-1);
  int z_nSize = zSize*(2*zSize-1);
  int n = 0;
  index_pair iToNum = index_to_number(2*xSize, x);
  int alpha = iToNum.index0;
  int beta = iToNum.index1;
  int x1 = alpha/2;
  int x2 = beta/2;

  atomicAdd(&QF_x1_x2_real[x2+x1*xSize], 0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].real());
  atomicAdd(&QF_x1_x2_imag[x2+x1*xSize], 0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].imag());
  atomicAdd(&QF_x1_x2_real[x1+x2*xSize], -0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].real());
  atomicAdd(&QF_x1_x2_imag[x1+x2*xSize], -0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].imag());
  
  // QF_x1_x2[x2+x1*xSize] += QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps];
  // QF_x1_x2[x1+x2*xSize] -= QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps];
}

__global__ void getQS_after_position_measurement_true(dcmplx *QField, double *QF_x1_x2_real, double *QF_x1_x2_imag, int *lattice){
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x_measure = lattice[6];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int x_nSize = xSize*(2*xSize-1);
  int y_nSize = ySize*(2*ySize-1);
  int z_nSize = zSize*(2*zSize-1);
  int n = 0;
  index_pair iToNum = index_to_number(2*xSize, x);
  int alpha = iToNum.index0;
  int beta = iToNum.index1;
  int x1 = alpha/2;
  int x2 = beta/2;

  if (x1==x_measure || x2==x_measure){
    atomicAdd(&QF_x1_x2_real[x2+x1*xSize], 0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].real());
    atomicAdd(&QF_x1_x2_imag[x2+x1*xSize], 0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].imag());
    atomicAdd(&QF_x1_x2_real[x1+x2*xSize], -0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].real());
    atomicAdd(&QF_x1_x2_imag[x1+x2*xSize], -0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].imag());
  }
  
  // QF_x1_x2[x2+x1*xSize] += QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps];
  // QF_x1_x2[x1+x2*xSize] -= QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps];
}

__global__ void getQS_after_position_measurement_false(dcmplx *QField, double *QF_x1_x2_real, double *QF_x1_x2_imag, int *lattice){
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x_measure = lattice[6];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int x_nSize = xSize*(2*xSize-1);
  int y_nSize = ySize*(2*ySize-1);
  int z_nSize = zSize*(2*zSize-1);
  int n = 0;
  index_pair iToNum = index_to_number(2*xSize, x);
  int alpha = iToNum.index0;
  int beta = iToNum.index1;
  int x1 = alpha/2;
  int x2 = beta/2;

  if (x1!=x_measure && x2!=x_measure){
    atomicAdd(&QF_x1_x2_real[x2+x1*xSize], 0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].real());
    atomicAdd(&QF_x1_x2_imag[x2+x1*xSize], 0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].imag());
    atomicAdd(&QF_x1_x2_real[x1+x2*xSize], -0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].real());
    atomicAdd(&QF_x1_x2_imag[x1+x2*xSize], -0.5*QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps].imag());
  }
  
  // QF_x1_x2[x2+x1*xSize] += QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps];
  // QF_x1_x2[x1+x2*xSize] -= QField[n+z*spinComps+y*spinComps*z_nSize+x*z_nSize*y_nSize*spinComps];
}

__global__ void getRho_projected(double *QF_x1_x2_real, double *QF_x1_x2_imag, double *QF_projected_real, int *lattice){
  int xSize = lattice[0]; // x1 
  int ySize = lattice[0]; // x2 
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n = 0;

  double rho = QF_x1_x2_real[n+z*spinComps+y*spinComps*zSize+x*zSize*ySize*spinComps]*QF_x1_x2_real[n+z*spinComps+y*spinComps*zSize+x*zSize*ySize*spinComps]+QF_x1_x2_imag[n+z*spinComps+y*spinComps*zSize+x*zSize*ySize*spinComps]*QF_x1_x2_imag[n+z*spinComps+y*spinComps*zSize+x*zSize*ySize*spinComps];
  atomicAdd(&QF_projected_real[x], rho);

}

__global__ void getQS_r1_r2_2d(dcmplx *QField, double *QF_r1_r2_real, double *QF_r1_r2_imag, int *lattice){
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int x_nSize = xSize*ySize*(2*xSize*ySize-1);
  int n = 0;
  index_pair iToNum = index_to_number(2*xSize*ySize, x);
  int alpha = iToNum.index0;
  int beta = iToNum.index1;
  index_quad alphaBeta2d = num_to2d_num(2*xSize, alpha, beta);
  int alpha_x = alphaBeta2d.index0;
  int alpha_y = alphaBeta2d.index1;
  int beta_x = alphaBeta2d.index2;
  int beta_y = alphaBeta2d.index3;
  //printf("ind=%d,alpha_x=%d,alpha_y=%d,beta_x=%d,beta_y=%d || ",x,alpha_x,alpha_y,beta_x,beta_y);
  int x1 = alpha_x/2;
  int y1 = alpha_y;
  int x2 = beta_x/2;
  int y2 = beta_y;
  //printf("ind=%d,x1=%d,y1=%d,x2=%d,y2=%d || ",x,x1,y1,x2,y2);
  //printf("4d ind = %d || ",y2+x2*ySize+y1*ySize*xSize+x1*ySize*ySize*xSize);

  atomicAdd(&QF_r1_r2_real[y2+x2*ySize+y1*ySize*xSize+x1*ySize*xSize*ySize], 0.5*QField[x].real());
  atomicAdd(&QF_r1_r2_imag[y2+x2*ySize+y1*ySize*xSize+x1*ySize*xSize*ySize], 0.5*QField[x].imag());
  atomicAdd(&QF_r1_r2_real[y1+x1*ySize+y2*ySize*xSize+x2*ySize*xSize*ySize], -0.5*QField[x].real());
  atomicAdd(&QF_r1_r2_imag[y1+x1*ySize+y2*ySize*xSize+x2*ySize*xSize*ySize], -0.5*QField[x].imag());

  
}

__global__ void getRho_projected_2d(double *QF_r1_r2_real, double *QF_r1_r2_imag, double *rho_projected_real, int *lattice){
  int xSize = lattice[0]; // Lx 
  int ySize = lattice[2]; // Ly
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x; //x2+x1*xSize
  int y = blockIdx.y * blockDim.y + threadIdx.y; //y2+y1*ySize
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int x1 = x/xSize;
  int x2 = x%xSize;
  int y1 = y/ySize;
  int y2 = y%ySize;

  double rho = QF_r1_r2_real[y2+x2*ySize+y1*ySize*xSize+x1*ySize*ySize*xSize]*QF_r1_r2_real[y2+x2*ySize+y1*ySize*xSize+x1*ySize*ySize*xSize]
              +QF_r1_r2_imag[y2+x2*ySize+y1*ySize*xSize+x1*ySize*ySize*xSize]*QF_r1_r2_imag[y2+x2*ySize+y1*ySize*xSize+x1*ySize*ySize*xSize];
  atomicAdd(&rho_projected_real[y1+x1*ySize], rho);

}


__global__ void calcPurity(double *QF_x1_x2_real, double *QF_x1_x2_imag, double *purity, int *lattice){
  int xSize = lattice[0]; // x1 
  int ySize = lattice[0]; // x2 
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n = 0;
  dcmplx i(0.,1.);
  dcmplx purity_add(0.,0.);
  for (int x3 = 0; x3<blockDim.x*gridDim.x; x3++){
    for (int x4 = 0; x4<blockDim.y*gridDim.y; x4++){
      purity_add += (QF_x1_x2_real[n+z*spinComps+y*spinComps*zSize+x*zSize*ySize*spinComps]+i*QF_x1_x2_imag[n+z*spinComps+y*spinComps*zSize+x*zSize*ySize*spinComps])*(QF_x1_x2_real[n+z*spinComps+y*spinComps*zSize+x3*zSize*ySize*spinComps]
                    -i*QF_x1_x2_imag[n+z*spinComps+y*spinComps*zSize+x3*zSize*ySize*spinComps])*(QF_x1_x2_real[n+z*spinComps+x4*spinComps*zSize+x3*zSize*ySize*spinComps]
                    +i*QF_x1_x2_imag[n+z*spinComps+x4*spinComps*zSize+x3*zSize*ySize*spinComps])*(QF_x1_x2_real[n+z*spinComps+x4*spinComps*zSize+x*zSize*ySize*spinComps]-i*QF_x1_x2_imag[n+z*spinComps+x4*spinComps*zSize+x*zSize*ySize*spinComps]);
    }
  }

  atomicAdd(&purity[0], purity_add.real());

}


/////////////////////////////////////////////////////////
/////////////*  Calculate analytic field  *//////////////
/////////////////////////////////////////////////////////
__device__ dcmplx phi(double x,double t, dcmplx *An, dcmplx *Bn,double L){
  dcmplx val = 0.;
  dcmplx i(0.,1.);
  for (int l = 0; l<L; l++){
    double n = (double) l;
    val += (An[l]*cos(2.*pi*(n)*x/L)+Bn[l]*sin(2.*pi*(n)*x/L))*exp(-i*(4.*pi*pi*t)*((n)*(n))/(L*L));
  }
  return val;
}

/*
__device__ dcmplx phi(double x,double t,dcmplx *An,dcmplx *Bn,double L){
  dcmplx val = 0.;
  dcmplx i(0.,1.);
  for (int l = 0; l<L; l++){
    double n = (double) l;
    val += (An[l]*exp(i*(2.*pi)*(n)*x/L)+Bn[l]*exp(-i*(2.*pi)*(n)*x/L))*exp(-i*(4.*pi*pi*t)*((n)*(n))/(L*L));
  }
  return val;
}
*/



__device__ dcmplx psi_odd(double x, double y, double t, dcmplx *An, dcmplx *Bn, dcmplx *An2, dcmplx *Bn2, double L){
  return (1./sqrt(2.))*(phi(x,t,An,Bn,L)*phi(y,t,An2,Bn2,L) - phi(y,t,An,Bn,L)*phi(x,t,An2,Bn2,L));
}


__global__ void getAnalyticField(double *QFieldAnalyticReal, double *QFieldAnalyticImag, dcmplx *An, dcmplx *Bn, dcmplx *An2, dcmplx *Bn2, int *lattice, double *time){
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  double t = time[0]; //t*t_scale
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int x_nSize = xSize*(2*xSize-1);
  int y_nSize = ySize*(2*ySize-1);
  int z_nSize = zSize*(2*zSize-1);
  index_pair iToNum = index_to_number(2*xSize, x);
  int alpha = iToNum.index0;
  int beta = iToNum.index1;
  int x1 = alpha/2;
  int x2 = beta/2;

  dcmplx field = psi_odd((double)x1,(double)x2,(double)t,An,Bn,An2,Bn2,(double)xSize);
  atomicAdd(&QFieldAnalyticReal[x2+x1*xSize], 0.25*field.real());
  atomicAdd(&QFieldAnalyticImag[x2+x1*xSize], 0.25*field.imag());
  atomicAdd(&QFieldAnalyticReal[x1+x2*xSize], -0.25*field.real());
  atomicAdd(&QFieldAnalyticImag[x1+x2*xSize], -0.25*field.imag());

}

""")