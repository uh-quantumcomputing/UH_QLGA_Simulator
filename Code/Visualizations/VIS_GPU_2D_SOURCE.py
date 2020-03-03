#CUDA Code
from pycuda.compiler import SourceModule
gpuSource = SourceModule("""
#include <pycuda-complex.hpp>
#include <stdio.h>

typedef pycuda::complex<double> dcmplx;

const int vectorSize = 10;
const int spinComps = 5;
const double pi = 3.14159265358979323846264338328;

/* QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize] */


__global__ void test(dcmplx *QField, dcmplx *vortField, dcmplx *rhoField, int *lattice)
{

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


__global__ void aveVorticity(dcmplx *vortField, dcmplx *vortFieldAve, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int xL = (x-1+xSize)%(xSize);
  int xR = (x+1)%(xSize);
  int yL = (y-1+ySize)%(ySize);
  int yR = (y+1)%(ySize);
  vortFieldAve[y + x*ySize] =  1./9.*(vortField[yL + xL*ySize] +
                                      vortField[yL + xR*ySize] +
                                      vortField[yL + x*ySize] +
                                      vortField[yR + xL*ySize] +
                                      vortField[yR + xR*ySize] +
                                      vortField[yR + x*ySize] +
                                      vortField[y + xL*ySize] +
                                      vortField[y + xR*ySize] +
                                      vortField[y + x*ySize]);  
}

__global__ void getPlotDetailsPhaseAndDensity(dcmplx *QField, dcmplx *vortField, dcmplx *rhoField, dcmplx *bosonField, double *phaseField, dcmplx *VxField, dcmplx *VyField, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];

  /* For Vorticity */
  double phiRightAbove[ spinComps ];
  double phiRightBelow[ spinComps ];
  double phiLeftAbove[ spinComps ];
  double phiLeftBelow[ spinComps ];

  /* For Velocity */
  double phiRight[ spinComps ];
  double phiLeft[ spinComps ];
  double phiAbove[ spinComps ];
  double phiBelow[ spinComps ];
  

  double Vx = 0.;
  double Vy = 0.;
  double VxAbove = 0.;
  double VxBelow = 0.;
  double VyLeft = 0.;
  double VyRight = 0.;
  dcmplx i(0.,1.);
  int xLeft = (x-1+xSize)%(xSize);
  int xRight = (x+1)%(xSize);
  int yBelow = (y-1+ySize)%(ySize);
  int yAbove = (y+1)%(ySize);

  phaseField[y+x*ySize] = 0;

  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }

  for(n = 0; n < spinComps; n++){
    phi[n] = QField[2*n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
    
    /* For Vorticity */
    phiRightAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
    phiRightBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);         
    phiLeftAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);
    phiLeftBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);  

    /* For Velocity */
    phiRight[n] = phase(QField[2*n+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
    phiLeft[n] = phase(QField[2*n+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);         
    phiAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]);
    phiBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]);   
  
    /* For Phase */
    phaseField[y+x*ySize] += phase(phi[n]);
  }

  while(phaseField[y+x*ySize] > 2.*pi){
    phaseField[y+x*ySize] -= 2.*pi;
  } 

  rhoField[y+x*ySize] = 0;

  for(n = 0; n < vectorSize; n++){
    rhoField[y+x*ySize] += psi[n]*conj(psi[n]);
  }
  bosonField[y+x*ySize] = 0;
  for(n = 0; n < spinComps; n++){
    bosonField[y+x*ySize] += phi[n]*conj(phi[n]);
  }
  for(n = 0; n < spinComps; n++){     
      Vx += (1./2.)*phaseSubtract(phiRight[n], phiLeft[n]);
      Vy += (1./2.)*phaseSubtract(phiAbove[n], phiBelow[n]);     
      VxAbove += (1./2.)*phaseSubtract(phiRightAbove[n], phiLeftAbove[n]);
      VxBelow += (1./2.)*phaseSubtract(phiRightBelow[n], phiLeftBelow[n]);
      VyLeft += (1./2.)*phaseSubtract(phiLeftAbove[n], phiLeftBelow[n]);
      VyRight += (1./2.)*phaseSubtract(phiRightAbove[n], phiRightBelow[n]);
  }
  VxField[z+y*zSize+x*zSize*ySize] = Vx;
  VyField[z+y*zSize+x*zSize*ySize] = Vy;
  vortField[z+y*zSize+x*zSize*ySize] = (1./2.)*((VyRight - VyLeft)-(VxAbove-VxBelow));

}

__global__ void getPlotDetailsVorticity(dcmplx *QField, dcmplx *vortField, dcmplx *rhoField, dcmplx *bosonField, double *phaseField, dcmplx *VxField, dcmplx *VyField, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];

  /* For Vorticity */
  double phiRightAbove[ spinComps ];
  double phiRightBelow[ spinComps ];
  double phiLeftAbove[ spinComps ];
  double phiLeftBelow[ spinComps ];

  /* For Velocity */
  double phiRight[ spinComps ];
  double phiLeft[ spinComps ];
  double phiAbove[ spinComps ];
  double phiBelow[ spinComps ];
  

  double Vx [ spinComps ];
  double Vy [ spinComps ];
  double VxAbove [ spinComps ];
  double VxBelow [ spinComps ];
  double VyLeft [ spinComps ];
  double VyRight [ spinComps ];
  dcmplx i(0.,1.);
  int xLeft = (x-1+xSize)%(xSize);
  int xRight = (x+1)%(xSize);
  int yBelow = (y-1+ySize)%(ySize);
  int yAbove = (y+1)%(ySize);

  phaseField[y+x*ySize] = 0;

  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }

  for(n = 0; n < spinComps; n++){
    phi[n] = QField[2*n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
    
    /* For Vorticity */
    phiRightAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
    phiRightBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);         
    phiLeftAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);
    phiLeftBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);  

    /* For Velocity */
    phiRight[n] = phase(QField[2*n+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
    phiLeft[n] = phase(QField[2*n+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);         
    phiAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]);
    phiBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]);   
  
    /* For Phase */
    phaseField[y+x*ySize] += phase(phi[n]);
  }

  while(phaseField[y+x*ySize] > 2.*pi){
    phaseField[y+x*ySize] -= 2.*pi;
  } 

  rhoField[y+x*ySize] = 0;

  for(n = 0; n < vectorSize; n++){
    rhoField[y+x*ySize] += psi[n]*conj(psi[n]);
  }
  for(n = 0; n < spinComps; n++){
  bosonField[n+y*spinComps+x*ySize*spinComps] = 0;
  }
  for(n = 0; n < spinComps; n++){
    bosonField[n+y*spinComps+x*ySize*spinComps] = phi[n]*conj(phi[n]);
  }
  for(n = 0; n < spinComps; n++){     
      Vx[n]= (1./2.)*phaseSubtract(phiRight[n], phiLeft[n]);
      Vy[n]= (1./2.)*phaseSubtract(phiAbove[n], phiBelow[n]);     
      VxAbove[n] += (1./2.)*phaseSubtract(phiRightAbove[n], phiLeftAbove[n]);
      VxBelow[n]= (1./2.)*phaseSubtract(phiRightBelow[n], phiLeftBelow[n]);
      VyLeft[n]= (1./2.)*phaseSubtract(phiLeftAbove[n], phiLeftBelow[n]);
      VyRight[n]= (1./2.)*phaseSubtract(phiRightAbove[n], phiRightBelow[n]);
  }
  VxField[z+y*zSize+x*zSize*ySize] = Vx[0];
  VyField[z+y*zSize+x*zSize*ySize] = Vy[0];
  for (n = 0; n < spinComps; n++){ 
    vortField[n + y*spinComps+x*spinComps*ySize] = (1./2.)*((VyRight[n] - VyLeft[n])-(VxAbove[n]-VxBelow[n]));
  }
}









__global__ void getPlotDetailsMayavi(dcmplx *QField, dcmplx *rhoField, double *phaseField, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int comp = lattice[6];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];
  
  dcmplx i(0.,1.);

  phaseField[y+x*ySize] = 0;

  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }

  for(n = 0; n < spinComps; n++){
    phi[n] = QField[2*n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];   
  }
  /* For Phase */
  phaseField[y+x*ySize] = phase(phi[comp]);

  while(phaseField[y+x*ySize] > 2.*pi){
    phaseField[y+x*ySize] -= 2.*pi;
  } 

  rhoField[y+x*ySize] = psi[2*comp]*conj(psi[2*comp]) + psi[2*comp+1]*conj(psi[2*comp+1]);

}


__global__ void getPlotDetailsMayavi_three_d(dcmplx *QField, dcmplx *rhoField, double *phaseField, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int comp = lattice[6];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];
  
  dcmplx i(0.,1.);

  phaseField[z+y*zSize+x*ySize*zSize] = 0;

  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }

  for(n = 0; n < spinComps; n++){
    phi[n] = QField[2*n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];   
  }
  /* For Phase */
  phaseField[z+y*zSize+x*ySize*zSize] = phase(phi[comp]);

  while(phaseField[z+y*zSize+x*ySize*zSize] > 2.*pi){
    phaseField[z+y*zSize+x*ySize*zSize] -= 2.*pi;
  } 

  rhoField[z+y*zSize+x*ySize*zSize] = psi[2*comp]*conj(psi[2*comp]) + psi[2*comp+1]*conj(psi[2*comp+1]);

}


__global__ void getPlotDetails(dcmplx *QField, dcmplx *vortField, dcmplx *rhoField, double *phaseField, dcmplx *VxField, dcmplx *VyField, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];

  /* For Vorticity */
  double phiRightAbove[ spinComps ];
  double phiRightBelow[ spinComps ];
  double phiLeftAbove[ spinComps ];
  double phiLeftBelow[ spinComps ];

  /* For Velocity */
  double phiRight[ spinComps ];
  double phiLeft[ spinComps ];
  double phiAbove[ spinComps ];
  double phiBelow[ spinComps ];
  

  double Vx = 0.;
  double Vy = 0.;
  double VxAbove = 0.;
  double VxBelow = 0.;
  double VyLeft = 0.;
  double VyRight = 0.;
  dcmplx i(0.,1.);
  int xLeft = (x-1+xSize)%(xSize);
  int xRight = (x+1)%(xSize);
  int yBelow = (y-1+ySize)%(ySize);
  int yAbove = (y+1)%(ySize);

  phaseField[y+x*ySize] = 0;

  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }

  for(n = 0; n < spinComps; n++){
    phi[n] = QField[2*n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
    
    /* For Vorticity */
    phiRightAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
    phiRightBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);         
    phiLeftAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);
    phiLeftBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);  

    /* For Velocity */
    phiRight[n] = phase(QField[2*n+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
    phiLeft[n] = phase(QField[2*n+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);         
    phiAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]);
    phiBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]);   
  
    /* For Phase */
    phaseField[y+x*ySize] += phase(phi[n]);
  }

  while(phaseField[y+x*ySize] > 2.*pi){
    phaseField[y+x*ySize] -= 2.*pi;
  } 

  rhoField[y+x*ySize] = 0;
  for(n = 0; n < vectorSize; n++){
    rhoField[y+x*ySize] += psi[n]*conj(psi[n]);
  }

  for(n = 0; n < spinComps; n++){     
      Vx += (1./2.)*phaseSubtract(phiRight[n], phiLeft[n]);
      Vy += (1./2.)*phaseSubtract(phiAbove[n], phiBelow[n]);     
      VxAbove += (1./2.)*phaseSubtract(phiRightAbove[n], phiLeftAbove[n]);
      VxBelow += (1./2.)*phaseSubtract(phiRightBelow[n], phiLeftBelow[n]);
      VyLeft += (1./2.)*phaseSubtract(phiLeftAbove[n], phiLeftBelow[n]);
      VyRight += (1./2.)*phaseSubtract(phiRightAbove[n], phiRightBelow[n]);
  }
  VxField[z+y*zSize+x*zSize*ySize] = Vx;
  VyField[z+y*zSize+x*zSize*ySize] = Vy;
  vortField[z+y*zSize+x*zSize*ySize] = (1./2.)*((VyRight - VyLeft)-(VxAbove-VxBelow));

}

__global__ void getPlotDetailsDS(dcmplx *QField, dcmplx *vortField, dcmplx *rhoField, double *phaseField, dcmplx *VxField, dcmplx *VyField, dcmplx *rhoFieldComp, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];

  /* For Vorticity */
  double phiRightAbove[ spinComps ];
  double phiRightBelow[ spinComps ];
  double phiLeftAbove[ spinComps ];
  double phiLeftBelow[ spinComps ];

  /* For Velocity */
  double phiRight[ spinComps ];
  double phiLeft[ spinComps ];
  double phiAbove[ spinComps ];
  double phiBelow[ spinComps ];
  

  double Vx = 0.;
  double Vy = 0.;
  double VxAbove = 0.;
  double VxBelow = 0.;
  double VyLeft = 0.;
  double VyRight = 0.;
  dcmplx i(0.,1.);
  int xLeft = (x-1+xSize)%(xSize);
  int xRight = (x+1)%(xSize);
  int yBelow = (y-1+ySize)%(ySize);
  int yAbove = (y+1)%(ySize);

  phaseField[y+x*ySize] = 0;

  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }

  for(n = 0; n < spinComps; n++){
    phi[n] = QField[2*n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
    
    /* For Vorticity */
    phiRightAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
    phiRightBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);         
    phiLeftAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);
    phiLeftBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);  

    /* For Velocity */
    phiRight[n] = phase(QField[2*n+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
    phiLeft[n] = phase(QField[2*n+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);         
    phiAbove[n] = phase(QField[2*n+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]);
    phiBelow[n] = phase(QField[2*n+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]
              +QField[2*n+1+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]);   
  
    /* For Phase */
    phaseField[y+x*ySize] += phase(phi[n]);
  }

  while(phaseField[y+x*ySize] > 2.*pi){
    phaseField[y+x*ySize] -= 2.*pi;
  } 

  rhoField[y+x*ySize] = 0;
  for(n = 0; n < vectorSize; n++){
    rhoField[y+x*ySize] += psi[n]*conj(psi[n]);
  }

  rhoFieldComp[n+y*vectorSize+x*ySize*vectorSize] = 0;
  for(n = 0; n < vectorSize; n++){
    rhoFieldComp[n+y*vectorSize+x*ySize*vectorSize] += psi[n]*conj(psi[n]);
  }

  for(n = 0; n < spinComps; n++){     
      Vx += (1./2.)*phaseSubtract(phiRight[n], phiLeft[n]);
      Vy += (1./2.)*phaseSubtract(phiAbove[n], phiBelow[n]);     
      VxAbove += (1./2.)*phaseSubtract(phiRightAbove[n], phiLeftAbove[n]);
      VxBelow += (1./2.)*phaseSubtract(phiRightBelow[n], phiLeftBelow[n]);
      VyLeft += (1./2.)*phaseSubtract(phiLeftAbove[n], phiLeftBelow[n]);
      VyRight += (1./2.)*phaseSubtract(phiRightAbove[n], phiRightBelow[n]);
  }
  VxField[z+y*zSize+x*zSize*ySize] = Vx;
  VyField[z+y*zSize+x*zSize*ySize] = Vy;
  vortField[z+y*zSize+x*zSize*ySize] = (1./2.)*((VyRight - VyLeft)-(VxAbove-VxBelow));

}

__global__ void getPlotDetailsForComponent(dcmplx *QField, dcmplx *vortField, dcmplx *rhoField, dcmplx *phaseField, dcmplx *VxField, dcmplx *VyField, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int component = lattice[6];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];

  /* For Vorticity */
  double phiRightAbove[ spinComps ];
  double phiRightBelow[ spinComps ];
  double phiLeftAbove[ spinComps ];
  double phiLeftBelow[ spinComps ];

  /* For Velocity */
  double phiRight[ spinComps ];
  double phiLeft[ spinComps ];
  double phiAbove[ spinComps ];
  double phiBelow[ spinComps ];
  

  double Vx = 0.;
  double Vy = 0.;
  double VxAbove = 0.;
  double VxBelow = 0.;
  double VyLeft = 0.;
  double VyRight = 0.;
  dcmplx i(0.,1.);
  int xLeft = (x-1+xSize)%(xSize);
  int xRight = (x+1)%(xSize);
  int yBelow = (y-1+ySize)%(ySize);
  int yAbove = (y+1)%(ySize);

  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }

  phi[component] = QField[2*component+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];


  /* For Vorticity */
  phiRightAbove[component] = phase(QField[2*component+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+yAbove*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
  phiRightBelow[component] = phase(QField[2*component+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+yBelow*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);         
  phiLeftAbove[component] = phase(QField[2*component+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+yAbove*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);
  phiLeftBelow[component] = phase(QField[2*component+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+yBelow*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);  

  /* For Velocity */
  phiRight[component] = phase(QField[2*component+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+y*vectorSize*zSize+xRight*zSize*ySize*vectorSize]);
  phiLeft[component] = phase(QField[2*component+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+y*vectorSize*zSize+xLeft*zSize*ySize*vectorSize]);         
  phiAbove[component] = phase(QField[2*component+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+yAbove*vectorSize*zSize+x*zSize*ySize*vectorSize]);
  phiBelow[component] = phase(QField[2*component+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+yBelow*vectorSize*zSize+x*zSize*ySize*vectorSize]);   

  /* For Phase */
  phaseField[y+x*ySize] = phase(phi[component]);

   

  rhoField[y+x*ySize] = 0;
  for(n = 2*component; n <= 2*component + 1; n++){
  rhoField[y+x*ySize] += psi[n]*conj(psi[n]);
  }

  Vx = (1./2.)*phaseSubtract(phiRight[component], phiLeft[component]);
  Vy = (1./2.)*phaseSubtract(phiAbove[component], phiBelow[component]);     
  VxAbove = (1./2.)*phaseSubtract(phiRightAbove[component], phiLeftAbove[component]);
  VxBelow = (1./2.)*phaseSubtract(phiRightBelow[component], phiLeftBelow[component]);
  VyLeft = (1./2.)*phaseSubtract(phiLeftAbove[component], phiLeftBelow[component]);
  VyRight = (1./2.)*phaseSubtract(phiRightAbove[component], phiRightBelow[component]);
  

  VxField[z+y*zSize+x*zSize*ySize] = Vx;
  VyField[z+y*zSize+x*zSize*ySize] = Vy;
  vortField[z+y*zSize+x*zSize*ySize] = (1./2.)*((VyRight - VyLeft)-(VxAbove-VxBelow));

}

__global__ void getPlotDetailsForMF(dcmplx *QField, dcmplx *rhoField, double *phaseField, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int component = lattice[6];
  //printf("mf = %d", component);
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];

  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }

  phi[component] = QField[2*component+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]
            +QField[2*component+1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];

  /* For Phase */
  phaseField[y+x*ySize] = phase(phi[component]);

   

  rhoField[y+x*ySize] = 0;
  for(n = 2*component; n <= 2*component + 1; n++){
  rhoField[y+x*ySize] += psi[n]*conj(psi[n]);
  }

}

__global__ void getTotalRho(dcmplx *QField, dcmplx *rhoField, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int component = lattice[6];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];
  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }
  rhoField[y+x*ySize] = 0;
  for(n = 0; n <= vectorSize ; n++){
  rhoField[y+x*ySize] += psi[n]*conj(psi[n]);
  }
  }


__global__ void getTotalRhoComp(dcmplx *QField, dcmplx *rhoField, int *lattice)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int component = lattice[6];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ spinComps ];
  dcmplx psi[ vectorSize ];
  for(n = 0; n < vectorSize; n++){
    psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
  }
  rhoField[y+x*ySize] = 0;
  for(n = 2*component; n <= 2*component + 1 ; n++){
    rhoField[y+x*ySize] += psi[n]*conj(psi[n]);
  }

}



""")