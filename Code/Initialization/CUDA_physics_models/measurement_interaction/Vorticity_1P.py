def get_CUDA(dimensions, vectorSize):
  return'''
  __device__  double density (dcmplx z)
{   
    dcmplx rho = Mul(z,conj(z));
    return rho.real();

}

__device__  double phase_vort (dcmplx z)
{   
    if (z.imag() == 0. && z.real()==0.){
    return 0.;
    }
    double R = atan2(z.imag(), z.real());
    return R + pi;

}

__device__ __forceinline__ double phaseSubtract (double phase1, double phase2)
{   
    double phaseDiff = phase1 - phase2;
    bool edgeCase1 = (phaseDiff<-pi);
    bool edgeCase2 = (phaseDiff>pi);
    if (edgeCase1){
      return 2.*pi + phaseDiff; 
    } else if (edgeCase2){
      return phaseDiff - 2.*pi;
    } else {
      return phaseDiff;
    }
    
}
__global__ void measurement(dcmplx *QField, int *vortField, int *lattice, int *gpu_params)
{
  int xSize = lattice[0];
  int ySize = lattice[2];
  int zSize = lattice[4];
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  int n;
  dcmplx phi[ 5 ];
  dcmplx psi[ 10 ];

  /* For Vorticity */
  double phiRightAbove[ 5 ];
  double phiRightBelow[ 5 ];
  double phiLeftAbove[ 5 ];
  double phiLeftBelow[ 5 ];

  /* For density */
  dcmplx rho;
  dcmplx rhoRightAbove[ 5 ];
  dcmplx rhoRightBelow[ 5 ];
  dcmplx rhoLeftAbove[ 5 ];
  dcmplx rhoLeftBelow[ 5 ];
  

  double Vx [ 5 ];
  double Vy [ 5 ];
  double VxAbove [ 5 ];
  double VxBelow [ 5 ];
  double VyLeft [ 5 ];
  double VyRight [ 5 ];
  dcmplx i(0.,1.);
  int xLeft = (x-1+xSize)%(xSize);
  int xRight = (x+1)%(xSize);
  int yBelow = (y-1+ySize)%(ySize);
  int yAbove = (y+1)%(ySize);

  for(n = 0; n < 5; n++){
      phiRightAbove[n] = phase_vort(QField[2*n+z*10+yAbove*10*zSize+xRight*zSize*ySize*10]
                +QField[2*n+1+z*10+yAbove*10*zSize+xRight*zSize*ySize*10]);
      phiRightBelow[n] = phase_vort(QField[2*n+z*10+yBelow*10*zSize+xRight*zSize*ySize*10]
                +QField[2*n+1+z*10+yBelow*10*zSize+xRight*zSize*ySize*10]);         
      phiLeftAbove[n] = phase_vort(QField[2*n+z*10+yAbove*10*zSize+xLeft*zSize*ySize*10]
                +QField[2*n+1+z*10+yAbove*10*zSize+xLeft*zSize*ySize*10]);
      phiLeftBelow[n] = phase_vort(QField[2*n+z*10+yBelow*10*zSize+xLeft*zSize*ySize*10]
                +QField[2*n+1+z*10+yBelow*10*zSize+xLeft*zSize*ySize*10]);
    }
  for(n = 0; n < 5; n++){ 
      rhoRightAbove[n] = density(QField[2*n+z*10+yAbove*10*zSize+xRight*zSize*ySize*10]
                +QField[2*n+1+z*10+yAbove*10*zSize+xRight*zSize*ySize*10]);
      rhoRightBelow[n] = density(QField[2*n+z*10+yBelow*10*zSize+xRight*zSize*ySize*10]
                +QField[2*n+1+z*10+yBelow*10*zSize+xRight*zSize*ySize*10]);         
      rhoLeftAbove[n] = density(QField[2*n+z*10+yAbove*10*zSize+xLeft*zSize*ySize*10]
                +QField[2*n+1+z*10+yAbove*10*zSize+xLeft*zSize*ySize*10]);
      rhoLeftBelow[n] = density(QField[2*n+z*10+yBelow*10*zSize+xLeft*zSize*ySize*10]
                +QField[2*n+1+z*10+yBelow*10*zSize+xLeft*zSize*ySize*10]);
  } 

  for(n = 0; n < 5; n++){         
      VxAbove[n] = (1./2.)*phaseSubtract(phiRightAbove[n], phiLeftAbove[n]);
      VxBelow[n] = (1./2.)*phaseSubtract(phiRightBelow[n], phiLeftBelow[n]);
      VyLeft[n] = (1./2.)*phaseSubtract(phiLeftAbove[n], phiLeftBelow[n]);
      VyRight[n] = (1./2.)*phaseSubtract(phiRightAbove[n], phiRightBelow[n]);
  }
  
  for (n = 0; n < 5; n++){ 
    int count = 0;
    double vort = (1./2.)*((VyRight[n] - VyLeft[n])-(VxAbove[n]-VxBelow[n]));
    if (vort > .2){
      rho = density(QField[2*n+z*10+y*10*zSize+x*zSize*ySize*10]
                +QField[2*n+1+z*10+y*10*zSize+x*zSize*ySize*10]);
      if (rho.real() < rhoLeftAbove[n].real()){
        count++;
      }
      if (rho.real() < rhoRightAbove[n].real()){
          count++;
      }
      if (rho.real() < rhoRightBelow[n].real()){
            count++;
      }
      if (rho.real() < rhoLeftBelow[n].real()){
              count++;
      }
      if (count > 3){
        vortField[2*(2*n + z*10 + y*10*zSize+xLeft*10*ySize*zSize)] = 1;
        vortField[2*(2*n + z*10 + y*10*zSize+xRight*10*ySize*zSize)] = 1;
        vortField[2*(2*n + z*10 + yAbove*10*zSize+x*10*ySize*zSize)] = 1;
        vortField[2*(2*n + z*10 + yBelow*10*zSize+x*10*ySize*zSize)] = 1;
        vortField[2*(2*n + z*10 + y*10*zSize+x*10*ySize*zSize)] = 1;
      }
    }
    count = 0;
    if (vort < -.2){
      rho = density(QField[2*n+z*10+y*10*zSize+x*zSize*ySize*10]
                +QField[2*n+1+z*10+y*10*zSize+x*zSize*ySize*10]);
      if (rho.real() < rhoLeftAbove[n].real()){
        count++;
      }
      if (rho.real() < rhoRightAbove[n].real()){
          count++;
      }
      if (rho.real() < rhoRightBelow[n].real()){
            count++;
      }
      if (rho.real() < rhoLeftBelow[n].real()){
           count++;
      }
      if (count > 3){
        vortField[2*(2*n + 1 + z*10 + y*10*zSize+xLeft*10*ySize*zSize)] = 1;
        vortField[2*(2*n + 1 + z*10 + y*10*zSize+xRight*10*ySize*zSize)] = 1;
        vortField[2*(2*n + 1 + z*10 + yAbove*10*zSize+x*10*ySize*zSize)] = 1;
        vortField[2*(2*n + 1 + z*10 + yBelow*10*zSize+x*10*ySize*zSize)] = 1;
        vortField[2*(2*n + 1 + z*10 + y*10*zSize+x*10*ySize*zSize)] = 1;
      }
    }
  }  

}'''
