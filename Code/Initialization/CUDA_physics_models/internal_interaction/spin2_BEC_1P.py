def get_CUDA(dimensions, vectorSize, G0 = 1., G1 = 1., G2 = 1., MU = 1., scaling = 30., **kwargs):
	return r'''
		__global__ void internal(dcmplx *QField, dcmplx *QField2, int* lattice, int* gpu_params)
	{   
	    int xSize = lattice[0]*gpu_params[2];
	    int ySize = lattice[2];
	    int zSize = lattice[4];
	    int vectorSize = lattice[12];
	    dcmplx g0 = ''' + str(G0) + r''';
	    dcmplx g1 = ''' + str(G1) + r''';
	    dcmplx g2 = ''' + str(G2) + r''';
	    dcmplx mu2 =''' + str(MU) + r''';
	    dcmplx scaling = ''' + str(scaling) + r''';
	    int x = blockIdx.x * blockDim.x + threadIdx.x;
	    int y = blockIdx.y * blockDim.y + threadIdx.y;
	    int z = blockIdx.z * blockDim.z + threadIdx.z;
	    int n, j, nn;
	    dcmplx k(1.,0.);
	    dcmplx A(scaling.real()/256., 0.); 
	    dcmplx mu = Mul(mu2, Mul(k, k));
	    dcmplx tau = Mul(A, A);
	    dcmplx Nterm = 0.;
	    dcmplx rho = 0.;
	    dcmplx B00L = 0.;
	    dcmplx B00R = 0.;
	    dcmplx NLPot = 0.;
	    dcmplx new_g2(g2.real()/5. , 0.);              
	    double TA = __dmul_rn((Mul(g1, tau)).real(), 1./''' + str(int(4*dimensions)) + r'''.); // trotterization with N = 4*dims
	    double TA2 = __dmul_rn((Mul(new_g2, tau)).real(), 1.);
	    dcmplx epsilon(1./sqrt(2.), 0.);
	    double delta = 0.0000000000000001;
	    dcmplx root2(sqrt(2.), 0);
	    dcmplx root3(sqrt(3.), 0);
	    dcmplx root6(sqrt(6.), 0.);
	    dcmplx i(0.,1./''' + str(int(4*dimensions)) + r'''.); // trotterization with N = 4*dims
	    dcmplx one(1., 0.); 
	    dcmplx two(2., 0.);
	    dcmplx three(3., 0.); 
	    dcmplx four(4., 0.);
	    dcmplx six(6., 0.);
	    dcmplx nine(9., 0.);
	    dcmplx Fp = 0.;
	    dcmplx Fm = 0.;
	    dcmplx Fz = 0.;
	    double F = 0.;
	    double FHalf = 0.;
	    double FsquaredHalf = 0.;
	    double Fsquared = 0.;
	    dcmplx csHalf = 0.;
	    dcmplx isnHalf = 0.;
	    dcmplx cs = 0.;
	    dcmplx isn = 0.;
	    dcmplx psi[''' + str(int(vectorSize)) + r'''];
	    dcmplx phi[''' + str(int(vectorSize/2)) + r'''];  

	    for (n=0; n<''' + str(int(vectorSize)) + r'''; n=n+1){
	        psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	        }
	    for(nn = 0; nn < ''' + str(int(vectorSize/2)) + r'''; nn++){
	        phi[nn] = psi[2*nn] + psi[2*nn+1];
	        }
	    rho = Mul(phi[0], conj(phi[0])) + Mul(phi[1], conj(phi[1])) + Mul(phi[2], conj(phi[2])) + 
	              Mul(phi[3], conj(phi[3])) + Mul(phi[4], conj(phi[4]));
	    dcmplx Fp_first_part = Mul(conj(phi[0]), phi[1]) + Mul(conj(phi[3]), phi[4]);
	    dcmplx Fp_second_part = Mul(conj(phi[1]), phi[2]) + Mul(conj(phi[2]), phi[3]);
	    dcmplx Fp_combined = Mul(two, Fp_first_part) + Mul(root6, Fp_second_part); 
	    Fp = Mul(epsilon, Fp_combined);                                                                
	    Fm = conj(Fp);
	    Fz = Mul(two, Mul(conj(phi[0]), phi[0]) - Mul(conj(phi[4]), phi[4])) + Mul(conj(phi[1]), phi[1]) - Mul(conj(phi[3]), phi[3]);
	    FsquaredHalf= (Mul3(two, Fp, Fm) + Mul3(four, Fz, Fz)).real();
	    Fsquared=(Mul3(six, Fp, Fm) + Mul3(nine, Fz, Fz)).real();
	    FHalf = sqrt(FsquaredHalf);
	    F = sqrt(Fsquared);
	    NLPot = exp(Mul3(-i, (Mul(rho,g0) + mu), tau));
	    if(abs(rho.real())>delta){
	      Nterm = (exp(Mul3(-i, TA2, rho.real()))-(dcmplx)1.);
	      Nterm = dcmplx(Nterm.real()/rho.real(), Nterm.imag()/rho.real());   // line 358 
	    }
	    else {
	      Nterm = -Mul(i, TA2);
	    }

	    if(abs(Fsquared)>delta){
	      cs=Mul(dcmplx(1./Fsquared, 0.), dcmplx(cos(__dmul_rn(TA,F))-1., 0.));
	      isn=Mul(dcmplx(0., 1./F), dcmplx(sin(__dmul_rn(TA,F)), 0.));
	    }
	    else{
	      cs=dcmplx(0., 0.);
	      isn=i;
	    }
	    if(abs(FsquaredHalf)>delta){
	      csHalf=Mul(dcmplx(1./FsquaredHalf, 0.) , dcmplx(cosC(__dmul_rn(TA,FHalf))-1. , 0.));
	      isnHalf=Mul(dcmplx(0., 1./FHalf), dcmplx(sinC(__dmul_rn(TA, FHalf)), 0.));
	    }
	    else{
	      csHalf= dcmplx(0., 0.);
	      isnHalf=i;
	    }

	    /*Collide Potential*/

	    for (j=0; j<''' + str(int(vectorSize)) + r'''; j++){
	        QField[j+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=Mul(NLPot,QField[j+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]);
	    }

	    /*Update Fields*/
	    for (n=0; n<''' + str(int(vectorSize)) + r'''; n=n+1){
	        psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	        }
	    for(nn = 0; nn < ''' + str(int(vectorSize/2)) + r'''; nn++){
	        phi[nn] = psi[2*nn] + psi[2*nn+1];
	        } 
	    
	    /*Collide Spin*/

	    /*Lefthanded*/
	    QField[0+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	            Mul(psi[0], one - Mul3(two,Fz,isnHalf)+Mul(csHalf,FsquaredHalf))
	            +Mul3(psi[2], -Mul3(root2, isnHalf, Fm), Mul(cs, Mul3(nine,Fz,Fz)+Mul3(three,Fm,Fp)) + one-Mul3(three,Fz,isn))
	            +Mul3(psi[4], -Mul3(root2, isnHalf, Fm), Mul3(root3, Fm, Mul3(three,cs,Fz)-isn))
	            +Mul3(psi[6], -Mul3(root2, isnHalf, Fm), Mul4(three, cs, Fm, Fm))
	            );
	    QField[2+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	             Mul4(psi[0], -root2, isnHalf, Fp)
	            +Mul3(psi[2],  (one + Mul3(two, Fz, isnHalf)+ Mul(csHalf,FsquaredHalf)) , Mul(cs, Mul3(nine,Fz,Fz) + Mul3(three, Fm, Fp)) + one-Mul3(three,Fz,isn))
	            +Mul3(psi[4], Mul3(root3, Fm, Mul3(three, cs, Fz) - isn), one+Mul3(two,Fz,isnHalf)+Mul(csHalf,FsquaredHalf))
	            +Mul3(psi[6], one+Mul3(two, Fz, isnHalf)+Mul(csHalf,FsquaredHalf), Mul4(three, cs, Fm, Fm))
	            );
	    QField[4+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	             Mul3(psi[2], Mul(root3, Fp), Mul3(three,cs,Fz)-isn)
	            +Mul(psi[4], one+Mul4(six,cs,Fm,Fp))
	            +Mul3(psi[6], Mul(root3, Fm), Mul3(-three,cs,Fz)-isn)
	            );
	    QField[6+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	             Mul3(psi[2], one-Mul3(two, Fz, isnHalf)+Mul(csHalf,FsquaredHalf), Mul4(three,cs,Fp,Fp))
	            +Mul4(psi[4], one-Mul3(two, Fz, isnHalf)+Mul(csHalf,FsquaredHalf), Mul(root3, Fp) , Mul3(-three, cs, Fz) -isn)
	            +Mul3(psi[6], one-Mul3(two, Fz, isnHalf)+Mul(csHalf,FsquaredHalf), Mul(cs, Mul3(nine,Fz,Fz)+Mul3(three,Fm,Fp))+one+Mul3(three,Fz, isn))
	            +Mul4(psi[8], -root2, isnHalf, Fm)
	            );
	    QField[8+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	             Mul3(psi[2], Mul3(-root2, isnHalf, Fp), Mul4(three,cs,Fp,Fp))
	            +Mul4(psi[4], Mul3(-root2, isnHalf, Fp),  Mul(root3, Fp) , Mul3(-three, cs, Fz) -isn)
	            +Mul3(psi[6], Mul3(-root2, isnHalf, Fp), Mul(cs, Mul3(nine,Fz,Fz)+Mul3(three,Fm,Fp))+one+Mul3(three,Fz, isn))
	            +Mul(psi[8], one+Mul3(two, Fz, isnHalf) + Mul(csHalf, FsquaredHalf))
	            ); 


	    /*Righthanded*/
	    
	    QField[1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	             Mul(psi[1], one - Mul3(2.,Fz,isnHalf)+Mul(csHalf,FsquaredHalf))
	            +Mul3(psi[3], -Mul3(root2, isnHalf, Fm), Mul(cs, Mul3(nine,Fz,Fz)+Mul3(three,Fm,Fp)) +one-Mul3(three,Fz,isn))
	            +Mul3(psi[5], -Mul3(root2, isnHalf, Fm), Mul3(root3, Fm, Mul3(three,cs,Fz)-isn))
	            +Mul3(psi[7], -Mul3(root2, isnHalf, Fm), Mul4(three, cs, Fm, Fm))
	            );
	    QField[3+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	             Mul4(psi[1], -root2, isnHalf, Fp)
	            +Mul3(psi[3],  (one + Mul3(two, Fz, isnHalf)+ Mul(csHalf,FsquaredHalf)) , Mul(cs, Mul3(nine,Fz,Fz) + Mul3(three, Fm, Fp)) + one-Mul3(three,Fz,isn))
	            +Mul3(psi[5], Mul3(root3, Fm, Mul3(three, cs, Fz) - isn), one+Mul3(two,Fz,isnHalf)+Mul(csHalf,FsquaredHalf))
	            +Mul3(psi[7], one+Mul3(two, Fz, isnHalf)+Mul(csHalf,FsquaredHalf), Mul4(three, cs, Fm, Fm))
	            );
	    QField[5+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	             Mul3(psi[3], Mul(root3, Fp), Mul3(three,cs,Fz)-isn)
	            +Mul(psi[5], one+Mul4(6.,cs,Fm,Fp))
	            +Mul3(psi[7], Mul(root3, Fm), Mul3(-three,cs,Fz)-isn)
	            );
	    QField[7+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	             Mul3(psi[3], one-Mul3(two, Fz, isnHalf)+Mul(csHalf,FsquaredHalf), Mul4(three,cs,Fp,Fp))
	            +Mul4(psi[5], one-Mul3(two, Fz, isnHalf)+Mul(csHalf,FsquaredHalf), Mul(root3, Fp) , Mul3(-three, cs, Fz) -isn)
	            +Mul3(psi[7], one-Mul3(two, Fz, isnHalf)+Mul(csHalf,FsquaredHalf), Mul(cs, Mul3(nine,Fz,Fz)+Mul3(three,Fm,Fp))+one+Mul3(three,Fz, isn))
	            +Mul4(psi[9], -root2, isnHalf, Fm)
	            );
	    QField[9+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=(
	             Mul3(psi[3], Mul3(-root2, isnHalf, Fp), Mul4(three,cs,Fp,Fp))
	            +Mul4(psi[5], Mul3(-root2, isnHalf, Fp),  Mul(root3, Fp) , Mul3(-three, cs, Fz) -isn)
	            +Mul3(psi[7], Mul3(-root2, isnHalf, Fp), Mul(cs, Mul3(nine,Fz,Fz)+ Mul3(three,Fm,Fp))+one+Mul3(three,Fz, isn))
	            +Mul(psi[9], one+Mul3(two, Fz, isnHalf) + Mul(csHalf, FsquaredHalf))
	            );

	    /*Update Fields*/
	    for (n=0; n<''' + str(int(vectorSize)) + r'''; n=n+1){
	        psi[n] = QField[n+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize];
	        }
	    for(nn = 0; nn < ''' + str(int(vectorSize/2)) + r'''; nn++){
	        phi[nn] = psi[2*nn] + psi[2*nn+1];
	        }

	    B00L = Mul(phi[0], psi[8]) - Mul(phi[1], psi[6]) + Mul(phi[2], psi[4]) - Mul(phi[3], psi[2]) + Mul(phi[4], psi[0]);
	    B00R = Mul(phi[0], psi[9]) - Mul(phi[1], psi[7]) + Mul(phi[2], psi[5]) - Mul(phi[3], psi[3]) + Mul(phi[4], psi[1]); 
	    
	    /*CollideA*/ 
	    /*Lefthanded*/
	    QField[0+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[0]+Mul3(Nterm, conj(phi[4]), B00L);
	    QField[2+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[2]-Mul3(Nterm, conj(phi[3]), B00L);
	    QField[4+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[4]+Mul3(Nterm, conj(phi[2]), B00L);
	    QField[6+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[6]-Mul3(Nterm, conj(phi[1]), B00L);
	    QField[8+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[8]+Mul3(Nterm, conj(phi[0]), B00L);

	    /*Righthanded*/
	    QField[1+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[1]+Mul3(Nterm, conj(phi[4]), B00R);
	    QField[3+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[3]-Mul3(Nterm, conj(phi[3]), B00R);
	    QField[5+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[5]+Mul3(Nterm, conj(phi[2]), B00R);
	    QField[7+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[7]-Mul3(Nterm, conj(phi[1]), B00R);
	    QField[9+z*vectorSize+y*vectorSize*zSize+x*zSize*ySize*vectorSize]=psi[9]+Mul3(Nterm, conj(phi[0]), B00R);
	}
	'''