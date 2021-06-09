#define M 8
#define LX1 8
#define LY1 8
#define LZ1 8

__kernel void pre_ax(__global double * restrict r,
                        __global double * restrict w,
                        __global double * restrict p,
                        __global double * restrict g1,
                        __global double * restrict g2,
                        __global double * restrict g3,
                        __global double * restrict g4,
                        __global double * restrict g5,
                        __global double * restrict g6,
                        __global double * restrict dx,
                        __global double * restrict dxt,
                        __global double * restrict beta, 
                        int N){
    
    for(unsigned ele = 0; ele < N; ele += LX1*LY1*LZ1){
        double shur[LX1*LY1*LZ1];
        double shus[LX1*LY1*LZ1];
        double shut[LX1*LY1*LZ1];
        double shw[LX1*LY1*LZ1];
        double shg1[LX1*LY1*LZ1];
        double shg2[LX1*LY1*LZ1];
        double shg3[LX1*LY1*LZ1];
        double shg4[LX1*LY1*LZ1];
        double shg5[LX1*LY1*LZ1];
        double shg6[LX1*LY1*LZ1];
        double shu[LX1*LY1*LZ1];
        double shdx[LX1*LY1];
        double shdxt[LX1*LY1];
        #pragma unroll 32
        for(unsigned ij=0; ij<LX1*LY1; ++ij){
            shdx[ij] = dx[ij];
            shdxt[ij] = dxt[ij];
        }

        #pragma unroll 32
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            double temp = r[ijk + ele] + beta[0] * p[ijk + ele];
            shu[ijk] =  temp;
            p[ijk + ele] = temp;
            shg1[ijk] = g1[ijk + ele];
            shg2[ijk] = g2[ijk + ele];
            shg3[ijk] = g3[ijk + ele];
            shg4[ijk] = g4[ijk + ele];
            shg5[ijk] = g5[ijk + ele];
            shg6[ijk] = g6[ijk + ele];
        }
        #pragma loop_coalesce
        #pragma ii 1
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll 4
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double G00 = shg1[ijk];
                    double G01 = shg4[ijk];
                    double G02 = shg5[ijk];
                    double G11 = shg2[ijk];
                    double G12 = shg6[ijk];
                    double G22 = shg3[ijk];
                    double rtmp = 0.0;
                    double stmp = 0.0;
                    double ttmp = 0.0;
                    #pragma unroll 
                    for (unsigned l = 0; l<LX1; l++){
                      rtmp += shdxt[l+i*LX1] * shu[l+j*LX1 +k*LX1*LY1];
                      stmp += shdxt[l+j*LX1] * shu[i+l*LX1 + k*LX1*LY1];
                      ttmp += shdxt[l+k*LX1] * shu[ij + l*LX1*LY1];
                    }
                    shur[ijk] = G00*rtmp
                             + G01*stmp
                             + G02*ttmp;
                    shus[ijk] = G01*rtmp
                             + G11*stmp
                             + G12*ttmp;
                    shut[ijk]  = G02*rtmp
                             + G12*stmp
                             + G22*ttmp;
                }
            }
        }
        #pragma loop_coalesce 
        #pragma ii 1
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll 4
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    double wijke = 0.0;
                    #pragma unroll 
                    for(unsigned l = 0; l<LX1; l++){
                        wijke += shdx[l + i*LX1] * shur[l+j*LX1+k*LX1*LY1];
                        wijke += shdx[l + j*LX1] * shus[i+l*LX1+k*LX1*LY1];
                        wijke += shdx[l + k*LX1] * shut[i+j*LX1+l*LX1*LY1];
                    }
                    shw[ijk] = wijke;
                }
            }
        }

        #pragma unroll 32 
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk)
            w[ijk + ele] = shw[ijk];
    }


}

__kernel void post_ax(__global double * restrict x, 
                         __global const double * restrict p,
                         __global double * restrict r,
                         __global double * restrict w,
                         __global const double * restrict mult,
                         __global double * restrict rtz1,
                         __global double * restrict rtz2,
                         __global double * restrict beta,
                         int N)
{  
    double res = 0.0;
    
    #pragma unroll 32 
    for( int i = 0; i < N; ++i){
        res +=  w[i]*p[i]*mult[i];
    }

    double pap = res;
    double alpha = rtz1[0]/pap;
    printf("post ax %f, %f, %d \n",pap,rtz1[0], N);
    res = 0.0;

    #pragma unroll 32 
    for( int i = 0; i < N; ++i){
    	x[i] = x[i] + alpha * p[i];
    	r[i] = r[i] - alpha * w[i];
        res +=  r[i]*r[i]*mult[i];
    }
    rtz2[0] = rtz1[0];
    rtz1[0] = res;
  
    beta[0] = rtz1[0]/rtz2[0];

  }


