#define M 8
#define M2 8
#define LX1 8
#define LY1 8
#define LZ1 8
__kernel void cg(__global float * restrict x, 
                 __global float * restrict p,
                 __global float * restrict r,
                 __global float * restrict w,
                 __global const float * restrict mult,
                 __global const float * restrict g1,
                 __global const float * restrict g2,
                 __global const float * restrict g3,
                 __global const float * restrict g4,
                 __global const float * restrict g5,
                 __global const float * restrict g6,
                 __global const float * restrict dx,
                 __global const float * restrict dxt,
                 __global const int * restrict mask,
                 __global float * restrict rtz1,
                 __global float * restrict rtz2,
                 __global float * restrict beta,
                 const int N,
                 __global const int * restrict b,
                 __global const int * restrict gd,
                 __global const int * restrict dg,
                 __global float * restrict v,
                 const int m,
                 const int o,
                 const int nb)
{
   beta[0] = 5.0;   
   #pragma ivdep
   for(unsigned ele = 0; ele < N; ele += LX1*LY1*LZ1){
        float shur[LX1*LY1*LZ1];
        float shus[LX1*LY1*LZ1];
        float shut[LX1*LY1*LZ1];
        float shw[LX1*LY1*LZ1];
        float shg1[LX1*LY1*LZ1];
        float shg2[LX1*LY1*LZ1];
        float shg3[LX1*LY1*LZ1];
        float shg4[LX1*LY1*LZ1];
        float shg5[LX1*LY1*LZ1];
        float shg6[LX1*LY1*LZ1];
        float shu[LX1*LY1*LZ1];
        float shdx[LX1*LY1];
        float shdxt[LX1*LY1];
        #pragma unroll 64
        for(unsigned ij=0; ij<LX1*LY1; ++ij){
            shdx[ij] = dx[ij];
            shdxt[ij] = dxt[ij];
        }

        #pragma unroll 64
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            float temp = p[ijk + ele];
            shu[ijk] =  temp;
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
            #pragma unroll 2
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    float G00 = shg1[ijk];
                    float G01 = shg2[ijk];
                    float G02 = shg3[ijk];
                    float G11 = shg4[ijk];
                    float G12 = shg5[ijk];
                    float G22 = shg6[ijk];
                    float rtmp = 0.0;
                    float stmp = 0.0;
                    float ttmp = 0.0;
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
            #pragma unroll 2
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    float wijke = 0.0;
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

        #pragma unroll 64 
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk)
            w[ijk + ele] = shw[ijk];
    }
}



