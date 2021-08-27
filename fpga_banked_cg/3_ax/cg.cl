#define M 8
#define M2 8
#define LX1 8
#define LY1 8
#define LZ1 8
#define NBANKS 4
__kernel void cg(__global double * restrict x_1, 
                 __global double * restrict x_2,
                 __global double * restrict x_3,
                 __global double * restrict x_4,
                 __global double * restrict p_1,
                 __global double * restrict p_2,
                 __global double * restrict p_3,
                 __global double * restrict p_4,
                 __global double * restrict r_1,
                 __global double * restrict r_2,
                 __global double * restrict r_3,
                 __global double * restrict r_4,
                 __global double * restrict w_1,
                 __global double * restrict w_2,
                 __global double * restrict w_3,
                 __global double * restrict w_4,
                 __global const double * restrict mult_1,
                 __global const double * restrict mult_2,
                 __global const double * restrict mult_3,
                 __global const double * restrict mult_4,
                 __global const double * restrict g1_1,
                 __global const double * restrict g1_2,
                 __global const double * restrict g1_3,
                 __global const double * restrict g1_4,
                 __global const double * restrict g2_1,
                 __global const double * restrict g2_2,
                 __global const double * restrict g2_3,
                 __global const double * restrict g2_4,
                 __global const double * restrict g3_1,
                 __global const double * restrict g3_2,
                 __global const double * restrict g3_3,
                 __global const double * restrict g3_4,
                 __global const double * restrict g4_1,
                 __global const double * restrict g4_2,
                 __global const double * restrict g4_3,
                 __global const double * restrict g4_4,
                 __global const double * restrict g5_1,
                 __global const double * restrict g5_2,
                 __global const double * restrict g5_3,
                 __global const double * restrict g5_4,
                 __global const double * restrict g6_1,
                 __global const double * restrict g6_2,
                 __global const double * restrict g6_3,
                 __global const double * restrict g6_4,
                 __global const double * restrict dx,
                 __global const double * restrict dxt,
                 __global int * restrict mask,
                 __global double * restrict rtz1,
                 __global double * restrict rtz2,
                 __global double * restrict beta,
                 const int N,
                 __global const int * restrict b,
                 __global const int * restrict gd,
                 __global const int * restrict dg,
                 __global double * restrict v,
                 const int m,
                 const int o,
                 const int nb)
{   
     __global double * restrict wn[NBANKS];
    wn[0] = w_1;
    wn[1] = w_2;
    wn[2] = w_3;
    wn[3] = w_4;
    __global double * restrict pn[NBANKS];
    pn[0] = p_1;
    pn[1] = p_2;
    pn[2] = p_3;
    pn[3] = p_4;
    __global double * restrict rn[NBANKS];
    rn[0] = r_1;
    rn[1] = r_2;
    rn[2] = r_3;
    rn[3] = r_4;
    __global double * restrict xn[NBANKS];
    xn[0] = x_1;
    xn[1] = x_2;
    xn[2] = x_3;
    xn[3] = x_4;
    __global const double * restrict g1[NBANKS];
    g1[0] = g1_1;
    g1[1] = g1_2;
    g1[2] = g1_3;
    g1[3] = g1_4;
    __global const double * restrict g2[NBANKS];
    g2[0] = g2_1;
    g2[1] = g2_2;
    g2[2] = g2_3;
    g2[3] = g2_4;
    __global const double * restrict g3[NBANKS];
    g3[0] = g3_1;
    g3[1] = g3_2;
    g3[2] = g3_3;
    g3[3] = g3_4;
    __global const double * restrict g4[NBANKS];
    g4[0] = g4_1;
    g4[1] = g4_2;
    g4[2] = g4_3;
    g4[3] = g4_4;
    __global const double * restrict g5[NBANKS];
    g5[0] = g5_1;
    g5[1] = g5_2;
    g5[2] = g5_3;
    g5[3] = g5_4;
    __global const double * restrict g6[NBANKS];
    g6[0] = g6_1;
    g6[1] = g6_2;
    g6[2] = g6_3;
    g6[3] = g6_4;
 
    beta[0] = 5.0;
    #pragma ivdep
    for(unsigned ele = 0; ele < N/NBANKS; ele += LX1*LY1*LZ1/NBANKS){
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

        #pragma loop_coalesce
        #pragma ii 1
        #pragma ivdep
        for(int ijk=0; ijk<LX1*LY1*LZ1; ijk+=32){
            int b_ijk = ijk >> 2; 
            #pragma unroll
            for(unsigned i = 0; i < NBANKS; i++){
                #pragma unroll
                for(unsigned j = 0; j < M; j++){
                    double temp = rn[i][j + b_ijk + ele] + beta[0] * pn[i][j + b_ijk + ele];
                    shu[ijk+i*M+j] =  temp;
                    shg1[ijk+i*M+j] = g1[i][j + b_ijk + ele];
                    shg2[ijk+i*M+j] = g2[i][j + b_ijk + ele];
                    shg3[ijk+i*M+j] = g3[i][j + b_ijk + ele];
                    shg4[ijk+i*M+j] = g4[i][j + b_ijk + ele];
                    shg5[ijk+i*M+j] = g5[i][j + b_ijk + ele];
                    shg6[ijk+i*M+j] = g6[i][j + b_ijk + ele];
                }
            }
        }
        
        #pragma loop_coalesce
        #pragma ii 1
        for (unsigned k=0; k<LZ1; ++k){
            for(unsigned j = 0; j < LY1; j++){
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    double G00 = shg1[ijk];
                    double G01 = shg2[ijk];
                    double G02 = shg3[ijk];
                    double G11 = shg4[ijk];
                    double G12 = shg5[ijk];
                    double G22 = shg6[ijk];
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

        #pragma loop_coalesce
        #pragma ii 1
        #pragma ivdep
        for(int ijk=0; ijk<LX1*LY1*LZ1; ijk+=32){
            int b_ijk = ijk >> 2; 
            #pragma unroll
            for(unsigned i = 0; i < NBANKS; i++){
                #pragma unroll
                for(unsigned j = 0; j < M; j++){
                    wn[i][j + b_ijk + ele] = shw[ijk+i*M+j];
                    xn[i][j + b_ijk + ele] = shu[ijk+i*M+j];
                }
            }
        }
    }

}
