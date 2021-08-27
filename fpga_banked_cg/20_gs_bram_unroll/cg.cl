#define M 8
#define M2 8
#define LX1 8
#define LY1 8
#define LZ1 8
#define NBANKS 4
#define MAX_DEG 16
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
    double __attribute__((doublepump)) shw[1024*512];
    double  __attribute__((doublepump)) shw2[1024*512];
    #pragma loop_coalesce
    #pragma ii 1
    for(int k=0; k <(N>>5); k+=1){
        #pragma unroll
        for(unsigned i = 0; i < NBANKS; i++){
            #pragma unroll
            for(unsigned j = 0; j < M; j++){
                shw[j+i*8+k*32] = wn[i][j+k*8];
                shw2[j+i*8+k*32] = wn[i][j+k*8];
            }
        }
     }
 
   
    //gather-scatter
    int k = 0;
    #pragma ivdep
    #pragma unroll 2
    for(int i = 0; i < nb; i++){
        int idx[MAX_DEG];
        int blk_len = b[i];
        double tmp = shw[idx[0]];
        #pragma unroll
        for(int j = 0; j < MAX_DEG; j++){
            idx[j] = gd[k+j] - 1;
        }
        #pragma ii 1
        for(int j = 0; j < blk_len; j++){
            tmp += shw[idx[j]];
        }
        #pragma ii 1
        for(int j = 0; j < blk_len; j++){
            shw2[idx[j]] = tmp;
        }
        k = k + blk_len;
    }
    #pragma ivdep
    #pragma ii 1
    #pragma unroll 2
    for(int i = (o-1); i < m; i+=2){
        int idx1 = gd[i] - 1;
        int idx2 = gd[i+1] - 1;
        double tmp =shw[idx1] + shw[idx2];
        shw2[idx1] = tmp;
        shw2[idx2] = tmp;
    }
    #pragma loop_coalesce
    #pragma ii 1
    for(int k=0; k <(N>>5); k+=1){
        #pragma unroll
        for(unsigned i = 0; i < NBANKS; i++){
            #pragma unroll
            for(unsigned j = 0; j < M; j++){
                wn[i][j+k*8]=shw2[j+i*8+k*32];
            }
        }
     }
}
