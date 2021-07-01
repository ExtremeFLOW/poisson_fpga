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
    
    //gather-scatter
    int k = 0;
    #pragma ivdep
    for(int i = 0; i < nb; i++){
        double rtr_copies3[M2];
        int idx[8];
        int bank[8];
        int blk_len = b[i];
        int k1 = gd[k]-1;
        int j1 = k1 % 32;
        int id1 = k1 >> 5;
        bank[0] = j1 >> 3;
        int off1 = j1 % 8;
        idx[0] =  off1 + id1*8;
        
        double tmp = wn[bank[0]][idx[0]];
        #pragma unroll
        for(int i = 0; i < M2; ++i) 
            rtr_copies3[i] = 0.0;
        #pragma ivdep
        #pragma unroll
        for(int j = 1; j < blk_len; j++){
            int k2 = gd[k+j]-1;
            int j2 = k2 % 32;
            int id2 = k2 >> 5;
            bank[j] = j2>> 3;
            int off2 = j2 % 8;
            idx[j] =  off2 + id2*8;
        }

        #pragma ivdep
        #pragma ii 1
        for(int j = 1; j < blk_len; j++){
            double cur = rtr_copies3[M2-1] + wn[bank[j]][idx[j]];
            #pragma unroll
            for(unsigned j = M2-1; j>0; j--){
                rtr_copies3[j] = rtr_copies3[j-1];
            }
            rtr_copies3[0] = cur;
        }
        #pragma unroll
        for(unsigned i = 0; i < M2; i++)
            tmp += rtr_copies3[i];
        #pragma ivdep
        #pragma unroll
        for(int j = 0; j < blk_len; j++){
            wn[bank[j]][idx[j]] = tmp;
        }
        k = k + blk_len;
    }
    #pragma ivdep
    #pragma unroll 8
    #pragma ii 1
    for(int i = (o-1); i < m; i+=2){
        int k1 = gd[i]-1;
        int j1 = k1 % 32;
        int id1 = k1 >> 5;
        int ji1 = j1 >> 3;
        int off1 = j1 % 8;
        int idx1 =  off1 + id1*8;
        
        int k2 = gd[i+1]-1;
        int j2 = k2 % 32;
        int id2 = k2 >> 5;
        int ji2 = j2 >> 3;
        int off2 = j2 % 8;
        int idx2 =  off2 + id2*8;
        
        double tmp =wn[ji1][idx1] + wn[ji2][idx2];
        wn[ji1][idx1] = tmp;
        wn[ji2][idx2] = tmp;
    }
}
