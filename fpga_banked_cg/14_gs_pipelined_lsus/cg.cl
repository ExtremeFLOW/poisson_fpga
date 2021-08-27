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
                 __global double * restrict w_3, __global double * restrict w_4, __global const double * restrict mult_1,
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
    
    
    #pragma ivdep
    #pragma unroll 8
    for(int i = (o-1); i < m; i+=2){
        int k1 = __burst_coalesced_load(&gd[i])-1;
        int k2 = __burst_coalesced_load(&gd[i+1])-1;
 
        int j1 = k1 % 32;
        int id1 = k1 >> 5;
        int ji1 = j1 >> 3;
        int off1 = j1 % 8;
        int idx1 =  off1 + id1*8;
        
        int j2 = k2 % 32;
        int id2 = k2 >> 5;
        int ji2 = j2 >> 3;
        int off2 = j2 % 8;
        int idx2 =  off2 + id2*8;
        
        double tmp =__pipelined_load(&wn[ji1][idx1]) + __pipelined_load(&wn[ji2][idx2]);
        __pipelined_store(&wn[ji1][idx1],tmp);
        __pipelined_store(&wn[ji2][idx2],tmp);
    }
}
