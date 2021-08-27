#define M 8
#define M2 8
#define LX1 8
#define LY1 8
#define LZ1 8
__kernel void cg(__global double * restrict x, 
                 __global double * restrict p,
                 __global double * restrict r,
                 __global double * restrict w,
                 __global const double * restrict mult,
                 __global double * restrict g1,
                 __global double * restrict g2,
                 __global double * restrict g3,
                 __global double * restrict g4,
                 __global double * restrict g5,
                 __global double * restrict g6,
                 __global double * restrict dx,
                 __global double * restrict dxt,
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
    //gather-scatter
    int k = 0;
    #pragma ivdep
    for(int i = 0; i < nb; i++){
        double rtr_copies3[M2];
        int idx[8];
        int blk_len = b[i];
        idx[0] = gd[k] - 1;
        double tmp = w[idx[0]];
        #pragma unroll
        for(int i = 0; i < M2; ++i) 
            rtr_copies3[i] = 0.0;
        for(int j = 1; j < blk_len; j++){
            idx[j] = gd[k+j] - 1;
            double cur = rtr_copies3[M2-1] + w[idx[j]];
            #pragma unroll
            for(unsigned j = M2-1; j>0; j--){
                rtr_copies3[j] = rtr_copies3[j-1];
            }
            rtr_copies3[0] = cur;
        }
        #pragma unroll
        for(unsigned i = 0; i < M2; i++)
            tmp += rtr_copies3[i];
        for(int j = 0; j < blk_len; j++){
            w[idx[j]] = tmp;
        }
        k = k + blk_len;
    }
    #pragma ivdep
    for(int i = (o-1); i < m; i+=2){
        int idx1 = gd[i] - 1;
        int idx2 = gd[i+1] - 1;
        double tmp =w[idx1] + w[idx2];
        w[idx1] = tmp;
        w[idx2] = tmp;
    }

}


