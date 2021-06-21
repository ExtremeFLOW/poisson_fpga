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
    double rtr_copies[M];
    double res = 0.0;
    int ma = mask[0];
    #pragma ii 1
    for(int i = 1; i < (ma + 1); i++){
       int k = mask[i];
       w[k-1] = 0.0; 
    }
    #pragma unroll
    for(int i = 0; i < M; ++i) 
        rtr_copies[i] = 0;
    #pragma ii 1
    for( int i = 0; i < N; i+=M){
        double cur = rtr_copies[M-1];
        #pragma unroll
        for( int k = 0; k < M; k++) 
           cur += w[i+k]*p[i+k]*mult[i+k];
        #pragma unroll
        for(unsigned j = M-1; j>0; j--){
            rtr_copies[j] = rtr_copies[j-1];
        }
        rtr_copies[0] = cur; 
    }
 
    #pragma unroll
    for(unsigned i = 0; i < M; i++){
        res += rtr_copies[i];
    }


    double pap = res;
    double alpha = rtz1[0]/pap;
    res = 0.0;

    //for( int i = 0; i < N; ++i){
    //	x[i] = x[i] + alpha * p[i];
    //	r[i] = r[i] - alpha * w[i];
    //    res +=  r[i]*r[i]*mult[i];
    //}
    double rtr_copies2[M];
    #pragma unroll
    for(int i = 0; i < M; ++i) 
        rtr_copies2[i] = 0;
    #pragma ii 1
    for( int i = 0; i < N; i+=M){
        double cur = rtr_copies2[M-1];
        #pragma unroll
        for( int k = 0; k < M; k++){ 
    	    x[i+k] = x[i+k] + alpha * p[i+k];
    	    double r_new= r[i+k] - alpha * w[i+k];
            cur +=  r_new*r_new*mult[i+k];
            r[i+k] = r_new;
        }
        #pragma unroll
        for(unsigned j = M-1; j>0; j--){
            rtr_copies2[j] = rtr_copies2[j-1];
        }
        rtr_copies2[0] = cur; 
    }
 
    #pragma unroll
    for(unsigned i = 0; i < M; i++){
        res += rtr_copies2[i];
    }

    rtz2[0] = rtz1[0];
    rtz1[0] = res;
  
    beta[0] = rtz1[0]/rtz2[0];
}


