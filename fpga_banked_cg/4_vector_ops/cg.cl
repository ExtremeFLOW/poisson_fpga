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
    double rtr_copies[M];
    double res = 0.0;
    int ma = mask[0];
    __global double * restrict wn[NBANKS];
    wn[0] = w_1;
    wn[1] = w_2;
    wn[2] = w_3;
    wn[3] = w_4;
    #pragma ii 1
    #pragma ivdep
    for(int i = 1; i < (ma + 1); i+=M){
       #pragma unroll
       for(int l = 0; l < M; l++){
          int k = mask[i+l]-1;
          int j = k % 32;
          int id = k >> 5;
          int ji = j >> 3;
          int off = j % 8;
          int idx =  off + id*8;
          wn[ji][idx] = 0.0;
       }
    }
    #pragma unroll
    for(int i = 0; i < M; ++i) 
        rtr_copies[i] = 0;
    #pragma ii 1
    for( int i = 0; i < N/NBANKS; i+=M){
        double cur = rtr_copies[M-1];
        #pragma unroll
        for( int k = 0; k < M; k++){ 
           cur += w_1[i+k]*p_1[i+k]*mult_1[i+k];
           cur += w_2[i+k]*p_2[i+k]*mult_2[i+k];
           cur += w_3[i+k]*p_3[i+k]*mult_3[i+k];
           cur += w_4[i+k]*p_4[i+k]*mult_4[i+k];
        }
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
    for( int i = 0; i < N/NBANKS; i+=M){
        double cur = rtr_copies2[M-1];
        #pragma unroll
        for( int k = 0; k < M; k++){ 
    	    x_1[i+k] = x_1[i+k] + alpha * p_1[i+k];
    	    x_2[i+k] = x_2[i+k] + alpha * p_2[i+k];
    	    x_3[i+k] = x_3[i+k] + alpha * p_3[i+k];
    	    x_4[i+k] = x_4[i+k] + alpha * p_4[i+k];
    	    double r_new_1= r_1[i+k] - alpha * w_1[i+k];
    	    double r_new_2= r_2[i+k] - alpha * w_2[i+k];
    	    double r_new_3= r_3[i+k] - alpha * w_3[i+k];
    	    double r_new_4= r_4[i+k] - alpha * w_4[i+k];
            cur +=  r_new_1*r_new_1*mult_1[i+k];
            cur +=  r_new_2*r_new_2*mult_2[i+k];
            cur +=  r_new_3*r_new_3*mult_3[i+k];
            cur +=  r_new_4*r_new_4*mult_4[i+k];
            r_1[i+k] = r_new_1;
            r_2[i+k] = r_new_2;
            r_3[i+k] = r_new_3;
            r_4[i+k] = r_new_4;
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
