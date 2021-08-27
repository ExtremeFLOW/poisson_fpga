#define M 8
#define E 128
#define M2 8
#define LX1 8
#define LY1 8
#define LZ1 8
#define MAX_DEG 8
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
    double shw1[LX1*LY1*LZ1*E];
    double shw2[LX1*LY1*LZ1*E];
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
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            shw1[ijk + ele] = shw[ijk];
            shw2[ijk + ele] = shw[ijk];
        }
    }
  
   
    //gather-scatter
    int k = 0;
    #pragma ivdep
    for(int i = 0; i < nb; i++){
        int idx[MAX_DEG];
        int blk_len = b[i];
        double tmp = shw1[idx[0]];
        #pragma unroll
        for(int j = 0; j < MAX_DEG; j++){
            idx[j] = gd[k+j] - 1;
        }
        #pragma ii 1
        for(int j = 0; j < blk_len; j++){
            tmp += shw1[idx[j]];
        }
        #pragma ii 1
        for(int j = 0; j < blk_len; j++){
            shw2[idx[j]] = tmp;
        }
        k = k + blk_len;
    }
    #pragma ivdep
    #pragma ii 1
    for(int i = (o-1); i < m; i+=2){
        int idx1 = gd[i] - 1;
        int idx2 = gd[i+1] - 1;
        double tmp =shw1[idx1] + shw1[idx2];
        shw2[idx1] = tmp;
        shw2[idx2] = tmp;
    }
    #pragma loop_coalesce
    for(int ele=0; ele <(N<<9); ele+=LX1*LY1*LZ1){
        #pragma unroll
        for(unsigned ijk = 0; ijk < LX1*LY1*LZ1; ijk++){
            w[ijk+ele]=shw2[ijk+ele];
        }
    }
}


