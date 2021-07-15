#define M 8
#define M2 8
#define M_gs 1
#define LX1 8
#define LY1 8
#define LZ1 8
#define NBANKS 4

typedef struct
{
	double add1[M_gs];
	double add2[M_gs];
	int bank1[M_gs];
	int bank2[M_gs];
	int id1[M_gs];
	int id2[M_gs];
} gs_data;
#pragma OPENCL EXTENSION cl_intel_channels : enable

channel gs_data gs_add_channel __attribute__((depth(128)));


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
        #pragma ivdep
        for(int ijk=0; ijk<LX1*LY1*LZ1; ijk+=32){
            int b_ijk = ijk >> 2;
            #pragma unroll
            for(unsigned i = 0; i < M; i++){
                int b_i = i % M;
                double temp = r_1[b_i + b_ijk + ele] + beta[0] * p_1[b_i + b_ijk + ele];
                shu[ijk+i] =  temp;
                p_1[b_i + b_ijk + ele] = temp;
                shg1[ijk+i] = g1_1[b_i + b_ijk + ele];
                shg2[ijk+i] = g2_1[b_i + b_ijk + ele];
                shg3[ijk+i] = g3_1[b_i + b_ijk + ele];
                shg4[ijk+i] = g4_1[b_i + b_ijk + ele];
                shg5[ijk+i] = g5_1[b_i + b_ijk + ele];
                shg6[ijk+i] = g6_1[b_i + b_ijk + ele];
            }
            #pragma unroll
            for(unsigned i = M; i < 2*M; i++){
                int b_i = i % M;
                double temp = r_2[b_i + b_ijk + ele] + beta[0] * p_2[b_i + b_ijk + ele];
                shu[ijk+i] =  temp;
                p_2[i -M +b_ijk + ele] = temp;
                shg1[ijk+i] = g1_2[b_i + b_ijk + ele];
                shg2[ijk+i] = g2_2[b_i + b_ijk + ele];
                shg3[ijk+i] = g3_2[b_i + b_ijk + ele];
                shg4[ijk+i] = g4_2[b_i + b_ijk + ele];
                shg5[ijk+i] = g5_2[b_i + b_ijk + ele];
                shg6[ijk+i] = g6_2[b_i + b_ijk + ele];
            }
            #pragma unroll
            for(unsigned i = 2*M; i < 3*M; i++){
                int b_i = i % M;
                double temp = r_3[b_i + b_ijk + ele] + beta[0] * p_3[b_i + b_ijk + ele];
                shu[ijk+i] =  temp;
                p_3[b_i + b_ijk + ele] = temp;
                shg1[ijk+i] = g1_3[b_i + b_ijk + ele];
                shg2[ijk+i] = g2_3[b_i + b_ijk + ele];
                shg3[ijk+i] = g3_3[b_i + b_ijk + ele];
                shg4[ijk+i] = g4_3[b_i + b_ijk + ele];
                shg5[ijk+i] = g5_3[b_i + b_ijk + ele];
                shg6[ijk+i] = g6_3[b_i + b_ijk + ele];
            }
            #pragma unroll
            for(unsigned i = 3*M; i < 4*M; i++){
                int b_i = i % M;
                double temp = r_4[b_i + b_ijk + ele] + beta[0] * p_4[b_i + b_ijk + ele];
                shu[ijk+i] =  temp;
                p_4[b_i + b_ijk + ele] = temp;
                shg1[ijk+i] = g1_4[b_i + b_ijk + ele];
                shg2[ijk+i] = g2_4[b_i + b_ijk + ele];
                shg3[ijk+i] = g3_4[b_i + b_ijk + ele];
                shg4[ijk+i] = g4_4[b_i + b_ijk + ele];
                shg5[ijk+i] = g5_4[b_i + b_ijk + ele];
                shg6[ijk+i] = g6_4[b_i + b_ijk + ele];
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
 
        for(int ijk=0; ijk<LX1*LY1*LZ1; ijk+=32){
            int b_ijk = ijk >> 2; 
            #pragma unroll
            for(unsigned i = 0; i < M; i++){
                int b_i = i % M;
                w_1[b_i+b_ijk + ele] = shw[ijk+i];
            }
            #pragma unroll
            for(unsigned i = 1*M; i < 2*M; i++){
                int b_i = i % M;
                w_2[b_i+b_ijk + ele] = shw[ijk+i];
            }
            #pragma unroll
            for(unsigned i = 2*M; i < 3*M; i++){
                int b_i = i % M;
                w_3[b_i+b_ijk + ele] = shw[ijk+i];
            }
            #pragma unroll
            for(unsigned i = 3*M; i < 4*M; i++){
                int b_i = i % M;
                w_4[b_i+b_ijk + ele] = shw[ijk+i];
            }
        }
    }
    

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
        for(int j = 1; j < blk_len; j++){
            int k2 = gd[k+j]-1;
            int j2 = k2 % 32;
            int id2 = k2 >> 5;
            bank[j] = j2>> 3;
            int off2 = j2 % 8;
            idx[j] =  off2 + id2*8;
        
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
        for(int j = 0; j < blk_len; j++){
            wn[bank[j]][idx[j]] = tmp;
        }
        k = k + blk_len;
    }
     
    #pragma ivdep
    #pragma ii 1
    for(int i = (o-1); i < m; i+=2*M_gs){
        gs_data gs_info;
        #pragma unroll 
        for(int j = 0; j < M_gs; j++){
            int k1 = __burst_coalesced_load(&gd[i+j]) - 1;
            int k2 = __burst_coalesced_load(&gd[i+j+1]) - 1;
            
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
                
            gs_info.add1[j] = __pipelined_load(&wn[ji1][idx1]);
            gs_info.add2[j] = __pipelined_load(&wn[ji2][idx2]);
            gs_info.bank1[j] = ji1;
            gs_info.id1[j] = idx1;
            gs_info.bank2[j] = ji2;
            gs_info.id2[j] = idx2;
        }
        write_channel_intel(gs_add_channel,gs_info);
    }
}

__attribute__((scheduler_target_fmax_mhz(300)))
__kernel void gs_add(__global double * restrict w_1,
                     __global double * restrict w_2,
                     __global double * restrict w_3,
                     __global double * restrict w_4,
                     const int m,
                     const int o
                     ){
    
    __global double * restrict wn[NBANKS];
    wn[0] = w_1;
    wn[1] = w_2;
    wn[2] = w_3;
    wn[3] = w_4;
    #pragma ivdep
    #pragma ii 1
    for(int i = (o-1); i < m; i+=2*M_gs){
        gs_data gs_info = read_channel_intel(gs_add_channel);
        #pragma unroll 
        for(int j = 0; j < M_gs; j++){
            double tmp = gs_info.add1[j] + gs_info.add2[j];
            wn[gs_info.bank1[j]][gs_info.id1[j]] = tmp;
            wn[gs_info.bank2[j]][gs_info.id2[j]] = tmp;
            //__pipelined_store(&wn[gs_info.bank1[j]][gs_info.id1[j]], tmp);
            //__pipelined_store(&wn[gs_info.bank2[j]][gs_info.id2[j]], tmp);

        }
    }
   
}


