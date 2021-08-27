#define M 8
#define M2 8
#define M_gs 1
#define LX1 8
#define LY1 8
#define LZ1 8
#define NBANKS 4
#define MAX_DEG 8

typedef struct
{
	double add1[M_gs];
	double add2[M_gs];
	int bank1[M_gs];
	int bank2[M_gs];
	int id1[M_gs];
	int id2[M_gs];
} gs_data;

typedef struct
{       
        double val;
	int bank;
	int idx;
        int blk_len;
} gs_data_blk;

#pragma OPENCL EXTENSION cl_intel_channels : enable

channel gs_data gs_add_channel __attribute__((depth(128)));
channel gs_data_blk gs_add_blk_channel __attribute__((depth(128)));


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
                 __global double const * restrict w_1,
                 __global double const * restrict w_2,
                 __global double const * restrict w_3,
                 __global double const * restrict w_4,
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

    __global const double * restrict wn[NBANKS];
    wn[0] = w_1;
    wn[1] = w_2;
    wn[2] = w_3;
    wn[3] = w_4;


    //gather-scatter
    int k = 0;
    #pragma ivdep
    #pragma ii 1
    for(int i = 0; i < (o-1); i++){
        gs_data_blk blk_data;
        int k2 = gd[i]-1;
        int j2 = k2 % 128;
        int id2 = k2 >> 7;
        blk_data.bank = j2 >> 5;
        int off2 = j2 % 32;
        blk_data.idx =  off2 + id2*32;
        blk_data.val = wn[blk_data.bank][blk_data.idx];
        write_channel_intel(gs_add_blk_channel,blk_data);
    }
    
}

__kernel void gs_blk(__global double * restrict w_1,
                     __global double * restrict w_2,
                     __global double * restrict w_3,
                     __global double * restrict w_4,
                     const int nb
                     ){
    
    __global double * restrict wn[NBANKS];
    wn[0] = w_1;
    wn[1] = w_2;
    wn[2] = w_3;
    wn[3] = w_4;
    #pragma ivdep
    #pragma ii 1
    for(int i = 0; i < (nb-1); i++){
        gs_data_blk gs_blk_info = read_channel_intel(gs_add_blk_channel);
        double val = gs_blk_info.val;
        wn[gs_blk_info.bank][gs_blk_info.idx] = val;
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
    
}


