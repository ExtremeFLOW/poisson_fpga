#define M 8
#define M2 8
#define LX1 8
#define LY1 8
#define LZ1 8
#define POW 9
#define FPX float
__kernel void cg(__global float * restrict x, 
                 __global float * restrict p,
                 __global float * restrict r,
                 __global float * restrict w,
                 __global const float * restrict mult,
                 __global const float * restrict x_cord,
		 __global const float * restrict y_cord,
                 __global const float * restrict z_cord,
                 __global const float * restrict jx,
                 __global const float * restrict jxt,
                 __global const float * restrict w3,
                 __global const float * restrict dx,
                 __global const float * restrict dxt,
                 __global int * restrict mask,
                 __global float * restrict rtz1,
                 __global float * restrict rtz2,
                 __global float * restrict beta,
                 const int N,
                 __global const int * restrict b,
                 __global const int * restrict gd,
                 __global const int * restrict dg,
                 __global const float * restrict jacinv,
                 const int m,
                 const int o,
                 const int nb)
{   

    float shw3[LX1*LY1*LZ1];
    float shjx[LX1*2];
    float shjxt[LX1*2];
    float shdx[LX1*LY1];
    float shdxt[LX1*LY1];

    #pragma unroll
    for(unsigned i = 0; i < 16; i++){
        shjx[i] = jx[i];
        shjxt[i] = jxt[i];
    }
    #pragma unroll 32
    for(unsigned ij=0; ij<LX1*LY1; ++ij){
        shdx[ij] = dx[ij];
        shdxt[ij] = dxt[ij];
    }

    #pragma unroll 64
    for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
        shw3[ijk] = w3[ijk];
    }
    


    #pragma ivdep
    for(unsigned e = 0; e < (N>>POW); e++){
        float x_c[8];
        float y_c[8];
        float z_c[8];
        float shur[LX1*LY1*LZ1];
        float shus[LX1*LY1*LZ1];
        float shut[LX1*LY1*LZ1];
        float shw[LX1*LY1*LZ1];
        float shu[LX1*LY1*LZ1];
        float tmp_x1[LX1*2*2];
        float tmp_y1[LX1*2*2];
        float tmp_z1[LX1*2*2];
        float tmp_x2[LX1*LY1*2];
        float tmp_y2[LX1*LY1*2];
        float tmp_z2[LX1*LY1*2];
        float tmpx[LX1*LY1*LZ1];
        float tmpy[LX1*LY1*LZ1];
        float tmpz[LX1*LY1*LZ1];
        float shjv[LX1*LY1*LZ1];
        int ele = e * LX1*LY1*LZ1;
      

        #pragma unroll 32
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            float temp =  p[ijk + ele];
            shjv[ijk] = jacinv[ijk+ele];
            shu[ijk] =  temp;
        }
        #pragma unroll 128
        for (unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            shw[ijk] = shu[ijk] + shjv[ijk]; 
            
        }

        #pragma unroll 32
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            w[ijk + ele] = shw[ijk];
        }
    }
}

