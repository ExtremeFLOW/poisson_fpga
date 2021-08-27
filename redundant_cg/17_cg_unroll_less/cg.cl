#define M 16
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
    float res = 0.0;
    int ma = mask[0];
    for(int i = 1; i < (ma + 1); i++){
       int k = mask[i];
       w[k-1] = 0.0; 
    }

    for( int i = 0; i < N; i+=M){
        #pragma unroll
        for( int k = 0; k < M; k++){
           res += w[i+k]*p[i+k]*mult[i+k];
        }
    }

    float pap = res;
    float alpha = rtz1[0]/pap;
    float res2 = 0.0;

    for( int i = 0; i < N; i+=M){
        #pragma unroll
        for( int k = 0; k < M; k++){ 
            x[i+k] = x[i+k] + alpha * p[i+k];
            float r_new= r[i+k] - alpha * w[i+k];
            res2 +=  r_new*r_new*mult[i+k];
            r[i+k] = r_new;
        }
    }

    rtz2[0] = rtz1[0];
    rtz1[0] = res2;
  
    beta[0] = rtz1[0]/rtz2[0];
 
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
        #pragma unroll
        for(unsigned i = 0; i < 8; i++){
            x_c[i] = x_cord[i+e*8];
            y_c[i] = y_cord[i+e*8];
            z_c[i] = z_cord[i+e*8];
        }
        #pragma loop_coalesce
        for(unsigned j = 0; j < 2*2; j++){
            #pragma unroll
            for(unsigned i = 0; i < LX1; i++){
                float temp1 = 0.0;
                float temp2 = 0.0;
                float temp3 = 0.0;
                #pragma unroll
                for(unsigned l = 0; l < 2; l++){
                    temp1 += shjx[i+l*LX1]*x_c[l+j*2];
                    temp2 += shjx[i+l*LX1]*y_c[l+j*2];
                    temp3 += shjx[i+l*LX1]*z_c[l+j*2];
                }
                tmp_x1[i+j*LX1] = temp1; 
                tmp_y1[i+j*LX1] = temp2; 
                tmp_z1[i+j*LX1] = temp3; 
            }
        }
        #pragma loop_coalesce
        #pragma ii 1
        #pragma ivdep
        for(unsigned k = 0; k < 2; k++){ 
            for(unsigned i = 0; i < LX1; i++){
                #pragma unroll
                for(unsigned j = 0; j < LX1; j++){
                    float temp1 = 0.0;
                    float temp2 = 0.0;
                    float temp3 = 0.0;
                    #pragma unroll
                    for(unsigned l = 0; l < 2; l++){
                        temp1 += tmp_x1[2*LX1*k + i + l*LX1]*shjxt[2*j+l];
                        temp2 += tmp_y1[2*LX1*k + i + l*LX1]*shjxt[2*j+l];
                        temp3 += tmp_z1[2*LX1*k + i + l*LX1]*shjxt[2*j+l];
                    }
                    tmp_x2[k*LX1*LX1+i*LX1+j] = temp1; 
                    tmp_y2[k*LX1*LX1+i*LX1+j] = temp2; 
                    tmp_z2[k*LX1*LX1+i*LX1+j] = temp3; 
                }
            }
        }
        #pragma loop_coalesce
        #pragma ii 1
        for(unsigned j = 0; j < LX1; j++){
            #pragma unroll 32
            for(unsigned i = 0; i < LX1*LX1; i++){
                float temp1 = 0.0;
                float temp2 = 0.0;
                float temp3 = 0.0;
                #pragma unroll
                for(unsigned l = 0; l < 2; l++){
                    temp1 += tmp_x2[i+l*LX1*LX1]*shjxt[l+j*2];
                    temp2 += tmp_y2[i+l*LX1*LX1]*shjxt[l+j*2];
                    temp3 += tmp_z2[i+l*LX1*LX1]*shjxt[l+j*2];
                }
                tmpx[j*LX1*LX1+i] = temp1; 
                tmpy[j*LX1*LX1+i] = temp2; 
                tmpz[j*LX1*LX1+i] = temp3; 
            }
        }

        #pragma unroll 64
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            float temp = r[ijk + ele] + beta[0] * p[ijk + ele];
            shjv[ijk] = jacinv[ijk+ele];
            shu[ijk] =  temp;
        }
        #pragma loop_coalesce
        #pragma ii 1
        for (unsigned k=0; k<LZ1; ++k){
            #pragma unroll 2
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll 
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    float rtmp = 0.0;
                    float stmp = 0.0;
                    float ttmp = 0.0;
                    float rtmpx = 0.0;
                    float stmpx = 0.0;
                    float ttmpx = 0.0;
                    float rtmpy = 0.0;
                    float stmpy = 0.0;
                    float ttmpy = 0.0;
                    float rtmpz = 0.0;
                    float stmpz = 0.0;
                    float ttmpz = 0.0;
                    #pragma unroll
                    for (unsigned l = 0; l<LX1; l++){
                      rtmp += shdxt[l+i*LX1] * shu[l+j*LX1 +k*LX1*LY1];
                      stmp += shdxt[l+j*LX1] * shu[i+l*LX1 + k*LX1*LY1];
                      ttmp += shdxt[l+k*LX1] * shu[ij + l*LX1*LY1];
                      rtmpx += shdxt[l+i*LX1] * tmpx[l+j*LX1 +k*LX1*LY1];
                      stmpx += shdxt[l+j*LX1] * tmpx[i+l*LX1 + k*LX1*LY1];
                      ttmpx += shdxt[l+k*LX1] * tmpx[ij + l*LX1*LY1];
                      
                      rtmpy += shdxt[l+i*LX1] * tmpy[l+j*LX1 +k*LX1*LY1];
                      stmpy += shdxt[l+j*LX1] * tmpy[i+l*LX1 + k*LX1*LY1];
                      ttmpy += shdxt[l+k*LX1] * tmpy[ij + l*LX1*LY1];
                      
                      rtmpz += shdxt[l+i*LX1] * tmpz[l+j*LX1 +k*LX1*LY1];
                      stmpz += shdxt[l+j*LX1] * tmpz[i+l*LX1 + k*LX1*LY1];
                      ttmpz += shdxt[l+k*LX1] * tmpz[ij + l*LX1*LY1];
                    }
                    float dxdr =rtmpx;
                    float dxds =stmpx;
                    float dxdt =ttmpx;
                    float dydr =rtmpy;
                    float dyds =stmpy;
                    float dydt =ttmpy;
                    float dzdr =rtmpz;
                    float dzds =stmpz;
                    float dzdt =ttmpz;
                    float drdx= dyds*dzdt-dydt*dzds;
                    float drdy= dxdt*dzds-dxds*dzdt;
                    float drdz= dxds*dydt-dxdt*dyds;
                    float dsdx= dydt*dzdr-dydr*dzdt;
                    float dsdy= dxdr*dzdt-dxdt*dzdr;
                    float dsdz= dxdt*dydr-dxdr*dydt;
                    float dtdx= dydr*dzds-dyds*dzdr;
                    float dtdy= dxds*dzdr-dxdr*dzds;
                    float dtdz= dxdr*dyds-dxds*dydr;
       
                    float g11 = drdx*drdx+drdy*drdy+drdz*drdz; 
                    float g22 = dsdx*dsdx+dsdy*dsdy+dsdz*dsdz;
                    float g33 = dtdx*dtdx+dtdy*dtdy+dtdz*dtdz;
                    float g12 = drdx*dsdx+drdy*dsdy+drdz*dsdz;
                    float g13 = drdx*dtdx+drdy*dtdy+drdz*dtdz;
                    float g23 = dsdx*dtdx+dsdy*dtdy+dsdz*dtdz;
       
                    g11=g11*shjv[ijk];
                    g22=g22*shjv[ijk];
                    g33=g33*shjv[ijk];
                    g12=g12*shjv[ijk];
                    g13=g13*shjv[ijk];
                    g23=g23*shjv[ijk];
                    g11=g11*shw3[ijk];
                    g22=g22*shw3[ijk];
                    g12=g12*shw3[ijk];
                    g33=g33*shw3[ijk];
                    g13=g13*shw3[ijk];
                    g23=g23*shw3[ijk];


                    shur[ijk] = g11*rtmp
                             + g12*stmp
                             + g13*ttmp;
                    shus[ijk] = g12*rtmp
                             + g22*stmp
                             + g23*ttmp;
                    shut[ijk]  = g13*rtmp
                             + g23*stmp
                             + g33*ttmp;
                }
            }
        }
        #pragma loop_coalesce 
        #pragma ii 1
        for (unsigned k=0; k<LZ1; ++k){
            #pragma unroll 2
            for(unsigned j = 0; j < LY1; j++){
                #pragma unroll
                for(unsigned i = 0; i < LX1; i++){
                    int ij = i + j*LX1;
                    int ijk = ij + k*LX1*LY1;
                    
                    float wijke = 0.0;
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

        #pragma unroll 64
        for(unsigned ijk=0; ijk<LX1*LY1*LZ1; ++ijk){
            w[ijk + ele] = shw[ijk];
            p[ijk + ele] = shu[ijk];
        }
    }

    //gather-scatter
    int k = 0;
    #pragma ivdep
    for(int i = 0; i < nb; i++){
        int idx[M2];
        int blk_len = b[i];
        idx[0] = gd[k] - 1;
        float cur = w[idx[0]];
        for(int j = 1; j < blk_len; j++){
            idx[j] = gd[k+j] - 1;
            cur = cur + w[idx[j]];
        }
        for(int j = 0; j < blk_len; j++){
            w[idx[j]] = cur;
        }
        k = k + blk_len;
    }
    #pragma ivdep
    for(int i = (o-1); i < m; i+=2){
        int idx1 = gd[i] - 1;
        int idx2 = gd[i+1] - 1;
        float tmp =w[idx1] + w[idx2];
        w[idx1] = tmp;
        w[idx2] = tmp;
    }

}


