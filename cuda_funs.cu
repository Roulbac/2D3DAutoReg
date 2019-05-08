#include <math.h>


__device__ const float MAX_FLOAT32 = 3.4028e+038;
__device__ const float MIN_FLOAT32 = -3.4028e+038;
__device__ const float EPS_FLOAT32 = 2.22045e-016;

__global__ void backprojectPixel(
        const int h, const int w, float *dsts,
        const float *minv, const float *kinv, const int z_sign)
{
    int blockId = blockIdx.x + blockIdx.y*gridDim.x;
    int threadId = blockId * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;
    int i = threadId / w;
    int j = threadId - i*w;
    if (threadId < h*w)
    {
        float dotx = z_sign*(kinv[0]*i + kinv[1]*j + kinv[2]*1);
        float doty = z_sign*(kinv[3]*i + kinv[4]*j + kinv[5]*1);
        float dotz = z_sign*(kinv[6]*i + kinv[7]*j + kinv[8]*1);
        dsts[3*threadId + 0] = minv[0]*dotx + minv[1]*doty + minv[2]*dotz + minv[3]*1;
        dsts[3*threadId + 1] = minv[4]*dotx + minv[5]*doty + minv[6]*dotz + minv[7]*1;
        dsts[3*threadId + 2] = minv[8]*dotx + minv[9]*doty + minv[10]*dotz + minv[11]*1;
    }
}

__device__ int getIJK(
        const float src, const float dst, const float minAxyz,
        const float amin, const float b, const float s)
{
    return floor((src + 0.5*(minAxyz + amin)*(dst-src) - b)/s);
}

__device__ void getAlphas(
        const float b, const float s, const float p1, 
        const float p2, const int n, float *amin, float *amax)
{
   if (fabsf(p2 - p1) < EPS_FLOAT32) {
       *amin = MIN_FLOAT32;
       *amax = MAX_FLOAT32;
   }
   else{
       *amin = (b - p1)/(p2 - p1);
       *amax = (b+(n-1)*s - p1)/(p2 - p1);
       if (*amin > *amax){
           float temp = *amin;
           *amin = *amax;
           *amax = temp;
       }
   }
}

__device__ float getAx(
        const float p1, const float p2, const int n,
        const float b, const float s, const float axmin,
        const float axmax, const float amin, const float amax)
{
    float ax = 0;
    int imin = 1;
    int imax = n-2;
    if(fabsf(p2 - p1) < EPS_FLOAT32){
        ax = MAX_FLOAT32;
    }
    else if (p1 < p2){
        if(fabsf(amin - axmin) > EPS_FLOAT32){
            imin = floor((p1 + amin*(p2-p1) - b)/s + 1);
        }
        ax = ((b + imin*s) - p1)/(p2 - p1);
    }
    else {
        if(fabsf(amin - axmin) > EPS_FLOAT32){
            imax = ceil((p1 + amin*(p2 - p1) - b)/s - 1);
        }
        ax = ((b + imax*s) - p1)/(p2 - p1);
    }
    return ax;
}

__global__ void traceRay(
        const float *src, const float *dsts, float *raysums,
        const float *rho, const float *bs, const int *ns,
        const float *spacing, const int h, const int w)
{
    int blockId = blockIdx.x + blockIdx.y*gridDim.x;
    int threadId = blockId * (blockDim.x*blockDim.y) + (threadIdx.y*blockDim.x) + threadIdx.x;
    // int i = threadId / w;
    // int j = threadId - i*w;
    if (threadId < h*w){
        float dstx = dsts[3*threadId + 0];
        float dsty = dsts[3*threadId + 1];
        float dstz = dsts[3*threadId + 2];
        float bx = bs[0];
        float by = bs[1];
        float bz = bs[2];
        float srcx = src[0];
        float srcy = src[1];
        float srcz = src[2];
        int nx = ns[0];
        int ny = ns[1];
        int nz = ns[2];
        float sx = spacing[0];
        float sy = spacing[1];
        float sz = spacing[2];
        float axmin, axmax, aymin, aymax, azmin, azmax;
        getAlphas(bx, sx, srcx, dstx, nx, &axmin, &axmax);
        getAlphas(by, sy, srcy, dsty, ny, &aymin, &aymax);
        getAlphas(bz, sz, srcz, dstz, nz, &azmin, &azmax);
        float amin = fmaxf(axmin, fmaxf(aymin, azmin));
        float amax = fminf(axmax, fminf(aymax, azmax));
        if(amin > amax || (amin < 0 || amin == 0)){
            raysums[threadId] = 0;
            return;
        }   
        else {
            float ptx = srcx + amin*(dstx - srcx);
            float pty = srcy + amin*(dsty - srcy);
            float ptz = srcz + amin*(dstz - srcz);
            if( ((ptx > (bx + (nx-1)*sx)) || (ptx < bx)) ||
                ((pty > (by + (ny-1)*sy)) || (pty < by)) ||
                ((ptz > (bz + (nz-1)*sz)) || (ptz < bz))   ) {
                raysums[threadId] = 0;
                return;
            }
            float ax = getAx(srcx, dstx, nx, bx, sx, axmin, axmax, amin, amax);
            float ay = getAx(srcy, dsty, ny, by, sy, aymin, aymax, amin, amax);
            float az = getAx(srcz, dstz, nz, bz, sz, azmin, azmax, amin, amax);
            float dconv = sqrtf(
                    (dstx-srcx)*(dstx-srcx) +
                    (dsty-srcy)*(dsty-srcy) +
                    (dstz-srcz)*(dstz-srcz)  );
            float d12 = 0;
            float ac = amin;
            float minAxyz = fminf(ax, fminf(ay, az));
            int i = getIJK(srcx, dstx, minAxyz, amin, bx, sx);
            int j = getIJK(srcy, dsty, minAxyz, amin, by, sy);
            int k = getIJK(srcz, dstz, minAxyz, amin, bz, sz);
            while((-1 < i && i < (nx-1)) && 
                  (-1 < j && j < (ny-1)) &&
                  (-1 < k && k < (nz-1))   ){
                if(ax == minAxyz){
                   d12 = d12 + (ax - ac)*dconv*rho[i + j*(nx-1) + k*(nx-1)*(ny-1)]; 
                   i = (srcx<dstx)?(i+1):(i-1);
                   ac = ax;
                   ax = ax + sx/fabsf(dstx - srcx);
                }
                else if(ay == minAxyz){
                   d12 = d12 + (ay - ac)*dconv*rho[i + j*(nx-1) + k*(nx-1)*(ny-1)]; 
                   j = (srcy<dsty)?(j+1):(j-1);
                   ac = ay;
                   ay = ay + sy/fabsf(dsty - srcy);
                }
                else {
                   d12 = d12 + (az - ac)*dconv*rho[i + j*(nx-1) + k*(nx-1)*(ny-1)]; 
                   k = (srcz<dstz)?(k+1):(k-1);
                   ac = az;
                   az = az + sz/fabsf(dstz - srcz);
                }
                minAxyz = fminf(ax, fminf(ay, az));
            }
            raysums[threadId] = d12;
        }
    }
}
