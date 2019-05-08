# include <math.h>


__constant__ const float MAX_FLOAT32 = 3.4028e+038;
__constant__ const float MIN_FLOAT32 = -3.4028e+038;
__constant__ const float EPS_FLOAT32 = 2.22045e-016;
__constant__ float MU_WATER = 0.85684356;
__constant__ float MU_AIR = 0.0007937816;

__global__ void backprojectPixel(
    const int h, const int w, float * dsts,
    const float * minv, const float * kinv, const int z_sign)
{
    int blockId = blockIdx.x + blockIdx.y*gridDim.x;
    int localId = (threadIdx.y*blockDim.x) + threadIdx.x;
    int threadId = blockId * (blockDim.x*blockDim.y) + localId;

    __shared__ float sKinv[9];
    __shared__ float sMinv[12];

    if (localId < 9){
        sKinv[localId] = kinv[localId];
    }
    else if (localId < 21){
        sMinv[localId - 9] = minv[localId - 9];
    }
    __syncthreads()

    int i = threadId / w;
    int j = threadId - i*w;
    if (threadId < h*w)
    {
        float dotx = z_sign*(sKinv[0]*i + sKinv[1]*j + sKinv[2]*1);
        float doty = z_sign*(sKinv[3]*i + sKinv[4]*j + sKinv[5]*1);
        float dotz = z_sign*(sKinv[6]*i + sKinv[7]*j + sKinv[8]*1);
        dsts[3*threadId + 0] = sMinv[0]*dotx + sMinv[1]*doty + sMinv[2]*dotz + sMinv[3]*1;
        dsts[3*threadId + 1] = sMinv[4]*dotx + sMinv[5]*doty + sMinv[6]*dotz + sMinv[7]*1;
        dsts[3*threadId + 2] = sMinv[8]*dotx + sMinv[9]*doty + sMinv[10]*dotz + sMinv[11]*1;
    }
}

__device__ int getIJK(
    const float & src, const float & dst, const float & minAxyz,
    const float & amin, const float & b, const float & s)
{
    return floor((src + 0.5*(minAxyz + amin)*(dst-src) - b)/s);
}

__device__ void getAlphas(
    const float & b, const float & s, const float & p1,
    const float & p2, const int & n, float & amin, float & amax)
{
    if (fabsf(p2 - p1) < EPS_FLOAT32) {
        amin = MIN_FLOAT32;
        amax = MAX_FLOAT32;
    }
    else{
        amin = (b - p1)/(p2 - p1);
        amax = (b+(n-1)*s - p1)/(p2 - p1);
        if (amin > amax){
            float temp = amin;
            amin = amax;
            amax = temp;
        }
    }
}

__device__ float getAx(
    const float & p1, const float & p2, const int & n,
    const float & b, const float & s, const float & axmin,
    const float & axmax, const float & amin, const float & amax)
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
    const float * src, const float * dsts, float * raysums,
    const float * rho, const float * b, const float * sp,
    const int * n, const int h, const int w)
{
    int blockId = blockIdx.x + blockIdx.y*gridDim.x;
    int localId = (threadIdx.y*blockDim.x) + threadIdx.x;
    int threadId = blockId * (blockDim.x*blockDim.y) + localId;

    __shared__ float sB[3];
    __shared__ float sSp[3];
    __shared__ float sSrc[3];
    __shared__ int sN[3];

    if(localId < 3){
        sB[localId] = b[localId];
    } else if(localId < 6){
        sSp[localId - 3] = sp[localId - 3];
    } else if(localId < 9){
        sN[localId - 6] = n[localId - 6];
    } else if(localId < 12){
        sSrc[localId - 9] = src[localId - 9];
    }
    __syncthreads();

    if (threadId < h*w){
        float3 dst = make_float3(dsts[3*threadId], dsts[3*threadId + 1], dsts[3*threadId + 2]);
        float axmin, axmax, aymin, aymax, azmin, azmax;
        getAlphas(sB[0], sSp[0], sSrc[0], dst.x, sN[0], axmin, axmax);
        getAlphas(sB[1], sSp[1], sSrc[1], dst.y, sN[1], aymin, aymax);
        getAlphas(sB[2], sSp[2], sSrc[2], dst.z, sN[2], azmin, azmax);
        float amin = fmaxf(axmin, fmaxf(aymin, azmin));
        float amax = fminf(axmax, fminf(aymax, azmax));
        if(amin > amax | | (amin < 0)){
            raysums[threadId] = 1;
            return;
        }
        else {
            float3 pt = make_float3(sSrc[0] + amin*(dst.x - sSrc[0]),
                                    sSrc[1] + amin*(dst.y - sSrc[1]),
                                    sSrc[2] + amin*(dst.z - sSrc[2]));
            if(((pt.x > (sB[0] + (sN[0]-1)*sSp[0])) | | (pt.x < sB[0])) | |
                ((pt.y > (sB[1] + (sN[1]-1)*sSp[1])) | | (pt.y < sB[1])) | |
                ((pt.z > (sB[2] + (sN[2]-1)*sSp[2])) | | (pt.z < sB[2]))) {
                raysums[threadId] = 1;
                return;
            }
            float ax = getAx(sSrc[0], dst.x, sN[0], sB[0], sSp[0], axmin, axmax, amin, amax);
            float ay = getAx(sSrc[1], dst.y, sN[1], sB[1], sSp[1], aymin, aymax, amin, amax);
            float az = getAx(sSrc[2], dst.z, sN[2], sB[2], sSp[2], azmin, azmax, amin, amax);
            float dconv = sqrtf(
                    (dst.x-sSrc[0])*(dst.x-sSrc[0]) +
                    (dst.y-sSrc[1])*(dst.y-sSrc[1]) +
                    (dst.z-sSrc[2])*(dst.z-sSrc[2]));
            float d12 = 0;
            float ac = amin;
            float minAxyz = fminf(ax, fminf(ay, az));
            int i = getIJK(sSrc[0], dst.x, minAxyz, amin, sB[0], sSp[0]);
            int j = getIJK(sSrc[1], dst.y, minAxyz, amin, sB[1], sSp[1]);
            int k = getIJK(sSrc[2], dst.z, minAxyz, amin, sB[2], sSp[2]);
            while((-1 < i & & i < (sN[0]-1)) & &
                  (-1 < j & & j < (sN[1]-1)) & &
                  (-1 < k & & k < (sN[2]-1))){
                float mu = expf(MU_WATER-MU_AIR/1000*rho[i + j*(sN[0]-1) + k*(sN[0]-1)*(sN[1]-1)] + MU_WATER);
                if(ax == minAxyz){
                    d12 = d12 + (ax - ac)*dconv*mu;
                    i = (sSrc[0] < dst.x)?(i+1): (i-1);
                    ac = ax;
                    ax = ax + sSp[0]/fabsf(dst.x - sSrc[0]);
                }
                else if(ay == minAxyz){
                    d12 = d12 + (ay - ac)*dconv*mu;
                    j = (sSrc[1] < dst.y)?(j+1): (j-1);
                    ac = ay;
                    ay = ay + sSp[1]/fabsf(dst.y - sSrc[1]);
                }
                else {
                    d12 = d12 + (az - ac)*dconv*mu;
                    k = (sSrc[2] < dst.z)?(k+1): (k-1);
                    ac = az;
                    az = az + sSp[2]/fabsf(dst.z - sSrc[2]);
                }
                minAxyz = fminf(ax, fminf(ay, az));
            }
            raysums[threadId] = d12;
        }
    }
}
