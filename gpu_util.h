#ifndef GPU_UTIL_H
#define GPU_UTIL_H

#include "gpu/geometry_gpu.h"

void sendConstants2GPU();
void copySceneGeomotry();
void releaseSceneGeomotry();

//
static void copySquare(PrimGpuObj_host *, Square *);
void translateObjectGPU(vect3d vectorTr, int objectIndex);
void setObjectCenterGPU(vect3d vectorTr, int objectIndex);
void gpu_destroy();



__constant__ float epsi_gpu;  
__device__
float getEpsiGpu()
{
	return epsi_gpu;
}

#define AMBI_X 1
#define AMBI_Y 1
#define AMBI_Z 1


#endif