#include "gpu_util.h"

#include "global.h"
#include "consts.h"
#include "bbox.h"

#include <vector>
#include <cuda_runtime.h>


////////////

unsigned nCurrObjCount = 0;

PrimGpuObj *gpuObjs = NULL;
PrimGpuObj_host *hostObjs = NULL;

////////////

void gpu_destroy()
{
	releaseSceneGeomotry();
}

//////////////////

void sendConstants2GPU()
{
	//vect3d ambiColor; scene.getAmbiColor(ambiColor);
	//cudaError_t err = cudaMemcpyToSymbol(AmbiColor_gpu, ambiColor.data, sizeof(float) * 3/*, 0, cudaMemcpyHostToDevice*/);
	cudaError_t err = cudaMemcpyToSymbol(epsi_gpu, &epsi, sizeof(float) /*, 0,	cudaMemcpyHostToDevice*/);
	
	cudaThreadSynchronize();
}


void copySceneGeomotry()
{
#ifndef DATA_2D
	//%%%Originally 6
	//%%%at this time we have 2 gpuobjs, the big cube and the small one for the tool,
	//%%%so this set of functions have to be changed for something that accepts any # of geometry
	unsigned nTotalPrimaryCount = 12;
#else
	unsigned nTotalPrimaryCount = 6;
#endif

	//	1. Re-Alloc space for Objects
	if(gpuObjs)
	{
		cudaFree(gpuObjs);
	}
	cudaError_t err = cudaMalloc((void**)&gpuObjs, sizeof(PrimGpuObj) * nTotalPrimaryCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
		
	if(hostObjs)
	{
		free(hostObjs);
	}
	hostObjs = (PrimGpuObj_host*)malloc(sizeof(PrimGpuObj_host) * nTotalPrimaryCount);
	if(!hostObjs)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	//	2. Copy Objects
	unsigned nCurrPrimObjInx = 0;
	for(int i = 0; i < scene.getObjectNum(); i ++)
	{
		Object *pObj = scene.getObject(i);
		ObjType eType = pObj->getObjType();

		PrimGpuObj_host *pCurrPrimGpuObj = &hostObjs[nCurrPrimObjInx];

		//	Copy the common part
		pCurrPrimGpuObj->nId = pObj->_id;

		switch(eType)
		{
		case CUBE_CPU:
			{
				Cube *pCube = dynamic_cast<Cube*>(pObj);
				for(int m = 0; m < 6; m ++)
				{
					pCurrPrimGpuObj = &hostObjs[nCurrPrimObjInx];
					copySquare(pCurrPrimGpuObj, pCube->_vs[m]);	
#ifndef DATA_2D
					if(i == 1)
						pCurrPrimGpuObj->rayCast = false;
					else 
						pCurrPrimGpuObj->rayCast = true;
#endif
					nCurrPrimObjInx ++;
				}
			}
			break;

		default:
			printf("not supported obj type \n");
			return;
			break;
		}
	}//	copy for
	
	//	cuda copy objs
	//
	nCurrObjCount = nTotalPrimaryCount;
	err = cudaMemcpy(gpuObjs, hostObjs, sizeof(PrimGpuObj) * nTotalPrimaryCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	cudaThreadSynchronize();
}

void releaseSceneGeomotry()
{
	nCurrObjCount = 0;

	if(gpuObjs)
	{
		cudaFree(gpuObjs);
		gpuObjs = NULL;
	}

	if(hostObjs)
	{
		free(hostObjs);
		hostObjs = NULL;
	}	
}

static void copySquare(PrimGpuObj_host *pCurrPrimGpuObj, Square *pSqu)
{
	pCurrPrimGpuObj->eType = SQU_GPU;
				
	pCurrPrimGpuObj->nId = pSqu->_id;
	vecCopy(pCurrPrimGpuObj->_vNormal, pSqu->_vNormal);
	vecCopy(pCurrPrimGpuObj->_vWidthVec, pSqu->_vWidthVec); 
	vecCopy(pCurrPrimGpuObj->_vCenter, pSqu->_vCenter);

	pCurrPrimGpuObj->_nWidth = pSqu->_nWidth;
	pCurrPrimGpuObj->_nHeight = pSqu->_nHeight;	
	
	vecCopy(pCurrPrimGpuObj->_v2HeightVec, pSqu->_v2HeightVec);
	vecCopy(pCurrPrimGpuObj->_v2WidthVec, pSqu->_v2WidthVec);
}

//%%%
//Set an object' center in host memory and copy it to device. Since for now there are not many
//objects, this would be a good way of doing the translation
void setObjectCenterGPU(vect3d vectorTr, int objectIndex)
{
	//%%%Since right now just cubes
	int startIndex = 6 * objectIndex;
	for(int i = startIndex; i < startIndex + 6; i++)
	{
		hostObjs[i]._vCenter = vectorTr;
	}

	//Copy it to GPU
	cudaError_t err = cudaMemcpy(gpuObjs, hostObjs, sizeof(PrimGpuObj) * nCurrObjCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	cudaThreadSynchronize();
}
//Translate an object in host memory and copy it to device. Since for now there are not many
//objects, this would be a good way of doing the translation
void translateObjectGPU(vect3d vectorTr, int objectIndex)
{
	//%%%Since right now just cubes
	int startIndex = 6 * objectIndex;
	for(int i = startIndex; i < startIndex + 6; i++)
	{
		hostObjs[i]._vCenter = hostObjs[i]._vCenter + vectorTr;
	}

	//Copy it to GPU
	cudaError_t err = cudaMemcpy(gpuObjs, hostObjs, sizeof(PrimGpuObj) * nCurrObjCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}

	cudaThreadSynchronize();
}
//%%%