#include "nanorod.h"
#include "object.h"
#include "vector.h"
#include "gpu/ray_gpu.cu"

#include <vector>

unsigned nTriCount = 0;
PrimGpuObj *pNanoDevice = NULL;
PrimGpuObj_host *pNanoHost = NULL;

unsigned nPlaneTriCount = 0;
PrimGpuObj *pNanoPlaneDevice = NULL;
PrimGpuObj_host *pNanoPlaneHost = NULL;

//
#define EPSI_GPU 0.01

static void copyTriangle(PrimGpuObj_host *pCurrPrimGpuObj, ObjObject *pObjObj, Triangle *pCurrTri, bool bCap = false, float offset = 0)
{
	pCurrPrimGpuObj->eType = TRI_GPU;
	pCurrPrimGpuObj->nId = pCurrTri->_id;

	vecCopy(pCurrPrimGpuObj->_mat.specColor, pObjObj->_mat.specColor);
	vecCopy(pCurrPrimGpuObj->_mat.diffColor, pObjObj->_mat.diffColor);
	vecCopy(pCurrPrimGpuObj->_mat.ambiColor, pObjObj->_mat.ambiColor);
	
	for(int n = 0; n < 3; n ++)
	{
		vecCopy(pCurrPrimGpuObj->_vertices[n], pCurrTri->_vertices[n]);
		vecCopy(pCurrPrimGpuObj->_vnormal[n], pCurrTri->_vnormal[n]);
	}
	vecCopy(pCurrPrimGpuObj->_normal, pCurrTri->_normal);

	/*pObjObj->_bSmooth*/ 
	if(!bCap)
	{
		pCurrPrimGpuObj->_bSmooth = ((pCurrTri->_vertices[0][2] + 
									 pCurrTri->_vertices[1][2] + 
									 pCurrTri->_vertices[2][2]) / 3.f) <  - offset - 1;
	}
	else
	{
		pCurrPrimGpuObj->_bSmooth = false;
	}
								 
	pCurrPrimGpuObj->_bHasVNorm = pObjObj->_bHasVNorm;
}


//

void copyNanoGeo(ObjObject *pNanoGeo, float offset)
{
	nTriCount = 0;

	std::vector<Triangle *> tmpVec;
	for(int j = 0; j < pNanoGeo->getTriCount(); j ++)
	{
		Triangle *pCurrTri = pNanoGeo->getTriangle(j);
		//if(	((pCurrTri->_vertices[0][2] + 
		//	 pCurrTri->_vertices[1][2] + 
		//	 pCurrTri->_vertices[2][2]) / 3.f) > - offset - 1)
		{
			tmpVec.push_back(pCurrTri);
			nTriCount ++;
		}
	}//	for


	if(pNanoDevice)
	{
		cudaFree(pNanoDevice);
	}
	cudaError_t err = cudaMalloc((void**)&pNanoDevice, sizeof(PrimGpuObj) * nTriCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
		
	if(pNanoHost)
	{
		free(pNanoHost);
	}
	pNanoHost = (PrimGpuObj_host*)malloc(sizeof(PrimGpuObj_host) * nTriCount);
	if(!pNanoHost)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	//
	for(int j = 0; j < tmpVec.size(); j ++)
	{
		Triangle *pCurrTri = tmpVec[j];
		PrimGpuObj_host *pCurrHostObj = &pNanoHost[j];

		copyTriangle(pCurrHostObj, pNanoGeo, pCurrTri, false, offset);
	}//	for

	err = cudaMemcpy(pNanoDevice, pNanoHost, sizeof(PrimGpuObj_host) * nTriCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA Memcpy error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
	cudaThreadSynchronize();
}
void nanoGeoDestroy()
{
	if(pNanoDevice)
	{
		cudaFree(pNanoDevice);
		pNanoDevice = NULL;
	}

	if(pNanoHost)
	{
		free(pNanoHost);
		pNanoHost = NULL;
	}	
}

///		Nano Plane


void copyNanoPlane(ObjObject *pNanoGeo, float offset)
{
	nPlaneTriCount = 0;

	std::vector<Triangle *> tmpVec;
	for(int j = 0; j < pNanoGeo->getTriCount(); j ++)
	{
		Triangle *pCurrTri = pNanoGeo->getTriangle(j);
		//if(	((pCurrTri->_vertices[0][2] + 
		//	 pCurrTri->_vertices[1][2] + 
		//	 pCurrTri->_vertices[2][2]) / 3.f) > - offset - 1)
		{
			tmpVec.push_back(pCurrTri);
			nPlaneTriCount ++;
		}
	}//	for

	if(pNanoPlaneDevice)
	{
		cudaFree(pNanoPlaneDevice);
	}
	cudaError_t err = cudaMalloc((void**)&pNanoPlaneDevice, sizeof(PrimGpuObj) * nPlaneTriCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
		
	if(pNanoPlaneHost)
	{
		free(pNanoPlaneHost);
	}
	pNanoPlaneHost = (PrimGpuObj_host*)malloc(sizeof(PrimGpuObj_host) * nPlaneTriCount);
	if(!pNanoPlaneHost)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	//
	for(int j = 0; j < tmpVec.size(); j ++)
	{
		Triangle *pCurrTri = tmpVec[j];
		PrimGpuObj_host *pCurrHostObj = &pNanoPlaneHost[j];

		copyTriangle(pCurrHostObj, pNanoGeo, pCurrTri);
	}//	for

	err = cudaMemcpy(pNanoPlaneDevice, pNanoPlaneHost, sizeof(PrimGpuObj_host) * nPlaneTriCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA Memcpy error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
	cudaThreadSynchronize();
}

void nanoPlaneDestroy()
{
	if(pNanoPlaneDevice)
	{
		cudaFree(pNanoPlaneDevice);
		pNanoPlaneDevice = NULL;
	}

	if(pNanoPlaneHost)
	{
		free(pNanoPlaneHost);
		pNanoPlaneHost = NULL;
	}
}

///		Cap 0

unsigned nCap0TriCount = 0;
PrimGpuObj *pCap0Device = NULL;
PrimGpuObj_host *pCap0Host = NULL;

void copyInternalCap0(ObjObject *pCap0Obj, float offset)
{
	std::vector<Triangle *> tmpVec;

	for(int j = 0; j < pCap0Obj->getTriCount(); j ++)
	{
		Triangle *pCurrTri = pCap0Obj->getTriangle(j);
		if(	((pCurrTri->_vertices[0][2] + 
			 pCurrTri->_vertices[1][2] + 
			 pCurrTri->_vertices[2][2]) / 3.f) > - offset - 1)
		{
			tmpVec.push_back(pCurrTri);
			nCap0TriCount ++;
		}
	}//	for

	if(pCap0Device)
	{
		cudaFree(pCap0Device);
	}
	cudaError_t err = cudaMalloc((void**)&pCap0Device, sizeof(PrimGpuObj) * nCap0TriCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
		
	if(pCap0Host)
	{
		free(pCap0Host);
	}
	pCap0Host = (PrimGpuObj_host*)malloc(sizeof(PrimGpuObj_host) * nCap0TriCount);
	if(!pCap0Host)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	//
	for(int j = 0; j < tmpVec.size(); j ++)
	{
		PrimGpuObj_host *pCurrHostObj = &pCap0Host[j];
		copyTriangle(pCurrHostObj, pCap0Obj, tmpVec[j], true);
	}//	for

	err = cudaMemcpy(pCap0Device, pCap0Host, sizeof(PrimGpuObj_host) * nCap0TriCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA Memcpy error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
	cudaThreadSynchronize();
}

void internalCap0Destroy()
{
	if(pCap0Device)
	{
		cudaFree(pCap0Device);
		pCap0Device = NULL;
	}

	if(pCap0Host)
	{
		free(pCap0Host);
		pCap0Host = NULL;
	}	
}

///		Cap 1

unsigned nCap1TriCount = 0;
PrimGpuObj *pCap1Device = NULL;
PrimGpuObj_host *pCap1Host = NULL;

void copyInternalCap1(ObjObject *pCap1Obj, float offset)
{
	std::vector<Triangle *> tmpVec;

	for(int j = 0; j < pCap1Obj->getTriCount(); j ++)
	{
		Triangle *pCurrTri = pCap1Obj->getTriangle(j);
		if(	((pCurrTri->_vertices[0][2] + 
			 pCurrTri->_vertices[1][2] + 
			 pCurrTri->_vertices[2][2]) / 3.f) > -offset - 1)
		{
			tmpVec.push_back(pCurrTri);
			nCap1TriCount ++;
		}
	}//	for

	if(pCap1Device)
	{
		cudaFree(pCap1Device);
	}
	cudaError_t err = cudaMalloc((void**)&pCap1Device, sizeof(PrimGpuObj) * nCap1TriCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
		
	if(pCap1Host)
	{
		free(pCap1Host);
	}
	pCap1Host = (PrimGpuObj_host*)malloc(sizeof(PrimGpuObj_host) * nCap1TriCount);
	if(!pCap0Host)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	//
	for(int j = 0; j < tmpVec.size(); j ++)
	{
		PrimGpuObj_host *pCurrHostObj = &pCap1Host[j];
		copyTriangle(pCurrHostObj, pCap1Obj, tmpVec[j], true, offset);
	}//	for

	err = cudaMemcpy(pCap1Device, pCap1Host, sizeof(PrimGpuObj_host) * nCap1TriCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA Memcpy error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
	cudaThreadSynchronize();
}

void internalCap1Destroy()
{
	if(pCap1Device)
	{
		cudaFree(pCap1Device);
		pCap1Device = NULL;
	}

	if(pCap1Host)
	{
		free(pCap1Host);
		pCap1Host = NULL;
	}
}

///		Slice

unsigned nSliceTriCount = 0;
PrimGpuObj *pSliceDevice = NULL;
PrimGpuObj_host *pSliceHost = NULL;

void copySlice(ObjObject *pSliceObj)
{
	nSliceTriCount = pSliceObj->getTriCount();

	if(pSliceDevice)
	{
		cudaFree(pSliceDevice);
	}
	cudaError_t err = cudaMalloc((void**)&pSliceDevice, sizeof(PrimGpuObj) * nSliceTriCount);
	if(err != cudaSuccess)
	{
		printf("CUDA error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
		
	if(pSliceHost)
	{
		free(pSliceHost);
	}
	pSliceHost = (PrimGpuObj_host*)malloc(sizeof(PrimGpuObj_host) * nSliceTriCount);
	if(!pNanoHost)
	{
		printf("malloc failed %s, %s \n", __FILE__, __LINE__);
	}

	//
	for(int j = 0; j < nSliceTriCount; j ++)
	{
		Triangle *pCurrTri = pSliceObj->getTriangle(j);
		PrimGpuObj_host *pCurrHostObj = &pSliceHost[j];

		copyTriangle(pCurrHostObj, pSliceObj, pCurrTri);
	}//	for

	err = cudaMemcpy(pSliceDevice, pSliceHost, sizeof(PrimGpuObj_host) * nSliceTriCount, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("CUDA Memcpy error %s, %s, %s \n", __FILE__, __LINE__, cudaGetErrorString(err));
	}
	cudaThreadSynchronize();
}

void SliceDestroy()
{
	if(pSliceDevice)
	{
		cudaFree(pSliceDevice);
		pSliceDevice = NULL;
	}

	if(pSliceHost)
	{
		free(pSliceHost);
		pSliceHost = NULL;
	}
}

//////////////////////////////////////////////////////////////////////////

__device__
PrimGpuObj* isRodHit_gpu( PrimGpuObj *gpuObjs, unsigned nTriCount, Ray_gpu *ray, float *pt, vect3d_gpu &vNorm );

//////////////

__device__
bool isLightVisible( PrimGpuObj *gpuNodObjs, unsigned nTriCount, vect3d_gpu &vHitPoint, vect3d_gpu &vNorm, vect3d_gpu &lightPos)
{
	vect3d_gpu dir;
	points2vec_gpu(vHitPoint, lightPos, dir);	

	//	1. Check normal first
	if(dot_product_gpu(vNorm , dir) <= 0)
	{
		return false;
	}

	//	2. Check intersection then
	Ray_gpu ray(vHitPoint, dir);	float t; vect3d_gpu tmp;
	if(isRodHit_gpu( gpuNodObjs, nTriCount, &ray, &t, tmp ) == NULL)
	{
		return true;
	}
	//	hit, but farther than the light pos?
	return (t > (1.f));
}

__device__
void clampColor_gpu(vect3d_gpu &pColor)
{
	for(int i = 0; i < 3; i ++)
	{
		if(pColor.data[i] > 1.f) pColor.data[i] = 1.f;
	}
}

__device__
void color_multiply_gpu(vect3d_gpu &color1, vect3d_gpu &color2, vect3d_gpu &rColor)
{
	rColor.data[0] = color1.data[0] * color2.data[0];
	rColor.data[1] = color1.data[1] * color2.data[1];
	rColor.data[2] = color1.data[2] * color2.data[2];
}

__device__
void evalPhong(vect3d_gpu &start_point, vect3d_gpu &vHitPoint, vect3d_gpu &vNorm, PrimGpuObj *pObj, vect3d_gpu &lightPos, vect3d_gpu &retColor)
{
	//	Light params
	//
	float fAttenuation = 0.5;

	//	Get Eye2Point view vec
	vect3d_gpu dir;
	vect3d_gpu v2Eye;
	vect3d_gpu v2EyeNormalized;
	points2vec_gpu(vHitPoint, start_point , v2Eye);			
	vecCopy_gpu(v2EyeNormalized, v2Eye);
	normalize_gpu(v2EyeNormalized);	

	//	Get Point2Light vec
	vect3d_gpu v2Light;
	vect3d_gpu vLightPos;
	vect3d_gpu v2LightNormalized;

	vect3d_gpu pos;
	vecCopy_gpu(pos, lightPos);

	points2vec_gpu(vHitPoint, pos, v2Light);	
	vecCopy_gpu(v2LightNormalized, v2Light);
	normalize_gpu(v2LightNormalized);	// vec. L

	vect3d_gpu tmp0;	// ambient
	vect3d_gpu tmp1;	// diffuse
	vect3d_gpu tmp2;	// specular

	//	ambient part
	vect3d_gpu lightAmbColor(1, 1, 1);
	vect3d_gpu lightDiffColor(1, 1, 1);
	vect3d_gpu lightSpecColor(1, 1, 1);

	color_multiply_gpu(lightAmbColor, pObj->_mat.ambiColor, tmp0);	

	//	diffuse part		
	float v1 = dot_product_gpu(v2LightNormalized, vNorm);
	float c1 = (v1 > EPSI_GPU) ? v1 : 0;
	color_multiply_gpu(lightDiffColor, pObj->_mat.diffColor, tmp1);
	vecScale_gpu(tmp1, c1, tmp1);	

	// specular part
/*	vect3d_gpu vS;
	point2point_gpu(v2Light, v2Eye, vS);	normalize_gpu(vS);
	float v2 = dot_product_gpu(vS, vNorm);
	float c2 = (v2 > EPSI_GPU) ? v2 : 0;
	c2 = pow(c2, pObj->_mat.fShininess);
	color_multiply_gpu(lightSpecColor, pObj->_mat.specColor, tmp2);
	vecScale_gpu(tmp2, c2, tmp2);*/	

	//	add to light sum
	vect3d_gpu tmp;
	point2point_gpu(tmp, tmp0, tmp);	//	adding ambient color
	point2point_gpu(tmp, tmp1, tmp);			//	adding diffuse color
	point2point_gpu(tmp, tmp2, tmp);			//	adding specular color
	vecScale_gpu(tmp, fAttenuation, retColor);		//	calc. attenuation

	clampColor_gpu(retColor);
}	

//////////////////////////////////////////////////////////////////////////

__device__
bool isTriangleHit_gpu(	vect3d_gpu vertices[3], Ray_gpu &ray, 
						float *pt, float *pu, float *pv)
{
	//
	//	Real-Time Rendering 2nd, 13.7.2
	//
	vect3d_gpu e1; points2vec_gpu(vertices[0], vertices[1], e1);
	vect3d_gpu e2; points2vec_gpu(vertices[0], vertices[2], e2);

	vect3d_gpu p;  cross_product_gpu(ray.direction_vec, e2, p);
	float a  = dot_product_gpu(e1, p);
	
	if(a > -EPSI_GPU && a < EPSI_GPU)
	{
		return false;
	}

	float f  = 1.f / a;
	vect3d_gpu s; points2vec_gpu(vertices[0], ray.start_point, s);
	float u = f * dot_product_gpu(s, p);
	if(u < 0.f || u > 1.f)
	{
		return false;
	}

	vect3d_gpu q;	cross_product_gpu(s, e1, q);
	float v = f * dot_product_gpu(ray.direction_vec, q);
	if(v < 0.f || (u + v) > 1.f)
	{
		return false;
	}

	float t = f * dot_product_gpu(e2, q);
	if(t <= EPSI_GPU)
	{
		return false;
	}

	*pt = t;
	*pu = u;
	*pv = v;
	
	return true;
}

__device__
bool isTriHit_gpu(PrimGpuObj *pObj, Ray_gpu *ray, vect3d_gpu &tmpNorm, float *tmpT)
{
	
	float u = 0, v = 0;
	if(isTriangleHit_gpu(pObj->_vertices, *ray, tmpT, &u, &v))
	{
		if(pObj->_bSmooth && pObj->_bHasVNorm)
		{
			vect3d_gpu vSmoothNorm;
			point2point_gpu(pObj->_vnormal[1], vSmoothNorm, vSmoothNorm);
			vecScale_gpu(vSmoothNorm, u, vSmoothNorm);

			vect3d_gpu vnorm2;
			point2point_gpu(pObj->_vnormal[2], vnorm2, vnorm2);
			vecScale_gpu(vnorm2, v, vnorm2);

			vect3d_gpu vnorm3;
			point2point_gpu(pObj->_vnormal[0], vnorm3, vnorm3);
			vecScale_gpu(vnorm3, (1 - u - v), vnorm3);

			point2point_gpu(vSmoothNorm, vnorm2, vSmoothNorm);
			point2point_gpu(vSmoothNorm, vnorm3, tmpNorm);

			normalize_gpu(tmpNorm);
		}
		else
		{
			vecCopy_gpu(tmpNorm, pObj->_normal);
		}
		return true;
	}
	return false;
}

__device__
PrimGpuObj* isRodHit_gpu( PrimGpuObj *gpuObjs, unsigned nTriCount, Ray_gpu *ray, float *pt, vect3d_gpu &vNorm )
{
	vect3d_gpu norm;
	float t = 99999999.0;
	PrimGpuObj *pRetObj = NULL;

	//	Linear Search
	//
	for(unsigned i = 0; i < nTriCount; i ++)
	{
		vect3d_gpu tmpNorm;
		float tmpT = 0;

		PrimGpuObj *pObj = gpuObjs + i;

		if(isTriHit_gpu(pObj, ray, tmpNorm, &tmpT))
		{
			if(tmpT < t && tmpT > 0)
			{
				t = tmpT;
				vecCopy_gpu(norm, tmpNorm);					
				//fEmit  = pObj->_fEmitRatio; 
				//fReflR = pObj->_fReflectionRatio;
				//fRefrR = pObj->_fRefractionRatio;
				//fRefrRK= pObj->_fRefractionK;

				pRetObj = pObj;
			}//	if
		}//	if
	}//	for

	if( t < 99999999.0 && t > EPSI_GPU)
	{
		vecCopy_gpu(vNorm, norm);
		*pt = t;		
		//*pfEmit  = fEmit;
		//*pfReflR = fReflR; 
		//*pfRefrR = fRefrR; 
		//*pfRefrRK= fRefrRK;

		return pRetObj;
	}	

	return NULL;

}
