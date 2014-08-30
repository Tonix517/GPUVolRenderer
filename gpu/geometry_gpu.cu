#include "geometry_gpu.h"
#include "ray_gpu.cu"
#include "gpu_util.h"

__device__
bool isSquareHit_gpu(PrimGpuObj *pObj, Ray_gpu *ray, float *tmpT, float *deltax, float *deltay)
{
	//	The hit point on the plane
	vect3d_gpu op;
	points2vec_gpu(pObj->_vCenter, ray->start_point, op);

	float dn = dot_product_gpu(ray->direction_vec, pObj->_vNormal);
	if(dn == 0.f)
	{
		return false;
	}
	//ratio between dot product of (vector from center to start of ray and normal)
	//and ray direction and normal
	float t = - dot_product_gpu(op, pObj->_vNormal) / dn;
	//	NOTE: since it is a 0-thickness plane, we need this.
	if(t <= getEpsiGpu())
	{
		return false;
	}

	//	Get the hit point
	vect3d_gpu vHitPoint;
	vect3d_gpu pView; vecScale_gpu(ray->direction_vec, t, pView);
	point2point_gpu(ray->start_point, pView, vHitPoint);

	vect3d_gpu vHitVec;
	points2vec_gpu(vHitPoint, pObj->_vCenter, vHitVec);

	float dx = dot_product_gpu(vHitVec, pObj->_v2HeightVec) / pow( pObj->_nWidth /2 , 2);
	float dy = dot_product_gpu(vHitVec, pObj->_v2WidthVec) / pow( pObj->_nHeight /2 , 2);
	
	if( fabs(dy) <= 1.f && fabs(dx) <= 1.f )
	{
		*tmpT = t;
		*(deltax) = dx;
		*(deltay) = dy;

		return true;
	}
	return false;
}

//////		BBox
//////

__device__
bool isHitOnPlane(Ray_gpu &ray, float min, float max, AxisType eType)
{
	float start = 0, dir = 0;

	switch(eType)	
	{
	case X_AXIS:
		start = ray.start_point.data[0];
		dir = ray.direction_vec.data[0];
		break;

	case Y_AXIS:
		start = ray.start_point.data[1];
		dir = ray.direction_vec.data[1];
		break;

	case Z_AXIS:
		start = ray.start_point.data[2];
		dir = ray.direction_vec.data[2];
		break;
	}

	//	just between the slabs? yes
	if(start <= max && start > min)
	{
		return true;
	}
	
	//	no marching in this direction?
	if(dir == 0)
	{
		return false;
	}

	float toMinT = (min - start)/dir;
	float toMaxT = (max - start)/dir;

	if( start <= min)
	{
		return toMinT <= toMaxT;
	}
	if( start >= max)
	{
		return toMaxT <= toMinT;
	}

	return false;
}

extern PrimGpuObj *gpuObjs;


__device__
unsigned isHit_gpu(	PrimGpuObj *gpuObjs, Ray_gpu *ray,
					PrimGpuObj *pHits[4], float *deltas, float hitPoints[4][3], bool *toolHit)
{
	float t = 99999999.0;

	//
	//%%%these values need to be higher because now we can hit more than 2 
	//squares
	float ts[4] = {0};
	PrimGpuObj *objs[4] = {0};
	float tmpDeltas[8] = {0};
	*toolHit = false;
	//

	unsigned nHit = 0;

#ifndef DATA_2D
	//%%%Changed 6 to 12  
	for(int i = 0; i < 12; i ++)
#else
	for(int i = 0; i < 6; i ++)
#endif
	{
		PrimGpuObj *pObj = gpuObjs + i;

		float tmpT = t, dx = 0, dy = 0;
		if(isSquareHit_gpu(pObj, ray, &tmpT, &dx, &dy))
		{
#ifndef DATA_2D //If the square is not supposed to be rayCast then we are hitting the tool 
			if (pObj->rayCast == false)
			{
				*toolHit = true;	
				continue;
			}
#endif
			if(tmpT < t && (nHit == 1 ? tmpT != ts[0] : true))	// the latter one: cross the edge.. no repeat
			{
				ts[nHit] = tmpT;
				objs[nHit] = pObj;
				*(tmpDeltas + 0 + nHit * 2) = dx;
				*(tmpDeltas + 1 + nHit * 2) = dy;
				
				hitPoints[nHit][0] = ray->direction_vec[0] * tmpT + ray->start_point[0];
				hitPoints[nHit][1] = ray->direction_vec[1] * tmpT + ray->start_point[1];
				hitPoints[nHit][2] = ray->direction_vec[2] * tmpT + ray->start_point[2];

				nHit ++;
			}
		}
	}

#ifdef __DEVICE_EMULATION__
	if( nHit > 2 )
	{
		printf("Hit number not right.\n");
	}
#endif

	//	copy
	if(nHit == 2)
	{
		bool bReverseNeeded = ts[0] > ts[1];

		pHits[0] = bReverseNeeded ? objs[1] : objs[0];
		pHits[1] = bReverseNeeded ? objs[0] : objs[1];
		
		*(deltas + 0) = bReverseNeeded ? tmpDeltas[2] : tmpDeltas[0];
		*(deltas + 1) = bReverseNeeded ? tmpDeltas[3] : tmpDeltas[1];
		*(deltas + 2) = bReverseNeeded ? tmpDeltas[0] : tmpDeltas[2];
		*(deltas + 3) = bReverseNeeded ? tmpDeltas[1] : tmpDeltas[3];

		if(bReverseNeeded)
		{
			for(int i = 0; i < 3; i ++)
			{
				float tmp = hitPoints[0][i];
				hitPoints[0][i] = hitPoints[1][i];
				hitPoints[1][i] = tmp;
			}
		}
	}
	else if(nHit == 1)
	{
		pHits[0] = objs[0];
		pHits[1] = objs[1];
		
		*(deltas + 0) = tmpDeltas[0];
		*(deltas + 1) = tmpDeltas[1];
	}

	return nHit;
}



////////////
