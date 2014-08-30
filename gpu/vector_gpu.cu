#ifndef VECTOR_GPU_H
#define VECTOR_GPU_H

#ifdef __DEVICE_EMULATION__
#include <assert.h>
#include <stdio.h>
#endif

//

struct vect3d_gpu
{
	
	float data[3];

	inline __device__
	vect3d_gpu()
	{
		data[0] = 0;
		data[1] = 0;
		data[2] = 0;
	}

	inline __device__
	vect3d_gpu(float x, float y, float z)
	{
		data[0] = x;
		data[1] = y;
		data[2] = z;
	}

	inline __device__
	float & operator [](int i)
	{
		return data[i];
	}
};

static inline __device__
void points2vec_gpu(vect3d_gpu &vStartPoint, vect3d_gpu &vEndPoint, vect3d_gpu &vVec)
{
	vVec.data[0] = vEndPoint.data[0] - vStartPoint.data[0];
	vVec.data[1] = vEndPoint.data[1] - vStartPoint.data[1];
	vVec.data[2] = vEndPoint.data[2] - vStartPoint.data[2];
}

static inline __device__
void points2vec_gpu(float vStartPoint[3], float vEndPoint[3], vect3d_gpu &vVec)
{
	vVec.data[0] = vEndPoint[0] - vStartPoint[0];
	vVec.data[1] = vEndPoint[1] - vStartPoint[1];
	vVec.data[2] = vEndPoint[2] - vStartPoint[2];
}

static inline __device__
float vecLen_gpu(vect3d_gpu *vVec)
{
	return sqrtf( vVec->data[0] * vVec->data[0] + 
				  vVec->data[1] * vVec->data[1] + 
				  vVec->data[2] * vVec->data[2] );
}
 
static inline __device__
void vecScale_gpu(vect3d_gpu &vOrigVec, float fScale, vect3d_gpu &vScaledVec)
{
	vScaledVec.data[0] = fScale * vOrigVec.data[0];
	vScaledVec.data[1] = fScale * vOrigVec.data[1];
	vScaledVec.data[2] = fScale * vOrigVec.data[2];
}

static inline __device__
void point2point_gpu(vect3d_gpu &vStartPoint, vect3d_gpu &vVec, vect3d_gpu &vEndPoint)
{
	vEndPoint.data[0] = vStartPoint.data[0] + vVec.data[0];
	vEndPoint.data[1] = vStartPoint.data[1] + vVec.data[1];
	vEndPoint.data[2] = vStartPoint.data[2] + vVec.data[2];
}

static inline __device__
void cross_product_gpu(vect3d_gpu &vec1, vect3d_gpu &vec2, vect3d_gpu &vecr)
{
	vecr.data[0] = vec1.data[1] * vec2.data[2] - vec1.data[2] * vec2.data[1];
	vecr.data[1] = vec1.data[2] * vec2.data[0] - vec1.data[0] * vec2.data[2];
	vecr.data[2] = vec1.data[0] * vec2.data[1] - vec1.data[1] * vec2.data[0];
}

static inline __device__
float dot_product_gpu(vect3d_gpu &vec1, vect3d_gpu &vec2)
{
	return	vec1.data[0] * vec2.data[0] + 
			vec1.data[1] * vec2.data[1] + 
			vec1.data[2] * vec2.data[2];
}

static inline __device__
void normalize_gpu(vect3d_gpu &vec)
{
	float fLen = vecLen_gpu(&vec);
	if(fLen == 0) return;

	float v = __powf(fLen, -1); //1/fLen;
	vec.data[0] *= v;
	vec.data[1] *= v;
	vec.data[2] *= v;
}

static inline __device__
void vecCopy_gpu(vect3d_gpu &destVec, float *srcVec)
{
	destVec.data[0] = srcVec[0];
	destVec.data[1] = srcVec[1];
	destVec.data[2] = srcVec[2];
}

static inline __device__
void vecCopy_gpu(vect3d_gpu &destVec, vect3d_gpu &srcVec)
{
	destVec.data[0] = srcVec.data[0];
	destVec.data[1] = srcVec.data[1];
	destVec.data[2] = srcVec.data[2];
}


///
///		Secondary functions
///
static inline __device__
void reflectVec_gpu(vect3d_gpu &vOrigViewVec, vect3d_gpu &vNormal, vect3d_gpu &vReflectViewVec)
{	
	vect3d_gpu vReverseViewVec;	/*vReverseViewVec.init();*/ vecScale_gpu(vOrigViewVec, -1, vReverseViewVec);
	vect3d_gpu vDiagonalNormalVec; /*vDiagonalNormalVec.init();*/
	float fLen = dot_product_gpu(vReverseViewVec, vNormal) / vecLen_gpu(&vNormal) * 2.0f;

	vect3d_gpu vNormalizedNormal;/* vNormalizedNormal.init();*/
	point2point_gpu(vNormalizedNormal, vNormal, vNormalizedNormal);
	normalize_gpu(vNormalizedNormal);
	vecScale_gpu(vNormalizedNormal, fLen, vDiagonalNormalVec);
	point2point_gpu(vDiagonalNormalVec, vOrigViewVec, vReflectViewVec);	
}

static inline __device__
void refractVec_gpu(vect3d_gpu &vOrigViewVec, vect3d_gpu &vNormal, vect3d_gpu &vRefractedVec, float refraK)
{

	//	TODO: when view vec is very close to the plane. 
	//	      there'll be different behaviors of light

	//	Ref: http://en.wikipedia.org/wiki/Snell's_law
	//
	vect3d_gpu vOrigNormViewVec; /*vOrigNormViewVec.init();*/
	point2point_gpu(vOrigNormViewVec, vOrigViewVec, vOrigNormViewVec);
	normalize_gpu(vOrigNormViewVec);
	vect3d_gpu vMinusL; /*vMinusL.init();*/
	vecScale_gpu(vOrigNormViewVec, -1, vMinusL);

	float cos1 = dot_product_gpu(vNormal, vMinusL);
	float cos2 = sqrtf(1 - refraK * refraK * (1 - cos1 * cos1));	// cuda sqrt

#ifdef __DEVICE_EMULATION__
	//	assert( (1 - refraK * refraK * (1 - cos1 * cos1)) >= 0 );	
	if( !((1 - refraK * refraK * (1 - cos1 * cos1)) >= 0 ) )
	{
		printf(".data[refractVec_gpu] : Man, sth. is wrong...\n");
		return;
	}
#endif

	//	(n1/n2)*l
	vect3d_gpu tmp; /*tmp.init();*/
	point2point_gpu(tmp, vOrigNormViewVec, tmp);
	vecScale_gpu(tmp, refraK, tmp);

	//	(n1/n2*cos1 +- cos2)
	vecCopy_gpu(vRefractedVec, vNormal);
	if(cos1 > 0)
	{
		vecScale_gpu(vRefractedVec, refraK * cos1 - cos2, vRefractedVec);
	}
	else
	{
		vecScale_gpu(vRefractedVec, refraK * cos1 + cos2, vRefractedVec);
	}

	//	combined..
	point2point_gpu(tmp, vRefractedVec, vRefractedVec);
}

static inline __device__
void projectPoint_gpu(vect3d_gpu &pEyePos, vect3d_gpu &vViewVec, float t, vect3d_gpu &vTargetPoint)
{
	vect3d_gpu vStartPoint, tmp; /*vStartPoint.init(); tmp.init();*/
	point2point_gpu(vStartPoint, pEyePos, vStartPoint);
	point2point_gpu(tmp, vViewVec, tmp);	vecScale_gpu(tmp, t, tmp);
	point2point_gpu(vStartPoint, tmp, vTargetPoint);
}

static inline __device__
float point2line_gpu(	vect3d_gpu & vPoint, vect3d_gpu &pEyePos, vect3d_gpu &pViewVec, float *pT)
{
	vect3d_gpu k; /*k.init();*/
	points2vec_gpu(pEyePos, vPoint, k);

	float t = dot_product_gpu(k, pViewVec) / dot_product_gpu(pViewVec, pViewVec);	
	*pT = t;

	vect3d_gpu vLen; /*vLen.init();*/
	vect3d_gpu tmp;/* tmp.init();*/
	vect3d_gpu tmp2; /*tmp2.init();*/
	vecScale_gpu(pViewVec, t, tmp);
	vecScale_gpu(k, -1, tmp2);
	point2point_gpu(tmp2, tmp, vLen);

	return vecLen_gpu(&vLen);	
}

static inline __device__
float point2plane_gpu(float *point, float *planeCtr, vect3d_gpu &normalizedPlaneNorm)
{
	vect3d_gpu toVec;
	points2vec_gpu(planeCtr, point, toVec);

	return fabs(dot_product_gpu(normalizedPlaneNorm, toVec));
}
#endif