#include "vector.h"

///
///	NOTE: I really don't want them to run on CPU...
///

///
///		Secondary functions
///

void reflectVec(vect3d &vOrigViewVec, vect3d &vNormal, vect3d &vReflectViewVec)
{	
	vect3d vReverseViewVec;	vecScale(vOrigViewVec, -1, vReverseViewVec);
	vect3d vDiagonalNormalVec;
	float fLen = dot_product(vReverseViewVec, vNormal) / vecLen(vNormal) * 2.0f;

	vect3d vNormalizedNormal;
	point2point(vNormalizedNormal, vNormal, vNormalizedNormal);
	normalize(vNormalizedNormal);
	vecScale(vNormalizedNormal, fLen, vDiagonalNormalVec);
	point2point(vDiagonalNormalVec, vOrigViewVec, vReflectViewVec);	
}

void refractVec(vect3d &vOrigViewVec, vect3d &vNormal, vect3d &vRefractedVec, float refraK)
{

	//	TODO: when view vec is very close to the plane. 
	//	      there'll be different behaviors of light

	//	Ref: http://en.wikipedia.org/wiki/Snell's_law
	//
	vect3d vOrigNormViewVec;
	point2point(vOrigNormViewVec, vOrigViewVec, vOrigNormViewVec);
	normalize(vOrigNormViewVec);
	vect3d vMinusL;
	vecScale(vOrigNormViewVec, -1, vMinusL);

	float cos1 = dot_product(vNormal, vMinusL);
	float cos2 = sqrt(1 - refraK * refraK * (1 - cos1 * cos1));
	assert((1 - refraK * refraK * (1 - cos1 * cos1)) >= 0 );

	//	(n1/n2)*l
	vect3d tmp;
	point2point(tmp, vOrigNormViewVec, tmp);
	vecScale(tmp, refraK, tmp);

	//	(n1/n2*cos1 +- cos2)
	vecCopy(vRefractedVec, vNormal);
	if(cos1 > 0)
	{
		vecScale(vRefractedVec, refraK * cos1 - cos2, vRefractedVec);
	}
	else
	{
		vecScale(vRefractedVec, refraK * cos1 + cos2, vRefractedVec);
	}

	//	combined..
	point2point(tmp, vRefractedVec, vRefractedVec);
}

void projectPoint(vect3d &pEyePos, vect3d &vViewVec, float t, vect3d &vTargetPoint)
{
	vect3d vStartPoint, tmp;
	point2point(vStartPoint, pEyePos, vStartPoint);
	point2point(tmp, vViewVec, tmp);	vecScale(tmp, t, tmp);
	point2point(vStartPoint, tmp, vTargetPoint);
}

float point2line(	vect3d & vPoint, vect3d &pEyePos, vect3d &pViewVec, float *pT)
{
	vect3d k;
	points2vec(pEyePos, vPoint, k);

	float t = dot_product(k, pViewVec) / dot_product(pViewVec, pViewVec);	
	*pT = t;

	vect3d vLen;
	vect3d tmp;
	vect3d tmp2;
	vecScale(pViewVec, t, tmp);
	vecScale(k, -1, tmp2);
	point2point(tmp2, tmp, vLen);

	return vecLen(vLen);	
}

///
///	REF: Real-Time Rendering 2nd, 3.1.7, 3.1.8
///

void mat_on_vec3d(vect3d &p, float mat[3][3], vect3d &ret)
{
	ret[0] = mat[0][0] * p[0] + mat[0][1] * p[1] + mat[0][2] * p[2];
	ret[1] = mat[1][0] * p[0] + mat[1][1] * p[1] + mat[1][2] * p[2];
	ret[2] = mat[2][0] * p[0] + mat[2][1] * p[1] + mat[2][2] * p[2];
}

float r_mat[3][3] = {0};

void set_matrix(float sin, float cos, vect3d &r)
{
	r_mat[0][0] = cos + (1.f - cos) * r[0] * r[0];
	r_mat[0][1] = (1.f - cos) * r[0] * r[1] - r[2] * sin;
	r_mat[0][2] = (1.f - cos) * r[0] * r[2] + r[1] * sin;

	r_mat[1][0] = (1.f - cos) * r[0] * r[1] + r[2] * sin;
	r_mat[1][1] = cos + (1.f - cos) * r[1] * r[1];
	r_mat[1][2] = (1.f - cos) * r[1] * r[2] - r[0] * sin;

	r_mat[2][0] = (1.f - cos) * r[0] * r[2] - r[1] * sin;
	r_mat[2][1] = (1.f - cos) * r[1] * r[2] + r[0] * sin;
	r_mat[2][2] = cos + (1.f - cos) * r[2] * r[2];	
}

//	Rotation matrix for geometries
void mat_rot(vect3d &p, vect3d &ret_p)
{
	mat_on_vec3d(p, r_mat, ret_p);
}

