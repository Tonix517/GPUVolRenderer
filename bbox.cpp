#include "bbox.h"

#include "vector.h"
#include "consts.h"

///
///		BBox implementations
///

BBox::BBox()
	: _xmin(0), _xmax(0)
	, _ymin(0), _ymax(0)
	, _zmin(0), _zmax(0)
{ }
	
void BBox::setDim(float xmin, float xmax,
				float ymin, float ymax,
				float zmin, float zmax)
{
	_xmin = xmin; _xmax = xmax;
	_ymin = ymin; _ymax = ymax;
	_zmin = zmin; _zmax = zmax;
}

unsigned nBBoxHitCount = 0;
bool BBox::isHit(Ray &ray)
{
	nBBoxHitCount ++;
	return ( isHitOnPlane(ray, X_AXIS) && 
			 isHitOnPlane(ray, Y_AXIS) && 
			 isHitOnPlane(ray, Z_AXIS) );
}

void BBox::getBoundValues(AxisType eType, float *pmin, float *pmax)
{
	switch(eType)
	{
	case X_AXIS:
		*pmin = _xmin;
		*pmax = _xmax;
		break;

	case Y_AXIS:
		*pmin = _ymin;
		*pmax = _ymax;
		break;

	case Z_AXIS:
		*pmin = _zmin;
		*pmax = _zmax;
		break;
	};
}

bool BBox::isHitOnPlane(Ray &ray, AxisType eType)
{
	float min = 0, max = 0;
	float start = 0, dir = 0;

	switch(eType)	
	{
	case X_AXIS:
		min = _xmin;
		max = _xmax;
		start = ray.start_point[0];
		dir = ray.direction_vec[0];
		break;

	case Y_AXIS:
		min = _ymin;
		max = _ymax;
		start = ray.start_point[1];
		dir = ray.direction_vec[1];
		break;

	case Z_AXIS:
		min = _zmin;
		max = _zmax;
		start = ray.start_point[2];
		dir = ray.direction_vec[2];
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

void BBox::genBoundingSphereParam(float &fRad, vect3d &ctr)
{
	ctr[0] = (_xmin + _xmax) / 2.f;
	ctr[1] = (_ymin + _ymax) / 2.f;
	ctr[2] = (_zmin + _zmax) / 2.f;

	vect3d point;
	point[0] = _xmax;
	point[1] = _ymax;
	point[2] = _zmax;

	vect3d len;
	points2vec(ctr, point, len);
	fRad = vecLen(len);
}