#ifndef BBOX_H
#define BBOX_H

#include "ray.h"

enum AxisType {X_AXIS, Y_AXIS, Z_AXIS};

///
///		Bounding Box : axis-aligned
///	
///		NOTE: currently mainly for ObjObject
///
class BBox
{
public:
	BBox();
	
	void setDim(float xmin, float xmax,
				float ymin, float ymax,
				float zmin, float zmax);

	bool isHit(Ray &);

	void getBoundValues(AxisType, float *pmin, float *pmax);

	void genBoundingSphereParam(float &fRad, vect3d &ctr);

protected:

	bool isHitOnPlane(Ray &ray, AxisType);

public:
//protected:
	float _xmin, _xmax;
	float _ymin, _ymax;
	float _zmin, _zmax;
};

extern unsigned nBBoxHitCount;

#endif