#ifndef RAY_H
#define RAY_H

#include "vector.h"
#include <vector>

class Object;

typedef long long IdType;

struct Ray
{
public:
	//	For CUDA..
	Ray();
	void reset();

	Ray(vect3d &pStart, vect3d &pDir, bool pbIsInObj = false);

	void copy(Ray &);

	IdType id;

	vect3d start_point;
	vect3d direction_vec;
	vect3d color;

	//	to make integrator easier
	float fDeltaX, fDeltaY;	// within a PixelIntegrator

	//	For putting VPL on GPU only
	vect3d _hitPoint;
	vect3d _hitNorm;
};

void clampColor(vect3d &pColor);

#endif