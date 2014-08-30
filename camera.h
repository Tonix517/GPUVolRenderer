#ifndef CAMERA_H
#define CAMERA_H

#include "sampler.h"
#include "vector.h"

enum CamType {ORTHO, PERSP};

///
///		class Camera
///
class Camera
{
public:

	Camera(vect3d &pCtrPos, vect3d &pUpVec, vect3d &pViewVec, float fNtoF, float fPlaneRatio);
	virtual ~Camera()
	{
		if(_pSampler)
		{
			delete _pSampler;
			_pSampler = NULL;
		}
	}

	//	caller new
	void setSampler(SamplingType eType);

	void setMultiSamplingCount(unsigned nCount)
	{
		_nMultiSamplingCount = nCount;
	}
	unsigned getMultiSamplingCount(){ return _nMultiSamplingCount; }

//protected:

	float _fPlaneRatio;	// by length

	vect3d _ctrPos;	//	The center point of the near view plane
	vect3d _upVec;
	vect3d _rightVec;
	vect3d _dir;	// viewing vector (normalized)
	float _fN2F;	// the distance from near view plane to far view plane	

	Sampler *_pSampler;
	unsigned _nMultiSamplingCount;	// sampling count per pixel
};

//
//		class Orthogonal Camera
//
class OrthoCamera : public Camera
{
public:

	OrthoCamera(vect3d &pCtrPos, vect3d &pUpVec, vect3d &pViewVec, float fNtoF, float fPlaneRatio);

};

//
//		class Perspective Camera
//
class PerpCamera : public Camera
{
public:
	
	PerpCamera(float fEye2Near, vect3d &pCtrPos, vect3d &pUpVec, vect3d &pViewVec, float fNtoF, float fPlaneRatio);

//private:

	vect3d _eyePos;	// the distance from 'eye' to the near view plane
};

#endif
