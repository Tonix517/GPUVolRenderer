#include <stdlib.h>
#include "camera.h"
#include "vector.h"
#include "consts.h"
#include <assert.h>

Camera::Camera(vect3d &pCtrPos, vect3d &pUpVec, vect3d &pViewVec, float fNtoF, float fPlaneRatio)
	: _fN2F(fNtoF), _fPlaneRatio(fPlaneRatio), _pSampler(NULL)
{
	assert((fNtoF > 0) && (fPlaneRatio > 0));	

	//	viewing direction
	_dir[0] = pViewVec[0];
	_dir[1] = pViewVec[1];
	_dir[2] = pViewVec[2];
	normalize(_dir);

	//	Center point of the view plane
	_ctrPos[0] = pCtrPos[0];
	_ctrPos[1] = pCtrPos[1];
	_ctrPos[2] = pCtrPos[2];

	//	Up vec of the view plane, with vec-len as 1/2 of view plane height
	_upVec[0] = pUpVec[0];
	_upVec[1] = pUpVec[1];
	_upVec[2] = pUpVec[2];
	normalize(_upVec);
	vecScale(_upVec, WinHeight * _fPlaneRatio * 0.5, _upVec);

	//	right vec of the view plane, with vec-len as 1/2 of view plane width
	cross_product(_dir, _upVec, _rightVec);
	normalize(_rightVec);
	vecScale(_rightVec, WinWidth * _fPlaneRatio * 0.5, _rightVec);

	//	default: no multi-sampling
	_nMultiSamplingCount = 1;
}

void Camera::setSampler(SamplingType eType)
{
	if(_pSampler)
	{
		delete _pSampler;
	}
	switch(eType)
	{
	case STRATIFIED:
		_pSampler = new StratifiedSampler;
		break;

	case LOW_DISC:
		_pSampler = new LowDiscrepancySampler;
		break;

	case BEST_CANDID:
		_pSampler = new BestCandidateSampler;
		break;
	}
}


///
///
///

OrthoCamera::OrthoCamera(vect3d &pCtrPos, vect3d &pUpVec, vect3d &pViewVec, float fNtoF, float fPlaneRatio)
	: Camera(pCtrPos, pUpVec, pViewVec, fNtoF, fPlaneRatio)
{ }
///
///

PerpCamera::PerpCamera(float fEye2Near, vect3d &pCtrPos, vect3d &pUpVec, vect3d &pViewVec, float fNtoF, float fPlaneRatio)
	: Camera(pCtrPos, pUpVec, pViewVec, fNtoF, fPlaneRatio)
{
	assert(fEye2Near > 0);
	
	//
	vect3d vInverseVec;
	vecScale(_dir, -fEye2Near, vInverseVec);
	point2point(_ctrPos, vInverseVec, _eyePos);
}
