#ifndef CCAMERA_VOL_H
#define CCAMERA_VOL_H

#define _MSVC
#include "chai3d/src/chai3d.h"

//Our own camera class
class cCameraVol : public cCamera
{

public:
	cCameraVol(cWorld* iParent) : cCamera(iParent)
	{
	
	}
  
	void renderView(const int a_windowWidth, const int a_windowHeight, const int a_imageIndex = CHAI_MONO);

};

#endif