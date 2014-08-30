#ifndef TRACER_H
#define TRACER_H

#include "camera.h"
#include "bbox.h"
#include "gpu/ray_gpu.cu"
#include "nanorod.h"

class Tracer
{
public:

	static void setVolBBox(	float xmin, float xmax,
							float ymin, float ymax,
							float zmin, float zmax);

	static void computePixels_GPU(float *pDeviceFilm, unsigned nHeight, unsigned nWidth,
						float fViewPlaneRatio,
						float eye_x, float eye_y, float eye_z, 
						float viewPlaneCtr_x, float viewPlaneCtr_y, float viewPlaneCtr_z, 
						float _rightVec_x, float _rightVec_y, float _rightVec_z, 
						float _upVec_x, float _upVec_y, float _upVec_z,
						unsigned max_x, unsigned max_y, unsigned max_z,
						float *elecData, 
						int tf_mode, float fP0_val, float fP0_der, float fP1_val, float fP1_der,
						int nMultiSampleCount, float fSamplingDeltaFactor, float *rdmData, unsigned rdmCount, int bShowGeo,
						bool bClipPlaneEnabled, float planeCtr0, float planeCtr1, float planeCtr2, float planeNorm0, float planeNorm1, float planeNorm2, 
						float knotValue0, float knotValue1, float knotValue2, float knotValue3, float knotValue4, 
						int knotColor0, int knotColor1, int knotColor2, int knotColor3, int knotColor4,
						PrimGpuObj *pNanoDevice, unsigned nTriCount, float fNanoAlpha, int *idData, int mark[4], int bOnlyInRod,
						int mMode, float *deviceTexData, int texWidth, int texHeight, float fStart, float fEnd);
	///	I know the parameters are too many.. this is the consideration of efficiency

private:
	static BBox _bbox;
};

#endif