#include "tracer.h"
#include "ray.h"
#include "global.h"
#include "vector.h"
#include "consts.h"

#include "gpu/ray_gpu.cu"
#include "gpu/vector_gpu.cu" 
    
#include <cuda_runtime.h>  
#include <vector>
#include <assert.h> 
using namespace std; 
 
#include "gpu/ray_gpu.cu"
#include "gpu_util.h"
#include "gpu/geometry_gpu.cu"

BBox Tracer::_bbox;

////	Color Map Params

__device__
float gpuKnotValues[5] = {0};
__device__
int gpuKnotColors[5] = {0};

////
__device__ float tex_fStart = 0;
__device__ float tex_fEnd = 0;
__device__ int tex_Width = 0;
__device__ int tex_Height = 0;
__device__ float *tex_data = NULL;
__device__ int nDevPlaneSampleCount;

__device__ PrimGpuObj *pCap0Device_dev = NULL;
__device__ unsigned nCap0TriCount_dev = 0;

__device__ PrimGpuObj *pCap1Device_dev = NULL;
__device__ unsigned nCap1TriCount_dev = 0;

__device__ PrimGpuObj *pSliceDevice_dev = NULL;
__device__ unsigned nSliceTriCount_dev = 0;

__device__ PrimGpuObj *pPlaneDevice_dev = NULL;
__device__ unsigned nPlaneTriCount_dev = 0;

__global__ 
void _setTexInfo(float fStart, float fEnd, int texWidth, int texHeight, float *deviceTex, int nPlaneSampleCount, 
				 PrimGpuObj *pCap0, unsigned nCap0TriCount, PrimGpuObj *pCap1, unsigned nCap1TriCount,
				 PrimGpuObj *pSlice, unsigned nSliceTriCount,
				 float knotValue0, float knotValue1, float knotValue2, float knotValue3, float knotValue4, 
				 int knotColor0, int knotColor1, int knotColor2, int knotColor3, int knotColor4,
				 PrimGpuObj *pPlane, unsigned nPlaneTriCount)
{
	tex_fStart = fStart;
	tex_fEnd = fEnd;

	tex_Width = texWidth;
	tex_Height = texHeight;

	tex_data = deviceTex;

	nDevPlaneSampleCount = nPlaneSampleCount;

	pCap0Device_dev = pCap0;
	nCap0TriCount_dev = nCap0TriCount;

	pCap1Device_dev = pCap1;
	nCap1TriCount_dev = nCap1TriCount;

	pSliceDevice_dev = pSlice;
	nSliceTriCount_dev = nSliceTriCount;

	pPlaneDevice_dev = pPlane;
	nPlaneTriCount_dev = nPlaneTriCount;
	
	//
	gpuKnotValues[0] = knotValue0;
	gpuKnotValues[1] = knotValue1;
	gpuKnotValues[2] = knotValue2;
	gpuKnotValues[3] = knotValue3;
	gpuKnotValues[4] = knotValue4;

	gpuKnotColors[0] = knotColor0;
	gpuKnotColors[1] = knotColor1;
	gpuKnotColors[2] = knotColor2;
	gpuKnotColors[3] = knotColor3;
	gpuKnotColors[4] = knotColor4;
}

////
__device__
void clampColor_gpu(float *pColor)
{
	pColor[0] = pColor[0] > 1.f ? 1.f : (pColor[0] < 0.f ? 0.f : pColor[0]);
	pColor[1] = pColor[1] > 1.f ? 1.f : (pColor[1] < 0.f ? 0.f : pColor[1]);
	pColor[2] = pColor[2] > 1.f ? 1.f : (pColor[2] < 0.f ? 0.f : pColor[2]);
}

__device__
unsigned getThreadInx()
{
	return blockIdx.x * blockDim.x + threadIdx.x;
}

__device__
void genViewRay(Ray_gpu &ray, unsigned row, unsigned col, unsigned nWidth, unsigned nHeight,
						float fViewPlaneRatio,
						float eye_x, float eye_y, float eye_z, 
						float viewPlaneCtr_x, float viewPlaneCtr_y, float viewPlaneCtr_z, 
						float _rightVec_x, float _rightVec_y, float _rightVec_z, 
						float _upVec_x, float _upVec_y, float _upVec_z, float rdm[2], float fSamplingDeltaFactor)
{

	//	to fine the current primary ray starting point
	//
	vect3d_gpu nCurrCtr(viewPlaneCtr_x, viewPlaneCtr_y, viewPlaneCtr_z);
	
	//	right vec first
	vect3d_gpu rightVec(_rightVec_x, _rightVec_y, _rightVec_z);
	vecScale_gpu(rightVec, (col - nWidth/2.f) * fViewPlaneRatio / vecLen_gpu(&rightVec), rightVec);

	//	up vec second
	vect3d_gpu upVec(_upVec_x, _upVec_y, _upVec_z);
	vecScale_gpu(upVec, (row - nHeight/2.f) * fViewPlaneRatio/ vecLen_gpu(&upVec), upVec);
	
	point2point_gpu(nCurrCtr, rightVec, nCurrCtr);
	point2point_gpu(nCurrCtr, upVec, nCurrCtr);

	//	Randomize
	vect3d_gpu vDeltaXVec;
	vecScale_gpu(rightVec, fViewPlaneRatio * fSamplingDeltaFactor * rdm[0], vDeltaXVec);

	vect3d_gpu vDeltaYVec;
	vecScale_gpu(upVec, fViewPlaneRatio * fSamplingDeltaFactor * rdm[1], vDeltaYVec);

	point2point_gpu(nCurrCtr, vDeltaXVec, nCurrCtr);			
	point2point_gpu(nCurrCtr, vDeltaYVec, nCurrCtr);	
	
	//	
	vect3d_gpu eye(eye_x, eye_y, eye_z);
	vect3d_gpu viewDir;
	points2vec_gpu(eye, nCurrCtr, viewDir);
	normalize_gpu(viewDir);
	
	vecCopy_gpu(ray.start_point, nCurrCtr);
	vecCopy_gpu(ray.direction_vec, viewDir);
}

__device__
void getCellInx(PrimGpuObj *pObj, PrimGpuObj *gpuObjs, float *hitPoint, 
				unsigned max_x, unsigned max_y, unsigned max_z,
				int inx[3])
{
	inx[0] = hitPoint[0] + max_x * 1.f / 2.f;	inx[0] = inx[0] >= max_x ? (max_x - 1) : (inx[0] < 0.f ? 0.f : inx[0]);
	inx[1] = hitPoint[1] + max_y * 1.f / 2.f;	inx[1] = inx[1] >= max_y ? (max_y - 1) : (inx[1] < 0.f ? 0.f : inx[1]);
	inx[2] = hitPoint[2] + max_z * 1.f / 2.f;	inx[2] = inx[2] >= max_z ? (max_z - 1) : (inx[2] < 0.f ? 0.f : inx[2]);
}

__device__
float getElecCellValue(int x, int y, int z, float *elecData, int *idData, int mark[4], bool cutHalf = false)
{
	if( x < 0 || x >= VOL_X || 
		y < 0 || y >= VOL_Y ||
		z < 0 || z >= VOL_Z )	// Hard-code it for now
	{
		return 0;
	}
#ifndef DATA_2D
	if(cutHalf && y >= (VOL_Y/2))
#else
	if(cutHalf && z >= (VOL_Z/2))
#endif
	{
		return 0;
	}
	
	unsigned offset = x + y * VOL_X + z * VOL_X * VOL_Y;
	
	//	ID selection

	int currId = *(idData + offset);
	if( mark[currId - 1] == 0 )
	{
		return 0;
	}

	return *(elecData + offset);
}

__device__
float lerp(float v1, float ratio1, float v2)
{
#ifdef DATA_2D
	v1 = (v1 == INVALID_VOLVAL) ? 0 : v1;
	v2 = (v2 == INVALID_VOLVAL) ? 0 : v2;
#endif
	return v1 * (1.f - ratio1) + v2 * ratio1;
}

__device__
float getHermiteValue(float x, float fP0_val, float fP0_der, float fP1_val, float fP1_der)
{
	float x2 = x * x;
	float x3 = x * x * x;

	float p0 = 2 * x3 - 3 * x2 + 1;
	float p1 = -2 * x3 + 3 * x2;
	float p2 = x3 - 2 * x2 + x;
	float p3 = x3 - x2;

	return (fP0_val * p0 + fP1_val * p1 + fP0_der * p2 + fP1_der * p3);
}

/*
 *
 		0, "White"
		1, "Black"
		2, "Red"
		3, "Orange"
		4, "Yellow"
		5, "Green"
		6, "Cyan"
		7, "Blue"
		8, "Purple"
		9, "Gray"
 */
#define WHITE {1,1,1}
#define BLACK {0,0,0}
#define RED {1,0,0}
#define ORANGE {1,0.647,0}
#define YELLOW {1,1,0}
#define GREEN {0,1,0}
#define CYAN {0,1,1}
#define BLUE {0,0,1}
#define PURPLE {0.62745,0.12549,0.941176}
#define GRAY {0.8,0.8,0.8}

__device__
void custom_color_map(float value, float *pCurrPix, 
					  float knotValue[5], 
						int knotColor[5],
						 int mMode, float *deviceTexData, int texWidth, int texHeight, float fStart, float fEnd)
{
	if(mMode == 0)	//	Value-based
	{
		float _colors[10][3] = {
								WHITE,
								BLACK,
								RED,
								ORANGE,
								YELLOW,
								GREEN,
								CYAN,
								BLUE,
								PURPLE,
								GRAY
							  };
		//	Find starting inx & delta
		int inx0 = -1;
		float delta = 0;
		for(int i = 1; i < 5; i ++)
		{
			if( value < knotValue[i])
			{
				inx0 = i - 1;
				delta = (value - knotValue[i - 1]) / (knotValue[i] - knotValue[i - 1]);
				break;
			}
		}

		//	return color
		//
		if(inx0 >= 0 && inx0 <= 4)	// should fall into the range
		{
			*(pCurrPix + 0) = (1 - delta) * _colors[knotColor[inx0]][0] + delta * _colors[knotColor[inx0 + 1]][0];
			*(pCurrPix + 1) = (1 - delta) * _colors[knotColor[inx0]][1] + delta * _colors[knotColor[inx0 + 1]][1];
			*(pCurrPix + 2) = (1 - delta) * _colors[knotColor[inx0]][2] + delta * _colors[knotColor[inx0 + 1]][2];
		}
	}
	else	//	Picture based
	{
		//	Texture Image should be horizontal
		//
		if(value < fStart || value > fEnd)
		{
			*(pCurrPix + 0) = AMBI_X;
			*(pCurrPix + 1) = AMBI_Y;
			*(pCurrPix + 2) = AMBI_Z;
		}
		else
		{
			int offset = (value - fStart) / (fEnd - fStart) * texWidth;
			offset = offset > texWidth ? texWidth  : offset;

			//	Pick color from the mid
			float *pDataStart = deviceTexData + texHeight / 2 * 3 * texWidth;

			float *pColor = pDataStart + 3 * offset;
			*(pCurrPix + 0) = *(pColor + 0);
			*(pCurrPix + 1) = *(pColor + 1);
			*(pCurrPix + 2) = *(pColor + 2);
		}
	}
}

#ifdef DATA_2D

__device__
float getSampleValueByBilinear(float point[3], int maxx, int maxy, int maxz, float *elecData,
							   int *idData, int mark[4])
{
	int x_inx = fabs(point[0]);
	int y_inx = fabs(point[1]);
	int x_inx_p = x_inx < maxx ? (x_inx + 1): x_inx;
	int y_inx_p = y_inx < maxy ? (y_inx + 1): y_inx;

	float v0 = getElecCellValue(x_inx, y_inx, -1, elecData, idData, mark, false);
	float v1 = getElecCellValue(x_inx_p, y_inx, -1, elecData, idData, mark, false);;
	float v2 = getElecCellValue(x_inx, y_inx_p, -1, elecData, idData, mark, false);;
	float v3 = getElecCellValue(x_inx_p, y_inx_p, -1, elecData, idData, mark, false);;

	return lerp(lerp(v0, 0.5, v1), 0.5, lerp(v2, 0.5, v3));
}

#endif

///	Ray-marching
///
__device__ 
float getSampleValueByTrilinear(	Ray_gpu &ray, float point[3], int maxx, int maxy, int maxz, 
										float *elecData, int *idData, int mark[4], int *layer = NULL, bool bShowPlane = false )
{
	//	Distance from Ray to Cell Center
	int inx[3] = {	(int)(point[0] + (VOL_X/2)),
					(int)(point[1] + (VOL_Y/2)),
					(int)(point[2] + (VOL_Z/2)) };

#ifdef DATA_2D
	if( getElecCellValue(inx[0], inx[1], inx[2], elecData, idData, mark, bShowPlane) == INVALID_VOLVAL)
	{
		return INVALID_VOLVAL;
	}
#endif

	if( inx[0] >= 0 && inx[0] < VOL_X &&
		inx[1] >= 0 && inx[1] < VOL_Y &&
		inx[2] >= 0 && inx[2] < VOL_Z) 
	{

		vect3d_gpu ctr( inx[0] - (VOL_X/2) + 0.5, 
						inx[1] - (VOL_Y/2) + 0.5, 
						inx[2] - (VOL_Z/2) + 0.5 );

#if 1
		//	Some pPoint is out of the current cell (what the hell?!)
		//	so check this pPoint then
		//
		float delta[3] = {  (point[0] - ctr.data[0]), 
							(point[1] - ctr.data[1]),
							(point[2] - ctr.data[2]) };
		for(int i = 0; i < 3; i ++)
		{
			delta[i] = delta[i] > 1.f ? 1.f : delta[i];
			delta[i] = delta[i] <-1.f ?-1.f : delta[i];
		}
			
		//	Tri-linear
		int sign[3] = { delta[0] > 0 ? 1 : -1, 
						delta[1] > 0 ? 1 : -1, 
						delta[2] > 0 ? 1 : -1 };

		//	4 x
		float x_ny_nz = lerp(getElecCellValue(inx[0],           inx[1],           inx[2], elecData, idData, mark, bShowPlane) ,  delta[0] * sign[0], 
							 getElecCellValue(inx[0] + sign[0], inx[1],           inx[2], elecData, idData, mark, bShowPlane));
		float x_fy_nz = lerp(getElecCellValue(inx[0],           inx[1] + sign[1], inx[2], elecData, idData, mark, bShowPlane) ,  delta[0] * sign[0], 
							 getElecCellValue(inx[0] + sign[0], inx[1] + sign[1], inx[2], elecData, idData, mark, bShowPlane));

		float x_ny_fz = lerp(getElecCellValue(inx[0],           inx[1],           inx[2] + sign[2], elecData, idData, mark, bShowPlane) ,  delta[0] * sign[0], 
							 getElecCellValue(inx[0] + sign[0], inx[1],           inx[2] + sign[2], elecData, idData, mark, bShowPlane));
		float x_fy_fz = lerp(getElecCellValue(inx[0],           inx[1] + sign[1], inx[2] + sign[2], elecData, idData, mark, bShowPlane) ,  delta[0] * sign[0], 
							 getElecCellValue(inx[0] + sign[0], inx[1] + sign[1], inx[2] + sign[2], elecData, idData, mark, bShowPlane));

		float y_nz = lerp(x_ny_nz, delta[1] * sign[1], x_fy_nz);
		float y_fz = lerp(x_ny_fz, delta[1] * sign[1], x_fy_fz);

		if(layer)
		{
			unsigned offset = inx[0] + inx[1] * VOL_X + inx[2] * VOL_X * VOL_Y;
			*layer = *(idData + offset);
		}

		return lerp(y_nz, delta[2] * sign[2], y_fz);

#else
		if(layer)
		{
			unsigned offset = inx[0] + inx[1] * VOL_X + inx[2] * VOL_X * VOL_Y;
			*layer = *(idData + offset);
		}
		return getElecCellValue(inx[0],           inx[1],           inx[2], elecData, idData, mark, bShowPlane);
#endif
	}

	return 0;
}

__device__
float ray_marching(	Ray_gpu &ray,
					float *ret, int *nCount, float start_point[3], float end_point[3], int max[3], float *elecData, 
					int tf_mode, float fP0_val, float fP0_der, float fP1_val, float fP1_der, int bShowGeo, int &bInGeo,
					bool bClipPlaneEnabled, float planeCtr0, float planeCtr1, float planeCtr2, float planeNorm0, float planeNorm1, float planeNorm2,
					int *idData, int id0, int id1, int id2, int id3, int bShowSlice, bool bShowPlane = false)
{
	float fStep = 1;	// TODO: to be passed in

	//	Get total marching step len
	//
	vect3d_gpu vTotalVec;
	points2vec_gpu(start_point, end_point, vTotalVec);
	float fTotalLen = vecLen_gpu(&vTotalVec);

	//	Get Stepping Vector
	//
	vect3d_gpu point(start_point[0], start_point[1], start_point[2]);
	normalize_gpu(ray.direction_vec);
	vect3d_gpu stepVec, negStepVec;
	vecScale_gpu(ray.direction_vec, fStep, stepVec);
	vecScale_gpu(stepVec, -1, negStepVec);

	//	Marching!
	//
	int count = 0; // total marching count
	float fTotalWeight = 0;
	float fTotalVal = 0;
	float fCurrMarchingLen = 0;


	int mark[4] = {id0, id1, id2, id3};
	while(fCurrMarchingLen <= fTotalLen)
	{
		point2point_gpu(point, stepVec, point);
		int layer = -1;
		float vol_val = getSampleValueByTrilinear( ray, point.data, VOL_X, VOL_Y, VOL_Z, elecData, idData, mark, (!bClipPlaneEnabled)?&layer:NULL, bShowPlane );

#ifdef DATA_2D	
		if(vol_val == INVALID_VOLVAL)
		{
			continue;
		}
#endif
		
		if(!bClipPlaneEnabled)
		{
			float val = vol_val;

			switch(tf_mode)
			{
			case 0:	// Average
				fTotalVal += val;
				fTotalWeight += 1;
				break;

			case 2:	//	Hermite mode
				float fHmtFactor = getHermiteValue(fCurrMarchingLen/fTotalLen, fP0_val, fP0_der, fP1_val, fP1_der);
				fTotalWeight += fHmtFactor;
				fTotalVal += val * fHmtFactor;
				break;
			};

		}
		else	//!bClipPlaneEnabled
		{
			float currPlaneCtr[3] = {planeCtr0, planeCtr1, planeCtr2};
			
			vect3d_gpu planeVec(planeNorm0, planeNorm1, planeNorm2);
			normalize_gpu(planeVec);

			count = 1;
			fTotalWeight = 1;

			if(point2plane_gpu(point.data, currPlaneCtr, planeVec) < 1)
			{
				fTotalVal = vol_val;
				break;
			}
		}// if(!bClipPlaneEnabled)

		fCurrMarchingLen += fStep;
		count ++;
	}//	while

	// Eliminate the weird color on the volume edges
	*ret = (!bClipPlaneEnabled && count < 10) ? 0 : fTotalVal / (bClipPlaneEnabled ? 1 : fTotalWeight);

	*nCount = count;
	return fTotalWeight;
}

#include "nanorod.cu"

__global__
void _computePixels_GPU(float *pDeviceFilm, PrimGpuObj *gpuObjs, unsigned nHeight, unsigned nWidth, 
						float xmin, float xmax,
						float ymin, float ymax,
						float zmin, float zmax,
						float fViewPlaneRatio,
						float eye_x, float eye_y, float eye_z, 
						float viewPlaneCtr_x, float viewPlaneCtr_y, float viewPlaneCtr_z, 
						float _rightVec_x, float _rightVec_y, float _rightVec_z, 
						float _upVec_x, float _upVec_y, float _upVec_z,
						unsigned max_x, unsigned max_y, unsigned max_z,
						float *elecData, int tf_mode, float fP0_val, float fP0_der, float fP1_val, float fP1_der,
						int nMultiSampleCount, float fSamplingDeltaFactor, float *rdmData, unsigned rdmCount, int bShowGeo,
						bool bClipPlaneEnabled, float planeCtr0, float planeCtr1, float planeCtr2, float planeNorm0, float planeNorm1, float planeNorm2,						
						 PrimGpuObj *pNanoDevice, unsigned nTriCount, float fNanoAlpha, 
						 int *idData, int id0, int id1, int id2, int id3, int bOnlyInRod,
						 int mMode, int bShowSlice, int bShowPlane, float fPlaneAlpha)
{

	unsigned tid = getThreadInx();

	if(tid < nHeight * nWidth)
	{
		float *pCurrPix = pDeviceFilm + tid * 3;
		
		Ray_gpu primeRay;
		float primeRdm[2] = {0, 0};

		genViewRay(primeRay, tid / nWidth, tid % nWidth, nWidth, nHeight,
						fViewPlaneRatio, 
						eye_x, eye_y, eye_z, 
						viewPlaneCtr_x, viewPlaneCtr_y, viewPlaneCtr_z, 
						_rightVec_x, _rightVec_y, _rightVec_z, 
						_upVec_x, _upVec_y, _upVec_z, primeRdm, fSamplingDeltaFactor);

		bool bHitSlice = false;
		float fSliceValue = 0;
		bool bHitPlane = false;
		float fPlaneValue = 0;
		float toolDepth = 1.f; //%%%for blending the tool
		float toolColor[3]={0.75390625,0.2109375,0.23828125};
		//	BBox for Volume
		if( isHitOnPlane(primeRay, xmin, xmax, X_AXIS) &&
			isHitOnPlane(primeRay, ymin, ymax, Y_AXIS) && 
			isHitOnPlane(primeRay, zmin, zmax, Z_AXIS) ) 
		{
	
			//	Voxels
			//
			float delta[4] = {0}; // 1st -> 2nd, x -> y
			PrimGpuObj *pObjs[2] = {0};
			float hitPoints[4][3] = {0};//%%%hitPoints can be 4 now with the little cube

			float voxColor[3] = {0};
			float sliceColor[3] = {0};
			bool toolHit;
			unsigned nHit = isHit_gpu(gpuObjs, &primeRay, pObjs, delta, hitPoints, &toolHit);
			if(nHit > 0)  
			{ 
				int inx0[3] = {0};				
				getCellInx(pObjs[0], gpuObjs, hitPoints[0], 
								max_x, max_y, max_z,
								inx0);
				
				float fCount = 0;
				int bInGeo = 0;

				int max[3] = {VOL_X, VOL_Y, VOL_Z};
				int mark[4] = {id0, id1, id2, id3};

				float value = 0;

				if(toolHit)//%%%then the little cube got hit
				{
					//get depth of center of tool
					float depth;
					depth = VOL_Y/2 - gpuObjs[6]._vCenter.data[1];
					toolDepth = depth;

					//normalize
					toolDepth /= (float)(VOL_Y);
				}
				if(nHit == 2)
				{
					//Only necessary when using dda_ray_casting
					int inx1[3] = {0};	
					getCellInx(pObjs[1], gpuObjs, hitPoints[1], 
								max_x, max_y, max_z,
								inx1);

#if 0
					fCount = dda_ray_casting(	primeRay, &value, inx0, inx1, maxes, elecData, tf_mode, fP0_val, fP0_der, fP1_val, fP1_der, bShowGeo, bInGeo,
												bClipPlaneEnabled, planeCtr0, planeCtr1, planeCtr2, planeNorm0, planeNorm1, planeNorm2,
												idData, id0, id1, id2, id3 );
#else
					int marchCount = 0;
					if(!bClipPlaneEnabled)
					{
						
						//	Show slice
						//
						if(bShowSlice)
						{

							for(int i = 0; i < nMultiSampleCount; i ++)	/// TODO: Multi-sample for only Nanorod
							{
								Ray_gpu ray;
								unsigned rdmInx = tid * 2 * nMultiSampleCount % rdmCount;
								float rdm[2] = {rdmData[rdmInx + i * 2], rdmData[rdmInx + i * 2 + 1]};

								genViewRay(ray, tid / nWidth, tid % nWidth, nWidth, nHeight,
												fViewPlaneRatio, 
												eye_x, eye_y, eye_z, 
												viewPlaneCtr_x, viewPlaneCtr_y, viewPlaneCtr_z, 
												_rightVec_x, _rightVec_y, _rightVec_z, 
												_upVec_x, _upVec_y, _upVec_z, rdm, fSamplingDeltaFactor);
								
								float sliceT = 0;
								vect3d_gpu sliceNorm;
								PrimGpuObj *pHitTri = NULL;
								if( pHitTri = isRodHit_gpu( pSliceDevice_dev, nSliceTriCount_dev, &ray, &sliceT, sliceNorm ))
								{
									bHitSlice = true;

									vect3d_gpu hitPoint;
									vect3d_gpu marchVec;
									vecScale_gpu(ray.direction_vec, sliceT, marchVec);
									point2point_gpu(ray.start_point, marchVec, hitPoint);

								fSliceValue += getSampleValueByTrilinear( ray, hitPoint.data, VOL_X, VOL_Y, VOL_Z, elecData, idData, mark, NULL, bShowPlane != 0 );
								}		
							}//	for

							fSliceValue /= nMultiSampleCount;
							custom_color_map(fSliceValue, sliceColor, gpuKnotValues, gpuKnotColors, mMode, tex_data, tex_Width, tex_Height, tex_fStart, tex_fEnd);

						}//	if(bShowSlice)
						
						//	Show Plane
						//
						if(bShowPlane)
						{

							fPlaneValue  = 0;
							for(int i = 0; i < nMultiSampleCount; i ++)	/// TODO: Multi-sample for only Nanorod
							{
								Ray_gpu ray;
								unsigned rdmInx = tid * 2 * nMultiSampleCount % rdmCount;
								float rdm[2] = {rdmData[rdmInx + i * 2], rdmData[rdmInx + i * 2 + 1]};

								genViewRay(ray, tid / nWidth, tid % nWidth, nWidth, nHeight,
												fViewPlaneRatio, 
												eye_x, eye_y, eye_z, 
												viewPlaneCtr_x, viewPlaneCtr_y, viewPlaneCtr_z, 
												_rightVec_x, _rightVec_y, _rightVec_z, 
												_upVec_x, _upVec_y, _upVec_z, rdm, fSamplingDeltaFactor);
								
								float planeT = 0;
								vect3d_gpu planeNorm;
								PrimGpuObj *pHitTri = NULL;
								if( pHitTri = isRodHit_gpu( pPlaneDevice_dev, nPlaneTriCount_dev, &ray, &planeT, planeNorm ))
								{
									bHitPlane = true;

									vect3d_gpu hitPoint;
									vect3d_gpu marchVec;
									vecScale_gpu(ray.direction_vec, planeT, marchVec);
									point2point_gpu(ray.start_point, marchVec, hitPoint);
//#ifndef DATA_2D
									fPlaneValue += getSampleValueByTrilinear( ray, hitPoint.data, VOL_X, VOL_Y, VOL_Z, elecData, idData, mark, NULL, true );
//#else
//									fPlaneValue += getSampleValueByBilinear(hitPoint.data, DIM_X, DIM_Y, VOL_Z, elecData, idData, mark);
//#endif
								}	
							}//	for

							fPlaneValue = fPlaneValue / nMultiSampleCount;

						}//	if(bShowPlane)
						//If nothing is selected, and the planes are hit, just do this
						fCount = ray_marching(	primeRay, &value, 
												&marchCount, hitPoints[0], hitPoints[1], max, elecData, 
												tf_mode, fP0_val, fP0_der, fP1_val, fP1_der, bShowGeo, bInGeo,
												bClipPlaneEnabled, planeCtr0, planeCtr1, planeCtr2, planeNorm0, planeNorm1, planeNorm2,
												idData, id0, id1, id2, id3, bShowSlice, bShowPlane);
					}
					else	//if(!bClipPlaneEnabled)
					{
						float totalVal = 0;

						for(int i = 0; i < nMultiSampleCount; i ++)	/// TODO: Multi-sample for only Nanorod
						{
							Ray_gpu ray;
							unsigned rdmInx = tid * 2 * nMultiSampleCount % rdmCount;
							float rdm[2] = {rdmData[rdmInx + i * 2], rdmData[rdmInx + i * 2 + 1]};

							genViewRay(ray, tid / nWidth, tid % nWidth, nWidth, nHeight,
											fViewPlaneRatio, 
											eye_x, eye_y, eye_z, 
											viewPlaneCtr_x, viewPlaneCtr_y, viewPlaneCtr_z, 
											_rightVec_x, _rightVec_y, _rightVec_z, 
											_upVec_x, _upVec_y, _upVec_z, rdm, fSamplingDeltaFactor);

							float tmpVal = 0;
							fCount = ray_marching(	ray, &tmpVal, 
												&marchCount, hitPoints[0], hitPoints[1], max, elecData, 
												tf_mode, fP0_val, fP0_der, fP1_val, fP1_der, bShowGeo, bInGeo,
												bClipPlaneEnabled, planeCtr0, planeCtr1, planeCtr2, planeNorm0, planeNorm1, planeNorm2,
												idData, id0, id1, id2, id3, bShowSlice );

							totalVal += tmpVal;
						}//for

						value = totalVal / nMultiSampleCount;
					}//if(!bClipPlaneEnabled)
#endif
				}// if(nHit == 2)

				//	Customed Color-map		
				//

				custom_color_map(	value, voxColor, gpuKnotValues, gpuKnotColors, mMode, tex_data, tex_Width, tex_Height, tex_fStart, tex_fEnd);
			}//	hit or not
			else
			{
				voxColor[0] = AMBI_X;
				voxColor[1] = AMBI_Y;
				voxColor[2] = AMBI_Z;
			}
			
			//	Nanorod
			//
			vect3d_gpu nanoColor;	
			bool bHitNano = false;
			if(bHitSlice) bHitNano = true;
			if(bShowGeo && !bHitSlice)
			{
				for(int i = 0; i < nMultiSampleCount; i ++)	/// TODO: Multi-sample for only Nanorod
				{
					Ray_gpu ray;
					unsigned rdmInx = tid * 2 * nMultiSampleCount % rdmCount;
					float rdm[2] = {rdmData[rdmInx + i * 2], rdmData[rdmInx + i * 2 + 1]};

					genViewRay(ray, tid / nWidth, tid % nWidth, nWidth, nHeight,
									fViewPlaneRatio, 
									eye_x, eye_y, eye_z, 
									viewPlaneCtr_x, viewPlaneCtr_y, viewPlaneCtr_z, 
									_rightVec_x, _rightVec_y, _rightVec_z, 
									_upVec_x, _upVec_y, _upVec_z, rdm, fSamplingDeltaFactor);
					
					float nanoT = 0;
					vect3d_gpu norm;
					PrimGpuObj *pHitTri = NULL;
					if( pHitTri = isRodHit_gpu( pNanoDevice, nTriCount, &ray, &nanoT, norm ))
					{
						vect3d_gpu lightPos(0, 200, 0);

						//	Cap 0
						//
						bool bHitCap0 = false;
						vect3d_gpu cap0Color;
						float cap0T = 0;
						vect3d_gpu normCap0;
						PrimGpuObj *pHitCap0 = isRodHit_gpu( pCap0Device_dev, nCap0TriCount_dev, &ray, &cap0T, normCap0 );
						if(pHitCap0)
						{
							vect3d_gpu tmpColor;

							vect3d_gpu marchVec;
							vecScale_gpu(ray.direction_vec, cap0T, marchVec);
							vect3d_gpu hitPoint;
							point2point_gpu(ray.start_point, marchVec, hitPoint);

							if(!bShowPlane ||  bShowPlane && hitPoint[1] < 0)
							{
								evalPhong(ray.start_point, hitPoint, normCap0, pHitCap0, lightPos, tmpColor);
								point2point_gpu(cap0Color, tmpColor, cap0Color);

								bHitCap0 = true;
							}
						}

						//	Cap 1
						//
						bool bHitCap1 = false;
						vect3d_gpu cap1Color;
						float cap1T = 0;
						vect3d_gpu normCap1;
						PrimGpuObj *pHitCap1 = isRodHit_gpu( pCap1Device_dev, nCap1TriCount_dev, &ray, &cap1T, normCap1 );
						if(pHitCap1)
						{
							vect3d_gpu tmpColor;

							vect3d_gpu marchVec;
							vecScale_gpu(ray.direction_vec, cap1T, marchVec);
							vect3d_gpu hitPoint;
							point2point_gpu(ray.start_point, marchVec, hitPoint);

							if(!bShowPlane ||  bShowPlane && hitPoint[1] < 0)
							{
								evalPhong(ray.start_point, hitPoint, normCap1, pHitCap1, lightPos, tmpColor);
								point2point_gpu(cap1Color, tmpColor, cap1Color);

								bHitCap1 = true;
							}
						}
						
						//	Rod itself
						//
						bool bHitRod = false;
						vect3d_gpu tmpColor;

						vect3d_gpu marchVec;
						vecScale_gpu(ray.direction_vec, nanoT, marchVec);
						vect3d_gpu hitPoint;
						point2point_gpu(ray.start_point, marchVec, hitPoint);

						if(!bShowPlane ||  bShowPlane && hitPoint[1] < 0)
						{
							evalPhong(ray.start_point, hitPoint, norm, pHitTri, lightPos, tmpColor);
							point2point_gpu(nanoColor, tmpColor, nanoColor);

							bHitRod = true;
						}

						//	Blend Rod\Cap0\Cap1
						if(tf_mode == 1)
						{
							vect3d_gpu lightGreen(212.f/255.f, 231.f/255.f, 178.f/255.f);
							vect3d_gpu   midGreen(167.f/255.f, 214.f/255.f, 148.f/255.f);
							vect3d_gpu darkGreen(153.f/255.f, 183.f/255.f,  74.f/255.f);
							//vect3d_gpu darkGreen(0.2, 0.2, 0.2);

							if(bHitCap1) vecScale_gpu(darkGreen, vecLen_gpu(&cap1Color), cap1Color);
							if(bHitCap0) vecScale_gpu(midGreen,  vecLen_gpu(&cap0Color), cap0Color);
							vecScale_gpu(lightGreen, vecLen_gpu(&nanoColor), nanoColor);
						}

						if(tf_mode == 1)
						{
							//float alpha0 = 0.8;
							float alphaNano = 0.25;
							float alphaCap0 = 0.6;
							float alphaCap1 = 0.7;

							////	blend
							//
							vecScale_gpu(cap0Color, alphaCap0, cap0Color);
							vecScale_gpu(cap1Color, alphaCap1, cap1Color);
							vecScale_gpu(nanoColor, alphaNano, nanoColor);

							point2point_gpu(nanoColor, cap0Color, nanoColor);
							point2point_gpu(nanoColor, cap1Color, nanoColor);

							for(int i = 0; i < 3; i ++)
							{
								nanoColor.data[i] = nanoColor.data[i] > 1 ? 1 : nanoColor.data[i];
								nanoColor.data[i] = nanoColor.data[i] < 0 ? 0 : nanoColor.data[i];
							}
							bHitNano = true;
						}
						else if( tf_mode != -1 && (bHitRod || bHitPlane) )
						{
							vecScale_gpu(cap1Color, fNanoAlpha, cap1Color);
							vecScale_gpu(cap0Color, 1 - fNanoAlpha, cap0Color);
							point2point_gpu(cap0Color, cap1Color, cap0Color);

							vecScale_gpu(nanoColor, fNanoAlpha, nanoColor);
							vecScale_gpu(cap0Color, 1 - fNanoAlpha, cap0Color);
							point2point_gpu(nanoColor, cap0Color, nanoColor);

							for(int i = 0; i < 3; i ++)
							{
								nanoColor.data[i] = nanoColor.data[i] > 1 ? 1 : nanoColor.data[i];
								nanoColor.data[i] = nanoColor.data[i] < 0 ? 0 : nanoColor.data[i];
							}
							bHitNano = true;
						}
						
					}			
					else
					{
						if(bOnlyInRod)
						{
							nanoColor.data[0] += AMBI_X;
							nanoColor.data[1] += AMBI_Y;
							nanoColor.data[2] += AMBI_Z;
						}
						else
						{
							nanoColor.data[0] += voxColor[0];
							nanoColor.data[1] += voxColor[1];
							nanoColor.data[2] += voxColor[2];
						}
					}
				}//	for

				vecScale_gpu(nanoColor, 1.f / nMultiSampleCount, nanoColor);
			}//	if(bShowgeo)

			if(bHitNano)
			{
				if(tf_mode == 1)
				{
					*(pCurrPix + 0) = nanoColor.data[0];
					*(pCurrPix + 1) = nanoColor.data[1];
					*(pCurrPix + 2) = nanoColor.data[2];

				}
				else
				{
					if(bHitSlice)
					{
						float tmp1 = 0.5;
						*(pCurrPix + 0) = voxColor[0] * (1 - tmp1) + sliceColor[0] * tmp1;
						*(pCurrPix + 1) = voxColor[1] * (1 - tmp1) + sliceColor[1] * tmp1;
						*(pCurrPix + 2) = voxColor[2] * (1 - tmp1) + sliceColor[2] * tmp1;
					}
					else
					{
						*(pCurrPix + 0) = voxColor[0] * (1.0 - fNanoAlpha) + nanoColor.data[0] * fNanoAlpha;
						*(pCurrPix + 1) = voxColor[1] * (1.0 - fNanoAlpha) + nanoColor.data[1] * fNanoAlpha;
						*(pCurrPix + 2) = voxColor[2] * (1.0 - fNanoAlpha) + nanoColor.data[2] * fNanoAlpha;
					}

					//	blend plane
					if(bShowPlane && bHitPlane)
					{
						float planeColor[3] = {0};
#ifndef DATA_2D
						fPlaneValue *= 3;
#endif
						custom_color_map( fPlaneValue, planeColor, gpuKnotValues, gpuKnotColors, mMode, tex_data, tex_Width, tex_Height, tex_fStart, tex_fEnd);
						//
						float t = 0.5;
						*(pCurrPix + 0) = planeColor[0] * t + voxColor[0] * (1.0 - t);
						*(pCurrPix + 1) = planeColor[1] * t + voxColor[1] * (1.0 - t);
						*(pCurrPix + 2) = planeColor[2] * t + voxColor[2] * (1.0 - t);
					}
					else if(bShowPlane && !bHitPlane)
					{
						*(pCurrPix + 0) = voxColor[0] * (fNanoAlpha)/* + nanoColor.data[0] * (1 - fNanoAlpha)*/;
						*(pCurrPix + 1) = voxColor[1] * (fNanoAlpha)/* + nanoColor.data[1] * (1 - fNanoAlpha)*/;
						*(pCurrPix + 2) = voxColor[2] * (fNanoAlpha)/* + nanoColor.data[2] * (1 - fNanoAlpha)*/;
					}
				}
				//%%%blend with tool
				*(pCurrPix + 0) = toolColor[0] * (1.0 - toolDepth) + *(pCurrPix + 0) * toolDepth;
				*(pCurrPix + 1) = toolColor[1] * (1.0 - toolDepth) + *(pCurrPix + 1) * toolDepth;
				*(pCurrPix + 2) = toolColor[2] * (1.0 - toolDepth) + *(pCurrPix + 2) * toolDepth;
			}// hitNano
			else
			{
				if(!bOnlyInRod)
				{
					*(pCurrPix + 0) = 1.f * 0.5f + (0.5f)*voxColor[0];
					*(pCurrPix + 1) = 1.f * 0.5f + (0.5f)*voxColor[1];
					*(pCurrPix + 2) = 1.f * 0.5f + (0.5f)*voxColor[2];

					//	blend plane
					if(bShowPlane && bHitPlane)
					{
						float planeColor[3] = {0};
						
						custom_color_map( fPlaneValue, planeColor, gpuKnotValues, gpuKnotColors, mMode, tex_data, tex_Width, tex_Height, tex_fStart, tex_fEnd);
						//
#ifndef DATA_2D
						*(pCurrPix + 0) = planeColor[0]/* * fNanoAlpha + voxColor[0] * (1.0 - fNanoAlpha)*/;
						*(pCurrPix + 1) = planeColor[1]/* * fNanoAlpha + voxColor[1] * (1.0 - fNanoAlpha)*/;
						*(pCurrPix + 2) = planeColor[2]/* * fNanoAlpha + voxColor[2] * (1.0 - fNanoAlpha)*/;
#else
						*(pCurrPix + 0) = planeColor[0] * fNanoAlpha + voxColor[0] * (1.0 - fNanoAlpha);
						*(pCurrPix + 1) = planeColor[1] * fNanoAlpha + voxColor[1] * (1.0 - fNanoAlpha);
						*(pCurrPix + 2) = planeColor[2] * fNanoAlpha + voxColor[2] * (1.0 - fNanoAlpha);
#endif
					}
				}
				else
				{
					*(pCurrPix + 0) = AMBI_X;
					*(pCurrPix + 1) = AMBI_Y;
					*(pCurrPix + 2) = AMBI_Z;
				}
				//%%%blend with tool
				*(pCurrPix + 0) = toolColor[0] * (1.0 - toolDepth) + *(pCurrPix + 0) * toolDepth;
				*(pCurrPix + 1) = toolColor[1] * (1.0 - toolDepth) + *(pCurrPix + 1) * toolDepth;
				*(pCurrPix + 2) = toolColor[2] * (1.0 - toolDepth) + *(pCurrPix + 2) * toolDepth;
			}
		}// if bbox
		else
		{
			//	Not hit
			*(pCurrPix + 0) = AMBI_X;
			*(pCurrPix + 1) = AMBI_Y;
			*(pCurrPix + 2) = AMBI_Z;
			//%%%blend with tool
			*(pCurrPix + 0) = toolColor[0] * (1.0 - toolDepth) + *(pCurrPix + 0) * toolDepth;
			*(pCurrPix + 1) = toolColor[1] * (1.0 - toolDepth) + *(pCurrPix + 1) * toolDepth;
			*(pCurrPix + 2) = toolColor[2] * (1.0 - toolDepth) + *(pCurrPix + 2) * toolDepth;
		}
	}// if tid
}

////
void Tracer::setVolBBox(	float xmin, float xmax,
							float ymin, float ymax, 
							float zmin, float zmax)
{
	_bbox.setDim( xmin, xmax,
				  ymin, ymax,
				  zmin, zmax);
}

void Tracer::computePixels_GPU(float *pDeviceFilm, unsigned nHeight, unsigned nWidth,
								   float fViewPlaneRatio,
								   float eye_x, float eye_y, float eye_z, 
									float viewPlaneCtr_x, float viewPlaneCtr_y, float viewPlaneCtr_z, 
									float _rightVec_x, float _rightVec_y, float _rightVec_z, 
									float _upVec_x, float _upVec_y, float _upVec_z,
									unsigned max_x, unsigned max_y, unsigned max_z,
									float *elecData, int tf_mode, float fP0_val, float fP0_der, float fP1_val, float fP1_der,
									int nMultiSampleCount, float fSamplingDeltaFactor, float *rdmData, unsigned rdmCount, int bShowGeo,
						bool bClipPlaneEnabled, float planeCtr0, float planeCtr1, float planeCtr2, float planeNorm0, float planeNorm1, float planeNorm2,
						float knotValue0, float knotValue1, float knotValue2, float knotValue3, float knotValue4, 
						int knotColor0, int knotColor1, int knotColor2, int knotColor3, int knotColor4,
						 PrimGpuObj *pNanoDevice, unsigned nTriCount, float fNanoAlpha, int *idData, int mark[4], int bOnlyInRod,
						 int mMode, float *deviceTexData, int texWidth, int texHeight, float fStart, float fEnd)
{

	_setTexInfo<<<1, 1>>>(fStart, fEnd, texWidth, texHeight, deviceTexData, nPlaneSampleCount,
							pCap0Device, nCap0TriCount, pCap1Device, nCap1TriCount,
							pSliceDevice, nSliceTriCount, 
							knotValue0, knotValue1, knotValue2, knotValue3, knotValue4, 
							knotColor0, knotColor1, knotColor2, knotColor3, knotColor4,
							pNanoPlaneDevice, nPlaneTriCount);
	cudaThreadSynchronize();

	unsigned nTotalPixel = nHeight * nWidth;	//	As below, yes, the resolution has to be 256x
	_computePixels_GPU<<<nTotalPixel / GpuBlockSize, GpuBlockSize>>>(pDeviceFilm, gpuObjs, nHeight, nWidth, 
											_bbox._xmin, _bbox._xmax, 
											_bbox._ymin, _bbox._ymax,
											_bbox._zmin, _bbox._zmax,
											fViewPlaneRatio,
											eye_x, eye_y, eye_z, 
											viewPlaneCtr_x, viewPlaneCtr_y, viewPlaneCtr_z, 
											_rightVec_x, _rightVec_y, _rightVec_z, 
											_upVec_x, _upVec_y, _upVec_z,
											max_x, max_y, max_z,
											elecData, tf_mode, fP0_val, fP0_der, fP1_val, fP1_der, 
											nMultiSampleCount, fSamplingDeltaFactor, rdmData, rdmCount, bShowGeo,
											bClipPlaneEnabled, planeCtr0, planeCtr1, planeCtr2, planeNorm0, planeNorm1, planeNorm2,
											pNanoDevice, nTriCount, fNanoAlpha, idData, mark[0], mark[1], mark[2], mark[3], bOnlyInRod,
											mMode, bShowSlice, bShowPlane, fPlaneAlpha); 
	cudaThreadSynchronize();

	//if(bClipPlaneEnabled)
	//{
	//	bilinear_convolution<<<nTotalPixel / GpuBlockSize, GpuBlockSize>>>(pDeviceFilm, nHeight, nWidth, nPlaneSampleCount);
	//	cudaThreadSynchronize();
	//}
}
