#include "consts.h"
#include "scene.h"
#include "film.h"
#include "gpu/geometry_gpu.h"
#include "tracer.h"
#include "vector.h"
#include "nanorod.h"
#include "IL/ilut.h"

#include <cuda_gl_interop.h>

#include <algorithm>
#include <assert.h>
#include <time.h>
#include <math.h>


Scene::Scene()
	: _pCamera(NULL)
	, _elecData(NULL)
	, _tf_mode(0)
	, _nRdmNum(100000)
	, _hostRdmData(NULL)
	, _deviceRdmData(NULL)
{			
	_ambientColor[0] = 0;			
	_ambientColor[1] = 0;			
	_ambientColor[2] = 0;	

	_max_x = 0;
	_max_y = 0;
	_max_z = 0;
}

Scene::~Scene()
{
	clear();

	if(_pFilm)
	{
		_pFilm->destroy();
		delete _pFilm;
		_pFilm = NULL;
	}

	if(_hostRdmData)
	{
		delete [] _hostRdmData;
		_hostRdmData = NULL;
	}
	
	if(_deviceRdmData)
	{
		cudaFree(_deviceRdmData);
		_deviceRdmData = NULL;
	}
}

void Scene::init()
{
	_pFilm = new Film;
	_pFilm->init(WinWidth, WinHeight);

	//	Init Rdm data
	_hostRdmData = new float[_nRdmNum];
	if(!_hostRdmData)
	{
		printf("Host Rdm Data allocation failed...\n");
	}

	for(unsigned i = 0; i < _nRdmNum; i ++)
	{
		_hostRdmData[i] = rand() % 10000 / 10000.f;
	}

	cudaError_t err = cudaMalloc(&_deviceRdmData, sizeof(float) * _nRdmNum);
	if(err != cudaSuccess)
	{
		printf("Device Rdm Data allocation failed...\n");
	}

	err = cudaMemcpy(_deviceRdmData, _hostRdmData, sizeof(float) * _nRdmNum, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("Device Rdm Data copy failed...\n");
	}
	
	delete [] _hostRdmData;
	_hostRdmData = NULL;
}

void Scene::clear()
{

	_pFilm->clear();

	if(_pCamera)
	{
		delete _pCamera;
		_pCamera = NULL;
	}

	if( !_vObjects.empty())
	{
		std::vector<Object *>::iterator iterObj = _vObjects.begin();
		for(; iterObj != _vObjects.end(); iterObj ++)
		{
			delete *iterObj;
		}
		_vObjects.clear();
	}
}

void Scene::setCamera(Camera *pCamera)
{
	if(_pCamera)
	{
		delete _pCamera;
	}

	_pCamera = pCamera;
}

//%%%
void Scene::setPreviousCameraCenter(Camera *pCamera)
{
	PerpCamera *pPCam = dynamic_cast<PerpCamera *>(pCamera);
	vecCopy(_previousCameraCenter, pPCam->_eyePos);
}

vect3d* Scene::computeCameraCenter(Camera *pCamera)
{
	PerpCamera *pPCam = dynamic_cast<PerpCamera *>(pCamera);
	return &pPCam->_eyePos;
}
//%%%

void Scene::setAmbiColor(vect3d &pColor)
{
	vecCopy(_ambientColor, pColor);
}

void Scene::getAmbiColor(vect3d &pColor)
{
	vecCopy(pColor, _ambientColor);
}

void Scene::compute()
{
	PerpCamera *pCam = dynamic_cast<PerpCamera *>(_pCamera);

	float *pDeviceFilm = NULL;
	GLuint bufId = _pFilm->GetFrameBufId();
	cudaError_t err = cudaGLMapBufferObject((void **)&pDeviceFilm, bufId);
	if(err != cudaSuccess)
	{
		printf("cudaGLMapBufferObject Error : %s \n", cudaGetErrorString(err));
	}

	Tracer::computePixels_GPU(pDeviceFilm,
							WinHeight, WinWidth, 
							ViewPlaneRatio,
							pCam->_eyePos[0], pCam->_eyePos[1], pCam->_eyePos[2], 
							pCam->_ctrPos[0], pCam->_ctrPos[1], pCam->_ctrPos[2], 
							pCam->_rightVec[0], pCam->_rightVec[1], pCam->_rightVec[2], 
							pCam->_upVec[0], pCam->_upVec[1], pCam->_upVec[2],
							_max_x, _max_y, _max_z, 
							_elecData, _tf_mode, fP0_val, fP0_der, fP1_val, fP1_der,
							nMultiSampleCount, fSamplingDeltaFactor, _deviceRdmData, _nRdmNum, bShowGeo,
							bClipPlaneEnabled, planeCtr[0], planeCtr[1], planeCtr[2], planeNorm[0], planeNorm[1], planeNorm[2],
							knotValues[0], knotValues[1], knotValues[2], knotValues[3], knotValues[4], 
							knotColors[0], knotColors[1], knotColors[2], knotColors[3], knotColors[4],
							pNanoDevice, nTriCount, fNanoAlpha, _idData, mark, bOnlyInRod,
							mMode, deviceTexData, texWidth, texHeight, fStart, fEnd
						);

	cudaGLUnmapBufferObject(bufId);
}

void Scene::render(const char *imgPath)
{
	_pFilm->render();

	//	show image
	if(imgPath)
	{
		////	Start to load
		//ILuint nCurrTexImg = 0;
		//ilGenImages(1, &nCurrTexImg);
		//ilBindImage(nCurrTexImg);	

		////	Get Image Info
		//if(ilLoadImage(imgPath))
		//{
		//	ILint nWidth = ilGetInteger(IL_IMAGE_WIDTH);
		//	ILint nHeight = ilGetInteger(IL_IMAGE_HEIGHT);

		//	unsigned texSize = nWidth * nHeight * 3 * sizeof(float);
		//	float *data_buf = new float[ texSize ];
		//	ilCopyPixels( 0, 0, 0, nWidth, nHeight, 1, IL_RGB, IL_FLOAT, data_buf);		

		//	//
		//	delete [] data_buf;
		//	ilDeleteImages(1, &nCurrTexImg);
		//}
		//else
		//{
		//	ILenum ilErr = ilGetError();
		//	printf("Error in LoadImage: %d [%s]\n", ilErr, imgPath);
		//	ilDeleteImages(1, &nCurrTexImg);
		//}
	}
}
