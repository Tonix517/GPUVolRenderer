#ifndef SCENE_H
#define SCENE_H

#include "film.h"
#include "camera.h"
#include "object.h"

#include <vector>

#define USE_KD_TREE

class Scene
{
public:

	Scene();
	~Scene();

	void init();
	void clear();

	//
	//	Camera Gettor/Settor
	//
	//%%%
	void setCamera(Camera *pCamera);
	Camera * getCamera()
	{
		return _pCamera;
	}
	vect3d * getPreviousCameraCenter()
	{
		return &_previousCameraCenter;
	}
	vect3d* computeCameraCenter(Camera *pCamera);
	void setPreviousCameraCenter(Camera *pCamera);
	//%%%
	//
	//	Ambient Light
	//
	void setAmbiColor(vect3d &pColor);
	void getAmbiColor(vect3d &pColor);

	void addObject(Object *pObj)
	{
		_vObjects.push_back(pObj);
	}

	unsigned getObjectNum() { return _vObjects.size(); }
	Object *getObject(unsigned index) { return _vObjects[index]; }

	void setDataDim(unsigned max_x, unsigned max_y, unsigned max_z)
	{
		_max_x = max_x;
		_max_y = max_y;
		_max_z = max_z;
	}

	void setElecData(float *elecData, int *idData)
	{
		_elecData = elecData;
		_idData = idData;
	}

	//
	//	It is paralleled with render()
	//
	void compute();
	
	//
	//	Render the scene
	//
	void render(const char *imgPath = NULL);	

	void setTFMode(int mode)
	{
		_tf_mode = mode;
	}

private:						

	std::vector<Object *>	_vObjects;

	Film	*_pFilm;
	Camera	*_pCamera;
	//%%%
	vect3d	_previousCameraCenter;
	//%%%
	unsigned _nRdmNum;
	float	*_hostRdmData;
	float	*_deviceRdmData;

	vect3d _ambientColor;

	//	Transfer Function mode
	int _tf_mode;

	//	max dims for data
	unsigned _max_x;
	unsigned _max_y;
	unsigned _max_z;

	float *_elecData;
	int *_idData;
};

#endif