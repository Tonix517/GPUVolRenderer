#ifndef OBJECT_H
#define OBJECT_H

#include "ray.h"
#include "bbox.h"
#include "vector.h"

///
///		struct material
///
struct material
{
	material()
	{
		for(int i = 0; i < 3; i ++)
		{
			specColor[i] = 0;
			diffColor[i] = 0;
			ambiColor[i] = 0;
		}

		fShininess = 1.0f;
	}

	vect3d specColor;
	vect3d diffColor;
	vect3d ambiColor;

	float fShininess;
};

//
enum ObjType {TRI_CPU, SQU_CPU, SPH_CPU, CUBE_CPU, OBJ_CPU, NONE_CPU};

///
///		class Object
///

class Object
{
public:

	Object(float fReflR = 1.f, float fRefrR = 0.f, float fRefrK = 1.f, float fEmitR = 1.0);

	virtual ~Object(){};

	virtual ObjType getObjType() = 0;

	//	Intersection
	virtual bool isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor = NULL) = 0;

	float getReflectionRatio() { return _fReflectionRatio; }
	float getRefractionRatio() { return _fRefractionRatio; }
	float getRefractionK()     { return _fRefractionK; }
	float getEmissionRatio()   { return _fEmitRatio; }

	//	Get info
	void setMaterial(vect3d &specColor, vect3d &diffColor, vect3d &ambiColor, float fShininess);
	material & getMaterial() { return _mat; }

	unsigned _id;
	//Ray *_pLastVisitingRay;
	IdType _nLastVisitingRay;

public:

	BBox * getBBox(){ return &_bbox; }

protected:

	virtual void updateBBox() = 0;

protected:

	BBox	_bbox;	

	//	Obj color itself occupies (1 - _fReflectionRatio - _fRefractionRatio)
	float _fReflectionRatio;	//	how much light contribution due to the reflection?
	
	float _fRefractionRatio;	//  how much light contribution due to the refraction?
	float _fRefractionK;

	float _fEmitRatio;

public:
	material _mat;	

};

///
///		class PrimaryObject
///
class PrimaryObject : public Object
{
public:
	PrimaryObject(float fReflR = 1.f, float fRefrR = 0.f, float fRefrK = 1.f, float fEmitR = 1.0)
		:Object(fReflR, fRefrR, fRefrK, fEmitR)
	{}
};


///
///		Primary - Sphere
///
bool isTriangleHit(	vect3d [3], Ray &ray, 
					float *pt, float *pu, float *pv);

class Triangle : public PrimaryObject
{
public:
	Triangle( float vVertices[3][3], float *facetNormal, 
				float fReflR = 1.f, float fRefrR = 0.f, float fRefrK = 1.f, float fEmitR = 1.0);
	
	ObjType getObjType()
	{	return TRI_CPU; }

	void setSmooth(bool bSmooth)
	{	_bSmooth = bSmooth; }

	bool isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor = NULL);

	void setVNorm(float vnorm[3][3]);

	void updateBBox();

//private:
	vect3d	_vertices[3];
	vect3d	_normal;

	vect3d 	_vnormal[3];

//protected:
	bool _bSmooth;
	bool _bHasVNorm;
	
};


///\	Square Object class
///
class Square : public PrimaryObject
{
public:
	//	WARNING: 1. the passed in vec. should be normalized !!
	//			 2. Dude, normal and WidthVec should be perpendicular to each other.
	//				you need to promise it.
	Square(vect3d &pCenter, vect3d &pNormal, vect3d &pHeadVec, float nWidth, float nHeight,
			float fReflR = 1.f, float fRefrR = 0.f, float fRefrK = 1.f, float fEmitR = 1.0);
	
	ObjType getObjType()
	{	return SQU_CPU; }

	bool isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor = NULL);

protected:
	
	void updateBBox();

//protected:
public:

	//	Directions
	vect3d _vNormal;
	vect3d _vWidthVec;

	//	Positions
	vect3d _vCenter;
	float _nWidth;
	float _nHeight;	

	//	For Calc.
	//
	vect3d _v2HeightVec;
	vect3d _v2WidthVec;
};

///\	Cube Object class
///
class Cube : public PrimaryObject
{
public:

	///	WARNING: all vectors should be normalized !!!
	Cube(	float fLen, float fWidth, float fHeight,
			vect3d &pCenterPos,
			vect3d &pVerticalVec,
			vect3d &pHorizonVec,				
			float fReflR = 1.f, float fRefrR = 0.f, float fRefrK = 1.f, float fEmitR = 1.0
		 );		

	~Cube();

	ObjType getObjType()
	{	return CUBE_CPU; }

	bool isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor = NULL);
	
	void setColor( 	vect3d &pAmbColor, 
					vect3d &pDiffColor,
					vect3d &pSpecColor, 
					float fShininess)
	{			
		
		for(int i = 0; i < 6; i++)
		{
			_vs[i]->setMaterial(pSpecColor, pDiffColor, pAmbColor, fShininess);
		}
		
		PrimaryObject::setMaterial(	pSpecColor, pDiffColor, pAmbColor, fShininess);
	}

protected:

	void updateBBox();

private:	

	float _fLength;		
	float _fWidth;
	float _fHeight;

	vect3d _vCenterPos;
	vect3d _verticalVec;
	vect3d _horizonVec;

public:
	Square*	_vs[6];

};

#endif