#ifndef OBJ_OBJECT_H
#define OBJ_OBJECT_H

#include "object.h"
#include "vector.h"


//#define USE_KD_TREE_OBJ

//unsigned nObjDepth = 3;

///
///		class ObjObject
///	
///		NOTE: Triangle intersection logic is in this class
///
class ObjObject : public Object
{
public:
	ObjObject(float fReflR = 1.f, float fRefrR = 0.f, float fRefrK = 1.f, float fEmitR = 1.0);
	~ObjObject();

	ObjType getObjType()
	{	return OBJ_CPU; }
	
	bool load(char *pObjPath);

	void scale(float x, float y, float z);
	void translate(float x, float y, float z);
	void rotate(float angle, vect3d& axis);

	void setSmooth(bool bSmooth);

	bool isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor = NULL);

#ifdef USE_KD_TREE_OBJ	
	void buildKdTree();
#endif

//	For GPU use
	unsigned getTriCount()
	{	return _nTriNum;	}

	Triangle *getTriangle(unsigned inx)
	{	return _triangles[inx];	}
//

protected:
	
	void updateBBox();

public:
	bool	_bSmooth;	
	bool	_bHasVNorm;

protected:

	///
	///	NOTE: As you may know, GLM stores all data collectively and makes reference into the arrays
	///		  when rendering. However, this memory reference can slow down my rendering a lot due to
	///		  the overly memory access.
	///
	unsigned _nTriNum;
	Triangle **_triangles;

public:
	//	kd-tree
#ifdef USE_KD_TREE_OBJ
	kd_node *_pKdNode;
#endif
};

#endif