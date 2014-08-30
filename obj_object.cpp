#include "obj_object.h"

//	WARNING: this macro is originally defined in glm.c
//			 when you are using GLMtriangle by yourself,
//			 you have to define it by yourself. This macro
//			 controls the structure of the GLMtriangle...
#define MATERIAL_BY_FACE
#include "glm.h"

#include "consts.h"

#include <stdio.h>

#include <assert.h>
#include <math.h>
#include <time.h>

///
///		ObjObject implementations
///

ObjObject::ObjObject(float fReflR, float fRefrR, float fRefrK, float fEmitR)
	: Object(fReflR, fRefrR, fRefrK, fEmitR)
	, _bSmooth(false)
	, _bHasVNorm(false)
{
	///
	///	Since there's no ObjObject in GPU, and I want to make primaryObj id the same as the 
	///	GPU primaryObj array index, I do this. This should not impact the current functional 
	///	code.
	///
	nCurrObj --;

#ifdef USE_KD_TREE_OBJ
	_pKdNode = new kd_node;
#endif	
}

ObjObject::~ObjObject()
{
#ifdef USE_KD_TREE_OBJ
	if(_pKdNode)
	{
		delete _pKdNode;
		_pKdNode = new kd_node;
	}
#endif

	if(_triangles)
	{
		for(int i = 0; i < _nTriNum; i ++)
		{
			delete _triangles[i];
		}
		delete [] _triangles;
	}
}

void ObjObject::setSmooth(bool bSmooth)
{
	_bSmooth = bSmooth;
	for(int i = 0; i < _nTriNum; i ++)
	{
		_triangles[i]->setSmooth(bSmooth);
	}
}

bool ObjObject::load(char *pObjPath)
{
	assert(pObjPath);

	GLMmodel *pModel = glmReadOBJ(pObjPath);
	if (pModel)
	{	
		unsigned numTri = pModel->numtriangles;
		if(numTri > 0)
		{
			_nTriNum = numTri;
			_triangles = new Triangle*[numTri];

			unsigned nCurrCount = 0;

			GLMgroup *group = pModel->groups;
			while (group) 
			{
				for(unsigned i = 0; i < group->numtriangles; i++) 
				{			
					GLMtriangle *triangle = & pModel->triangles[group->triangles[i]];

					GLfloat points[3][3] = {0};
					float vnorm[3][3] = {0};
					float norm[3] = {0};

					for(int j = 0; j< 3; j++)
					{         
						assert(triangle->vindices[j]>=1 && triangle->vindices[j]<= pModel->numvertices);

						//	Vertex											
						points[j][0] = pModel->vertices[3 * triangle->vindices[j]    ];
						points[j][1] = pModel->vertices[3 * triangle->vindices[j] + 1];
						points[j][2] = pModel->vertices[3 * triangle->vindices[j] + 2];
						
									
						//	Vertex Normal
						if( triangle->nindices[j] != -1)
						{
							vnorm[j][0] = pModel->normals[3 * triangle->nindices[j]];
							vnorm[j][1] = pModel->normals[3 * triangle->nindices[j] + 1];
							vnorm[j][2] = pModel->normals[3 * triangle->nindices[j] + 2];	
							_bHasVNorm = true;
						}
						else
						{
							_bHasVNorm = false;
						}
					}

					//	Facet Normal
					norm[0] = pModel->facetnorms[triangle->findex * 3];
					norm[1] = pModel->facetnorms[triangle->findex * 3 + 1];
					norm[2] = pModel->facetnorms[triangle->findex * 3 + 2];

					//	Material
					_triangles[nCurrCount] = new Triangle(points, norm, _fReflectionRatio, _fRefractionRatio, _fRefractionK);
					if(_bHasVNorm)
					{
						_triangles[nCurrCount]->setVNorm(vnorm);
					}

					if(triangle->material /*&& triangle->material != group->material*/) 
					{
						GLuint material = triangle->material;
						GLMmaterial *materialp = &pModel->materials[material];
						
						vect3d spec, diff, ambi;
						vecCopy(spec, materialp->specular);
						vecCopy(diff, materialp->diffuse);
						vecCopy(ambi, materialp->ambient);
						this->setMaterial(spec, diff, ambi, materialp->shininess);

						//_triangles[nCurrCount]->setMaterial(materialp->specular, materialp->diffuse, materialp->ambient, materialp->shininess);
						
					}
					else
					{
						//printf("Warning: This object doesn't have internal material\n");
					}

					nCurrCount ++;
				}//	group

				group = group->next;
			}//	while

			assert(nCurrCount == pModel->numtriangles);
		}// if

		glmDelete(pModel);

		updateBBox();

		return true;
	}

	printf("GML obj file load failure!\n");
	return false;
}

void ObjObject::updateBBox()
{
	float xmin = 999999.f, xmax = -999999.f;
	float ymin = 999999.f, ymax = -999999.f;
	float zmin = 999999.f, zmax = -999999.f;

	for(int i = 0; i < _nTriNum; i ++)
	{
		for(int j = 0; j < 3; j ++)
		{
			float x = _triangles[i]->_vertices[j][0];
			float y = _triangles[i]->_vertices[j][1];
			float z = _triangles[i]->_vertices[j][2];

			if( x < xmin) xmin = x;
			if( x > xmax) xmax = x;
			if( y < ymin) ymin = y;
			if( y > ymax) ymax = y;
			if( z < zmin) zmin = z;
			if( z > zmax) zmax = z;
		}
		_triangles[i]->updateBBox();
	}
	//printf("{%.2f, %.2f}-{%.2f, %.2f}-{%.2f, %.2f}\n", xmin, xmax, ymin, ymax, zmin, zmax);
	_bbox.setDim(xmin, xmax, ymin, ymax, zmin, zmax);
}

#ifdef USE_KD_TREE_OBJ
void ObjObject::buildKdTree()
{
	for(int i = 0; i < _nTriNum; i ++)
	{
		_pKdNode->addObject(_triangles[i]);
	}
	_pKdNode->updateBBox();
	printf(" total [%d] triangles \n", _nTriNum);

	if(::nObjDepth != 0)
	{
		clock_t nStart = clock();

		kd_node::nObjDepth = (::nObjDepth == -1)?(8 + 1.3 * log(_nTriNum * 1.f)): ::nObjDepth;
		printf("[Building KD-tree for Obj [%d]...\n", ::nObjDepth);

		_pKdNode->split(false);	

		printf(" Done %.2f \n", (clock() - nStart) / 1000.f);
	}	
}
#endif

void ObjObject::scale(float x, float y, float z)
{
	for(int i = 0; i < _nTriNum; i ++)
	{
		for(int j = 0; j < 3; j ++)
		{
			_triangles[i]->_vertices[j][0] *= x;
			_triangles[i]->_vertices[j][1] *= y;
			_triangles[i]->_vertices[j][2] *= z;
		}
	}
	updateBBox();
}

void ObjObject::translate(float x, float y, float z)
{
	for(int i = 0; i < _nTriNum; i ++)
	{
		for(int j = 0; j < 3; j ++)
		{
			_triangles[i]->_vertices[j][0] += x;
			_triangles[i]->_vertices[j][1] += y;
			_triangles[i]->_vertices[j][2] += z;
		}
	}

	updateBBox();//	TODO: maybe don't need to iterate 
}

void ObjObject::rotate(float angle, vect3d& axis)
{
	angle *= - PIon180;

	set_matrix(sinf(angle), cosf(angle), axis);

	for(int i = 0; i < _nTriNum; i ++)
	{
		vect3d tmp;
		for(int j = 0; j < 3; j ++)
		{
			mat_rot(_triangles[i]->_vertices[j], tmp);	vecCopy(_triangles[i]->_vertices[j], tmp);
			//	set a bool for existance of vnormal
			mat_rot(_triangles[i]->_vnormal[j], tmp);	vecCopy(_triangles[i]->_vnormal[j], tmp);
		}
		mat_rot(_triangles[i]->_normal, tmp);	vecCopy(_triangles[i]->_normal, tmp);
	}

	updateBBox();
}

bool ObjObject::isHit(Ray &ray, vect3d &pNormal, float *pt, vect3d *pTexColor)
{
#ifndef USE_KD_TREE_OBJ
	if(!_bbox.isHit(ray))
	{
		return false;	
	}

	vect3d norm;
	float t = 999999.f;

	float u = 0, v = 0;
	unsigned nHitTriInx = 0;

	for(int i = 0; i < _nTriNum; i ++)
	{
		float currT = 999999.f;
		vect3d currNorm;
		vect3d currTexColor;

		bool bHit = _triangles[i]->isHit(ray, currNorm,	&currT);
		if(bHit && (currT < t))
		{
			t = currT;
			vecCopy(norm, currNorm);
		}
	}

	if(t < 999999.f && t > epsi)
	{
		*pt = t;
		vecCopy(pNormal, norm);
		
		return true;
	}

	return false;

#else

	assert(_pKdNode);

	vect3d norm;
	float tmpT = 0.f;

	Object *pObj = _pKdNode->isHit(ray, norm, &tmpT, pTexColor);	
	if(pObj)
	{
		*pt = tmpT;
		vecCopy(pNormal, norm);
		
		return true;
	}
	return false;

#endif
}