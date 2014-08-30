#ifndef NANOROD_H
#define NANOROD_H

#include "obj_object.h"
#include "gpu/geometry_gpu.h"

extern unsigned nTriCount;
extern PrimGpuObj *pNanoDevice;
extern PrimGpuObj_host *pNanoHost;

void copyNanoGeo(ObjObject *pNanoGeo, float offset);
void nanoGeoDestroy();

extern unsigned nPlaneTriCount;
extern PrimGpuObj *pNanoPlaneDevice;
extern PrimGpuObj_host *pNanoPlaneHost;

void copyNanoPlane(ObjObject *pNanoGeo, float offset);
void nanoPlaneDestroy();

///		Cap 0

extern unsigned nCap0TriCount;
extern PrimGpuObj *pCap0Device;
extern PrimGpuObj_host *pCap0Host;

void copyInternalCap0(ObjObject *pObj, float offset);
void internalCap0Destroy();

///		Cap 1

extern unsigned nCap1TriCount;
extern PrimGpuObj *pCap1Device;
extern PrimGpuObj_host *pCap1Host;

void copyInternalCap1(ObjObject *pObj, float offset);
void internalCap1Destroy();

///		Slice

extern unsigned nSliceTriCount;
extern PrimGpuObj *pSliceDevice;
extern PrimGpuObj_host *pSliceHost;

void copySlice(ObjObject *pObj);
void SliceDestroy();

#endif