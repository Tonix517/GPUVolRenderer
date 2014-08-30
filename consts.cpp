#include "consts.h"

//	512 is to make aligned memory for GPU. coalesce memory
unsigned WinWidth  = 512;
unsigned WinHeight = 512;
unsigned GpuBlockSize = 512;

unsigned WinLeft   = 200;
unsigned WinTop    = 50;

char * WinTitle = "GPU Volume Rendering Engine";

float ViewPlaneRatio = 0.01;

int nMultiSampleCount = 1;

float epsi = 0.01;

float fSamplingDeltaFactor = 0.01;

unsigned nCurrObj = 0;

int bShowGeo = 1;

//	Data Selection
int mark[4] = {1, 1, 1, 1};

int iTfMode = 0;	// average mode is the default
float fP0_val = 0;
float fP0_der = 4;
float fP1_val = 0;
float fP1_der = -4;

//	Plane
int bClipPlaneEnabled = 0;
float planeCtr[3] = {0, 0, 0};
float planeNorm[3]= {0, 1, 0};
int nPlaneSampleCount = 1;

//	Color-map
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
 */
//float knotValues[5] = {-0.03, -0.015, 0, 0.0135, 0.027};	// elec
float knotValues[5] = {-0.059, -0.0295, 0, 0.031, 0.062};	// elec
int knotColors[5] = {0, 6, 7, 2, 8};

int mMode = 1;
float *deviceTexData = 0;
int texWidth = 0;
int texHeight = 0;
float fStart = 0;
float fEnd = 1;

float fNanoAlpha = 0.3;
int bOnlyInRod = 0;

int bShowSlice = 1;
int bShowPlane = 0;
float fPlaneAlpha = 0.5;

//	Color-map Pic
float picWidth = 0.5;
float picHeight = 0.01;
float picX0 = 0.1;
float picY0 = 0.05;
