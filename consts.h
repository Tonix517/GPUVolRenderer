#ifndef CONSTS_H
#define CONSTS_H

//	Volume MACRO
#define INVALID_VOLVAL 0xFFFFFFFF 

////	2D Electric Data
//#define DATA_2D
//#define DIM_X 44
//#define DIM_Y 44
//#define VOL_X 96
//#define VOL_Y 44
//#define VOL_Z 96
//#define ID_PATH  ""
//#define FIELD_PATH "data/solid_000050.tsv"
//#define DATA_PATH "_data_2d.bin"
//#define CM_IMG "data/2d_cm.jpg"

//	Electric Data
#define VOL_X 100
#define VOL_Y 100
#define VOL_Z 156
#define ID_PATH  "VisData\\microID.txt"
#define FIELD_PATH "VisData\\ElectricField.txt"
#define DATA_PATH "_data_elec.bin"
#define CM_IMG "images/cm1.jpg"

////	Poly - spon
//#define VOL_X 100
//#define VOL_Y 100
//#define VOL_Z 156
//#define ID_PATH  "R55C38W12fine\\Data\\microID.txt"
//#define FIELD_PATH "R55C38W12fine\\Data\\SpontaneousPolarization.txt"
//#define DATA_PATH "_data_spon_pola.bin"
//#define Z_ONLY
//#define CM_IMG "images/cm1.jpg"

////	Poly - piezo
//#define VOL_X 100
//#define VOL_Y 100
//#define VOL_Z 156
//#define ID_PATH  "R55C38W12fine\\Data\\microID.txt"
//#define FIELD_PATH "R55C38W12fine\\Data\\PiezoelectricPolarization.txt"
//#define DATA_PATH "_data_piezo_pola.bin"
//#define Z_ONLY
//#define CM_IMG "images/cm1.jpg"

////	Poly - Dielect
//#define VOL_X 100
//#define VOL_Y 100
//#define VOL_Z 156
//#define ID_PATH  "R55C38W12fine\\Data\\microID.txt"
//#define FIELD_PATH "R55C38W12fine\\Data\\DielectricPolarization.txt"
//#define DATA_PATH "_data_dielec_pola.bin"
//#define Z_ONLY
//#define CM_IMG "images/redblue.jpg"
//
////	Poly - SQW - dielec
//#define VOL_X 93
//#define VOL_Y 93
//#define VOL_Z 157
//#define ID_PATH  "SQW\\Data\\microID.txt"
//#define FIELD_PATH "SQW\\Data\\DielectricPolarization.txt"
//#define DATA_PATH "_data_dielec_sqw.bin"
//#define CM_IMG "images\\redblue.jpg"

////	Poly - SQW - piezo
//#define VOL_X 93
//#define VOL_Y 93
//#define VOL_Z 157
//#define ID_PATH  "SQW\\Data\\microID.txt"
//#define FIELD_PATH "SQW\\Data\\PiezoelectricPolarization.txt"
//#define DATA_PATH "_data_piezo_sqw.bin"
//#define CM_IMG "images\\redblue.jpg"

////	Poly - SQW - Spon
//#define VOL_X 93
//#define VOL_Y 93
//#define VOL_Z 157
//#define ID_PATH  "SQW\\Data\\microID.txt"
//#define FIELD_PATH "SQW\\Data\\SpontaneousPolarization.txt"
//#define DATA_PATH "_data_spon_sqw.bin"
//#define CM_IMG "images\\redblue.jpg"
//#define Z_ONLY

////  Poly - SQW - Dielect
//#define VOL_X 93
//#define VOL_Y 93
//#define VOL_Z 157
//#define ID_PATH  "SQW\\Data\\microID.txt"
//#define FIELD_PATH "SQW\\Data\\DielectricPolarization.txt"
//#define DATA_PATH "_data_dielect_sqw.bin"
//#define CM_IMG "images\\redblue.jpg"

////	VR55C40W20
//#define VOL_X 147
//#define VOL_Y 147
//#define VOL_Z 120
//#define ID_PATH  "VR55C40W20\\microID.txt"
//#define FIELD_PATH "VR55C40W20\\ElectricField.txt"
//#define DATA_PATH "_data_W20.bin"
//#define CM_IMG "images\\heat.jpg"

////		VR55C40W6
//#define VOL_X 128
//#define VOL_Y 128
//#define VOL_Z 104
//#define ID_PATH  "VR55C40W6\\microID.txt"
//#define FIELD_PATH "VR55C40W6\\ElectricField.txt"
//#define DATA_PATH "_data_W6.bin"
//#define CM_IMG "images\\heat.jpg"

////		VR55C10W20
//#define VOL_X 107
//#define VOL_Y 107
//#define VOL_Z 87
//#define ID_PATH  "VR55C10W20\\microID.txt"
//#define FIELD_PATH "VR55C10W20\\ElectricField.txt"
//#define DATA_PATH "_data_W10.bin"
//#define CM_IMG "images\\heat.jpg"

//	Windows Params
extern unsigned WinWidth;
extern unsigned WinHeight;
extern unsigned GpuBlockSize;

extern unsigned WinLeft;
extern unsigned WinTop;

extern char * WinTitle;

extern float ViewPlaneRatio;

//	
extern float epsi;

extern int nMultiSampleCount;
extern float fSamplingDeltaFactor;
extern unsigned nCurrObj;

//
extern int bShowGeo;

//
extern int iTfMode;
extern float fP0_val;
extern float fP0_der;
extern float fP1_val;
extern float fP1_der;

//
#define PIon180 (0.017453292222)
#define PI (3.1415926)

//	Clip Plane
extern int bClipPlaneEnabled;
extern float planeCtr[3];
extern float planeNorm[3];
extern int nPlaneSampleCount;

//	Color Map
extern float knotValues[5];
extern int knotColors[5];
extern int mMode;
extern float *deviceTexData;
extern int texWidth;
extern int texHeight;
extern float fStart;
extern float fEnd;

//	Data Selection
extern int mark[4];

//	Nano Alpha
extern float fNanoAlpha;
extern int bOnlyInRod;

//	Slice\Cut-plane
extern int bShowSlice;
extern int bShowPlane;
extern float fPlaneAlpha;

//	Color-map Pic - all by percentage based on WinWidth & WinHeight
extern float picWidth;
extern float picHeight;
extern float picX0;
extern float picY0;

#endif