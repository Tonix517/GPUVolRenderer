// DataProcess.cpp : Defines the entry point for the console application.
//


#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>

#include "../consts.h"

bool processData(	char *pDestPath,unsigned x, unsigned y, unsigned z, 
					char *pMacroIdPath, char *pElecFldPath)
{

	unsigned nTotalCellNum = x * y * z;

	float *pElecData = new float[3 * nTotalCellNum];
	if(pElecData == NULL)
	{
		return false;
	}

	int *pIdData = new int[nTotalCellNum];
	if(pIdData == NULL)
	{
		return false;
	}

	///		Load ElecField 
	///
	FILE *fp0 = fopen(pElecFldPath, "rb");
	if(fp0)
	{
		printf("Loading Electronic Field Data file...\n");

		//	Skip the header first
		char buf[150] = {0};
		fscanf(fp0, "%s%s%s", buf, buf + 50, buf + 100);

		//	GO!
		unsigned nCount = 0;
		while(!feof(fp0))
		{
			char buf[150] = {0};
			size_t nRet = fscanf(fp0, "%s%s%s", buf, buf + 50, buf + 100);
			
			float x = atof(buf);
			float y = atof(buf + 50);
			float z = atof(buf + 100);
			
			*(pElecData + nCount * 3 + 0) = x;
			*(pElecData + nCount * 3 + 1) = y;
			*(pElecData + nCount * 3 + 2) = z;			

			if(nRet < 3)	break;

			nCount ++;
			if(nCount == nTotalCellNum)	 break;

			if(nCount % 10000 == 0)
			{
				printf("\b\b\b\b\b\b\b\b%.2f", nCount * 1.f / nTotalCellNum);
			}
		}//	while

		printf("\nTotal Cell : %d \n", nCount);
		fclose(fp0);
	}
	else
	{
		delete [] pIdData;
		delete [] pElecData;
		return false;
	}

	float *pCubeCells = new float[nTotalCellNum];
	assert(pCubeCells);

	///		Load MacroID 
	///
	FILE *fp = fopen(pMacroIdPath, "rb");
	if(fp)
	{
		printf("Loading MacroID file...\n");

		//	Skip the header first
		char buf[200] = {0};
		fscanf(fp, "%s%s%s%s", buf, buf + 50, buf + 100, buf + 150);

		float range[4][2] = {	 { 10, -10 },
								 { 10, -10 },
								 { 10, -10 },
								 { 10, -10 } };

		//	GO!
		unsigned nCount = 0;
		while(!feof(fp))
		{
			char buf[200] = {0};
			size_t nRet = fscanf(fp, "%s%s%s%s", buf, buf + 50, buf + 100, buf + 150);
			if(nRet < 4)
			{
				break;
			}

			float *pPlc = pElecData + nCount * 3;
#ifndef Z_ONLY
			float val = sqrt(	*(pPlc + 0) * *(pPlc + 0) +
								*(pPlc + 1) * *(pPlc + 1) +
								*(pPlc + 2) * *(pPlc + 2) );
#else
			float val = *(pPlc + 2);
#endif
			*(pCubeCells + nCount) = val;

			int inx = atoi(buf + 150);
			*(pIdData + nCount) = inx;
			
			range[inx-1][0] = val < range[inx-1][0] ? val : range[inx-1][0];
			range[inx-1][1] = val > range[inx-1][1] ? val : range[inx-1][1];

			nCount ++;
			if(nCount % 10000 == 0)
			{
				printf("\b\b\b\b\b\b\b\b%.2f", nCount * 1.f / nTotalCellNum);
			}

			if(nCount >= nTotalCellNum)	 break;
		}//	while

		float gRange[2] = {0xFFFFF, -0xFFFFF};

		printf("Sub-ranges:\n");
		for(int i = 0; i < 4; i ++)
		{
			printf("ID - [%d] : %.20f  ->  %.20f = [%.20f]\n", i, range[i][0], range[i][1],  range[i][1] - range[i][0]);
			gRange[0] = gRange[0] > range[i][0] ? range[i][0] : gRange[0];
			gRange[1] = gRange[1] < range[i][1] ? range[i][1] : gRange[1];			
		}

		printf("Global-range:\n");
		printf("GLOBAL: { %.20f, %.20f } = %.20f\n", gRange[0], gRange[1], gRange[1] - gRange[0]);

		printf("\nTotal Cell : %d \n", nCount);
		fclose(fp);

#if 0
		float gMax = -0xFFFFFF;
		float gMin =  0xFFFFFF;
		for(int i = 0; i < 4; i ++)
		{
			gMax = gMax <  range[i][1] ? range[i][1] : gMax;
			gMin = gMin >  range[i][0] ? range[i][0] : gMin;
		}
		printf("Normalizing data... \n");

		for(unsigned i = 0; i < nTotalCellNum; i ++)
		{
			*(pCubeCells + i) = (*(pCubeCells + i) - gMin) / (gMax - gMin);

			if(i % 10000 == 0)
			{
				printf("\b\b\b\b\b\b\b\b%.2f", i * 1.f / nTotalCellNum);
			}
		}

		printf(" Done. \n");
#endif

		//	Write To Bin File
		//
		FILE *fDp = fopen(pDestPath, "wb");
		if(fDp)
		{
			printf("Writing to file... \n");
			for(int i = 0; i < nTotalCellNum; i ++)
			{
				fprintf(fDp, "%f %d ", *(pCubeCells + i), *(pIdData + i));
				if(i % 10000 == 0)
				{
					printf("\b\b\b\b\b\b\b\b\b\b\b%.2f", i * 1.f / nTotalCellNum);
				}
			}
			fclose(fDp);
			printf("\nDone\n");
		}
		else
		{
			printf("Erro when creating dest file.\n");
			delete [] pCubeCells;
			delete [] pElecData;
			delete [] pIdData;
		
			return false;
		}

		delete [] pCubeCells;
		delete [] pElecData;
		delete [] pIdData;
		return true;
	}

	delete [] pIdData;
	delete [] pCubeCells;
	delete [] pElecData;
	return false;
}

#ifdef DATA_2D
bool processData2D(	char *pDestPath,unsigned x, unsigned y, unsigned z, 
					char *pElecFldPath)
{

	unsigned nTotalCellNum = DIM_X * DIM_Y;

	float *pElecData = new float[4 * nTotalCellNum];
		assert(pElecData);

	///		Load 2DField 
	///
	///		min-max
	float sc_x[2] = {0xFFFFFF, -0xFFFFFF};
	float  v_x[2] = {0xFFFFFF, -0xFFFFFF};

	FILE *fp0 = fopen(pElecFldPath, "rb");
	if(fp0)
	{
		printf("Loading 2D Field Data file...\n");

		//	Skip the header first
		char buf[200] = {0};
		fscanf(fp0, "%s%s%s%s%s", buf, buf + 50, buf + 100, buf + 100, buf + 150);

		//	GO!
		unsigned nCount = 0;
		while(!feof(fp0))
		{
			size_t nRet = fscanf(fp0, "%s%s%s%s", buf, buf + 50, buf + 100, buf + 150);
			
			float x = atof(buf);
			float y = atof(buf + 50);
			float sc = atof(buf + 100);
			float v = atof(buf + 100);

			sc_x[0] = sc < sc_x[0] ? sc : sc_x[0];
			sc_x[1] = sc > sc_x[1] ? sc : sc_x[1];
			v_x[0] = v < v_x[0] ? v : v_x[0];
			v_x[1] = v > v_x[1] ? v : v_x[1];
			
			*(pElecData + nCount * 4 + 0) = x;
			*(pElecData + nCount * 4 + 1) = y;
			*(pElecData + nCount * 4 + 2) = sc;			
			*(pElecData + nCount * 4 + 3) = v;			

			if(nRet < 4)	break;

			nCount ++;
			if(nCount == nTotalCellNum)	 break;

			if(nCount % VOL_Y == 0)
			{
				printf("\b\b\b\b\b\b\b\b%.2f", nCount * 1.f / nTotalCellNum);
			}
		}//	while

		printf("\nTotal Cell : %d \n", nCount);
		fclose(fp0);
	}
	else
	{
		delete [] pElecData;
		return false;
	}

	//
	printf(" == SC : {%.10f, %.10f}\n ==  V : {%.10f, %.10f}\n", sc_x[0], sc_x[1], v_x[0], v_x[1]);
	
	sc_x[0] = 0xFFFFFF;	sc_x[1] = -0xFFFFFF;
	v_x[0] = 0xFFFFFF;	v_x[1] = -0xFFFFFF;

	//	Write To Bin File
	//
	FILE *fDp = fopen(pDestPath, "wb");
	if(fDp)
	{
		printf("Writing to file... \n");
		unsigned nTotal = VOL_X * VOL_Y * VOL_Z;

		int dx = VOL_X - DIM_X;
		int dy = VOL_Y - DIM_Y;

		for(int k = 0; k < VOL_Z; k ++)
		{
			for(int j = 0; j < VOL_Y; j ++)
			for(int i = 0; i < VOL_X; i ++)
			{
				float sc = INVALID_VOLVAL;
				float v = INVALID_VOLVAL;

				//	Radius from pole
				float r = sqrt( powf(fabs(i * 1.f - VOL_X/2), 2.f) + powf(fabs(k * 1.f - VOL_Z/2), 2.f) );
				if(r < DIM_X)
				{
					int x_inx = (int)r;
					int y_inx = j;
					assert(x_inx >= 0 && x_inx < DIM_X);
					assert(y_inx >= 0 && y_inx < DIM_Y);

					int absInx = x_inx + y_inx * (DIM_X);
					sc = *(pElecData + absInx * 4 + 2); 
					v  = *(pElecData + absInx * 4 + 3);

					if(sc != INVALID_VOLVAL)
					{
						sc_x[0] = sc < sc_x[0] ? sc : sc_x[0];
						sc_x[1] = sc > sc_x[1] ? sc : sc_x[1];
					}
					if(v != INVALID_VOLVAL)
					{
						v_x[0] = v < v_x[0] ? v : v_x[0];
						v_x[1] = v > v_x[1] ? v : v_x[1];
					}
				}

				fprintf(fDp, "%f %f ", sc, v);			
			}
			printf("\b\b\b\b\b\b\b\b\b\%.2f", (k + 1) * 1.f/VOL_Z);
		}
		fclose(fDp);
		//
		printf(" == SC : {%.10f, %.10f}\n ==  V : {%.10f, %.10f}\n", sc_x[0], sc_x[1], v_x[0], v_x[1]);
		printf("\nDone\n");
	}
	else
	{
		printf("Erro when creating dest file.\n");
		delete [] pElecData;
			
		return false;
	}

	delete [] pElecData;
	return true;
}
#endif

int main()
{
#ifndef DATA_2D
	bool bRet = processData(DATA_PATH, VOL_X, VOL_Y, VOL_Z, ID_PATH, FIELD_PATH);
#else
	bool bRet = processData2D(DATA_PATH, VOL_X, VOL_Y, VOL_Z, FIELD_PATH);
#endif

	if(!bRet)
	{
		printf("Something is wrong...\n");
	}
	else
	{
		printf("Finished...\n");
	}
	system("pause");

	return 0;
}

