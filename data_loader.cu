#include <stdio.h>
#include <assert.h>

#include "consts.h"

bool loadData(	unsigned x, unsigned y, unsigned z, 
				char *pDataPath,
				float *&deviceData, int *&idData,
				float *&hostData, int *&hostIdData)
{

	unsigned nTotalCellNum = x * y * z;

	float *pCubeCells = new float[nTotalCellNum];
	if(pCubeCells == NULL)
	{
		return false;
	}

	int *pIdData = new int[nTotalCellNum];
	if(pIdData == NULL)
	{
		return false;
	}

	///		Info (min, max)
	float id_range[4][2] = { { 0xFFFFF, -0xFFFFF },
							 { 0xFFFFF, -0xFFFFF }, 
							 { 0xFFFFF, -0xFFFFF }, 
							 { 0xFFFFF, -0xFFFFF } };

	///		Loading
	///
	FILE *fp0 = fopen(pDataPath, "rb");
	if(fp0)
	{
		printf("Loading Data file...\n");

		for(int i = 0; i < nTotalCellNum; i ++)
		{
			fscanf(fp0, "%f %d ",  pCubeCells + i, pIdData + i);

			//	Get Range
			int inx = *(pIdData + i) - 1;
			float val = *(pCubeCells + i);

			if(val < id_range[inx][0]) id_range[inx][0] = val;
			if(val > id_range[inx][1]) id_range[inx][1] = val;

			//if(inx == -1)
			//{
			//	printf("[%f, %f, %f, %d]\n", *(pCubeCells + 3 * i + 0), 
			//									*(pCubeCells + 3 * i + 1), 
			//									*(pCubeCells + 3 * i + 2), 
			//									*(pIdData + i));
			//}
			if(i % 10000 == 0)
			{
				printf("\b\b\b\b\b\b\b\b\b\b%.2f", i * 1.f / nTotalCellNum);
			}
		}
		printf("\nDone\n");

		float gRange[2] = {0xFFFFF, -0xFFFFF};

		printf("Data Ranges:\n");
		for(int i = 0; i < 4; i ++)
		{
			printf("ID - [%d] : %f  ->  %f \n", i, id_range[i][0], id_range[i][1]);
			gRange[0] = gRange[0] > id_range[i][0] ? id_range[i][0] : gRange[0];
			gRange[1] = gRange[1] < id_range[i][1] ? id_range[i][1] : gRange[1];
		}

		printf("GLOBAL: [%.10f, %.10f]\n", gRange[0], gRange[1]);
		fStart = gRange[0];
		fEnd   = gRange[1];
		fclose(fp0);
	}
	else
	{
		printf("Opening Data file error... \n Please run DataProcess first\n");
		system("pause");
		delete [] pIdData;
		delete [] pCubeCells;
		return false;
	}

	//	Copy to CUDA memory
	cudaError_t err = cudaMemcpy( deviceData, pCubeCells, sizeof(float) * nTotalCellNum, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("Cell Data Copying Error! \n");
	}

	err = cudaMemcpy( idData, pIdData, sizeof(int) * nTotalCellNum, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("Cell Data Copying Error! \n");
	}
	
	//%%% Keep the data for the haptic tool
	memcpy(hostData, pCubeCells, sizeof(float) * nTotalCellNum);
	memcpy(hostIdData, pIdData, sizeof(int) * nTotalCellNum);
	//%%%

	delete [] pCubeCells;
	delete [] pIdData;
	return true;
}

bool loadData2D(	unsigned x, unsigned y, unsigned z, 
					char *pDataPath, float *&deviceData, float *&hostData)
{
	unsigned nTotalCellNum = x * y * z;

	float *pCubeCells = new float[nTotalCellNum];
	if(pCubeCells == NULL)
	{
		return false;
	}

	float gRange[2] = {0xFFFFFF, -0xFFFFFF};

	///		Loading
	///
	FILE *fp0 = fopen(pDataPath, "rb");
	if(fp0)
	{
		printf("Loading Data file...\n");

		float dmp;
		for(int i = 0; i < nTotalCellNum; i ++)
		{
			fscanf(fp0, "%f %f ",  pCubeCells + i, &dmp);

			//	Get Range			
			float val = *(pCubeCells + i);
			if(val != INVALID_VOLVAL)
			{
				gRange[1] = val > gRange[1] ? val : gRange[1];
				gRange[0] = val < gRange[0] ? val : gRange[0];
			}
			if(i % VOL_X == 0)
			{
				printf("\b\b\b\b\b\b\b\b\b\b%.2f", i * 1.f / nTotalCellNum);
			}
		}
		printf("\nDone\n");

		printf("GLOBAL: [%.10f, %.10f]\n", gRange[0], gRange[1]);
		fStart = gRange[0];
		fEnd   = gRange[1];
		fclose(fp0);
	}
	else
	{
		printf("Opening Data file error... \n Please run DataProcess first\n");
		system("pause");
		delete [] pCubeCells;
		return false;
	}

	//	Copy to CUDA memory
	cudaError_t err = cudaMemcpy( deviceData, pCubeCells, sizeof(float) * nTotalCellNum, cudaMemcpyHostToDevice);
	if(err != cudaSuccess)
	{
		printf("Cell Data Copying Error! \n");
	}
	
	//%%% Keep the data for the haptic tool
	memcpy(hostData, pCubeCells, sizeof(float) * nTotalCellNum);
	//%%%

	delete [] pCubeCells;
	return true;
}