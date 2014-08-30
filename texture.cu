#include "texture.h"
#include "consts.h"

#include "IL/ilut.h"

bool loadTexture(const char *imgPath)
{
	//	Start to load
	ILuint nCurrTexImg = 0;
	ilGenImages(1, &nCurrTexImg);
	ilBindImage(nCurrTexImg);	

	//	Get Image Info
	if(ilLoadImage(imgPath))
	{
		ILint nWidth = ilGetInteger(IL_IMAGE_WIDTH);
		ILint nHeight = ilGetInteger(IL_IMAGE_HEIGHT);

		unsigned texSize = nWidth * nHeight * 3 * sizeof(float);
		float *data_buf = new float[ texSize ];
		ilCopyPixels( 0, 0, 0, nWidth, nHeight, 1, IL_RGB, IL_FLOAT, data_buf);		

		if(deviceTexData)
		{
			cudaFree(&deviceTexData);
		}
		cudaError_t err = cudaMalloc(&deviceTexData, texSize);
		if(err != cudaSuccess)
		{
			printf("tex cudaMalloc failed...\n");
		}

		err = cudaMemcpy(deviceTexData, data_buf, texSize, cudaMemcpyHostToDevice);
		if(err != cudaSuccess)
		{
			printf("tex cudaMalloc failed...\n");
		}

		texWidth = nWidth;
		texHeight = nHeight;

		delete [] data_buf;
		ilDeleteImages(1, &nCurrTexImg);
	}
	else
	{
		ILenum ilErr = ilGetError();
		printf("Error in LoadImage: %d [%s]\n", ilErr, imgPath);
		ilDeleteImages(1, &nCurrTexImg);

		return false;
	}

	printf("%s loaded successfully ... \n", imgPath);
	return true;
}