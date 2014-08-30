#include "global.h"

extern void gpu_destroy();
#include "IL/ilut.h"
ILuint nCurrImg = 1;

#include <time.h>
#include <stdlib.h>

#include <cuda_runtime.h>

//
//	Engine..
//
Scene scene;

void global_init()
{
	//	DevIL init
	//
	ilInit();
	ilutRenderer(ILUT_OPENGL);
	ilutEnable(ILUT_OPENGL_CONV);

	//ilOriginFunc(IL_ORIGIN_UPPER_LEFT);
	//ilEnable(IL_ORIGIN_SET);

	ilGenImages(1, &nCurrImg);
	ilBindImage(nCurrImg);	

	//
	srand(clock());
}

void global_destroy()
{
	gpu_destroy();

	//	DevIL finalization
	ilDeleteImages(1, &nCurrImg);
}