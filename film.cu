#include "film.h"
#include "consts.h"
#include "ray.h"

#include <cuda_gl_interop.h>

#include <assert.h>
#include <stdio.h>

//	static vars init
void *Film::_pFrameBuf = 0;
GLuint Film::nBufferId = 0;
unsigned Film::nWidth = 0, Film::nHeight = 0;

#define BUFFER_OFFSET(i) ((char *)NULL + (i))

//

void Film::init(unsigned x, unsigned y)
{
	nWidth = x; nHeight = y;
	assert(x > 0 && y > 0);

	//	memory setup
	assert(_pFrameBuf == 0);
	_pFrameBuf = malloc( sizeof(float) * 3 * nWidth * nHeight);	
	assert(_pFrameBuf);

	clear();

	//	OGL setup
	if(GL_ARB_vertex_buffer_object)
	{
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1); //%%%Changed this from Tony's sizeof(float)

		glGenBuffers(1, &nBufferId);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, nBufferId);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, nBufferId);
		//	this call copys the pixel data to promise real-time update
		glBufferData( GL_PIXEL_UNPACK_BUFFER, 
					  3 * nWidth * nHeight * sizeof(float), 
					  _pFrameBuf, 
					  GL_DYNAMIC_DRAW); // this buffer is intended to be modified many times
										// and rendered many times

		//	Register to CUDA
		cudaError_t err0 = cudaGLRegisterBufferObject(nBufferId);
		if(err0 != cudaSuccess)
		{
			printf("cudaGLRegisterBufferObject Failed : %s\n", cudaGetErrorString(err0));
		}

		//	check error
		GLenum err = glGetError();
		if(err != GL_NO_ERROR)
		{
			printf("[GL ERROR] %s - %d : 0x%x\n", __FILE__, __LINE__, err);
		}
	}
	else
	{
		printf("[ERROR] OpenGL version is too low to support vertex buffer !\n");
	}

}

void Film::clear()
{
	for(int i = 0; i < nWidth; i ++)
	for(int j = 0; j < nHeight; j ++)
	{
		float v = 0;
		if( (i/128 + j/128) % 2 )
		{
			v = 1.f;
		}
		((float*)_pFrameBuf)[(j * nWidth + i) * 3 + 0] = v;
		((float*)_pFrameBuf)[(j * nWidth + i) * 3 + 1] = v;
		((float*)_pFrameBuf)[(j * nWidth + i) * 3 + 2] = v;
	}
}

void Film::destroy()
{
	cudaGLUnregisterBufferObject(nBufferId);
	glDeleteBuffers(1, &nBufferId);

	if(_pFrameBuf)
	{
		free(_pFrameBuf);
		_pFrameBuf = NULL;
	}
}

void Film::clampFilmColor(float *pColor)
{
	assert(pColor);

	for(int i = 0; i < 3; i ++)
	{
		if(pColor[i] > 1.f) pColor[i] = 1.f;
	}
}

void Film::render()
{
	assert(_pFrameBuf != NULL);	

	glClear(GL_COLOR_BUFFER_BIT);
	glRasterPos2i(0, 0);

	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, nBufferId);
	//
	////	this call copys the pixel data to promise real-time update
	//glBufferData( GL_PIXEL_UNPACK_BUFFER, 
	//			  3 * nWidth * nHeight * sizeof(float), 
	//			  _pFrameBuf, 
	//			  GL_DYNAMIC_DRAW); // this buffer is intended to be modified many times
	//								// and rendered many times

	glDrawPixels(nWidth, nHeight, GL_RGB, GL_FLOAT, BUFFER_OFFSET(0));

	//	check error
	GLenum err = glGetError();
	if(err != GL_NO_ERROR)
	{
		printf("[GL ERROR] %s - %d : 0x%x\n", __FILE__, __LINE__, err);
	}
}
