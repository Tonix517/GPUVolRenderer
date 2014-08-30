#ifndef FILM_H
#define FILM_H

#include "GL/glee.h"

#include <stdlib.h>

class Film
{
public:
	
	static void init(unsigned x, unsigned y);
	static void clear();
	static void destroy();

	static void render();

	static GLuint GetFrameBufId()
	{
		return nBufferId;
	}

private:
	static void clampFilmColor(float *);

private:

	static GLuint nBufferId;

	static unsigned nWidth, nHeight;

	static void *_pFrameBuf;
};
#endif