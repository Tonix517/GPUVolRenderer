#include "sampler.h"

#include <stdlib.h>
#include <assert.h>

void StratifiedSampler::getNextSample(float *pX, float *pY)
{
	*pX = (rand() % 10 - 5) / 10.f;
	*pY = (rand() % 10 - 5) / 10.f;
};

void LowDiscrepancySampler::getNextSample(float *pX, float *pY)
{
	assert("not implemented yet.");
};

void BestCandidateSampler::getNextSample(float *pX, float *pY)
{
	assert("not implemented yet.");
};