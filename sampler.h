#ifndef SAMPLER_H
#define SAMPLER_H

enum SamplingType {STRATIFIED, LOW_DISC, BEST_CANDID};

class Sampler
{
public:
	virtual void getNextSample(float *pX, float *pY) = 0;
};

class StratifiedSampler : public Sampler
{
public:
	void getNextSample(float *pX, float *pY);
};

class LowDiscrepancySampler : public Sampler
{
public:
	void getNextSample(float *pX, float *pY);
};

class BestCandidateSampler : public Sampler
{
public:
	void getNextSample(float *pX, float *pY);
};

#endif