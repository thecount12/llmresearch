#include "model.h"

int
greedy_sample(float *logits, int n)
{
	int i, best;
	float bestv;

	best = 0;
	bestv = logits[0];
	for(i = 1; i < n; i++){
		if(logits[i] > bestv){
			bestv = logits[i];
			best = i;
		}
	}
	return best;
}

int
sample_with_temperature(float *logits, int n, float temperature)
{
	int i;
	float r, cdf;

	if(temperature <= 0.0f)
		return greedy_sample(logits, n);

	for(i = 0; i < n; i++)
		logits[i] /= temperature;
	softmax(logits, n);

	r = (float)nrand(1000000) / 1000000.0f;
	cdf = 0.0f;
	for(i = 0; i < n; i++){
		cdf += logits[i];
		if(r <= cdf)
			return i;
	}
	return n - 1;
}
