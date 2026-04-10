#include "common.h"
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

void
sampler_seed(ulong seed)
{
	srand((long)seed);
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

void
dump_logits_topk(int fd, float *logits, int n, int k)
{
	int round, j, u;
	int picked[8];
	int besti;
	int nk;

	if(logits == nil || n <= 0)
		return;
	nk = k;
	if(nk > 8)
		nk = 8;
	if(nk > n)
		nk = n;
	for(round = 0; round < nk; round++){
		besti = -1;
		for(j = 0; j < n; j++){
			for(u = 0; u < round; u++)
				if(picked[u] == j)
					goto skipj;
			if(besti < 0 || logits[j] > logits[besti])
				besti = j;
		skipj:;
		}
		if(besti < 0)
			break;
		picked[round] = besti;
		fprint(fd, " %d:%g", besti, logits[besti]);
	}
	fprint(fd, "\n");
}
