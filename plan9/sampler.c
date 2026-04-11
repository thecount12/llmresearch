#include "common.h"
#include "model.h"

/*
 * Match llama.cpp common_init_result: forbid sampling EOG / control-ish ids so greedy
 * matches reference when those logits would otherwise win.
 */
void
decode_logits_mask(Model *m, float *logits)
{
	static int qwen2_eog[] = {
		128247,	/* </s> */
		151643,	/* <|endoftext|> */
		151645,	/* <|im_end|> */
		151662, 151663, 151664
	};
	float neg;
	int n, i;

	if(m == nil || logits == nil)
		return;
	n = m->cfg.vocab_size;
	neg = -1e30f;
	if(m->cfg.arch == ArchQwen2){
		for(i = 0; i < (int)(sizeof qwen2_eog / sizeof qwen2_eog[0]); i++){
			if(qwen2_eog[i] >= 0 && qwen2_eog[i] < n)
				logits[qwen2_eog[i]] = neg;
		}
	}
	if(m->eos_token_id >= 0 && m->eos_token_id < n)
		logits[m->eos_token_id] = neg;
	if(m->pad_token_id >= 0 && m->pad_token_id < n)
		logits[m->pad_token_id] = neg;
}

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
