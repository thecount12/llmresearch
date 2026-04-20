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
	softmax_f64norm(logits, n);

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

/*
 * Generic float vector fingerprint (same sumsq/cksum as hf_logits_fingerprint).
 * arch: short label (e.g. qwen2, llama); kind: embed, L0, pre_logits, ...
 */
void
vec_fingerprint(int fd, const char *arch, int pos, const char *kind, float *v, int n)
{
	double sumsq;
	unsigned long long cksum;
	union { float f; unsigned u; } uu;
	int i;

	if(v == nil || n <= 0 || arch == nil || kind == nil)
		return;
	sumsq = 0.0;
	cksum = 1469598103934665603ULL;
	for(i = 0; i < n; i++){
		double x;

		x = (double)v[i];
		sumsq += x * x;
		uu.f = v[i];
		cksum ^= (unsigned long long)uu.u ^ ((unsigned long long)i << 1);
		cksum *= 1099511628211ULL;
	}
	fprint(fd, "lumen_dbg: arch=%s pos=%d kind=%s dim=%d sumsq=%.18g cksum=%llux\n",
		arch, pos, kind, n, sumsq, cksum);
}

/*
 * Human-readable float dump for diffing against hf_hidden_ref.py (same prefix grep as lumen_dbg).
 */
void
vec_dump_raw(int fd, const char *arch, int pos, const char *kind, float *v, int n)
{
	int i;

	if(v == nil || n <= 0 || arch == nil || kind == nil)
		return;
	fprint(fd, "lumen_dbg_raw: arch=%s pos=%d kind=%s n=%d", arch, pos, kind, n);
	for(i = 0; i < n; i++)
		fprint(fd, " %.9g", (double)v[i]);
	fprint(fd, "\n");
}

/*
 * One stderr block for comparing pre-mask next-token logits to HF (hf_logits_ref.py).
 * sumsq = sum(logits[i]^2); cksum = 64-bit mix of float bits and index (not cryptographic).
 */
void
hf_logits_fingerprint(int fd, float *logits, int n)
{
	int i, g, round, j, u;
	int picked[8];
	int besti;
	int nk;
	double sumsq;
	unsigned long long cksum;
	union { float f; unsigned u; } uu;

	if(logits == nil || n <= 0)
		return;
	g = greedy_sample(logits, n);
	sumsq = 0.0;
	cksum = 1469598103934665603ULL;
	for(i = 0; i < n; i++){
		double x;

		x = (double)logits[i];
		sumsq += x * x;
		uu.f = logits[i];
		cksum ^= (unsigned long long)uu.u ^ ((unsigned long long)i << 1);
		cksum *= 1099511628211ULL;
	}
	fprint(fd, "lumen_hf: pre_mask greedy_id=%d greedy_logit=%g sumsq=%.18g cksum=%llux\n",
		g, logits[g], sumsq, cksum);
	fprint(fd, "lumen_hf: pre_mask top5");
	nk = 5;
	if(nk > n)
		nk = n;
	for(round = 0; round < nk; round++){
		besti = -1;
		for(j = 0; j < n; j++){
			for(u = 0; u < round; u++)
				if(picked[u] == j)
					goto skipfp;
			if(besti < 0 || logits[j] > logits[besti])
				besti = j;
		skipfp:;
		}
		if(besti < 0)
			break;
		picked[round] = besti;
		fprint(fd, " %d:%g", besti, logits[besti]);
	}
	fprint(fd, "\n");
}
