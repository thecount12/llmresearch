#include "common.h"
#include "model.h"

void
vec_zero(float *x, int n)
{
	int i;

	for(i = 0; i < n; i++)
		x[i] = 0.0f;
}

void
vec_copy(float *dst, float *src, int n)
{
	memmove(dst, src, n * sizeof(float));
}

void
accum(float *dst, float *src, int n)
{
	int i;

	for(i = 0; i < n; i++)
		dst[i] += src[i];
}

void
scale(float *x, float s, int n)
{
	int i;

	for(i = 0; i < n; i++)
		x[i] *= s;
}

float
dot(float *a, float *b, int n)
{
	int i;
	float sum;

	sum = 0.0f;
	for(i = 0; i < n; i++)
		sum += a[i] * b[i];
	return sum;
}

void
softmax(float *x, int n)
{
	int i;
	float maxv, sum;

	if(n <= 0)
		return;
	maxv = x[0];
	for(i = 1; i < n; i++)
		if(x[i] > maxv)
			maxv = x[i];
	sum = 0.0f;
	for(i = 0; i < n; i++){
		x[i] = exp(x[i] - maxv);
		sum += x[i];
	}
	if(sum == 0.0f)
		return;
	for(i = 0; i < n; i++)
		x[i] /= sum;
}

void
rmsnorm(float *out, float *x, float *weight, int n, float eps)
{
	int i;
	float ss, scalev;

	ss = 0.0f;
	for(i = 0; i < n; i++)
		ss += x[i] * x[i];
	ss /= n;
	scalev = 1.0f / sqrt(ss + eps);
	for(i = 0; i < n; i++)
		out[i] = weight[i] * (x[i] * scalev);
}

void
matvec(float *out, float *w, float *x, int nout, int nin)
{
	int i, j;
	float sum;

	for(i = 0; i < nout; i++){
		sum = 0.0f;
		for(j = 0; j < nin; j++)
			sum += w[i * nin + j] * x[j];
		out[i] = sum;
	}
}

/*
 * GGUF stores token_embd.weight as [dim, vocab] row-major: table[d*vocab + t].
 * Simple/toy format uses [vocab, dim]: table[t*dim + d].
 */
void
embed_lookup_gguf(float *dst, float *table, int token, Config *cfg)
{
	int d, v;

	v = cfg->vocab_size;
	for(d = 0; d < cfg->dim; d++)
		dst[d] = table[d * v + token];
}

/*
 * GGUF output.weight is [dim, vocab]: logits[v] = sum_d w[d*vocab+v] * x[d].
 */
void
matvec_logits_gguf(float *out, float *w, float *x, int dim, int vocab)
{
	int v, d;
	float sum;

	for(v = 0; v < vocab; v++){
		sum = 0.0f;
		for(d = 0; d < dim; d++)
			sum += w[d * vocab + v] * x[d];
		out[v] = sum;
	}
}

/*
 * GGUF attn_{k,v}.weight is [dim, kdim] row-major; PyTorch Linear uses [kdim, dim].
 * out[i] = sum_r w[r*kdim+i]*x[r].
 */
void
matvec_k_proj_gguf(float *out, float *w, float *x, int kdim, int dim)
{
	int i, r;
	float sum;

	for(i = 0; i < kdim; i++){
		sum = 0.0f;
		for(r = 0; r < dim; r++)
			sum += w[r * kdim + i] * x[r];
		out[i] = sum;
	}
}

/*
 * GGUF ffn_gate / ffn_up is [dim, hidden]; matmul wants [hidden, dim].
 */
void
matvec_gate_up_gguf(float *out, float *w, float *x, int hidden, int dim)
{
	int h, r;
	float sum;

	for(h = 0; h < hidden; h++){
		sum = 0.0f;
		for(r = 0; r < dim; r++)
			sum += w[r * hidden + h] * x[r];
		out[h] = sum;
	}
}

/*
 * GGUF ffn_down is [hidden, dim]; matmul wants [dim, hidden].
 */
void
matvec_ffn_down_gguf(float *out, float *w, float *x, int dim, int hidden)
{
	int d, h;
	float sum;

	for(d = 0; d < dim; d++){
		sum = 0.0f;
		for(h = 0; h < hidden; h++)
			sum += w[h * dim + d] * x[h];
		out[d] = sum;
	}
}

float
silu(float x)
{
	return x / (1.0f + exp(-x));
}

void
rope_apply(float *q, float *k, int pos, Config *cfg)
{
	int h, ic, head_dim, half, kv_head_dim, kv_repeat, kv_head;
	float base, ang, c, s;
	float q0, q1, k0, k1;
	int i0, i1;

	head_dim = cfg->dim / cfg->n_heads;
	half = head_dim / 2;
	kv_head_dim = head_dim;
	kv_repeat = cfg->n_heads / cfg->n_kv_heads;
	base = cfg->rope_freq_base > 0 ? cfg->rope_freq_base : 10000.0f;

	for(h = 0; h < cfg->n_heads; h++){
		for(ic = 0; ic < half; ic++){
			/* Same θ schedule as ggml: angle = pos * base^(-2*ic/head_dim) */
			ang = pos * pow(base, -2.0f * (float)ic / (float)head_dim);
			c = cos(ang);
			s = sin(ang);

			if(cfg->rope_type == RopeNeox){
				/* GGML_ROPE_TYPE_NEOX: rotate (ic, ic + half) */
				i0 = ic;
				i1 = ic + half;
				q0 = q[h * head_dim + i0];
				q1 = q[h * head_dim + i1];
				q[h * head_dim + i0] = q0 * c - q1 * s;
				q[h * head_dim + i1] = q0 * s + q1 * c;

				kv_head = h / kv_repeat;
				k0 = k[kv_head * kv_head_dim + i0];
				k1 = k[kv_head * kv_head_dim + i1];
				k[kv_head * kv_head_dim + i0] = k0 * c - k1 * s;
				k[kv_head * kv_head_dim + i1] = k0 * s + k1 * c;
			}else{
				/* LLaMA-style: consecutive pairs (2*ic, 2*ic+1) */
				i0 = 2 * ic;
				i1 = i0 + 1;
				q0 = q[h * head_dim + i0];
				q1 = q[h * head_dim + i1];
				q[h * head_dim + i0] = q0 * c - q1 * s;
				q[h * head_dim + i1] = q0 * s + q1 * c;

				kv_head = h / kv_repeat;
				k0 = k[kv_head * kv_head_dim + i0];
				k1 = k[kv_head * kv_head_dim + i1];
				k[kv_head * kv_head_dim + i0] = k0 * c - k1 * s;
				k[kv_head * kv_head_dim + i1] = k0 * s + k1 * c;
			}
		}
	}
}
