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

/*
 * Softmax with float64 normalization sum (closer to PyTorch F.softmax(..., dtype=float32)).
 */
void
softmax_f64norm(float *x, int n)
{
	int i;
	float maxv;
	double sum;

	if(n <= 0)
		return;
	maxv = x[0];
	for(i = 1; i < n; i++)
		if(x[i] > maxv)
			maxv = x[i];
	sum = 0.0;
	for(i = 0; i < n; i++){
		x[i] = (float)exp((double)(x[i] - maxv));
		sum += (double)x[i];
	}
	if(sum == 0.0)
		return;
	for(i = 0; i < n; i++)
		x[i] = (float)((double)x[i] / sum);
}

void
rmsnorm(float *out, float *x, float *weight, int n, float eps)
{
	int i;
	double ss, scalev;

	/* HF Qwen2RMSNorm: variance in float32; double here matches that better than float acc. */
	ss = 0.0;
	for(i = 0; i < n; i++){
		double t = (double)x[i];
		ss += t * t;
	}
	ss /= (double)n;
	scalev = 1.0 / sqrt(ss + (double)eps);
	for(i = 0; i < n; i++)
		out[i] = (float)((double)weight[i] * (double)x[i] * scalev);
}

void
matvec(float *out, float *w, float *x, int nout, int nin)
{
	int i, j;
	double sum;

	for(i = 0; i < nout; i++){
		sum = 0.0;
		for(j = 0; j < nin; j++)
			sum += (double)w[i * nin + j] * (double)x[j];
		out[i] = (float)sum;
	}
}

/*
 * GGUF token_embd.weight after dequant (bytes follow ggml: ne[0] innermost):
 * embed_layout 0: tensor ne = (vocab, dim) — table[d*vocab + t] ([dim,vocab] logical)
 * embed_layout 1: tensor ne = (dim, vocab) — table[t*dim + d] (HF embed_tokens rows)
 * Simple/toy format uses [vocab, dim] like layout 1 but uses copy_embedding_simple().
 */
void
embed_lookup_gguf(float *dst, float *table, int token, Config *cfg)
{
	int d, v;

	v = cfg->vocab_size;
	if(cfg->embed_layout == 1){
		for(d = 0; d < cfg->dim; d++)
			dst[d] = table[token * cfg->dim + d];
		return;
	}
	for(d = 0; d < cfg->dim; d++)
		dst[d] = table[d * v + token];
}

/*
 * GGUF output.weight: same memory layout as token_embd (see Config.embed_layout).
 * layout 0: ne (vocab, dim) — logits[v] = sum_d w[d*vocab+v]*x[d]
 * layout 1: ne (dim, vocab) — logits[v] = sum_d w[v*dim+d]*x[d]
 */
void
matvec_logits_gguf(float *out, float *w, float *x, int dim, int vocab, int embed_layout)
{
	int v, d;
	double sum;

	if(embed_layout == 1){
		for(v = 0; v < vocab; v++){
			sum = 0.0;
			for(d = 0; d < dim; d++)
				sum += (double)w[v * dim + d] * (double)x[d];
			out[v] = (float)sum;
		}
		return;
	}
	for(v = 0; v < vocab; v++){
		sum = 0.0;
		for(d = 0; d < dim; d++)
			sum += (double)w[d * vocab + v] * (double)x[d];
		out[v] = (float)sum;
	}
}

/*
 * GGUF attn_k / attn_v are [dim, kdim] row-major = transpose of PyTorch [kdim, dim].
 * attn_q / attn_out are [dim, dim] same order as PyTorch; use matvec(), not this.
 * out[i] = sum_r w[r*kdim+i]*x[r].
 */
void
matvec_k_proj_gguf(float *out, float *w, float *x, int kdim, int dim)
{
	int i, r;
	double sum;

	for(i = 0; i < kdim; i++){
		sum = 0.0;
		for(r = 0; r < dim; r++)
			sum += (double)w[r * kdim + i] * (double)x[r];
		out[i] = (float)sum;
	}
}

/*
 * GGUF ffn_gate / ffn_up is [dim, hidden]; matmul wants [hidden, dim].
 */
void
matvec_gate_up_gguf(float *out, float *w, float *x, int hidden, int dim)
{
	int h, r;
	double sum;

	for(h = 0; h < hidden; h++){
		sum = 0.0;
		for(r = 0; r < dim; r++)
			sum += (double)w[r * hidden + h] * (double)x[r];
		out[h] = (float)sum;
	}
}

/*
 * GGUF ffn_down is [hidden, dim]; matmul wants [dim, hidden].
 */
void
matvec_ffn_down_gguf(float *out, float *w, float *x, int dim, int hidden)
{
	int d, h;
	double sum;

	for(d = 0; d < dim; d++){
		sum = 0.0;
		for(h = 0; h < hidden; h++)
			sum += (double)w[h * dim + d] * (double)x[h];
		out[d] = (float)sum;
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
	double based;

	head_dim = cfg->dim / cfg->n_heads;
	half = head_dim / 2;
	kv_head_dim = head_dim;
	kv_repeat = cfg->n_heads / cfg->n_kv_heads;
	base = cfg->rope_freq_base > 0 ? cfg->rope_freq_base : 10000.0f;
	based = (double)base;

	for(h = 0; h < cfg->n_heads; h++){
		for(ic = 0; ic < half; ic++){
			/* Same θ schedule as ggml / HF inv_freq: pos * base^(-2*ic/head_dim) */
			ang = (float)((double)pos * pow(based, -2.0 * (double)ic / (double)head_dim));
			c = cos((double)ang);
			s = sin((double)ang);

			if(cfg->rope_type == RopeNeox){
				/* GGML_ROPE_TYPE_NEOX: rotate (ic, ic + half) */
				i0 = ic;
				i1 = ic + half;
				q0 = q[h * head_dim + i0];
				q1 = q[h * head_dim + i1];
				q[h * head_dim + i0] = q0 * c - q1 * s;
				q[h * head_dim + i1] = q0 * s + q1 * c;

				/*
				 * GQA: k is shared — only n_kv_heads blocks. Rotate each kv head once,
				 * not once per query head (would apply RoPE kv_repeat times per block).
				 */
				if(h % kv_repeat == 0){
					kv_head = h / kv_repeat;
					k0 = k[kv_head * kv_head_dim + i0];
					k1 = k[kv_head * kv_head_dim + i1];
					k[kv_head * kv_head_dim + i0] = k0 * c - k1 * s;
					k[kv_head * kv_head_dim + i1] = k0 * s + k1 * c;
				}
			}else{
				/* LLaMA-style: consecutive pairs (2*ic, 2*ic+1) */
				i0 = 2 * ic;
				i1 = i0 + 1;
				q0 = q[h * head_dim + i0];
				q1 = q[h * head_dim + i1];
				q[h * head_dim + i0] = q0 * c - q1 * s;
				q[h * head_dim + i1] = q0 * s + q1 * c;

				if(h % kv_repeat == 0){
					kv_head = h / kv_repeat;
					k0 = k[kv_head * kv_head_dim + i0];
					k1 = k[kv_head * kv_head_dim + i1];
					k[kv_head * kv_head_dim + i0] = k0 * c - k1 * s;
					k[kv_head * kv_head_dim + i1] = k0 * s + k1 * c;
				}
			}
		}
	}
}
