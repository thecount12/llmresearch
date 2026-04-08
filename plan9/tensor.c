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
rmsnorm(float *out, float *x, float *weight, int n)
{
	int i;
	float ss, scalev;
	float eps;

	eps = 1e-5f;
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

float
silu(float x)
{
	return x / (1.0f + exp(-x));
}

void
rope_apply(float *q, float *k, int pos, Config *cfg)
{
	int h, i, head_dim, kv_head_dim, kv_repeat, kv_head;
	float theta, freq, c, s;
	float q0, q1, k0, k1;

	head_dim = cfg->dim / cfg->n_heads;
	kv_head_dim = head_dim;
	kv_repeat = cfg->n_heads / cfg->n_kv_heads;

	for(h = 0; h < cfg->n_heads; h++){
		for(i = 0; i + 1 < head_dim; i += 2){
			theta = (float)i / (float)head_dim;
			freq = pow(10000.0f, -theta);
			c = cos(pos * freq);
			s = sin(pos * freq);

			q0 = q[h * head_dim + i];
			q1 = q[h * head_dim + i + 1];
			q[h * head_dim + i] = q0 * c - q1 * s;
			q[h * head_dim + i + 1] = q0 * s + q1 * c;

			kv_head = h / kv_repeat;
			k0 = k[kv_head * kv_head_dim + i];
			k1 = k[kv_head * kv_head_dim + i + 1];
			k[kv_head * kv_head_dim + i] = k0 * c - k1 * s;
			k[kv_head * kv_head_dim + i + 1] = k0 * s + k1 * c;
		}
	}
}
