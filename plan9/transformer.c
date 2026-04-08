#include "common.h"
#include "model.h"

static int
kv_dim(Config *cfg)
{
	return (cfg->dim / cfg->n_heads) * cfg->n_kv_heads;
}

static void
copy_embedding(float *dst, float *table, int token, Config *cfg)
{
	memmove(dst, table + token * cfg->dim, cfg->dim * sizeof(float));
}

int
transformer_forward(Model *m, RunState *s, int token, int pos, char *err, int nerr)
{
	Config *cfg;
	LayerWeights *lw;
	float *kcache, *vcache, *qhead, *kvec, *vvec;
	float score, inv_scale;
	int l, i, t, h, dim, head_dim, n_heads, n_kv_heads, kv_repeat, kdim, kvh;

	USED(err);
	USED(nerr);

	cfg = &m->cfg;
	dim = cfg->dim;
	n_heads = cfg->n_heads;
	n_kv_heads = cfg->n_kv_heads;
	head_dim = dim / n_heads;
	kv_repeat = n_heads / n_kv_heads;
	kdim = kv_dim(cfg);

	if(token < 0 || token >= cfg->vocab_size)
		return -1;
	if(pos < 0 || pos >= cfg->seq_len)
		return -1;

	copy_embedding(s->x, m->token_embedding_table, token, cfg);

	for(l = 0; l < cfg->n_layers; l++){
		lw = &m->layers[l];

		rmsnorm(s->xb, s->x, lw->rms_att_weight, dim);
		matvec(s->q, lw->wq, s->xb, dim, dim);
		matvec(s->k, lw->wk, s->xb, kdim, dim);
		matvec(s->v, lw->wv, s->xb, kdim, dim);
		rope_apply(s->q, s->k, pos, cfg);

		kcache = s->cache.k + (l * cfg->seq_len + pos) * kdim;
		vcache = s->cache.v + (l * cfg->seq_len + pos) * kdim;
		vec_copy(kcache, s->k, kdim);
		vec_copy(vcache, s->v, kdim);

		vec_zero(s->xb2, dim);
		inv_scale = 1.0f / sqrt((float)head_dim);

		for(h = 0; h < n_heads; h++){
			qhead = s->q + h * head_dim;
			kvh = h / kv_repeat;

			for(t = 0; t <= pos; t++){
				kvec = s->cache.k + (l * cfg->seq_len + t) * kdim + kvh * head_dim;
				score = dot(qhead, kvec, head_dim) * inv_scale;
				s->att[h * cfg->seq_len + t] = score;
			}
			softmax(s->att + h * cfg->seq_len, pos + 1);

			for(i = 0; i < head_dim; i++)
				s->xb2[h * head_dim + i] = 0.0f;

			for(t = 0; t <= pos; t++){
				vvec = s->cache.v + (l * cfg->seq_len + t) * kdim + kvh * head_dim;
				score = s->att[h * cfg->seq_len + t];
				for(i = 0; i < head_dim; i++)
					s->xb2[h * head_dim + i] += score * vvec[i];
			}
		}

		matvec(s->xb, lw->wo, s->xb2, dim, dim);
		accum(s->x, s->xb, dim);

		rmsnorm(s->xb, s->x, lw->rms_ffn_weight, dim);
		matvec(s->hb, lw->w1, s->xb, cfg->hidden_dim, dim);
		matvec(s->hb2, lw->w3, s->xb, cfg->hidden_dim, dim);
		for(i = 0; i < cfg->hidden_dim; i++)
			s->hb[i] = silu(s->hb[i]) * s->hb2[i];
		matvec(s->xb, lw->w2, s->hb, dim, cfg->hidden_dim);
		accum(s->x, s->xb, dim);
	}

	rmsnorm(s->xb, s->x, m->rms_final_weight, dim);
	matvec(s->logits, m->wcls, s->xb, cfg->vocab_size, dim);
	return 0;
}
