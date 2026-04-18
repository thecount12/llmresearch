#include "common.h"
#include "model.h"

static char *
arch_str(Config *cfg)
{
	if(cfg->arch == ArchQwen2)
		return "qwen2";
	if(cfg->arch == ArchLlama)
		return "llama";
	return "?";
}

static void
forward_debug_emit(ForwardDebug *dbg, Config *cfg, int pos, const char *kind, float *v, int n)
{
	if(dbg == nil || !dbg->enabled)
		return;
	if(dbg->pos_filter >= 0 && pos != dbg->pos_filter)
		return;
	vec_fingerprint(dbg->fd, arch_str(cfg), pos, kind, v, n);
}

static int
kv_dim(Config *cfg)
{
	return (cfg->dim / cfg->n_heads) * cfg->n_kv_heads;
}

static void
copy_embedding_simple(float *dst, float *table, int token, Config *cfg)
{
	memmove(dst, table + token * cfg->dim, cfg->dim * sizeof(float));
}

int
transformer_forward(Model *m, RunState *s, int token, int pos, char *err, int nerr, ForwardDebug *dbg)
{
	Config *cfg;
	LayerWeights *lw;
	float *kcache, *vcache, *qhead, *kvec, *vvec;
	float score, inv_scale;
	int l, i, t, h, dim, head_dim, n_heads, n_kv_heads, kv_repeat, kdim, kvh;
	int t_start, natt, ti;
	char kbuf[24];

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

	if(m->loader_kind == LoaderGGUF)
		embed_lookup_gguf(s->x, m->token_embedding_table, token, cfg);
	else
		copy_embedding_simple(s->x, m->token_embedding_table, token, cfg);
	forward_debug_emit(dbg, cfg, pos, "embed", s->x, dim);

	for(l = 0; l < cfg->n_layers; l++){
		lw = &m->layers[l];

		rmsnorm(s->xb, s->x, lw->rms_att_weight, dim, cfg->rms_eps);
		snprint(kbuf, sizeof kbuf, "L%d_norm", l);
		forward_debug_emit(dbg, cfg, pos, kbuf, s->xb, dim);
		/*
		 * GGUF maps attn_k/v as [dim, kdim] (transpose of PyTorch [kdim, dim]); use matvec_k_proj_gguf.
		 * attn_q and attn_out are [dim, dim] with the same axis order as PyTorch [out, in] row-major;
		 * use matvec like the HF linear (do not use matvec_k_proj_gguf — that would apply W^T).
		 */
		matvec(s->q, lw->wq, s->xb, dim, dim);
		if(m->loader_kind == LoaderGGUF){
			matvec_k_proj_gguf(s->k, lw->wk, s->xb, kdim, dim);
			matvec_k_proj_gguf(s->v, lw->wv, s->xb, kdim, dim);
		}else{
			matvec(s->k, lw->wk, s->xb, kdim, dim);
			matvec(s->v, lw->wv, s->xb, kdim, dim);
		}
		if(lw->bq != nil){
			accum(s->q, lw->bq, dim);
			accum(s->k, lw->bk, kdim);
			accum(s->v, lw->bv, kdim);
		}
		rope_apply(s->q, s->k, pos, cfg);
		snprint(kbuf, sizeof kbuf, "L%d_rope", l);
		forward_debug_emit(dbg, cfg, pos, kbuf, s->q, dim);
		snprint(kbuf, sizeof kbuf, "L%d_krope", l);
		forward_debug_emit(dbg, cfg, pos, kbuf, s->k, kdim);

		kcache = s->cache.k + (l * cfg->seq_len + pos) * kdim;
		vcache = s->cache.v + (l * cfg->seq_len + pos) * kdim;
		vec_copy(kcache, s->k, kdim);
		vec_copy(vcache, s->v, kdim);

		vec_zero(s->xb2, dim);
		inv_scale = 1.0f / sqrt((float)head_dim);

		t_start = 0;
		if(cfg->sliding_window > 0){
			t_start = pos + 1 - cfg->sliding_window;
			if(t_start < 0)
				t_start = 0;
		}
		natt = pos + 1 - t_start;

		for(h = 0; h < n_heads; h++){
			qhead = s->q + h * head_dim;
			kvh = h / kv_repeat;

			for(ti = 0; ti < natt; ti++){
				t = t_start + ti;
				kvec = s->cache.k + (l * cfg->seq_len + t) * kdim + kvh * head_dim;
				score = dot(qhead, kvec, head_dim) * inv_scale;
				s->att[h * cfg->seq_len + t] = score;
			}
			softmax(s->att + h * cfg->seq_len + t_start, natt);
			if(l == 0 && h == 0 && dbg != nil && dbg->enabled &&
			   (dbg->pos_filter < 0 || pos == dbg->pos_filter)){
				snprint(kbuf, sizeof kbuf, "L0_probs_h0");
				forward_debug_emit(dbg, cfg, pos, kbuf,
					s->att + h * cfg->seq_len + t_start, natt);
			}

			for(i = 0; i < head_dim; i++)
				s->xb2[h * head_dim + i] = 0.0f;

			for(ti = 0; ti < natt; ti++){
				t = t_start + ti;
				vvec = s->cache.v + (l * cfg->seq_len + t) * kdim + kvh * head_dim;
				score = s->att[h * cfg->seq_len + t];
				for(i = 0; i < head_dim; i++)
					s->xb2[h * head_dim + i] += score * vvec[i];
			}
		}

		snprint(kbuf, sizeof kbuf, "L%d_preatn", l);
		forward_debug_emit(dbg, cfg, pos, kbuf, s->xb2, dim);

		matvec(s->xb, lw->wo, s->xb2, dim, dim);
		accum(s->x, s->xb, dim);
		snprint(kbuf, sizeof kbuf, "L%d_attn", l);
		forward_debug_emit(dbg, cfg, pos, kbuf, s->x, dim);

		rmsnorm(s->xb, s->x, lw->rms_ffn_weight, dim, cfg->rms_eps);
		if(m->loader_kind == LoaderGGUF){
			matvec_gate_up_gguf(s->hb, lw->w1, s->xb, cfg->hidden_dim, dim);
			matvec_gate_up_gguf(s->hb2, lw->w3, s->xb, cfg->hidden_dim, dim);
		}else{
			matvec(s->hb, lw->w1, s->xb, cfg->hidden_dim, dim);
			matvec(s->hb2, lw->w3, s->xb, cfg->hidden_dim, dim);
		}
		for(i = 0; i < cfg->hidden_dim; i++)
			s->hb[i] = silu(s->hb[i]) * s->hb2[i];
		if(m->loader_kind == LoaderGGUF)
			matvec_ffn_down_gguf(s->xb, lw->w2, s->hb, dim, cfg->hidden_dim);
		else
			matvec(s->xb, lw->w2, s->hb, dim, cfg->hidden_dim);
		accum(s->x, s->xb, dim);
		snprint(kbuf, sizeof kbuf, "L%d", l);
		forward_debug_emit(dbg, cfg, pos, kbuf, s->x, dim);
	}

	rmsnorm(s->xb, s->x, m->rms_final_weight, dim, cfg->rms_eps);
	forward_debug_emit(dbg, cfg, pos, "pre_logits", s->xb, dim);
	if(m->loader_kind == LoaderGGUF)
		matvec_logits_gguf(s->logits, m->wcls, s->xb, dim, cfg->vocab_size, cfg->embed_layout);
	else
		matvec(s->logits, m->wcls, s->xb, cfg->vocab_size, dim);
	return 0;
}
