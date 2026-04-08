#include "common.h"
#include "model.h"

static void
seterr(char *err, int nerr, char *msg)
{
	if(err == nil || nerr <= 0)
		return;
	snprint(err, nerr, "%s", msg);
}

static int
kv_dim(Config *cfg)
{
	return (cfg->dim / cfg->n_heads) * cfg->n_kv_heads;
}

int
validate_config(Config *cfg, char *err, int nerr)
{
	if(cfg->vocab_size <= 0){
		seterr(err, nerr, "bad vocab_size");
		return -1;
	}
	if(cfg->dim <= 0 || cfg->hidden_dim <= 0){
		seterr(err, nerr, "bad model dims");
		return -1;
	}
	if(cfg->n_layers <= 0 || cfg->n_heads <= 0 || cfg->n_kv_heads <= 0){
		seterr(err, nerr, "bad layer/head counts");
		return -1;
	}
	if(cfg->seq_len <= 0){
		seterr(err, nerr, "bad seq_len");
		return -1;
	}
	if(cfg->dim % cfg->n_heads != 0){
		seterr(err, nerr, "dim not divisible by n_heads");
		return -1;
	}
	if(cfg->n_heads % cfg->n_kv_heads != 0){
		seterr(err, nerr, "n_heads not divisible by n_kv_heads");
		return -1;
	}
	if(cfg->rms_eps <= 0)
		cfg->rms_eps = 1e-5f;
	return 0;
}

static void *
xalloc(ulong n, char *err, int nerr)
{
	void *p;

	p = mallocz(n, 1);
	if(p == nil)
		seterr(err, nerr, "mallocz failed");
	return p;
}

static int
alloc_layer(LayerWeights *lw, Config *cfg, char *err, int nerr)
{
	int dim, hidden, kdim;

	dim = cfg->dim;
	hidden = cfg->hidden_dim;
	kdim = kv_dim(cfg);

	lw->rms_att_weight = xalloc(dim * sizeof(float), err, nerr);
	lw->wq = xalloc(dim * dim * sizeof(float), err, nerr);
	lw->wk = xalloc(kdim * dim * sizeof(float), err, nerr);
	lw->wv = xalloc(kdim * dim * sizeof(float), err, nerr);
	lw->wo = xalloc(dim * dim * sizeof(float), err, nerr);
	lw->rms_ffn_weight = xalloc(dim * sizeof(float), err, nerr);
	lw->w1 = xalloc(hidden * dim * sizeof(float), err, nerr);
	lw->w2 = xalloc(dim * hidden * sizeof(float), err, nerr);
	lw->w3 = xalloc(hidden * dim * sizeof(float), err, nerr);

	if(lw->rms_att_weight == nil || lw->wq == nil || lw->wk == nil ||
	   lw->wv == nil || lw->wo == nil || lw->rms_ffn_weight == nil ||
	   lw->w1 == nil || lw->w2 == nil || lw->w3 == nil)
		return -1;
	return 0;
}

int
alloc_model(Model *m, Config *cfg, char *err, int nerr)
{
	int i;

	memset(m, 0, sizeof(*m));
	m->cfg = *cfg;
	if(validate_config(&m->cfg, err, nerr) < 0)
		return -1;

	m->token_embedding_table = xalloc(m->cfg.vocab_size * m->cfg.dim * sizeof(float), err, nerr);
	m->layers = xalloc(m->cfg.n_layers * sizeof(LayerWeights), err, nerr);
	m->rms_final_weight = xalloc(m->cfg.dim * sizeof(float), err, nerr);
	m->wcls = xalloc(m->cfg.vocab_size * m->cfg.dim * sizeof(float), err, nerr);
	if(m->token_embedding_table == nil || m->layers == nil || m->rms_final_weight == nil || m->wcls == nil){
		free_model(m);
		return -1;
	}
	for(i = 0; i < m->cfg.n_layers; i++){
		if(alloc_layer(&m->layers[i], &m->cfg, err, nerr) < 0){
			free_model(m);
			return -1;
		}
	}
	m->owns_memory = 1;
	return 0;
}

void
free_model(Model *m)
{
	int i;
	LayerWeights *lw;

	if(m == nil)
		return;
	if(m->layers != nil){
		for(i = 0; i < m->cfg.n_layers; i++){
			lw = &m->layers[i];
			free(lw->rms_att_weight);
			free(lw->wq);
			free(lw->wk);
			free(lw->wv);
			free(lw->wo);
			free(lw->rms_ffn_weight);
			free(lw->w1);
			free(lw->w2);
			free(lw->w3);
		}
	}
	free(m->token_embedding_table);
	free(m->layers);
	free(m->rms_final_weight);
	free(m->wcls);
	memset(m, 0, sizeof(*m));
}

int
alloc_run_state(RunState *s, Config *cfg, char *err, int nerr)
{
	int dim, hidden, seq, heads, kdim, layers;

	memset(s, 0, sizeof(*s));
	dim = cfg->dim;
	hidden = cfg->hidden_dim;
	seq = cfg->seq_len;
	heads = cfg->n_heads;
	layers = cfg->n_layers;
	kdim = kv_dim(cfg);

	s->x = xalloc(dim * sizeof(float), err, nerr);
	s->xb = xalloc(dim * sizeof(float), err, nerr);
	s->xb2 = xalloc(dim * sizeof(float), err, nerr);
	s->hb = xalloc(hidden * sizeof(float), err, nerr);
	s->hb2 = xalloc(hidden * sizeof(float), err, nerr);
	s->q = xalloc(dim * sizeof(float), err, nerr);
	s->k = xalloc(kdim * sizeof(float), err, nerr);
	s->v = xalloc(kdim * sizeof(float), err, nerr);
	s->att = xalloc(heads * seq * sizeof(float), err, nerr);
	s->logits = xalloc(cfg->vocab_size * sizeof(float), err, nerr);
	s->cache.k = xalloc(layers * seq * kdim * sizeof(float), err, nerr);
	s->cache.v = xalloc(layers * seq * kdim * sizeof(float), err, nerr);

	if(s->x == nil || s->xb == nil || s->xb2 == nil || s->hb == nil ||
	   s->hb2 == nil || s->q == nil || s->k == nil || s->v == nil ||
	   s->att == nil || s->logits == nil || s->cache.k == nil || s->cache.v == nil){
		free_run_state(s);
		return -1;
	}
	return 0;
}

void
free_run_state(RunState *s)
{
	if(s == nil)
		return;
	free(s->x);
	free(s->xb);
	free(s->xb2);
	free(s->hb);
	free(s->hb2);
	free(s->q);
	free(s->k);
	free(s->v);
	free(s->att);
	free(s->logits);
	free(s->cache.k);
	free(s->cache.v);
	memset(s, 0, sizeof(*s));
}

void
clear_kv_cache(RunState *s, Config *cfg)
{
	memset(s->cache.k, 0, cfg->n_layers * cfg->seq_len * kv_dim(cfg) * sizeof(float));
	memset(s->cache.v, 0, cfg->n_layers * cfg->seq_len * kv_dim(cfg) * sizeof(float));
}

static void
fill_identity(float *w, int n)
{
	int i;

	memset(w, 0, n * n * sizeof(float));
	for(i = 0; i < n; i++)
		w[i * n + i] = 1.0f;
}

static void
fill_rect_identity(float *w, int rows, int cols)
{
	int i, j, n;

	memset(w, 0, rows * cols * sizeof(float));
	n = rows < cols ? rows : cols;
	for(i = 0; i < n; i++){
		j = i * cols + i;
		w[j] = 1.0f;
	}
}

int
init_toy_model(Model *m, char *err, int nerr)
{
	Config cfg;
	int i, j;
	LayerWeights *lw;

	memset(&cfg, 0, sizeof(cfg));
	cfg.vocab_size = 256;
	cfg.dim = 32;
	cfg.hidden_dim = 64;
	cfg.n_layers = 2;
	cfg.n_heads = 4;
	cfg.n_kv_heads = 4;
	cfg.seq_len = 64;
	cfg.rms_eps = 1e-5f;

	if(alloc_model(m, &cfg, err, nerr) < 0)
		return -1;

	for(i = 0; i < cfg.vocab_size; i++){
		for(j = 0; j < cfg.dim; j++)
			m->token_embedding_table[i * cfg.dim + j] = ((i + j) % 17) / 17.0f;
	}

	for(i = 0; i < cfg.dim; i++)
		m->rms_final_weight[i] = 1.0f;

	fill_rect_identity(m->wcls, cfg.vocab_size, cfg.dim);

	for(i = 0; i < cfg.n_layers; i++){
		lw = &m->layers[i];
		for(j = 0; j < cfg.dim; j++){
			lw->rms_att_weight[j] = 1.0f;
			lw->rms_ffn_weight[j] = 1.0f;
		}
		fill_identity(lw->wq, cfg.dim);
		fill_identity(lw->wo, cfg.dim);
		fill_rect_identity(lw->wk, kv_dim(&cfg), cfg.dim);
		fill_rect_identity(lw->wv, kv_dim(&cfg), cfg.dim);
		fill_rect_identity(lw->w1, cfg.hidden_dim, cfg.dim);
		fill_rect_identity(lw->w2, cfg.dim, cfg.hidden_dim);
		fill_rect_identity(lw->w3, cfg.hidden_dim, cfg.dim);
	}
	return 0;
}

int
detect_loader_kind(char *path)
{
	int n;

	n = strlen(path);
	if(n >= 5 && strcmp(path + n - 5, ".gguf") == 0)
		return LoaderGGUF;
	if(n >= 4 && strcmp(path + n - 4, ".p9m") == 0)
		return LoaderSimple;
	if(n >= 4 && strcmp(path + n - 4, ".bin") == 0)
		return LoaderSimple;
	return LoaderUnknown;
}

int
load_model_auto(Model *m, char *path, char *err, int nerr)
{
	int kind;

	kind = detect_loader_kind(path);
	switch(kind){
	case LoaderSimple:
		return load_model_simple(m, path, err, nerr);
	case LoaderGGUF:
		return load_model_gguf(m, path, err, nerr);
	default:
		seterr(err, nerr, "unknown model format");
		return -1;
	}
}

int
load_model_gguf(Model *m, char *path, char *err, int nerr)
{
	USED(m);
	USED(path);
	seterr(err, nerr, "gguf loader not implemented yet");
	return -1;
}
