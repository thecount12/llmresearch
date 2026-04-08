#ifndef PLAN9_MODEL_H
#define PLAN9_MODEL_H

#include <u.h>
#include <libc.h>

typedef struct Config Config;
typedef struct LayerWeights LayerWeights;
typedef struct Model Model;
typedef struct KVCache KVCache;
typedef struct RunState RunState;

enum {
	LoaderUnknown,
	LoaderSimple,
	LoaderGGUF
};

struct Config {
	int vocab_size;
	int dim;
	int hidden_dim;
	int n_layers;
	int n_heads;
	int n_kv_heads;
	int seq_len;
	float rms_eps;
};

struct LayerWeights {
	float *rms_att_weight;
	float *wq;
	float *wk;
	float *wv;
	float *wo;
	float *rms_ffn_weight;
	float *w1;
	float *w2;
	float *w3;
};

struct Model {
	Config cfg;
	float *token_embedding_table;
	LayerWeights *layers;
	float *rms_final_weight;
	float *wcls;
	int owns_memory;
};

struct KVCache {
	float *k;
	float *v;
};

struct RunState {
	float *x;
	float *xb;
	float *xb2;
	float *hb;
	float *hb2;
	float *q;
	float *k;
	float *v;
	float *att;
	float *logits;
	KVCache cache;
};

int validate_config(Config *cfg, char *err, int nerr);

int alloc_model(Model *m, Config *cfg, char *err, int nerr);
void free_model(Model *m);
int init_toy_model(Model *m, char *err, int nerr);
int detect_loader_kind(char *path);
int load_model_auto(Model *m, char *path, char *err, int nerr);
int load_model_simple(Model *m, char *path, char *err, int nerr);
int load_model_gguf(Model *m, char *path, char *err, int nerr);

int alloc_run_state(RunState *s, Config *cfg, char *err, int nerr);
void free_run_state(RunState *s);
void clear_kv_cache(RunState *s, Config *cfg);

void vec_zero(float *x, int n);
void vec_copy(float *dst, float *src, int n);
void accum(float *dst, float *src, int n);
void scale(float *x, float s, int n);
float dot(float *a, float *b, int n);
void softmax(float *x, int n);
void rmsnorm(float *out, float *x, float *weight, int n, float eps);
void matvec(float *out, float *w, float *x, int nout, int nin);
float silu(float x);
void rope_apply(float *q, float *k, int pos, Config *cfg);

int transformer_forward(Model *m, RunState *s, int token, int pos, char *err, int nerr);
int greedy_sample(float *logits, int n);
int sample_with_temperature(float *logits, int n, float temperature);

#endif
