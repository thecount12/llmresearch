#ifndef PLAN9_MODEL_H
#define PLAN9_MODEL_H

typedef struct Config Config;
typedef struct LayerWeights LayerWeights;
typedef struct Model Model;
typedef struct KVCache KVCache;
typedef struct RunState RunState;
typedef struct ForwardDebug ForwardDebug;

enum {
	LoaderUnknown,
	LoaderSimple,
	LoaderGGUF
};

enum {
	ArchLlama,
	ArchQwen2
};

enum {
	RopeNormal = 0,	/* interleaved pairs (i,i+1); LLaMA-style */
	RopeNeox = 2	/* GPT-NeoX / GGML_ROPE_TYPE_NEOX: pairs (i, i+head_dim/2); Qwen2 */
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
	float rope_freq_base;	/* RoPE θ base: Llama ~1e4, Qwen2 ~1e6 */
	int sliding_window;	/* 0 = full causal; else limit KV positions */
	int arch;		/* ArchLlama, ArchQwen2, … */
	int rope_type;		/* RopeNormal or RopeNeox (match GGUF / llama.cpp) */
};

struct LayerWeights {
	float *rms_att_weight;
	float *wq;
	float *wk;
	float *wv;
	float *wo;
	float *bq;	/* Qwen2 GGUF: attn_{q,k,v}.bias; nil for Llama */
	float *bk;
	float *bv;
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
	int loader_kind;
	char **token_str;	/* GGUF tokenizer.ggml.tokens; nil if not loaded */
	int eos_token_id;	/* tokenizer.ggml.eos_token_id from GGUF, or -1 */
	int pad_token_id;	/* tokenizer.ggml.padding_token_id from GGUF, or -1 */
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

/*
 * Optional forward trace (stderr): one line per checkpoint (embed, L0.., pre_logits).
 * pos_filter: -1 = dump at every token position; else only when pos == pos_filter.
 */
struct ForwardDebug {
	int enabled;
	int fd;
	int pos_filter;
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
void embed_lookup_gguf(float *dst, float *table, int token, Config *cfg);
void matvec_logits_gguf(float *out, float *w, float *x, int dim, int vocab);
void matvec_k_proj_gguf(float *out, float *w, float *x, int kdim, int dim);
void matvec_gate_up_gguf(float *out, float *w, float *x, int hidden, int dim);
void matvec_ffn_down_gguf(float *out, float *w, float *x, int dim, int hidden);
float silu(float x);
void rope_apply(float *q, float *k, int pos, Config *cfg);

int greedy_sample(float *logits, int n);
int sample_with_temperature(float *logits, int n, float temperature);
void sampler_seed(ulong seed);
void dump_logits_topk(int fd, float *logits, int n, int k);
void hf_logits_fingerprint(int fd, float *logits, int n);
void vec_fingerprint(int fd, const char *arch, int pos, const char *kind, float *v, int n);
void decode_logits_mask(Model *m, float *logits);

int transformer_forward(Model *m, RunState *s, int token, int pos, char *err, int nerr, ForwardDebug *dbg);

#endif
