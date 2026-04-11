#include "common.h"
#include "model.h"

typedef struct GGUFInfo GGUFInfo;
typedef struct GGUFMap GGUFMap;
typedef struct GGUFTensorPlan GGUFTensorPlan;
typedef struct GGUFLoadPlan GGUFLoadPlan;

enum {
	GGUFUint8 = 0,
	GGUFInt8,
	GGUFUint16,
	GGUFInt16,
	GGUFUint32,
	GGUFInt32,
	GGUFFloat32,
	GGUFBool,
	GGUFString,
	GGUFArray,
	GGUFUint64,
	GGUFInt64,
	GGUFFloat64
};

enum {
	GGMLTypeF32 = 0,
	GGMLTypeF16 = 1,
	GGMLTypeQ4_0 = 2,
	GGMLTypeQ8_0 = 8
};

enum {
	QK4_0 = 32,
	QK8_0 = 32
};

struct GGUFInfo {
	ulong version;
	uvlong tensor_count;
	uvlong kv_count;
	char architecture[64];
	uvlong vocab_size;
	uvlong dim;
	uvlong hidden_dim;
	uvlong n_layers;
	uvlong n_heads;
	uvlong n_kv_heads;
	uvlong seq_len;
	ulong alignment;
	char **vocab_tokens;
	uvlong vocab_tokens_n;
	float rope_freq_base;	/* 0 = not in KV */
	float rms_eps_kv;	/* 0 = not in KV */
	uvlong sliding_window;
};

struct GGUFMap {
	int token_embd;
	int output_norm;
	int output;
	int attn_norm;
	int attn_q;
	int attn_k;
	int attn_v;
	int attn_out;
	int ffn_norm;
	int ffn_gate;
	int ffn_down;
	int ffn_up;
	int unknown;
	int q4_0;
	int q8_0;
	int unsupported;
	ulong first_type;
	char first_name[96];
};

struct GGUFTensorPlan {
	char name[96];
	float *dst;
	uvlong count;
	uvlong off;
	ulong type;
};

struct GGUFLoadPlan {
	GGUFTensorPlan *items;
	int n;
	int cap;
	int tied_output;
};

static int
readfull9(int fd, void *buf, int n)
{
	uchar *p;
	int m, got;

	p = buf;
	got = 0;
	while(got < n){
		m = read(fd, p + got, n - got);
		if(m <= 0)
			return -1;
		got += m;
	}
	return 0;
}

static int
readu8(int fd, uchar *out)
{
	return readfull9(fd, out, 1);
}

static int
readu16(int fd, ushort *out)
{
	uchar b[2];

	if(readfull9(fd, b, 2) < 0)
		return -1;
	*out = (ushort)b[0] | ((ushort)b[1] << 8);
	return 0;
}

static int
readu32(int fd, ulong *out)
{
	uchar b[4];

	if(readfull9(fd, b, 4) < 0)
		return -1;
	*out = (ulong)b[0]
		| ((ulong)b[1] << 8)
		| ((ulong)b[2] << 16)
		| ((ulong)b[3] << 24);
	return 0;
}

static int
readu64(int fd, uvlong *out)
{
	uchar b[8];

	if(readfull9(fd, b, 8) < 0)
		return -1;
	*out = (uvlong)b[0]
		| ((uvlong)b[1] << 8)
		| ((uvlong)b[2] << 16)
		| ((uvlong)b[3] << 24)
		| ((uvlong)b[4] << 32)
		| ((uvlong)b[5] << 40)
		| ((uvlong)b[6] << 48)
		| ((uvlong)b[7] << 56);
	return 0;
}

static int
readf32(int fd, float *out)
{
	union {
		ulong u;
		float f;
	} v;

	if(readu32(fd, &v.u) < 0)
		return -1;
	*out = v.f;
	return 0;
}

static int
readf64(int fd, double *out)
{
	union {
		uvlong u;
		double d;
	} v;

	if(readu64(fd, &v.u) < 0)
		return -1;
	*out = v.d;
	return 0;
}

static float
u32asf(ulong u)
{
	union {
		ulong u;
		float f;
	} x;

	x.u = u & (ulong)0xffffffff;
	return x.f;
}

/*
 * IEEE fp16 -> fp32. Normals use explicit int exponent + uint32 bits (no float
 * multiply loop) so biased exp is never built via ulong underflow.
 * Subnormals use double scaling only.
 */
static float
f16tof32(ushort h)
{
	int exp, frac;
	int e;
	ulong u;
	double m, scale;
	int k;

	exp = (h >> 10) & 0x1f;
	frac = h & 0x3ff;

	if(exp == 0){
		if(frac == 0)
			return (h & 0x8000) ? -0.0f : 0.0f;
		m = (double)frac / 1024.0;
		scale = 1.0;
		for(k = 0; k < 14; k++)
			scale *= 0.5;
		return (float)(((h & 0x8000) ? -1.0 : 1.0) * m * scale);
	}
	if(exp == 31){
		if(frac == 0){
			/* sign bit: (h&0x8000)<<16 -> bit 31; never use (int)(1<<31) (UB on 32-bit int) */
			u = ((ulong)(h & 0x8000) << 16) | ((ulong)0xff << 23);
			return u32asf(u);
		}
		return 0.0f / 0.0f;
	}

	e = exp - 15 + 127;
	u = ((ulong)(h & 0x8000) << 16) | ((ulong)e << 23) | ((ulong)frac << 13);
	return u32asf(u);
}

static int
seekabs(int fd, vlong off)
{
	if(seek(fd, off, 0) < 0)
		return -1;
	return 0;
}

/* SmolLM Q4 ffn_down first-block scale is half 0x2a88 -> float bits 0x3d510000. */
static int
gguf_f16_selftest(char *err, int nerr)
{
	ushort h;
	union {
		float f;
		ulong u;
	} vu;

	h = 0x2a88;
	vu.f = f16tof32(h);
	if((vu.u & (ulong)0xffffffff) != (ulong)0x3d510000){
		snprint(err, nerr,
			"gguf f16 self-test: half=%#ux want bits %#lux got %#lux (recompile loader-gguf.c)",
			h, (ulong)0x3d510000, vu.u & (ulong)0xffffffff);
		return -1;
	}
	return 0;
}

static int
readstr(int fd, char **out)
{
	uvlong n;
	char *s;

	if(readu64(fd, &n) < 0)
		return -1;
	if(n > 1UL * 1024 * 1024)
		return -1;
	s = malloc(n + 1);
	if(s == nil)
		return -1;
	if(n > 0 && readfull9(fd, s, (int)n) < 0){
		free(s);
		return -1;
	}
	s[n] = 0;
	*out = s;
	return 0;
}

static int skipvalue(int fd, ulong type);

static int
skiparray(int fd)
{
	ulong elem_type;
	uvlong n, i;

	if(readu32(fd, &elem_type) < 0)
		return -1;
	if(readu64(fd, &n) < 0)
		return -1;
	for(i = 0; i < n; i++)
		if(skipvalue(fd, elem_type) < 0)
			return -1;
	return 0;
}

static int
skipvalue(int fd, ulong type)
{
	uchar u8;
	ushort u16;
	ulong u32;
	uvlong u64;
	float f32;
	double f64;
	char *s;

	switch(type){
	case GGUFUint8:
	case GGUFInt8:
	case GGUFBool:
		return readu8(fd, &u8);
	case GGUFUint16:
	case GGUFInt16:
		return readu16(fd, &u16);
	case GGUFUint32:
	case GGUFInt32:
		return readu32(fd, &u32);
	case GGUFFloat32:
		return readf32(fd, &f32);
	case GGUFUint64:
	case GGUFInt64:
		return readu64(fd, &u64);
	case GGUFFloat64:
		return readf64(fd, &f64);
	case GGUFString:
		if(readstr(fd, &s) < 0)
			return -1;
		free(s);
		return 0;
	case GGUFArray:
		return skiparray(fd);
	default:
		return -1;
	}
}

static void
copystr0(char *dst, int ndst, char *src)
{
	if(ndst <= 0)
		return;
	snprint(dst, ndst, "%s", src);
}

static int
addplan(GGUFLoadPlan *gp, char *name, float *dst, uvlong count, uvlong off, ulong type)
{
	if(gp->n >= gp->cap)
		return -1;
	snprint(gp->items[gp->n].name, sizeof gp->items[gp->n].name, "%s", name);
	gp->items[gp->n].dst = dst;
	gp->items[gp->n].count = count;
	gp->items[gp->n].off = off;
	gp->items[gp->n].type = type;
	gp->n++;
	return 0;
}

static int
checkdims1(ulong ndims, uvlong *dims, uvlong d0)
{
	return ndims == 1 && dims[0] == d0;
}

static int
checkdims2(ulong ndims, uvlong *dims, uvlong d0, uvlong d1)
{
	return ndims == 2 && dims[0] == d0 && dims[1] == d1;
}

static void
free_vocab_gi(GGUFInfo *gi)
{
	uvlong i;

	if(gi == nil || gi->vocab_tokens == nil)
		return;
	for(i = 0; i < gi->vocab_tokens_n; i++)
		if(gi->vocab_tokens[i] != nil)
			free(gi->vocab_tokens[i]);
	free(gi->vocab_tokens);
	gi->vocab_tokens = nil;
	gi->vocab_tokens_n = 0;
}

static void
setinfo_u64(GGUFInfo *gi, char *key, uvlong val)
{
	int nkey;

	nkey = strlen(key);
	if(nkey >= strlen(".context_length") && strcmp(key + nkey - strlen(".context_length"), ".context_length") == 0){
		gi->seq_len = val;
		return;
	}
	if(nkey >= strlen(".embedding_length") && strcmp(key + nkey - strlen(".embedding_length"), ".embedding_length") == 0){
		gi->dim = val;
		return;
	}
	if(nkey >= strlen(".feed_forward_length") && strcmp(key + nkey - strlen(".feed_forward_length"), ".feed_forward_length") == 0){
		gi->hidden_dim = val;
		return;
	}
	if(nkey >= strlen(".block_count") && strcmp(key + nkey - strlen(".block_count"), ".block_count") == 0){
		gi->n_layers = val;
		return;
	}
	if(nkey >= strlen(".attention.head_count") && strcmp(key + nkey - strlen(".attention.head_count"), ".attention.head_count") == 0){
		gi->n_heads = val;
		return;
	}
	if(nkey >= strlen(".attention.head_count_kv") && strcmp(key + nkey - strlen(".attention.head_count_kv"), ".attention.head_count_kv") == 0){
		gi->n_kv_heads = val;
		return;
	}
	if(nkey >= strlen(".attention.sliding_window") && strcmp(key + nkey - strlen(".attention.sliding_window"), ".attention.sliding_window") == 0){
		gi->sliding_window = val;
		return;
	}
}

static void
setinfo_float(GGUFInfo *gi, char *key, double val)
{
	int nkey;

	nkey = strlen(key);
	if(nkey >= strlen(".rope.freq_base") && strcmp(key + nkey - strlen(".rope.freq_base"), ".rope.freq_base") == 0){
		gi->rope_freq_base = (float)val;
		return;
	}
	if(nkey >= strlen(".attention.layer_norm_rms_epsilon") && strcmp(key + nkey - strlen(".attention.layer_norm_rms_epsilon"), ".attention.layer_norm_rms_epsilon") == 0){
		gi->rms_eps_kv = (float)val;
		return;
	}
}

static int
parse_metadata_value(int fd, GGUFInfo *gi, char *key, ulong type)
{
	uchar u8;
	ushort u16;
	ulong u32;
	uvlong u64;
	float f32;
	double f64;
	char *s;

	switch(type){
	case GGUFUint8:
	case GGUFInt8:
	case GGUFBool:
		if(readu8(fd, &u8) < 0)
			return -1;
		if(strcmp(key, "tokenizer.ggml.tokens") == 0)
			gi->vocab_size = u8;
		return 0;
	case GGUFUint16:
	case GGUFInt16:
		return readu16(fd, &u16);
	case GGUFUint32:
	case GGUFInt32:
		if(readu32(fd, &u32) < 0)
			return -1;
		if(strcmp(key, "general.alignment") == 0)
			gi->alignment = u32;
		if(strcmp(key, "tokenizer.ggml.tokens") == 0)
			gi->vocab_size = u32;
		setinfo_u64(gi, key, u32);
		return 0;
	case GGUFFloat32:
		if(readf32(fd, &f32) < 0)
			return -1;
		setinfo_float(gi, key, f32);
		return 0;
	case GGUFUint64:
	case GGUFInt64:
		if(readu64(fd, &u64) < 0)
			return -1;
		if(strcmp(key, "tokenizer.ggml.tokens") == 0)
			gi->vocab_size = u64;
		setinfo_u64(gi, key, u64);
		return 0;
	case GGUFFloat64:
		if(readf64(fd, &f64) < 0)
			return -1;
		setinfo_float(gi, key, f64);
		return 0;
	case GGUFString:
		if(readstr(fd, &s) < 0)
			return -1;
		if(strcmp(key, "general.architecture") == 0)
			copystr0(gi->architecture, sizeof gi->architecture, s);
		free(s);
		return 0;
	case GGUFArray:
		if(strcmp(key, "tokenizer.ggml.tokens") == 0){
			ulong elem_type;
			uvlong n, i;
			char **toks;
			char *one;

			if(readu32(fd, &elem_type) < 0)
				return -1;
			if(readu64(fd, &n) < 0)
				return -1;
			if(elem_type != GGUFString)
				return -1;
			gi->vocab_size = n;
			toks = mallocz(n * sizeof(char*), 1);
			if(toks == nil)
				return -1;
			for(i = 0; i < n; i++){
				if(readstr(fd, &one) < 0){
					while(i > 0){
						i--;
						free(toks[i]);
					}
					free(toks);
					return -1;
				}
				toks[i] = one;
			}
			gi->vocab_tokens = toks;
			gi->vocab_tokens_n = n;
			return 0;
		}
		return skiparray(fd);
	default:
		return -1;
	}
}

static int
parse_metadata(int fd, GGUFInfo *gi)
{
	uvlong i;
	char *key;
	ulong type;

	for(i = 0; i < gi->kv_count; i++){
		if(readstr(fd, &key) < 0){
			free_vocab_gi(gi);
			return -1;
		}
		if(readu32(fd, &type) < 0){
			free(key);
			free_vocab_gi(gi);
			return -1;
		}
		if(parse_metadata_value(fd, gi, key, type) < 0){
			free(key);
			free_vocab_gi(gi);
			return -1;
		}
		free(key);
	}
	return 0;
}

static int
parseblklayer(char *name, char *suffix)
{
	char *p, *q;
	int n;

	p = name;
	if(strncmp(p, "blk.", 4) != 0)
		return -1;
	p += 4;
	n = 0;
	while(*p >= '0' && *p <= '9'){
		n = n * 10 + (*p - '0');
		p++;
	}
	if(*p != '.')
		return -1;
	q = p + 1;
	if(strcmp(q, suffix) == 0)
		return n;
	return -1;
}

static void
marktype(GGUFMap *gm, char *name, ulong ggml_type)
{
	if(ggml_type == GGMLTypeF32)
		return;
	if(ggml_type == GGMLTypeF16)
		return;
	if(ggml_type == GGMLTypeQ4_0){
		gm->q4_0++;
		return;
	}
	if(ggml_type == GGMLTypeQ8_0){
		gm->q8_0++;
		return;
	}
	gm->unsupported++;
	if(gm->first_type == 0){
		gm->first_type = ggml_type;
		copystr0(gm->first_name, sizeof gm->first_name, name);
	}
}

static int
maptensor(Model *m, GGUFMap *gm, GGUFLoadPlan *gp, char *name, ulong ggml_type, ulong ndims, uvlong *dims, uvlong off)
{
	int layer;
	Config *cfg;
	int kdim;

	cfg = &m->cfg;
	kdim = (cfg->dim / cfg->n_heads) * cfg->n_kv_heads;

	/* Qwen2 GGUF may ship precomputed RoPE tables; forward uses rope.freq_base instead. */
	if(strcmp(name, "rope_freqs.weight") == 0)
		return 0;

	if(strcmp(name, "token_embd.weight") == 0){
		if(!checkdims2(ndims, dims, cfg->dim, cfg->vocab_size))
			return -1;
		marktype(gm, name, ggml_type);
		gm->token_embd++;
		return addplan(gp, name, m->token_embedding_table, cfg->vocab_size * cfg->dim, off, ggml_type);
	}
	if(strcmp(name, "output_norm.weight") == 0){
		if(!checkdims1(ndims, dims, cfg->dim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->output_norm++;
		return addplan(gp, name, m->rms_final_weight, cfg->dim, off, ggml_type);
	}
	if(strcmp(name, "output.weight") == 0){
		if(!checkdims2(ndims, dims, cfg->dim, cfg->vocab_size))
			return -1;
		marktype(gm, name, ggml_type);
		gm->output++;
		return addplan(gp, name, m->wcls, cfg->vocab_size * cfg->dim, off, ggml_type);
	}

	layer = parseblklayer(name, "attn_norm.weight");
	if(layer >= 0 && layer < cfg->n_layers){
		if(!checkdims1(ndims, dims, cfg->dim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->attn_norm++;
		return addplan(gp, name, m->layers[layer].rms_att_weight, cfg->dim, off, ggml_type);
	}
	layer = parseblklayer(name, "attn_q.weight");
	if(layer >= 0 && layer < cfg->n_layers){
		if(!checkdims2(ndims, dims, cfg->dim, cfg->dim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->attn_q++;
		return addplan(gp, name, m->layers[layer].wq, cfg->dim * cfg->dim, off, ggml_type);
	}
	layer = parseblklayer(name, "attn_k.weight");
	if(layer >= 0 && layer < cfg->n_layers){
		if(!checkdims2(ndims, dims, cfg->dim, kdim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->attn_k++;
		return addplan(gp, name, m->layers[layer].wk, kdim * cfg->dim, off, ggml_type);
	}
	layer = parseblklayer(name, "attn_v.weight");
	if(layer >= 0 && layer < cfg->n_layers){
		if(!checkdims2(ndims, dims, cfg->dim, kdim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->attn_v++;
		return addplan(gp, name, m->layers[layer].wv, kdim * cfg->dim, off, ggml_type);
	}
	layer = parseblklayer(name, "attn_output.weight");
	if(layer >= 0 && layer < cfg->n_layers){
		if(!checkdims2(ndims, dims, cfg->dim, cfg->dim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->attn_out++;
		return addplan(gp, name, m->layers[layer].wo, cfg->dim * cfg->dim, off, ggml_type);
	}
	layer = parseblklayer(name, "ffn_norm.weight");
	if(layer >= 0 && layer < cfg->n_layers){
		if(!checkdims1(ndims, dims, cfg->dim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->ffn_norm++;
		return addplan(gp, name, m->layers[layer].rms_ffn_weight, cfg->dim, off, ggml_type);
	}
	layer = parseblklayer(name, "ffn_gate.weight");
	if(layer >= 0 && layer < cfg->n_layers){
		if(!checkdims2(ndims, dims, cfg->dim, cfg->hidden_dim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->ffn_gate++;
		return addplan(gp, name, m->layers[layer].w1, cfg->hidden_dim * cfg->dim, off, ggml_type);
	}
	layer = parseblklayer(name, "ffn_down.weight");
	if(layer >= 0 && layer < cfg->n_layers){
		if(!checkdims2(ndims, dims, cfg->hidden_dim, cfg->dim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->ffn_down++;
		return addplan(gp, name, m->layers[layer].w2, cfg->dim * cfg->hidden_dim, off, ggml_type);
	}
	layer = parseblklayer(name, "ffn_up.weight");
	if(layer >= 0 && layer < cfg->n_layers){
		if(!checkdims2(ndims, dims, cfg->dim, cfg->hidden_dim))
			return -1;
		marktype(gm, name, ggml_type);
		gm->ffn_up++;
		return addplan(gp, name, m->layers[layer].w3, cfg->hidden_dim * cfg->dim, off, ggml_type);
	}

	gm->unknown++;
	return 0;
}

static int
parse_tensor_infos(int fd, uvlong n, Model *m, GGUFMap *gm, GGUFLoadPlan *gp)
{
	uvlong i, j, off;
	uvlong dims[4];
	ulong ndims, ggml_type;
	char *name;

	memset(gm, 0, sizeof(*gm));
	for(i = 0; i < n; i++){
		if(readstr(fd, &name) < 0)
			return -1;
		if(readu32(fd, &ndims) < 0){
			free(name);
			return -1;
		}
		if(ndims > 4){
			free(name);
			return -1;
		}
		for(j = 0; j < ndims; j++){
			if(readu64(fd, &dims[j]) < 0){
				free(name);
				return -1;
			}
		}
		if(readu32(fd, &ggml_type) < 0){
			free(name);
			return -1;
		}
		if(readu64(fd, &off) < 0){
			free(name);
			return -1;
		}
		if(maptensor(m, gm, gp, name, ggml_type, ndims, dims, off) < 0){
			free(name);
			return -1;
		}
		free(name);
	}
	return 0;
}

static int
hasrequired(GGUFInfo *gi, GGUFMap *gm)
{
	if(gm->token_embd < 1 || gm->output_norm < 1)
		return 0;
	if(gm->output < 1 && gm->token_embd < 1)
		return 0;
	if(gm->attn_norm < gi->n_layers || gm->attn_q < gi->n_layers ||
	   gm->attn_k < gi->n_layers || gm->attn_v < gi->n_layers ||
	   gm->attn_out < gi->n_layers || gm->ffn_norm < gi->n_layers ||
	   gm->ffn_gate < gi->n_layers || gm->ffn_down < gi->n_layers ||
	   gm->ffn_up < gi->n_layers)
		return 0;
	return 1;
}

static int
loadf32tensor(int fd, vlong data_base, GGUFTensorPlan *tp)
{
	if(seekabs(fd, data_base + (vlong)tp->off) < 0)
		return -1;
	return readfull9(fd, tp->dst, (int)(tp->count * sizeof(float)));
}

static int
loadf16tensor(int fd, vlong data_base, GGUFTensorPlan *tp)
{
	uvlong i;
	ushort h;

	if(seekabs(fd, data_base + (vlong)tp->off) < 0)
		return -1;
	for(i = 0; i < tp->count; i++){
		if(readu16(fd, &h) < 0)
			return -1;
		tp->dst[i] = f16tof32(h);
	}
	return 0;
}

static int
loadq40tensor(int fd, vlong data_base, GGUFTensorPlan *tp, char *err, int nerr)
{
	uvlong blocks, b;
	ushort dh;
	uchar qs[QK4_0 / 2];
	float d;
	int i;

	if(tp->count % QK4_0 != 0)
		return -1;
	if(seekabs(fd, data_base + (vlong)tp->off) < 0)
		return -1;
	blocks = tp->count / QK4_0;
	for(b = 0; b < blocks; b++){
		if(readu16(fd, &dh) < 0)
			return -1;
		if(readfull9(fd, qs, sizeof qs) < 0)
			return -1;
		d = f16tof32(dh);
		if(d != d || d > 1.0e6f || d < -1.0e6f){
			if(err != nil && nerr > 0)
				snprint(err, nerr,
					"bad q4_0 scale: tensor=%s block=%llud tensor_off=%llud file_off=%llud raw_half=%#ux d=%g q0=%ux q1=%ux",
					tp->name, b, tp->off, (uvlong)(data_base + (vlong)tp->off + (vlong)b * 18), dh, d, qs[0], qs[1]);
			return -1;
		}
		for(i = 0; i < QK4_0 / 2; i++){
			tp->dst[b * QK4_0 + i] = d * (float)((int)(qs[i] & 0x0f) - 8);
			tp->dst[b * QK4_0 + i + QK4_0 / 2] = d * (float)((int)(qs[i] >> 4) - 8);
		}
	}
	return 0;
}

static int
loadq80tensor(int fd, vlong data_base, GGUFTensorPlan *tp)
{
	uvlong blocks, b;
	ushort dh;
	uchar qs[QK8_0];
	float d;
	int i;
	schar q;

	if(tp->count % QK8_0 != 0)
		return -1;
	if(seekabs(fd, data_base + (vlong)tp->off) < 0)
		return -1;
	blocks = tp->count / QK8_0;
	for(b = 0; b < blocks; b++){
		if(readu16(fd, &dh) < 0)
			return -1;
		if(readfull9(fd, qs, sizeof qs) < 0)
			return -1;
		d = f16tof32(dh);
		for(i = 0; i < QK8_0; i++){
			q = (schar)qs[i];
			tp->dst[b * QK8_0 + i] = d * q;
		}
	}
	return 0;
}

static int
checktensor(GGUFTensorPlan *tp, char *err, int nerr)
{
	uvlong i;
	float v;
	union {
		float f;
		ulong u;
	} vu;

	for(i = 0; i < tp->count; i++){
		v = tp->dst[i];
		if(v != v || v > 1.0e6f || v < -1.0e6f){
			vu.f = v;
			snprint(err, nerr, "bad tensor values: %s idx=%llud val=%g bits=%#lux type=%lud",
				tp->name, i, v, vu.u & (ulong)0xffffffff, tp->type);
			return -1;
		}
	}
	return 0;
}

static int
loadmappedtensors(int fd, vlong data_base, GGUFLoadPlan *gp, char *err, int nerr)
{
	int i;

	for(i = 0; i < gp->n; i++){
		switch(gp->items[i].type){
		case GGMLTypeF32:
			if(loadf32tensor(fd, data_base, &gp->items[i]) < 0)
				return -1;
			break;
		case GGMLTypeF16:
			if(loadf16tensor(fd, data_base, &gp->items[i]) < 0)
				return -1;
			break;
		case GGMLTypeQ4_0:
			if(loadq40tensor(fd, data_base, &gp->items[i], err, nerr) < 0)
				return -1;
			break;
		case GGMLTypeQ8_0:
			if(loadq80tensor(fd, data_base, &gp->items[i]) < 0)
				return -1;
			break;
		default:
			return -1;
		}
		if(checktensor(&gp->items[i], err, nerr) < 0)
			return -1;
	}
	return 0;
}

int
load_model_gguf(Model *m, char *path, char *err, int nerr)
{
	int fd;
	uchar magic[4];
	ulong version, rem;
	vlong data_base;
	Config cfg;
	GGUFInfo gi;
	GGUFMap gm;
	GGUFLoadPlan gp;

	memset(&gi, 0, sizeof gi);
	memset(&gm, 0, sizeof gm);
	memset(&gp, 0, sizeof gp);
	memset(&cfg, 0, sizeof cfg);
	gi.alignment = 32;

	fd = open(path, OREAD);
	if(fd < 0){
		snprint(err, nerr, "open failed: %s", path);
		return -1;
	}
	if(readfull9(fd, magic, 4) < 0){
		snprint(err, nerr, "short read: %s", path);
		close(fd);
		return -1;
	}
	if(magic[0] != 'G' || magic[1] != 'G' || magic[2] != 'U' || magic[3] != 'F'){
		snprint(err, nerr, "bad gguf magic: %s", path);
		close(fd);
		return -1;
	}
	if(readu32(fd, &version) < 0 || readu64(fd, &gi.tensor_count) < 0 || readu64(fd, &gi.kv_count) < 0){
		snprint(err, nerr, "truncated gguf header");
		close(fd);
		return -1;
	}
	gi.version = version;

	if(parse_metadata(fd, &gi) < 0){
		snprint(err, nerr, "failed parsing gguf metadata");
		close(fd);
		return -1;
	}

	if(gi.architecture[0] == 0)
		copystr0(gi.architecture, sizeof gi.architecture, "unknown");
	if(gi.n_kv_heads == 0 && gi.n_heads > 0)
		gi.n_kv_heads = gi.n_heads;

	cfg.vocab_size = gi.vocab_size;
	cfg.dim = gi.dim;
	cfg.hidden_dim = gi.hidden_dim;
	cfg.n_layers = gi.n_layers;
	cfg.n_heads = gi.n_heads;
	cfg.n_kv_heads = gi.n_kv_heads;
	cfg.seq_len = gi.seq_len;
	cfg.rms_eps = gi.rms_eps_kv;
	cfg.rope_freq_base = gi.rope_freq_base;
	if(gi.sliding_window > (uvlong)0x7fffffff)
		cfg.sliding_window = 0x7fffffff;
	else
		cfg.sliding_window = (int)gi.sliding_window;

	cfg.arch = ArchLlama;
	if(strstr(gi.architecture, "qwen2") != nil || strstr(gi.architecture, "Qwen2") != nil){
		cfg.arch = ArchQwen2;
		cfg.rope_type = RopeNeox;
	}else{
		cfg.rope_type = RopeNormal;
	}

	if(alloc_model(m, &cfg, err, nerr) < 0){
		free_vocab_gi(&gi);
		close(fd);
		return -1;
	}
	m->loader_kind = LoaderGGUF;
	m->token_str = gi.vocab_tokens;
	gi.vocab_tokens = nil;
	gi.vocab_tokens_n = 0;

	gp.cap = 3 + cfg.n_layers * 9;
	gp.items = mallocz(gp.cap * sizeof(GGUFTensorPlan), 1);
	if(gp.items == nil){
		snprint(err, nerr, "mallocz failed");
		close(fd);
		free_model(m);
		return -1;
	}

	if(parse_tensor_infos(fd, gi.tensor_count, m, &gm, &gp) < 0){
		snprint(err, nerr, "failed parsing gguf tensor table");
		close(fd);
		free(gp.items);
		free_model(m);
		return -1;
	}

	if(!hasrequired(&gi, &gm)){
		snprint(err, nerr,
			"gguf parsed but tensor mapping incomplete: arch=%s layers=%llud token_embd=%d output_norm=%d output=%d attn_norm=%d attn_q=%d attn_k=%d attn_v=%d attn_out=%d ffn_norm=%d ffn_gate=%d ffn_down=%d ffn_up=%d unknown=%d",
			gi.architecture, gi.n_layers, gm.token_embd, gm.output_norm, gm.output,
			gm.attn_norm, gm.attn_q, gm.attn_k, gm.attn_v, gm.attn_out,
			gm.ffn_norm, gm.ffn_gate, gm.ffn_down, gm.ffn_up, gm.unknown);
		close(fd);
		free(gp.items);
		free_model(m);
		return -1;
	}

	if(gm.unsupported > 0){
		snprint(err, nerr,
			"gguf mapped: arch=%s version=%lud tensors=%llud layers=%llud dim=%llud heads=%llud kv_heads=%llud vocab=%llud ctx=%llud tied_output=%d; unsupported ggml tensor type=%lud tensor=%s",
			gi.architecture, gi.version, gi.tensor_count, gi.n_layers, gi.dim,
			gi.n_heads, gi.n_kv_heads, gi.vocab_size, gi.seq_len, gm.output < 1, gm.first_type, gm.first_name);
		close(fd);
		free(gp.items);
		free_model(m);
		return -1;
	}

	data_base = seek(fd, 0, 1);
	if(data_base < 0){
		snprint(err, nerr, "seek failed after tensor table");
		close(fd);
		free(gp.items);
		free_model(m);
		return -1;
	}
	if(gi.alignment > 1){
		rem = data_base % gi.alignment;
		if(rem != 0)
			data_base += gi.alignment - rem;
	}

	if(gguf_f16_selftest(err, nerr) < 0){
		close(fd);
		free(gp.items);
		free_model(m);
		return -1;
	}

	if(loadmappedtensors(fd, data_base, &gp, err, nerr) < 0){
		if(err[0] == 0)
			snprint(err, nerr, "failed loading mapped gguf tensors");
		close(fd);
		free(gp.items);
		free_model(m);
		return -1;
	}
	close(fd);

	if(gm.output < 1){
		memmove(m->wcls, m->token_embedding_table, cfg.vocab_size * cfg.dim * sizeof(float));
		gp.tied_output = 1;
	}

	free(gp.items);
	snprint(err, nerr,
		"gguf loaded: arch=%s version=%lud tensors=%llud kv=%llud dim=%llud layers=%llud heads=%llud kv_heads=%llud vocab=%llud ctx=%llud tied_output=%d q4_0=%d q8_0=%d",
		gi.architecture, gi.version, gi.tensor_count, gi.kv_count, gi.dim,
		gi.n_layers, gi.n_heads, gi.n_kv_heads, gi.vocab_size, gi.seq_len, gp.tied_output, gm.q4_0, gm.q8_0);
	return 0;
}
