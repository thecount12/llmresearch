#include "common.h"
#include "model.h"

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

typedef struct GGUFInfo GGUFInfo;
typedef struct GGUFMap GGUFMap;
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
	int quantized;
	ulong first_type;
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
		return readf32(fd, &f32);
	case GGUFUint64:
	case GGUFInt64:
		if(readu64(fd, &u64) < 0)
			return -1;
		if(strcmp(key, "tokenizer.ggml.tokens") == 0)
			gi->vocab_size = u64;
		setinfo_u64(gi, key, u64);
		return 0;
	case GGUFFloat64:
		return readf64(fd, &f64);
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

			if(readu32(fd, &elem_type) < 0)
				return -1;
			if(readu64(fd, &n) < 0)
				return -1;
			gi->vocab_size = n;
			for(i = 0; i < n; i++)
				if(skipvalue(fd, elem_type) < 0)
					return -1;
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
		if(readstr(fd, &key) < 0)
			return -1;
		if(readu32(fd, &type) < 0){
			free(key);
			return -1;
		}
		if(parse_metadata_value(fd, gi, key, type) < 0){
			free(key);
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
	int n, want;

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
	want = strlen(suffix);
	if(strcmp(q, suffix) == 0)
		return n;
	return -1;
}

static int
istensorf32(ulong ggml_type)
{
	return ggml_type == 0;
}

static void
updatemap(GGUFMap *gm, char *name, ulong ggml_type)
{
	if(!istensorf32(ggml_type)){
		gm->quantized++;
		if(gm->first_type == 0)
			gm->first_type = ggml_type;
	}

	if(strcmp(name, "token_embd.weight") == 0){
		gm->token_embd++;
		return;
	}
	if(strcmp(name, "output_norm.weight") == 0){
		gm->output_norm++;
		return;
	}
	if(strcmp(name, "output.weight") == 0){
		gm->output++;
		return;
	}
	if(parseblklayer(name, "attn_norm.weight") >= 0){
		gm->attn_norm++;
		return;
	}
	if(parseblklayer(name, "attn_q.weight") >= 0){
		gm->attn_q++;
		return;
	}
	if(parseblklayer(name, "attn_k.weight") >= 0){
		gm->attn_k++;
		return;
	}
	if(parseblklayer(name, "attn_v.weight") >= 0){
		gm->attn_v++;
		return;
	}
	if(parseblklayer(name, "attn_output.weight") >= 0){
		gm->attn_out++;
		return;
	}
	if(parseblklayer(name, "ffn_norm.weight") >= 0){
		gm->ffn_norm++;
		return;
	}
	if(parseblklayer(name, "ffn_gate.weight") >= 0){
		gm->ffn_gate++;
		return;
	}
	if(parseblklayer(name, "ffn_down.weight") >= 0){
		gm->ffn_down++;
		return;
	}
	if(parseblklayer(name, "ffn_up.weight") >= 0){
		gm->ffn_up++;
		return;
	}
	gm->unknown++;
}

static int
parse_tensor_infos(int fd, uvlong n, GGUFMap *gm)
{
	uvlong i, j, dimv, off;
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
		for(j = 0; j < ndims; j++){
			if(readu64(fd, &dimv) < 0){
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
		updatemap(gm, name, ggml_type);
		free(name);
	}
	return 0;
}

static int
hasrequired(GGUFInfo *gi, GGUFMap *gm)
{
	if(gm->token_embd < 1 || gm->output_norm < 1 || gm->output < 1)
		return 0;
	if(gm->attn_norm < gi->n_layers || gm->attn_q < gi->n_layers ||
	   gm->attn_k < gi->n_layers || gm->attn_v < gi->n_layers ||
	   gm->attn_out < gi->n_layers || gm->ffn_norm < gi->n_layers ||
	   gm->ffn_gate < gi->n_layers || gm->ffn_down < gi->n_layers ||
	   gm->ffn_up < gi->n_layers)
		return 0;
	return 1;
}

int
load_model_gguf(Model *m, char *path, char *err, int nerr)
{
	int fd;
	uchar magic[4];
	ulong version;
	GGUFInfo gi;
	GGUFMap gm;

	USED(m);

	memset(&gi, 0, sizeof gi);
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
	if(parse_tensor_infos(fd, gi.tensor_count, &gm) < 0){
		snprint(err, nerr, "failed parsing gguf tensor table");
		close(fd);
		return -1;
	}
	close(fd);

	if(gi.architecture[0] == 0)
		copystr0(gi.architecture, sizeof gi.architecture, "unknown");
	if(gi.vocab_size == 0)
		gi.vocab_size = 0;
	if(gi.n_kv_heads == 0 && gi.n_heads > 0)
		gi.n_kv_heads = gi.n_heads;

	if(!hasrequired(&gi, &gm)){
		snprint(err, nerr,
			"gguf parsed but tensor mapping incomplete: arch=%s layers=%llud token_embd=%d output_norm=%d output=%d attn_norm=%d attn_q=%d attn_k=%d attn_v=%d attn_out=%d ffn_norm=%d ffn_gate=%d ffn_down=%d ffn_up=%d unknown=%d",
			gi.architecture, gi.n_layers, gm.token_embd, gm.output_norm, gm.output,
			gm.attn_norm, gm.attn_q, gm.attn_k, gm.attn_v, gm.attn_out,
			gm.ffn_norm, gm.ffn_gate, gm.ffn_down, gm.ffn_up, gm.unknown);
		return -1;
	}
	if(gm.quantized > 0){
		snprint(err, nerr,
			"gguf mapped: arch=%s version=%lud tensors=%llud layers=%llud dim=%llud heads=%llud kv_heads=%llud vocab=%llud ctx=%llud; quantized tensor loading not implemented (first ggml_type=%lud, mapped layers ok)",
			gi.architecture, gi.version, gi.tensor_count, gi.n_layers, gi.dim,
			gi.n_heads, gi.n_kv_heads, gi.vocab_size, gi.seq_len, gm.first_type);
		return -1;
	}

	snprint(err, nerr,
		"gguf mapped: arch=%s version=%lud tensors=%llud kv=%llud dim=%llud layers=%llud heads=%llud kv_heads=%llud vocab=%llud ctx=%llud; float32 tensor loading not implemented",
		gi.architecture, gi.version, gi.tensor_count, gi.kv_count, gi.dim,
		gi.n_layers, gi.n_heads, gi.n_kv_heads, gi.vocab_size, gi.seq_len);
	return -1;
}
