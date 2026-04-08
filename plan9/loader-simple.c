#include "model.h"

typedef struct SimpleHeader SimpleHeader;
struct SimpleHeader {
	uchar magic[4];
	uint32 version;
	uint32 vocab_size;
	uint32 dim;
	uint32 hidden_dim;
	uint32 n_layers;
	uint32 n_heads;
	uint32 n_kv_heads;
	uint32 seq_len;
	float rms_eps;
};

static int
kv_dim(Config *cfg)
{
	return (cfg->dim / cfg->n_heads) * cfg->n_kv_heads;
}

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
read_tensor(int fd, float *dst, int n)
{
	return readfull9(fd, dst, n * sizeof(float));
}

int
load_model_simple(Model *m, char *path, char *err, int nerr)
{
	SimpleHeader hdr;
	Config cfg;
	int fd, i, kdim;
	LayerWeights *lw;

	fd = open(path, OREAD);
	if(fd < 0){
		snprint(err, nerr, "open failed: %s", path);
		return -1;
	}

	if(readfull9(fd, &hdr, sizeof(hdr)) < 0){
		snprint(err, nerr, "short read: %s", path);
		close(fd);
		return -1;
	}
	if(hdr.magic[0] != 'P' || hdr.magic[1] != '9' || hdr.magic[2] != 'D' || hdr.magic[3] != 'M'){
		snprint(err, nerr, "bad magic: %s", path);
		close(fd);
		return -1;
	}
	if(hdr.version != 1){
		snprint(err, nerr, "unsupported version");
		close(fd);
		return -1;
	}

	memset(&cfg, 0, sizeof(cfg));
	cfg.vocab_size = hdr.vocab_size;
	cfg.dim = hdr.dim;
	cfg.hidden_dim = hdr.hidden_dim;
	cfg.n_layers = hdr.n_layers;
	cfg.n_heads = hdr.n_heads;
	cfg.n_kv_heads = hdr.n_kv_heads;
	cfg.seq_len = hdr.seq_len;
	cfg.rms_eps = hdr.rms_eps;

	if(alloc_model(m, &cfg, err, nerr) < 0){
		close(fd);
		return -1;
	}

	kdim = kv_dim(&cfg);
	if(read_tensor(fd, m->token_embedding_table, cfg.vocab_size * cfg.dim) < 0)
		goto Short;
	for(i = 0; i < cfg.n_layers; i++){
		lw = &m->layers[i];
		if(read_tensor(fd, lw->rms_att_weight, cfg.dim) < 0) goto Short;
		if(read_tensor(fd, lw->wq, cfg.dim * cfg.dim) < 0) goto Short;
		if(read_tensor(fd, lw->wk, kdim * cfg.dim) < 0) goto Short;
		if(read_tensor(fd, lw->wv, kdim * cfg.dim) < 0) goto Short;
		if(read_tensor(fd, lw->wo, cfg.dim * cfg.dim) < 0) goto Short;
		if(read_tensor(fd, lw->rms_ffn_weight, cfg.dim) < 0) goto Short;
		if(read_tensor(fd, lw->w1, cfg.hidden_dim * cfg.dim) < 0) goto Short;
		if(read_tensor(fd, lw->w2, cfg.dim * cfg.hidden_dim) < 0) goto Short;
		if(read_tensor(fd, lw->w3, cfg.hidden_dim * cfg.dim) < 0) goto Short;
	}
	if(read_tensor(fd, m->rms_final_weight, cfg.dim) < 0) goto Short;
	if(read_tensor(fd, m->wcls, cfg.vocab_size * cfg.dim) < 0) goto Short;

	close(fd);
	return 0;

Short:
	snprint(err, nerr, "truncated tensor data");
	close(fd);
	free_model(m);
	return -1;
}
