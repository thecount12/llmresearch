#include "common.h"
#include "model.h"

static void
usage(void)
{
	fprint(2, "usage: %s [-m model.bin] [-c ctx] [-n steps] [-p prompt] [-P tokfile] [-B] [-t temp] [-s seed] [-v] [-g] [-e] [-a] [-F]\n", argv0);
	fprint(2, "       -c  max context length (KV cache / attention); default caps large GGUF seq_len to 4096; -c 0 = use model max\n");
	fprint(2, "       -p  each byte is a token id (toy / byte-vocab only); for GGUF BPE models use encode_prompt_hf.py → -P\n");
	fprint(2, "       -P  file of prompt token ids (overrides -p); default: ASCII integers; with -B: binary int32 LE\n");
	fprint(2, "       -B  with -P: read int32 little-endian (4 bytes per id), not text\n");
	fprint(2, "       -s  seed RNG for temperature sampling (Plan 9 nrand/srand)\n");
	fprint(2, "       -v  verbose (config / loader on stderr)\n");
	fprint(2, "       -g  print top logits each generation step (stderr; before sampling)\n");
	fprint(2, "       -e  print each greedy token id (and piece string) to stderr for comparison with HF/llama.cpp\n");
	fprint(2, "       -F  after prompt: print pre-mask logits fingerprint (greedy, sumsq, cksum, top-5) on stderr; use hf_logits_ref.py on host\n");
	fprint(2, "       -a  pretty print: map common HF-style token strings (e.g. Ġ→space, Ċ→newline)\n");
	fprint(2, "            token pieces are buffered so UTF-8 bytes split across tokens decode correctly\n");
	fprint(2, "       model path must be passed with -m (first arg alone is not the file)\n");
	fprint(2, "       no -m means use the built-in toy model\n");
	exits("usage");
}

static int
readfull_local(int fd, void *buf, int n)
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

/*
 * Read entire file (max ~1MB); caller must free. Returns nil on error.
 */
static char *
read_file_all(char *path, char *err, int nerr)
{
	int fd;
	vlong len;
	char *buf;

	fd = open(path, OREAD);
	if(fd < 0){
		snprint(err, nerr, "open %s failed", path);
		return nil;
	}
	len = seek(fd, 0, 2);
	if(len < 0){
		snprint(err, nerr, "seek %s failed", path);
		close(fd);
		return nil;
	}
	if(len > 1<<20){
		snprint(err, nerr, "file too large: %s", path);
		close(fd);
		return nil;
	}
	if(seek(fd, 0, 0) < 0){
		snprint(err, nerr, "rewind %s failed", path);
		close(fd);
		return nil;
	}
	buf = malloc((ulong)len + 1);
	if(buf == nil){
		snprint(err, nerr, "malloc failed");
		close(fd);
		return nil;
	}
	if(readfull_local(fd, buf, (int)len) < 0){
		snprint(err, nerr, "read %s failed", path);
		free(buf);
		close(fd);
		return nil;
	}
	close(fd);
	buf[len] = 0;
	return buf;
}

/*
 * Parse integers from buf into ids[0..maxn-1]. Lines starting with # are skipped.
 * Returns count, or -1 on syntax error.
 */
static int
parse_prompt_ids(char *buf, int *ids, int maxn, char *err, int nerr)
{
	char *p;
	int n, v, sign, d;

	n = 0;
	p = buf;
	while(*p){
		while(*p == ' ' || *p == '\t' || *p == '\n' || *p == '\r' || *p == ',')
			p++;
		if(*p == 0)
			break;
		if(*p == '#'){
			while(*p && *p != '\n')
				p++;
			continue;
		}
		sign = 1;
		if(*p == '-'){
			sign = -1;
			p++;
		}
		if(*p < '0' || *p > '9'){
			snprint(err, nerr, "bad token id near: %.40s", p);
			return -1;
		}
		v = 0;
		while(*p >= '0' && *p <= '9'){
			d = *p - '0';
			if(v > (2147483647 - d) / 10){
				snprint(err, nerr, "token id overflow");
				return -1;
			}
			v = v * 10 + d;
			p++;
		}
		if(n >= maxn){
			snprint(err, nerr, "too many prompt ids (max %d)", maxn);
			return -1;
		}
		ids[n++] = sign * v;
	}
	return n;
}

static int
load_i32_le(uchar *p)
{
	unsigned u;

	u = (unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24);
	return (int)u;
}

/*
 * Binary prompt file: raw int32 little-endian, 4 bytes per id. Returns count or -1.
 */
static int
load_prompt_ids_binary(char *path, int *ids, int maxn, char *err, int nerr)
{
	int fd, i, n;
	vlong len;
	uchar *buf;

	fd = open(path, OREAD);
	if(fd < 0){
		snprint(err, nerr, "open %s failed", path);
		return -1;
	}
	len = seek(fd, 0, 2);
	if(len < 0){
		snprint(err, nerr, "seek %s failed", path);
		close(fd);
		return -1;
	}
	if(len == 0){
		close(fd);
		return 0;
	}
	if(len % 4 != 0){
		snprint(err, nerr, "binary -P file size must be a multiple of 4 (int32 LE)");
		close(fd);
		return -1;
	}
	n = len / 4;
	if(n > maxn){
		snprint(err, nerr, "too many prompt ids (max %d)", maxn);
		close(fd);
		return -1;
	}
	if(seek(fd, 0, 0) < 0){
		snprint(err, nerr, "rewind %s failed", path);
		close(fd);
		return -1;
	}
	buf = malloc((ulong)len);
	if(buf == nil){
		snprint(err, nerr, "malloc failed");
		close(fd);
		return -1;
	}
	if(readfull_local(fd, buf, (int)len) < 0){
		snprint(err, nerr, "read %s failed", path);
		free(buf);
		close(fd);
		return -1;
	}
	close(fd);
	for(i = 0; i < n; i++)
		ids[i] = load_i32_le(buf + i * 4);
	free(buf);
	return n;
}

static int
clamp_token(int token, int vocab_size)
{
	if(token < 0)
		return 0;
	if(token < vocab_size)
		return token;
	return token % vocab_size;
}

static void
validate_prompt_ids(int *ids, int n, int vocab_size, char *path)
{
	int i;

	for(i = 0; i < n; i++)
		if(ids[i] < 0 || ids[i] >= vocab_size)
			sysfatal("prompt id %d out of range [0,%d): %d (file %s)",
				i, vocab_size, ids[i], path);
}

/*
 * Bounded UTF-8: 0 = one complete character of *nout bytes; -1 = need more
 * bytes; -2 = invalid lead/continuation (caller should drop 1 byte).
 */
static int
utf8_peek_len(uchar *p, int len, int *nout)
{
	uchar c;
	ulong cp;

	if(len <= 0)
		return -1;
	c = p[0];
	if(c < 0x80){
		*nout = 1;
		return 0;
	}
	if((c & 0xe0) == 0xc0){
		if(len < 2)
			return -1;
		if((p[1] & 0xc0) != 0x80)
			return -2;
		cp = ((ulong)(c & 0x1f) << 6) | (ulong)(p[1] & 0x3f);
		if(cp < 0x80)
			return -2;
		*nout = 2;
		return 0;
	}
	if((c & 0xf0) == 0xe0){
		if(len < 3)
			return -1;
		if((p[1] & 0xc0) != 0x80 || (p[2] & 0xc0) != 0x80)
			return -2;
		cp = ((ulong)(c & 0x0f) << 12) | ((ulong)(p[1] & 0x3f) << 6) | (ulong)(p[2] & 0x3f);
		if(cp < 0x800 || (cp >= 0xd800 && cp <= 0xdfff))
			return -2;
		*nout = 3;
		return 0;
	}
	if((c & 0xf8) == 0xf0){
		if(len < 4)
			return -1;
		if((p[1] & 0xc0) != 0x80 || (p[2] & 0xc0) != 0x80 || (p[3] & 0xc0) != 0x80)
			return -2;
		cp = ((ulong)(c & 0x07) << 18) | ((ulong)(p[1] & 0x3f) << 12)
			| ((ulong)(p[2] & 0x3f) << 6) | (ulong)(p[3] & 0x3f);
		if(cp < 0x10000 || cp > 0x10ffff || (cp >= 0xd800 && cp <= 0xdfff))
			return -2;
		*nout = 4;
		return 0;
	}
	return -2;
}

enum {
	EmitUtf8Max = 65536
};

static uchar emit_utf8_buf[EmitUtf8Max];
static int emit_utf8_len;

/*
 * HuggingFace BPE: map common space/newline codepoints to ASCII when -a.
 * q points to one complete UTF-8 codepoint of n bytes (1..4).
 */
static void
emit_one_codepoint_pretty(uchar *q, int n)
{
	if(n == 2 && q[0] == 0xc4 && q[1] == 0xa0){
		fprint(1, " ");
		return;
	}
	if(n == 2 && q[0] == 0xc4 && q[1] == 0x8a){
		fprint(1, "\n");
		return;
	}
	if(n == 3 && q[0] == 0xe2 && q[1] == 0x96 && q[2] == 0x81){
		fprint(1, " ");
		return;
	}
	if(n == 2 && q[0] == 0xc2 && q[1] == 0xa0){
		fprint(1, " ");
		return;
	}
	if(n == 1 && q[0] < 0x80){
		fprint(1, "%c", q[0]);
		return;
	}
	write(1, q, n);
}

static void
emit_utf8_drain(int pretty)
{
	int n;
	int r;

	for(;;){
		if(emit_utf8_len <= 0)
			return;
		r = utf8_peek_len(emit_utf8_buf, emit_utf8_len, &n);
		if(r == -1)
			return;
		if(r == -2){
			fprint(1, "%c", emit_utf8_buf[0]);
			memmove(emit_utf8_buf, emit_utf8_buf + 1, emit_utf8_len - 1);
			emit_utf8_len--;
			continue;
		}
		if(pretty)
			emit_one_codepoint_pretty(emit_utf8_buf, n);
		else
			write(1, emit_utf8_buf, n);
		memmove(emit_utf8_buf, emit_utf8_buf + n, emit_utf8_len - n);
		emit_utf8_len -= n;
	}
}

static void
emit_utf8_append_piece(char *s, int pretty)
{
	int l;
	int cap;

	if(s == nil)
		return;
	l = strlen(s);
	cap = sizeof emit_utf8_buf;
	while(emit_utf8_len + l > cap){
		emit_utf8_drain(pretty);
		if(emit_utf8_len + l > cap){
			fprint(2, "lumen: warning: UTF-8 emit buffer full; truncating piece\n");
			l = cap - emit_utf8_len;
			if(l <= 0)
				return;
		}
	}
	memmove(emit_utf8_buf + emit_utf8_len, s, l);
	emit_utf8_len += l;
	emit_utf8_drain(pretty);
}

/* Flush incomplete UTF-8 tail as raw bytes (e.g. before fallback emit or at EOF). */
static void
emit_utf8_flush_tail_raw(void)
{
	if(emit_utf8_len <= 0)
		return;
	write(1, emit_utf8_buf, emit_utf8_len);
	emit_utf8_len = 0;
}

static void
emit_utf8_before_fallback(int pretty)
{
	emit_utf8_drain(pretty);
	emit_utf8_flush_tail_raw();
}

static void
emit_utf8_finish(int pretty)
{
	emit_utf8_drain(pretty);
	emit_utf8_flush_tail_raw();
}

static void
emit_token(Model *m, int token, int pretty)
{
	if(m->token_str != nil && token >= 0 && token < m->cfg.vocab_size
	    && m->token_str[token] != nil){
		emit_utf8_append_piece(m->token_str[token], pretty);
		return;
	}
	emit_utf8_before_fallback(pretty);
	if(token >= 32 && token < 127){
		fprint(1, "%c", token);
		return;
	}
	if(token == '\n' || token == '\t'){
		fprint(1, "%c", token);
		return;
	}
	fprint(1, "<%d>", token);
}

void
main(int argc, char **argv)
{
	Model model;
	RunState state;
	char err[512];
	char *model_path, *prompt, *prompt_file;
	int *prompt_ids;
	int steps, pos, i, token, next, promptlen, n_prompt_ids, prompt_tok_count, nstr, verbose, dump_logits, emit_ids, hf_fingerprint, pretty, bin_prompt, have_seed;
	int cap_ctx;	/* -1 = default policy; 0 = full model seq_len; >0 = cap */
	int orig_seq;
	ulong seed;
	float temperature;

	memset(&model, 0, sizeof(model));
	memset(&state, 0, sizeof(state));
	model_path = nil;
	prompt = "";
	prompt_file = nil;
	prompt_ids = nil;
	n_prompt_ids = 0;
	steps = 32;
	temperature = 0.0f;
	verbose = 0;
	dump_logits = 0;
	emit_ids = 0;
	hf_fingerprint = 0;
	pretty = 0;
	bin_prompt = 0;
	have_seed = 0;
	seed = 0;
	cap_ctx = -1;

	ARGBEGIN{
	case 'm':
		model_path = EARGF(usage());
		break;
	case 'c':
		cap_ctx = atoi(EARGF(usage()));
		break;
	case 'n':
		steps = atoi(EARGF(usage()));
		break;
	case 'p':
		prompt = EARGF(usage());
		break;
	case 'P':
		prompt_file = EARGF(usage());
		break;
	case 'B':
		bin_prompt = 1;
		break;
	case 't':
		temperature = atof(EARGF(usage()));
		break;
	case 's':
		seed = strtoul(EARGF(usage()), nil, 0);
		have_seed = 1;
		break;
	case 'v':
		verbose = 1;
		break;
	case 'g':
		dump_logits = 1;
		break;
	case 'e':
		emit_ids = 1;
		break;
	case 'F':
		hf_fingerprint = 1;
		break;
	case 'a':
		pretty = 1;
		break;
	default:
		usage();
	}ARGEND

	if(bin_prompt && prompt_file == nil)
		sysfatal("-B requires -P");

	if(model_path != nil){
		if(load_model_auto(&model, model_path, err, sizeof err) < 0)
			sysfatal("%s", err);
	}else{
		if(init_toy_model(&model, err, sizeof err) < 0)
			sysfatal("%s", err);
	}

	orig_seq = model.cfg.seq_len;
	if(cap_ctx == -1){
		if(orig_seq > 4096)
			model.cfg.seq_len = 4096;
	}else if(cap_ctx == 0)
		model.cfg.seq_len = orig_seq;
	else if(cap_ctx > 0){
		if(cap_ctx > orig_seq)
			model.cfg.seq_len = orig_seq;
		else
			model.cfg.seq_len = cap_ctx;
	}else
		usage();

	if(verbose && model.cfg.seq_len != orig_seq)
		fprint(2, "seq_len: using %d (model metadata %d; -c 0 for full, -c N to set)\n",
			model.cfg.seq_len, orig_seq);

	if(have_seed)
		sampler_seed(seed);

	if(prompt_file != nil){
		char *pbuf;

		prompt_ids = malloc(model.cfg.seq_len * sizeof(int));
		if(prompt_ids == nil)
			sysfatal("malloc failed");
		if(bin_prompt){
			n_prompt_ids = load_prompt_ids_binary(prompt_file, prompt_ids, model.cfg.seq_len, err, sizeof err);
			if(n_prompt_ids < 0)
				sysfatal("%s", err);
		}else{
			pbuf = read_file_all(prompt_file, err, sizeof err);
			if(pbuf == nil)
				sysfatal("%s", err);
			n_prompt_ids = parse_prompt_ids(pbuf, prompt_ids, model.cfg.seq_len, err, sizeof err);
			free(pbuf);
			if(n_prompt_ids < 0)
				sysfatal("%s", err);
		}
		validate_prompt_ids(prompt_ids, n_prompt_ids, model.cfg.vocab_size, prompt_file);
	}

	if(verbose){
		fprint(2, "loader_kind=%d arch=%d vocab=%d dim=%d layers=%d heads=%d kv_heads=%d seq_len=%d\n",
			model.loader_kind, model.cfg.arch, model.cfg.vocab_size, model.cfg.dim,
			model.cfg.n_layers, model.cfg.n_heads, model.cfg.n_kv_heads, model.cfg.seq_len);
		fprint(2, "rope_freq_base=%g rms_eps=%g sliding_window=%d rope_type=%d\n",
			model.cfg.rope_freq_base, model.cfg.rms_eps, model.cfg.sliding_window,
			model.cfg.rope_type);
		if(model.token_str != nil){
			for(i = 0, nstr = 0; i < model.cfg.vocab_size; i++)
				if(model.token_str[i] != nil)
					nstr++;
			fprint(2, "token_str entries: %d / %d\n", nstr, model.cfg.vocab_size);
		}else
			fprint(2, "token_str: (nil)\n");
		if(prompt_file != nil){
			fprint(2, "prompt: %d token ids from %s%s (-P overrides -p)\n", n_prompt_ids, prompt_file,
				bin_prompt ? " (binary int32 LE)" : " (ASCII)");
			if(n_prompt_ids > 0 && model.token_str != nil){
				int nshow;

				nshow = n_prompt_ids;
				if(nshow > 6)
					nshow = 6;
				for(i = 0; i < nshow; i++){
					int id;

					id = prompt_ids[i];
					if(id >= 0 && id < model.cfg.vocab_size && model.token_str[id] != nil)
						fprint(2, "  prompt[%d] id=%d piece=%s\n", i, id, model.token_str[id]);
					else
						fprint(2, "  prompt[%d] id=%d piece=(no string)\n", i, id);
				}
			}
		}
	}

	if(prompt_file == nil && model.loader_kind == LoaderGGUF && strlen(prompt) > 0)
		fprint(2, "lumen: warning: -p uses raw bytes as token ids; for Qwen/Llama GGUF use host `encode_prompt_hf.py` and -P\n");
	if(prompt_file != nil && model.loader_kind == LoaderGGUF)
		fprint(2, "lumen: note: ids in -P must come from the same tokenizer as this GGUF (e.g. encode_prompt_hf.py -m Qwen/Qwen2.5-0.5B-Instruct for Qwen2.5 GGUF)\n");
	if(model.loader_kind == LoaderGGUF && model.token_str != nil)
		fprint(2, "lumen: note: stdout is UTF-8 pieces from GGUF; if decoded text looks wrong on any UTF-8 viewer, compare token ids with a reference (e.g. llama.cpp) using -e\n");

	if(prompt_file != nil && n_prompt_ids + steps > model.cfg.seq_len)
		sysfatal("prompt (%d tok) + steps (%d) exceeds seq_len=%d; raise -c or shorten -n",
			n_prompt_ids, steps, model.cfg.seq_len);
	if(prompt_file == nil){
		promptlen = strlen(prompt);
		if((promptlen > 0 ? promptlen : 1) + steps > model.cfg.seq_len)
			sysfatal("prompt + steps exceeds seq_len=%d; raise -c or shorten -n", model.cfg.seq_len);
	}

	if(alloc_run_state(&state, &model.cfg, err, sizeof err) < 0)
		sysfatal("%s", err);
	clear_kv_cache(&state, &model.cfg);

	pos = 0;
	token = clamp_token(' ', model.cfg.vocab_size);
	if(prompt_file != nil){
		if(n_prompt_ids > 0){
			for(i = 0; i < n_prompt_ids && pos < model.cfg.seq_len; i++){
				token = prompt_ids[i];
				if(transformer_forward(&model, &state, token, pos, err, sizeof err) < 0)
					sysfatal("forward failed at prompt token %d", pos);
				pos++;
			}
		}else{
			if(transformer_forward(&model, &state, token, pos, err, sizeof err) < 0)
				sysfatal("forward failed");
			pos++;
		}
	}else{
		promptlen = strlen(prompt);
		for(i = 0; i < promptlen && pos < model.cfg.seq_len; i++){
			token = clamp_token((uchar)prompt[i], model.cfg.vocab_size);
			if(transformer_forward(&model, &state, token, pos, err, sizeof err) < 0)
				sysfatal("forward failed at prompt token %d", pos);
			pos++;
		}
		if(promptlen == 0){
			if(transformer_forward(&model, &state, token, pos, err, sizeof err) < 0)
				sysfatal("forward failed");
			pos++;
		}
	}

	prompt_tok_count = pos; /* tokens fed before first gen step (-P or -p) */
	if(hf_fingerprint){
		fprint(2, "lumen_hf: after_prompt prompt_tok=%d next-token logits (pre_mask, before EOG mask)\n",
			prompt_tok_count);
		hf_logits_fingerprint(2, state.logits, model.cfg.vocab_size);
	}
	for(i = 0; i < steps && pos < model.cfg.seq_len; i++){
		if(dump_logits){
			/* Leading \\n so stderr lines do not glue to stdout tokens on one tty line. */
			fprint(2, "\nlogits top-8 step %d (pre-mask):", i);
			dump_logits_topk(2, state.logits, model.cfg.vocab_size, 8);
		}
		/*
		 * HF cross-check: greedy next id 19482 (Ġsyntax) only for single-token prompt [14990].
		 * With multiple -P ids, compare logits to HF for that exact prefix instead.
		 * Print pre-mask logits and unmasked argmax; then apply EOG mask like llama.cpp.
		 */
		if(emit_ids && i == 0 && model.cfg.vocab_size > 131306){
			int um;

			um = greedy_sample(state.logits, model.cfg.vocab_size);
			fprint(2, "step0 pre-mask: logit[19482]=%g logit[131306]=%g unmasked_argmax=%d (logit=%g)\n",
				state.logits[19482], state.logits[131306], um, state.logits[um]);
		}
		decode_logits_mask(&model, state.logits);
		if(emit_ids && i == 0 && model.cfg.vocab_size > 19482){
			if(prompt_tok_count == 1)
				fprint(2, "step0 post-mask: logit[19482]=%g (HF greedy-next ref for prompt [14990] only; unchanged if not EOG-masked)\n",
					state.logits[19482]);
			else
				fprint(2, "step0 post-mask: logit[19482]=%g (not the HF [14990]→19482 ref; compare HF on this full prompt prefix)\n",
					state.logits[19482]);
		}
		if(temperature > 0.0f)
			next = sample_with_temperature(state.logits, model.cfg.vocab_size, temperature);
		else
			next = greedy_sample(state.logits, model.cfg.vocab_size);
		if(emit_ids && i == 0)
			fprint(2, "step0 post-mask: greedy id=%d logit=%g\n", next, state.logits[next]);
		if(emit_ids){
			fprint(2, "gen[%d] id=%d", i, next);
			if(model.token_str != nil && next >= 0 && next < model.cfg.vocab_size
			    && model.token_str[next] != nil)
				fprint(2, " piece=%s", model.token_str[next]);
			fprint(2, "\n");
		}
		emit_token(&model, next, pretty);
		token = next;
		if(transformer_forward(&model, &state, token, pos, err, sizeof err) < 0)
			sysfatal("forward failed at generation step %d", i);
		pos++;
	}
	emit_utf8_finish(pretty);
	fprint(1, "\n");

	free_run_state(&state);
	free(prompt_ids);
	free_model(&model);
	exits(nil);
}
