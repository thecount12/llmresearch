#include "common.h"
#include "model.h"

static void
usage(void)
{
	fprint(2, "usage: %s [-m model.bin] [-n steps] [-p prompt] [-t temp] [-v] [-g]\n", argv0);
	fprint(2, "       -v  verbose (config / loader on stderr)\n");
	fprint(2, "       -g  print top logits each generation step (stderr; before sampling)\n");
	fprint(2, "       model path must be passed with -m (first arg alone is not the file)\n");
	fprint(2, "       no -m means use the built-in toy model\n");
	exits("usage");
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
emit_token(Model *m, int token)
{
	if(m->token_str != nil && token >= 0 && token < m->cfg.vocab_size
	    && m->token_str[token] != nil){
		fprint(1, "%s", m->token_str[token]);
		return;
	}
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
	char *model_path, *prompt;
	int steps, pos, i, token, next, promptlen, nstr, verbose, dump_logits;
	float temperature;

	memset(&model, 0, sizeof(model));
	memset(&state, 0, sizeof(state));
	model_path = nil;
	prompt = "";
	steps = 32;
	temperature = 0.0f;
	verbose = 0;
	dump_logits = 0;

	ARGBEGIN{
	case 'm':
		model_path = EARGF(usage());
		break;
	case 'n':
		steps = atoi(EARGF(usage()));
		break;
	case 'p':
		prompt = EARGF(usage());
		break;
	case 't':
		temperature = atof(EARGF(usage()));
		break;
	case 'v':
		verbose = 1;
		break;
	case 'g':
		dump_logits = 1;
		break;
	default:
		usage();
	}ARGEND

	if(model_path != nil){
		if(load_model_auto(&model, model_path, err, sizeof err) < 0)
			sysfatal("%s", err);
	}else{
		if(init_toy_model(&model, err, sizeof err) < 0)
			sysfatal("%s", err);
	}

	if(verbose){
		fprint(2, "loader_kind=%d vocab=%d dim=%d layers=%d heads=%d kv_heads=%d seq_len=%d\n",
			model.loader_kind, model.cfg.vocab_size, model.cfg.dim,
			model.cfg.n_layers, model.cfg.n_heads, model.cfg.n_kv_heads, model.cfg.seq_len);
		if(model.token_str != nil){
			for(i = 0, nstr = 0; i < model.cfg.vocab_size; i++)
				if(model.token_str[i] != nil)
					nstr++;
			fprint(2, "token_str entries: %d / %d\n", nstr, model.cfg.vocab_size);
		}else
			fprint(2, "token_str: (nil)\n");
	}

	if(alloc_run_state(&state, &model.cfg, err, sizeof err) < 0)
		sysfatal("%s", err);
	clear_kv_cache(&state, &model.cfg);

	pos = 0;
	token = clamp_token(' ', model.cfg.vocab_size);
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

	for(i = 0; i < steps && pos < model.cfg.seq_len; i++){
		if(dump_logits){
			/* Leading \\n so stderr lines do not glue to stdout tokens on one tty line. */
			fprint(2, "\nlogits top-8 step %d:", i);
			dump_logits_topk(2, state.logits, model.cfg.vocab_size, 8);
		}
		if(temperature > 0.0f)
			next = sample_with_temperature(state.logits, model.cfg.vocab_size, temperature);
		else
			next = greedy_sample(state.logits, model.cfg.vocab_size);
		emit_token(&model, next);
		token = next;
		if(transformer_forward(&model, &state, token, pos, err, sizeof err) < 0)
			sysfatal("forward failed at generation step %d", i);
		pos++;
	}
	fprint(1, "\n");

	free_run_state(&state);
	free_model(&model);
	exits(nil);
}
