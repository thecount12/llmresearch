#include "model.h"

static void
usage(void)
{
	fprint(2, "usage: %s [-m model.bin] [-n steps] [-p prompt] [-t temp]\n", argv0);
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
emit_token(int token)
{
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
	char err[128];
	char *model_path, *prompt;
	int steps, pos, i, token, next, promptlen;
	float temperature;

	memset(&model, 0, sizeof(model));
	memset(&state, 0, sizeof(state));
	model_path = nil;
	prompt = "";
	steps = 32;
	temperature = 0.0f;

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
		if(temperature > 0.0f)
			next = sample_with_temperature(state.logits, model.cfg.vocab_size, temperature);
		else
			next = greedy_sample(state.logits, model.cfg.vocab_size);
		emit_token(next);
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
