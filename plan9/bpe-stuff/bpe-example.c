#include <u.h>
#include <libc.h>

typedef struct Merge Merge;
typedef struct StrList StrList;
typedef struct VocabEntry VocabEntry;
typedef struct Vocab Vocab;

struct Merge {
	char *pair;
	int rank;
};

struct StrList {
	char **items;
	int n;
};

struct VocabEntry {
	char *token;
	int id;
};

struct Vocab {
	VocabEntry *items;
	int n;
};

static void
usage(void)
{
	fprint(2, "usage: %s [-f textfile] [-v vocab.txt] merges.txt [word]\n", argv0);
	exits("usage");
}

static int
isspace9(int c)
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static char *
join_pair(char *a, char *b)
{
	unsigned la, lb;
	char *s;

	la = strlen(a);
	lb = strlen(b);
	s = malloc(la + lb + 2);
	if(s == nil)
		sysfatal("malloc failed");
	memcpy(s, a, la);
	s[la] = '\x01';
	memcpy(s + la + 1, b, lb);
	s[la + lb + 1] = 0;
	return s;
}

static char *
xstrndup(char *s, int n)
{
	char *p;

	p = malloc(n + 1);
	if(p == nil)
		sysfatal("malloc failed");
	memmove(p, s, n);
	p[n] = 0;
	return p;
}

static char *
readfile(char *path)
{
	int fd, n, cap, len, newcap;
	char tmp[4096];
	char *buf;

	fd = open(path, OREAD);
	if(fd < 0)
		return nil;

	buf = nil;
	cap = 0;
	len = 0;
	while((n = read(fd, tmp, sizeof tmp)) > 0){
		if(len + n + 1 > cap){
			newcap = cap ? cap * 2 : 4096;
			while(newcap < len + n + 1)
				newcap *= 2;
			buf = realloc(buf, newcap);
			if(buf == nil)
				sysfatal("realloc failed");
			cap = newcap;
		}
		memmove(buf + len, tmp, n);
		len += n;
	}
	close(fd);

	if(buf == nil){
		buf = malloc(1);
		if(buf == nil)
			sysfatal("malloc failed");
		buf[0] = 0;
		return buf;
	}
	buf[len] = 0;
	return buf;
}

static char *
skipspace(char *p)
{
	while(*p != 0 && isspace9((uchar)*p))
		p++;
	return p;
}

static Merge *
load_merges(char *path, int *out_count)
{
	char *buf, *p, *e, *a, *b;
	Merge *arr;
	int cap, n;

	buf = readfile(path);
	if(buf == nil)
		return nil;

	arr = nil;
	cap = 0;
	n = 0;
	p = buf;
	while(*p != 0){
		e = strchr(p, '\n');
		if(e != nil)
			*e = 0;

		a = skipspace(p);
		if(*a != 0 && *a != '#'){
			b = a;
			while(*b != 0 && !isspace9((uchar)*b))
				b++;
			if(*b != 0){
				*b++ = 0;
				b = skipspace(b);
				if(*b != 0){
					if(n == cap){
						cap = cap ? cap * 2 : 256;
						arr = realloc(arr, cap * sizeof(Merge));
						if(arr == nil)
							sysfatal("realloc failed");
					}
					arr[n].pair = join_pair(a, b);
					arr[n].rank = n;
					n++;
				}
			}
		}

		if(e == nil)
			break;
		p = e + 1;
	}

	free(buf);
	*out_count = n;
	return arr;
}

static Vocab *
load_vocab(char *path)
{
	char *buf, *p, *e, *line, *sep, *idp;
	Vocab *v;
	int cap, n;

	buf = readfile(path);
	if(buf == nil)
		return nil;

	v = malloc(sizeof(Vocab));
	if(v == nil)
		sysfatal("malloc failed");
	v->items = nil;
	v->n = 0;

	cap = 0;
	n = 0;
	p = buf;
	while(*p != 0){
		e = strchr(p, '\n');
		if(e != nil)
			*e = 0;

		line = p;
		line = skipspace(line);
		if(*line != 0 && *line != '#'){
			sep = strrchr(line, '\t');
			if(sep == nil){
				sep = strrchr(line, ' ');
				while(sep != nil && sep > line && sep[-1] == ' ')
					sep--;
			}
			if(sep != nil){
				*sep = 0;
				idp = skipspace(sep + 1);
				if(*idp != 0){
					if(n == cap){
						cap = cap ? cap * 2 : 256;
						v->items = realloc(v->items, cap * sizeof(VocabEntry));
						if(v->items == nil)
							sysfatal("realloc failed");
					}
					v->items[n].token = strdup(line);
					if(v->items[n].token == nil)
						sysfatal("strdup failed");
					v->items[n].id = atoi(idp);
					n++;
				}
			}
		}

		if(e == nil)
			break;
		p = e + 1;
	}

	free(buf);
	v->n = n;
	return v;
}

static int
vocab_lookup(Vocab *v, char *token)
{
	int i;

	if(v == nil)
		return -1;
	for(i = 0; i < v->n; i++)
		if(strcmp(v->items[i].token, token) == 0)
			return v->items[i].id;
	return -1;
}

static int
merge_lookup(Merge *arr, int n, char *x, char *y)
{
	char *p;
	int i, found;

	p = join_pair(x, y);
	found = -1;
	for(i = 0; i < n; i++){
		if(strcmp(arr[i].pair, p) == 0){
			found = arr[i].rank;
			break;
		}
	}
	free(p);
	return found;
}

/*
 * This is byte-level splitting, so it works on raw input bytes.
 * It is still not a full GPT-2 tokenizer because it does not do
 * byte-to-unicode remapping or regex pretokenization.
 */
static StrList
split_bytes(char *s)
{
	StrList sl;
	unsigned i, L;
	char *c;

	L = strlen(s);
	sl.items = malloc(L * sizeof(char *));
	if(sl.items == nil)
		sysfatal("malloc failed");
	sl.n = 0;
	for(i = 0; i < L; i++){
		c = malloc(2);
		if(c == nil)
			sysfatal("malloc failed");
		c[0] = s[i];
		c[1] = 0;
		sl.items[sl.n++] = c;
	}
	return sl;
}

static char **
bpe_encode_word(char *word, Merge *merges, int merges_n, int *out_tokens_n)
{
	StrList symbols;
	char **out, *m;
	unsigned la, lb;
	int i, best_i, best_rank, r;

	symbols = split_bytes(word);
	for(;;){
		if(symbols.n < 2)
			break;

		best_i = -1;
		best_rank = 0x7fffffff;
		for(i = 0; i < symbols.n - 1; i++){
			r = merge_lookup(merges, merges_n, symbols.items[i], symbols.items[i + 1]);
			if(r >= 0 && r < best_rank){
				best_rank = r;
				best_i = i;
			}
		}
		if(best_i < 0)
			break;

		la = strlen(symbols.items[best_i]);
		lb = strlen(symbols.items[best_i + 1]);
		m = malloc(la + lb + 1);
		if(m == nil)
			sysfatal("malloc failed");
		memmove(m, symbols.items[best_i], la);
		memmove(m + la, symbols.items[best_i + 1], lb);
		m[la + lb] = 0;

		free(symbols.items[best_i]);
		free(symbols.items[best_i + 1]);
		symbols.items[best_i] = m;
		if(symbols.n - best_i - 2 > 0)
			memmove(&symbols.items[best_i + 1], &symbols.items[best_i + 2],
				(symbols.n - best_i - 2) * sizeof(char *));
		symbols.n--;
	}

	out = malloc(symbols.n * sizeof(char *));
	if(out == nil)
		sysfatal("malloc failed");
	for(i = 0; i < symbols.n; i++)
		out[i] = symbols.items[i];
	free(symbols.items);
	*out_tokens_n = symbols.n;
	return out;
}

static void
free_tokens(char **toks, int n)
{
	int i;

	for(i = 0; i < n; i++)
		free(toks[i]);
	free(toks);
}

static void
free_merges(Merge *merges, int n)
{
	int i;

	for(i = 0; i < n; i++)
		free(merges[i].pair);
	free(merges);
}

static void
free_vocab(Vocab *v)
{
	int i;

	if(v == nil)
		return;
	for(i = 0; i < v->n; i++)
		free(v->items[i].token);
	free(v->items);
	free(v);
}

static void
print_encoded_word(char *word, Merge *merges, int merges_n, Vocab *vocab)
{
	char **toks;
	int ntok, i, id;

	toks = bpe_encode_word(word, merges, merges_n, &ntok);
	print("word: %s\n", word);
	for(i = 0; i < ntok; i++){
		id = vocab_lookup(vocab, toks[i]);
		if(vocab != nil)
			print("  %s\t%d\n", toks[i], id);
		else
			print("  %s\n", toks[i]);
	}
	free_tokens(toks, ntok);
}

static void
encode_text(char *text, Merge *merges, int merges_n, Vocab *vocab)
{
	char *p, *start, *word;
	int n;

	p = text;
	while(*p != 0){
		while(*p != 0 && isspace9((uchar)*p))
			p++;
		if(*p == 0)
			break;

		start = p;
		while(*p != 0 && !isspace9((uchar)*p))
			p++;
		n = p - start;
		word = xstrndup(start, n);
		print_encoded_word(word, merges, merges_n, vocab);
		free(word);
	}
}

void
main(int argc, char **argv)
{
	Merge *merges;
	Vocab *vocab;
	char *mergespath, *vocabpath, *textpath, *word, *text;
	int merges_n;

	vocab = nil;
	vocabpath = nil;
	textpath = nil;

	ARGBEGIN{
	case 'f':
		textpath = EARGF(usage());
		break;
	case 'v':
		vocabpath = EARGF(usage());
		break;
	default:
		usage();
	}ARGEND

	if(argc < 1)
		usage();

	mergespath = argv[0];
	word = "hello";
	if(argc > 1)
		word = argv[1];

	merges = load_merges(mergespath, &merges_n);
	if(merges == nil)
		sysfatal("failed to load %s", mergespath);

	if(vocabpath != nil){
		vocab = load_vocab(vocabpath);
		if(vocab == nil)
			sysfatal("failed to load %s", vocabpath);
	}

	if(textpath != nil){
		text = readfile(textpath);
		if(text == nil)
			sysfatal("failed to load %s", textpath);
		encode_text(text, merges, merges_n, vocab);
		free(text);
	}else{
		print_encoded_word(word, merges, merges_n, vocab);
	}

	free_vocab(vocab);
	free_merges(merges, merges_n);
	exits(nil);
}
