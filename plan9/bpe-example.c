#include <u.h>
#include <libc.h>

typedef struct Merge Merge;
typedef struct StrList StrList;

struct Merge {
	char *pair;
	int rank;
};

struct StrList {
	char **items;
	int n;
};

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
		memcpy(buf + len, tmp, n);
		len += n;
	}
	close(fd);

	if(buf == nil)
		return nil;
	buf[len] = 0;
	return buf;
}

static Merge *
load_merges(char *path, int *out_count)
{
	char *buf, *p, *e, *sp, *a, *b, *tb;
	Merge *arr;
	int cap, n;

	buf = readfile(path);
	if(buf == nil)
		return nil;

	arr = nil;
	cap = 0;
	n = 0;
	p = buf;
	while(p != nil && *p != 0){
		e = strchr(p, '\n');
		if(e != nil)
			*e = 0;

		if(*p != 0){
			sp = strchr(p, ' ');
			if(sp != nil){
				*sp = 0;
				a = p;
				b = sp + 1;
				tb = b + strlen(b);
				while(tb > b && (tb[-1] == '\r' || tb[-1] == ' ' || tb[-1] == '\t')){
					tb[-1] = 0;
					tb--;
				}

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

		if(e != nil)
			p = e + 1;
		else
			break;
	}

	free(buf);
	*out_count = n;
	return arr;
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
		memcpy(m, symbols.items[best_i], la);
		memcpy(m + la, symbols.items[best_i + 1], lb);
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

void
main(int argc, char **argv)
{
	Merge *merges;
	char **toks;
	int merges_n, tcount, i;
	char *word, *path;

	ARGBEGIN{
	default:
		break;
	}ARGEND

	path = "merges.txt";
	word = "hello";
	if(argc > 0)
		path = argv[0];
	if(argc > 1)
		word = argv[1];

	merges = load_merges(path, &merges_n);
	if(merges == nil)
		sysfatal("failed to load %s", path);

	toks = bpe_encode_word(word, merges, merges_n, &tcount);
	for(i = 0; i < tcount; i++){
		print("%s\n", toks[i]);
		free(toks[i]);
	}
	free(toks);
	for(i = 0; i < merges_n; i++)
		free(merges[i].pair);
	free(merges);
	exits(nil);
}
