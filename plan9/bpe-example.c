#include <u.h>
#include <libc.h>

typedef struct {
  char *pair;   /* delimiter-separated a\x01b */
  int rank;
} Merge;

typedef struct {
  char **items;
  int n;
} StrList;

static char *
join_pair(char *a, char *b)
{
  unsigned la = strlen(a), lb = strlen(b);
  char *s = malloc(la + lb + 2);
  if(s == nil) sysfatal("malloc failed");
  memcpy(s, a, la);
  s[la] = '\x01';
  memcpy(s+la+1, b, lb);
  s[la+1+lb] = 0;
  return s;
}

/* Read entire file into a buffer (null-terminated). Caller must free buf. */
static char *
readfile(char *path)
{
  int fd;
  char *buf = nil;
  int cap = 0;
  int len = 0;
  char tmp[4096];
  int n;

  fd = open(path, OREAD);
  if(fd < 0) return nil;
  while((n = read(fd, tmp, sizeof tmp)) > 0){
    if(len + n + 1 > cap){
      int newcap = cap ? cap*2 : 4096;
      while(newcap < len + n + 1) newcap *= 2;
      buf = realloc(buf, newcap);
      if(buf == nil) sysfatal("realloc failed");
      cap = newcap;
    }
    memcpy(buf + len, tmp, n);
    len += n;
  }
  close(fd);
  if(buf == nil) return nil;
  buf[len] = 0;
  return buf;
}

/* Load merges file where each line is "a b" */
Merge *
load_merges(char *path, int *out_count)
{
  char *buf = readfile(path);
  if(buf == nil) return nil;
  Merge *arr = nil;
  int cap = 0;
  int n = 0;

  char *p = buf;
  while(p && *p){
    /* find end of line */
    char *e = strchr(p, '\n');
    if(e) *e = 0;
    /* skip empty lines */
    if(*p != 0){
      /* find first space */
      char *sp = strchr(p, ' ');
      if(sp){
        *sp = 0;
        char *a = p;
        char *b = sp + 1;
        /* trim trailing whitespace on b */
        char *tb = b + strlen(b) - 1;
        while(tb >= b && (*tb == '\r' || *tb == ' ' || *tb == '\t')) *tb-- = 0;

        if(n == cap){
          cap = cap ? cap*2 : 256;
          arr = realloc(arr, cap * sizeof(Merge));
          if(arr == nil) sysfatal("realloc failed");
        }
        arr[n].pair = join_pair(a, b);
        arr[n].rank = n;
        n++;
      }
    }
    if(e) p = e + 1; else break;
  }
  free(buf);
  *out_count = n;
  return arr;
}

int
merge_lookup(Merge *arr, int n, char *x, char *y)
{
  char *p = join_pair(x, y);
  int found = -1;
  for(int i = 0; i < n; i++){
    if(strcmp(arr[i].pair, p) == 0){ found = arr[i].rank; break; }
  }
  free(p);
  return found;
}

StrList
split_bytes(char *s)
{
  unsigned L = strlen(s);
  StrList sl;
  sl.items = malloc(L * sizeof(char*));
  if(sl.items == nil) sysfatal("malloc failed");
  sl.n = 0;
  for(unsigned i = 0; i < L; i++){
    char *c = malloc(2);
    if(c == nil) sysfatal("malloc failed");
    c[0] = s[i];
    c[1] = 0;
    sl.items[sl.n++] = c;
  }
  return sl;
}

char **
bpe_encode_word(char *word, Merge *merges, int merges_n, int *out_tokens_n)
{
  StrList symbols = split_bytes(word);
  while(1){
    if(symbols.n < 2) break;
    int best_i = -1;
    int best_rank = 0x7fffffff;
    for(int i = 0; i < symbols.n - 1; i++){
      int r = merge_lookup(merges, merges_n, symbols.items[i], symbols.items[i+1]);
      if(r >= 0 && r < best_rank){ best_rank = r; best_i = i; }
    }
    if(best_i < 0) break;
    unsigned la = strlen(symbols.items[best_i]);
    unsigned lb = strlen(symbols.items[best_i+1]);
    char *m = malloc(la + lb + 1);
    if(m == nil) sysfatal("malloc failed");
    memcpy(m, symbols.items[best_i], la);
    memcpy(m+la, symbols.items[best_i+1], lb);
    m[la+lb] = 0;
    free(symbols.items[best_i]);
    free(symbols.items[best_i+1]);
    symbols.items[best_i] = m;
    if(symbols.n - best_i - 2 > 0)
      memmove(&symbols.items[best_i+1], &symbols.items[best_i+2], (symbols.n-best_i-2)*sizeof(char*));
    symbols.n -= 1;
  }
  char **out = malloc(symbols.n * sizeof(char*));
  if(out == nil) sysfatal("malloc failed");
  for(int i = 0; i < symbols.n; i++) out[i] = symbols.items[i];
  free(symbols.items);
  *out_tokens_n = symbols.n;
  return out;
}

/* Simple example: load merges.txt and encode "hello" */
void
main(int argc, char **argv)
{
  int merges_n = 0;
  Merge *merges = load_merges("merges.txt", &merges_n);
  if(merges == nil){
    print("failed to load merges.txt\n");
    exits("no merges");
  }
  int tcount = 0;
  char **toks = bpe_encode_word("hello", merges, merges_n, &tcount);
  for(int i = 0; i < tcount; i++){
    print("%s\n", toks[i]);
    free(toks[i]);
  }
  free(toks);
  for(int i = 0; i < merges_n; i++) free(merges[i].pair);
  free(merges);
  exits(nil);
}
#include <u.h>
#include <libc.h>
#include <bio.h>


typedef struct {
  char *pair;
  int rank;
} Merge;

typedef struct {
  char **items;
  int n;
} StrList;

/* join_pair: no const to suit older Plan 9 compilers */
static char *
join_pair(char *a, char *b)
{
  unsigned la = strlen(a), lb = strlen(b);
  char *s = malloc(la + lb + 2);
  if(s == nil) sysfatal("malloc failed");
  memcpy(s, a, la);
  s[la] = '\x01';
  memcpy(s+la+1, b, lb);
  s[la+1+lb] = 0;
  return s;
}

/* load_merges: use Biobuf to read lines; parse "a b" pairs */
Merge *
load_merges(char *path, int *out_count)
{
  Biobuf *bp;
  char *line;
  Merge *arr = nil;
  int cap = 0;
  int n = 0;

  bp = Bopen(path, OREAD);
  if(bp == nil) return nil;

  while((line = Brdline(bp, '\n')) != nil){
    /* strip newline */
    char *nl = strchr(line, '\n');
    if(nl) *nl = 0;

    /* find first whitespace separator */
    char *sp = strchr(line, ' ');
    if(sp == nil) continue;
    *sp = 0;
    char *a = line;
    char *b = sp + 1;

    if(n == cap){
      cap = cap ? cap*2 : 256;
      arr = realloc(arr, cap * sizeof(Merge));
      if(arr == nil) sysfatal("realloc failed");
    }
    arr[n].pair = join_pair(a, b);
    arr[n].rank = n;
    n++;
  }

  Bclose(bp);

  /* optional: simple qsort if available */
  if(n > 1) qsort(arr, n, sizeof(Merge),
    (int(*)(const void*,const void*)) strcmp /* wrapper below may be better */);

  *out_count = n;
  return arr;
}

/* simple linear lookup (avoids relying on bsearch implementation) */
int
merge_lookup(Merge *arr, int n, char *x, char *y)
{
  char *p = join_pair(x, y);
  int found = -1;
  for(int i=0;i<n;i++){
    if(strcmp(arr[i].pair, p) == 0){ found = arr[i].rank; break; }
  }
  free(p);
  return found;
}

/* split_bytes: use unsigned for lengths */
StrList
split_bytes(char *s)
{
  unsigned L = strlen(s);
  StrList sl;
  sl.items = malloc(L * sizeof(char*));
  if(sl.items == nil) sysfatal("malloc failed");
  sl.n = 0;
  for(unsigned i=0;i<L;i++){
    char *c = malloc(2);
    if(c == nil) sysfatal("malloc failed");
    c[0] = s[i];
    c[1] = 0;
    sl.items[sl.n++] = c;
  }
  return sl;
}

int main()
{
	Merge *merges;
	int merges_n; 
	merges = load_merges("merges.txt", &merges_n);
	int tcount;

	char **toks = bpe_encode_word("hello", merges, merges_n, &tcount);

	return 0;
}
/* Example usage:
   Merge *merges; int merges_n;
   merges = load_merges("merges.txt", &merges_n);
   int tcount; char **toks = bpe_encode_word("hello", merges, merges_n, &tcount);
*/
