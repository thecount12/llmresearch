#include <u.h>
#include <libc.h>


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

/* Example usage:
   Merge *merges; int merges_n;
   merges = load_merges("merges.txt", &merges_n);
   int tcount; char **toks = bpe_encode_word("hello", merges, merges_n, &tcount);
*/
