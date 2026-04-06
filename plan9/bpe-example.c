#include <u.h>
#include <libc.h>


typedef struct {
  char *pair;   // e.g. "t+h" where delimiter can be '\x01'
  int rank;
} Merge;

typedef struct {
  char **items;
  int n;
} StrList;

static char 
*join_pair(const char *a, const char *b)
{
  unsigned int la = strlen(a), lb = strlen(b);
  char *s = malloc(la + lb + 2);
  memcpy(s, a, la);
  s[la] = '\x01'; // delimiter
  memcpy(s+la+1, b, lb);
  s[la+1+lb] = 0;
  return s;
}

static int 
merge_cmp(void *pa, void *pb)
{
  const Merge *a = pa; const Merge *b = pb;
  return strcmp(a->pair, b->pair);
}

Merge 
*load_merges(char *path, int *out_count) 
{
  FILE *f = fopen(path, "r");
  if(!f) return nil;
  Merge *arr = nil; int cap=0, n=0;
  char a[256], b[256];
  while(fscanf(f, "%255s %255s", a, b)==2){
    if(n==cap){ cap = cap?cap*2:256; arr = realloc(arr, cap * sizeof(Merge)); }
    arr[n].pair = join_pair(a,b);
    arr[n].rank = n;
    n++;
  }
  fclose(f);
  qsort(arr, n, sizeof(Merge), merge_cmp);
  *out_count = n;
  return arr;
}

int 
merge_lookup(Merge *arr, int n,char *x, char *y) 
{
  char *p = join_pair(x,y);
  Merge key = { .pair = p, .rank = -1 };
  Merge *found = bsearch(&key, arr, n, sizeof(Merge), merge_cmp);
  free(p);
  return found ? found->rank : -1;
}

StrList 
split_bytes(char *s) 
{
  size_t L = strlen(s);
  StrList sl = { malloc(L * sizeof(char*)), 0 };
  for(size_t i=0;i<L;i++){
    char *c = malloc(2);
    c[0] = s[i];
    c[1] = 0;
    sl.items[sl.n++] = c;
  }
  return sl;
}

char 
**bpe_encode_word(char *word, Merge *merges, int merges_n, int *out_tokens_n) {
  StrList symbols = split_bytes(word);
  // greedy merging loop
  while(1){
    if(symbols.n < 2) break;
    int best_i = -1;
    int best_rank = INT_MAX;
    for(int i=0;i<symbols.n-1;i++){
      int r = merge_lookup(merges, merges_n, symbols.items[i], symbols.items[i+1]);
      if(r>=0 && r < best_rank){ best_rank = r; best_i = i; }
    }
    if(best_i < 0) break;
    // merge best_i and best_i+1
    size_t la = strlen(symbols.items[best_i]), lb = strlen(symbols.items[best_i+1]);
    char *m = malloc(la + lb + 1);
    memcpy(m, symbols.items[best_i], la);
    memcpy(m+la, symbols.items[best_i+1], lb);
    m[la+lb] = 0;
    free(symbols.items[best_i]);
    free(symbols.items[best_i+1]);
    // shift left and replace
    symbols.items[best_i] = m;
    memmove(&symbols.items[best_i+1], &symbols.items[best_i+2], (symbols.n-best_i-2)*sizeof(char*));
    symbols.n -= 1;
  }
  // return array of token strings
  char **out = malloc(symbols.n * sizeof(char*));
  for(int i=0;i<symbols.n;i++){ out[i] = symbols.items[i]; }
  free(symbols.items);
  *out_tokens_n = symbols.n;
  return out;
}

/* Example usage:
   Merge *merges; int merges_n;
   merges = load_merges("merges.txt", &merges_n);
   int tcount; char **toks = bpe_encode_word("hello", merges, merges_n, &tcount);
*/
