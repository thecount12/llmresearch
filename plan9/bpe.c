#include <u.h>
#include <libc.h>

typedef struct Pair Pair;
struct Pair {
	uint32 t1, t2; // The pair of tokens (the key)
	long count;    // Frequency (the value)
	Pair *next;    // For collision chaining
};

typedef struct {
	Pair **buckets;
	int size;
} HashTable;

/* FNV-1a Hash for two 32-bit integers */
uint32
hash_pair(uint32 t1, uint32 t2, int size)
{
	uint32 h = 2166136261U;
	// Hash first token
	h ^= t1;
	h *= 16777619;
	// Hash second token
	h ^= t2;
	h *= 16777619;
	return h % size;
}

/* Initialize Table */
HashTable*
mktable(int size)
{
	HashTable *ht;
	ht = malloc(sizeof(HashTable));
	if(ht == nil) return nil;
	
	ht->size = size;
	ht->buckets = malloc(size * sizeof(Pair*));
	if(ht->buckets == nil){
		free(ht);
		return nil;
	}
	memset(ht->buckets, 0, size * sizeof(Pair*));
	return ht;
}

/* Increment count for a pair */
void
increment(HashTable *ht, uint32 t1, uint32 t2)
{
	uint32 h = hash_pair(t1, t2, ht->size);
	Pair *p;

	/* 1. Search if it already exists */
	for(p = ht->buckets[h]; p != nil; p = p->next){
		if(p->t1 == t1 && p->t2 == t2){
			p->count++;
			return;
		}
	}

	/* 2. Not found: Create new Pair struct */
	p = malloc(sizeof(Pair));
	if(p == nil) return;
	p->t1 = t1;
	p->t2 = t2;
	p->count = 1;
	p->next = ht->buckets[h]; // Insert at head of chain
	ht->buckets[h] = p;
}

/* Freeing the table */
void
freetable(HashTable *ht)
{
	int i;
	Pair *p, *tmp;

	for(i = 0; i < ht->size; i++){
		p = ht->buckets[i];
		while(p != nil){
			tmp = p->next;
			free(p); // Free each Pair struct
			p = tmp;
		}
	}
	free(ht->buckets);
	free(ht);
}
