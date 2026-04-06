.NH
Enbeddings
.PP
Embeddings are two-dimensional, and we plot them in two dimensional scatter plot 
for visualization. Word embeding techniques such as Word2Vec, have words 
corresponding to simliar concepts that appear close to each other in embedding space. Kind of like a grouping of birds to each other and countries would be grouped
to each other. High-dimensional embeddings are two difficult to represent in a 3
dimmeninal representation. The steps for preparing the embeddings used by an 
LLM, which include splitting text into words, and converting words into tokens, 
and turning tokens into embedding vectors is tokenization. 
.NH 2
Tokenizing text
.PP
We split input text into indidual tokens, a required preprocessing step for creating embeddings for an LLM. We will attempt to break up words and special characters.
.LP
Here is a copy from https://en.wikisource.org/wiki/The_Verdict
hget -o the-verdict.txt https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt
.LP
Let's open with C and read exactly 99 characters.
.P1
; touch verdict.c

#include <u.h>
#include <libc.h>
/* usage: verdict filename.txt */
void
main(int argc, char **argv)
{
  int fd;
  int i;
  char buf[99];
  //long n;

  if (argc == 1)
    fd = 0;
  else if ((fd = open(argv[1], OREAD)) < 0)
    perror(argv[1]);
  read(fd, buf, 256);
  // print("input file: %s\en", buf);  // testing
  print("----\n");
  for (i = 0; i < 99; i++) {
    print("%c", buf[i]);
  }
  close(fd);
  exits(nil);
}
.P2
.LP
Compile:
.P1
6c -FVw verdict.c
6l -o verdict verdict.6
.P2
.LP
Results:
.P1 
; verdict the-verdict.txt
----
I HAD always thought Jack Gisburn rather a cheap genius--though a good fellow enough--so it was no
.P2
.PP
The next step. Is to break everything up into words and special characters. Since we are 
dealing with a low level languages its a bit difficult to illustrate that, so we have to 
code in the extra brackets"[]" and commas to show a proper array. 
.P1
; touch word_split

#include <u.h>
#include <libc.h>

void
main(int, char* [])
{
  int i;
  char ar[] = "Hello, world. This, is a test.";
  print("['");
  for (i = 0; i < sizeof(ar); i++) {
    if (ar[i]==' ') {
      print("', '");
    }
    if (ar[i]!=' ') {
      print("%c", ar[i]);
    }
  }
  print("']\n");
}
.P2
.LP
Compile:
.P1
; 6c -FVw word_split.c
; 6l -o wordsplit word_split.6
.P2
.LP
Results:
.P1
wordsplit
['Hello,', 'world.', 'This,', 'is', 'a', 'test.']
.P2
.LP
The next step is to parse the period "." and the comma ",". We will start with 
the period first
.P1
; edit word_split.c

#include <u.h>
#include <libc.h>

void
main(int, char* [])
{
  int i;
  char ar[] = "Hello, world. This, is a test.";
  print("['");
  for (i = 0; i < sizeof(ar); i++) {
    if (ar[i]==' ') {
      print("', '");
    }
    if (ar[i]!=' ' && ar[i]!=',') {
      print("%c", ar[i]);
    }
    if (ar[i]==',') {
      print("'%c ',", ar[i]);
    }
  }
  print("']\n");
}
.P2
.LP
Results:
.P1
; wordsplit
['Hello', ',', 'world.', 'This', ',', 'is', 'a', 'test.']
.P2
.LP
Next we isolate the period.
.P1
; edit word_split.c

#include <u.h>
#include <libc.h>

void
main(int, char* [])
{
  int i;
  char ar[] = "Hello, world. This, is a test.";
  print("['");
  for (i = 0; i < sizeof(ar); i++) {
    if (ar[i]==' ') {
      print("', '");
    }
    if (ar[i]!=' ' && ar[i]!=',' && ar[i]!='.') {
      print("%c", ar[i]);
    }
    if (ar[i]==',') {
      print("'%c ',", ar[i]);
    }
    if (ar[i]=='.') {
      print("', '%c", ar[i]);
    }
  }
  print("']\n");
}
.P2
.LP
Results:
.P1
wordsplit
['Hello', ',', 'world', '.', 'This', ',', 'is', 'a', 'test', '.']
.P2
.PP
As you can seee this is extra work to display the proper results for visualization. A
case block could prove useful by tricky. A structure with a dynamic array would probably be better. Let's take a look
at that example. THe wordsplit below will be in two parts
.LP
Part I
.P1
; touch wordSplit.c

#include <u.h>
#include <libc.h>
#include <ctype.h>

typedef struct TokenArray {
	char **tokens;
	int size;
	int capacity;
} TokenArray;

void
inittokens(TokenArray *arr)
{
	arr->size = 0;
	print("arr size: %d\n", arr->size);
	arr->capacity = 16; // Initial capacity
	print("capacity: %d\n", arr->capacity);
	arr->tokens = malloc(arr->capacity * sizeof(char*));
	print("tokens memory: %p\n", arr->tokens);
	if (arr->tokens == nil) {
		sysfatal("malloc failed");
	}
}

// Doubles the capacity of the token array.
void
growtokens(TokenArray *arr)
{
	char **new_tokens;
	arr->capacity *= 2;
	new_tokens = realloc(arr->tokens, arr->capacity * sizeof(char*));
	if (new_tokens == nil) {
		sysfatal("realloc failed");
	}
	arr->tokens = new_tokens;
}

// Adds a new token (word or symbol) to the array.
void
addtoken(TokenArray *arr, char *token)
{
	if (arr->size >= arr->capacity) {
		growtokens(arr);
	}
	arr->tokens[arr->size] = strdup(token);
	if (arr->tokens[arr->size] == nil) {
		sysfatal("strdup failed");
	}
	arr->size++;
}
.P2
.LP
Part II
.P1
// Frees all memory allocated for the token array.
void
freetokens(TokenArray *arr)
{
	int i;
	for (i = 0; i < arr->size; i++) {
		free(arr->tokens[i]);
	}
	free(arr->tokens);
}

// Checks if a character is a symbol
int
issymbol(int c)
{
	static char *symbols = ",.:;?!\"()\'\t\n ";
	return c != 0 && strchr(symbols, c) != nil;
}

void
parser(char *text, TokenArray *arr)
{
	char *start;
	char *end;
	char token_buffer[1024]; // Temporary buffer for tokens\

	start = text;
	while (*start != '\0') {
		// Skip leading whitespace
		while (isspace(*start)) {
			print("hit space:::\n");
			start++;
		}
		if (*start == '\0') {
			break;
		}
		// Handle the special two-character symbol '--'
		if (*start == '-' && *(start + 1) == '-') {
			addtoken(arr, "--");
			print("added --\n");
			print("and start is: %d\n", *start);
			start += 2;
			continue;
		}
		// Handle single-character symbols
		if (issymbol(*start)) {
			token_buffer[0] = *start;
			token_buffer[1] = '\0';
			addtoken(arr, token_buffer);
			print("added character:\n");
			start++;
			continue;
		}
		// Handle words
		end = start;
		while (*end != '\0' && !isspace(*end) && !issymbol(*end) 
			&& !(*end == '-' && *(end + 1) == '-')) {
			end++;
		}

		if (end > start) {
			long word_len = end - start;
			if (word_len < sizeof(token_buffer)) {
				strncpy(token_buffer, start, word_len);
				token_buffer[word_len] = '\0';
				addtoken(arr, token_buffer);
			} else {
				// Handle very long words if needed, though unlikely for this example
			}
		}
		start = end;
	}
	
}

void
main(int, char* [])
{
	char *text = "Hello, world! This is a test--with some symbols and numbers 123.";
	TokenArray tokens;
	inittokens(&tokens);
	parser(text, &tokens);

	// Print the contents of the token array
	print("--- Token Array ---\n");
	int i;
	for (i = 0; i < tokens.size; i++) {
		print("token[%d]: \"%s\"\n", i, tokens.tokens[i]);
	}
	print("-------------------\n");

	freetokens(&tokens);

	exits(nil);
}
.P2
.LP
Compile:
.P1
;6c -FVw wordSplit.c
;6l -o split2 wordSplit.6
.P2
Results:
.P1
; split2
arr size: 0
capacity: 16
tokens memory: 4018e8
added character:
hit space:::
added character:
hit space:::
hit space:::
hit space:::
hit space:::
added --
and start is: 45
hit space:::
hit space:::
hit space:::
hit space:::
hit space:::
added character:
--- Token Array ---
token[0]: "Hello"
token[1]: ","
token[2]: "world"
token[3]: "!"
token[4]: "This"
token[5]: "is"
token[6]: "a"
token[7]: "test"
token[8]: "--"
token[9]: "with"
token[10]: "some"
token[11]: "symbols"
token[12]: "and"
token[13]: "numbers"
token[14]: "123"
token[15]: "."
.P2
.PP
This example is considerably more code and extra functions to handle the work load but scales considerably. Now let's 
re-open and process the whole file. 
.P1
; storyCount.c

void
main(int, char* [])
{
  int fd;
  int i;
  char *filedata;
  long n;
  Dir *d;
  char *filename = "the-verdict.txt";
  d = dirstat(filename);
  if (d == nil) {
    print("could not stat file '%s': %r\en", filename);
    exits("file error");
  }
  // Allocate memory for file content (+1 for null terminator)
  filedata = malloc(d->length + 1);
  if (filedata == nil) {
    print("malloc failed: %r\en", filename);
    free(d);
    exits("memory error");
  }
  free(d);
  // Open file for reading
  fd = open(filename, OREAD);
  if (fd < 0) {
    print("could not open file '%s': %r\en", filename);
    free(filedata);
    exits("file error");
  }
  // Read the entire file into the buffer
  n = read(fd, filedata, d->length);
  if (n < 0) {
    print("error reading file '%s': %r\en", filename);
    free(filedata);
    close(fd);
    exits("read error");
  }
  filedata[n] = '\0'; // Null-terminate the buffer
  close(fd);

  TokenArray tokens;
  inittokens(&tokens);
  parser(filedata, &tokens);

  // Print the contents of the token array
  print("--- Token Array ---\en");

  for (i = 0; i < tokens.size; i++) {
    print("token[%d]: \"%s\"\en", i, tokens.tokens[i]);
  }
  print("-------------------\en");

  freetokens(&tokens);
  exits(nil);
}
.P2
.LP
I did forget to add a symbol "_" so you will need to edit issymbol() function. 
.P1
issymbol(int c)
{
	static char *symbols = ",.:;?!_\e"()\e'\et\en ";
	return c != 0 && strchr(symbols, c) != nil;
}
.P2
.LP
Compile:
.P1
; 6c -FVw wordCount.c
; 6l -o count wordCount.6
.P2
Results:
.P1
token[4683]: "exterminating"
token[4684]: "our"
token[4685]: "kind"
token[4686]: "of"
token[4687]: "art"
token[4688]: "."
token[4689]: """
-------------------
.P2
.PP
So it should be exactly 4690 tokens. It took a while to analyze but I discovered it was missing a hyphen "-". The code
above parses two hyphens "--". 
.PP
In the next example that follows we have to do something a little bit different. We need to build a hash table so we can
keep track of the tokenization of unique words and symbols. We need a way to be able to read in the raw text and tokenize
or encode then decode back again.
.LP
Part I
.P1
; touch simpleTokenizer.c

#include <u.h>
#include <libc.h>
#include <ctype.h>

// --- Hash Table Implementation ---

unsigned int hash(const char *str, int size) {
    unsigned int hash_val = 0;
    while (*str) {
        hash_val = (hash_val << 5) + hash_val + *str;
        str++;
    }
    return hash_val % size;
}

typedef struct Entry {
    char *key;
    int value;
    struct Entry *next;
} Entry;

typedef struct Hashtable {
    Entry **entries;
    int size;
} Hashtable;

Hashtable* create_hashtable(int size) {
    Hashtable *table = malloc(sizeof(Hashtable));
    if (table == nil) sysfatal("malloc: %r");
    table->size = size;
    table->entries = malloc(sizeof(Entry*) * size);
    if (table->entries == nil) sysfatal("malloc: %r");
    memset(table->entries, 0, sizeof(Entry*) * size);
    return table;
}

void add_entry(Hashtable *table, char *key, int value) {
    unsigned int slot = hash(key, table->size);
    Entry *new_entry = malloc(sizeof(Entry));
    if (new_entry == nil) sysfatal("malloc: %r");
    new_entry->key = strdup(key);
    new_entry->value = value;
    new_entry->next = table->entries[slot];
    table->entries[slot] = new_entry;
}

int get_value(Hashtable *table, char *key, int *value) {
    unsigned int slot = hash(key, table->size);
    Entry *entry = table->entries[slot];
    while (entry != nil) {
        if (strcmp(entry->key, key) == 0) {
            *value = entry->value;
            return 1; // Success
        }
        entry = entry->next;
    }
    return 0; // Not found
}

void free_hashtable(Hashtable *table) {
    for (int i = 0; i < table->size; i++) {
        Entry *entry = table->entries[i];
        while (entry != nil) {
            Entry *temp = entry;
            entry = entry->next;
            free(temp->key);
            free(temp);
        }
    }
    free(table->entries);
    free(table);
}
.P2
.LP
Part II
.P1
// --- SimpleTokenizerV1 Implementation ---

typedef struct SimpleTokenizerV1 {
    Hashtable *str_to_int;
    char **int_to_str;
    int vocab_size;
} SimpleTokenizerV1;

typedef struct TokenArray {
    char **tokens;
    int size;
    int capacity;
} TokenArray;

void addtoken(TokenArray *arr, char *token) {
    if (arr->size >= arr->capacity) {
        arr->capacity = arr->capacity ? arr->capacity * 2 : 10;
        arr->tokens = realloc(arr->tokens, arr->capacity * sizeof(char*));
        if (arr->tokens == nil) {
            sysfatal("realloc: %r");
        }
    }
    arr->tokens[arr->size++] = strdup(token);
    if (arr->tokens[arr->size - 1] == nil) {
        sysfatal("strdup: %r");
    }
}

void inittokens(TokenArray *arr) {
    arr->tokens = nil;
    arr->size = 0;
    arr->capacity = 0;
}

void freetokens(TokenArray *arr) {
    int i;
    for (i = 0; i < arr->size; i++) {
        free(arr->tokens[i]);
    }
    free(arr->tokens);
}

int issymbol(char c) {
    switch (c) {
        case ',':
        case '_':
        case '.':
        case '!':
        case ';':
        case '?':
        case '(':
        case ')':
        case '\t':
        case '\n':
        case '\"':
        case '\'':
        case ':':
            return 1;
        default:
            return 0;
    }
}

void parser_with_separators(char *text, TokenArray *arr) {
    char *p = text;
    char *start;
    char token_buffer[2]; // Use a buffer for single characters

    while (*p != '\0') {
        start = p;
        if (*p == '-' && *(p + 1) == '-') {
            addtoken(arr, "--");
            p += 2;
        } else if (issymbol(*p) || isspace(*p)) {
            token_buffer[0] = *p;
            token_buffer[1] = '\0';
            addtoken(arr, token_buffer);
            p++;
        } else {
            while (*p != '\0' && !isspace(*p) && !issymbol(*p) && !(*p == '-' && *(p + 1) == '-')) {
                p++;
            }
            long len = p - start;
            if (len > 0) {
                char *word = malloc(len + 1);
                memmove(word, start, len);
                word[len] = '\0';
                addtoken(arr, word);
                free(word);
            }
        }
    }
}

SimpleTokenizerV1* newSimpleTokenizerV1(TokenArray *vocab_tokens) {
    SimpleTokenizerV1 *tokenizer = malloc(sizeof(SimpleTokenizerV1));
    if (tokenizer == nil) sysfatal("malloc: %r");

    tokenizer->vocab_size = vocab_tokens->size;
    tokenizer->str_to_int = create_hashtable(tokenizer->vocab_size * 2);
    tokenizer->int_to_str = malloc(sizeof(char*) * tokenizer->vocab_size);
    if (tokenizer->int_to_str == nil) sysfatal("malloc: %r");

    for (int i = 0; i < tokenizer->vocab_size; i++) {
        add_entry(tokenizer->str_to_int, vocab_tokens->tokens[i], i);
        tokenizer->int_to_str[i] = strdup(vocab_tokens->tokens[i]);
    }
    return tokenizer;
}

TokenArray* encode(SimpleTokenizerV1 *tokenizer, char *text) {
    TokenArray *preprocessed = malloc(sizeof(TokenArray));
    inittokens(preprocessed);
    parser_with_separators(text, preprocessed);
    
    TokenArray *ids = malloc(sizeof(TokenArray));
    inittokens(ids);
    int id;

    for (int i = 0; i < preprocessed->size; i++) {
        if (get_value(tokenizer->str_to_int, preprocessed->tokens[i], &id)) {
            // Using a temporary buffer for the integer string
            char buf[20];
            sprint(buf, "%d", id);
            addtoken(ids, buf);
        } else {
            // Handle unknown token, if necessary
        }
    }
    freetokens(preprocessed);
    free(preprocessed);

    return ids;
}

char* decode(SimpleTokenizerV1 *tokenizer, TokenArray *ids) {
    TokenArray *parts = malloc(sizeof(TokenArray));
    inittokens(parts);
    int id;

    for (int i = 0; i < ids->size; i++) {
        id = atoi(ids->tokens[i]);
        if (id >= 0 && id < tokenizer->vocab_size) {
            addtoken(parts, tokenizer->int_to_str[id]);
        }
    }

    char *raw_text = nil;
    if (parts->size > 0) {
        raw_text = strdup(parts->tokens[0]);
        for (int i = 1; i < parts->size; i++) {
            char *current_token = parts->tokens[i];
            if (raw_text == nil) {
                raw_text = strdup(current_token);
            } else if (issymbol(*current_token) || isspace(*current_token)) {
                char *merged = smprint("%s%s", raw_text, current_token);
                free(raw_text);
                raw_text = merged;
            } else {
                char *merged = smprint("%s %s", raw_text, current_token);
                free(raw_text);
                raw_text = merged;
            }
        }
    } else {
        raw_text = strdup("");
    }
    
    freetokens(parts);
    free(parts);
    return raw_text;
}

int cmpstr(const void* a, const void* b) {
    const char* aa = *(const char**)a;
    const char* bb = *(const char**)b;
    return strcmp(aa, bb);
}
.P2
.LP
Main function
.P1
void
main(int, char* [])
{
    int fd;
    char *filename = "the-verdict.txt";
    char *filedata;
    long n;
    Dir *d;
    TokenArray all_tokens;
    TokenArray unique_tokens;
    SimpleTokenizerV1 *tokenizer;
    TokenArray *ids;
    char *decoded_text;
    int i;

    // Read file and parse tokens
    d = dirstat(filename);
    if (d == nil) {
        fprint(2, "could not stat file '%s': %r\n", filename);
        exits("file error");
    }
    filedata = malloc(d->length + 1);
    if (filedata == nil) {
        fprint(2, "malloc failed: %r\n");
        free(d);
        exits("memory error");
    }
    free(d);
    fd = open(filename, OREAD);
    if (fd < 0) {
        fprint(2, "could not open file '%s': %r\n", filename);
        free(filedata);
        exits("file error");
    }
    n = read(fd, filedata, d->length);
    if (n < 0) {
        fprint(2, "error reading file '%s': %r\n", filename);
        free(filedata);
        close(fd);
        exits("read error");
    }
    filedata[n] = '\0';
    close(fd);

    // Get all unique tokens to build the vocabulary
    inittokens(&all_tokens);
    parser_with_separators(filedata, &all_tokens);


    if (all_tokens.size == 0) {
        print("No tokens found.\n");
        freetokens(&all_tokens);
        free(filedata); // prod
        exits(nil);
    }

    qsort(all_tokens.tokens, all_tokens.size, sizeof(char*), cmpstr);

    inittokens(&unique_tokens);
    addtoken(&unique_tokens, all_tokens.tokens[0]);

    for (i = 1; i < all_tokens.size; i++) {
        if (strcmp(all_tokens.tokens[i], all_tokens.tokens[i-1]) != 0) {
            addtoken(&unique_tokens, all_tokens.tokens[i]);
        }
    }
    freetokens(&all_tokens);

    // Initialize the tokenizer
    tokenizer = newSimpleTokenizerV1(&unique_tokens);

    // Encode the original text
    ids = encode(tokenizer, filedata);
  

    // Decode the ids
    decoded_text = decode(tokenizer, ids);

    // Print results
    print("--- Original Text ---\n");
    print("%s\n", filedata);
    print("--- Decoded Text ---\n");
    print("%s\n", decoded_text);
    print("--- Tokens (IDs) ---\n");
    for (i = 0; i < ids->size; i++) {
        print("id[%d]: %s\n", i, ids->tokens[i]);
    }

    // Cleanup
    free(filedata);
    freetokens(&unique_tokens);
    free_hashtable(tokenizer->str_to_int);
    for (i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->int_to_str[i]);
    }
    free(tokenizer->int_to_str);
    free(tokenizer);
    freetokens(ids);
    free(ids);
    free(decoded_text);
    exits(nil);
}
.P2
.LP
Compile:
.P1
; 6c -FVw simpleTokenizerv.c
; 6l -o simpleT simpleTokenizer.6
.P2
.LP
Results: This will run for about a minute. It's extremely long. 
.P1
...
id[8392]: 711
id[8393]: 1
id[8394]: 415
id[8395]: 1
id[8396]: 738
id[8397]: 1
id[8398]: 595
id[8399]: 1
id[8400]: 724
id[8401]: 1
id[8402]: 176
id[8403]: 9
id[8404]: 3
.P2
.PP
This looks to be about 8404 tokens. Kind of diffult to verify if we are doing anything righ. Let's process a small sample
of this to validate if we are encoding and decoding correctly. Let's modify the main section. 
.P1
; edit simpleTokenizer.c

void
main(int, char* [])
{
	// small string test
    char *textstring = "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.";
	//char *textstring = "Hello, do you like tea?";
    int fd;
    char *filename = "the-verdict.txt";
    char *filedata;
    long n;
    Dir *d;
    TokenArray all_tokens;
    TokenArray unique_tokens;
    SimpleTokenizerV1 *tokenizer;
    TokenArray *ids;
    char *decoded_text;
    int i;

    // Get all unique tokens to build the vocabulary
    inittokens(&all_tokens);
    //parser_with_separators(filedata, &all_tokens); // prod
	parser_with_separators(textstring, &all_tokens); // small string test


    if (all_tokens.size == 0) {
        print("No tokens found.\n");
        freetokens(&all_tokens);
        //free(filedata); // prod
        exits(nil);
    }

    qsort(all_tokens.tokens, all_tokens.size, sizeof(char*), cmpstr);

    inittokens(&unique_tokens);
    //addtoken(&unique_tokens, all_tokens.tokens[0]); // prod
	// small string test uncomment if block
	if (all_tokens.size > 0) {
        addtoken(&unique_tokens, all_tokens.tokens[0]);
    }

    for (i = 1; i < all_tokens.size; i++) {
        if (strcmp(all_tokens.tokens[i], all_tokens.tokens[i-1]) != 0) {
            addtoken(&unique_tokens, all_tokens.tokens[i]);
        }
    }
    freetokens(&all_tokens);

    // Initialize the tokenizer
    tokenizer = newSimpleTokenizerV1(&unique_tokens);

    // Encode the original text
    //ids = encode(tokenizer, filedata); // prod
    ids = encode(tokenizer, textstring); // small string test

    // Decode the ids
    decoded_text = decode(tokenizer, ids);

    // Print results
    print("--- Original Text ---\n");
    //print("%s\n", filedata); // prod
    print("%s\n", textstring); // small string test
    print("--- Decoded Text ---\n");
    print("%s\n", decoded_text);
    print("--- Tokens (IDs) ---\n");
	// uncomment for prod
    //for (i = 0; i < ids->size; i++) {
    //    print("id[%d]: %s\n", i, ids->tokens[i]);
    //}
	// uncomment for string test
	for (i = 0; i < ids->size; i++) {
        print("id[%d]: %s\n", i, ids->tokens[i]);
    }


    // Cleanup
    //free(filedata); // uncomment for prod
    freetokens(&unique_tokens);
    free_hashtable(tokenizer->str_to_int);
    for (i = 0; i < tokenizer->vocab_size; i++) {
        free(tokenizer->int_to_str[i]);
    }
    free(tokenizer->int_to_str);
    free(tokenizer);
    freetokens(ids);
    free(ids);
    free(decoded_text);
    exits(nil);
}
.P2
.LP
Compile:
.P1
; 6c -FVw simpleTokenizer.c
; 6l -o simpleT simpleTokenizer.6
.P2
.LP
Results:
.P1
simpleT
--- Original Text ---
It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.
--- Decoded Text ---
It' s  the  last  he  painted,  you  know,  Mrs.  Gisburn  said  with  pardonable  pride.
--- Tokens (IDs) ---
id[0]: 5
id[1]: 1
id[2]: 13
id[3]: 0
id[4]: 15
id[5]: 0
id[6]: 9
id[7]: 0
id[8]: 7
id[9]: 0
...
.P2
.LP
It appears to be working. However we have extra white spaces. This doesn't bother me at the moment. I may have to do major 
refactoring use an algorithm that also has better performance. 
