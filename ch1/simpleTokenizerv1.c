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

void
main(int, char* [])
{
	// small string test
    //char *textstring = "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.";
	char *textstring = "Hello, do you like tea?";
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
// comment out for :small string test
/*
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
*/ 
// small string test end

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