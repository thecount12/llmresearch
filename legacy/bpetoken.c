#include <u.h>
#include <libc.h>
#include <ctype.h>

// --- Data Structures ---

// A hash table to store pair frequencies during training
typedef struct PairFreq {
    char *key;      // The pair, e.g., "th"
    int freq;       // Its frequency
    struct PairFreq *next;
} PairFreq;

typedef struct FreqTable {
    PairFreq **entries;
    int size;
} FreqTable;

typedef struct Entry {
    char *key;
    int value;
    struct Entry *next;
} Entry;

typedef struct Hashtable {
    Entry **entries;
    int size;
} Hashtable;

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


// A tokenizer object to hold the vocabulary and merge rules
typedef struct BPETokenizer {
    Hashtable *vocab;      // Maps string tokens to integer IDs
    char **id_to_token;    // Maps integer IDs back to string tokens
    TokenArray *merges;    // Stores the merge rules, e.g., "t h" -> "th"
    int vocab_size;
} BPETokenizer;

unsigned int hash(const char *str, int size) {
    unsigned int hash_val = 0;
    while (*str) {
        hash_val = (hash_val << 5) + hash_val + *str;
        str++;
    }
    return hash_val % size;
}


FreqTable* create_freq_table(int size) {
    FreqTable *table = malloc(sizeof(FreqTable));
    if (table == nil) sysfatal("malloc: %r");
    table->size = size;
    table->entries = malloc(sizeof(PairFreq*) * size);
    if (table->entries == nil) {
        free(table);
        sysfatal("malloc: %r");
    }
    memset(table->entries, 0, sizeof(PairFreq*) * size);
    return table;
}

PairFreq* get_freq_entry(FreqTable *table, char *key) {
    unsigned int slot = hash(key, table->size);
    for (PairFreq *entry = table->entries[slot]; entry != nil; entry = entry->next) {
        if (strcmp(entry->key, key) == 0) {
            return entry;
        }
    }
    return nil;
}

void add_freq_entry(FreqTable *table, char *key) {
    unsigned int slot = hash(key, table->size);
    PairFreq *entry = table->entries[slot];

    // Check if the key already exists
    while (entry != nil) {
        if (strcmp(entry->key, key) == 0) {
            entry->freq++; // Found it, just increment the frequency
            return;
        }
        entry = entry->next;
    }

    // Key not found, create a new entry
    PairFreq *new_entry = malloc(sizeof(PairFreq));
    if (new_entry == nil) sysfatal("malloc: %r");
    new_entry->key = strdup(key);
    new_entry->freq = 1; // Initialize frequency to 1
    new_entry->next = table->entries[slot];
    table->entries[slot] = new_entry;
}

char* find_most_frequent_pair(FreqTable *table) {
    int max_freq = 0;
    char *most_freq_pair = nil;

    for (int i = 0; i < table->size; i++) {
        for (PairFreq *entry = table->entries[i]; entry != nil; entry = entry->next) {
            if (entry->freq > max_freq) {
                max_freq = entry->freq;
                if (most_freq_pair) {
                    free(most_freq_pair);
                }
                most_freq_pair = strdup(entry->key);
            }
        }
    }
    return most_freq_pair;
}

void free_freq_table(FreqTable *table) {
    for (int i = 0; i < table->size; i++) {
        for (PairFreq *entry = table->entries[i]; entry != nil; ) {
            PairFreq *temp = entry;
            entry = entry->next;
            free(temp->key);
            free(temp);
        }
    }
    free(table->entries);
    free(table);
}


// --- Helper Functions for Hash Tables and TokenArray ---
// (Use the implementations from previous examples)

// --- Hash Table Implementation ---


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

// --- Core BPE Functions ---

// Helper function to merge the most frequent pair
TokenArray*
merge_most_frequent(TokenArray *tokens, char *pair)
{
    TokenArray *new_tokens = malloc(sizeof(TokenArray));
    if (new_tokens == nil) sysfatal("malloc: %r");
    inittokens(new_tokens);
    
    char *space = strchr(pair, ' ');
    if (space == nil) sysfatal("invalid pair format");

    *space = '\0';
    char *first = pair;
    char *second = space + 1;
    
    char *merged_token = smprint("%s%s", first, second);
    if (merged_token == nil) sysfatal("smprint: %r");
    
    for(int i = 0; i < tokens->size; ) {
        if (i < tokens->size - 1 && strcmp(tokens->tokens[i], first) == 0 && strcmp(tokens->tokens[i+1], second) == 0) {
            addtoken(new_tokens, merged_token);
            i += 2;
        } else {
            addtoken(new_tokens, tokens->tokens[i]);
            i++;
        }
    }
    
    freetokens(tokens);
    free(tokens);
    free(merged_token);
    
    return new_tokens;
}

// Training function
BPETokenizer* learn_bpe(char *corpus, int desired_vocab_size) {
    // 1. Initial tokenization (character-level)
    TokenArray *tokens = malloc(sizeof(TokenArray));
    if (tokens == nil) sysfatal("malloc: %r");
    inittokens(tokens);
    char *p = corpus;
    while (*p) {
        char *buf = malloc(2);
        if (buf == nil) sysfatal("malloc: %r");
        buf[0] = *p;
        buf[1] = '\0';
        addtoken(tokens, buf);
        free(buf);
        p++;
    }

    // Allocate the merges array
    TokenArray *merges = malloc(sizeof(TokenArray));
    if (merges == nil) sysfatal("malloc: %r");
    inittokens(merges);

    // 2. Main merging loop
    for (int merge_count = tokens->size; merge_count < desired_vocab_size; merge_count++) {
        FreqTable *freq_table = create_freq_table(tokens->size * 2);
        for (int j = 0; j < tokens->size - 1; j++) {
            char *pair = smprint("%s %s", tokens->tokens[j], tokens->tokens[j + 1]);
            if (pair == nil) sysfatal("smprint: %r");
            add_freq_entry(freq_table, pair);
            free(pair);
        }
        
        char *most_freq_pair = find_most_frequent_pair(freq_table);
        if (most_freq_pair == nil) {
            free_freq_table(freq_table);
            break;
        }

        // Add the most frequent pair to the merges array
        addtoken(merges, most_freq_pair);
        
        // Merge the most frequent pair in the current tokens
        tokens = merge_most_frequent(tokens, most_freq_pair);
        free(most_freq_pair);
        free_freq_table(freq_table);
    }
    
    // 3. Construct and return the BPETokenizer object
    BPETokenizer *bpe = malloc(sizeof(BPETokenizer));
    if (bpe == nil) sysfatal("malloc: %r");
    
    // Store the merges array in the tokenizer
    bpe->merges = merges;
    
    // Create final vocabulary and mappings
    bpe->vocab_size = tokens->size;
    bpe->vocab = create_hashtable(bpe->vocab_size * 2);
    bpe->id_to_token = malloc(sizeof(char*) * bpe->vocab_size);
    if (bpe->id_to_token == nil) sysfatal("malloc: %r");

    for(int i = 0; i < tokens->size; i++) {
        add_entry(bpe->vocab, tokens->tokens[i], i);
        bpe->id_to_token[i] = strdup(tokens->tokens[i]);
    }
    
    freetokens(tokens);
    free(tokens);
    
    return bpe;
}
TokenArray*
bpe_encode(BPETokenizer *bpe, char *text)
{
    // 1. Initial character-level tokenization
    TokenArray *tokens = malloc(sizeof(TokenArray));
    if (tokens == nil) sysfatal("malloc: %r");
    inittokens(tokens);
    char *p = text;
    while (*p) {
        // Allocate a buffer of 2 bytes: one for the char and one for the null terminator.
        char *buf = malloc(2);
        if (buf == nil) sysfatal("malloc: %r");
        
        // Copy the single character *p into the first byte of the buffer.
        buf[0] = *p;
        
        // Add the null terminator to make it a valid C string.
        buf[1] = '\0';
        
        // The addtoken function can now handle the C string.
        addtoken(tokens, buf);
        
        // Free the temporary buffer.
        free(buf);
        
        p++;
    }

    // REMOVE THIS RETURN STATEMENT:
    // return tokens;

	if (bpe == nil || bpe->merges == nil) {
        sysfatal("Invalid BPE tokenizer or merges array.");
    }

    // 2. Apply merge rules iteratively
    for (int i = 0; i < bpe->merges->size; i++) {
        TokenArray *new_tokens = malloc(sizeof(TokenArray));
        if (new_tokens == nil) sysfatal("malloc: %r");
        inittokens(new_tokens);

        char *pair = bpe->merges->tokens[i];
        char *space = strchr(pair, ' ');
        if (space == nil) sysfatal("invalid merge format");
        *space = '\0';
        char *first = pair;
        char *second = space + 1;
        char *merged_token = smprint("%s%s", first, second);

        for (int j = 0; j < tokens->size; ) {
            if (j < tokens->size - 1 && strcmp(tokens->tokens[j], first) == 0 && strcmp(tokens->tokens[j+1], second) == 0) {
                addtoken(new_tokens, merged_token);
                j += 2;
            } else {
                addtoken(new_tokens, tokens->tokens[j]);
                j++;
            }
        }
        
        freetokens(tokens);
        free(tokens);
        tokens = new_tokens;
        free(merged_token);
    }
    
    // 3. Convert final subwords to integer IDs
    TokenArray *ids = malloc(sizeof(TokenArray));
    if (ids == nil) sysfatal("malloc: %r");
    inittokens(ids);
    int id;

    for (int i = 0; i < tokens->size; i++) {
        if (get_value(bpe->vocab, tokens->tokens[i], &id)) {
            char buf[12]; // Buffer for int to string conversion
            sprint(buf, "%d", id);
            addtoken(ids, buf);
        }
    }
    
    freetokens(tokens);
    free(tokens);
    
    return ids;
}

char*
bpe_decode(BPETokenizer *bpe, TokenArray *ids)
{
    // 1. Convert integer IDs to subword tokens
    TokenArray *tokens = malloc(sizeof(TokenArray));
    if (tokens == nil) sysfatal("malloc: %r");
    inittokens(tokens);
    
    for (int i = 0; i < ids->size; i++) {
        int id = atoi(ids->tokens[i]);
        if (id >= 0 && id < bpe->vocab_size) {
            addtoken(tokens, bpe->id_to_token[id]);
        }
    }

    // 2. Concatenate subword tokens to form the final text
    char *decoded_text = nil;
    if (tokens->size > 0) {
        // Concatenate all tokens without adding extra spaces.
        decoded_text = strdup(tokens->tokens[0]);
        if (decoded_text == nil) sysfatal("strdup: %r");

        for (int i = 1; i < tokens->size; i++) {
            char *merged = smprint("%s%s", decoded_text, tokens->tokens[i]);
            free(decoded_text);
            decoded_text = merged;
        }

        // Your existing logic for fixing spaces around punctuation is still valid
        // and should work on the correctly concatenated text.
        char *final_text = nil;
        char *p = decoded_text;
        char *start = decoded_text;
        while (*p) {
            // Check for space followed by punctuation
            if (isspace(*p) && issymbol(*(p+1))) {
                long len = p - start;
                char *temp = malloc(len + 1);
                memmove(temp, start, len);
                temp[len] = '\0';
                if (final_text == nil) {
                    final_text = temp;
                } else {
                    char *merged = smprint("%s%s", final_text, temp);
                    free(final_text);
                    final_text = merged;
                    free(temp);
                }
                start = p + 1;
            }
            p++;
        }
        if (start < p) {
            if (final_text == nil) {
                final_text = strdup(start);
            } else {
                char *merged = smprint("%s%s", final_text, start);
                free(final_text);
                final_text = merged;
            }
        }
        if (final_text == nil) final_text = strdup("");
        free(decoded_text);
        decoded_text = final_text;

    } else {
        // Handle the empty case
        decoded_text = strdup("");
    }
    
    freetokens(tokens);
    free(tokens);
    return decoded_text;
}
// --- Main function to demonstrate usage ---

void
main(int argc, char *argv[])
{
    // Small string test using the BPE tokenizer
    char *textstring = "It's the last he painted, you know, Mrs. Gisburn said with pardonable pride.";
    int desired_vocab_size = 100; // Example size
    BPETokenizer *bpe_tokenizer;
    TokenArray *ids;
    char *decoded_text;
    int i;

    // Learn the BPE tokenizer
    print("--- Learning BPE tokenizer ---\n");
    bpe_tokenizer = learn_bpe(textstring, desired_vocab_size);
    if (bpe_tokenizer == nil) {
        exits("bpe learning failed");
    }

    // Encode the original text
    print("--- Encoding Text ---\n");
    ids = bpe_encode(bpe_tokenizer, textstring);

    // Decode the ids
    print("--- Decoding IDs ---\n");
    decoded_text = bpe_decode(bpe_tokenizer, ids);

    // Print results
    print("--- Original Text ---\n");
    print("%s\n", textstring);
    print("--- Decoded Text ---\n");
    print("%s\n", decoded_text);
    print("--- Encoded Tokens (IDs) ---\n");
    for (i = 0; i < ids->size; i++) {
        print("id[%d]: %s\n", i, ids->tokens[i]);
    }
    print("----------------------------\n");

    // Cleanup for BPE
    freetokens(bpe_tokenizer->merges);
    free(bpe_tokenizer->merges);
    free_hashtable(bpe_tokenizer->vocab);
    for (i = 0; i < bpe_tokenizer->vocab_size; i++) {
        free(bpe_tokenizer->id_to_token[i]);
    }
    free(bpe_tokenizer->id_to_token);
    free(bpe_tokenizer);
    freetokens(ids);
    free(ids);
    free(decoded_text);

    exits(nil);
}


/* backup
void
main(int argc, char *argv[])
{
    // ... (Your previous file reading and parsing code) ...
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
// comment block below for :small string test
// BEGIN COMMENT BLOCK
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
// END COMMENT BLOCK
// small string test: comment block above end

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
    //decoded_text = decode(tokenizer, ids);

    // Print results
    print("--- Original Text ---\n");
    //print("%s\n", filedata); // prod
    print("%s\n", textstring); // small string test
    print("--- Decoded Text ---\n");
    //print("%s\n", decoded_text);
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
    //free(decoded_text);

    // Learn the BPE tokenizer
    // BPETokenizer *bpe = learn_bpe(filedata, DESIRED_VOCAB_SIZE);

    // Encode some text
    // TokenArray *ids = bpe_encode(bpe, "some new text");

    // Decode the ids back
    // char *decoded_text = bpe_decode(bpe, ids);

    // ... (Cleanup) ...

    exits(nil);
}
*/