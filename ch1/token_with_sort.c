#include <u.h>
#include <libc.h>
#include <ctype.h>

// Your TokenArray and parser-related code (as provided previously)

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

void
parser(char *text, TokenArray *arr)
{
	char *start;
	char *end;
	char token_buffer[1024]; // Temporary buffer for tokens

	start = text;
	while (*start != '\0') {
		while (isspace(*start)) {
			start++;
		}
		if (*start == '\0') {
			break;
		}

		if (*start == '-' && *(start + 1) == '-') {
			addtoken(arr, "--");
			start += 2;
			continue;
		}

		if (issymbol(*start)) {
			token_buffer[0] = *start;
			token_buffer[1] = '\0';
			addtoken(arr, token_buffer);
			start++;
			continue;
		}

		end = start;
		while (*end != '\0' && !isspace(*end) && !issymbol(*end) && !(*end == '-' && *(end + 1) == '-')) {
			end++;
		}

		if (end > start) {
			long word_len = end - start;
			if (word_len < sizeof(token_buffer)) {
				strncpy(token_buffer, start, word_len);
				token_buffer[word_len] = '\0';
				addtoken(arr, token_buffer);
			}
		}
		start = end;
	}
}

// Comparison function for qsort to sort strings
int cmpstr(const void* a, const void* b) {
    const char* aa = *(const char**)a;
    const char* bb = *(const char**)b;
    return strcmp(aa, bb);
}

void
main(int argc, char *argv[])
{
    int fd;
    char *filename = "the-verdict.txt";
    char *filedata;
    long n;
    Dir *d;
    TokenArray all_tokens;
    TokenArray unique_tokens;
    int i;

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

    // 1. Parse all tokens
    inittokens(&all_tokens);
    parser(filedata, &all_tokens);
    free(filedata);

    if (all_tokens.size == 0) {
        print("No tokens found.\n");
        exits(nil);
    }

    // 2. Sort the tokens
    qsort(all_tokens.tokens, all_tokens.size, sizeof(char*), cmpstr);

    // 3. Create a new array with only unique tokens
    inittokens(&unique_tokens);
    addtoken(&unique_tokens, all_tokens.tokens[0]); // Add the first token

    for (i = 1; i < all_tokens.size; i++) {
        if (strcmp(all_tokens.tokens[i], all_tokens.tokens[i-1]) != 0) {
            addtoken(&unique_tokens, all_tokens.tokens[i]);
        }
    }
    freetokens(&all_tokens); // Clean up the original array

    // 4. Print the unique sorted tokens and the vocabulary size
    print("--- Unique Sorted Tokens ---\n");
    for (i = 0; i < unique_tokens.size; i++) {
        print("token[%d]: \"%s\"\n", i, unique_tokens.tokens[i]);
    }
    print("----------------------------\n");
    print("Vocabulary size: %d\n", unique_tokens.size);

    freetokens(&unique_tokens);
    exits(nil);
}
