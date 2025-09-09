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
	static char *symbols = ",.:;?!_\"()\'\t\n ";
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
		while (*end != '\0' && !isspace(*end) && !issymbol(*end) && !(*end == '-' && *(end + 1) == '-')) {
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
  int fd;
  int i;
  char *filedata;
  long n;
  Dir *d;
  char *filename = "the-verdict.txt";
  d = dirstat(filename);
  if (d == nil) {
    print("could not stat file '%s': %r\n", filename);
    exits("file error");
  }
  // Allocate memory for file content (+1 for null terminator)
  filedata = malloc(d->length + 1);
  if (filedata == nil) {
    print("malloc failed: %r\n", filename);
    free(d);
    exits("memory error");
  }
  free(d);
  // Open file for reading
  fd = open(filename, OREAD);
  if (fd < 0) {
    print("could not open file '%s': %r\n", filename);
    free(filedata);
    exits("file error");
  }
  // Read the entire file into the buffer
  n = read(fd, filedata, d->length);
  if (n < 0) {
    print("error reading file '%s': %r\n", filename);
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
  print("--- Token Array ---\n");

  for (i = 0; i < tokens.size; i++) {
    print("token[%d]: \"%s\"\n", i, tokens.tokens[i]);
  }
  print("-------------------\n");

  freetokens(&tokens);
  exits(nil);
}
