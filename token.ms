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
at that example. 
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
.LP
asdf
