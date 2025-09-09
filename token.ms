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
  // print("input file: %s\n", buf);  // testing
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
; 6c -FVw word_split.c
; 6l -o wordsplit word_split.6
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
.LP
As you can seee this is extra work to display the proper results for visualization. A
case block could prove useful. 