#include <u.h>
#include <libc.h>

void
main(int, char* [])
{
  int i;
  char ar[] = "Hello, world!; Is This--, a new: test? Or not number (\"one\") or [\'two\']";
  print("['");
  for (i = 0; i < sizeof(ar); i++) {
    if (ar[i]==' ') { // space
      print("', '");
    }
    if (ar[i]!=' ' && ar[i]!=',' && ar[i]!='.' && ar[i]!=':' && ar[i]!='?'
        && ar[i]!=';' && ar[i]!='!' && ar[i]!='(' && ar[i]!=')') { //symbols
      print("%c", ar[i]);
    }
    if (ar[i]==',') {
      print("'%c ',", ar[i]); // comma
    }
    if (ar[i]=='.') {
      print("', '%c", ar[i]); // period
    }
    if (ar[i]==':') {
      print("', '%c", ar[i]); // colon
    }
    if (ar[i]=='?') {
      print("', '%c", ar[i]); // question mark
    }
	if (ar[i]==';') {
      print("', '%c", ar[i]); // semi colon
	}
	if (ar[i]=='!') {
      print("', '%c", ar[i]); // exlimation
    }
	if (ar[i]=='(') {
      print("%c', '", ar[i]); // left parenthesis
    }
	if (ar[i]==')') {
      print("', '%c", ar[i]); // right parenthesis
    }
  }
  print("']\n");
}
