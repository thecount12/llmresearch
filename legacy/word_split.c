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
    } // && ar[i]!='.'
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
