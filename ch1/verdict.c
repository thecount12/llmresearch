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
