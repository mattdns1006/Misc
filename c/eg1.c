#include <stdio.h>
#include <stdio.h>

int strlength(char *s)
{
	int n;
	for (n=0; *s != '\0'; s++)
		n++;
	return n;
}
int main(int argc, char *argv[])
{
	char *s;
	s = argv[1];
	printf("%s\n",s);
	int size;
	size = strlength(s);
	printf("Size = %d\n.",size);
}

