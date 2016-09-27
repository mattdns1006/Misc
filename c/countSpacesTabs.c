#include <stdio.h>

int main()
{
	int c, nt, ns;
	ns = nt = 0;
	printf("EOF = %d\n",EOF);
	while ((c = getchar())!= '\n'){

		if (c == '\t')
			++nt;
		else if (c == ' ')
			++ns;
	}
	printf("There were %d tabs and %d spaces. \n",nt,ns);
	return 0;
}
