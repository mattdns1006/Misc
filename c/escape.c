#include <stdio.h>

//int escape(s, t)

int main()
{
	int c;
	char *esc;
	while((c=getchar()) != EOF){
		switch(c){
			case '\n':
				esc = " \\n ";
				break;
			case '\t':
				esc = " \\t ";
				break;
			default:

				break;
		}
		if(esc)
			printf("%s",esc);
		esc = NULL;
		printf("%c",c);

	}
	printf("finished");
	return 0;
}
