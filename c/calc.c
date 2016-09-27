#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#define MAX 100

/* Calculator */
int main()
{
	char commands[MAX] = {'1'};	
	int isDig[MAX] = {0};
	char c;
	char *top = commands;
	char *bottom = commands;
	char *i;
	double operands;
	double *ptrOperands = &operands;

	puts("Press '. Enter' to exit");
	while((c = getchar()) != EOF){

		if(c == '\n'){
			
			for(i=bottom; i< top; i++){
				if(isdigit(*i)){
					operands = (double)atof(i);
					printf("address %p = %f is digit\n",&operands,*ptrOperands);

				}else{
					printf("P = %p, V = %c\n",i,*i);
				}
			}
		}
		*top = c;
		top++;

	}	
	
	return 0;
}

