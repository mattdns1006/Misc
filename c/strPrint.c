#include<stdio.h>
#include<stdlib.h>

void strPrint(char*s);

void strPrint(char*s)
{
	while(*s){
		printf("%p = %c. \n",s,*s);
		s++;
	}
	printf("\n");
}

