#include<stdio.h>
#include<stdlib.h>

void strCat(char* s, char* t);
void strPrint(char*s);

int main()
{
	char str1[20] = "";
	char str2[20] = "";
	printf("Enter first and second strings. \n");
	scanf("%s %s",str1,str2);
	printf("Strings before \n");
	strPrint(str1);
	strPrint(str2);

	printf("Concatenated \n");
	strCat(str1,str2);
	strPrint(str1);
		
}

void strPrint(char*s)
{
	while(*s){
		printf("%p = %c. \n",s,*s);
		s++;
	}
	printf("\n");
}

void strCat(char *s, char *t)
{
	while(*s)
		s++;
	while(*t){
		*s++ = *t;
		t++;
	}
}
