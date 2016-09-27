#include<stdio.h>
#include<stdlib.h>
#include"strPrint.c"

int strEnd(char* s1, char* s2);
void strPrint(char*s);
int strLength(char*s);

int main()
{
	char str1[20] = "";
	char str2[20] = "";
	printf("Enter first and second strings. \n");
	scanf("%s %s",str1,str2);
	printf("Strings \n");
	strPrint(str1);
	strPrint(str2);
	int end = strEnd(str1,str2);
	printf("Result = %d. \n",end);
}

int strEnd(char* s1, char* s2)
{

	int l1, l2;
	l1 = strLength(s1);
	l2 = strLength(s2);
	
	printf("Lengths of strings = {%d, %d} \n",l1,l2);

	while(*s1){
		s1++;
	}
	while(*s2){
		s2++;
	}
	if(l2 > l1){
		printf("length(s2) > length(s1)\n");
		return 0;
	}
	int i = 0;
	int t;
	while(1){
		if(i>l2)
			break;
		s1--;
		s2--;
		i++;
		t = (*s1 == *s2) ? 1 : 0;
		if(t == 0)
			return 0;
		printf("%c %c (%d) \n",*s1,*s2,t);
	}
		
	return 1;
}

int strLength(char *s)
{
	int len =0;	
	while(*s++)
		len++;
	return len;
}
		


