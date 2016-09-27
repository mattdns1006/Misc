#include <stdio.h>
int strIdx(char s[], char t[]);

int main(int argc, char *argv[])
{
	if(argv[1] == NULL | argv[2] == NULL)
	{
		printf("Missing args. \n");
		return -1;
	}
	strIdx(argv[1],argv[2]);
	return 0;
}

int strIdx(char s[], char t[])
{
	int i, j, k;
	for (i = 0; s[i] != '\0'; i++)
		continue;
	for (j = i-1 ; j >= 0; j--){
		if(s[j] == t[0]){
			printf("position of %c in %s == s[%d]. \n",t[0], s, j);
			return 0;
		}
	}
	return -1;
}

