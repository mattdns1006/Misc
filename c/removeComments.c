#include <stdio.h>
#include <stdlib.h>
#include <string.h>

char* ReadFile(char *filename)
{	
	char *buffer = NULL;
	int stringSize, readSize;
	FILE *handler = fopen(filename, "r");
	if (handler)
	{
		fseek(handler, 0, SEEK_END);
		stringSize = ftell(handler);
		rewind(handler);
		buffer = (char*) malloc(sizeof(char) * (stringSize + 1));
		readSize = fread(buffer,sizeof(char), stringSize, handler);
		buffer[stringSize] = '\0';
		if (stringSize != readSize){
			free(buffer);
			buffer = NULL;
		}
		fclose(handler);
	}
	return buffer;
}

int main(int argc, char *argv[])
{
	char *fname = argv[1];
	char *string = ReadFile(fname);	
	int len = strlen(string);
	typedef enum{ true = 1, false = 0} bool;
	if(string)
	{	
		//puts(string);
		int i=0;
		char c, cNext;
		bool or1, or2;
		while ( string[i] != '\0'){
			c = string[i];
			cNext = string[i+1];
			//if((c == '/' & cNext == '/') | (c == '/' & cNext == '*'))
			or1 = ((cNext == '/') | (cNext == '*'));
			or2 = ((c == '/') | (c == '*'));
			if(	((c == '/') & or1) | ((cNext == '/') & or2)	      )
				printf("!!!HERE!!!");

			printf("%c",c);
			i++;
		}
		free(string);
	}

	printf("Length of file = %d characters\n.",len);
	return 0;
}
