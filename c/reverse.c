#include<stdio.h>
#include<stdlib.h>

// Adapt ideas of printd to convert integer to string
void reverse(char str[], char out[]);
int length(char str[]);
void reverseRec(char in[], int i, int j);

int main(int argc, char *argv[])
{
	char in[100], out[100];
	printf("Enter string..\n");
	scanf("%s",in);
	printf("you entered %s ..\n",in);
	reverse(in,out);
	printf("reversed = %s \n",out);
	printf("\n\n");
	printf("Inplace reversed before = %s\n",in);
	int i,j;
	i = 0;
	j = length(in)-1;
	reverseRec(in,i,j);
	printf("Inplace reversed after = %s\n",in);

}

void swap(char *in, int i, int j)
{
	char temp;
	temp = in[i];
	in[i] = in[j];
	in[j] = temp;
}

void reverseRec(char in[],int i,int j)
{
	if(i<j){
		swap(in,i,j);
		i++;
		j--;
		reverseRec(in,i,j);
	}
}

void reverse(char str[], char out[])
{
	int length =0;
	while(1){
		if(str[length] == '\0')
			break;
		length++;
	}

	printf("String has len = %d \n",length);

	int j = 0;
	int lengthCopy = length;
	for(j=0;j<lengthCopy;j++){
		out[j] = str[length-1];
		length--;
	}
}

int length(char str[])
{
	int length =0;
	while(1){
		if(str[length] == '\0')
			break;
		length++;
	}
	return length;
}


