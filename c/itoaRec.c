#include<stdio.h>
#include<stdlib.h>

// Adapt ideas of printd to convert integer to string

void intToStrRec(int n, char out[])
{
	static int i;
	if(n/10){
		printf("Recursive step %d \n",n);
		intToStrRec(n/10, out);
	}else{
	i = 0;
		if(n<0)
			out[i++]='-';
	}
	printf("string so far %s \n",out);
	out[i++] = abs(n) % 10 + '0';
	out[i] = '\0';
}

int main(int argc, char *argv[])
{
	int eg;
	char egC[100]; 
	printf("Convert 1 unit int %d to string eg \n",eg);
	scanf("%d",&eg);
	int i=0;



	int number;
	char out[100];
	printf("Enter int..\n");
	scanf("%d",&number);
	intToStrRec(number,out);
	printf("%s \n",out);
}
