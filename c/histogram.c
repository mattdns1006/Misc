#include <stdio.h>
#define WORDSIZE 10

/* Prints histogram of words in a sentance */ 

int main()
{
	char word, line; 
	int hist[WORDSIZE];
	int i,j, nWords, wordSize;
	nWords = wordSize = 0;
	for(i=0;i< WORDSIZE;i++)
		hist[i] =0;

	while((line = getchar()) != '\n')
		if(line == ' ')	{
			++nWords;
			++hist[wordSize];

			printf("Total number of letters = %d \n", wordSize);
			wordSize = 0;
		}
		else{
			++wordSize;
		}

	printf("Histogram ==> \n ");
	int peak =0;
	for(i=0;i< WORDSIZE;i++)
		if(hist[i] > peak)
			peak = hist[i];
		printf("%d" , hist[i]);

	printf("\n");

	// Actual histogram 
	char geom[WORDSIZE][peak];
	for(i = 0; i < WORDSIZE; i ++)
		for(j = 0; j < peak; j++){
			if(hist[i] > j)
				geom[i][j] = '-';
			else{
				geom[i][j] = '\0';
			}
	}
	// Print hist  
	for(i = 0; i < WORDSIZE; i ++){
		printf("%d ",i);
		for(j = 0; j < peak; j++){
			printf("%c" , geom[i][j]);
		}
		printf("\n");
	}
	

	printf("Total number of words = %d \n", nWords);
	return 0;
}
