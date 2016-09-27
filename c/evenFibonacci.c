#include<stdio.h>

// Even fibonacci numbers
/*
int main()
{
	//Initialize array to first two numbers
	int fibonacciNumbers[2] = {1,2};
	int last = (sizeof(fibonacciNumbers)/sizeof(fibonacciNumbers[0]));
	printf("Size of array initialized to %d\n",last);
	int i = 2;
	int largestNumberOfDigits = 0;
	while(largestNumberOfDigits<10000){
		fibonacciNumbers[i]=fibonacciNumbers[i-1]+fibonacciNumbers[i-2];
		printf("Fibonacci numbers so far are: %d\n",fibonacciNumbers[i]);
		largestNumberOfDigits = countDigits(fibonacciNumbers[i]);
		printf("Number of digits in last number %d\n",largestNumberOfDigits);
		++i;
	}

	return 0;
}
*/

int main()
{
	//Initialize array to first two numbers
	int fib1 = 1, fib2 = 2, fib3;	
	int numDigits = 1;
	int count = 0;
	int tempFib2 = 0;
	while(numDigits<100){
		fib3 = fib1 + fib2;
		tempFib2 = fib2;
		numDigits = countDigits(fib3);
		printf("Iterations = %d\n",count);
		fib2 = fib3;
		fib1 = tempFib2;
		printf("Current Fibonacci number %d\n", fib3);
		printf("This number has %d digits\n", numDigits);
		

	}

	return 0;
}


int countDigits(int x){

	// Counts the number of digits of a number
	int count = 0;
	while(x!=0){
		x/=10;
		++count;
	}
	printf("Num of digits = %d",count);
	return count;
}
