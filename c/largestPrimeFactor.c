#include<stdio.h>
//#include<gmp.h>
/*
		The prime factors of 13195 are 5, 7, 13 and 29.

			What is the largest prime factor of the number 600851475143 ?
			*/

int main(int argc, char** argv){

	const long x = 600851475143;
	long j=3;
	long long i=0;
	long long primes[100000] ={2} ;
	long primeIndex = 0;
	int isPrime = 1; // Is the number prime or not
	long long largestPrimeFactor = 2;

	for(j=3;j<x/2;j++){
		/*printf("Checking weather or not %d is prime, i.e. whether to add it to primes.\n",j);
		printf("Value of j = %d\n",j);
		//Loop over each prime value in primes
		//*/
		for(i=0;i<=primeIndex;i++){
			/*
			printf("Value of i = %d\n",i);
			printf("Value of primes[i] = %d\n",primes[i]);
				*/
			if(j%primes[i]==0){
				//Number is divisible by a prime hence it is not a prime number (exit loop)
				isPrime = 0;
				break;
			}
		}
		if(isPrime ==1){ 
			primeIndex = primeIndex + 1;
			primes[primeIndex] = j;
			//printf("%ld\n is a prime number.",j);
			if(x%j==0){
				largestPrimeFactor = j;
			}

		}
		isPrime = 1; //Reset to true agin for next j

	printf("Largest prime number is = %llu\n",largestPrimeFactor);	
	}
	printf("Largest prime number is = %llu\n",largestPrimeFactor);	
	return 0;
}

