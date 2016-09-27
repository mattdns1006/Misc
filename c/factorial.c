#include <stdio.h>

int factorial(int n){
	int i = 0;
	int fac = 1;
	for (i=1; i<=n; i++){
		fac = fac * i;
	}
	return fac;
}

int main(int argc, char* argv[]) {

	int number = 5;
	int fac_five = factorial(number);
	printf(fac_five);
	//printf(number,"! is %d\n",fac_five);
	return 0;
}

