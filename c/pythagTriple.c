#include<stdio.h>

int main(void){

	/* We use Eulics formulae a = m^2 - n^2, b = 2mn,  c = m^2 + n^2 where 
	 * a b c form a pythag triple.
	 */
	int n=1;
	int m=2;
	int a =3, b = 4, c= 5;
	int sum;
	int foundNumbers = 0;
	printf("a^2 + b^2 = %d\n",square(a) + square(b));
	int targetSum = 1000;

	for(m=2;m<=1000;m++){

		for(n=1;n<m;n++){
		
			printf("m = %d, n = %d\n",m,n);

			a = eulicidA(m,n);
			b = eulicidB(m,n);
			c = eulicidC(m,n);

			printf("a = %d, b = %d , c = %d\n",a,b,c);
			sum = a + b + c;

			if(sum>targetSum){
				break;
			}

			if(sum==targetSum){
				printf("Found Numbers....breaking");
				foundNumbers = 1;
				break;
			}


		}
		
		if(foundNumbers==1){
			break;
		}

	}
	printf("Numbers are = %d,%d,%d",a,b,c);
	int productNumbers = a*b*c;
	printf("Their product is = %d",productNumbers);

	return 0;

}



int square(int x){
	return x*x;
}
int eulicidA(int m,int n){
	return square(m)-square(n);
}
int eulicidB(int m,int n){
	return 2*m*n;
}
int eulicidC(int m,int n){
	return square(m)+square(n);
}

