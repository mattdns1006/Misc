#include<stdio.h>
#include<vector>
#include<iostream>
#include<algorithm>
#include<string>
using namespace std;
int factorial(int x){
	if(x==1){
		return 1;
	}else{
		return x*factorial(x-1);
	}
}

int main(){

	int myints[] = {0,1,2,3,4,5,6,7,8,9};
	int nDigits= sizeof(myints)/sizeof(myints[0]);
	long nPerms = factorial(nDigits);
	cout << "Number of digits = " <<  nDigits << endl;
	cout << "Number of permutations is "<< nPerms  << endl;
	
	long count = 0;
	for(int i=0;i<1e6;++i){
		
		for(int j=0;j<nDigits;++j){
			cout<<myints[j];
		}
		cout << endl;
		std::next_permutation(myints,myints+nDigits);
	}
		
	return 0;
}
