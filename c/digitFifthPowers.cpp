/*
surprisingly there are only three numbers that can be written as the sum of fourth powers of their digits:

1634 = 14 + 64 + 34 + 44
8208 = 84 + 24 + 04 + 84
9474 = 94 + 44 + 74 + 44
As 1 = 14 is not a sum it is not included.

The sum of these numbers is 1634 + 8208 + 9474 = 19316.

Find the sum of all the numbers that can be written as the sum of fifth powers of their digits
*/

#include<iostream>
using namespace std;
#include<cmath>

template<int D> int getDigit(int val) {return getDigit<D-1>(val/10);}
template<> int getDigit<1>(int val) {return val % 10;}

int main(){

	int n = 1000;
	for(int i=0;i<=n;++i){
		cout << i << " has " << getDigit<i>(123) <<"\n";
		
	}

}
