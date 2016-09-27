/*
The following iterative sequence is defined for the set of positive integers:

n → n/2 (n is even)
	n → 3n + 1 (n is odd)

	Using the rule above and starting with 13, we generate the following sequence:

	13 → 40 → 20 → 10 → 5 → 16 → 8 → 4 → 2 → 1
	It can be seen that this sequence (starting at 13 and finishing at 1) contains 10 terms. Although it has not been proved yet (Collatz Problem), it is thought that all starting numbers finish at 1.

	Which starting number, under one million, produces the longest chain?

	NOTE: Once the chain starts the terms are allowed to go above one million.
*/

#include<iostream>
#include<string>
#include<cmath>
using namespace std;

inline long long nextCollatzeNumber(long n){
	if(n%2==0){
		// If even
	       return n/2;
	}else{
		// If odd
		return 3*n + 1;
	}
}

int main()
{

	//cout << "Please enter first number of Collatze sequence: "; cin >> seq;
	
	long long decrementor = 0;
	long long largestCount = 0;
	long long largestCollatzeSeqNum = 0;
	while(decrementor!=pow(10,6))
	{
		long long seq = pow(10,6) - decrementor;
		long long init = seq;
		cout << "First number is " << seq << endl;
		bool atOne = 0;
		long long count = 0;

		while(atOne == 0){
			seq = nextCollatzeNumber(seq);

			//cout << "Next number in Collatze sequence is "<< seq  << endl;
			if(seq==1){
				atOne = 1;

				if(count>largestCount){
					largestCount = count;
					largestCollatzeSeqNum = init;
					cout << "Starting value of "<< init << endl;
					cout << "New largest with total count of << " << count << endl;
				}
			}
			count +=1;
		}

		decrementor += 1;

	}
	cout << "Largest Collatze starting number = " << largestCollatzeSeqNum << endl;
	return 0;
}
