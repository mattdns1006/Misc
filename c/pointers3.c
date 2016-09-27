#include<stdio.h>

void main(){
	int myarray[4] = {1,2,3,4};
	int *ptr = myarray;
	printf("Address of array is %p\n",ptr);
	printf("Address of first element of array is %p\n",&myarray[0]);
	int i;
	for(i=0;i<4;i++){
		printf("ith element is %d\n",myarray[i]);
		printf("ith element's pointer is %p\n",&myarray[i]);
	}
}
