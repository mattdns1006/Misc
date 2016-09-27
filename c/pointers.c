#include<stdio.h>

int main(){
	int *ptr;
	int val = 1;
	ptr = &val;
	printf("The value of var is %d\n",val);
	printf("It's address is %p\n",ptr);
	int deref = *ptr;
	printf("We can dereference the value from the pointer %d\n",deref);

	return 0;
}
