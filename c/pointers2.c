#include<stdio.h>

void main(){
	int *uninit;
	int *nullptd = 0;
	void *vptr;
	int val = 1;
	int *iptr;
	int *backptr;

	iptr = &val;
	vptr = iptr;
	//printf("iptr=%p,vptr=%p\n",(void*)iptr,(void*)vptr);
	printf("iptr=%p,vptr=%p\n",(void*)iptr,(void*)vptr);
	printf("The value of val is %d\n",*iptr);
	//return 0;
}
