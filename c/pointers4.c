#include<stdio.h>

void main(){

	struct person {
		int age;
		char *name;
	};
	struct person first;
	struct person *ptr;
	first.age = 21;
	char *fullname = "fullname";
	first.name = "fullname";
	ptr = &first;
	printf("age = %d, name = %s\n",first.age,ptr->name);
}
