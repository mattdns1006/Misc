#include <stdio.h>

#define swap(t,x,y) {t _temp; _temp = x; x = y; y = _temp;}

int main()
{
	int x,y;
	x =3;
	y =5;
	printf("before %d %d \n",x,y);
	swap(int,x,y);
	printf("after %d %d \n",x,y);
	return 0;
}
