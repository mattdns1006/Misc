#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

static int days[13] = {
		{0,31,28,31,30,31,30,31,31,30,31,30,31},
		{0,31,29,31,30,31,30,31,31,30,31,30,31} };

static char *monthName[] = { "Illegal month", "Jan","Feb","Mar","Apr","May","Jun",				 "Jul","Aug","Sep","Oct","Nov","Dec" };

int dayOfYear(int year, int month, int day);
int ass(int year, int month, int day);

int main(int argc, char *argv[])
{
	int year , month, day;
	printf("Enter yyyy mm dd \n");
	scanf("%d %d %d",&year,&month,&day);
	ass(year,month,day);
	dayOfYear(year,month,day);
	
}

int ass (int year, int month, int day)
{
	int count =0;
	while(year>1){
		year /= 10;
		count++;	
	}

	assert( month > 0 && month < 13 );
	assert( day < 32 && day > 0);
	return 0;
}

int dayOfYear(int year, int month, int day)
{
	int leap; //Leap year?
	leap = (year % 4 == 0) ? 1 : 0;

	int ndays = 0;
	int i;
	for(i=1;i<month;i++)
		ndays += days[leap][i];
	ndays += day;

	char *name = monthName[month];
	printf("Number of days passed on %d %s %d = %d\n",day,name,year, ndays);
	return 0;
}
