#include "ccv.h"

int main(int argc, char** argv)
{
	ccv_array_t* array = ccv_array_new(2, 4);
	int i;
	i = 1;
	ccv_array_push(array, &i);
	i = 2;
	ccv_array_push(array, &i);
	i = 3;
	ccv_array_push(array, &i);
	i = 4;
	ccv_array_push(array, &i);
	i = 5;
	ccv_array_push(array, &i);
	for (i = 0; i < array->rnum; i++)
		printf("%d : %d\n", i, ((int*)ccv_array_get(array, i))[0]);
	ccv_array_clear(array);
	i = 3;
	ccv_array_push(array, &i);
	i = 4;
	ccv_array_push(array, &i);
	i = 5;
	ccv_array_push(array, &i);
	for (i = 0; i < array->rnum; i++)
		printf("%d : %d\n", i, ((int*)ccv_array_get(array, i))[0]);
	ccv_array_free(array);
	return 0;
}
