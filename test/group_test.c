#include "ccv.h"

int is_equal(const void* r1, const void* r2, void* data)
{
	int a = *(int*)r1;
	int b = *(int*)r2;
	return a == b;
}

int main(int argc, char** argv)
{
	ccv_array_t* array = ccv_array_new(2, 4);
	int i;
	i = 1;
	ccv_array_push(array, &i);
	i = 2;
	ccv_array_push(array, &i);
	i = 2;
	ccv_array_push(array, &i);
	i = 2;
	ccv_array_push(array, &i);
	i = 5;
	ccv_array_push(array, &i);
	i = 3;
	ccv_array_push(array, &i);
	i = 4;
	ccv_array_push(array, &i);
	i = 5;
	ccv_array_push(array, &i);
	ccv_array_t* idx = NULL;
	ccv_array_group(array, &idx, is_equal, 0);
	for (i = 0; i < array->rnum; i++)
		printf("%d : %d\n", *(int*)ccv_array_get(array, i), *(int*)ccv_array_get(idx, i));
	ccv_array_free(array);
	ccv_array_free(idx);
	return 0;
}
