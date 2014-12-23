#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

#define BATCH_SIZE (8)

int main(int argc, char** argv)
{
	assert(argc >= 3);
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
	if (image != 0)
	{
		ccv_convnet_t* convnet = ccv_convnet_read(0, argv[2]);
		ccv_dense_matrix_t* input = 0;
		ccv_convnet_input_formation(convnet->input, image, &input);
		ccv_matrix_free(image);
		unsigned int elapsed_time = get_current_time();
		ccv_array_t* rank = 0;
		ccv_convnet_classify(convnet, &input, 1, &rank, 5, 1);
		elapsed_time = get_current_time() - elapsed_time;
		int i;
		for (i = 0; i < rank->rnum - 1; i++)
		{
			ccv_classification_t* classification = (ccv_classification_t*)ccv_array_get(rank, i);
			printf("%d %f ", classification->id + 1, classification->confidence);
		}
		ccv_classification_t* classification = (ccv_classification_t*)ccv_array_get(rank, rank->rnum - 1);
		printf("%d %f\n", classification->id + 1, classification->confidence);
		printf("elapsed time %dms\n", elapsed_time);
		ccv_array_free(rank);
		ccv_matrix_free(input);
		ccv_convnet_free(convnet);
	} else {
		FILE* r = fopen(argv[1], "rt");
		if (argc == 4)
			chdir(argv[3]);
		if(r)
		{
			ccv_convnet_t* convnet = ccv_convnet_read(1, argv[2]);
			int i, j, k = 0;
			ccv_dense_matrix_t* images[BATCH_SIZE] = {
				0
			};
			size_t len = 1024;
			char* file = (char*)malloc(len);
			ssize_t read;
			while((read = getline(&file, &len, r)) != -1)
			{
				while(read > 1 && isspace(file[read - 1]))
					read--;
				file[read] = 0;
				if (images[k % BATCH_SIZE] != 0)
					ccv_matrix_free(images[k % BATCH_SIZE]);
				ccv_dense_matrix_t* image = 0;
				ccv_read(file, &image, CCV_IO_ANY_FILE | CCV_IO_RGB_COLOR);
				assert(image != 0);
				images[k % BATCH_SIZE] = 0;
				ccv_convnet_input_formation(convnet->input, image, images + (k % BATCH_SIZE));
				ccv_matrix_free(image);
				++k;
				if (k % BATCH_SIZE == 0)
				{
					ccv_array_t* ranks[BATCH_SIZE] = {
						0
					};
					ccv_convnet_classify(convnet, images, 1, ranks, 5, BATCH_SIZE);
					for (i = 0; i < BATCH_SIZE; i++)
					{
						for (j = 0; j < ranks[i]->rnum - 1; j++)
						{
							ccv_classification_t* classification = (ccv_classification_t*)ccv_array_get(ranks[i], j);
							printf("%d %f ", classification->id + 1, classification->confidence);
						}
						ccv_classification_t* classification = (ccv_classification_t*)ccv_array_get(ranks[i], ranks[i]->rnum - 1);
						printf("%d %f\n", classification->id + 1, classification->confidence);
						ccv_array_free(ranks[i]);
					}
				}
			}
			if (k % BATCH_SIZE != 0)
			{
				if (k < BATCH_SIZE) // special casing this
					for (i = k; i < BATCH_SIZE; i++)
						images[i] = images[0]; // padding to BATCH_SIZE batch size
				ccv_array_t* ranks[BATCH_SIZE] = {
					0
				};
				ccv_convnet_classify(convnet, images, 1, ranks, 5, BATCH_SIZE);
				for (i = 0; i < (k % BATCH_SIZE); i++)
				{
					for (j = 0; j < ranks[i]->rnum - 1; j++)
					{
						ccv_classification_t* classification = (ccv_classification_t*)ccv_array_get(ranks[i], j);
						printf("%d %f ", classification->id + 1, classification->confidence);
					}
					ccv_classification_t* classification = (ccv_classification_t*)ccv_array_get(ranks[i], ranks[i]->rnum - 1);
					printf("%d %f\n", classification->id + 1, classification->confidence);
					ccv_array_free(ranks[i]);
				}
				for (i = (k % BATCH_SIZE); i < BATCH_SIZE; i++)
					ccv_array_free(ranks[i]);
				for (i = 0; i < ccv_min(BATCH_SIZE, k); i++)
					ccv_matrix_free(images[i]);
			}
			ccv_convnet_free(convnet);
			free(file);
			fclose(r);
		}
	}
	ccv_drain_cache();
	return 0;
}
