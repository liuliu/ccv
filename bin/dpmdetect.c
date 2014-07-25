#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

unsigned int get_current_time()
{
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int count_models(const char* directory, int *model_list)
{
	int num_models = 0;

	FILE* r = fopen(directory, "rt");
	if(r)
	{
		size_t len = 1024;
		char* line = (char*)malloc(len);
		ssize_t read;
		/* check if it is a model file */
		if((read = getline(&line, &len, r)) != -1)
		{
			while(read > 1 && isspace(line[read - 1]))
				read--;
			line[read] = 0;
			if (strlen(line) == 1 && line[0] == '.')
				return 1;
			/* if it reaches here, it must be a list of model files */
			(*model_list) = 1;
			num_models++;
		}
		while((read = getline(&line, &len, r)) != -1)
			num_models++;
	}
	return num_models;
}

ccv_dpm_mixture_model_t** read_models(const char* directory, int num_models, int model_list)
{
	ccv_dpm_mixture_model_t** models = (ccv_dpm_mixture_model_t**)ccmalloc(sizeof(ccv_dpm_mixture_model_t*) * num_models);
	if (num_models > 1 || model_list)
	{
		int i;
		FILE* r = fopen(directory, "rt");
		size_t len = 1024;
		char* line = (char*)malloc(len);
		ssize_t read;
		for (i = 0; i < num_models; i++)
		{
			if((read = getline(&line, &len, r)) != -1)
			{
				while(read > 1 && isspace(line[read - 1]))
					read--;
				line[read] = 0;
				models[i] = ccv_dpm_read_mixture_model(line);
			}
		}
	}
	else
		models[0] = ccv_dpm_read_mixture_model(directory);

	return models;
}

int main(int argc, char** argv)
{
	assert(argc >= 3);
	int i, j, num_models, model_list = 0;
	ccv_enable_default_cache();
	ccv_dense_matrix_t* image = 0;
	ccv_read(argv[1], &image, CCV_IO_ANY_FILE);
	num_models = count_models(argv[2], &model_list);
	ccv_dpm_mixture_model_t** models = read_models(argv[2], num_models, model_list);
	if (image != 0)
	{
		unsigned int elapsed_time = get_current_time();
		ccv_array_t* seq = ccv_dpm_detect_objects(image, models, num_models, ccv_dpm_default_params);
		elapsed_time = get_current_time() - elapsed_time;
		if (seq)
		{
			for (i = 0; i < seq->rnum; i++)
			{
				ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
				printf("%d %d %d %d %f %d\n", comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->classification.confidence, comp->pnum);
				for (j = 0; j < comp->pnum; j++)
					printf("| %d %d %d %d %f\n", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].classification.confidence);
			}
			printf("total : %d in time %dms\n", seq->rnum, elapsed_time);
			ccv_array_free(seq);
		} else {
			printf("elapsed time %dms\n", elapsed_time);
		}
		ccv_matrix_free(image);
	} else {
		FILE* r = fopen(argv[1], "rt");
		if (argc == 4)
			chdir(argv[3]);
		if(r)
		{
			size_t len = 1024;
			char* file = (char*)malloc(len);
			ssize_t read;
			while((read = getline(&file, &len, r)) != -1)
			{
				while(read > 1 && isspace(file[read - 1]))
					read--;
				file[read] = 0;
				image = 0;
				ccv_read(file, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
				assert(image != 0);
				ccv_array_t* seq = ccv_dpm_detect_objects(image, models, num_models, ccv_dpm_default_params);
				if (seq != 0)
				{
					for (i = 0; i < seq->rnum; i++)
					{
						ccv_root_comp_t* comp = (ccv_root_comp_t*)ccv_array_get(seq, i);
						printf("%s %d %d %d %d %f %d\n", file, comp->rect.x, comp->rect.y, comp->rect.width, comp->rect.height, comp->classification.confidence, comp->pnum);
						for (j = 0; j < comp->pnum; j++)
							printf("| %d %d %d %d %f\n", comp->part[j].rect.x, comp->part[j].rect.y, comp->part[j].rect.width, comp->part[j].rect.height, comp->part[j].classification.confidence);
					}
					ccv_array_free(seq);
				}
				ccv_matrix_free(image);
			}
			free(file);
			fclose(r);
		}
	}
	ccv_drain_cache();
	for (i = 0; i < num_models; i++)
		ccv_dpm_mixture_model_free(models[i]);
	ccfree(models);
	return 0;
}
