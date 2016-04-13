#include "ccv.h"
#include <sys/time.h>
#include <ctype.h>

static unsigned int get_current_time(void)
{
	struct timeval tv;
	gettimeofday(&tv, 0);
	return tv.tv_sec * 1000 + tv.tv_usec / 1000;
}

int main(int argc, char** argv)
{
    int interval,min_neighbors, scale_invariant, direction, size ,low_thresh,high_thresh;
    int max_height,min_height,min_area, letter_occlude_thresh;
    int intensity_thresh, letter_thresh, breakdown; 
    double height_ratio, distance_ratio, intersect_ratio, elongate_ratio,breakdown_ratio;
    double same_word_thresh[2],aspect_ratio,std_ratio, thickness_ratio;
    int opt;
    while ((opt = getopt(argc,argv,"a:b:c:d:e:f:g:h:i:j:k:l:m:n:o:p:q:r:s:t:u:v:w:x:")) != -1)
    {
        switch (opt)
        {
            case 'a': interval=atoi(optarg); break;
            case 'b': min_neighbors=atoi(optarg); break;
            case 'c': scale_invariant=atoi(optarg); break;
            case 'd': direction=atoi(optarg); break;
            case 'e': same_word_thresh[0]=atof(optarg); break;
            case 'f': same_word_thresh[1]=atof(optarg); break;
            case 'g':size=atoi(optarg); break;
            case 'h':low_thresh=atoi(optarg); break;
            case 'i':high_thresh=atoi(optarg); break;
            case 'j':max_height=atoi(optarg); break;
            case 'k':min_height=atoi(optarg); break;
            case 'l':min_area=atoi(optarg); break;
            case 'm':letter_occlude_thresh=atoi(optarg); break;
            case 'n':aspect_ratio=atof(optarg); break;
            case 'o':std_ratio=atof(optarg); break;
            case 'p':thickness_ratio=atof(optarg); break;
            case 'q':height_ratio=atof(optarg); break;
            case 'r':intensity_thresh=atoi(optarg); break;
            case 's':letter_thresh=atoi(optarg); break;
            case 't':distance_ratio=atof(optarg); break;
            case 'u':intersect_ratio=atof(optarg); break;
            case 'v':elongate_ratio=atof(optarg); break;
            case 'w':breakdown=atoi(optarg); break;
            case 'x':breakdown_ratio=atof(optarg); break;
        }
    }

    ccv_swt_param_t ccv_swt_mine_params = 
    {
        .interval = interval,
        .same_word_thresh = { same_word_thresh[0], same_word_thresh[1] }, 
        .min_neighbors = min_neighbors,
        .scale_invariant = scale_invariant,
        .size = size,
        .low_thresh = low_thresh,
        .high_thresh = high_thresh,
        .max_height = max_height,
        .min_height = min_height,
        .min_area = min_area,
        .letter_occlude_thresh = letter_occlude_thresh,
        .aspect_ratio = aspect_ratio,
        .std_ratio = std_ratio,
        .thickness_ratio = thickness_ratio,
        .height_ratio = height_ratio,
        .intensity_thresh = intensity_thresh,
        .distance_ratio = distance_ratio,
        .intersect_ratio = intersect_ratio,
        .letter_thresh = letter_thresh,
        .elongate_ratio = elongate_ratio,
        .breakdown = breakdown,
        .breakdown_ratio = breakdown_ratio,
    };

    ccv_enable_default_cache();
    ccv_dense_matrix_t* image = 0;
    char *filename=argv[optind++];
    ccv_read(filename, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
    if (image != 0)
    {
        unsigned int elapsed_time = get_current_time();
        ccv_array_t* words = ccv_swt_detect_words(image, ccv_swt_mine_params);
        elapsed_time = get_current_time() - elapsed_time;
        if (words)
        {
            int i;
            for (i = 0; i < words->rnum; i++)
            {
                ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
                printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);
            }
            printf("total : %d in time %dms\n", words->rnum, elapsed_time);
            ccv_array_free(words);
        }
        ccv_matrix_free(image);
    } else {
        FILE* r = fopen(filename, "rt");
        if (argc == 3)
            chdir(argv[2]);
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
                printf("%s\n", file);
                ccv_read(file, &image, CCV_IO_GRAY | CCV_IO_ANY_FILE);
                ccv_array_t* words = ccv_swt_detect_words(image, ccv_swt_mine_params);
                int i;
                for (i = 0; i < words->rnum; i++)
                {
                    ccv_rect_t* rect = (ccv_rect_t*)ccv_array_get(words, i);
                    printf("%d %d %d %d\n", rect->x, rect->y, rect->width, rect->height);
                }
                ccv_array_free(words);
                ccv_matrix_free(image);
            }
            free(file);
            fclose(r);
        }
    }
    ccv_drain_cache();
    return 0;
}

