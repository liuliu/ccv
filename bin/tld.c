#include <ccv.h>
#ifdef HAVE_AVCODEC
#include <libavcodec/avcodec.h>
#endif
#ifdef HAVE_AVFORMAT
#include <libavformat/avformat.h>
#endif
#ifdef HAVE_SWSCALE
#include <libswscale/swscale.h>
#endif

int main(int argc, char** argv)
{
#ifdef HAVE_AVCODEC
#ifdef HAVE_AVFORMAT
#ifdef HAVE_SWSCALE
	assert(argc == 6);
	ccv_rect_t box = ccv_rect(atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]));
	// box.width = box.width - box.x + 1;
	// box.height = box.height - box.y + 1;
	// printf("%d,%d,%d,%d,%f\n", box.x, box.y, box.width + box.x - 1, box.height + box.y - 1, 1.0f);
	printf("%05d: %d %d %d %d %f\n", 0, box.x, box.y, box.width, box.height, 1.0f);
	// init av-related structs
	AVFormatContext* ic = 0;
	int video_stream = -1;
	AVStream* video_st = 0;
	AVFrame* picture = 0;
	AVFrame rgb_picture;
	memset(&rgb_picture, 0, sizeof(AVPicture));
	AVPacket packet;
	memset(&packet, 0, sizeof(AVPacket));
	av_init_packet(&packet);
	av_register_all();
	avformat_network_init();
	// load video and codec
	avformat_open_input(&ic, argv[1], 0, 0);
	avformat_find_stream_info(ic, 0);
	int i;
	for (i = 0; i < ic->nb_streams; i++)
	{
		AVCodecContext* enc = ic->streams[i]->codec;
		enc->thread_count = 2;
		if (AVMEDIA_TYPE_VIDEO == enc->codec_type && video_stream < 0)
		{
			AVCodec* codec = avcodec_find_decoder(enc->codec_id);
			if (!codec || avcodec_open2(enc, codec, 0) < 0)
				continue;
			video_stream = i;
			video_st = ic->streams[i];
			picture = avcodec_alloc_frame();
			rgb_picture.data[0] = (uint8_t*)ccmalloc(avpicture_get_size(PIX_FMT_RGB24, enc->width, enc->height));
			avpicture_fill((AVPicture*)&rgb_picture, rgb_picture.data[0], PIX_FMT_RGB24, enc->width, enc->height);
			break;
		}
	}
	int got_picture = 0;
	while (!got_picture)
	{
		int result = av_read_frame(ic, &packet);
		if (result == AVERROR(EAGAIN))
			continue;
		avcodec_decode_video2(video_st->codec, picture, &got_picture, &packet);
	}
	ccv_enable_default_cache();
	struct SwsContext* picture_ctx = sws_getCachedContext(0, video_st->codec->width, video_st->codec->height, video_st->codec->pix_fmt, video_st->codec->width, video_st->codec->height, PIX_FMT_RGB24, SWS_BICUBIC, 0, 0, 0);
	sws_scale(picture_ctx, (const uint8_t* const*)picture->data, picture->linesize, 0, video_st->codec->height, rgb_picture.data, rgb_picture.linesize);
	ccv_dense_matrix_t* x = 0;
	ccv_read(rgb_picture.data[0], &x, CCV_IO_RGB_RAW | CCV_IO_GRAY, video_st->codec->height, video_st->codec->width, rgb_picture.linesize[0]);
	ccv_tld_t* tld = ccv_tld_new(x, box, ccv_tld_default_params);
	ccv_dense_matrix_t* y = 0;
	for (;;)
	{
		got_picture = 0;
		int result = av_read_frame(ic, &packet);
		if (result == AVERROR(EAGAIN))
			continue;
		avcodec_decode_video2(video_st->codec, picture, &got_picture, &packet);
		if (!got_picture)
			break;
		sws_scale(picture_ctx, (const uint8_t* const*)picture->data, picture->linesize, 0, video_st->codec->height, rgb_picture.data, rgb_picture.linesize);
		ccv_read(rgb_picture.data[0], &y, CCV_IO_RGB_RAW | CCV_IO_GRAY, video_st->codec->height, video_st->codec->width, rgb_picture.linesize[0]);
		ccv_tld_info_t info;
		ccv_comp_t newbox = ccv_tld_track_object(tld, x, y, &info);
		/*
		// printf("%04d: performed learn: %d, performed track: %d, successfully track: %d; %d passed fern detector, %d passed nnc detector, %d merged, %d confident matches, %d close matches\n", tld->count, info.perform_learn, info.perform_track, info.track_success, info.ferns_detects, info.nnc_detects, info.clustered_detects, info.confident_matches, info.close_matches);
		ccv_dense_matrix_t* image = 0;
		ccv_read(rgb_picture.data[0], &image, CCV_IO_RGB_RAW | CCV_IO_RGB_COLOR, video_st->codec->height, video_st->codec->width, rgb_picture.linesize[0]);
		// draw out
		// for (i = 0; i < tld->top->rnum; i++)
		if (tld->found)
		{
			ccv_comp_t* comp = &newbox; // (ccv_comp_t*)ccv_array_get(tld->top, i);
			if (comp->rect.x >= 0 && comp->rect.x + comp->rect.width < image->cols &&
				comp->rect.y >= 0 && comp->rect.y + comp->rect.height < image->rows)
			{
				int x, y;
				for (x = comp->rect.x; x < comp->rect.x + comp->rect.width; x++)
				{
					image->data.u8[image->step * comp->rect.y + x * 3] =
					image->data.u8[image->step * (comp->rect.y + comp->rect.height - 1) + x * 3] = 255;
					image->data.u8[image->step * comp->rect.y + x * 3 + 1] =
					image->data.u8[image->step * (comp->rect.y + comp->rect.height - 1) + x * 3 + 1] =
					image->data.u8[image->step * comp->rect.y + x * 3 + 2] =
					image->data.u8[image->step * (comp->rect.y + comp->rect.height - 1) + x * 3 + 2] = 0;
				}
				for (y = comp->rect.y; y < comp->rect.y + comp->rect.height; y++)
				{
					image->data.u8[image->step * y + comp->rect.x * 3] =
					image->data.u8[image->step * y + (comp->rect.x + comp->rect.width - 1) * 3] = 255;
					image->data.u8[image->step * y + comp->rect.x * 3 + 1] =
					image->data.u8[image->step * y + (comp->rect.x + comp->rect.width - 1) * 3 + 1] =
					image->data.u8[image->step * y + comp->rect.x * 3 + 2] =
					image->data.u8[image->step * y + (comp->rect.x + comp->rect.width - 1) * 3 + 2] = 0;
				}
			}
		}
		char filename[1024];
		sprintf(filename, "tld-out/output-%04d.png", tld->count);
		ccv_write(image, filename, 0, CCV_IO_PNG_FILE, 0);
		ccv_matrix_free(image);
		if (tld->found)
			printf("%d,%d,%d,%d,%f\n", newbox.rect.x, newbox.rect.y, newbox.rect.width + newbox.rect.x - 1, newbox.rect.height + newbox.rect.y - 1, newbox.classification.confidence);
		else
			printf("NaN,NaN,NaN,NaN,NaN\n");
		*/
		if (tld->found)
			printf("%05d: %d %d %d %d %f\n", tld->count, newbox.rect.x, newbox.rect.y, newbox.rect.width, newbox.rect.height, newbox.classification.confidence);
		else
			printf("%05d: --------------\n", tld->count);
		x = y;
		y = 0;
	}
	ccv_matrix_free(x);
	ccv_tld_free(tld);
	ccfree(rgb_picture.data[0]);
	ccv_disable_cache();
#endif
#endif
#endif
	return 0;
}
