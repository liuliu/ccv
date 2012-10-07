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
	struct SwsContext* picture_ctx = sws_getCachedContext(0, video_st->codec->width, video_st->codec->height, video_st->codec->pix_fmt, video_st->codec->width, video_st->codec->height, PIX_FMT_RGB24, SWS_BICUBIC, 0, 0, 0);
	sws_scale(picture_ctx, (const uint8_t* const*)picture->data, picture->linesize, 0, video_st->codec->height, rgb_picture.data, rgb_picture.linesize);
	ccv_dense_matrix_t* x = 0;
	ccv_read(rgb_picture.data[0], &x, CCV_IO_RGB_RAW | CCV_IO_NO_COPY, video_st->codec->height, video_st->codec->width, rgb_picture.linesize[0]);
	ccv_write(x, "output.png", 0, CCV_IO_PNG_FILE, 0);
	ccv_matrix_free(x);
#endif
#endif
#endif
	return 0;
}
