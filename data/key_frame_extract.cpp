/**
 * Simplest FFmpeg Decoder
 *
 * leixiaohua1020@126.com
 * Communication University of China / Digital TV Technology
 * http://blog.csdn.net/leixiaohua1020
 *
 * This software is a simplest decoder based on FFmpeg.
 * It decodes video to YUV pixel data.
 * It uses libavcodec and libavformat.
 * Suitable for beginner of FFmpeg.
 *
 */

#include <stdio.h>

#define __STDC_CONSTANT_MACROS

#ifdef _WIN32
//Windows
extern "C"
{
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
#include "libswscale/swscale.h"
#include "libavutil/imgutils.h"
};
#else
//Linux...
#ifdef __cplusplus
extern "C"
{
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/imgutils.h>
#ifdef __cplusplus
};
#endif
#endif


#include <string>
#include <stdexcept>
#include <unordered_map>
#include <iostream>

using namespace std;

struct OpenFile {
	bool open = false;
	AVRational frame_base_;
	AVRational stream_base_;
	int frame_count_;

	int vid_stream_idx_;
	int last_frame_;

	AVFormatContext* pFormatCtx;
	AVCodecContext* pCodecCtx;
	AVCodec* pCodec;
};

unordered_map<string, OpenFile> open_files_;

extern "C" OpenFile& open_file(string filename, int index=-1);

extern "C" void seek_pos(OpenFile& file, int frame, int mode=0);

extern "C" int find_key_frames(char* filename, int *frame_index, int max_num);

extern "C" OpenFile& open_file(string filename, int index)
{
	OpenFile& file = open_files_[filename+to_string(index)];
	if(!file.open){
		file.pFormatCtx = avformat_alloc_context();

		if(avformat_open_input(&file.pFormatCtx, filename.c_str(), NULL, NULL)!=0){
			throw std::runtime_error(std::string("Could not open file ") + filename);
		}
		if(avformat_find_stream_info(file.pFormatCtx, NULL)<0){
			throw std::runtime_error(std::string("Could not find stream information in ")+ filename);
		}

		file.vid_stream_idx_ = -1;
		for(unsigned int i=0; i<file.pFormatCtx->nb_streams;i++)
			if(file.pFormatCtx->streams[i]->codec->codec_type == AVMEDIA_TYPE_VIDEO){
				file.vid_stream_idx_=i;
				break;
			}

		if(file.vid_stream_idx_==-1){
			throw std::runtime_error(std::string("Could not find video stream in ") + filename);
		}
		
		file.pCodecCtx=file.pFormatCtx->streams[file.vid_stream_idx_]->codec;
		file.pCodec=avcodec_find_decoder(file.pCodecCtx->codec_id);
		if(file.pCodec == NULL){
			throw std::runtime_error(std::string("Could not find codec in ") + filename);
		}
		
#ifdef FFMPEG_THREAD
		file.pCodecCtx->thread_count = 10;
		file.pCodecCtx->thread_type = FF_THREAD_SLICE;
		file.pCodecCtx->flags |= CODEC_FLAG_TRUNCATED;
#endif

		if(avcodec_open2(file.pCodecCtx, file.pCodec, NULL) < 0){
			throw std::runtime_error(std::string("Could not open codec in ") + filename);
		}

		AVStream* stream = file.pFormatCtx->streams[file.vid_stream_idx_];

		file.stream_base_ = stream->time_base;
		file.frame_base_ = AVRational{stream->avg_frame_rate.den,
									  stream->avg_frame_rate.num};

		file.frame_count_ = av_rescale_q(stream->duration, stream->time_base, file.frame_base_);
		
//		printf("Frame_count is %d ; Frame_base is %d %d \n", file.frame_count_, file.frame_base_.den, file.frame_base_.num);		

		file.open = true;
	}	
	return file;
}


extern "C" void seek_pos(OpenFile& file, int frame, int mode)
{
	int ret = 0;
	int seek_time = av_rescale_q(frame, file.frame_base_, file.stream_base_);
	//printf("Seeking to frame %d timestamp %d \n", frame, seek_time);
	if(mode == 0)
  		ret = av_seek_frame(file.pFormatCtx, file.vid_stream_idx_,seek_time, AVSEEK_FLAG_FRAME); //forward seek to the next key frame 
	else if(mode == 1)
		ret = av_seek_frame(file.pFormatCtx, file.vid_stream_idx_,seek_time, AVSEEK_FLAG_ANY); //seek to any frame even if it's none key frame
	else if(mode == 2)
		ret = av_seek_frame(file.pFormatCtx, file.vid_stream_idx_,seek_time, AVSEEK_FLAG_BACKWARD); //seek to the last key frame

	if (ret < 0) {
	//	printf("Seeking ERROR \n");
	}	
}

/**
 * loop the video file once to get the key frame index 
 * just judge whether the frame is a key frame, not decode
 *
 */
extern "C" int find_key_frames(char* filename, int *frame_index, int max_num){
	av_register_all();
	avformat_network_init();
	
	OpenFile& file = open_file(filename);
	
	AVPacket* packet;
	int count=0;
	
	seek_pos(file, 0, 2);

	packet=(AVPacket *)av_malloc(sizeof(AVPacket));

	while(av_read_frame(file.pFormatCtx, packet)>=0){
		if(packet->stream_index==file.vid_stream_idx_){
			int frame = av_rescale_q(packet->pts, file.stream_base_, file.frame_base_);

			if(packet->flags & AV_PKT_FLAG_KEY)
			{
				//printf("Key frame! %d \n", frame);
			
				//key_frame_index[count] = frame; 
				*( frame_index + count)= frame;
				
				count++;
				if(count >= max_num)
				{
					printf("redundent frames! \n");
					break;
				}
				seek_pos(file, frame+1, 0);
			}
			av_free_packet(packet);
		}	
	}

	avcodec_flush_buffers(file.pCodecCtx);
	return count;	
}

