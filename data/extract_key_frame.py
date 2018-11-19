from ctypes import *
import os
def extract_key_frame(filename, max_frame_num):
	""" The function receive three paras to extract key_frame index from a video:
			filename: the video filename;
			max_frame_num: the max number of key frames users want to get
	
		The return value is 
			frames_indices: the list of the key frame_index;
			length: the length of the frames_indices.
	"""
	if not os.path.isfile(filename):
		raise ValueError('The File does not exsists!')

	frames_indices = []
	lib = cdll.LoadLibrary('./key_frame_extract.so')
	frame_array = (c_int * max_frame_num)()
	length = c_int(0)
	frame_count = lib.find_key_frames(filename.encode('utf-8'), frame_array, byref(length), c_int(max_frame_num))

	for i in range(length.value):
		frames_indices.append(frame_array[i])

	return frames_indices, frame_count


