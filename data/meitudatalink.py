import os
import glob
import argparse

def softlink_all_videos(source_dir,dest_dir):
    """
    soft symbol link all videos all in a normal dir,instead of serpraterly
    :param source_dir: the meitu origin data_dir
    :param dest_dir: the dest collection dir
    :return: succeed or not
    """
    if not os.path.exists(dest_dir):
        os.mkdir(dest_dir)
    def symblink(dir):
        for root,dirs,files in os.walk(dir):
            for file_name in files:
                origin = os.path.join(root,file_name)
                dest = os.path.join(dest_dir,file_name)
                try:
                    os.symlink(origin,dest)
                except FileExistsError:
                    continue
            for sub_dir in dirs:
                sub_dir = os.path.join(root,sub_dir)
                symblink(sub_dir)
    symblink(source_dir)



if __name__=='__main__':
    parser = argparse.ArgumentParser(description='get the input dir and output dir')
    parser.add_argument('--video_dir',type=str,
                        default='/data/jh/notebooks/hudengjun/meitu/videos/val',help='this is a description')
    parser.add_argument('--output_dir',type=str,
                        default='/data/jh/notebooks/hudengjun/meitu/videos/val_collection')
    args = parser.parse_args()
    source_dir = args.video_dir
    dest_dir = args.output_dir
    softlink_all_videos(source_dir,dest_dir)


