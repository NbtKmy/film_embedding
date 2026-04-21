from scenedetect import detect, AdaptiveDetector, split_video_ffmpeg
import argparse



def split_video(video_path: str, output_dir: str):
    scene_list = detect(video_path, AdaptiveDetector())
    split_video_ffmpeg(video_path, scene_list, output_dir=output_dir)

if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Split video into scenes')
    arg_parser.add_argument('input', type=str, help='Path to the input video file')
    arg_parser.add_argument('output', help='Path to the output directory')
    args = arg_parser.parse_args()

    split_video(args.input, args.output)