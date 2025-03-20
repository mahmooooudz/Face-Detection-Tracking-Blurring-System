from video_processor import VideoProcessor

def main():
    input_video = 'VideoTest.mp4'
    output_video = 'TestResult.mp4'
    processor = VideoProcessor(input_video)
    all_tracked_faces = processor.process_video(output_path=output_video, show_bbox=True)
    
    for face_id, appearances in all_tracked_faces.items():
        print(f"Face {face_id} appeared {len(appearances)} times")

if __name__ == '__main__':
    main()