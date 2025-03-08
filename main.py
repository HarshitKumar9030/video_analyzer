from src.video_processor import VideoProcessor
from src.gemini_client import save_summary

def main():
    # you can use the video below for testing purposes 
    video_path = "./x.mp4"  # Update with your video path
    processor = VideoProcessor(video_path)
    
    # Process video and get the summary
    summary = processor.process_video()
    
    # Print summary to console
    print("\n----- SUMMARY -----\n")
    print(summary)
    print("\n-------------------\n")
    
    # Save summary to file
    save_summary(summary, video_path)

if __name__ == "__main__":
    main()