import cv2
import os
import numpy as np
from tqdm import tqdm
import time
from collections import defaultdict
from .frame_analyzer import FrameAnalyzer
from .gemini_client import GeminiClient
from .config import API_KEY

class VideoProcessor:
    def __init__(self, video_path, frame_sample_rate=1, scene_threshold=30):
        """
        Initialize the video processor
        
        Args:
            video_path: Path to the video file
            frame_sample_rate: Process every nth frame (default: 1)
            scene_threshold: Threshold for scene change detection (0-255)
        """
        self.video_path = video_path
        self.frame_sample_rate = frame_sample_rate
        self.scene_threshold = scene_threshold
        self.frames = []
        self.frame_data = []
        self.scenes = []
        self.analyzer = FrameAnalyzer()
        self.gemini_client = GeminiClient(api_key=API_KEY)
        self.summary = None
        
    def process_video(self):
        """Process the entire video and return a summary"""
        print(f"Processing video: {self.video_path}")
        self.extract_frames()
        self.analyze_frames()
        summary = self.generate_summary()
        return summary
        
    def extract_frames(self):
        """Extract frames from the video with scene change detection"""
        if not os.path.exists(self.video_path):
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
            
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {self.video_path}")
            
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Total frames: {total_frames}, FPS: {fps}")
        print(f"Sampling every {self.frame_sample_rate} frame(s)")
        
        prev_frame = None
        current_scene = []
        scene_count = 0
        
        # Process frames with progress bar
        with tqdm(total=total_frames) as pbar:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Update progress bar
                pbar.update(1)
                
                # Only process every nth frame
                if frame_idx % self.frame_sample_rate == 0:
                    # Scene change detection
                    if prev_frame is not None:
                        # Convert frames to grayscale for comparison
                        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                        
                        # Calculate absolute difference between frames
                        frame_diff = cv2.absdiff(curr_gray, prev_gray)
                        mean_diff = np.mean(frame_diff)
                        
                        # If difference exceeds threshold, mark as a new scene
                        if mean_diff > self.scene_threshold:
                            if current_scene:
                                self.scenes.append(current_scene)
                                scene_count += 1
                                print(f"Scene {scene_count} detected with {len(current_scene)} frames")
                            current_scene = []
                    
                    # Store the frame
                    self.frames.append(frame)
                    current_scene.append(frame)
                    prev_frame = frame.copy()
                
                frame_idx += 1
                
        # Add the last scene if it exists
        if current_scene:
            self.scenes.append(current_scene)
            scene_count += 1
            
        print(f"Extracted {len(self.frames)} frames across {scene_count} scenes")
        cap.release()
        
    def analyze_frames(self, verbose=False):
        """Analyze the extracted frames"""
        print("Analyzing frames...")
        self.frame_data = []
        
        for i, frame in enumerate(tqdm(self.frames)):
            # Analyze each frame using the FrameAnalyzer
            frame_analysis = self.analyzer.analyze_frame(frame)
            
            # Add frame index for reference
            frame_analysis['frame_index'] = i
            frame_analysis['timestamp'] = i * self.frame_sample_rate / self.get_video_fps()
            
            if verbose:
                print(f"Frame {i}: {frame_analysis.get('text', '')[:30]}...")
            
            self.frame_data.append(frame_analysis)
            
        print(f"Analyzed {len(self.frame_data)} frames")
    
    def get_video_fps(self):
        """Get the FPS of the video"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps
    
    def get_scene_data(self):
        """Group frame data by scenes"""
        scene_data = []
        
        frame_to_scene = {}
        for scene_idx, scene in enumerate(self.scenes):
            for frame in scene:
                for i, f in enumerate(self.frames):
                    if np.array_equal(frame, f):
                        frame_to_scene[i] = scene_idx
                        break
        
        scene_frames = defaultdict(list)
        for frame_info in self.frame_data:
            frame_idx = frame_info['frame_index']
            if frame_idx in frame_to_scene:
                scene_idx = frame_to_scene[frame_idx]
                scene_frames[scene_idx].append(frame_info)
        
        for scene_idx, frames in scene_frames.items():
            scene_text = ""
            for frame in frames:
                if 'text' in frame and frame['text']:
                    scene_text += f"{frame['text']} "
            
            scene_data.append({
                'scene_index': scene_idx,
                'start_time': frames[0]['timestamp'] if frames else 0,
                'end_time': frames[-1]['timestamp'] if frames else 0,
                'duration': frames[-1]['timestamp'] - frames[0]['timestamp'] if frames else 0,
                'text_content': scene_text.strip(),
                'frame_count': len(frames)
            })
        
        return scene_data

    def generate_summary(self):
        """Generate a summary using the GeminiClient"""
        print("Generating summary using Gemini...")
        
        scene_data = self.get_scene_data()
        
        video_info = {
            'filename': os.path.basename(self.video_path),
            'frames_processed': len(self.frames),
            'scenes_detected': len(self.scenes),
            'total_duration': self.frame_data[-1]['timestamp'] if self.frame_data else 0,
            'scenes': scene_data,
            'frame_data': self.frame_data
        }
        
        try:
            self.summary = self.gemini_client.generate_summary(video_info)
        except Exception as e:
            print(f"Error generating summary: {e}")
            self.summary = f"Failed to generate summary: {str(e)}"
        
        print("Summary generation complete")
        return self.summary