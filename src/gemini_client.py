import google.generativeai as genai
import json
import time
import os

class GeminiClient:
    def __init__(self, api_key):
        self.api_key = api_key
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash')
    
    def generate_summary(self, video_info):
        """Generate a summary of the video using Gemini Flash"""
        try:
            # Create a prompt for the video summary
            prompt = self._create_summary_prompt(video_info)
            
            response = self.model.generate_content(prompt)
            
            if hasattr(response, 'text'):
                summary = response.text
                # Add a small delay to allow gRPC to clean up properly
                time.sleep(0.5)
                return summary
            else:
                return "Summary generation failed: No text in response"
                
        except Exception as e:
            print(f"Error in Gemini API: {str(e)}")
            return f"Summary generation failed: {str(e)}"
    
    def _create_summary_prompt(self, video_info):
        """Create a well-structured prompt for the Gemini model"""
        # Basic video information
        prompt = f"""You are a video analysis expert. Create a comprehensive summary and notes for the following video:
        
Video: {video_info['filename']}
Duration: {video_info['total_duration']:.2f} seconds
Scenes detected: {video_info['scenes_detected']}
        
"""
        
        if 'scenes' in video_info and video_info['scenes']:
            prompt += "\n## SCENE BREAKDOWN:\n"
            for scene in video_info['scenes']:
                prompt += f"\nScene {scene['scene_index']+1} ({scene['start_time']:.2f}s - {scene['end_time']:.2f}s):\n"
                if scene['text_content']:
                    prompt += f"Text detected: {scene['text_content'][:500]}...\n" if len(scene['text_content']) > 500 else f"Text detected: {scene['text_content']}\n"
        
        # Add analysis instructions
        prompt += """
Based on the above information, please provide:
1. A concise summary of the video content
2. Key points or important information extracted from the video
3. Any notable observations about the structure or content
4. Well-organized notes that could be used for reference

Format your response as a well-structured document with headings, bullet points and proper organization.
"""
        
        return prompt
    
    def generate_notes(self, processed_data):
        """Generate detailed notes from the video data"""
        try:
            # Create a specialized prompt for detailed notes
            prompt = f"""As an expert note-taker, create detailed, organized notes from this video content. 
Focus on creating a hierarchical structure that captures main ideas and supporting details.

Content information:
{json.dumps(processed_data, indent=2, default=str)[:4000]}

Format your notes as markdown with:
- Clear headings and subheadings
- Bullet points for key concepts
- Numbered lists for sequential information
- Code blocks or tables where appropriate
- Bold text for important terms or concepts

Your notes should be comprehensive yet concise, capturing the essence of the content.
"""
            
            # Generate content
            response = self.model.generate_content(prompt)
            
            if hasattr(response, 'text'):
                return response.text
            else:
                return "Notes generation failed: No text in response"
                
        except Exception as e:
            print(f"Error in Gemini API: {str(e)}")
            return f"Notes generation failed: {str(e)}"

def save_summary(summary, video_path):
    """Save the generated summary to a file"""
    output_dir = "output/summaries"
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_name}_summary.md")
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    
    print(f"Summary saved to: {output_path}")