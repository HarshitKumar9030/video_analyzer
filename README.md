# Video Analyzer

## Overview
The Video Analyzer project is designed to process video inputs, extract and preprocess content from each frame, and generate notes or summaries using Google's Gemini API. This project leverages various Python libraries for video processing and image manipulation.

## Project Structure
```
video-analyzer
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── video_processor.py
│   ├── frame_analyzer.py
│   ├── gemini_client.py
│   ├── utils
│   │   ├── __init__.py
│   │   └── image_utils.py
│   └── config.py
├── main.py
├── requirements.txt
├── .env.example 
└── README.md
```

## Installation
1. Clone the repository:
   ```
   git clone <repository-url>
   cd video-analyzer
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   ```

3. Activate the virtual environment:
   - On Windows:
     ```
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```
     source venv/bin/activate
     ```

4. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Configuration
- Open src/config.py and update the required variables.
- Open main.py and update the video file path.

## Usage
To run the video analyzer, execute the following command:
```
python main.py
```

# Future Enhancements
- Implement a GUI for the video analyzer.
- Add support for audio processing.

## Contributing
Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

