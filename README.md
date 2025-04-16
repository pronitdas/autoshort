# Video Text Overlay App

A simple Streamlit application that allows you to:
1. Upload a video
2. Add text overlay (manually or generate using LLMs)
3. Convert text to speech
4. Combine everything into a final video

## Features

- Video upload support (MP4, MOV, AVI)
- Text overlay with customizable appearance
- Text-to-speech conversion
- Integration with local LLMs (Ollama and LM Studio)
- Easy download of processed videos

## Setup

### Local Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
streamlit run app.py
```

### Docker Setup

1. Build the container:
```bash
docker build -t video-text-overlay .
```

2. Run the container:
```bash
docker run -p 8501:8501 video-text-overlay
```

## LLM Integration

The app supports two local LLM options:

### Ollama
- Install Ollama from: https://ollama.ai/
- Run Ollama locally
- Default endpoint: http://localhost:11434

### LM Studio
- Install LM Studio from: https://lmstudio.ai/
- Run your chosen model locally
- Default endpoint: http://localhost:1234/v1/completions

## Usage

1. Open the app in your browser (default: http://localhost:8501)
2. Choose your LLM settings in the sidebar (optional)
3. Upload a video file
4. Either:
   - Enter text directly in the text area
   - Or use an LLM to generate text by entering a prompt
5. Click "Process Video" to create the final video
6. Download the processed video using the download button

## Notes

- The app creates temporary files that are cleaned up after processing
- Text overlay is centered and sized to 80% of video width
- Audio is generated using pyttsx3 (offline TTS)
