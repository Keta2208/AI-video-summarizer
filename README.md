# AI Video Summarizer 🎥🤖

Welcome to the **AI Video Summarizer**, a powerful tool that leverages natural language processing (NLP) and machine learning to provide concise summaries for YouTube videos and uploaded video files. Whether you're a student, content creator, or professional, this tool can help you quickly extract key insights from videos.

## 🚀 Features

- **YouTube Video Support**: Summarize any YouTube video by providing its URL.
- **Uploaded Video Support**: Upload your own videos, and the tool will transcribe and summarize them.
- **Multi-Model Summarization**: The tool uses advanced models like T5 for summarization, Whisper for transcription, and spaCy for NLP-based sentence segmentation.
- **Video Type Classification**: Automatically classifies the video into categories like Motivational, Educational, Business, and more.
- **User-Friendly Interface**: Simple web interface for easy interaction.

## ⚙️ Requirements

Before running the project locally or deploying it, ensure you have the following:

- **Python 3.11** (or compatible versions)
- **Virtual Environment** for managing dependencies
- **Installed Libraries**:
  - Flask
  - transformers
  - youtube-transcript-api
  - pytube
  - torch
  - openai-whisper
  - spacy
  - numpy
  - pandas
  - requests
  - tqdm
  - waitress

## 💻 Installation

### 1. Clone the Repository

Start by cloning the repository:

```bash
git clone https://github.com/Keta2208/AI-video-summarizer.git
cd AI-video-summarize
```

### 2. Set Up Virtual Environment
Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/macOS
venv\Scripts\activate  # For Windows
```




