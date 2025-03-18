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

### 3. Install Dependencies
Install the required libraries using pip:

```bash
pip install -r requirements.txt
```

### 4. Download spaCy Model
Download the spaCy language model en_core_web_sm:

```bash
python -m spacy download en_core_web_sm
```

### 🚀 Running the Project Locally
Once the setup is complete, you can run the application locally using Flask:

```bash
python app_advanced.py
```
Your app should now be accessible at http://127.0.0.1:5001/ in your web browser.


### 📑 How It Works
Transcription: The app extracts the transcript from YouTube videos using youtube-transcript-api or transcribes uploaded videos using Whisper.
Text Segmentation: We use spaCy for better segmentation of the transcript into sentences and sections.
Summarization: The app uses the T5 model for generating summaries of video content.
Classification: The app classifies the video into categories like Motivational, Educational, Business, and more using the Zero-Shot Classification model.

### 🎨 UI Design
The user interface is designed with Tailwind CSS for a responsive and modern look. It features:

Dark/Light Mode Toggle: Easily switch between light and dark modes for a better user experience.
Collapsible Sections: Summary sections are collapsible for easier navigation.
Attractive Layout: A minimalist and clean layout with a glassmorphism effect for a sleek design.

### 🔧 Built With
Flask: For building the web application.
spaCy: For text processing and NLP tasks.
Transformers: For powerful text summarization using T5.
Whisper: For transcribing audio to text.
YouTube Transcript API: For fetching YouTube video transcripts.
Pytube: For interacting with YouTube videos.

### Thank you for checking out the AI Video Summarizer! 🚀🎬



