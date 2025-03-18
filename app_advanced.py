from flask import Flask, render_template, request
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from pytube import extract
import whisper
import os
import spacy
from concurrent.futures import ThreadPoolExecutor
import re

# ✅ Load spaCy NLP Model for better sentence segmentation
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

# ✅ Load Summarization Model (T5 for structured bullet points)
print("⏳ Loading summarization model...")
summarizer = pipeline("summarization", model="t5-large", max_length=500)
print("✅ Summarization model loaded!")

# ✅ Load NLP Model for Video Type Classification
print("⏳ Loading classification model...")
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
print("✅ Classification model loaded!")

# ✅ Function to Extract YouTube Transcript
def get_transcript(video_id):
    """Fetches the transcript from YouTube using YouTubeTranscriptApi."""
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
        return " ".join([entry['text'] for entry in transcript])
    except Exception as e:
        print(f"❌ Error fetching transcript: {e}")
        return None

# ✅ Function to Transcribe Uploaded Videos
def transcribe_video(video_path):
    """Transcribes a locally uploaded video using Whisper AI."""
    model = whisper.load_model("tiny")  # ✅ Using the "tiny" model for faster processing
    result = model.transcribe(video_path)
    return result["text"]

# ✅ Function to Classify Video Type
def classify_video(text):
    """Classifies the type of video based on its transcript."""
    labels = ["Motivational", "Self-Improvement", "Educational", "Science", "Business", "Technology", 
              "Health & Fitness", "Entertainment", "News", "Documentary", "Finance", "Psychology", 
              "History", "Political", "Sports"]
    result = classifier(text, candidate_labels=labels)
    return result["labels"][0]

# ✅ Function to Split Text into Sections Using spaCy NLP
def split_into_sections(text):
    """Splits text into structured sections using spaCy NLP for better segmentation."""
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]

    sections = []
    chunk_size = 20  # 🔹 Increased chunk size for more context in summaries

    for i in range(0, len(sentences), chunk_size):
        section = " ".join(sentences[i:i + chunk_size])
        sections.append(section)

    print(f"✅ Sections created: {len(sections)}")
    return sections

# ✅ Function to Summarize Based on Video Type (Bullet Points + Sections)
def summarize_large_transcript(transcript_text, video_type):
    """Summarizes the video transcript into structured bullet points and sections."""

    prompt_mapping = {
        "Motivational": "Summarize the most powerful insights and life lessons in structured bullet points.",
        "Self-Improvement": "Summarize the best productivity habits, mindset strategies, and personal growth techniques in structured bullet points.",
        "Educational": "Summarize the core scientific concepts, key arguments, and real-world examples in structured bullet points.",
        "Business": "Summarize financial trends, key business strategies, and economic principles in structured bullet points.",
        "Documentary": "Summarize the key historical insights, expert perspectives, and major discoveries in structured bullet points."
    }
    
    prompt = prompt_mapping.get(video_type, "Summarize the key ideas of this video in structured bullet points.")

    # ✅ Handle Short Texts First (Videos < 5 min)
    if len(transcript_text.split()) < 1000:
        summary = summarizer(
            f"{prompt}\n{transcript_text}",
            max_new_tokens=500, temperature=0.3, repetition_penalty=2.5, do_sample=False
        )
        return format_summary(summary[0]['summary_text'], with_sections=True)

    sections = split_into_sections(transcript_text)

    # ✅ Process summaries in parallel to reduce loading time
    def process_section(section):
        return summarizer(
            f"{prompt}\n{section}",
            max_new_tokens=500, temperature=0.3, repetition_penalty=2.5, do_sample=True
        )[0]['summary_text']

    with ThreadPoolExecutor() as executor:
        summaries = list(executor.map(process_section, sections))

    structured_summaries = [format_summary(summary, with_sections=True, index=i+1) for i, summary in enumerate(summaries)]
    return "<br>".join(structured_summaries)

# ✅ Function to Format Summary as Bullet Points & Sections
def format_summary(text, with_sections=False, index=1):
    """Ensures the summary is formatted as bullet points with sections & no AI-generated comments."""
    
    text = re.sub(r"https?://\S+", "", text)  # Remove links
    text = re.sub(r"\(.*?\)", "", text)  # Remove text inside parentheses
    text = re.sub(r"\s+", " ", text).strip()  # Remove extra spaces

    # ✅ Removing unwanted AI-generated phrases
    text = re.sub(r"(thank you for watching.*?|subscribe for more.*?|click here.*?|for more information.*?|narrator:.*)", "", text, flags=re.IGNORECASE)

    sentences = re.split(r'(?<=[.!?])\s+', text)
    bullet_points = [f"- {sentence.strip()}" for sentence in sentences if len(sentence) > 10]

    # ✅ If sections are enabled, add Section Titles
    if with_sections:
        return f"<h3>🟢 Section {index}</h3><br>" + "<br>".join(bullet_points)

    return "<br>".join(bullet_points)

# ✅ Flask Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    """Handles summarization for both YouTube and uploaded videos."""
    video_url = request.form.get('video_url')
    uploaded_file = request.files.get('video_file')

    if video_url:
        video_id = extract.video_id(video_url)
        transcript_text = get_transcript(video_id)
    elif uploaded_file:
        video_path = os.path.join("uploads", uploaded_file.filename)
        os.makedirs("uploads", exist_ok=True)
        uploaded_file.save(video_path)
        transcript_text = transcribe_video(video_path)
    else:
        return "<h1>Error:</h1><p>❌ No video provided.</p>"

    if not transcript_text:
        return "<h1>Error:</h1><p>❌ Could not retrieve transcript. It may not be available for this video.</p>"

    video_type = classify_video(transcript_text)
    summary = summarize_large_transcript(transcript_text, video_type)

    return render_template('summary.html', summary=summary)

# ✅ Run Flask App
if __name__ == '__main__':
    from waitress import serve
    serve(app, host="0.0.0.0", port=8080)

