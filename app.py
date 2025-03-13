# Install required dependencies (run this in your terminal or Colab first if needed)
# pip install flask mistralai requests matplotlib Pillow numpy zyphra

from flask import Flask, request, render_template, send_file, jsonify, url_for
import json
import numpy as np
import re
import requests
import os
import base64
from typing import Dict, List, Optional, Union
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from io import BytesIO
from mistralai import Mistral
from zyphra import ZyphraClient

# API Keys (replace with your actual keys)
MISTRAL_API_KEY = "oxc4QDUxkd5sI2GrdTZSeb4PEquy4jcq"  # Mistral AI
GROQ_API_KEY = "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"  # Groq
ZYPHRA_API_KEY = "zsk-f0257a6b875712b55e5034445da5132e067429909ad007dc53617ed4ee35710c"  # Zyphra

app = Flask(__name__)

# Ensure static folder exists for serving audio files
os.makedirs('static', exist_ok=True)

# NewsToSpeechPipeline class
class NewsToSpeechPipeline:
    def __init__(self, api_key: str):
        self.client = ZyphraClient(api_key=api_key)

    def process_script(self,
                      script: str,
                      output_path: str = "news_audio.webm",
                      language_iso_code: str = "en-us",
                      speaking_rate: float = 15.0,
                      model: str = "zonos-v0.1-transformer",
                      mime_type: str = "audio/mp3",
                      emotion_settings: Optional[Dict[str, float]] = None,
                      voice_reference_path: Optional[str] = None,
                      **kwargs) -> str:
        params = {
            "text": script,
            "language_iso_code": language_iso_code,
            "speaking_rate": speaking_rate,
            "model": model,
            "mime_type": mime_type,
            "output_path": output_path
        }
        if emotion_settings:
            from zyphra.models.audio import EmotionWeights
            emotions = EmotionWeights(**emotion_settings)
            params["emotion"] = emotions
        if voice_reference_path and os.path.exists(voice_reference_path):
            with open(voice_reference_path, "rb") as f:
                audio_base64 = base64.b64encode(f.read()).decode('utf-8')
            params["speaker_audio"] = audio_base64
        params.update(kwargs)
        try:
            result_path = self.client.audio.speech.create(**params)
            print(f"Audio generated successfully and saved to {result_path}")
            return result_path
        except Exception as e:
            print(f"Error generating audio: {str(e)}")
            raise

def process_image_to_structured_news(image_bytes: bytes, output_file: str = None) -> Dict:
    client = Mistral(api_key=MISTRAL_API_KEY)
    try:
        encoded_image = base64.b64encode(image_bytes).decode()
        base64_data_url = f"data:image/jpeg;base64,{encoded_image}"
        print("Sending image to Mistral OCR API...")
        ocr_response = client.ocr.process(
            model="mistral-ocr-latest",
            document={"type": "image_url", "image_url": base64_data_url}
        )
        ocr_markdown = ocr_response.pages[0].markdown if ocr_response.pages else ""
        print("OCR completed successfully!")
        print("Extracting structured data using Mistral LLM...")
        news_structure = {
            "headline": "The main title of the news article",
            "source": "News agency or publication source",
            "location": "Dateline location",
            "body_text": ["Array of paragraphs from the article"],
            "date": "Publication date if available, null if not found"
        }
        prompt = f"""
        This is the OCR text extracted from a news article image:
        <BEGIN_OCR_TEXT>
        {ocr_markdown}
        </END_OCR_TEXT>
        Convert this into a structured JSON with the following format:
        {json.dumps(news_structure, indent=2)}
        The output should be strictly JSON with no additional commentary.
        """
        chat_response = client.chat.complete(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0
        )
        structured_data = json.loads(chat_response.choices[0].message.content)
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {output_file}")
        return structured_data
    except Exception as e:
        error_data = {"error": f"Processing failed: {str(e)}"}
        if output_file:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(error_data, f, indent=2)
        return error_data

def generate_war_reporter_script_groq(summary, headline, location):
    prompt = f"""You are an experienced war correspondent reporting from dangerous conflict zones.
LOCATION: {location if location else "the conflict zone"}
HEADLINE: {headline}
SUMMARY: {summary}
Create a dramatic news report script with the following:
1. Introduce yourself with a unique war reporter persona name and briefly describe where you're reporting from.
2. Present the news in a dramatic, tense style typical of frontline reporting.
3. Include background sounds or environment descriptions in [brackets].
4. Add short pauses indicated by (pause) where appropriate for dramatic effect.
5. End with a signature sign-off phrase and your reporter name.
FORMAT YOUR RESPONSE AS A COMPLETE SCRIPT READY FOR TEXT-TO-SPEECH:
"""
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_API_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama3-8b-8192",
                "messages": [
                    {"role": "system", "content": "You are an AI specialized in creating dramatic war correspondent scripts."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1024
            },
            timeout=15
        )
        if response.status_code == 200:
            return response.json().get("choices", [{}])[0].get("message", {}).get("content", "").strip()
        else:
            print(f"Groq API Error: {response.status_code}, {response.text}")
    except requests.RequestException as e:
        print(f"Groq API request failed: {str(e)}")
    print("Falling back to rule-based generation due to API failure.")
    return rule_based_script_generation(summary, headline, location)

def rule_based_script_generation(summary, headline, location):
    reporter_names = ["Alex Harker", "Morgan Wells", "Jamie Frost", "Casey Rivers", 
                     "Taylor Stone", "Jordan Reed", "Sam Fletcher", "Riley Hayes"]
    sign_offs = ["Reporting live from the frontlines, {reporter}.",
                 "Back to you in the studio, this is {reporter}.",
                 "This is {reporter}, reporting from {location}.",
                 "For World News Network, this is {reporter} signing off.",
                 "The situation remains fluid. From {location}, I'm {reporter} reporting."]
    sounds = ["[distant explosions]", "[helicopter overhead]", "[sirens wailing]",
             "[crowd noise]", "[wind howling]", "[radio static]", "[gunfire in distance]"]
    np.random.seed(42)
    reporter = np.random.choice(reporter_names)
    sign_off = np.random.choice(sign_offs).format(location=location or "the conflict zone", reporter=reporter)
    opening_sound = np.random.choice(sounds)
    middle_sound = np.random.choice(sounds)
    summary_parts = summary.split('\n\n')
    first_part = summary_parts[0] if summary_parts else summary
    rest_parts = ' '.join(summary_parts[1:]) if len(summary_parts) > 1 else ""
    script = (f"{opening_sound} This is {reporter}, reporting live from {location or 'the conflict zone'}. (pause)\n\n"
              f"{headline}. (pause)\n\n"
              f"{first_part}\n\n"
              f"{middle_sound} (pause)\n\n")
    if rest_parts:
        script += f"{rest_parts}\n\n"
    script += f"{sign_off}"
    return script

def simple_sentence_tokenize(text):
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def rank_sentences(sentences):
    if not sentences:
        return []
    keywords = ['arrested', 'charges', 'navy', 'officials', 'incident', 
                'killed', 'attack', 'explosion', 'conflict', 'military']
    scores = []
    for i, sentence in enumerate(sentences):
        position_score = 1.5 if i < len(sentences) * 0.2 else 1.2 if i > len(sentences) * 0.8 else 1.0
        length_score = 1.2 if 10 <= len(sentence.split()) <= 25 else 0.8
        content_score = 1.3 if any(keyword in sentence.lower() for keyword in keywords) else 1.0
        total_score = position_score * length_score * content_score
        scores.append((i, total_score))
    return scores

def extract_top_sentences(sentences, scores, num_sentences=5):
    if not sentences or not scores:
        return []
    num_sentences = min(5, len(sentences))
    scores.sort(key=lambda x: x[1], reverse=True)
    top_indices = sorted([idx for idx, _ in scores[:num_sentences]])
    return [sentences[i] for i in top_indices]

def add_war_reporter_style(sentences):
    if not sentences:
        return ""
    transition_phrases = ["Reports from the front line indicate", "Our correspondents on the ground confirm",
                          "Breaking news from the conflict zone:", "Eyewitness accounts suggest",
                          "According to military analysts,", "The situation remains tense as"]
    np.random.seed(42)
    styled_text = f"{np.random.choice(transition_phrases)} {sentences[0]}\n\n"
    for i, sentence in enumerate(sentences[1:-1], 1):
        if i % 2 == 0 and len(sentences) > 3:
            styled_text += f"{np.random.choice(transition_phrases)} {sentence} "
        else:
            styled_text += f"{sentence} "
    if len(sentences) > 1:
        styled_text += f"\n\nThe situation continues to develop as {sentences[-1].lower()}"
    return styled_text

def process_json_news(json_data):
    try:
        headline = json_data.get('headline', 'Breaking News from Conflict Zone')
        body_text = ''
        raw_body_text = json_data.get('body_text', '')
        if isinstance(raw_body_text, List):
            body_text = ' '.join(raw_body_text)
        elif isinstance(raw_body_text, str):
            body_text = raw_body_text
        else:
            body_text = str(raw_body_text)
        location = json_data.get('location', 'the conflict zone')
        full_text = f"{headline}. {body_text}".strip()
        cleaned_text = re.sub(r'\s+', ' ', full_text)
        sentences = simple_sentence_tokenize(cleaned_text)
        scores = rank_sentences(sentences)
        top_sentences = extract_top_sentences(sentences, scores)
        war_reporter_summary = add_war_reporter_style(top_sentences)
        news_script = generate_war_reporter_script_groq(war_reporter_summary, headline, location)
        return {
            'headline': headline,
            'location': location,
            'summary': war_reporter_summary,
            'news_script': news_script
        }
    except Exception as e:
        print(f"Error processing news: {str(e)}")
        return {
            'headline': 'Error Processing News',
            'location': 'Unknown',
            'summary': 'An error occurred while processing the news data.',
            'news_script': 'Unable to generate script due to an error.'
        }

def clean_script(script: str) -> str:
    return re.sub(r'\[.*?\]|\(pause\)', '', script).strip()

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'Empty file name'}), 400
        
        # Process the image
        image_bytes = image_file.read()
        
        # Save JSON to static folder
        json_path = os.path.join('static', 'news_article.json')
        json_data = process_image_to_structured_news(image_bytes, json_path)
        
        if 'error' in json_data:
            return jsonify({'error': json_data['error']}), 500
        
        # Process the extracted JSON data
        processed_data = process_json_news(json_data)
        
        # Generate audio from the script
        try:
            tts_pipeline = NewsToSpeechPipeline(api_key=ZYPHRA_API_KEY)
            audio_path = os.path.join('static', 'summary_report.mp3')
            cleaned_script = clean_script(processed_data['news_script'])
            tts_pipeline.process_script(
                script=cleaned_script,
                output_path=audio_path,
                speaking_rate=15.0,
                emotion_settings={"serious": 0.8, "urgent": 0.7}
            )
            audio_url = url_for('static', filename='summary_report.mp3')
        except Exception as e:
            print(f"TTS Error: {str(e)}")
            audio_url = None
        
        return jsonify({
            'json_data': json_data,
            'summary': processed_data['summary'],
            'news_script': processed_data['news_script'],
            'audio_url': audio_url
        })
    
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True)