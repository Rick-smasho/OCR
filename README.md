# Hackgrid

## Team Members

| Name   | Phone Number |
|--------|--------------|
| Sachiv | 9444209374   |
| Niyas  | 7418971997   |

#  AI-Powered  WarZone Reporter

## 🌐 Project Overview

ConflictCast is an innovative application that transforms static newspaper articles about conflict zones into dynamic, personalized news reports ready for broadcast. Using advanced OCR technology, NLP processing, and voice synthesis, the system creates compelling war correspondent scripts that bring news to life with professional delivery and emotional nuance.

## 🔑 Key Features

- **Image-to-Script Pipeline**: Upload newspaper images to extract text using Mistral OCR
- **Intelligent Content Analysis**: Automatically extracts headlines, locations, and critical information
- **War Correspondent Style**: Transforms formal news into dramatic frontline reporting
- **Professional Voice Synthesis**: Converts scripts into broadcast-ready audio with emotional inflection
- **Customizable Output**: Adjust speaking rate and emotion settings for different segments

## 🛠️ Technology Stack

- **Backend**: Flask with Python
- **OCR**: MistralAI OCR for text extraction
- **NLP**: Custom sentence ranking & MistralAI/Groq for structured data extraction
- **Voice Synthesis**: Zyphra API for realistic news reporter voices
- **Frontend**: HTML/JS/CSS (not included in the repository)

## 📋 Project Structure

```
ConflictCast/
├── voice.py           # Voice synthesis module using Zyphra API
├── app.py             # Main Flask application with integrated functionality
├── templates
      ├──index.html    # Main page
├── static  # Static assets (CSS, JS, images)

```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Flask
- OpenCV
- MistralAI Python SDK
- Zyphra Voice API access

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/Rick-smasho/OCR.git
   cd OCR
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   ```
   export MISTRAL_API_KEY="your_mistral_api_key"
   export GROQ_API_KEY="your_groq_api_key"
   export ZYPHRA_API_KEY="your_zyphra_api_key"
   ```

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## 💡 Usage

### Processing Newspaper Articles

1. Upload a newspaper image through the web interface
2. The system will process the image using OCR
3. Review the extracted content and generated news script
4. Generate an audio report with war correspondent style and emotional inflection

### Example Workflow

```python
# Example of processing a news article from text
from app import process_json_news

news_data = {
    "headline": "Taliban reject court move to arrest its top officials",
    "body_text": [
        "The Taliban on Friday rejected a court move to arrest two of their top officials for persecuting women, accusing the court of baseless accusations and misbehaviour.",
        "The International Criminal Court's chief prosecutor Karim Khan announced on Thursday he had requested arrest warrants for two top Taliban officials, including the leader Hibatullah Akhundzada.",
        "Since they took back control of the country in 2021, the Taliban have barred women from jobs, most public spaces and education beyond sixth grade.",
        "A Foreign Ministry statement condemned the ICC request."
    ],
    "location": "KABUL"
}

result = process_json_news(news_data)
print(result['news_script'])
```

## 🎬 Demo

(https://drive.google.com/drive/folders/1pwhqWNKP5NUnZhNk-Yd_nAgiqP3x2W0Y?usp=sharing)]

## 📈 Future Enhancements

- Real-time news processing from RSS feeds
- Multi-language support for global conflict coverage
- Video generation with AI anchors and background footage
- Mobile app for field journalists to quickly convert articles to audio reports
- Integration with popular podcast and news platforms

## 🔒 API Keys and Security

**Important:** The repository contains placeholder API keys. For security:

1. Never commit real API keys to your repository
2. Use environment variables for all sensitive keys
3. Implement proper API key rotation and security practices


## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgements

- MistralAI for OCR technology
- Groq for NLP processing capabilities
- Zyphra for voice synthesis technology
- The open-source community for various libraries and tools
