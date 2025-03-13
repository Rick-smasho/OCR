# OCR War Reporter Generator

This project uses OCR (Optical Character Recognition) to extract text from news article images, then generates a dramatic war reporter script and audio based on the content.

## Features

- Upload news article images for OCR text extraction
- Convert extracted text into structured JSON data
- Generate dramatic war reporter scripts from the news content
- Convert scripts to audio with text-to-speech
- Modern web interface with responsive design

## Technologies Used

- **Backend**: Flask (Python)
- **OCR**: Mistral AI OCR API
- **Text Generation**: Mistral AI and Groq LLM APIs
- **Text-to-Speech**: Zyphra API
- **Frontend**: HTML, CSS (Tailwind CSS), JavaScript

## Setup and Installation

1. Clone this repository:
   ```
   git clone https://github.com/Rick-smasho/OCR.git
   cd OCR
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up API keys:
   - Mistral AI API key
   - Groq API key
   - Zyphra API key

4. Run the application:
   ```
   python app.py
   ```

5. Open your browser and navigate to `http://localhost:5000`

## Usage

1. Upload a news article image using the web interface
2. Click "Generate Report" to process the image
3. View the extracted text, generated script, and listen to the audio
4. Download or copy the results as needed

## License

MIT License