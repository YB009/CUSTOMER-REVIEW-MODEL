
## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <project-directory>
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the web server:
```bash
python app/main.py
```

2. Open your browser and navigate to `http://localhost:5000`

3. Enter your reviews in the text area and click "Analyze Topics"

## API Endpoints

- `POST /analyze`: Analyze reviews and return topics
  - Request body: `{"reviews": ["review1", "review2", ...]}`
  - Response: `{"topics": [["word1", "word2", ...], ...]}`

## Dependencies

- Flask
- NumPy
- NLTK
- Flask-CORS

## License

MIT License

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request