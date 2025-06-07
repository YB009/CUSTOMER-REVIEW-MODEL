from app import create_app
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt_tab')

app = create_app()

if __name__ == '__main__':
    app.run(debug=True) 