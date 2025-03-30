# Semantic Book Recommender

> 🚀 **Live Demo:** [Try the app on Hugging Face Spaces](https://ashish-kharde1-book-recommendation.hf.space/)

A machine learning-powered book recommendation system that suggests books based on natural language descriptions, categories, and emotional tones.

## Features

- Natural language book search using semantic similarity
- Category-based filtering
- Emotional tone filtering (Happy, Surprising, Angry, Suspenseful, Sad)
- Visual interface with book covers and descriptions
- Multi-stage recommendation pipeline using Google's Generative AI

## Setup

1. Clone the repository:
```bash
git clone https://github.com/Ashish-kharde1/book-recommender.git
cd book-recommender
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up API keys:
    - Create a `.env` file in the project root:
```bash
GOOGLE_API_KEY=your_key_here
```

5. Download and prepare the data:
   - Run the data preparation notebooks in order:
```bash
jupyter notebook data-exploration.ipynb   # Initial data cleaning
jupyter notebook test-classification.ipynb  # Category classification
jupyter notebook sentiment-analysis.ipynb   # Emotion analysis
jupyter notebook vector-search.ipynb        # Search embeddings setup
```

6. Verify setup:
   - Check that the following files exist:
     - `books_with_emotion.csv`
     - `tagged_description.txt`
     - `cover_not_found.jpg` (default book cover image)
   - Make sure the ChromaDB vector store is initialized

7. Run the application:
```bash
python app.py
```

8. Access the interface:
   - Open your browser
   - Go to `http://localhost:7860`
   - The interface should show the book recommendation system

## Troubleshooting

Common issues and solutions:
- If ChromaDB fails to initialize: Clear the `.chroma` directory and rerun vector-search.ipynb
- If model downloads fail: Check your internet connection and retry
- If the API key doesn't work: Verify the key in Google Cloud Console and check the .env file format

## Usage

Run the web interface:
```bash
python app.py
```

The interface allows you to:
- Enter natural language descriptions of books you're interested in
- Filter by category (Fiction, Non-fiction, Children's books, etc.)
- Filter by emotional tone
- View book covers and descriptions of recommended titles

## Project Structure

- `app.py` - Main application with Gradio interface
- `requirements.txt` - Project dependencies
- Notebooks:
  - `data-exploration.ipynb` - Initial data analysis
  - `test-classification.ipynb` - Book category classification
  - `sentiment-analysis.ipynb` - Emotional tone analysis
  - `vector-search.ipynb` - Semantic search implementation

## Technical Details

- Uses Google's Generative AI for embeddings
- Implements ChromaDB for vector storage and similarity search
- Emotion classification using DistilRoBERTa
- Category classification using BART zero-shot learning
- Built with LangChain, Pandas, and Gradio

## Deployment

### Live Demo
The application is currently deployed and available at:
- [Hugging Face Spaces](https://ashish-kharde1-book-recommendation.hf.space/) - Try the live demo!

### Deployment Options

1. Hugging Face Spaces
   - Fork the repository
   - Create a new Space on Hugging Face
   - Connect your GitHub repository
   - Add environment variables for API keys
   - Deploy using the provided `requirements.txt`


### Deployment Requirements
- Python 3.8+
- 2GB RAM minimum
- Google Cloud API key
- Internet connection for model downloads
- Storage space for embeddings (~500MB)

## License

Made with ❤️ by Ashish Kharde © 2025
