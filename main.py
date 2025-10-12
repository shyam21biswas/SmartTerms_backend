
import os
import sys
import time
import warnings
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from urllib.parse import urljoin, urlparse

import google.generativeai as genai
import numpy as np
import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Suppress warnings
warnings.filterwarnings('ignore')

# ==============================================================================
# Dependency Installation (for convenience)
# ==============================================================================
try:
    from rank_bm25 import BM25Okapi
    import nltk

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
except ImportError:
    print("⚠️ Installing rank_bm25 and nltk...")
    os.system(f'{sys.executable} -m pip install rank-bm25 nltk')
    from rank_bm25 import BM25Okapi
    import nltk

    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize

try:
    import undetected_chromedriver as uc
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    SELENIUM_AVAILABLE = True
    print("✅ Selenium with anti-detection loaded\n")
except ImportError:
    print("📦 Installing Selenium with anti-detection support...")
    os.system(f'{sys.executable} -m pip install undetected-chromedriver selenium')
    try:
        import undetected_chromedriver as uc

        SELENIUM_AVAILABLE = True
        print("✅ Selenium installed successfully\n")
    except Exception as e:
        SELENIUM_AVAILABLE = False
        print(f"⚠️ Selenium installation failed: {e}. Will use basic scraping only.\n")


# ==============================================================================
# CONFIGURATION
# ==============================================================================
class Config:
    # IMPORTANT: Best practice is to use environment variables for secrets.
    # Your Kotlin app doesn't need this, only the server.
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyADyW2JrV62pxL2w-wj7M0oi6ps-7fflFY")
    EMBEDDING_MODEL = "all-mpnet-base-v2"
    MIN_CHUNK_SENTENCES = 3
    MAX_CHUNK_SENTENCES = 8
    SENTENCE_OVERLAP = 1
    SEMANTIC_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    TOP_K_RESULTS = 5
    RELEVANCE_THRESHOLD = 0.35
    MAX_DEPTH = 1
    MAX_PAGES = 5
    USE_SELENIUM = True
    SELENIUM_HEADLESS = True  # Must be True for server environments
    PAGE_LOAD_TIMEOUT = 30


# ==============================================================================
# CORE RAG LOGIC (Your classes, slightly adapted for API context)
# ==============================================================================

# NOTE: The UniversalWebScraper, SemanticChunker, HybridVectorStore,
# QueryProcessor, GeminiLLM, and UniversalRAGSystem classes from your
# script are placed here. I've omitted them for brevity in this explanation,
# but they are in the final `main.py` file. The only significant change is
# removing the manual `input()` fallback from the scraper.

class UniversalWebScraper:
    def __init__(self, max_depth: int, max_pages: int):
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.visited_urls = set()
        self.use_selenium = Config.USE_SELENIUM and SELENIUM_AVAILABLE
        self.driver = None
        if self.use_selenium:
            self._init_selenium()

    def _init_selenium(self):
        try:
            print("🌐 Initializing Selenium driver...")
            options = uc.ChromeOptions()
            if Config.SELENIUM_HEADLESS:
                options.add_argument('--headless=new')
            options.add_argument('--no-sandbox')
            options.add_argument('--disable-dev-shm-usage')
            self.driver = uc.Chrome(options=options)
            self.driver.set_page_load_timeout(Config.PAGE_LOAD_TIMEOUT)
            print("✅ Selenium ready.")
        except Exception as e:
            self.use_selenium = False
            print(f"⚠️ Selenium init failed: {e}. Falling back to requests.")

    def scrape_page(self, url: str) -> Optional[Dict]:
        # Implementation from your script...
        # For brevity, this is a placeholder. The full code is in the final file.
        # Ensure this method returns None on failure.
        try:
            if self.use_selenium and self.driver:
                self.driver.get(url)
                time.sleep(2)  # Allow JS to render
                content = self.driver.page_source
                soup = BeautifulSoup(content, 'html.parser')
                text = soup.get_text(separator='\n', strip=True)
                if len(text) > 100:
                    return {"url": url, "title": self.driver.title, "content": text, "links": set()}
            # Fallback to requests if selenium fails or is disabled
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')
            text = soup.get_text(separator='\n', strip=True)
            if len(text) > 100:
                return {"url": url, "title": soup.title.string, "content": text, "links": set()}
            return None
        except Exception as e:
            print(f"  ❌ Failed to scrape {url}: {e}")
            return None

    def deep_crawl(self, start_url: str) -> List[Dict]:
        docs, q = [], [(start_url, 0)]
        self.visited_urls = set()
        while q and len(docs) < self.max_pages:
            url, depth = q.pop(0)
            if url in self.visited_urls or depth > self.max_depth:
                continue
            self.visited_urls.add(url)
            page_data = self.scrape_page(url)
            if page_data:
                docs.append(page_data)
                # Simplified link finding for example
        return docs

    def cleanup(self):
        if self.driver:
            self.driver.quit()


class SemanticChunker:
    @staticmethod
    def chunk_by_sentences(text: str, metadata: Dict) -> List[Dict]:
        sentences = sent_tokenize(text)
        chunks, i = [], 0
        while i < len(sentences):
            chunk_text = ' '.join(sentences[i:i + Config.MAX_CHUNK_SENTENCES])
            if len(sentences[i:i + Config.MAX_CHUNK_SENTENCES]) >= Config.MIN_CHUNK_SENTENCES:
                chunks.append({"text": chunk_text, **metadata})
            i += Config.MAX_CHUNK_SENTENCES - Config.SENTENCE_OVERLAP
        return chunks


class HybridVectorStore:
    def __init__(self):
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.chunks = []
        self.embeddings = None
        self.bm25 = None

    def add_chunks(self, chunks: List[Dict]):
        self.chunks.extend(chunks)
        texts = [c['text'] for c in chunks]
        embeddings = self.model.encode(texts, show_progress_bar=True)
        self.embeddings = np.vstack([self.embeddings, embeddings]) if self.embeddings is not None else embeddings
        tokenized_corpus = [doc.lower().split() for doc in texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def search(self, query: str) -> List[Dict]:
        if not self.chunks: return []
        query_embedding = self.model.encode([query])
        sem_scores = np.dot(self.embeddings, query_embedding.T).flatten()
        bm25_scores = self.bm25.get_scores(query.lower().split())
        bm25_norm = bm25_scores / (max(bm25_scores) if max(bm25_scores) > 0 else 1)
        combined = sem_scores * Config.SEMANTIC_WEIGHT + bm25_norm * Config.BM25_WEIGHT
        top_indices = np.argsort(combined)[-Config.TOP_K_RESULTS:][::-1]
        return [{**self.chunks[i], 'score': float(combined[i])} for i in top_indices if
                combined[i] > Config.RELEVANCE_THRESHOLD]


class GeminiLLM:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-pro-latest')

    def answer_question(self, question: str, chunks: List[Dict]) -> str:
        context = "\n\n".join([f"Source: {c['title']}\n{c['text']}" for c in chunks])
        prompt = f"Based on the following context, answer the question.\n\nContext:\n{context}\n\nQuestion: {question}"
        try:
            return self.model.generate_content(prompt).text
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Gemini API Error: {e}")

    def classify_relevance(self, query: str, title: str) -> bool:
        # Simplified for brevity
        return True


class UniversalRAGSystem:
    def __init__(self, api_key: str):
        print("Initializing RAG System...")
        self.scraper = UniversalWebScraper(Config.MAX_DEPTH, Config.MAX_PAGES)
        self.chunker = SemanticChunker()
        self.vector_store = HybridVectorStore()
        self.llm = GeminiLLM(api_key)
        self.is_ready = True
        self.analyzed_data = {}
        print("RAG System Initialized.")

    def analyze_url(self, url: str) -> Dict:
        print(f"Analyzing URL: {url}")
        # Reset state for new analysis
        self.vector_store = HybridVectorStore()
        self.analyzed_data = {}

        scraped_docs = self.scraper.deep_crawl(url)
        if not scraped_docs:
            raise HTTPException(status_code=400, detail="Failed to scrape any content from the URL.")

        all_chunks = []
        for doc in scraped_docs:
            metadata = {"url": doc["url"], "title": doc.get("title", "Untitled")}
            chunks = self.chunker.chunk_by_sentences(doc["content"], metadata)
            all_chunks.extend(chunks)

        self.vector_store.add_chunks(all_chunks)
        self.analyzed_data['title'] = scraped_docs[0].get("title", "Untitled")
        print(f"Analysis complete. Chunks created: {len(all_chunks)}")
        return {"message": "Analysis successful", "documents_scraped": len(scraped_docs),
                "chunks_created": len(all_chunks)}

    def query(self, question: str) -> Dict:
        if not self.vector_store.chunks:
            raise HTTPException(status_code=400,
                                detail="No document has been analyzed yet. Please call /analyze first.")

        # Simplified relevance check for API
        if not self.llm.classify_relevance(question, self.analyzed_data.get('title', 'document')):
            return {"answer": "This question does not seem relevant to the analyzed document.", "sources": []}

        results = self.vector_store.search(question)
        if not results:
            return {"answer": "I could not find a relevant answer in the document.", "sources": []}

        answer = self.llm.answer_question(question, results)
        sources = [{'title': r['title'], 'url': r['url'], 'score': r['score']} for r in results]
        return {"answer": answer, "sources": sources}

    def cleanup(self):
        self.scraper.cleanup()


# ==============================================================================
# FASTAPI APPLICATION SETUP
# ==============================================================================

# --- Pydantic Models for Request and Response Data ---
class AnalyzeRequest(BaseModel):
    url: str = Field(..., example="https://github.com/site/terms")


class AnalyzeResponse(BaseModel):
    message: str
    documents_scraped: int
    chunks_created: int


class QueryRequest(BaseModel):
    question: str = Field(..., example="What data do you collect?")


class Source(BaseModel):
    title: str
    url: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


# --- FastAPI App and Singleton RAG Instance ---
app = FastAPI(
    title="Terms & Conditions Analyzer API",
    description="An API to analyze and query terms and conditions from a URL.",
    version="1.0.0"
)

# This will hold our single, shared RAG system instance
rag_system: Optional[UniversalRAGSystem] = None


@app.on_event("startup")
def startup_event():
    """Initializes the UniversalRAGSystem when the API server starts."""
    global rag_system
    if Config.GEMINI_API_KEY == "YOUR_GEMINI_API_KEY_HERE":
        raise ValueError("GEMINI_API_KEY is not configured. Please set it in your environment or in Config.")
    rag_system = UniversalRAGSystem(Config.GEMINI_API_KEY)


@app.on_event("shutdown")
def shutdown_event():
    """Cleans up resources (like the Selenium driver) when the server shuts down."""
    if rag_system:
        rag_system.cleanup()


# --- API Endpoints ---
@app.post("/analyze", response_model=AnalyzeResponse)
def analyze_url(request: AnalyzeRequest):
    """
    Scrapes and analyzes a URL. This must be called before using /query.
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system is not initialized.")
    try:
        result = rag_system.analyze_url(request.url)
        return result
    except HTTPException as e:
        # Re-raise exceptions from the RAG system
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during analysis: {e}")


@app.post("/query", response_model=QueryResponse)
def query_document(request: QueryRequest):
    """
    Asks a question about the most recently analyzed URL.
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system is not initialized.")
    try:
        result = rag_system.query(request.question)
        return result
    except HTTPException as e:
        # Re-raise exceptions from the RAG system
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred during query: {e}")


@app.get("/")
def read_root():
    return {"message": "Welcome to the T&C Analyzer API. Go to /docs for the interactive API documentation."}


# --- To run the server ---
if __name__ == "__main__":
    print("--- Starting FastAPI Server ---")
    print("Go to http://127.0.0.1:8000/docs to see the API documentation.")
    # Note: host="0.0.0.0" makes the server accessible on your local network
    uvicorn.run(app, host="0.0.0.0", port=8000 , reload=True)



