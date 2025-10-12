"""
SmartTerms FastAPI - Optimized for Low Memory Deployment
Memory optimized version for 512MB RAM environments
"""

import os
import sys
import time
import warnings
import gc
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import urljoin, urlparse

import google.generativeai as genai
import numpy as np
import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Aggressive memory management
gc.set_threshold(700, 10, 10)
warnings.filterwarnings('ignore')

# ==============================================================================
# INSTALL DEPENDENCIES
# ==============================================================================
try:
    from rank_bm25 import BM25Okapi
    import nltk

    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📦 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize
except ImportError:
    print("⚠️ Installing rank_bm25 and nltk...")
    os.system(f'{sys.executable} -m pip install rank-bm25 nltk')
    from rank_bm25 import BM25Okapi
    import nltk

    nltk.download('punkt', quiet=True)
    from nltk.tokenize import sent_tokenize


# ==============================================================================
# CONFIGURATION - OPTIMIZED FOR LOW MEMORY
# ==============================================================================
class Config:
    # Get API key from environment variable (best practice)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyADyW2JrV62pxL2w-wj7M0oi6ps-7fflFY")

    # MEMORY OPTIMIZATION: Use smaller embedding model
    EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 80MB instead of 420MB

    # Chunking settings
    MIN_CHUNK_SENTENCES = 2  # Reduced
    MAX_CHUNK_SENTENCES = 6  # Reduced
    SENTENCE_OVERLAP = 1

    # Search settings
    SEMANTIC_WEIGHT = 0.7
    BM25_WEIGHT = 0.3
    TOP_K_RESULTS = 3  # Reduced from 5
    RELEVANCE_THRESHOLD = 0.3

    # Crawling limits - REDUCED FOR MEMORY
    MAX_DEPTH = 0  # Only scrape main page
    MAX_PAGES = 1  # Only one page

    # Selenium DISABLED for memory savings
    USE_SELENIUM = False
    PAGE_LOAD_TIMEOUT = 15


# ==============================================================================
# SIMPLE WEB SCRAPER (No Selenium)
# ==============================================================================
class SimpleWebScraper:
    """Lightweight scraper without Selenium"""

    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

    def scrape_page(self, url: str) -> Optional[Dict]:
        """Scrape single page with requests"""
        try:
            print(f"  📄 Scraping: {url[:70]}...")
            response = self.session.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')

            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()

            title = soup.title.string.strip() if soup.title else "Document"
            content = soup.get_text(separator='\n', strip=True)
            lines = [line.strip() for line in content.split('\n') if line.strip() and len(line.strip()) > 3]
            content = '\n'.join(lines)

            if len(content) < 200:
                print(f"  ⚠️ Low content: {len(content)} chars")
                return None

            print(f"  ✅ Scraped: {len(content)} chars")
            return {
                "url": url,
                "title": title,
                "content": content,
                "scraped_at": datetime.now().isoformat()
            }
        except Exception as e:
            print(f"  ❌ Error: {str(e)}")
            return None

    def scrape(self, url: str) -> List[Dict]:
        """Scrape URL (single page only for memory efficiency)"""
        print(f"\n🌐 Scraping URL: {url}")
        result = self.scrape_page(url)
        if result:
            return [result]
        return []

    def cleanup(self):
        """Cleanup session"""
        self.session.close()


# ==============================================================================
# SEMANTIC CHUNKER
# ==============================================================================
class SemanticChunker:
    @staticmethod
    def chunk_by_sentences(text: str, metadata: Dict) -> List[Dict]:
        """Chunk text by sentences"""
        sentences = sent_tokenize(text)
        if not sentences:
            return []

        chunks = []
        i = 0
        chunk_index = 0

        while i < len(sentences):
            chunk_sentences = sentences[i:i + Config.MAX_CHUNK_SENTENCES]

            if len(chunk_sentences) >= Config.MIN_CHUNK_SENTENCES:
                chunk_text = ' '.join(chunk_sentences).strip()
                chunks.append({
                    "text": chunk_text,
                    "chunk_index": chunk_index,
                    **metadata
                })
                chunk_index += 1

            i += Config.MAX_CHUNK_SENTENCES - Config.SENTENCE_OVERLAP

        return chunks


# ==============================================================================
# HYBRID VECTOR STORE
# ==============================================================================
class HybridVectorStore:
    """Hybrid search with semantic + keyword matching"""

    def __init__(self):
        print(f"🧠 Loading embedding model: {Config.EMBEDDING_MODEL}...")
        self.model = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.chunks = []
        self.embeddings = None
        self.bm25 = None
        gc.collect()  # Clean up after loading model
        print("✅ Model loaded")

    def add_chunks(self, chunks: List[Dict]):
        """Add and index chunks"""
        if not chunks:
            return

        print(f"💾 Indexing {len(chunks)} chunks...")
        self.chunks.extend(chunks)

        texts = [c['text'] for c in chunks]
        new_embeddings = self.model.encode(texts, show_progress_bar=False)

        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        # BM25 index
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

        gc.collect()  # Clean up
        print(f"✅ Indexed {len(self.chunks)} total chunks")

    def search(self, query: str) -> List[Dict]:
        """Hybrid search"""
        if not self.chunks or self.embeddings is None:
            return []

        # Semantic search
        query_embedding = self.model.encode([query])[0]
        sem_scores = np.dot(self.embeddings, query_embedding) / (
                np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_embedding)
        )

        # BM25 search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_norm = bm25_scores / (max(bm25_scores) if max(bm25_scores) > 0 else 1)

        # Combine scores
        combined = sem_scores * Config.SEMANTIC_WEIGHT + bm25_norm * Config.BM25_WEIGHT
        top_indices = np.argsort(combined)[-Config.TOP_K_RESULTS:][::-1]

        results = []
        for idx in top_indices:
            if combined[idx] >= Config.RELEVANCE_THRESHOLD:
                results.append({
                    **self.chunks[idx],
                    'score': float(combined[idx])
                })

        return results


# ==============================================================================
# GEMINI LLM HANDLER
# ==============================================================================
class GeminiLLM:
    def __init__(self, api_key: str):
        print("🤖 Initializing Gemini API...")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini ready")

    def generate_summary(self, content: str, title: str) -> str:
        """Generate document summary"""
        prompt = f"""Summarize this document in simple language:

Title: {title}
Content: {content[:3000]}

Format:
🎯 Quick Overview (2-3 sentences)
📋 Key Points (5-7 bullets with emojis)
👤 User Obligations
🚫 Restrictions
⚠️ Important Info

Keep it simple and scannable."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def answer_question(self, question: str, chunks: List[Dict]) -> str:
        """Answer question based on chunks"""
        context = "\n\n".join([f"[{c['title']}]\n{c['text']}" for c in chunks])

        prompt = f"""Answer based ONLY on the context below.

Context:
{context}

Question: {question}

Provide a clear, concise answer."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"

    def classify_relevance(self, query: str, doc_title: str) -> bool:
        """Check if query is relevant"""
        # Simple keyword-based check for speed
        common_terms = ['data', 'privacy', 'refund', 'cancel', 'payment', 'terms',
                        'policy', 'account', 'rights', 'liability', 'warranty']
        query_lower = query.lower()

        # Check if query contains common T&C terms
        if any(term in query_lower for term in common_terms):
            return True

        # Check for obviously off-topic queries
        off_topic = ['weather', 'sports', 'recipe', 'capital', 'who is']
        if any(term in query_lower for term in off_topic):
            return False

        return True  # Default to relevant


# ==============================================================================
# MAIN RAG SYSTEM
# ==============================================================================
class RAGSystem:
    """Main RAG orchestration"""

    def __init__(self, api_key: str):
        print("\n" + "=" * 70)
        print("🚀 Initializing SmartTerms RAG System")
        print("=" * 70 + "\n")

        self.scraper = SimpleWebScraper()
        self.chunker = SemanticChunker()
        self.vector_store = HybridVectorStore()
        self.llm = GeminiLLM(api_key)
        self.documents = {}

        print("\n✅ RAG System Ready\n")

    def analyze_url(self, url: str) -> Dict:
        """Scrape and analyze URL"""
        print(f"\n{'=' * 70}")
        print(f"📄 Analyzing: {url}")
        print(f"{'=' * 70}\n")

        # Reset for new analysis
        self.vector_store = HybridVectorStore()
        self.documents = {}
        gc.collect()

        # Scrape
        scraped_docs = self.scraper.scrape(url)

        if not scraped_docs:
            raise HTTPException(
                status_code=400,
                detail="Failed to scrape URL. Website may be blocking requests or URL is invalid."
            )

        # Process documents
        all_chunks = []
        for doc in scraped_docs:
            metadata = {
                "url": doc["url"],
                "title": doc["title"]
            }

            chunks = self.chunker.chunk_by_sentences(doc["content"], metadata)
            all_chunks.extend(chunks)

            self.documents["main_title"] = doc["title"]
            self.documents["main_content"] = doc["content"]

        # Index chunks
        self.vector_store.add_chunks(all_chunks)

        # Generate summary
        print("\n📝 Generating summary...")
        summary = self.llm.generate_summary(
            self.documents["main_content"],
            self.documents["main_title"]
        )
        self.documents["summary"] = summary

        print(f"\n✅ Analysis complete! Chunks: {len(all_chunks)}\n")

        return {
            "message": "Analysis successful",
            "documents_scraped": len(scraped_docs),
            "chunks_created": len(all_chunks),
            "summary": summary
        }

    def query(self, question: str) -> Dict:
        """Query the analyzed document"""
        if not self.documents:
            raise HTTPException(
                status_code=400,
                detail="No document analyzed yet. Call /analyze first."
            )

        # Check for summary request
        question_lower = question.lower().strip()
        if any(kw in question_lower for kw in ['summary', 'summarize', 'overview', 'tldr']):
            return {
                "answer": self.documents.get("summary", "No summary available"),
                "sources": []
            }

        # Check relevance
        if not self.llm.classify_relevance(question, self.documents.get("main_title", "")):
            return {
                "answer": "❌ This question doesn't seem relevant to the document.\n\n💡 Try: 'summary', 'what data is collected', 'refund policy', etc.",
                "sources": []
            }

        # Search for relevant chunks
        results = self.vector_store.search(question)

        if not results:
            return {
                "answer": "❌ No relevant information found in the document.\n\n💡 Try asking for a 'summary' first.",
                "sources": []
            }

        # Generate answer
        print(f"💬 Answering: {question}")
        answer = self.llm.answer_question(question, results)

        sources = [
            {
                "title": r["title"],
                "url": r["url"],
                "score": round(r["score"], 2)
            }
            for r in results
        ]

        return {
            "answer": answer,
            "sources": sources
        }

    def cleanup(self):
        """Cleanup resources"""
        self.scraper.cleanup()
        gc.collect()


# ==============================================================================
# FASTAPI APPLICATION
# ==============================================================================

# Pydantic Models
class AnalyzeRequest(BaseModel):
    url: str = Field(..., example="https://github.com/site/terms")


class AnalyzeResponse(BaseModel):
    message: str
    documents_scraped: int
    chunks_created: int
    summary: str


class QueryRequest(BaseModel):
    question: str = Field(..., example="What data do you collect?")


class Source(BaseModel):
    title: str
    url: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]


class HealthResponse(BaseModel):
    status: str
    message: str


# FastAPI App
app = FastAPI(
    title="SmartTerms API",
    description="AI-powered Terms & Conditions analyzer with RAG",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, specify your Android app's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance
rag_system: Optional[RAGSystem] = None


@app.on_event("startup")
async def startup_event():
    """Initialize RAG system on startup"""
    global rag_system

    if not Config.GEMINI_API_KEY or Config.GEMINI_API_KEY == "YOUR_API_KEY":
        raise ValueError("GEMINI_API_KEY not configured!")

    rag_system = RAGSystem(Config.GEMINI_API_KEY)
    gc.collect()  # Clean up after initialization


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if rag_system:
        rag_system.cleanup()


# API Endpoints
@app.get("/", response_model=HealthResponse)
def root():
    """Health check endpoint"""
    return {
        "status": "online",
        "message": "SmartTerms API is running. Visit /docs for API documentation."
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Detailed health check"""
    return {
        "status": "healthy",
        "message": f"RAG system: {'Ready' if rag_system else 'Not initialized'}"
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_url(request: AnalyzeRequest):
    """
    Analyze a Terms & Conditions URL

    - **url**: The URL to scrape and analyze

    Returns document summary and metadata
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        result = rag_system.analyze_url(request.url)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/query", response_model=QueryResponse)
async def query_document(request: QueryRequest):
    """
    Ask a question about the analyzed document

    - **question**: Your question about the document

    Returns AI-generated answer with sources
    """
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        result = rag_system.query(request.question)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


# Run server
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("🚀 Starting SmartTerms FastAPI Server")
    print("=" * 70)
    print("\n📝 API Documentation: http://127.0.0.1:8000/docs")
    print("🔍 Health Check: http://127.0.0.1:8000/health\n")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False  # Disable reload in production
    )