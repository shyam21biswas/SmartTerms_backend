"""
SmartTerms FastAPI - Ultra Lightweight (No ML Models)
Memory optimized for 512MB RAM - Keyword search only
"""

import os
import sys
import warnings
import gc
from datetime import datetime
from typing import List, Dict, Optional

import google.generativeai as genai
import requests
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Memory management
gc.set_threshold(700, 10, 10)
warnings.filterwarnings('ignore')

# Install dependencies
try:
    from rank_bm25 import BM25Okapi
    import nltk
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("📦 Downloading NLTK data...")
        nltk.download('punkt', quiet=True)
        nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize
except ImportError:
    print("⚠️ Installing dependencies...")
    os.system(f'{sys.executable} -m pip install rank-bm25 nltk')
    from rank_bm25 import BM25Okapi
    import nltk
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
    from nltk.tokenize import sent_tokenize


# Configuration
class Config:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    MIN_CHUNK_SENTENCES = 2
    MAX_CHUNK_SENTENCES = 6
    SENTENCE_OVERLAP = 1
    TOP_K_RESULTS = 3
    RELEVANCE_THRESHOLD = 0.1  # Lower threshold for keyword-only search


# Web Scraper
class SimpleWebScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }

    def scrape_page(self, url: str) -> Optional[Dict]:
        try:
            print(f"  📄 Scraping: {url[:70]}...")
            response = self.session.get(url, headers=self.headers, timeout=15, allow_redirects=True)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, 'html.parser')
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
        print(f"\n🌐 Scraping URL: {url}")
        result = self.scrape_page(url)
        return [result] if result else []

    def cleanup(self):
        self.session.close()


# Semantic Chunker
class SemanticChunker:
    @staticmethod
    def chunk_by_sentences(text: str, metadata: Dict) -> List[Dict]:
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


# Keyword-Only Vector Store (No ML Models)
class KeywordVectorStore:
    """Lightweight keyword search - no embeddings needed"""

    def __init__(self):
        print("🔍 Initializing keyword search (no ML models)...")
        self.chunks = []
        self.bm25 = None
        print("✅ Search ready")

    def add_chunks(self, chunks: List[Dict]):
        if not chunks:
            return

        print(f"💾 Indexing {len(chunks)} chunks...")
        self.chunks.extend(chunks)

        texts = [c['text'] for c in chunks]
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

        gc.collect()
        print(f"✅ Indexed {len(self.chunks)} total chunks")

    def search(self, query: str) -> List[Dict]:
        if not self.chunks or self.bm25 is None:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Normalize scores
        max_score = max(scores) if max(scores) > 0 else 1
        normalized_scores = scores / max_score

        # Get top results
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:Config.TOP_K_RESULTS]

        results = []
        for idx in top_indices:
            if normalized_scores[idx] >= Config.RELEVANCE_THRESHOLD:
                results.append({
                    **self.chunks[idx],
                    'score': float(normalized_scores[idx])
                })

        return results


# Gemini LLM
class GeminiLLM:
    def __init__(self, api_key: str):
        print("🤖 Initializing Gemini API...")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Gemini ready")

    def generate_summary(self, content: str, title: str) -> str:
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
        common_terms = ['data', 'privacy', 'refund', 'cancel', 'payment', 'terms',
                        'policy', 'account', 'rights', 'liability', 'warranty']
        query_lower = query.lower()

        if any(term in query_lower for term in common_terms):
            return True

        off_topic = ['weather', 'sports', 'recipe', 'capital', 'who is']
        if any(term in query_lower for term in off_topic):
            return False

        return True


# RAG System
class RAGSystem:
    def __init__(self, api_key: str):
        print("\n" + "=" * 70)
        print("🚀 Initializing SmartTerms RAG System (Lightweight)")
        print("=" * 70 + "\n")

        self.scraper = SimpleWebScraper()
        self.chunker = SemanticChunker()
        self.vector_store = KeywordVectorStore()
        self.llm = GeminiLLM(api_key)
        self.documents = {}

        print("\n✅ RAG System Ready\n")

    def analyze_url(self, url: str) -> Dict:
        print(f"\n{'=' * 70}")
        print(f"📄 Analyzing: {url}")
        print(f"{'=' * 70}\n")

        self.vector_store = KeywordVectorStore()
        self.documents = {}
        gc.collect()

        scraped_docs = self.scraper.scrape(url)

        if not scraped_docs:
            raise HTTPException(
                status_code=400,
                detail="Failed to scrape URL. Website may be blocking requests or URL is invalid."
            )

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

        self.vector_store.add_chunks(all_chunks)

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
        if not self.documents:
            raise HTTPException(
                status_code=400,
                detail="No document analyzed yet. Call /analyze first."
            )

        question_lower = question.lower().strip()
        if any(kw in question_lower for kw in ['summary', 'summarize', 'overview', 'tldr']):
            return {
                "answer": self.documents.get("summary", "No summary available"),
                "sources": []
            }

        if not self.llm.classify_relevance(question, self.documents.get("main_title", "")):
            return {
                "answer": "❌ This question doesn't seem relevant to the document.\n\n💡 Try: 'summary', 'what data is collected', 'refund policy', etc.",
                "sources": []
            }

        results = self.vector_store.search(question)

        if not results:
            return {
                "answer": "❌ No relevant information found in the document.\n\n💡 Try asking for a 'summary' first.",
                "sources": []
            }

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
        self.scraper.cleanup()
        gc.collect()


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
    description="AI-powered Terms & Conditions analyzer (Lightweight)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag_system: Optional[RAGSystem] = None


@app.on_event("startup")
async def startup_event():
    global rag_system

    if not Config.GEMINI_API_KEY:
        print("⚠️ WARNING: GEMINI_API_KEY not found!")
        raise ValueError("GEMINI_API_KEY environment variable is required!")

    rag_system = RAGSystem(Config.GEMINI_API_KEY)
    gc.collect()


@app.on_event("shutdown")
async def shutdown_event():
    if rag_system:
        rag_system.cleanup()


@app.get("/", response_model=HealthResponse)
def root():
    return {
        "status": "online",
        "message": "SmartTerms API is running. Visit /docs for API documentation."
    }


@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "message": f"RAG system: {'Ready' if rag_system else 'Not initialized'}"
    }


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_url(request: AnalyzeRequest):
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
    if not rag_system:
        raise HTTPException(status_code=503, detail="RAG system not initialized")

    try:
        result = rag_system.query(request.question)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        reload=False
    )
