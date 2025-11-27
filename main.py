"""
SmartTerms FastAPI - OPTIMIZED for Gemini API Rate Limits
Only 1 Gemini call per analyze, 1 per query (total: 2 calls max per session)
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
    TOP_K_RESULTS = 5  # Increased for better context
    RELEVANCE_THRESHOLD = 0.1


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


# Keyword-Only Vector Store
class KeywordVectorStore:
    def __init__(self):
        print("🔍 Initializing keyword search...")
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
        
        max_score = max(scores) if max(scores) > 0 else 1
        normalized_scores = scores / max_score

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:Config.TOP_K_RESULTS]

        results = []
        for idx in top_indices:
            if normalized_scores[idx] >= Config.RELEVANCE_THRESHOLD:
                results.append({
                    **self.chunks[idx],
                    'score': float(normalized_scores[idx])
                })

        return results


# Gemini LLM - OPTIMIZED
class GeminiLLM:
    def __init__(self, api_key: str):
        print("🤖 Initializing Gemini API...")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-flash')
        self.call_count = 0  # Track API calls
        print("✅ Gemini ready")

    def generate_summary(self, content: str, title: str) -> str:
        """ONLY 1 GEMINI CALL - for summary generation"""
        self.call_count += 1
        print(f"🔥 Gemini API Call #{self.call_count} - Generating summary")
        
        prompt = f"""Analyze this Terms & Conditions document and create a comprehensive summary.

Title: {title}
Content: {content[:5000]}

Provide a detailed summary in this exact format:

🎯 QUICK OVERVIEW
[2-3 sentences explaining what this document covers]

📋 KEY POINTS
• [Important point 1]
• [Important point 2]
• [Important point 3]
• [Important point 4]
• [Important point 5]

👤 YOUR OBLIGATIONS
• [What you must do]
• [What you agree to]

🚫 RESTRICTIONS
• [What you cannot do]
• [Limitations on service]

⚠️ IMPORTANT NOTICES
• [Critical information]
• [Things to watch out for]

Keep it clear and actionable."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def answer_question(self, question: str, chunks: List[Dict], cached_summary: str = None) -> str:
        """ONLY 1 GEMINI CALL - for answering questions"""
        self.call_count += 1
        print(f"🔥 Gemini API Call #{self.call_count} - Answering question")
        
        # Use more context from chunks
        context = "\n\n".join([f"Section {i+1}:\n{c['text']}" for i, c in enumerate(chunks)])

        prompt = f"""You are analyzing a Terms & Conditions document. Answer the user's question based ONLY on the provided context.

Context from document:
{context[:4000]}

User's Question: {question}

Provide a clear, direct answer. If the information is not in the context, say so. Be specific and cite relevant parts."""

        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"Error: {str(e)}"


# RAG System
class RAGSystem:
    def __init__(self, api_key: str):
        print("\n" + "=" * 70)
        print("🚀 Initializing SmartTerms RAG System (API Optimized)")
        print("=" * 70 + "\n")

        self.scraper = SimpleWebScraper()
        self.chunker = SemanticChunker()
        self.vector_store = KeywordVectorStore()
        self.llm = GeminiLLM(api_key)
        self.documents = {}

        print("\n✅ RAG System Ready\n")

    def analyze_url(self, url: str) -> Dict:
        """Scrape and analyze - ONLY 1 GEMINI CALL"""
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

        print(f"\n✅ Analysis complete!")
        print(f"📊 Total Gemini API calls for this analysis: 1")
        print(f"📦 Chunks created: {len(all_chunks)}\n")

        return {
            "message": "Analysis successful",
            "documents_scraped": len(scraped_docs),
            "chunks_created": len(all_chunks),
            "summary": summary,
            "api_calls_used": 1
        }

    def query(self, question: str) -> Dict:
        """Query document - ONLY 1 GEMINI CALL (or 0 if returning cached summary)"""
        if not self.documents:
            raise HTTPException(
                status_code=400,
                detail="No document analyzed yet. Call /analyze first."
            )

        question_lower = question.lower().strip()
        
        # NO GEMINI CALL - Return cached summary
        if any(kw in question_lower for kw in ['summary', 'summarize', 'overview', 'tldr', 'sum']):
            print("💾 Returning cached summary (0 API calls)")
            return {
                "answer": self.documents.get("summary", "No summary available"),
                "sources": [],
                "api_calls_used": 0
            }

        # Search for relevant chunks (NO GEMINI CALL)
        results = self.vector_store.search(question)

        if not results:
            # NO GEMINI CALL - Simple response
            print("❌ No relevant chunks found (0 API calls)")
            return {
                "answer": "I couldn't find relevant information about that in the document. Try asking for a 'summary' to see what topics are covered.",
                "sources": [],
                "api_calls_used": 0
            }

        # ONLY 1 GEMINI CALL - For answering
        print(f"💬 Answering question with {len(results)} relevant chunks")
        answer = self.llm.answer_question(question, results, self.documents.get("summary"))

        sources = [
            {
                "title": r["title"],
                "url": r["url"],
                "score": round(r["score"], 2)
            }
            for r in results
        ]

        print(f"📊 API calls used for this query: 1\n")

        return {
            "answer": answer,
            "sources": sources,
            "api_calls_used": 1
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
    api_calls_used: int = Field(default=1, description="Number of Gemini API calls")


class QueryRequest(BaseModel):
    question: str = Field(..., example="What data do you collect?")


class Source(BaseModel):
    title: str
    url: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: List[Source]
    api_calls_used: int = Field(default=1, description="Number of Gemini API calls")


class HealthResponse(BaseModel):
    status: str
    message: str


# FastAPI App
app = FastAPI(
    title="SmartTerms API",
    description="AI-powered T&C analyzer - Optimized for API limits",
    version="2.0.0"
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
        "message": "SmartTerms API v2.0 - Optimized for API limits"
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
