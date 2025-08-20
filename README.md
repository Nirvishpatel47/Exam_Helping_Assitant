# Exam Tutor AI
This project is a secure and intelligent AI Exam Tutor powered by a Retrieval-Augmented Generation (RAG) system. The application helps students prepare for exams by generating questions, providing detailed explanations, and ensuring a safe learning environment based on a provided document.

# Features
## Secure & Context-Bound Q&A:
Generates exam-style questions and answers strictly from the provided document context, preventing hallucinations and maintaining data integrity.

## Comprehensive Document Support: 
Capable of processing various file types including PDF, CSV, and TXT, with robust handling of PDFs containing both text and images.

## Hybrid Retrieval System: 
Employs an Ensemble Retriever combining both Vector Search (FAISS) and Keyword Search (BM25) to ensure highly accurate and relevant document retrieval.

## Conversational Memory: 
Utilizes ConversationSummaryBufferMemory to maintain context during a conversation, summarizing past exchanges to support a fluid, ongoing dialogue without overwhelming the model's context window.

## Security & Input Sanitization: 
Includes a layer of input sanitization to detect and prevent common security threats like prompt injection and malicious queries.

# Technologies Used
LangChain: The core framework for building the RAG pipeline.

Google Gemini API: Used for both embeddings (text-embedding-004) and the generative large language model (gemini-1.5-flash).

FAISS: For efficient similarity search and vector storage.

BM25Retriever: For keyword-based retrieval.

pdfloader: For robust document loading.

Python: The primary programming language.

dotenv: For managing API keys and environment variables securely.

Logging: Implements a rotating file handler for logging user questions and AI responses for monitoring and security.
