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

# Steps to run this on your system
## Step 1: Install Python and Git
If you don't already have them, you need to install Python and Git on your computer.

Install Python: Download and install Python from the official website. Be sure to check the box that says "Add Python to PATH" during installation.

Install Git: Download and install Git from the official website.

## Step 2: Get the Code and Install Libraries
Clone the Repository: Open your terminal or command prompt and run the following commands to download the project code.

Bash
`git clone https://github.com/Nirvishpatel47/Exam_Helping_Assistant.git`
Install all libraries at once: Run the following command. This will install all required packages globally on your system.

`pip install pandas python-dotenv langchain-community langchain-core langchain-google-genai streamlit langchain-community faiss-cpu``
## Step 3: Configure Your API Key
Your code needs a secret key to work.
Get your Google Gemini API Key from Google AI Studio.
Create a file named .env in the project's main folder.
Open the .env file and add this line, replacing the text in quotes with your actual key:

Code snippet:

`GEMINI_API_KEY="YOUR_API_KEY_HERE"`

## Step 4: Run the Application
Now you can run the program directly.
Put your document (e.g., abc.pdf) in the same folder as your Python script.

Open your terminal in that folder.

Run the script using the python command:

Bash:
`python your_filename.py`
The program will then prompt you to enter the document path and your questions.
