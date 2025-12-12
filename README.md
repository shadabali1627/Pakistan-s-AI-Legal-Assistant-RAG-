# AI Legal Assistant RAG Chatbot

![Status](https://img.shields.io/badge/Status-Active-success)

[**Live Demo**](https://pakistan-legal-assistant.gleeze.com)

An AI-powered **Pakistan Legal Assistant** Chatbot that uses Retrieval-Augmented Generation (RAG) to provide accurate legal information specifically related to **Pakistan's law** based on uploaded documents. The application features a modern React frontend and a robust FastAPI backend powered by LangChain and Google's Gemini models.

## 🚀 Features

-   **RAG-powered Chatbot**: Queries a vector database to retrieve relevant legal context before answering.
-   **Document Ingestion**: Supports PDF and docx ingestion for building the knowledge base.
-   **Modern UI**: Built with React and Vite, featuring a responsive design.
-   **FastAPI Backend**: High-performance asynchronous API for handling chat requests.
-   **Vector Search**: Uses ChromaDB for efficient similarity search of legal documents.

## 🛠️ Tech Stack

### Frontend
-   **Framework**: React (Vite)
-   **Routing**: React Router
-   **Styling**: CSS Modules / Vanilla CSS
-   **Markdown Support**: `react-markdown`, `remark-gfm`

### Backend
-   **Framework**: FastAPI
-   **LLM Integration**: LangChain, Google Gemini (GenAI)
-   **Vector Store**: ChromaDB
-   **Document Processing**: `pypdf`, `unstructured`

## 📂 Project Structure

```
├── backend/            # FastAPI backend application
│   ├── data/           # Directory for legal documents
│   ├── routes/         # API routes
│   ├── utils/          # Utility functions
│   ├── vectorstore/    # ChromaDB storage
│   ├── app.py          # Main application entry point
│   ├── ingest.py       # Script to ingest documents
│   ├── rag_pipeline.py # RAG logic and chain
│   └── requirements.txt
│
└── frontend/           # React frontend application
    ├── src/            # Source code
    ├── public/         # Public assets
    ├── index.html      # Entry HTML
    ├── package.json    # Dependencies
    └── vite.config.js  # Vite configuration
```

## ⚡ Getting Started

### Prerequisites
-   Node.js (v18+)
-   Python (v3.9+)
-   Google API Key (for Gemini)

### 1. Backend Setup

Navigate to the backend directory:
```bash
cd backend
```

Create and activate a virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Set up environment variables:
Create a `.env` file in the `backend` directory and add your Google API key:
```env
GOOGLE_API_KEY=your_api_key_here
```

(Optional) Ingest documents:
If you have documents in the `data/` folder, run the ingestion script:
```bash
python ingest.py
```

Run the backend server:
```bash
uvicorn app:app --reload
```
The API will be available at `http://localhost:8000`.

### 2. Frontend Setup

Navigate to the frontend directory:
```bash
cd frontend
```

Install dependencies:
```bash
npm install
```

Run the development server:
```bash
npm run dev
```
The application will run at `http://localhost:5173`.

## 📝 License

This project is licensed under the MIT License.
