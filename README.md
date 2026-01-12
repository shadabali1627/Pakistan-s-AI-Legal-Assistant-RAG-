# AI Legal Assistant RAG Chatbot ⚖️🤖

![Status](https://img.shields.io/badge/Status-Deployed-success)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![React](https://img.shields.io/badge/React-18-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)

[**🌐 Live Demo: https://ai-legal-assistant.webredirect.org**](https://ai-legal-assistant.webredirect.org)

An advanced **Retrieval-Augmented Generation (RAG)** chatbot designed to answer questions about **Pakistan's Law System**. It ingests legal documents (PDFs), vectorizes them for semantic search, and uses Google's Gemini LLMs to generate accurate, cited responses.

---

## 🧠 AI Models & Technology

We use state-of-the-art Google AI models via the LangChain framework:

| Component | Model / Technology | Description |
| :--- | :--- | :--- |
| **LLM (Generation)** | **Gemini 1.5 Flash** (or `gemma-2-9b-it`) | Generates human-like, context-aware answers. Configurable via `.env`. |
| **Embeddings** | **text-embedding-004** | Converts legal text into 768-dimensional vectors for semantic search. |
| **Vector Database** | **ChromaDB** | Stores and indexes document embeddings for fast retrieval. |
| **Orchestration** | **LangChain** | Manages the RAG pipeline (Retrieval + Generation). |
| **Database** | **MongoDB Atlas** | Stores user chat history and session metadata. |

---

## 🚀 Features

-   **⚡ Real-Time Streaming**: Responses are streamed token-by-token (typewriter effect) using Server-Sent Events (SSE).
-   **📚 Citation Support**: Every answer cites the source PDF and page number used.
-   **🔐 Secure Authentication**: Google OAuth 2.0 integration for secure user login.
-   **📱 Responsive UI**: Modern, mobile-friendly interface built with React & Tailwind/CSS.
-   **☁️ Cloud Native**: Fully deployed on **AWS EC2** with Nginx reverse proxy and Systemd auto-healing.
-   **🧹 Auto-History**: Saves your conversations to MongoDB for easy retrieval.

---

## 🛠️ Tech Stack

### **Frontend**
-   **Framework**: React (Vite)
-   **State**: React Hooks (Custom Auth & Chat hooks)
-   **HTTP Client**: Axios (with Interceptors)
-   **Styling**: Modern CSS (Glassmorphism design)

### **Backend**
-   **Framework**: FastAPI (Async/Await)
-   **Server**: Uvicorn (ASGI)
-   **Search**: ChromaDB (Local vector store)
-   **PDF Processing**: `pypdf`, `recursive-character-text-splitter`

### **Deployment (DevOps)**
-   **Server**: AWS EC2 (Ubuntu 24.04 LTS)
-   **WebServer**: Nginx (Reverse Proxy, SSL, Compression)
-   **Process Manager**: Systemd (`ai-legal-assistant.service`)
-   **Security**: Let's Encrypt SSL, UFW Firewall

---

## 📂 Project Structure

```
├── backend/            # FastAPI Application
│   ├── data/pdfs/      # Folder for raw PDF documents
│   ├── routes/         # API Endpoints (Auth, Chat)
│   ├── vectorstore/    # ChromaDB persistent storage
│   ├── app.py          # Main entry point
│   ├── ingest.py       # RAG Ingestion Script
│   └── rag_pipeline.py # Core RAG Logic
│
└── frontend/           # React Application
    ├── src/            # Components & Hooks
    ├── dist/           # Production Build
    └── vite.config.js  # Build Configuration
```

---

## ⚡ Deployment & Setup Guide

### 1. Local Development by following steps:

**Backend:**
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
# Create .env file with GOOGLE_API_KEY, MONGODB_URL
uvicorn app:app --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### 2. AWS EC2 Deployment Commands

**Restart Server:**
```bash
sudo systemctl restart ai-legal-assistant
```

**View Logs:**
```bash
sudo journalctl -u ai-legal-assistant -f
```

**Re-Ingest Data (Update Knowledge Base):**
```bash
sudo systemctl stop ai-legal-assistant
cd ~/Pakistan-s-AI-Legal-Assistant-RAG-
source backend/.venv/bin/activate
python -m backend.ingest
sudo systemctl start ai-legal-assistant
```

---

## 📝 License

This project is licensed under the MIT License.
