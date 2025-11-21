# LiveKit Voice AI Recruiter/Trainer

## Overview

This project is a real-time voice AI assistant that acts as a personal recruiter and interview trainer. It allows users to upload their resume (PDF), receive actionable feedback, and practice interview questions—all with live transcription and a natural conversation experience. The system uses **LiveKit Agents** for voice communication and leverages **Retrieval-Augmented Generation (RAG)** to ground the AI’s responses in the actual resume content.

***

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [RAG Integration Details](#rag-integration-details)
- [Model choices (TTS, LLM, STT) and customization](#model-choices-tts-llm-stt-and-customization)
- [Tools and Frameworks](#tools-and-frameworks)
- [Setup & Running Locally](#setup--running-locally)
- [Design Decisions & Assumptions](#design-decisions--assumptions)
- [Troubleshooting](#troubleshooting)

***

## System Architecture

1. **Frontend (agent-starter-react)**
    - React/Next.js frontend UI for calls and live transcription
    - Allows users to join a LiveKit room and interact by voice or chat
    - Transcripts are displayed live in the UI (with improved design for readability)

2. **Backend (agent-starter-python)**
    - Python agent using LiveKit Agents SDK
    - Loads and semantically indexes the uploaded PDF resume for retrieval
    - Connects to the same LiveKit room as the web client and responds to queries via voice

3. **LiveKit Cloud (Voice Infrastructure)**
    - Handles real-time low-latency audio, agent orchestration, and token authentication

***

## RAG Integration Details

- **Document Ingestion:**  
  The backend agent loads the provided resume PDF using LangChain’s `PyPDFLoader`.
- **Text Chunking:**  
  Resume content is split into overlapping 1,000-character segments using `RecursiveCharacterTextSplitter`, for more contextually relevant search.
- **Semantic Vector Search:**  
  Chunks are embedded using `HuggingFaceEmbeddings` (e.g., `all-MiniLM-L6-v2`) and indexed in a local Chroma vector database.
- **Retrieval:**  
  User queries are matched to the most relevant chunks from the uploaded resume via vector similarity, and this information is made available to the AI as grounding context for LLM generation.
- **Conversation:**  
  The agent uses the retrieved context to provide personalized coaching and feedback, simulating a recruiter/interviewer.

***

## Model Choices (TTS, LLM, STT) and Customization

#### **Speech-to-Text (STT)**
- **Model:** `assemblyai/universal-streaming`
- **Purpose:** Converts user speech to live transcriptions and text queries for the agent.
- **Customization:** You can swap in other STT providers/models via `inference.STT(model=...)` if desired (e.g., Whisper, Deepgram, Google STT).

#### **Language Model (LLM)**
- **Model:** `openai/gpt-4.1-mini`
- **Purpose:** Generates agent responses, interview coaching, resume feedback, and simulated Q&A.
- **Customization:** The system supports any LLM compatible with the LiveKit Agents framework. Swap `inference.LLM(model=...)` to use another model (e.g., Claude, Gemini, custom/hosted LLMs).

#### **Text-to-Speech (TTS)**
- **Model:** `cartesia/sonic-3` (`voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"`)
- **Purpose:** Converts agent responses to clear, natural-sounding voice audio for real-time dialogue.
- **Customization:** Easily change TTS providers/voices by updating the `model=` or `voice=` fields in your agent configuration.

***

#### **How to Customize:**
In `agent-starter-python/src/agent_room.py`, update these lines:
```python
session = AgentSession(
    stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
    llm=inference.LLM(model="openai/gpt-4.1-mini"),
    tts=inference.TTS(
        model="cartesia/sonic-3", 
        voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
    ),
    ...
)
```
- **Swap in** your preferred model keys or voice IDs for any/all modalities.
- Most open and commercial options are supported, as long as a compatible API/SDK is available to LiveKit Agents.

- **STT, LLM, TTS models** are explicitly set and easily customizable for your needs.
- Configuration is in the backend agent Python code.
- This flexibility lets you experiment, tune, or upgrade the system as new models become available.

***

## Tools and Frameworks

- **Frontend:**  
  - React / Next.js  
  - Tailwind CSS for UI  
  - LiveKit Components React SDK
- **Backend:**  
  - Python 3  
  - LiveKit Agents Python SDK  
  - LangChain, Chroma DB, HuggingFace Transformers
- **Realtime Voice:**  
  - LiveKit Cloud (API, token server, and agent management)

***

## Setup & Running Locally

### 1. Clone the Repositories
- Clone `livekit-ai-interview-coach`.

### 2. Backend Agent (Python/RAG)
- Install Python dependencies in `agent-starter-python`.
  ```sh
  cd agent-starter-python
  pip install -r requirements.txt
  ```
- Configure your `.env.local` for the correct PDF path.
- Start the agent **with the correct LiveKit room name**:
  ```sh
  python src/agent_room.py console
  ```

### 3. Frontend (React)
- Install dependencies.
  ```sh
  cd agent-starter-react
  pnpm install   # Or npm install
  ```
- Fill out `.env.local` with LiveKit API keys/server.
- Set a **fixed room name** (recommended for consistent testing) in your `/api/connection-details/route.ts` and use that for both frontend and backend agent.
- Run locally:
  ```sh
  pnpm dev
  # or
  npm run dev
  ```

### 4. Usage
- Access http://localhost:3000 in your browser.
- Join a call, speak, and upload your resume.
- Your agent will respond to queries and coach you as a recruiter.

***

## Design Decisions & Assumptions

### Trade-Offs & Limitations
- **Real-time RAG:** Uses local Chroma DB; for very large resumes/docs, scale may be limited without cloud vector storage.
- **Transcript Display:** Now always visible, but appearance can be further polished for accessibility.
- **LiveKit Room Coordination:** Both agent and web client must use the exact same room name for a session.
- **Voice Animation:** Now smaller and less obtrusive, but visibility is a trade-off with subtlety.

### Hosting Assumptions
- LiveKit Cloud is used for signaling/audio/agent orchestration.
- Credentials are supplied via `.env.local`.

### RAG Assumptions
- **Vector DB:** Local Chroma (swap for managed DB for scale).
- **Chunk Size:** Empirically set to 1000 chars with overlap for best balance.
- **Embeddings:** All MiniLM-L6-v2, for good semantic matching with high perf.

### LiveKit Agent Design
- Python agent launches, joins target LiveKit room, and listens/responds via STT/LLM/TTS.
- Prompts tailored for recruiter/interview coach persona (modifiable in code).

***

## Troubleshooting

- **Transcript/Agent Not Working:** Ensure both frontend and backend use the **same room name**.
- **Visualization Overlap:** Adjust wrapper div size/opacity in TileLayout or session-view components.
- **Resume Not Detected:** Confirm file path in backend and upload process in browser.

***
## Check out the project demo video:
https://www.youtube.com/watch?v=T1JiDi_H3HY&t=171s

***

## Credits

- Built using [LiveKit](https://livekit.io/), [LangChain](https://www.langchain.com/), [Chroma](https://www.trychroma.com/), [HuggingFace](https://huggingface.co/), [Next.js](https://nextjs.org/).
- https://www.perplexity.ai/search/failed-to-connect-to-livekit-s-340faG1vSAuU7ZR5dTjt8w#77

***
