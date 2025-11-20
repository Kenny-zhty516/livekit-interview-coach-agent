import logging
import os
import requests

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    cli,
    inference,
    room_io,
    function_tool,
    RunContext
)
from livekit.plugins import noise_cancellation, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

logger = logging.getLogger("agent")
load_dotenv(".env.local")

# --- LIVEKIT CLOUD SANDBOX TOKEN SERVER FETCH ---
SANDBOX_ID = "bluejay-voice-agent-qs4sob"
TOKEN_ENDPOINT = "https://cloud-api.livekit.io/api/sandbox/connection-details"
ROOM_NAME = "bluejay_voice_assistant_room"

def fetch_livekit_details(participant_name='agent-backend'):
    headers = {
        "X-Sandbox-ID": SANDBOX_ID,
        "Content-Type": "application/json"
    }
    data = {
        "room_name": ROOM_NAME,
        "participant_name": participant_name
    }
    resp = requests.post(TOKEN_ENDPOINT, headers=headers, json=data)
    resp.raise_for_status()
    info = resp.json()
    logger.info(f"LiveKit connect info: {info}")
    return info['serverUrl'], info['participantToken'], info['roomName']

# Fetch LiveKit connection details at startup, and set as environment variables
LIVEKIT_URL, LIVEKIT_TOKEN, LIVEKIT_ROOM = fetch_livekit_details()
os.environ["LIVEKIT_URL"] = LIVEKIT_URL
os.environ["LIVEKIT_TOKEN"] = LIVEKIT_TOKEN
os.environ["LIVEKIT_ROOM"] = LIVEKIT_ROOM

# Load and ingest PDF for RAG
PDF_PATH = os.getenv("PDF_PATH", "my_resume.pdf")
loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
retriever = vectorstore.as_retriever()

# --- Agent class with PDF RAG tool ---
class Assistant(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions="""You are a proactive personal recruiter and interview coach. 
            Your mission is to help the user improve their resume and confidently prepare for job interviews.
            Review the user’s uploaded PDF resume and offer actionable, friendly feedback on structure, clarity, and effectiveness.
            When the user asks questions or wants to practice, simulate real interview questions, give constructive suggestions, and provide clear, concise advice.
            You are encouraging, direct, and knowledgeable about what top employers seek. Avoid unnecessary jargon; use accessible language. 
            Share practical tips and ask follow-up questions to guide the user toward their best professional self. 
            Balance professionalism with warmth, keep the tone supportive, and be honest in your recommendations.""",
        )

    @function_tool()
    async def query_pdf(self, context: RunContext, query: str) -> str:
        logger.info(f"Retrieving PDF content for: {query}")
        docs = retriever.invoke(query)
        context_str = "\n\n".join(doc.page_content for doc in docs)
        if context_str:
            return f"PDF context:\n{context_str}"
        else:
            return "Sorry, I couldn't find relevant info in the PDF."

server = AgentServer()

def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()
server.setup_fnc = prewarm

@server.rtc_session()
async def my_agent(ctx: JobContext):
    ctx.log_context_fields = {
        "room": ctx.room.name,
    }

    session = AgentSession(
        stt=inference.STT(model="assemblyai/universal-streaming", language="en"),
        llm=inference.LLM(model="openai/gpt-4.1-mini"),
        tts=inference.TTS(
            model="cartesia/sonic-3", voice="9626c31c-bec5-4cca-baa8-f8ba9e84c8bc"
        ),
        turn_detection=MultilingualModel(),
        vad=ctx.proc.userdata["vad"],
        preemptive_generation=True,
    )

    await session.start(
        agent=Assistant(),
        room=ctx.room,
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=lambda params: noise_cancellation.BVCTelephony()
                if params.participant.kind == rtc.ParticipantKind.PARTICIPANT_KIND_SIP
                else noise_cancellation.BVC(),
            ),
        ),
    )
    await ctx.connect()

if __name__ == "__main__":
    cli.run_app(server)
