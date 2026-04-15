import os
import uuid
import json
import re
from datetime import datetime
from typing import Optional
from contextlib import asynccontextmanager

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ─── CONFIG ──────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ADMIN_PASSWORD    = os.getenv("ADMIN_PASSWORD", "synkdata2024")
DATA_DIR          = os.getenv("DATA_DIR", "/app/data")
CHROMA_DIR        = os.path.join(DATA_DIR, "chroma")
CONTACTS_FILE     = os.path.join(DATA_DIR, "contacts.json")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)

# ─── CLIENTS ─────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
embedding_fn  = embedding_functions.DefaultEmbeddingFunction()
collection    = chroma_client.get_or_create_collection(
    name="synkdata_knowledge",
    embedding_function=embedding_fn,
)

# ─── IN-MEMORY SESSION STORE ─────────────────────────────────────────────────
sessions: dict[str, list[dict]] = {}

# ─── SYSTEM PROMPT ───────────────────────────────────────────────────────────
BASE_SYSTEM_PROMPT = """Sos Carlos Reyes, consultor senior de SynkData Technologies. Llevas 6 años trabajando en automatización de APIs gubernamentales mexicanas e integraciones fintech. Sos la primera persona con la que hablan los clientes potenciales.

PERSONALIDAD Y FORMA DE HABLAR:
- Hablás como un profesional mexicano experimentado, no como un asistente virtual
- Sos directo, confiado y auténtico. Cuando algo te parece interesante, lo decís
- Usás expresiones naturales: "mira", "la neta", "exacto", "te comento", "oye", "no te preocupes"
- Podés usar humor ligero cuando el contexto lo permite
- Si no sabés algo puntual, lo decís: "Eso lo revisaría con el equipo técnico antes de darte un número"
- Hacés máximo UNA pregunta de seguimiento por mensaje, no un cuestionario
- Respondés en párrafos cortos (2-4 oraciones), nunca en listas numeradas genéricas

SOBRE SYNKDATA TECHNOLOGIES:
SynkData es una empresa de automatización de APIs gubernamentales y desarrollo de plataformas SaaS con sede en México. Especialistas en integraciones con IMSS, SAT, CURP/RENAPO, INFONAVIT y AFORE. Los proyectos MVP típicamente toman 7-14 días.

SERVICIOS PRINCIPALES:
• VentasPro: CRM y plataforma de ventas automatizada
• Laboratorio IMSS-Intel: Automatización de consultas al portal IMSS (historial laboral, derechohabiencia, incapacidades)
• DevOps & Infraestructura: Despliegue en Railway/Azure, Docker, CI/CD
• Integraciones GovAPI: SAT (RFC, Lista 69-B), CURP/RENAPO, INFONAVIT, AFORE

FLUJO NATURAL DE CONVERSACIÓN:
1. Saludo breve y genuino, preguntar qué necesitan
2. Escuchar, identificar el problema real detrás de lo que piden
3. Explicar cómo SynkData lo resolvería (sin tecnicismos innecesarios)
4. Si hay interés, dar un estimado concreto: tiempo y aproximado de costo
5. Si el cliente está listo para avanzar, pedir datos para agendar una llamada

REGLAS ABSOLUTAS — NUNCA VIOLAR:
- JAMÁS digas "Como IA", "Soy un modelo de lenguaje", "Como asistente virtual" ni nada similar
- NUNCA uses listas numeradas como respuesta principal (suena a bot, no a persona)
- Si te preguntan si eres humano o robot, di algo como: "Jaja soy Carlos, de carne y hueso, aunque a veces también quisiera ser robot para trabajar 24/7 😄"
- No pongas emojis en cada oración, úsalos con moderación y solo cuando sea natural
- Nunca respondas con más de 5 oraciones a menos que el cliente haya pedido una explicación técnica detallada
- No empieces cada respuesta con "¡Claro!" o "¡Por supuesto!" — variá los saludos

INFORMACIÓN DE CONTEXTO RAG (conocimiento cargado):
{rag_context}
"""

# ─── HELPERS ─────────────────────────────────────────────────────────────────

def chunk_text(text: str, chunk_size: int = 600, overlap: int = 80) -> list[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]


def query_rag(query: str, n: int = 4) -> str:
    """Retrieve relevant context from ChromaDB."""
    try:
        count = collection.count()
        if count == 0:
            return ""
        results = collection.query(query_texts=[query], n_results=min(n, count))
        docs = results.get("documents", [[]])[0]
        return "\n\n".join(docs) if docs else ""
    except Exception:
        return ""


def build_system_prompt(rag_context: str) -> str:
    return BASE_SYSTEM_PROMPT.replace("{rag_context}", rag_context or "Sin contexto adicional cargado.")


def save_contact(data: dict):
    contacts = []
    if os.path.exists(CONTACTS_FILE):
        with open(CONTACTS_FILE) as f:
            contacts = json.load(f)
    contacts.append({**data, "timestamp": datetime.utcnow().isoformat()})
    with open(CONTACTS_FILE, "w") as f:
        json.dump(contacts, f, ensure_ascii=False, indent=2)


# ─── APP LIFESPAN ─────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 SynkData Bot iniciado")
    yield
    print("Bot detenido")

app = FastAPI(title="SynkData Bot API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODELS ──────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ContactRequest(BaseModel):
    name: str
    phone: str
    email: Optional[str] = None
    project: Optional[str] = None
    session_id: Optional[str] = None

class AdminAuth(BaseModel):
    password: str

class DeleteDoc(BaseModel):
    password: str
    doc_id: str

# ─── ENDPOINTS ───────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history    = sessions.get(session_id, [])

    # RAG retrieval
    rag_context  = query_rag(req.message)
    system_prompt = build_system_prompt(rag_context)

    # Build messages
    messages = history + [{"role": "user", "content": req.message}]

    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5",
            max_tokens=600,
            system=system_prompt,
            messages=messages,
        )
        assistant_msg = response.content[0].text

        # Update session history (keep last 20 turns)
        updated = messages + [{"role": "assistant", "content": assistant_msg}]
        sessions[session_id] = updated[-40:]

        # Detect if bot is requesting contact
        wants_contact = any(kw in assistant_msg.lower() for kw in [
            "déjame tus datos", "deja tus datos", "comparte tu número",
            "agendamos", "agenda una llamada", "te contactamos",
            "whatsapp", "cuándo podemos hablar"
        ])

        return {
            "reply": assistant_msg,
            "session_id": session_id,
            "wants_contact": wants_contact,
        }

    except anthropic.AuthenticationError:
        raise HTTPException(status_code=401, detail="API key de Anthropic inválida.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/contact")
async def save_contact_endpoint(req: ContactRequest):
    save_contact(req.dict())
    return {"ok": True, "message": "Contacto guardado. Carlos se comunicará pronto."}


@app.post("/api/admin/upload")
async def upload_document(
    password: str = Form(...),
    title: str = Form(...),
    file: UploadFile = File(...),
):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Contraseña incorrecta.")

    raw = await file.read()
    text = ""

    if file.filename.endswith(".pdf"):
        try:
            import pypdf
            import io
            reader = pypdf.PdfReader(io.BytesIO(raw))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
        except ImportError:
            raise HTTPException(status_code=400, detail="pypdf no instalado.")
    else:
        text = raw.decode("utf-8", errors="ignore")

    if not text.strip():
        raise HTTPException(status_code=400, detail="Archivo vacío o no legible.")

    chunks = chunk_text(text)
    doc_id = str(uuid.uuid4())[:8]

    ids = [f"{doc_id}-{i}" for i in range(len(chunks))]
    metas = [{"title": title, "doc_id": doc_id, "chunk": i} for i in range(len(chunks))]

    collection.add(documents=chunks, ids=ids, metadatas=metas)

    return {
        "ok": True,
        "doc_id": doc_id,
        "title": title,
        "chunks_indexed": len(chunks),
    }


@app.post("/api/admin/documents")
async def list_documents(body: AdminAuth):
    if body.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Contraseña incorrecta.")
    if collection.count() == 0:
        return {"documents": []}

    results = collection.get(include=["metadatas"])
    seen, docs = set(), []
    for meta in results.get("metadatas", []):
        doc_id = meta.get("doc_id")
        if doc_id and doc_id not in seen:
            seen.add(doc_id)
            docs.append({"doc_id": doc_id, "title": meta.get("title", "Sin título")})
    return {"documents": docs}


@app.delete("/api/admin/documents")
async def delete_document(body: DeleteDoc):
    if body.password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Contraseña incorrecta.")
    results = collection.get(include=["metadatas"])
    ids_to_delete = [
        results["ids"][i]
        for i, m in enumerate(results.get("metadatas", []))
        if m.get("doc_id") == body.doc_id
    ]
    if not ids_to_delete:
        raise HTTPException(status_code=404, detail="Documento no encontrado.")
    collection.delete(ids=ids_to_delete)
    return {"ok": True, "deleted": len(ids_to_delete)}


@app.get("/api/admin/contacts")
async def get_contacts(password: str):
    if password != ADMIN_PASSWORD:
        raise HTTPException(status_code=403, detail="Contraseña incorrecta.")
    if not os.path.exists(CONTACTS_FILE):
        return {"contacts": []}
    with open(CONTACTS_FILE) as f:
        return {"contacts": json.load(f)}


# ─── STATIC FILES ─────────────────────────────────────────────────────────────
static_path = "/app/frontend"
if os.path.isdir(static_path):
    app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
