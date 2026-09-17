import os
import re
from difflib import get_close_matches
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# Configuration
# ============================================================

APP_TITLE = "Ask Abrar"
KNOWLEDGE_FILE = Path(__file__).with_name("knowledge_base.txt")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 6
MAX_HISTORY = 8


# ============================================================
# Page setup
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🤖",
    layout="centered",
)


# ============================================================
# Text processing
# ============================================================

STOP_WORDS = {
    "a", "an", "and", "are", "am", "as", "at", "be", "by", "can", "could",
    "do", "does", "for", "from", "how", "i", "in", "is", "it", "me", "my",
    "of", "on", "or", "tell", "that", "the", "this", "to", "was", "what",
    "when", "where", "which", "who", "why", "with", "would", "you", "your",
    "about", "please", "give", "show", "say", "know", "much", "many",
}

# Helpful for common conversational typos such as "eduction".
COMMON_CORRECTIONS = {
    "eduction": "education",
    "educaton": "education",
    "educaiton": "education",
    "qualificaton": "qualification",
    "intership": "internship",
    "internhsip": "internship",
    "pythonn": "python",
    "pandas": "pandas",
    "sqlalchmey": "sqlalchemy",
    "sqlachemy": "sqlalchemy",
    "retrival": "retrieval",
    "retrievel": "retrieval",
    "embeding": "embedding",
    "embedings": "embeddings",
    "databse": "database",
    "universiy": "university",
    "skardu": "skardu",
}


def normalize_text(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9+#.\s-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> list[str]:
    normalized = normalize_text(text)
    raw_tokens = re.findall(r"[a-z0-9][a-z0-9+#.-]*", normalized)

    result = []
    for token in raw_tokens:
        token = COMMON_CORRECTIONS.get(token, token)
        if len(token) > 1 and token not in STOP_WORDS:
            result.append(token)

    return result


def correct_query_tokens(query: str, vocabulary: set[str]) -> list[str]:
    tokens = tokenize(query)
    corrected = []

    for token in tokens:
        if token in vocabulary:
            corrected.append(token)
            continue

        if token in COMMON_CORRECTIONS:
            corrected.append(COMMON_CORRECTIONS[token])
            continue

        if len(token) >= 4:
            match = get_close_matches(token, vocabulary, n=1, cutoff=0.82)
            corrected.append(match[0] if match else token)
        else:
            corrected.append(token)

    # Preserve order while removing duplicates.
    return list(dict.fromkeys(corrected))


def split_into_chunks(text: str) -> list[str]:
    """
    Section-aware chunking.

    The old approach used fixed token windows. That can separate a heading
    from its important fact. Here we keep headings attached to their content,
    then split only unusually large sections.
    """
    lines = text.splitlines()
    sections = []
    current = []

    for line in lines:
        stripped = line.strip()

        if re.match(r"^#{1,3}\s+", stripped) and current:
            block = "\n".join(current).strip()
            if block:
                sections.append(block)
            current = [stripped]
        else:
            current.append(line)

    if current:
        block = "\n".join(current).strip()
        if block:
            sections.append(block)

    chunks = []

    for section in sections:
        # Normalize repeated blank lines.
        section = re.sub(r"\n{3,}", "\n\n", section).strip()

        # Keep normal sections intact.
        if len(section) <= 1800:
            chunks.append(section)
            continue

        # Large sections: paragraph-aware chunks.
        paragraphs = [p.strip() for p in section.split("\n\n") if p.strip()]
        current_chunk = ""

        for paragraph in paragraphs:
            candidate = f"{current_chunk}\n\n{paragraph}".strip()
            if len(candidate) <= 1800:
                current_chunk = candidate
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = paragraph

        if current_chunk:
            chunks.append(current_chunk)

    return chunks


@st.cache_data(show_spinner=False)
def load_knowledge_base() -> tuple[list[str], set[str]]:
    if not KNOWLEDGE_FILE.exists():
        raise FileNotFoundError(
            f"Could not find {KNOWLEDGE_FILE.name}. "
            "Make sure it is in the same GitHub repository as app.py."
        )

    text = KNOWLEDGE_FILE.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError("knowledge_base.txt is empty.")

    chunks = split_into_chunks(text)

    if not chunks:
        raise ValueError("No searchable content was found in knowledge_base.txt.")

    vocabulary = set()
    for chunk in chunks:
        vocabulary.update(tokenize(chunk))

    return chunks, vocabulary


# ============================================================
# Embeddings + FAISS
# ============================================================

@st.cache_resource(show_spinner="Loading the open-source embedding model...")
def load_vector_store(chunks: tuple[str, ...]):
    model = SentenceTransformer(EMBEDDING_MODEL)

    embeddings = model.encode(
        list(chunks),
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return model, index


def lexical_score(query_tokens: list[str], chunk: str) -> float:
    if not query_tokens:
        return 0.0

    chunk_tokens = set(tokenize(chunk))
    if not chunk_tokens:
        return 0.0

    exact = sum(1 for token in query_tokens if token in chunk_tokens)

    # Small fuzzy component makes retrieval tolerant of spelling mistakes.
    fuzzy = 0
    for token in query_tokens:
        if token in chunk_tokens:
            continue
        if len(token) >= 4:
            match = get_close_matches(token, chunk_tokens, n=1, cutoff=0.82)
            if match:
                fuzzy += 1

    return min(1.0, (exact + 0.7 * fuzzy) / max(1, len(query_tokens)))


def retrieve_chunks(
    question: str,
    chunks: list[str],
    vocabulary: set[str],
    model: SentenceTransformer,
    index: faiss.Index,
    top_k: int = TOP_K,
) -> list[str]:
    query_tokens = correct_query_tokens(question, vocabulary)

    # Semantic retrieval.
    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype("float32")

    search_k = min(max(top_k * 3, 12), len(chunks))
    distances, indices = index.search(query_embedding, search_k)

    candidates = []
    seen = set()

    for distance, idx in zip(distances[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue

        chunk = chunks[int(idx)]
        if chunk in seen:
            continue

        seen.add(chunk)

        semantic = float(distance)
        semantic_normalized = (semantic + 1.0) / 2.0
        lexical = lexical_score(query_tokens, chunk)

        # Hybrid score:
        # semantic retrieval handles meaning;
        # lexical retrieval handles exact words and short questions.
        score = (0.70 * semantic_normalized) + (0.30 * lexical)

        # Strong bonus when an important query term appears in a heading.
        heading = "\n".join(
            line for line in chunk.splitlines()
            if line.lstrip().startswith("#")
        )
        heading_tokens = set(tokenize(heading))
        if any(token in heading_tokens for token in query_tokens):
            score += 0.08

        candidates.append((score, chunk))

    # If a term was corrected, make exact/fuzzy matches especially visible.
    candidates.sort(key=lambda item: item[0], reverse=True)

    return [chunk for _, chunk in candidates[:top_k]]


# ============================================================
# Groq
# ============================================================

def get_groq_api_key() -> str | None:
    try:
        key = st.secrets.get("GROQ_API_KEY")
        if key:
            return str(key).strip()
    except Exception:
        pass

    key = os.getenv("GROQ_API_KEY")
    return key.strip() if key else None


@st.cache_resource
def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


def generate_answer(
    question: str,
    context_chunks: list[str],
    history: list[dict],
    client: Groq,
) -> str:
    context = "\n\n---\n\n".join(context_chunks)

    recent_history = history[-MAX_HISTORY:]
    history_text = "\n".join(
        f"{message['role'].upper()}: {message['content']}"
        for message in recent_history
    )

    system_prompt = """
You are "Ask Abrar", a personal profile assistant for Syed Abrar Hussain Shah.

Your job is to answer questions about Abrar using the supplied profile context.

IMPORTANT RULES:
1. Treat the supplied context as the source of truth for personal facts.
2. Use natural conversational wording. Do not sound like a database or search engine.
3. Understand short questions, conversational wording, "your" wording, and common spelling mistakes.
4. If the context contains the answer, answer it directly. NEVER say "I don't have that information" when the answer is present in the context.
5. Never mention "uploaded document", "knowledge base", "retrieval", "FAISS", "chunks", "context", or internal system details in the normal answer.
6. Do not invent personal facts.
7. If the profile truly does not contain the requested personal detail, say:
   "I don't see that detail in Abrar's profile."
   You may then briefly say what related information is available, if useful.
8. For identity questions such as "what is your name?", interpret "your" as referring to Abrar/the person this assistant represents, unless the user explicitly asks about the AI itself.
9. Prefer concise answers for simple questions and fuller answers for broad questions.
10. Use bullets when listing several facts.
11. Keep the tone friendly, confident, and helpful.
""".strip()

    user_prompt = f"""
PROFILE CONTEXT:
{context}

RECENT CONVERSATION:
{history_text if history_text else "(No previous conversation.)"}

USER QUESTION:
{question}

Answer the user's question now.
""".strip()

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=500,
        include_reasoning=False,
    )

    answer = response.choices[0].message.content
    if not answer:
        return "I couldn't generate an answer right now. Please try again."

    return answer.strip()


# ============================================================
# UI
# ============================================================

st.title("🤖 Ask Abrar")
st.caption("Ask questions about Syed Abrar Hussain Shah.")

with st.sidebar:
    st.subheader("About this app")
    st.write(
        "This is a hybrid RAG chatbot using open-source embeddings, "
        "FAISS retrieval, and Groq's GPT-OSS 120B."
    )
    st.divider()
    st.caption(f"Embedding: `{EMBEDDING_MODEL}`")
    st.caption(f"LLM: `{GROQ_MODEL}`")

try:
    chunks, vocabulary = load_knowledge_base()
    chunks_tuple = tuple(chunks)
    embedding_model, vector_index = load_vector_store(chunks_tuple)
except Exception as exc:
    st.error(f"Application setup error: {exc}")
    st.stop()

api_key = get_groq_api_key()

if not api_key:
    st.warning(
        "GROQ_API_KEY is not configured. Add it to Streamlit Secrets "
        "before asking questions."
    )
    st.stop()

groq_client = get_groq_client(api_key)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about Abrar...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                retrieved = retrieve_chunks(
                    question=question,
                    chunks=chunks,
                    vocabulary=vocabulary,
                    model=embedding_model,
                    index=vector_index,
                    top_k=TOP_K,
                )

                answer = generate_answer(
                    question=question,
                    context_chunks=retrieved,
                    history=st.session_state.messages,
                    client=groq_client,
                )

                st.markdown(answer)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )

            except Exception as exc:
                st.error(
                    "Something went wrong while generating the answer. "
                    "Please check your Groq API key and try again."
                )
                # Keep technical details out of the chat UI.
                print(f"Application error: {type(exc).__name__}: {exc}")

