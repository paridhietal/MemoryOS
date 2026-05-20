import sys
import streamlit as st
from groq import Groq
import chromadb
from sentence_transformers import SentenceTransformer
from collections import defaultdict
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)
chroma_client = chromadb.PersistentClient(path="./memory_db")
collection = chroma_client.get_or_create_collection(name="user_memory")
embed_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

DECAY_DAYS = 30

def run_memory_decay():
    all_data = collection.get(include=["metadatas"])
    ids = all_data["ids"]
    metas = all_data["metadatas"]
    cutoff = datetime.now() - timedelta(days=DECAY_DAYS)
    ids_to_delete = []
    for id, meta in zip(ids, metas):
        if meta and "last_accessed" in meta:
            last = datetime.fromisoformat(meta["last_accessed"])
            if last < cutoff:
                ids_to_delete.append(id)
    if ids_to_delete:
        collection.delete(ids=ids_to_delete)
        print(f"Memory decay: deleted {len(ids_to_delete)} old memories", flush=True)

run_memory_decay()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "memory_counter" not in st.session_state:
    st.session_state.memory_counter = 0
if "last_summary" not in st.session_state:
    st.session_state.last_summary = ""
if "last_memory_action" not in st.session_state:
    st.session_state.last_memory_action = ""

def get_topic_tag(summary):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Given this fact about a user, assign ONE short topic tag (1-2 words max, like RAG, Python, ML, NLP, Transformers, Data Science, etc). Only output the tag, nothing else.\nFact: {summary}"}]
    )
    return response.choices[0].message.content.strip().replace("[","").replace("]","")

def fetch_memories(user_message, n=3):
    query_embedding = embed_model.encode(user_message).tolist()
    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n,
            include=["documents", "metadatas"]
        )
        docs = results["documents"][0]
        ids = results["ids"][0]
        now = datetime.now().isoformat()
        for id in ids:
            try:
                existing = collection.get(ids=[id], include=["documents", "metadatas", "embeddings"])
                old_meta = existing["metadatas"][0] or {}
                old_meta["last_accessed"] = now
                collection.update(ids=[id], metadatas=[old_meta])
            except:
                pass
        return docs
    except:
        return []

def get_similar_memory(summary):
    embedding = embed_model.encode(summary).tolist()
    try:
        results = collection.query(
            query_embeddings=[embedding],
            n_results=1,
            include=["documents", "distances"]
        )
        if results["documents"][0]:
            return (
                results["ids"][0][0],
                results["documents"][0][0],
                results["distances"][0][0]
            )
    except Exception as e:
        print(f"DEBUG error: {e}", flush=True)
    return (None, None, None)

def merge_memories(old_memory, new_summary):
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Merge these two facts about a user into one single updated sentence. Keep all important information. Only output the merged sentence, nothing else.\nOld memory: {old_memory}\nNew memory: {new_summary}"}]
    )
    return response.choices[0].message.content.strip()

def store_memory_from_turn(user_msg, ai_response):
    conversation = f"User: {user_msg}\nAI: {ai_response}"
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": f"Read this conversation and write a single sentence summary of what the USER now understands or learned. Only output the summary, nothing else.\nConversation:\n{conversation}"}]
    )
    new_summary = response.choices[0].message.content.strip()
    tag = get_topic_tag(new_summary)
    now = datetime.now().isoformat()

    SIMILARITY_THRESHOLD = 0.95
    existing_id, existing_memory, distance = get_similar_memory(new_summary)
    print(f"DEBUG: id={existing_id}, distance={distance}", flush=True)

    if existing_id and distance < SIMILARITY_THRESHOLD:
        merged = merge_memories(existing_memory, new_summary)
        new_embedding = embed_model.encode(merged).tolist()
        collection.delete(ids=[existing_id])
        collection.add(
            documents=[merged],
            embeddings=[new_embedding],
            ids=[existing_id],
            metadatas=[{"tag": tag, "last_accessed": now}]
        )
        st.session_state.last_memory_action = "UPDATED"
        return merged
    else:
        new_embedding = embed_model.encode(new_summary).tolist()
        memory_id = f"mem_{st.session_state.memory_counter:03d}"
        st.session_state.memory_counter += 1
        collection.add(
            documents=[new_summary],
            embeddings=[new_embedding],
            ids=[memory_id],
            metadatas=[{"tag": tag, "last_accessed": now}]
        )
        st.session_state.last_memory_action = "NEW"
        return new_summary

def chat(user_message):
    memories = fetch_memories(user_message)
    memory_context = "What this user already knows:\n" + "\n".join(f"- {m}" for m in memories) if memories else "No prior knowledge stored yet."
    system_prompt = f"You are a helpful AI tutor. You remember what the user has learned before. Never re-explain things they already know. Pick up from where they left off.\n{memory_context}"
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
    )
    return response.choices[0].message.content.strip()

st.title("Memory AI Tutor")
st.caption("I remember what you have already learned - no need to start from scratch.")

with st.sidebar:
    st.header("Your Memories")
    all_data = collection.get(include=["documents", "metadatas"])
    docs = all_data["documents"]
    metas = all_data["metadatas"]

    if docs:
        grouped = defaultdict(list)
        for doc, meta in zip(docs, metas):
            tag = meta["tag"] if meta and "tag" in meta else "General"
            last = meta.get("last_accessed", "")[:10] if meta else "unknown"
            grouped[tag].append((doc, last))

        for tag, memories in grouped.items():
            st.markdown(f"**[{tag}]**")
            for mem, last in memories:
                st.write(f"• {mem}")
                st.caption(f"Last accessed: {last}")
            st.write("")
    else:
        st.write("No memories yet.")

    if st.button("Clear All Memories"):
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
        st.rerun()

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

user_input = st.chat_input("Ask me anything...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        response = chat(user_input)
        summary = store_memory_from_turn(user_input, response)
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.last_summary = summary
    st.rerun()

if st.session_state.last_summary:
    if st.session_state.last_memory_action == "UPDATED":
        st.caption(f"Memory UPDATED: {st.session_state.last_summary}")
    else:
        st.caption(f"New memory saved: {st.session_state.last_summary}")