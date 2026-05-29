# MemoryOS

### Why does every AI conversation start from zero?

I was prepping for exams, using AI tools every day. And every single day I had 
to re-explain myself. "I already know the basics of neural networks." 
"We already covered this yesterday." "Stop explaining what a vector is, 
I know what a vector is."

It was exhausting. A good human tutor remembers you. AI didn't.

So I built something that does.

---

## What it actually does

MemoryOS is a chat app that builds a knowledge profile of you over time. 
Every conversation gets summarized and stored. Next time you ask something, 
it already knows your background and picks up from where you left off.

But the interesting part isn't just storing memories — it's how it manages them.

Most systems just keep adding memories forever until it becomes noise. 
MemoryOS checks if a similar memory already exists before storing anything new. 
If you already have a memory about RAG and you just learned something new about RAG, 
it merges them into one updated memory instead of creating a duplicate.

I called it delta memory — only store what actually changed.

Memories also get tagged by topic automatically so the sidebar shows your 
knowledge grouped by subject. And anything you haven't touched in 30 days 
gets quietly deleted — just like how humans forget things they never use.

---

## The part that actually works well

Ask it something you've discussed before. It won't re-explain it. 
It'll say "you already know this, let's go deeper." That's the moment 
that made me think this was worth building.

---

## Tech stack

- **LLaMA 3.3 70B** via Groq — for chat, summarization, merging memories and topic tagging
- **ChromaDB** — vector database that saves memories to disk persistently
- **Sentence Transformers** — converts text to embeddings for semantic search
- **Streamlit** — UI
- **Python 3.10**

---

## Run it yourself

```bash
git clone https://github.com/paridhietal/MemoryOS.git
cd MemoryOS
pip install -r requirements.txt
```

Create a `.env` file:
