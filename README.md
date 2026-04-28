# 🚀 MeetIQ — AI Meeting Intelligence Agent
An advanced multi-agent AI system that transforms meeting preparation into a fast, structured, and data-driven process.
---

## 🌐 Live Demo

👉 **Try MeetIQ:**  
https://huggingface.co/spaces/Souvikbasur/MeetIQ

> ⚠️ Note: MeetIQ may show errors if Google Gemini or Serper API rate limits are exceeded. Wait for reset or upgrade API quota.
---

## 🎥 Demo Video

A full walkthrough of MeetIQ’s multi-agent pipeline is available in this repository. It demonstrates how the AI agents collaborate in real-time to generate an executive brief

---

## 🧠 Overview

**MeetIQ** is a **multi-agent AI meeting preparation system** that automatically generates **executive-level meeting briefs**.

It combines:
- Intelligent decision-making  
- Real-time web search  
- Context-aware memory  
- Strategic planning  

👉 Result: You walk into meetings **fully prepared, confident, and data-backed**.

---

## ✨ What MeetIQ Delivers

- 📄 Executive-ready briefing (900–1300 words)
- 🔍 Company research & insights  
- 📊 Industry trends & competitors  
- 🗺 Time-boxed meeting agenda  
- ❓ Anticipated questions & answers  
- 🎯 Actionable next steps  
- 📥 Export as PDF  

---

## 🤖 Multi-Agent System

MeetIQ is powered by **6 specialized AI agents**, each solving a specific part of the problem:

| Agent | Role |
|------|------|
| 🧠 Decision Agent | Orchestrates the pipeline (search, memory, depth, priority) |
| 🔍 Context Agent | Researches company background |
| 📊 Industry Agent | Analyzes market & competitors |
| 🗺 Strategy Agent | Builds agenda & strategy |
| 📋 Brief Agent | Generates executive briefing |
| ✨ Reflection Agent | Polishes and refines output |

---

## 🔍 Search Agent (Key Innovation)

MeetIQ includes a **dynamic Search Agent system** powered by real-time web search.

### ⚙️ How it works:
The **Decision Agent** determines:
- `SEARCH: ALWAYS` → deep research (4–5 searches)
- `SEARCH: LIGHT` → moderate research (2–3 searches)
- `SEARCH: MINIMAL` → minimal validation (1 search)

### 🔗 Tools Used:
- **Serper API** → Google Search results
- Integrated via CrewAI tools

### 💡 Why it matters:
- Avoids unnecessary API calls  
- Improves accuracy with real-time data  
- Adapts based on meeting complexity  

---

## 🧠 Memory System

- Stores past meetings in `memory.json`
- Injects relevant context into future runs
- Maintains continuity across sessions

### Features:
- Smart memory usage (only when relevant)
- Atomic file writes (no corruption)
- Works seamlessly on Hugging Face (`/tmp` storage)

---

## ⚙️ Tech Stack

### 🧩 Frameworks & Core
- **CrewAI** → Multi-agent orchestration  
- **Flask** → Backend API server  
- **Gunicorn** → Production server  

### 🤖 AI Models
- **Gemini 2.5 Flash** → High-quality output  
- **Gemini 2.5 Flash Lite** → Fast decision & research  

### 🔍 APIs & Tools
- **Serper API** → Web search  
- **CrewAI Tools** → Agent tool integration  

### 🎨 Frontend
- HTML, CSS, JavaScript  
- Marked.js → Markdown rendering  
- html2pdf.js → PDF export  

### 🐳 Deployment
- Docker  
- Hugging Face Spaces  

---

## 🚀 System Flow
User Input
→ 
🧠 Decision Agent
→ 
🔍 Context Agent (Search Enabled)
→ 
📊 Industry Agent (Search Enabled)
→ 
🗺 Strategy Agent
→ 
📋 Brief Agent
→ 
✨ Reflection Agent
→ 
📄 Final Executive Brief

---

## ⚡ Key Features

### 🧠 Smart Decision Engine
- Controls search, memory, and depth  
- Optimizes performance & cost  

### 🔍 Adaptive Search System
- Real-time web data  
- Context-aware search intensity  

### ⚡ Dual-Model Optimization
- Fast model for simple tasks  
- Powerful model for final output  
- Automatic fallback on rate limits  

### 🗂 Persistent Memory
- Stores and reuses past insights  
- Improves future outputs  

### 🖥 Premium UI
- Real-time agent pipeline  
- Transparent AI decisions  
- One-click PDF export  

---

## 💡 Why MeetIQ Matters

### ❌ Traditional Approach
- Hours of manual research  
- Unstructured notes  
- Missed insights  

### ✅ MeetIQ Approach
- 2-minute preparation  
- Structured executive brief  
- Data-driven decisions  

---

## 👥 Who Should Use This?

- 🧑‍💼 Founders & Entrepreneurs  
- 📊 Product Managers  
- 💼 Consultants  
- 🎓 Students
  
---

## 🛠️ Run Locally
### 1. Clone repo
```bash
git clone https://github.com/souvikbasuroy/MeetIQ.git
cd MeetIQ
