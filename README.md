# AI-PROJECT
# 🌿 AI Plant Care Assistant

An AI-powered assistant that helps people take care of their plants. Upload plant-care documents and a plant photo, then ask questions about **watering, sunlight, maintenance, and the medicinal and everyday uses of plants**. It runs fully **locally** using Ollama, so it is free and private.

---

## 📌 Features

- 📄 **Document Q&A (RAG):** upload plant-care files (`.txt` / `.pdf`) and get answers based on them
- 📷 **Image analysis:** upload a plant photo to identify the plant and spot visible problems
- 💧 **Care advice:** watering, sunlight, soil, fertilizer, pests and general maintenance
- 🌼 **Plant Uses & Benefits:** medicinal (traditional), culinary, household, environmental uses and safety/toxicity
- 📚 **Sources shown:** every answer shows which part of your documents was used
- 🎨 **Modern Streamlit interface** with quick-question buttons
- 🔒 **100% local:** no API keys, no cost, no data leaves your computer

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Language | Python 3.9+ |
| Web interface | Streamlit |
| Local LLM runtime | Ollama |
| Chat model | llama3.2 |
| Vision model | llava |
| Embedding model | nomic-embed-text |
| PDF reading | pypdf |
| Similarity search | NumPy (cosine similarity) |

---

## 🧠 How It Works

```
Documents -> split into chunks -> nomic-embed-text -> vectors
Question  -> embed -> cosine similarity -> top 3 matching chunks
Photo     -> llava -> plant name and health check
(Chunks + photo info + question) -> llama3.2 -> streamed answer + sources
```

---

## 📁 Project Structure

```
AI-PROJECT/
├── app.py                         # Main Streamlit application
├── requirements.txt               # Python dependencies
├── plant_care_knowledge_base.txt  # Knowledge base with 35 plants (upload this)
├── sample_plant_care.txt          # Small sample file for quick testing
├── tulsi-plant.jpeg               # Sample plant image for testing
├── .gitignore
└── README.md
```

---

## 🚀 Installation and Setup

### 1. Clone the repository
```bash
git clone https://github.com/KallaBrahmmani/AI-PROJECT.git
cd AI-PROJECT
```

### 2. Install Ollama
Download from https://ollama.com/download and make sure it is running.

### 3. Download the AI models (one time)
```bash
ollama pull llama3.2
ollama pull llava
ollama pull nomic-embed-text
```

### 4. Create a virtual environment and install packages
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
pip install -r requirements.txt
```

### 5. Run the app
```bash
streamlit run app.py
```
The app opens at http://localhost:8501

---

## 💡 How to Use

1. In the sidebar, upload `plant_care_knowledge_base.txt` and click **Process documents**.
2. *(Optional)* Upload a plant photo and click **Analyze plant**.
3. Go to **Ask the Assistant** and ask a question, or click a quick button.
4. Go to **Plant Uses & Benefits**, enter a plant name and click **Get uses**.

### Example questions
- How often should I water a rose plant?
- How much sunlight does tulsi need?
- Why are the leaves of my money plant turning yellow?
- What are the medicinal uses of aloe vera?
- Which plants are safe for pets?
- How do I control mealybugs naturally?

---

## 📸 Screenshots

> Add your screenshots to a `screenshots/` folder and link them here.

| Home | Image analysis | Plant uses |
|---|---|---|
| ![Home](screenshots/home.png) | ![Analysis](screenshots/analysis.png) | ![Uses](screenshots/uses.png) |

---

## ⚠️ Limitations

- Image identification may be inaccurate, and small local models can make mistakes.
- Answers are only as good as the documents provided.
- **Medicinal information is educational only and is not medical advice.** Consult a doctor before using any plant for health purposes.

---

## 🔮 Future Improvements

- Watering reminders and plant care scheduler
- Plant disease detection from leaf images
- Multilingual support (Hindi, Telugu)
- Mobile app version
- Larger and verified plant database

---

## 👩‍💻 Author

**Kalla Brahmmani**
GitHub: [@KallaBrahmmani](https://github.com/KallaBrahmmani)

---

## 📄 License

This project is for educational purposes.
