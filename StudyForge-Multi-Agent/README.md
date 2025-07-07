# 📚 StudyForge: Multi-Agent AI Assistant for Exam Revision

This project is a comprehensive, multi-agent academic assistant designed to help students efficiently revise for exams using their own PDF documents (such as textbooks, notes, or research papers). Built with **CrewAI**, **LlamaIndex**, and **LangChain**, it leverages the power of LLMs and agent orchestration to automate the creation of study materials and interactive Q&A.
---

## 😫 The Problem

Students are overwhelmed by long PDFs, textbooks, and scattered online content — especially during exams. There's no time to revise everything.

---

## 💡 The Solution

This assistant turns your PDFs into a structured study guide, complete with notes, MCQs, explanations, and even videos — in minutes.

---
> ⚡️ *Turn any textbook or notes PDF into a complete, interactive study assistant in minutes!*

## 🚀 Key Features

- **PDF Upload & Processing:** Students can upload their study PDFs, which are automatically chunked and indexed for semantic search and retrieval.
- **Multi-Agent Workflow:** Specialized agents collaborate to generate:
    - 📄 **Summary:** Concise overview of the document.
    - 📝 **Notes:** Bullet-point revision notes.
    - ❓ **MCQs:** Multiple-choice questions for self-assessment.
    - 💡 **Simplified Explanation:** Easy-to-understand explanations of complex topics.
    - 📺 **YouTube Resources:** Curated educational videos for each topic.
    - 🌐 **Web Resources:** High-quality web articles and references.
    - 🧩 **Comprehensive Compilation:** All outputs are combined into a structured markdown report.
- **Downloadable Output:** The final study guide can be downloaded as a markdown or PDF file for offline use.
- **Agentic RAG (Retrieval-Augmented Generation):**
    - **Smart Q&A:** Ask questions about your document. If the answer is not found but is related, the agent fetches information from the web.
    - **Context Awareness:** If the question is unrelated to the document, the agent politely declines to answer.
    - **Web Augmentation:** Even when the answer is found, the agent can supplement it with additional web resources.

---

## 🛠️ Technologies Used

- **CrewAI:** For orchestrating multiple specialized agents and tasks.
- **LlamaIndex:** For document ingestion, chunking, and vector-based retrieval.
- **LangChain:** For tool integration, prompt engineering, and agentic RAG.
- **Google Gemini & Mistral LLMs:** For high-quality language understanding and generation.
- **YouTube & Web Search APIs:** For fetching relevant external resources.
- **WeasyPrint:** For converting markdown study guides to PDF.

---

## 🧑‍💻 How It Works

1. **Document Ingestion:** Upload your PDF. The system splits it into manageable chunks and creates a semantic index.
2. **Agent Collaboration:** Each agent performs its specialized task (summarization, note-making, MCQ generation, etc.).
3. **Compilation:** Outputs from all agents are merged into a single, well-structured markdown document.
4. **Download:** Export your study guide as markdown or PDF.
5. **Interactive Q&A:** Use the Agentic RAG interface to ask follow-up questions, get explanations, or explore related topics with web-augmented answers.

---

## 🎯 Why Use This Project?

- **Saves Time:** Instantly generate summaries, notes, and practice questions from large PDFs.
- **Enhances Understanding:** Get simplified explanations and curated video/web resources.
- **Interactive Learning:** Ask questions and get context-aware, up-to-date answers.
- **All-in-One Solution:** Everything you need for exam prep in one workflow.
-  **Understand better** with simplified explanations
-  **Ask smart questions** and get contextual answers
-  **Study efficiently** with personalized summaries, MCQs, and curated resources

---

## 💡 Example Use Cases

- **Exam Revision:** Quickly turn a textbook or lecture PDF into a personalized study guide.
- **Self-Assessment:** Practice with automatically generated MCQs.
- **Deep Dives:** Use the Q&A agent to clarify doubts or explore related concepts.
- **Resource Discovery:** Find the best videos and articles for each topic in your syllabus.

---

## 📦 Project Workflow Overview

1. **Upload PDF →**  
2. **Agents Generate Outputs (Summary, Notes, MCQs, Explanations, Resources) →**  
3. **Compile to Markdown →**  
4. **Download as PDF/Markdown →**  
5. **Interactive Q&A with Agentic RAG**

---

## 📝 Next Steps

- Try uploading your own PDF and see the multi-agent workflow in action!
- Use the interactive chat to ask questions about your document or related topics.
- Download your personalized study guide for offline revision.

---

> **Empowering students to study smarter, not harder!**