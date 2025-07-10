
import warnings
from llama_index.core import (
    SimpleDirectoryReader, VectorStoreIndex, Settings,
    PromptTemplate
)
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.response_synthesizers import get_response_synthesizer, ResponseMode
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
warnings.filterwarnings('ignore')
import os
GOOGLE_API_KEY = "AIzaSyADmKoZt0vreo7lR73dmPqbYaaTFd0q_ZI"
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
from crewai import Agent, Task, Crew ,LLM

vector_index = None
query_engine = None
from langchain_google_genai import ChatGoogleGenerativeAI

llm_1 = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0,
    max_tokens=None,
    timeout=None,
    api_key="AIzaSyADmKoZt0vreo7lR73dmPqbYaaTFd0q_ZI",
    max_retries=2,
)
llm = LLM(
    model="gemini/gemini-2.5-flash",
    temperature=0.7,
    api_key="AIzaSyADmKoZt0vreo7lR73dmPqbYaaTFd0q_ZI", #
    provider="gemini" # 
)
# Build index once for the PDF
def load_and_index(input_pdf_path: str):
    global vector_index
    global query_engine
    reader = SimpleDirectoryReader(input_files=[input_pdf_path])
    documents = reader.load_data()
    print(f"Loaded {len(documents)} document(s).")

    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)

    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        embed_batch_size=200,
        api_key="AIzaSyADmKoZt0vreo7lR73dmPqbYaaTFd0q_ZI"
    )
    Settings.llm = llm_1
    Settings.embed_model = embed_model
    vector_index = VectorStoreIndex(nodes)
    print("Indexing complete.")
def run_prompted_rag(template_str: str, user_query: str):
    global vector_index
    global query_engine
    template = PromptTemplate(template_str)
    synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        text_qa_template=template
    )
    retriever = vector_index.as_retriever(similarity_top_k=4)
    query_engine = RetrieverQueryEngine(retriever=retriever, response_synthesizer=synthesizer)
    response = query_engine.query(user_query)
    return str(response)
SUMMARY_PROMPT = """You are an academic summarization expert. Summarize this content:
{context_str}
"""
SUMMARY_PROMPT = """You are an academic summarization expert. Summarize this content:
{context_str}
"""

NOTES_PROMPT = """Create concise bullet-point notes from this content:
{context_str}
"""

MCQ_PROMPT = """Generate 10-12 multiple choice questions based on the following:
{context_str}
"""

EXPLAIN_PROMPT = """Explain the following content in simple terms, like for a high school student:
{context_str}
"""
from crewai.tools import tool
@tool("Summarizer")
def summarizer_tool(query="Summarize"):
    """
    Summarizes PDF content using context-aware RAG
    """
    return run_prompted_rag(SUMMARY_PROMPT, query)

@tool("NoteMaker")
def notes_tool(query="Provide Notes in Bullet points"):
    """
    Creates bullet-point study notes
    """
    return run_prompted_rag(NOTES_PROMPT, query)

@tool("MCQGenerator")
def mcq_tool(query="Give MCQs"):
    """
    Creates MCQs from PDF
    """
    return run_prompted_rag(MCQ_PROMPT, query)

@tool("Simplifier")
def explanation_tool(query="Explain in simple terms"):
    """
    Explains the content in simple terms
    """
    return run_prompted_rag(EXPLAIN_PROMPT, query)
import os
os.environ["SERPER_API_KEY"] = "c363c1a26cc95ffca7bdcb87ce3082bec12f4f27"
from langchain_community.utilities import GoogleSerperAPIWrapper
# from crewai.tools import BaseTool
from crewai.tools import tool
@tool("MyWebSearchTool")
def web_search_tool(query=None):
    """
    Searches the web using Serper.dev and returns links to high-quality resources/articles for each topic.
    """
    search = GoogleSerperAPIWrapper()

    # Step 1: Get topics from the PDF
    ans_1 = query_engine.query(
        "Give all the topics which this document is based on as a Python list of strings, each item separated by a comma."
    )

    # Step 2: Extract topics from the response
    topics = re.findall(r"'(.*?)'", str(ans_1))

    results = []

    # Step 3: Search each topic and collect top 3 links
    for topic in topics:
        try:
            search_results = search.results(topic)
            links = []
            for result in search_results.get("organic", []):
                if "link" in result:
                    links.append(result["link"])
                if len(links) >= 3:
                    break
            if links:
                formatted = "\n".join(f"- {url}" for url in links)
                results.append(f"🌐 **{topic}**\n{formatted}\n")
        except Exception as e:
            results.append(f"🌐 **{topic}**\nError fetching results: {str(e)}\n")

    return "\n".join(results)

from langchain_community.tools import YouTubeSearchTool
YT = YouTubeSearchTool()
from crewai.tools import tool
# ans_1 = query_engine.query(
#     "Give all the topics which this document is based on as a Python list of strings, each item separated by a comma."
# )
import re
# from langchain.tools import tool

@tool("YTVideoSearch")
def youtube_search(query=None):
    """
    Searches YouTube for educational videos on the main topics in the document.
    Returns a dictionary with topics and associated video links.
    """
    from langchain_community.tools import YouTubeSearchTool

    YT = YouTubeSearchTool()

    # Step 1: Extract topics from the document
    ans_1 = query_engine.query(
        "Give all the topics which this document is based on as a Python list of strings, each item separated by a comma."
    )

    # Step 2: Parse topics from the response (if model returns Python-style string list)
    topics = re.findall(r"'(.*?)'", str(ans_1))  # fallback for string like: ['A' 'B']

    results = []

    # Step 3: Run YouTube search for each topic and collect results
    for topic in topics:
        video_result = YT.run(topic)
        results.append(f"🔎 **{topic}**\n{video_result}\n")

    # Step 4: Join all results into one string
    return "\n".join(results)
@tool("DownloadPDF")
def download_pdf_tool():
    """Downloads final_output.pdf in Colab."""
    try:
        from google.colab import files
        files.download("final_output.pdf")
        return "✅ Download triggered"
    except ImportError:
        return "⚠️ Download only supported in Colab."
    
summarizer_agent = Agent(
    role="Summarization Expert",
    goal="Summarize academic PDFs efficiently",
    backstory="You specialize in extracting summaries from technical content.",
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[summarizer_tool]

)

notes_agent = Agent(
    role="Note Making Assistant",
    goal="Generate clear and concise notes for revision",
    backstory="You turn complex PDFs into easy notes for students.",
   llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[notes_tool],
)

mcq_agent = Agent(
    role="Question Generator",
    goal="Create useful exam-oriented MCQs",
    backstory="You understand how to turn academic content into testable questions.",
    tools=[mcq_tool],
    llm= llm,
    allow_delegation=False,
    verbose=True,

)
download_agent = Agent(
    role="Download Manager",
    goal="Download the final_output.pdf file to the user's device",
    backstory="You are responsible for handling the download of files. You don't use any language model — just ensure the user gets the file.",
    tools=[download_pdf_tool],
    llm=llm,
    allow_delegation=False,
    verbose=True
)
explainer_agent = Agent(
    role="Simplification Agent",
    goal="Simplify academic content for better understanding",
    backstory="You are great at explaining complex topics in simple language.",
    tools=[explanation_tool],
    allow_delegation=False,
    max_iter=1,
    verbose=True,
  llm=llm,
)
composer_Agent = Agent(
    role="Final Composer",
    goal=(
        "Take the outputs from the summary, notes, MCQs, simplification, YouTube, and web search agents, "
        "and combine them into a single structured markdown report. Do not call any tools."
    ),
    backstory=(
        "You are skilled at organizing and formatting diverse content into a clean, readable study guide. "
        "You will use the provided outputs and combine them into a single markdown document."
    ),
    allow_delegation=False,
    verbose=True,
  llm=llm,
)

web_research_agent = Agent(
    role="Web Research Assistant",
    goal="Search the internet to gather supplementary information related to the academic topic.",
    backstory=(
        "You are a digital research assistant with access to real-time web tools. "
        "Your goal is to enhance academic topics by finding relevant articles, blogs, documentation, and real-world examples "
        "to give students broader context and deeper understanding."
    ),
    llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[web_search_tool]
)
youtube_finder_agent = Agent(
    role="YouTube Search Agent",
    goal="Find the best educational videos related to the given topic.",
    backstory=(
        "You help students by finding relevant and high-quality educational videos on YouTube. "
        "You return short descriptions along with links that will help explain the topic in audio-visual form."
    ),
   llm=llm,
    allow_delegation=False,
    verbose=True,
    tools=[youtube_search]
)

summary_task = Task(
    description="Summarize the PDF content for the user",
    expected_output="A clear and concise summary.",
    agent=summarizer_agent
    ,tools=[summarizer_tool]
)

notes_task = Task(
    description="Create bullet-point notes based on the content",
    expected_output="Concise revision notes",
    agent=notes_agent,
     tools=[notes_tool],
)
download_task = Task(
    description="Download the generated final_output.pdf file to the local system or browser via Colab.",
    expected_output="Trigger download of final_output.pdf",
    agent=download_agent
)
compose_task = Task(
    description=(
        "Take the pre-generated outputs from the summary, notes, MCQs, explanation, YouTube search, and web search, "
        "and combine them into a single clean markdown document. Do not regenerate any of the content — assume it is already available. "
        "Use clear section headers for each type of content."
    ),
    expected_output=(
        "A markdown document with sections:\n"
        "1. 📄 Summary\n"
        "2. 📝 Notes\n"
        "3. ❓ MCQs\n"
        "4. 💡 Explanation\n"
        "5. 📺 YouTube Resources\n"
        "6. 🌐 Web Articles\n"
    ),
    agent=composer_Agent,
    output_file="final_output.md"
)

mcq_task = Task(
    description="Generate multiple choice questions based on the document",
    expected_output="10-12 MCQs",
    agent=mcq_agent,
    tools=[mcq_tool]
)

explanation_task = Task(
    description="Simplify the PDF topic so it's easier to understand",
    expected_output="Simplified explanation",
    agent=explainer_agent,
    tools=[explanation_tool]
)
web_research_task = Task(
    description="Conduct a web search to find additional resources related to the topic",
    expected_output="List of 5-6 URLs from the web",
    agent=web_research_agent,
    tools=[web_search_tool]
)
youtube_search_task = Task(
    description="Search YouTube for relevant and high-quality educational videos on the given topic. Prioritize tutorial or lecture-style content.",
    expected_output="Top 5-6 video titles with their YouTube links and a short description of each.",
    agent=youtube_finder_agent,
    tools=[youtube_search]
)
crew = Crew(
    tasks=[summary_task, notes_task, mcq_task,explanation_task,youtube_search_task,web_research_task,compose_task],
    agents=[summarizer_agent, notes_agent, mcq_agent,explainer_agent,youtube_finder_agent,web_research_agent,composer_Agent],
    llm=llm,
    verbose=False,
)
def run_pdf_pipeline(pdf_path, user_query):
    print(f"Processing PDF: {pdf_path}",type(pdf_path))
    load_and_index(pdf_path)
    return crew.kickoff()
# result = run_pdf_pipeline("Agile.pdf", "Help me study this for my exam")
# from IPython.display import Markdown
# Extract markdown content from between the code blocks
# markdown_content1 = result.raw.split('```markdown\n')[0].split('\n```')[0]
# Markdown(markdown_content1)


# !pip install llama-index-llms-langchain llama_index
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from langchain_core.tools import Tool
from langchain_core.tools import tool
def return_query_engine(pdf_path):
 reader = SimpleDirectoryReader(input_files=[pdf_path])
 documents = reader.load_data()
 print(f"Loaded {len(documents)} document(s).")

 # Split into chunks
 splitter = SentenceSplitter(chunk_size=1024)
 nodes = splitter.get_nodes_from_documents(documents)
 # you can change the batch size , model_name to different varieties that google embeddings provide ,
 # go to https://ai.google.dev/gemini-api/docs/embeddings
 embed_model = GoogleGenAIEmbedding(
 model_name="text-embedding-004",
 embed_batch_size=200,
 api_key="AIzaSyADmKoZt0vreo7lR73dmPqbYaaTFd0q_ZI"
)
 # Set up LLM and embedding model
 Settings.llm = llm_1
 Settings.embed_model = embed_model
 # Create vector index
 vector_index = VectorStoreIndex(nodes)
 # Create query engine
 query_engine = vector_index.as_query_engine()
 return query_engine
import os
os.environ["SERPER_API_KEY"] = "c363c1a26cc95ffca7bdcb87ce3082bec12f4f27"
import json

def safe_str_to_dict(json_str: str) -> dict:
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"❌ JSON parsing error: {e}")
        return {}
# query_engine= return_query_engine(pdf_path="Agile.pdf")
@tool
def RUN_RAG(inputs: str) -> str:
    """
    Takes query and returns the RAG answer.
    """
    try:
        return str(query_engine.query(inputs))
    except Exception as e:
        return f"❌ Error querying PDF: {str(e)}"
@tool
def get_topic(input: str = "what is this document based on ?")->str:
  """
  takes the query and returns the topic on which this document is based on
  """
  return str(query_engine.query(input))
from langchain_community.utilities import GoogleSerperAPIWrapper
@tool
def search_web(inputs: str) -> str:
    """
    Searches the web using Serper.dev.
    """
    # inputs_dict=safe_str_to_dict(inputs)
    # query = inputs_dict.get("input")
    search = GoogleSerperAPIWrapper()
    return search.run(inputs)
tools = [RUN_RAG, search_web,get_topic]
from langchain.prompts import PromptTemplate

react_template = """
You are a helpful academic assistant designed to answer questions about a student's uploaded PDF document (e.g. textbook, notes, research paper).

You can use the following tools:
{tools}

You must use this format:

Question: the input question you must answer
Thought: you should always reason step by step before answering
Action: the action to take, must be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the complete, clear, and concise answer to the original question

Your reasoning flow should follow this logic:

1. First, try using the `RUN_RAG` tool to answer based on the uploaded document.
2. Now use the `get_topic` tool to know which topic the document is based on, dont put any input in this function , just run it.
3. If the document does not answer the question but the topic is still relevant (e.g., same subject), use the `search_web` tool to look it up.
4. Now you know the topic of the document, If the question is completely unrelated to the topic of the document, respond with:
   "I'm sorry, this question is unrelated to the uploaded material and I can't answer it."
5. If the document answers the question fully,  use `search_web` optionally to provide additional learning resources or some more things related to that answer/topic.
Always reflect before taking action. Be sure to cite observations from tools clearly in your final answer.

Begin!

Question: {input}
Thought:{agent_scratchpad}
"""


prompt = PromptTemplate(
    template=react_template,
    input_variables=["tools", "tool_names", "input", "agent_scratchpad"]
)
from langchain.agents import create_react_agent, AgentExecutor

agent = create_react_agent(llm=llm_1, tools=tools, prompt=prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    return_intermediate_steps=False
)


# === Streamlit Integration ===

# import streamlit as st
# import tempfile
# import os


# # Title and layout
# st.set_page_config(page_title="StudyForge", layout="wide")
# st.title("📚 StudyForge: Multi-Agent Exam Assistant")

# # PDF Upload Section
# st.sidebar.header("📄 Upload your Study PDF")
# uploaded_pdf = st.sidebar.file_uploader("Choose a PDF file", type=["pdf"])

# # Session state to persist query_engine
# if "query_engine" not in st.session_state:
#     st.session_state.query_engine = None

# if uploaded_pdf:
#     with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
#         tmp_file.write(uploaded_pdf.read())
#         tmp_path = tmp_file.name

#     st.success("✅ PDF uploaded successfully!")

#     with st.spinner("🧠 Indexing your PDF and setting up Agentic RAG..."):
#         # Your function to return query_engine (must be defined in your imports)
#         st.session_state.query_engine = return_query_engine(pdf_path=tmp_path)
#         query_engine=st.session_state.query_engine
#     st.success("📘 Agentic Q&A Engine Ready! Ask your doubts below.")

# # Follow-up Q&A Section
# st.header("💬 Ask Questions (Agentic RAG)")

# user_question = st.text_input("Type your question here...")

# if user_question:
#     if not uploaded_pdf or st.session_state.query_engine is None:
#         st.warning("Please upload a PDF first to enable Q&A.")
#     else:
#         with st.spinner("🤖 Thinking..."):
#             response = agent_executor.invoke({
#                 "input": user_question
#             })
#             st.markdown("**Answer:**")
#             st.write(response.get("output", "⚠️ No response generated."))

import streamlit as st
import tempfile
import os
import io
import sys
import re
from contextlib import redirect_stdout

# Title and layout
st.set_page_config(page_title="StudyForge: Agentic RAG", layout="wide")
st.title("📚 StudyForge: Agentic Q&A Assistant")

# Session state to persist query_engine
if "query_engine" not in st.session_state:
    st.session_state.query_engine = None

# Upload PDF
st.sidebar.header("📄 Upload your Study PDF")
uploaded_pdf = st.sidebar.file_uploader("Upload your study material", type=["pdf"])

if uploaded_pdf:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_pdf.read())
        tmp_path = tmp_file.name

    st.success("✅ PDF uploaded successfully!")

    with st.spinner("🔍 Indexing your PDF..."):
        st.session_state.query_engine = return_query_engine(pdf_path=tmp_path)
        query_engine = st.session_state.query_engine

    st.success("✅ Agentic Q&A Engine Ready!")

# Agentic Q&A
st.header("💬 Ask Questions About the Document")

user_question = st.text_input("❓ Your question:")

if user_question:
    if not uploaded_pdf or st.session_state.query_engine is None:
        st.warning("Please upload a PDF first.")
    else:
        with st.spinner("🤖 Thinking..."):
            log_buffer = io.StringIO()
            with redirect_stdout(log_buffer):
                response = agent_executor.invoke({"input": user_question})

            # Clean ANSI escape characters
            ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
            logs = ansi_escape.sub('', log_buffer.getvalue())

        st.markdown("### ✅ **Final Answer**")
        st.write(response.get("output", "⚠️ No response generated."))

        st.markdown("### 🧠 Agent Thought Process")
        with st.expander("📜 Show Agent Logs"):
            st.text(logs)
