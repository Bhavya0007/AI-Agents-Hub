# Spiritual AI Agent

This project is a **Spiritual AI Agent** built using [LangChain](https://python.langchain.com/), [Google Gemini](https://ai.google.dev/), and various web APIs. The agent provides empathetic, wise, and actionable spiritual guidance, including book recommendations, quotes, places, images, and videos tailored to the user's needs.

## Features

- **Intent Classification:** Understands the user's spiritual needs (e.g., emotional crisis, seeking knowledge, inspiration).
- **Book Recommendations:** Suggests relevant spiritual books using the Google Books API.
- **Spiritual Places:** Finds nearby temples, meditation centers, or yoga classes using Serper API.
- **Image Search:** Shows images of spiritual places.
- **YouTube Videos:** Recommends related talks, meditations, or discourses.
- **Inspirational Quotes:** Fetches motivational and spiritual quotes from ZenQuotes.
- **Web Search:** For additional spiritual resources.
- **Human Input Tool:** Asks clarifying questions when needed.
- **Summary Tool:** Gently summarizes the guidance, including links and resources.

## Setup

1. **Install Dependencies**

    ```bash
    pip install langchain-community langchain-google-genai youtube_search
    ```

2. **API Keys Required**
    - **Google AI API Key** (for Gemini)
    - **Serper API Key** (for web, places, and image search)

    The notebook will prompt you to enter these keys if not set in your environment.

3. **Run the Notebook**
    - Follow the cells in order, or use the provided agent executor to interact with the agent.

## Usage

- **Ask Spiritual Questions:**  
  The agent can help with finding peace, understanding spiritual concepts, recommending practices, or suggesting places and resources.
- **Example:**
  ```python
  response = agent_executor.invoke({
        "input": "Lately, I’ve been feeling lost and disconnected from my purpose. I want to find peace and reconnect with my spiritual side. Can you guide me?"
  })
  print(response["output"])
  ```

## Tools Integrated

- **Web_Search_Tool:** General spiritual web search.
- **Formatted_Places_Tool:** Human-readable list of spiritual places.
- **Formatted_Image_Tools:** Image links for places.
- **YouTube_Video_Search_Tool:** Spiritual video recommendations.
- **Spiritual_Books_Tool:** Book recommendations.
- **Spiritual_Quotes_Tool:** Inspirational quotes.
- **HumanInputTool:** Clarifies user intent or location.
- **Spiritual_Intent_Classifier:** Classifies user needs.
- **Summary_Tool:** Summarizes the response.

## Customization

- You can add or modify tools as needed.
- The prompt template can be adjusted for tone or flow.

## License

This project is for educational and research purposes. Please respect the terms of the APIs used.

---

*May your journey be peaceful and insightful!*