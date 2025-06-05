# MBA Data Science Major Project Repo

# 🤖 AI Governance Workbench

## Overview

The **AI Governance Workbench** is a comprehensive Streamlit application designed to explore and facilitate various aspects of Artificial Intelligence (AI) governance. This tool provides interactive dashboards, data visualization capabilities, document summarization, compliance auditing, and an AI chat assistant powered by Google Gemini, including Retrieval-Augmented Generation (RAG) from uploaded documents.

This project aims to offer a practical, hands-on environment for understanding the complexities of responsible AI development and deployment, emphasizing transparency, fairness, accountability, and security in AI systems.

## ✨ Features

* **🌐 Governance Dashboard:** A simulated overview of AI deployment, compliance, and risk trends with interactive charts.
* **📊 Data Visualizer:** Upload and explore your own structured datasets (.csv, .json, .xlsx) or utilize pre-loaded samples. Generate dynamic charts (bar, line, scatter, pie, box, violin, histogram, heatmap, geographic maps) and get immediate data summaries.
* **📄 Document Summarizer:** Upload policy documents (.pdf, .txt, .docx) to receive AI-generated summaries. Also features interactive flashcards for key concepts.
* **⚖️ Compliance Audit (Prototype):** A rule-based engine to check policy documents for the presence of key AI governance indicators, generating a downloadable markdown report.
* **💬 AI Chat Assistant (with RAG):** An interactive chatbot for general AI governance inquiries. When a document is loaded in the Summarizer, the chatbot can answer questions based on its content using Retrieval-Augmented Generation.

## 🚀 How to Run Locally

Follow these steps to get the AI Governance Workbench running on your local machine:

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Sumit-SC/Project.git
    cd Project
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    ```

3.  **Activate the Virtual Environment:**
    * **Windows:**
        ```bash
        .\venv\Scripts\activate
        ```
    * **macOS/Linux:**
        ```bash
        source venv/bin/activate
        ```

4.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *(If `requirements.txt` is not provided, you'll need to install the dependencies manually. Key ones include `streamlit`, `pandas`, `PyPDF2`, `python-docx`, `google-generativeai`, `plotly`, `numpy`, `faiss-cpu` (or `faiss-gpu`), `wordcloud`, `matplotlib`, `openpxl`.)*

5.  **Configure Google Gemini API Key:**
    * Create a `.streamlit` folder in the root of your project directory if it doesn't exist.
    * Inside `.streamlit`, create a file named `secrets.toml`.
    * Add your Google Gemini API key to this file:
        ```toml
        GOOGLE_API_KEY="YOUR_GEMINI_API_KEY_HERE"
        ```
    * You can obtain a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).

6.  **Run the Streamlit App:**
    ```bash
    streamlit run app.py
    ```
    (Assuming your main Streamlit script is named `app.py`. If it's `main.py` or similar, adjust the command.)

    The application should open in your default web browser.

## ☁️ Deploy to Streamlit Cloud

You can easily deploy this application to [Streamlit Community Cloud](https://streamlit.io/cloud).

1.  **Fork this repository** to your own GitHub account.
2.  Ensure your `secrets.toml` file is correctly set up in the `.streamlit` directory within your forked repository. **Note: For security, never commit your actual API key directly to a public repository.** Streamlit Cloud provides a secure way to add secrets during deployment.
3.  Go to [Streamlit Community Cloud](https://streamlit.io/cloud) and sign in.
4.  Click on "New app" from your dashboard.
5.  Select your forked repository and the branch containing your `app.py` (or main) file.
6.  In the "Advanced settings" section, you'll find an option to add secrets. Paste your `GOOGLE_API_KEY` exactly as it is in your `secrets.toml` file.
7.  Click "Deploy!"

Alternatively, use this direct deployment link:

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/Sumit-SC/Project/main/streamlit_app.py)


## 📚 Usage Guide

Once the application is running, navigate through the different sections using the sidebar:

* **🌐 Governance Dashboard:** Explore the simulated KPIs and charts. Use the multi-select to customize which charts are displayed.
* **📊 Data Visualizer:**
    * **Upload Data:** Use the file uploader to load your own CSV, JSON, or XLSX files. PDF, TXT, and DOCX files will have their text extracted and summarized.
    * **Pre-loaded Data:** Select from a variety of sample datasets.
    * **KPIs:** Choose a numerical column to see its average, sum, min, max, and other statistics.
    * **Single Chart Builder:** Select a chart type and relevant columns to create a specific visualization.
    * **Multi-Chart Dashboard:** Build a custom dashboard by selecting multiple chart types and configuring their columns. Each chart comes with a brief interpretation.
* **📄 Document Summarizer:**
    * **Upload Document:** Provide a PDF, TXT, or DOCX file.
    * **Pre-loaded/Sample:** Use pre-loaded documents or the built-in sample text.
    * **Generate Summary:** Get an AI-powered concise summary.
    * **Generate Flashcards:** Create interactive Q&A flashcards for self-testing.
* **⚖️ Compliance Audit (Prototype):**
    * **Upload Document:** Upload a policy document for analysis.
    * **Pre-loaded:** Select a sample policy document.
    * **Run Audit:** The system will check for predefined governance indicators and provide a compliance score and snippets.
    * **Download Report:** Export the audit findings in a markdown file.
* **💬 AI Chat Assistant:**
    * Type your questions about AI governance.
    * If a document was loaded in the Summarizer, the assistant will use its content (RAG) to provide more informed answers.

## 🛠️ Technologies Used

* **Frontend/App Framework:** [Streamlit](https://streamlit.io/)
* **AI Models:** [Google Gemini API](https://ai.google.dev/models/gemini) (`gemini-1.5-flash-latest`, `embedding-001`)
* **Data Manipulation:** [Pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/)
* **Plotting/Visualization:** [Plotly Express](https://plotly.com/python/plotly-express/), [Plotly Graph Objects](https://plotly.com/python/graph-objects/), [Matplotlib](https://matplotlib.org/)
* **Text Processing:** [PyPDF2](https://pypi.org/project/PyPDF2/), [python-docx](https://python-docx.readthedocs.io/en/latest/)
* **Vector Search (RAG):** [FAISS](https://github.com/facebookresearch/faiss)
* **Text Analysis:** [WordCloud](https://pypi.org/project/wordcloud/)

## 🚧 Future Enhancements

* Integration with more external AI governance frameworks and standards.
* Advanced NLP features for deeper document analysis (e.g., entity extraction, sentiment analysis specific to governance risks).
* User authentication and multi-user data persistence.
* More sophisticated compliance rule engine with custom rule definitions.
* Expanded pre-loaded real-world datasets and case studies.
* Interactive tutorials and guided tours for first-time users.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Contributions

Feel free to open issues or submit pull requests if you have suggestions, bug reports, or want to contribute to the project.

For questions or feedback, please reach out via GitHub issues.
