import streamlit as st
import pandas as pd
import PyPDF2
import matplotlib.pyplot as plt
import google.generativeai as genai
from io import StringIO

# --- Page Configuration (Always at the top) ---
st.set_page_config(
    page_title="AI Governance Analytics & LLM Interface",
    page_icon="🤖",
    layout="wide"
)

# --- Configure Google Gemini API Key securely ---
try:
    gemini_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    gemini_model = genai.GenerativeModel('gemini-pro')
except KeyError:
    st.error("Google API Key not found in Streamlit secrets.toml. Please add it as `GOOGLE_API_KEY`.")
    st.stop()
except Exception as e:
    st.error(f"Could not initialize Gemini API. Error: {e}")
    st.stop()

# --- Sidebar Options ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Choose Functionality:", [
    "📄 Document Summarizer",
    "📊 Data Visualizer",
    "🤖 AI Chat Assistant",
    "🌐 Governance Dashboard"
])

st.title("AI Governance Analytics & LLM Interface")
st.markdown("---") # Horizontal line for visual separation

# --- Dummy Data for Summarizer (for quick demo) ---
sample_policy_text = """
## AI Governance Framework: Principles and Practices

This document outlines the foundational principles and practical guidelines for the responsible development, deployment, and oversight of Artificial Intelligence (AI) systems within our organization. Our commitment to ethical AI is rooted in enhancing human well-being, fostering trust, and ensuring accountability across all AI initiatives.

**1. Fairness and Non-discrimination:**
AI systems must be designed and deployed to minimize bias and promote equitable outcomes. This includes rigorous testing for disparate impact across various demographic groups and continuous monitoring of model performance to detect and mitigate bias. Data used for training must be diverse and representative, and data collection practices must be ethical.

**2. Transparency and Explainability:**
The decision-making processes of AI systems should be understandable to stakeholders. We strive for sufficient transparency regarding the data, algorithms, and models used, appropriate to the context and potential impact of the AI. Where possible, explainable AI (XAI) techniques will be employed to provide insights into how AI reaches its conclusions.

**3. Accountability and Human Oversight:**
Clear lines of responsibility must be established for the development, deployment, and operation of AI systems. Humans must retain ultimate control and oversight, especially in high-stakes decisions. Mechanisms for appeal, redress, and human intervention will be integrated into AI-driven processes.

**4. Privacy and Security:**
Robust data privacy and security measures are paramount. AI systems will adhere to all relevant data protection regulations (e.g., GDPR, CCPA). Data minimization principles will be applied, and strong encryption and access controls will protect sensitive information. Regular security audits will be conducted.

**5. Robustness and Safety:**
AI systems must be resilient to errors, manipulation, and unexpected inputs. They should be designed to operate reliably and safely, even in unforeseen circumstances. Continuous testing, validation, and monitoring are essential to ensure the stability and integrity of AI models.

**6. Societal and Environmental Impact:**
We commit to assessing and mitigating the broader societal and environmental impacts of our AI systems. This includes considering implications for employment, social equity, and energy consumption. AI should be developed to contribute positively to society.

**Implementation Guidelines:**
* **Risk Assessment:** Conduct thorough AI risk assessments at each stage of the lifecycle.
* **Ethical Review Boards:** Establish multi-disciplinary ethical review boards for high-impact AI projects.
* **Training and Education:** Provide ongoing training for employees on AI ethics and responsible practices.
* **Monitoring and Auditing:** Implement continuous monitoring and regular independent audits of AI systems.
* **Stakeholder Engagement:** Engage with internal and external stakeholders to gather feedback and address concerns.

This framework is a living document and will be periodically reviewed and updated to reflect advancements in AI technology and evolving societal norms.
"""

# --- Dummy Data for Data Visualizer (for quick demo) ---
dummy_df = pd.DataFrame({
    'Year': [2020, 2021, 2022, 2023, 2024],
    'Compliance Score': [75, 80, 82, 85, 88],
    'Bias Detections': [12, 15, 10, 8, 5],
    'Domain': ['Healthcare', 'Finance', 'Education', 'Retail', 'Healthcare'],
    'Risk Level': ['Medium', 'High', 'Medium', 'Low', 'Medium']
})


# --- Summarizer Logic ---
if app_mode == "📄 Document Summarizer":
    st.header("Upload & Summarize PDF")
    st.info("Upload a PDF document to get an AI-generated summary using Google Gemini. Alternatively, use the sample policy text below for quick testing. "
            "For real-world examples, explore reports from institutions like the BIS or RBI linked in the sidebar.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("External Resources for Summarizer:")
    st.sidebar.markdown("- **AI Governance Documents Data (Kaggle):** [https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data](https://www.kaggle.com/datasets/umerhaddii/ai-governance-documents-data)")
    st.sidebar.markdown("- **Awesome AI Regulation (GitHub):** [https://github.com/ethicalml/awesome-artificial-intelligence-regulation](https://github.com/ethicalml/awesome-artificial-intelligence-regulation) (Find links to official PDF reports here)")
    st.sidebar.markdown("- **IFC Report on AI in Central Banks (BIS):** [https://www.bis.org/ifc/publ/ifc_report_18.pdf](https://www.bis.org/ifc/publ/ifc_report_18.pdf)")
    st.sidebar.markdown("- **RBI's FREE-AI Framework (IndiaAI Portal):** [https://indiaai.gov.in/article/rbi-s-framework-for-responsible-and-ethical-enablement-towards-ethical-ai-in-finance](https://indiaai.gov.in/article/rbi-s-framework-for-responsible-and-ethical-enablement-towards-ethical-ai-in-finance)")
    st.sidebar.markdown("---")

    uploaded_pdf = st.file_uploader("Upload PDF Document", type=["pdf"])
    
    selected_text = ""
    if uploaded_pdf:
        try:
            reader = PyPDF2.PdfReader(uploaded_pdf)
            for page in reader.pages:
                selected_text += page.extract_text()
            if not selected_text.strip():
                st.warning("Could not extract text from the uploaded PDF. It might be an image-based PDF or encrypted. Please try the sample text.")
                selected_text = ""
            else:
                st.subheader("Extracted Text Preview (from PDF)")
                st.write(selected_text[:1000] + ("..." if len(selected_text) > 1000 else ""))
        except PyPDF2.errors.PdfReadError:
            st.error("Invalid PDF file. Please upload a valid, unencrypted PDF.")
            selected_text = ""
        except Exception as e:
            st.error(f"An unexpected error occurred during PDF processing: {e}")
            selected_text = ""
    
    use_sample = st.checkbox("Use Sample AI Governance Policy Text for summarization", value=not bool(uploaded_pdf))

    if use_sample and not uploaded_pdf:
        selected_text = sample_policy_text
        st.subheader("Sample Text Preview (AI Governance Policy)")
        st.write(selected_text[:1000] + ("..." if len(selected_text) > 1000 else ""))
    
    if selected_text.strip():
        if st.button("Generate Summary"):
            with st.spinner("Summarizing your document with Gemini... This may take a moment."):
                try:
                    response = gemini_model.generate_content(
                        f"Summarize this policy document concisely, focusing on key takeaways and main points:\n\n{selected_text}"
                    )
                    summary = response.text
                    st.subheader("Summary")
                    st.success("Summary generated successfully!")
                    st.write(summary)
                except Exception as e:
                    st.error(f"Failed to generate summary with Gemini AI. Error: {e}")
                    st.warning("This could be due to text length, rate limits, or an issue with the Gemini API. Please try again or with a shorter document.")
    elif uploaded_pdf and not selected_text.strip():
        st.info("Please upload a PDF from which text can be extracted, or check 'Use Sample Policy Text'.")
    elif not uploaded_pdf and not use_sample:
        st.info("Upload a PDF or check 'Use Sample Policy Text' to begin summarization.")


# --- CSV Upload & Visualization ---
elif app_mode == "📊 Data Visualizer":
    st.header("Data Upload and Basic Visuals")
    st.info("Upload a CSV file to visualize its data. Only numeric columns can be plotted. A sample dataset is available for quick testing.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("External Resources for Data Visualizer:")
    st.sidebar.markdown("- **AI in Mitigating Cybersecurity Risks in Government (Kaggle):** [https://www.kaggle.com/datasets/salman1541983/ai-in-mitigating-cybersecurity-risks-in-government/data](https://www.kaggle.com/datasets/salman1541983/ai-in-mitigating-cybersecurity-risks-in-government/data)")
    st.sidebar.markdown("- **AI-Enhanced Cybersecurity Events Dataset (Kaggle):** [https://www.kaggle.com/datasets/hassaneskikri/ai-enhanced-cybersecurity-events-dataset](https://www.kaggle.com/datasets/hassaneskikri/ai-enhanced-cybersecurity-events-dataset)")
    st.sidebar.markdown("---")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    
    df = None
    use_dummy_data = st.checkbox("Use Sample Governance Data", value=not bool(uploaded_file))

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.info("Using uploaded CSV data.")
        except Exception as e:
            st.error(f"Error reading CSV file: {e}. Please ensure it's a valid CSV.")
            df = None
    
    if df is None and use_dummy_data:
        df = dummy_df
        st.info("Using sample governance data for visualization.")

    if df is not None:
        st.subheader("Data Preview:")
        st.dataframe(df.head())

        if not df.empty:
            numeric_columns = df.select_dtypes(include=['number']).columns
            if not numeric_columns.empty:
                column_to_plot = st.selectbox("Select a numeric column for visualization", numeric_columns)
                
                fig, ax = plt.subplots(figsize=(10, 6))
                
                if df[column_to_plot].nunique() < 20 and df[column_to_plot].dtype in ['int64', 'float64']:
                    df[column_to_plot].value_counts().sort_index().plot(kind='bar', ax=ax, color='skyblue')
                    ax.set_title(f'Counts of {column_to_plot}')
                    ax.set_ylabel('Count')
                    plt.xticks(rotation=45, ha='right')
                else:
                    df[column_to_plot].hist(ax=ax, bins=20, color='lightgreen', edgecolor='black')
                    ax.set_title(f'Histogram of {column_to_plot}')
                    ax.set_ylabel('Frequency')
                ax.set_xlabel(column_to_plot)
                st.pyplot(fig)
                plt.close(fig)

            else:
                st.warning("No numeric columns found in the uploaded CSV (or sample data) for plotting.")
        else:
            st.warning("The uploaded CSV file (or sample data) is empty.")
    else:
        st.info("Upload a CSV file or check 'Use Sample Governance Data' to view visualizations.")

# --- AI Chat Assistant (LLM-Based Query) ---
elif app_mode == "🤖 AI Chat Assistant":
    st.header("AI-Powered Chat on Governance Reports")
    st.info("Ask general questions about AI governance, policies, or ethical considerations to Google Gemini.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("Relevant AI Governance Reports & Articles:")
    st.sidebar.markdown("- **Global Finance Magazine (AI in Banks):** [https://gfmag.com/banking/financial-institutions-double-down-on-ai-but-will-it-deliver/](https://gfmag.com/banking/financial-institutions-double-down-on-ai-but-will-it-deliver/)")
    st.sidebar.markdown("- **Moody's (BIS report on AI in Central Banks):** [https://www.moodys.com/web/en/us/insights/regulatory-news/bis-report-discusses-ai-governance-at-central-banks.html](https://www.moodys.com/web/en/us/insights/regulatory-news/bis-report-discusses-ai-governance-at-central-banks.html)")
    st.sidebar.markdown("---")

    if "gemini_messages" not in st.session_state:
        st.session_state.gemini_messages = []
        st.session_state.gemini_messages.append({"role": "model", "parts": ["Hello! How can I help you with AI governance today?"]})

    for message in st.session_state.gemini_messages:
        with st.chat_message("user" if message["role"] == "user" else "assistant"):
            st.markdown(" ".join(message["parts"]))

    prompt = st.chat_input("Your query:")
    if prompt:
        st.session_state.gemini_messages.append({"role": "user", "parts": [prompt]})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Generating response with Gemini..."):
            try:
                response = gemini_model.generate_content(
                    st.session_state.gemini_messages
                )
                answer = response.text

                st.session_state.gemini_messages.append({"role": "model", "parts": [answer]})
                with st.chat_message("assistant"):
                    st.markdown(answer)

            except Exception as e:
                st.error(f"Failed to get response from Gemini AI. Error: {e}")
                st.warning("This could be due to conversation context length, rate limits, or an issue with the Gemini API. Please try again.")

# --- AI Governance Dashboard View ---
elif app_mode == "🌐 Governance Dashboard":
    st.header("Simulated Dashboard: AI Deployment & Governance Trends")
    st.info("This dashboard displays simulated trends related to AI adoption and governance challenges across various domains. The data is pre-populated for demonstration and inspiration.")
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Dashboard Inspiration & Data Sources:")
    st.sidebar.markdown("- **Multivariate Dataset on IoT and AI (Kaggle):** [https://www.kaggle.com/datasets/ziya07/multivariate-dataset-on-iot-and-ai](https://www.kaggle.com/datasets/ziya07/multivariate-dataset-on-iot-and-ai) (Contains AI adoption scores)")
    st.sidebar.markdown("- Consider reports from **WEF, OECD, UNESCO, Partnership on AI** for real-world metrics and trends.")
    st.sidebar.markdown("---")

    years = list(range(2018, 2025))
    ai_adoption = [15, 25, 35, 45, 55, 65, 72] # Percentage of AI adoption
    bias_incidents = [1, 2, 4, 7, 9, 11, 12] # Number of reported incidents

    dashboard_df = pd.DataFrame({
        'Year': years,
        'AI Deployment Rate (%)': ai_adoption,
        'Reported Bias Incidents': bias_incidents
    }).set_index('Year')

    st.subheader("AI Adoption vs. Governance Challenges (2018–2024)")
    
    st.line_chart(dashboard_df[['AI Deployment Rate (%)', 'Reported Bias Incidents']])

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.bar(years, ai_adoption, color='skyblue', label="AI Deployment Rate (%)", width=0.4, align='center')
    ax1.set_xlabel("Year")
    ax1.set_ylabel("AI Deployment Rate (%)", color='skyblue')
    ax1.tick_params(axis='y', labelcolor='skyblue')

    ax2 = ax1.twinx()
    ax2.plot(years, bias_incidents, color='red', marker='o', linestyle='-', label="Reported Incidents")
    ax2.set_ylabel("Reported Incidents", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    fig.suptitle("AI Adoption vs. Governance Challenges (2018–2024)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left", bbox_to_anchor=(0.1, 0.9))
    
    st.pyplot(fig)
    plt.close(fig)

    st.caption("Note: This data is simulated and represents hypothetical trends in AI deployment and associated privacy/bias issues across an unspecified sector. Refer to external links for real-world datasets.")