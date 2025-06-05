import streamlit as st
import pandas as pd
import PyPDF2
from docx import Document # For .docx support
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO
import numpy as np
import faiss # For vector similarity search
import time # For simulating loading times and potential delays
from wordcloud import WordCloud # For word cloud generation
import matplotlib.pyplot as plt # For displaying word cloud
import json # For JSON file handling


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="AI Governance Workbench - Developed by Sumit",
    page_icon="🤖",
    layout="wide",  # Use wide layout for better dashboarding
    initial_sidebar_state="expanded"
)

# --- Visible Page Title ---
st.title("AI Governance Workbench")
st.subheader("Developed by Sumit")

# Hide GitHub/Fork buttons and other Streamlit header items
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}   /* Hide the hamburger menu */
    footer {visibility: hidden;}      /* Hide footer */
    header {visibility: hidden;}      /* Hide the header (includes GitHub and Fork buttons) */
    </style>
"""

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# --- Configure Google Gemini API Key securely ---
gemini_model = None # Initialize to None
embedding_model = None # Initialize embedding model
try:
    gemini_api_key = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=gemini_api_key)
    # Using 'gemini-1.5-flash-latest' for text generation
    gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
    # Using 'embedding-001' for text embeddings (for RAG)
    embedding_model = genai.GenerativeModel('embedding-001')
except KeyError:
    st.error("Google API Key not found in Streamlit secrets.toml. Please add it as `GOOGLE_API_KEY`.")
    st.stop()
except Exception as e:
    st.error(f"Could not initialize Gemini API. Error: {e}")
    st.stop()

# --- Optional: Configure Mapbox token for Plotly density maps (for richer map tiles) ---
# You can get a free token from mapbox.com if desired.
try:
    px.set_mapbox_access_token(st.secrets["MAPBOX_TOKEN"])
except KeyError:
    st.warning("Mapbox token not found in secrets. Mapbox-powered maps may have limited detail.")
except Exception as e:
    st.warning(f"Could not set Mapbox access token: {e}")


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
* **Monitoring and Auditing:** Implement continuous monitoring and and regular independent audits of AI systems.
* **Stakeholder Engagement:** Engage with internal and external stakeholders to gather feedback and address concerns.

This framework is a living document and will be periodically reviewed and updated to reflect advancements in AI technology and evolving societal norms.
"""

# --- Dummy Data for Data Visualizer (for quick demo) ---
dummy_df = pd.DataFrame({
    'Year': [2020, 2021, 2022, 2023, 2024],
    'Compliance Score (%)': [75, 80, 82, 85, 88],
    'Bias Detections (Count)': [12, 15, 10, 8, 5],
    'Domain': ['Healthcare', 'Finance', 'Education', 'Retail', 'Healthcare'],
    'Risk Level': ['Medium', 'High', 'Medium', 'Low', 'Medium'],
    'AI Adoption Rate (%)': [30, 40, 55, 60, 75]
})

# Pre-loaded datasets (ensure these files exist in 'data/' or 'dummy_data/' folders)
preloaded_datasets = {
    "Sample Governance Data (Dummy)": "dummy_df", # Refers to the DataFrame defined above
    "AI Compliance Audit (Simulated Finance)": "data/ai_compliance_audit.csv",
    "Healthcare AI Ethical Concerns (Simulated)": "data/healthcare_ai_ethical_concerns.csv",
    "Adult Income Data (Bias Example - Real)": "data/adult.csv", # Download from Kaggle
    "Credit Card Fraud Detection (Finance - Real)": "data/creditcard.csv", # Large, might be slow
    "Cybersecurity Risks in Government (Real)": "data/cybersecurity_risks.csv",
    "E-commerce AI Recommendation Bias (Dummy)": "dummy_data/ecommerce_ai_bias.csv",
    "Government AI Adoption & Citizen Trust (Dummy)": "dummy_data/gov_ai_trust.csv",
    # --- Real PDF/DOCX files for the Summarizer/RAG/Audit ---
    "OECD AI Principles (Real PDF)": "data/oecd_ai_principles.pdf",
    "NIST AI Risk Management Framework (Real PDF)": "data/nist_ai_rmf.pdf",
    "EU Trustworthy AI Ethics Guidelines (Real PDF)": "data/eu_trustworthy_ai_ethics.pdf",
    "Sample Healthcare Report (Real DOCX)": "data/sample_report.docx", # Create this DOCX
    # --- NEW Real PDF/CSV files (refer to latest instructions to download/create) ---
    "EU AI Act (Official Text - Real PDF)": "data/eu_ai_act.pdf",
    "UK AI Regulation White Paper (Real PDF)": "data/uk_ai_white_paper.pdf",
    "UNESCO AI Ethics Recommendation (Real PDF)": "data/unesco_ai_ethics.pdf",
    "Deloitte AI Governance Roadmap (Real PDF)": "data/deloitte_ai_governance_roadmap.pdf",
    "Global AI Job Market Trends (Real CSV)": "data/ai_job_trends.csv",
    "AI Enhanced Cybersecurity Events (Real CSV)": "data/ai_cybersecurity_events.csv"
}

# --- Utility Functions for RAG ---

def get_text_chunks(text, chunk_size=1000, overlap_size=100):
    """Splits text into chunks of specified character length with overlap."""
    chunks = []
    if not text:
        return chunks

    start_index = 0
    while start_index < len(text):
        end_index = min(start_index + chunk_size, len(text))
        chunk = text[start_index:end_index]

        # Ensure we don't cut a word in half at the end of the chunk
        if end_index < len(text) and text[end_index].isalpha():
            next_space = text.find(' ', end_index)
            if next_space != -1:
                chunk = text[start_index:next_space]
                end_index = next_space
            else: # No space found, take rest of text as one chunk
                chunk = text[start_index:]
                end_index = len(text)

        chunks.append(chunk.strip())
        
        # Calculate next start index for overlap
        start_index = end_index - overlap_size
        if start_index < 0: # Ensure start_index doesn't go negative
            start_index = 0
        
        # If the last chunk was already added and the whole remaining text was added, break
        if end_index >= len(text):
            break

    return chunks

@st.cache_data(show_spinner="Generating embeddings for document...")
def generate_embeddings(text_chunks, model):
    """Generates embeddings for a list of text chunks."""
    embeddings = []
    # Gemini embedding API has a batch limit. Process in smaller batches if needed.
    # For this demo, assuming chunks are manageable.
    for i, chunk in enumerate(text_chunks):
        try:
            # Use 'embed_content' for individual chunks
            response = model.embed_content(model="embedding-001", content=chunk)
            embeddings.append(response['embedding'])
        except Exception as e:
            st.error(f"Error generating embedding for chunk {i}: {e}. Skipping chunk.")
            embeddings.append(None) # Append None if embedding fails for a chunk
            time.sleep(0.1) # Small delay on error to prevent hitting rate limits rapidly
    return np.array([e for e in embeddings if e is not None])

@st.cache_resource
def get_faiss_index(embeddings):
    """Creates and returns a FAISS index from embeddings."""
    if embeddings.size == 0:
        return None
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension) # L2 distance for similarity
    index.add(embeddings)
    return index

@st.cache_data(show_spinner="Retrieving relevant information...")
def retrieve_info(query, text_chunks, faiss_index, embedding_model, top_k=3):
    """Retrieves top_k most relevant chunks using FAISS."""
    if faiss_index is None:
        return []
    try:
        query_embedding_response = embedding_model.embed_content(model="embedding-001", content=query)
        query_embedding = np.array([query_embedding_response['embedding']])
        
        # Ensure query_embedding matches the index dimension
        if query_embedding.shape[1] != faiss_index.d:
            st.error(f"Query embedding dimension {query_embedding.shape[1]} does not match FAISS index dimension {faiss_index.d}.")
            return []

        D, I = faiss_index.search(query_embedding, top_k)
        retrieved_chunks = [text_chunks[idx] for idx in I[0]]
        return retrieved_chunks
    except Exception as e:
        st.error(f"Error during retrieval: {e}")
        return []

# --- Document Parsing Functions ---
def extract_text_from_pdf(uploaded_file):
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page_num in range(len(reader.pages)):
        text += reader.pages[page_num].extract_text() or ""
    return text

def extract_text_from_docx(uploaded_file):
    doc = Document(uploaded_file)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return "\n".join(text)

def extract_text_from_txt(uploaded_file):
    # Ensure file is read as UTF-8
    return uploaded_file.read().decode("utf-8")

# --- Ethical and Compliance Audit Rules (Prototype) ---
compliance_rules = {
    "Data Consent Statement": ["data consent", "informed consent", "consent for data usage"],
    "Bias Mitigation Strategy": ["bias mitigation", "fairness assessment", "de-biasing", "unbiased outcomes"],
    "Accountability Framework": ["accountability", "human oversight", "lines of responsibility", "redress mechanisms"],
    "Transparency/Explainability Mention": ["transparency", "explainability", "interpretability", "decision-making process"],
    "Security Measures": ["data security", "encryption", "access controls", "security audits"],
    "Risk Assessment Process": ["risk assessment", "impact assessment", "risk management framework"],
    "Stakeholder Engagement": ["stakeholder engagement", "public consultation", "feedback mechanisms"]
}

def conduct_compliance_audit(document_text):
    """
    Analyzes document text against predefined compliance rules.
    Returns a dictionary of rule_name: (found_status, found_snippet)
    """
    audit_results = {}
    doc_lower = document_text.lower()
    for rule, keywords in compliance_rules.items():
        found = False
        snippet = "Not found."
        for keyword in keywords:
            if keyword in doc_lower:
                found = True
                # Try to get a snippet around the keyword
                idx = doc_lower.find(keyword)
                start = max(0, idx - 50)
                end = min(len(doc_lower), idx + len(keyword) + 50)
                snippet = document_text[start:end].replace('\n', ' ')
                snippet = f"...{snippet.strip()}..."
                break
        audit_results[rule] = (found, snippet)
    return audit_results

# --- Sidebar Navigation ---
st.sidebar.title("AI Governance Workbench")
st.sidebar.markdown("Navigate through the tools:")

# Add an explicit label for the radio button, and hide it visually
app_mode = st.sidebar.radio(
    "Select App Mode", # Changed from "" to a meaningful label
    ["🌐 Governance Dashboard", "📊 Data Visualizer", "📄 Document Summarizer",
     "⚖️ Compliance Audit (Prototype)", "💬 AI Chat Assistant", "About"],
    label_visibility="hidden"
)

st.sidebar.markdown("---")
st.sidebar.info("Developed for exploring AI governance principles and tools.")

# --- Main Content Area based on app_mode ---

if app_mode == "About":
    st.header("About the AI Governance Workbench 🤖")
    st.markdown("---")

    st.markdown("""
    This application is designed to be a comprehensive platform for exploring various aspects of AI governance.
    It integrates tools for data analysis, document understanding, and interactive AI engagement,
    all under the umbrella of responsible AI development and deployment.

    **Key Features:**

    * **Governance Dashboard:** Provides a simulated overview of AI deployment, compliance, and risk trends.
    * **Data Visualizer:** Allows uploading and exploring real-world datasets related to AI's impact across different sectors (e-commerce, finance, health, government, etc.). Features interactive Plotly charts.
    * **Document Summarizer:** Helps in quickly understanding key policies, regulations, or research papers related to AI governance, supporting PDF, TXT, and DOCX formats.
    * **Compliance Audit (Prototype):** A rule-based engine to flag the presence of key AI governance elements in policy documents.
    * **AI Chat Assistant:** An interactive AI model for querying and discussing AI governance topics, enhanced with **Retrieval-Augmented Generation (RAG)** to answer questions directly from uploaded documents.

    **Purpose:**

    The primary goal is to provide a hands-on environment to understand the practical challenges and solutions in AI governance,
    emphasizing transparency, fairness, accountability, and security in AI systems.

    **Disclaimer:**
    All data used in this application, especially in the Governance Dashboard, is simulated or exemplary.
    Any real datasets integrated from public sources like Kaggle are for educational and demonstrative purposes only.
    Actual AI governance practices require careful consideration of proprietary data, regulatory frameworks, and expert human oversight.
    """)
    st.markdown("---")
    st.subheader("Technology Stack:")
    st.markdown("""
    * **Frontend:** Streamlit
    * **Backend/AI:** Google Gemini API (Gemini-1.5-Flash-latest for text, Embedding-001 for RAG)
    * **Data Handling:** Pandas, NumPy
    * **Document Parsing:** PyPDF2, python-docx
    * **Vector Search (for RAG):** FAISS
    * **Plotting:** Plotly Express, Plotly Graph Objects, Matplotlib
    * **Word Cloud:** WordCloud library
    """)

# --- AI Governance Dashboard View ---
elif app_mode == "🌐 Governance Dashboard":
    st.header("Comprehensive AI Governance Insights")
    st.markdown("---")

    st.info("This dashboard provides a simulated overview of AI deployment, compliance, and risk trends within an organization. It's designed to showcase key metrics and inspire data-driven governance decisions. Data is pre-populated for demonstration.")

    # --- Simulated Data for Dashboard ---
    years = list(range(2018, 2025))
    ai_adoption = [15, 25, 35, 45, 55, 65, 72] # Percentage of AI adoption
    bias_incidents = [1, 2, 4, 7, 9, 11, 12] # Number of reported incidents
    compliance_score = [70, 75, 78, 82, 85, 88, 90] # Percentage compliance score
    model_risk_index = [6.5, 7.0, 6.8, 6.2, 5.5, 5.0, 4.5] # Index from 1-10

    dashboard_data = {
        'Year': years,
        'AI Deployment Rate (%)': ai_adoption,
        'Reported Bias Incidents': bias_incidents,
        'Compliance Score (%)': compliance_score,
        'Model Risk Index': model_risk_index
    }
    dashboard_df = pd.DataFrame(dashboard_data).set_index('Year')

    # Simulated Incident Locations for Density Map (new dummy data for dashboard)
    incident_locations_data = {
        'Incident_ID': range(1, 31),
        'Latitude': np.random.uniform(30, 45, 30), # Latitude range for USA-ish area
        'Longitude': np.random.uniform(-120, -70, 30), # Longitude range for USA-ish area
        'Severity': np.random.randint(1, 6, 30), # 1 (low) to 5 (high)
        'Category': np.random.choice(['Bias', 'Security', 'Privacy', 'Performance', 'Fairness'], 30),
        'Year': np.random.choice(years, 30)
    }
    incident_locations_df = pd.DataFrame(incident_locations_data)

    # Simulated text for Word Cloud (new dummy data for dashboard)
    governance_keywords_text = """
    Ethics Transparency Accountability Fairness Bias Data Privacy Security Human Oversight Explainability Risk Management Compliance Policy Regulation Guidelines Trust Audit Impact Assessment Stakeholder Engagement Responsible AI AI Governance Principles Framework Safeguards Mitigation Development Deployment Monitoring Evaluation AI System Ethical AI Responsible AI Human-Centric AI
    Data Protection GDPR HIPAA CCPA Algorithmic Bias Discrimination Equity Inclusivity Audit Trails Explainable AI Interpretable AI Human-in-the-loop Model Monitoring Performance Drift Ethical Review Boards Societal Impact Environmental Impact Digital Divide Algorithmic Harms
    """

    st.subheader("Key Performance Indicators (KPIs)")

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Current AI Deployment Rate", value=f"{dashboard_df['AI Deployment Rate (%)'].iloc[-1]}%")
        with col2:
            st.metric(label="Total Reported Bias Incidents (2018-2024)", value=f"{dashboard_df['Reported Bias Incidents'].sum()}")
        with col3:
            st.metric(label="Latest Compliance Score", value=f"{dashboard_df['Compliance Score (%)'].iloc[-1]}%")
        with col4:
            st.metric(label="Current Model Risk Index", value=f"{dashboard_df['Model Risk Index'].iloc[-1]:.1f}")

    st.markdown("---")

    st.subheader("Interactive Governance Visualizations")

    chart_options = [
        "AI Deployment & Incidents Trend",
        "Compliance & Risk Trend",
        "Compliance Score Distribution (Histogram)",
        "Model Risk Index Distribution (Histogram)",
        "Key Governance Topics (Word Cloud)",
        "AI Incident Locations (Density Map)",
        "Dynamic Chart Builder" # Renamed to just "Dynamic Chart Builder"
    ]

    selected_charts = st.multiselect(
        "Select Charts to Display:",
        options=chart_options,
        default=["AI Deployment & Incidents Trend", "Compliance & Risk Trend"]
    )

    # --- Conditional Chart Rendering ---
    
    # Row for Trend Charts
    col_chart_trend1, col_chart_trend2 = st.columns(2)

    with col_chart_trend1:
        if "AI Deployment & Incidents Trend" in selected_charts:
            st.markdown("#### AI Deployment vs. Reported Bias Incidents")
            fig1 = px.line(dashboard_df, x=dashboard_df.index, y=['AI Deployment Rate (%)', 'Reported Bias Incidents'],
                           title='AI Deployment vs. Bias Incidents Trend',
                           labels={'value': 'Value', 'Year': 'Year', 'variable': 'Metric'})
            fig1.update_layout(hovermode="x unified")
            st.plotly_chart(fig1, use_container_width=True)
            st.caption("Observation: As AI deployment increases, so do reported bias incidents, highlighting the need for robust governance frameworks.")
    
    with col_chart_trend2:
        if "Compliance & Risk Trend" in selected_charts:
            st.markdown("#### Compliance Score vs. Model Risk Index")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=years, y=compliance_score, mode='lines+markers', name='Compliance Score (%)', yaxis='y1', line=dict(color='green')))
            fig2.add_trace(go.Scatter(x=years, y=model_risk_index, mode='lines+markers', name='Model Risk Index', yaxis='y2', line=dict(color='red', dash='dash')))

            fig2.update_layout(
                title_text="Compliance Score vs. Model Risk Index (2018–2024)",
                yaxis=dict(
                    title=dict(text="Compliance Score (%)", font=dict(color="green")), # Corrected syntax
                    tickfont=dict(color="green")
                ),
                yaxis2=dict(
                    title=dict(text="Model Risk Index (1-10)", font=dict(color="red")), # Corrected syntax
                    tickfont=dict(color="red"),
                    overlaying="y",
                    side="right"
                ),
                hovermode="x unified",
                legend=dict(x=0.01, y=0.99)
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.caption("Observation: Increasing compliance efforts seem to correlate with a decrease in the overall model risk index over time.")

    # Row for Histograms
    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        if "Compliance Score Distribution (Histogram)" in selected_charts:
            st.markdown("#### Distribution of Compliance Scores")
            fig_hist_compliance = px.histogram(dashboard_df, x='Compliance Score (%)', nbins=10,
                                               title='Frequency of Annual Compliance Scores',
                                               color_discrete_sequence=px.colors.qualitative.Plotly)
            st.plotly_chart(fig_hist_compliance, use_container_width=True)
            st.caption("Shows how frequently different compliance score ranges occurred.")

    with col_hist2:
        if "Model Risk Index Distribution (Histogram)" in selected_charts:
            st.markdown("#### Distribution of Model Risk Index")
            fig_hist_risk = px.histogram(dashboard_df, x='Model Risk Index', nbins=10,
                                         title='Frequency of Annual Model Risk Index',
                                         color_discrete_sequence=px.colors.qualitative.D3)
            st.plotly_chart(fig_hist_risk, use_container_width=True)
            st.caption("Shows the distribution of the calculated model risk index values.")

    # Row for Word Cloud and Density Map
    col_wordcloud, col_density_map = st.columns(2)

    with col_wordcloud:
        if "Key Governance Topics (Word Cloud)" in selected_charts:
            st.markdown("#### Dominant AI Governance Keywords")
            with st.spinner("Generating word cloud..."):
                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='viridis').generate(governance_keywords_text)
                plt.figure(figsize=(10, 5))
                plt.imshow(wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
                st.caption("Highlights frequently appearing terms in AI governance discussions.")
                plt.close() # Close plot to prevent display issues on subsequent runs

    with col_density_map:
        if "AI Incident Locations (Density Map)" in selected_charts:
            st.markdown("#### Density of Simulated AI Incidents by Location")
            with st.spinner("Generating density map..."):
                fig_density_map = px.density_mapbox(incident_locations_df, lat='Latitude', lon='Longitude', z='Severity', radius=10,
                                                    center=dict(lat=37.0902, lon=-95.7129), zoom=3,
                                                    mapbox_style="carto-positron",  # Try other styles like "open-street-map"
                                                    title='Simulated AI Incidents Density by Severity')
                st.plotly_chart(fig_density_map, use_container_width=True)
                st.caption("Visualizes areas with higher concentrations of simulated AI incidents based on severity.")

    # --- New Dynamic Chart Builder (Dashboard Data) ---
    if "Dynamic Chart Builder" in selected_charts:
        st.markdown("---")
        st.subheader("Build Your Own Chart (using simulated data or custom upload)")
        st.info("Select a data source and columns to create custom visualizations. The chart will update dynamically as you make selections.")

        # Data source selection for the dynamic chart builder on the dashboard
        data_source_option = st.radio(
            "Choose Data Source:",
            ("Simulated Dashboard Data", "Upload Your Own CSV"),
            key="dashboard_data_source_radio"
        )

        current_df_for_dynamic_chart = None
        if data_source_option == "Simulated Dashboard Data":
            current_df_for_dynamic_chart = dashboard_df.reset_index() # Reset index to make 'Year' a regular column
            st.markdown("Using internal simulated dashboard data.")
        else: # "Upload Your Own CSV"
            uploaded_dynamic_file = st.file_uploader("Upload CSV for Dynamic Chart", type=["csv"], key="dashboard_dynamic_uploader")
            if uploaded_dynamic_file is not None:
                try:
                    current_df_for_dynamic_chart = pd.read_csv(uploaded_dynamic_file)
                    st.success("Custom CSV data loaded for dynamic chart.")
                except Exception as e:
                    st.error(f"Error reading uploaded CSV: {e}")
                    current_df_for_dynamic_chart = None
            else:
                st.info("Upload a CSV file to use your own data for the dynamic chart builder.")

        if current_df_for_dynamic_chart is not None and not current_df_for_dynamic_chart.empty:
            
            all_cols_dyn = current_df_for_dynamic_chart.columns.tolist()
            numeric_cols_dyn = current_df_for_dynamic_chart.select_dtypes(include=['number']).columns.tolist()
            categorical_cols_dyn = current_df_for_dynamic_chart.select_dtypes(include=['object']).columns.tolist()
            
            col_dyn_chart_type, col_dyn_x, col_dyn_y = st.columns(3)

            with col_dyn_chart_type:
                dyn_chart_type = st.selectbox(
                    "Select Chart Type",
                    ["Bar Chart", "Line Chart", "Area Chart", "Scatter Plot", "Pie Chart", "Box Plot", "Violin Plot", "Histogram"],
                    key="dashboard_dyn_chart_type_select" # Added Histogram
                )
            
            with col_dyn_x:
                # X-axis options will vary based on chart type suitability
                if dyn_chart_type in ["Pie Chart"]:
                    dyn_x_axis_col = st.selectbox("Select Category/Names Column", all_cols_dyn, key="dashboard_dyn_x_pie")
                elif dyn_chart_type in ["Box Plot", "Violin Plot"]:
                    dyn_x_axis_col = st.selectbox("Select Grouping Column (Optional)", ["None"] + categorical_cols_dyn, key="dashboard_dyn_x_group")
                elif dyn_chart_type == "Histogram": # For histogram, X-axis can be numeric or categorical
                    dyn_x_axis_col = st.selectbox("Select X-axis Column", all_cols_dyn, key="dashboard_dyn_x_hist")
                else:
                    dyn_x_axis_col = st.selectbox("Select X-axis Column", all_cols_dyn, key="dashboard_dyn_x_other")

            with col_dyn_y:
                # Y-axis options will vary
                if dyn_chart_type in ["Pie Chart"]:
                    dyn_y_axis_col = st.selectbox("Select Value Column", numeric_cols_dyn, key="dashboard_dyn_y_pie")
                elif dyn_chart_type in ["Box Plot", "Violin Plot"]:
                    dyn_y_axis_col = st.selectbox("Select Value Column", numeric_cols_dyn, key="dashboard_dyn_y_value")
                elif dyn_chart_type == "Histogram": # Histogram Y-axis is frequency/count, so no explicit selection needed
                    dyn_y_axis_col = None # Not used directly by px.histogram for Y-axis
                else:
                    dyn_y_axis_col = st.selectbox("Select Y-axis Column", numeric_cols_dyn, key="dashboard_dyn_y_other")

            dyn_color_col = "None"
            if dyn_chart_type not in ["Pie Chart", "Word Cloud", "Density Map"]:
                 dyn_color_col = st.selectbox("Group by Color (Optional)", ["None"] + all_cols_dyn, key="dashboard_dyn_color_select")


            # Chart generation logic - now reactive without a button
            if dyn_x_axis_col and (dyn_y_axis_col or dyn_chart_type == "Histogram"): # Y-axis can be None for histogram
                st.markdown("---")
                st.subheader("Your Dynamic Plot:")

                fig_dyn = None
                try:
                    if dyn_chart_type == "Bar Chart":
                        # If x-axis is categorical or low unique count, group and average
                        if current_df_for_dynamic_chart[dyn_x_axis_col].dtype == 'object' or current_df_for_dynamic_chart[dyn_x_axis_col].nunique() < 50:
                            plot_df_dyn = current_df_for_dynamic_chart.groupby(dyn_x_axis_col)[dyn_y_axis_col].mean().reset_index()
                            fig_dyn = px.bar(plot_df_dyn, x=dyn_x_axis_col, y=dyn_y_axis_col,
                                             title=f'Dynamic: Average {dyn_y_axis_col} by {dyn_x_axis_col}',
                                             color=dyn_color_col if dyn_color_col != "None" else None)
                        else: # Otherwise, plot directly
                            fig_dyn = px.bar(current_df_for_dynamic_chart, x=dyn_x_axis_col, y=dyn_y_axis_col,
                                             title=f'Dynamic: {dyn_y_axis_col} by {dyn_x_axis_col}',
                                             color=dyn_color_col if dyn_color_col != "None" else None)

                    elif dyn_chart_type == "Line Chart":
                        fig_dyn = px.line(current_df_for_dynamic_chart, x=dyn_x_axis_col, y=dyn_y_axis_col,
                                          title=f'Dynamic: {dyn_y_axis_col} Trend over {dyn_x_axis_col}',
                                          color=dyn_color_col if dyn_color_col != "None" else None)
                    elif dyn_chart_type == "Area Chart":
                        fig_dyn = px.area(current_df_for_dynamic_chart, x=dyn_x_axis_col, y=dyn_y_axis_col,
                                          title=f'Dynamic: Area of {dyn_y_axis_col} over {dyn_x_axis_col}',
                                          color=dyn_color_col if dyn_color_col != "None" else None)
                    elif dyn_chart_type == "Scatter Plot":
                        fig_dyn = px.scatter(current_df_for_dynamic_chart, x=dyn_x_axis_col, y=dyn_y_axis_col,
                                             title=f'Dynamic: {dyn_y_axis_col} vs. {dyn_x_axis_col}',
                                             color=dyn_color_col if dyn_color_col != "None" else None,
                                             hover_data=all_cols_dyn)
                    elif dyn_chart_type == "Pie Chart":
                        plot_df_dyn = current_df_for_dynamic_chart.groupby(dyn_x_axis_col)[dyn_y_axis_col].sum().reset_index()
                        fig_dyn = px.pie(plot_df_dyn, names=dyn_x_axis_col, values=dyn_y_axis_col,
                                         title=f'Dynamic: Distribution of {dyn_y_axis_col} by {dyn_x_axis_col}')
                    elif dyn_chart_type == "Box Plot":
                        fig_dyn = px.box(current_df_for_dynamic_chart, x=dyn_x_axis_col if dyn_x_axis_col != "None" else None, y=dyn_y_axis_col,
                                         title=f'Dynamic: Box Plot of {dyn_y_axis_col}' + (f' by {dyn_x_axis_col}' if dyn_x_axis_col != "None" else ""),
                                         color=dyn_color_col if dyn_color_col != "None" else None)
                    elif dyn_chart_type == "Violin Plot":
                        fig_dyn = px.violin(current_df_for_dynamic_chart, x=dyn_x_axis_col if dyn_x_axis_col != "None" else None, y=dyn_y_axis_col,
                                            title=f'Dynamic: Violin Plot of {dyn_y_axis_col}' + (f' by {dyn_x_axis_col}' if dyn_x_axis_col != "None" else ""),
                                            color=dyn_color_col if dyn_color_col != "None" else None)
                    elif dyn_chart_type == "Histogram":
                        fig_dyn = px.histogram(current_df_for_dynamic_chart, x=dyn_x_axis_col,
                                               title=f'Dynamic: Distribution of {dyn_x_axis_col}',
                                               color=dyn_color_col if dyn_color_col != "None" else None)
                    
                    if fig_dyn:
                        st.plotly_chart(fig_dyn, use_container_width=True)
                    else:
                        st.warning("Could not generate dynamic chart. Please check column selections.")

                except Exception as e:
                    st.error(f"Error generating dynamic chart: {e}. Please ensure selected columns are suitable for the chart type.")
                    st.info("Hint: For Bar charts with categorical X, ensure Y is numerical. For Line/Scatter/Area, both X and Y should be suitable for plotting. For Box/Violin plots, Y should be numerical. For Histogram, X can be numerical or categorical.")
            else:
                st.info("Select valid columns to generate a dynamic plot.")
        else:
            st.info("Load data to enable the dynamic chart builder.")

    st.markdown("---")

    with st.expander("View Raw Dashboard Data (Simulated)"):
        st.dataframe(dashboard_df)
        st.caption("This data is simulated and represents hypothetical trends. For real-world analysis, integrate actual organizational data and industry reports.")
    
    with st.expander("View Raw Simulated Incident Data (for Density Map)"):
        st.dataframe(incident_locations_df)
        st.caption("This data is simulated for demonstration purposes.")

# --- Data Visualizer View ---
elif app_mode == "📊 Data Visualizer":
    st.header("Interactive Data Visualizer for AI Governance Data")
    st.markdown("---")
    st.info("Upload your own data file (.csv, .pdf, .txt, .docx, .json, .xlsx) or select from pre-loaded sample datasets. Interactive charts are available for structured data (CSV, JSON, XLSX). Text content will be displayed with a summary.")

    # --- Data Loading Section ---
    st.subheader("1. Load Your Data")
    col_upload, col_select = st.columns([1, 1])

    with col_upload:
        # Modified file uploader to accept all desired types
        uploaded_file = st.file_uploader("Upload Your Own File (.csv, .pdf, .txt, .docx, .json, .xlsx)", type=["csv", "pdf", "txt", "docx", "json", "xlsx"], label_visibility="collapsed")

    with col_select:
        # All preloaded_datasets can be listed here, then handled by type
        all_preloaded_data_options = ["None"] + list(preloaded_datasets.keys())
        selected_dataset_name = st.selectbox("Or Select a Pre-loaded Dataset", all_preloaded_data_options, label_visibility="collapsed")

    df = None
    file_content_text = "" # To store extracted text for non-CSV files

    if uploaded_file is not None:
        file_type = uploaded_file.type
        with st.spinner(f"Processing uploaded {file_type.split('/')[-1]} file..."):
            try:
                uploaded_file.seek(0) # Reset file pointer for re-reading
                if file_type == "text/csv":
                    df = pd.read_csv(uploaded_file)
                    st.success("Successfully loaded uploaded CSV data.")
                elif file_type == "application/json":
                    df = pd.read_json(uploaded_file)
                    st.success("Successfully loaded uploaded JSON data.")
                elif file_type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": # .xlsx
                    df = pd.read_excel(uploaded_file)
                    st.success("Successfully loaded uploaded XLSX data.")
                elif file_type == "application/pdf":
                    file_content_text = extract_text_from_pdf(uploaded_file)
                    st.success("Successfully extracted text from uploaded PDF.")
                elif file_type == "text/plain":
                    file_content_text = extract_text_from_txt(uploaded_file)
                    st.success("Successfully extracted text from uploaded TXT file.")
                elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingprocessingml.document":
                    file_content_text = extract_text_from_docx(uploaded_file)
                    st.success("Successfully extracted text from uploaded DOCX file.")
                else:
                    st.warning(f"Unsupported file type for full visualization: {file_type}. Attempting to display raw content.")
                    file_content_text = uploaded_file.read().decode("utf-8", errors='ignore') # Fallback for unknown types
                
                if not file_content_text.strip() and (df is None or df.empty):
                    st.warning("Could not extract meaningful content or load data from the uploaded file.")

            except Exception as e:
                st.error(f"Error processing uploaded file: {e}. Please ensure it's a valid file and its format is correct.")
                df = None
                file_content_text = ""

    elif selected_dataset_name and selected_dataset_name != "None":
        dataset_path_or_var = preloaded_datasets[selected_dataset_name]
        with st.spinner(f"Loading '{selected_dataset_name}'..."):
            try:
                if dataset_path_or_var == "dummy_df":
                    df = dummy_df
                    st.info("Using sample governance data.")
                elif dataset_path_or_var.endswith('.csv'):
                    df = pd.read_csv(dataset_path_or_var)
                    st.info(f"Using pre-loaded CSV dataset: '{selected_dataset_name}'.")
                elif dataset_path_or_var.endswith('.json'):
                    df = pd.read_json(dataset_path_or_var)
                    st.info(f"Using pre-loaded JSON dataset: '{selected_dataset_name}'.")
                elif dataset_path_or_var.endswith('.xlsx'):
                    df = pd.read_excel(dataset_path_or_var)
                    st.info(f"Using pre-loaded XLSX dataset: '{selected_dataset_name}'.")
                elif dataset_path_or_var.endswith('.pdf'):
                    with open(dataset_path_or_var, "rb") as f:
                        file_content_text = extract_text_from_pdf(f)
                    st.info(f"Using pre-loaded PDF document: '{selected_dataset_name}'.")
                elif dataset_path_or_var.endswith('.txt'):
                    with open(dataset_path_or_var, "r", encoding="utf-8") as f:
                        file_content_text = extract_text_from_txt(f)
                    st.info(f"Using pre-loaded TXT document: '{selected_dataset_name}'.")
                elif dataset_path_or_var.endswith('.docx'):
                    with open(dataset_path_or_var, "rb") as f:
                        file_content_text = extract_text_from_docx(f)
                    st.info(f"Using pre-loaded DOCX document: '{selected_dataset_name}'.")

                if not file_content_text.strip() and (df is None or df.empty):
                    st.warning("Could not load meaningful content from the selected pre-loaded file.")

            except FileNotFoundError:
                st.error(f"Error: Dataset file '{dataset_path_or_var}' not found. Please ensure it's in your project's 'data/' or 'dummy_data/' folder and named correctly.")
                df = None
                file_content_text = ""
            except Exception as e:
                st.error(f"An unexpected error occurred while loading pre-loaded file: {e}")
                df = None
                file_content_text = ""
    else:
        st.info("Upload a file or select a pre-loaded dataset to view its content and generate interactive visualizations.")

    # --- Conditional Display Logic based on loaded data type ---
    if df is not None and not df.empty:
        st.subheader("Data Preview:")
        st.dataframe(df.head())

        # --- General Data Summary (New) ---
        st.markdown("---")
        st.subheader("Overall Data Summary")
        num_rows, num_cols = df.shape
        st.markdown(f"The dataset contains **{num_rows} rows** and **{num_cols} columns**.")
        
        st.markdown("##### Column Information:")
        col_info_md = "```\n"
        col_info_md += f"{'Column Name':<25} {'Data Type':<15} {'Non-Null Count':<15}\n"
        col_info_md += "-" * 55 + "\n"
        for col in df.columns:
            dtype = str(df[col].dtype)
            non_null_count = df[col].count()
            col_info_md += f"{col:<25} {dtype:<15} {non_null_count:<15}\n"
        col_info_md += "```"
        st.markdown(col_info_md)

        numeric_cols_summary = df.select_dtypes(include=['number'])
        if not numeric_cols_summary.empty:
            st.markdown("##### Numerical Column Statistics:")
            st.dataframe(numeric_cols_summary.describe().transpose())
        
        categorical_cols_summary = df.select_dtypes(include=['object'])
        if not categorical_cols_summary.empty:
            st.markdown("##### Categorical Column Value Counts (Top 5):")
            for col in categorical_cols_summary.columns:
                st.write(f"**{col}**: {df[col].value_counts().head(5).to_dict()}")

        st.markdown("---")

        # --- Dynamic KPI Section ---
        st.subheader("2.1. Dynamic Key Performance Indicators (KPIs)")
        st.info("Select a numerical column to view its aggregated statistics.")

        numeric_columns_for_kpi = df.select_dtypes(include=['number']).columns.tolist()
        if numeric_columns_for_kpi:
            selected_kpi_column = st.selectbox(
                "Select Column for KPI Calculation",
                numeric_columns_for_kpi,
                key="visualizer_kpi_column"
            )

            if selected_kpi_column:
                st.markdown(f"**KPIs for: `{selected_kpi_column}`**")
                with st.container(border=True):
                    col_kpi1, col_kpi2, col_kpi3 = st.columns(3)
                    col_kpi4, col_kpi5, col_kpi6 = st.columns(3)

                    # Calculate KPIs
                    avg_val = df[selected_kpi_column].mean()
                    sum_val = df[selected_kpi_column].sum()
                    count_val = df[selected_kpi_column].count() # Count of non-nulls
                    min_val = df[selected_kpi_column].min()
                    max_val = df[selected_kpi_column].max()
                    unique_count = df[selected_kpi_column].nunique()

                    with col_kpi1:
                        st.metric(label=f"Average {selected_kpi_column}", value=f"{avg_val:,.2f}")
                    with col_kpi2:
                        st.metric(label=f"Sum of {selected_kpi_column}", value=f"{sum_val:,.2f}")
                    with col_kpi3:
                        st.metric(label=f"Count of Records", value=f"{count_val:,}")
                    with col_kpi4:
                        st.metric(label=f"Minimum {selected_kpi_column}", value=f"{min_val:,.2f}")
                    with col_kpi5:
                        st.metric(label=f"Maximum {selected_kpi_column}", value=f"{max_val:,.2f}")
                    with col_kpi6:
                        st.metric(label=f"Unique Values in {selected_kpi_column}", value=f"{unique_count:,}")
            else:
                st.info("No numerical columns available to calculate KPIs.")
        else:
            st.info("No numerical columns found in the loaded dataset for KPI calculation.")


        st.markdown("---")
        st.subheader("2.2. Choose Visualization Mode")
        visualization_mode = st.radio(
            "Select how you want to visualize your data:",
            ["Single Chart Builder", "Multi-Chart Dashboard"],
            key="data_visualizer_mode_selector"
        )

        if visualization_mode == "Single Chart Builder":
            st.markdown("---")
            st.subheader("3. Single Chart Builder")
            st.info("Select a single chart type and its columns to generate a custom visualization.")
            
            all_columns = df.columns.tolist()
            numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
            categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

            col_chart_type, col_x_axis, col_y_axis = st.columns(3)

            with col_chart_type:
                chart_type = st.selectbox(
                    "Select Chart Type",
                    ["Bar Chart", "Line Chart", "Area Chart", "Scatter Plot",
                     "Pie Chart", "Box Plot", "Violin Plot", "Histogram",
                     "Density Heatmap", "Map (Scattermapbox)"],
                    key="visualizer_chart_type_select"
                )

            with col_x_axis:
                if chart_type in ["Pie Chart"]:
                    x_axis_col = st.selectbox("Select Category/Names Column", all_columns, key="visualizer_x_pie")
                elif chart_type in ["Box Plot", "Violin Plot"]:
                    x_axis_col = st.selectbox("Select Grouping Column (Optional)", ["None"] + categorical_columns, key="visualizer_x_group")
                elif chart_type == "Histogram":
                    x_axis_col = st.selectbox("Select X-axis Column", all_columns, key="visualizer_x_hist")
                elif chart_type == "Map (Scattermapbox)":
                    x_axis_col = st.selectbox("Select Longitude Column", all_columns, key="visualizer_lon_map")
                elif chart_type == "Density Heatmap":
                     x_axis_col = st.selectbox("Select X-axis (for Heatmap)", all_columns, key="visualizer_x_heatmap")
                else:
                    x_axis_col = st.selectbox("Select X-axis Column", all_columns, key="visualizer_x_other")

            with col_y_axis:
                if chart_type in ["Pie Chart", "Histogram"]:
                    y_axis_col = st.selectbox("Select Value Column", numeric_columns, key="visualizer_y_pie_hist")
                elif chart_type in ["Box Plot", "Violin Plot"]:
                    y_axis_col = st.selectbox("Select Value Column", numeric_columns, key="visualizer_y_value")
                elif chart_type == "Map (Scattermapbox)":
                    y_axis_col = st.selectbox("Select Latitude Column", all_columns, key="visualizer_lat_map")
                elif chart_type == "Density Heatmap":
                    y_axis_col = st.selectbox("Select Y-axis (for Heatmap)", all_columns, key="visualizer_y_heatmap")
                else:
                    y_axis_col = st.selectbox("Select Y-axis Column", numeric_columns, key="visualizer_y_other")

            color_col = "None"
            size_col = "None"
            if chart_type not in ["Pie Chart", "Word Cloud", "Density Heatmap", "Histogram"]:
                color_col = st.selectbox("Group by Color (Optional)", ["None"] + all_columns, key="visualizer_color_select")
            if chart_type == "Map (Scattermapbox)":
                size_col = st.selectbox("Select Size Column (Optional)", ["None"] + numeric_columns, key="visualizer_size_map")
                mapbox_style = st.selectbox("Map Style", ["carto-positron", "open-street-map", "mapbox://styles/mapbox/light-v10", "mapbox://styles/mapbox/dark-v10"], key="mapbox_style")

            if (x_axis_col and y_axis_col) or (chart_type == "Histogram" and x_axis_col) or \
               (chart_type in ["Map (Scattermapbox)", "Density Heatmap"] and x_axis_col and y_axis_col):
                st.markdown("---")
                st.subheader("Your Interactive Plot:")

                fig = None
                try:
                    if chart_type == "Bar Chart":
                        if df[x_axis_col].dtype == 'object' or df[x_axis_col].nunique() < 50:
                            plot_df = df.groupby(x_axis_col)[y_axis_col].mean().reset_index()
                            fig = px.bar(plot_df, x=x_axis_col, y=y_axis_col,
                                         title=f'Average {y_axis_col} by {x_axis_col}',
                                         color=color_col if color_col != "None" else None)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"This bar chart displays the average of **{y_axis_col}** for each category in **{x_axis_col}**. It's useful for comparing numerical values across different groups.")
                        else:
                            fig = px.bar(df, x=x_axis_col, y=y_axis_col,
                                         title=f'{y_axis_col} by {x_axis_col}',
                                         color=color_col if color_col != "None" else None)
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"This bar chart shows individual values of **{y_axis_col}** for each entry in **{x_axis_col}**. It helps in understanding specific data points.")

                    elif chart_type == "Line Chart":
                        fig = px.line(df, x=x_axis_col, y=y_axis_col,
                                      title=f'{y_axis_col} Trend over {x_axis_col}',
                                      color=color_col if color_col != "None" else None)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"This line chart illustrates the trend of **{y_axis_col}** over **{x_axis_col}**. Ideal for time-series data or showing progression.")
                    elif chart_type == "Area Chart":
                        fig = px.area(df, x=x_axis_col, y=y_axis_col,
                                      title=f'Area of {y_axis_col} over {x_axis_col}',
                                      color=color_col if color_col != "None" else None)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"This area chart shows the magnitude of **{y_axis_col}** over **{x_axis_col}**, with the area under the line filled. Useful for visualizing cumulative totals or relative contributions.")
                    elif chart_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_axis_col, y=y_axis_col,
                                         title=f'{y_axis_col} vs. {x_axis_col}',
                                         color=color_col if color_col != "None" else None,
                                         hover_data=all_columns)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"This scatter plot visualizes the relationship between **{x_axis_col}** and **{y_axis_col}**. It helps in identifying correlations, clusters, or outliers in your data.")
                    elif chart_type == "Pie Chart":
                        plot_df = df.groupby(x_axis_col)[y_axis_col].sum().reset_index()
                        fig = px.pie(plot_df, names=x_axis_col, values=y_axis_col,
                                     title=f'Distribution of {y_axis_col} by {x_axis_col}')
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"This pie chart shows the proportional distribution of **{y_axis_col}** across different categories in **{x_axis_col}**. Each slice represents a percentage of the total.")
                    elif chart_type == "Box Plot":
                        fig = px.box(df, x=x_axis_col if x_axis_col != "None" else None, y=y_axis_col,
                                     title=f'Box Plot of {y_axis_col}' + (f' by {x_axis_col}' if x_axis_col != "None" else ""),
                                     color=color_col if color_col != "None" else None)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"This box plot illustrates the distribution of **{y_axis_col}**, including median, quartiles, and potential outliers. If a grouping column is selected, it compares distributions across groups.")
                    elif chart_type == "Violin Plot":
                        fig = px.violin(df, x=x_axis_col if x_axis_col != "None" else None, y=y_axis_col,
                                        title=f'Violin Plot of {y_axis_col}' + (f' by {x_axis_col}' if x_axis_col != "None" else ""),
                                        color=color_col if color_col != "None" else None)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"This violin plot shows the distribution of **{y_axis_col}** through a kernel density estimate. It's similar to a box plot but provides a richer view of data density, especially useful when comparing distributions across groups.")
                    elif chart_type == "Histogram":
                        fig = px.histogram(df, x=x_axis_col,
                                           title=f'Distribution of {x_axis_col}',
                                           color=color_col if color_col != "None" else None)
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"This histogram shows the frequency distribution of **{x_axis_col}**, indicating how many times values fall within specified ranges.")
                    elif chart_type == "Density Heatmap":
                        fig = px.density_heatmap(df, x=x_axis_col, y=y_axis_col,
                                                title=f'Density Heatmap of {x_axis_col} vs. {y_axis_col}')
                        st.plotly_chart(fig, use_container_width=True)
                        st.caption(f"This density heatmap visualizes the concentration of data points across two numerical variables, **{x_axis_col}** and **{y_axis_col}**. Darker areas indicate higher density.")
                    elif chart_type == "Map (Scattermapbox)":
                        lat_col_exists = y_axis_col in df.columns and pd.api.types.is_numeric_dtype(df[y_axis_col])
                        lon_col_exists = x_axis_col in df.columns and pd.api.types.is_numeric_dtype(df[x_axis_col])

                        if lat_col_exists and lon_col_exists:
                            fig = px.scatter_mapbox(df, lat=y_axis_col, lon=x_axis_col,
                                                    color=color_col if color_col != "None" else None,
                                                    size=size_col if size_col != "None" else None,
                                                    zoom=1, height=500,
                                                    mapbox_style=mapbox_style,
                                                    title=f'Geographic Distribution of Data Points')
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"This map plots individual data points using **Latitude ({y_axis_col})** and **Longitude ({x_axis_col})**. Color and size can represent additional data attributes, helping to visualize geographical patterns.")
                        else:
                            st.warning(f"Map visualization requires valid numeric Latitude ('{y_axis_col}') and Longitude ('{x_axis_col}') columns in the dataset. Please ensure your data contains appropriate geographical coordinates.")
                            fig = None
                    
                    if fig:
                        pass # Chart already shown with caption
                    else:
                        st.warning("Could not generate chart. Please check column selections and data types.")

                except Exception as e:
                    st.error(f"Error generating chart: {e}. Please ensure selected columns are suitable for the chart type and data is clean.")
                    st.info("Hint: For Bar charts with categorical X, ensure Y is numerical. For Line/Scatter/Area, both X and Y should be suitable for plotting. For Box/Violin plots, Y should be numerical. For Histogram, X can be numerical or categorical. For Map, ensure valid Latitude/Longitude columns. For Heatmap, ensure appropriate numerical columns.")
            else:
                st.info("Please select valid columns for the chosen chart type to generate a plot.")

        elif visualization_mode == "Multi-Chart Dashboard":
            st.markdown("---")
            st.subheader("3. Multi-Chart Dashboard Builder")
            st.info("Select multiple charts to build a custom dashboard from your loaded dataset. Charts will only appear if suitable data columns are available and configured.")

            all_cols = df.columns.tolist()
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            text_cols = [col for col in all_cols if pd.api.types.is_string_dtype(df[col])]


            available_dashboard_charts = []
            if len(numeric_cols) >= 2:
                available_dashboard_charts.append("Scatter Plot (Relationship)")
                available_dashboard_charts.append("Density Heatmap")
            if len(numeric_cols) >= 1:
                available_dashboard_charts.append("Histogram (Distribution)")
                available_dashboard_charts.append("Box Plot (Distribution)")
                available_dashboard_charts.append("Violin Plot (Distribution)")
                available_dashboard_charts.append("Line Chart (Trend)")
                available_dashboard_charts.append("Bar Chart (Comparison)")
                available_dashboard_charts.append("Pie Chart (Proportion)")

            # Check for geo columns for map
            has_lat_lon = False
            possible_lat_cols = [col for col in numeric_cols if 'lat' in col.lower() or 'latitude' in col.lower()]
            possible_lon_cols = [col for col in numeric_cols if 'lon' in col.lower() or 'longitude' in col.lower()]
            if possible_lat_cols and possible_lon_cols:
                available_dashboard_charts.append("Geographic Map (Scattermapbox)")
                has_lat_lon = True

            # Check for text columns for word cloud
            has_text_col = False
            if text_cols:
                available_dashboard_charts.append("Word Cloud (Text Analysis)")
                has_text_col = True

            selected_dashboard_charts = st.multiselect(
                "Select Dashboard Components:",
                options=available_dashboard_charts,
                key="data_visualizer_multi_charts_select"
            )

            if not selected_dashboard_charts:
                st.info("Select some charts to see your dashboard.")

            st.markdown("---")
            st.subheader("Your Custom Dashboard Visualizations:")

            for chart_name in selected_dashboard_charts:
                st.markdown(f"### {chart_name}")
                fig = None # Initialize fig for each chart

                try:
                    if chart_name == "Scatter Plot (Relationship)":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_col = st.selectbox(f"X-axis for {chart_name}", numeric_cols, key=f"multi_{chart_name}_x_vis")
                        with col2:
                            y_col = st.selectbox(f"Y-axis for {chart_name}", numeric_cols, key=f"multi_{chart_name}_y_vis")
                        with col3:
                            color_by = st.selectbox(f"Color by (Optional) for {chart_name}", ["None"] + all_cols, key=f"multi_{chart_name}_color_vis")
                        
                        if x_col and y_col:
                            fig = px.scatter(df, x=x_col, y=y_col,
                                             color=color_by if color_by != "None" else None,
                                             title=f'{y_col} vs. {x_col}')
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Shows the relationship between **{y_col}** and **{x_col}**. Color can represent a third variable.")
                        else:
                            st.info(f"Select X and Y axes for {chart_name}.")
                    
                    elif chart_name == "Histogram (Distribution)":
                        col1, col2 = st.columns(2)
                        with col1:
                            x_col = st.selectbox(f"Value Column for {chart_name}", numeric_cols, key=f"multi_{chart_name}_x_vis")
                        with col2:
                            color_by = st.selectbox(f"Group by (Optional) for {chart_name}", ["None"] + categorical_cols, key=f"multi_{chart_name}_group_vis")
                        
                        if x_col:
                            fig = px.histogram(df, x=x_col,
                                               color=color_by if color_by != "None" else None,
                                               title=f'Distribution of {x_col}')
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Displays the frequency distribution of **{x_col}**.")
                        else:
                            st.info(f"Select a value column for {chart_name}.")
                    
                    elif chart_name == "Box Plot (Distribution)":
                        col1, col2 = st.columns(2)
                        with col1:
                            y_col = st.selectbox(f"Value Column for {chart_name}", numeric_cols, key=f"multi_{chart_name}_y_vis")
                        with col2:
                            group_by = st.selectbox(f"Group by (Optional) for {chart_name}", ["None"] + categorical_cols, key=f"multi_{chart_name}_group_vis")
                        
                        if y_col:
                            fig = px.box(df, x=group_by if group_by != "None" else None, y=y_col,
                                         color=group_by if group_by != "None" else None,
                                         title=f'Box Plot of {y_col}' + (f' by {group_by}' if group_by != "None" else ""))
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Shows the distribution, outliers, and quartiles of **{y_col}**.")
                        else:
                            st.info(f"Select a value column for {chart_name}.")

                    elif chart_name == "Violin Plot (Distribution)":
                        col1, col2 = st.columns(2)
                        with col1:
                            y_col = st.selectbox(f"Value Column for {chart_name}", numeric_cols, key=f"multi_{chart_name}_y_vis")
                        with col2:
                            group_by = st.selectbox(f"Group by (Optional) for {chart_name}", ["None"] + categorical_cols, key=f"multi_{chart_name}_group_vis")
                        
                        if y_col:
                            fig = px.violin(df, x=group_by if group_by != "None" else None, y=y_col,
                                            color=group_by if group_by != "None" else None,
                                            title=f'Violin Plot of {y_col}' + (f' by {group_by}' if group_by != "None" else ""))
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Displays the distribution of **{y_col}** through a kernel density estimate, showing density and statistical properties.")
                        else:
                            st.info(f"Select a value column for {chart_name}.")

                    elif chart_name == "Line Chart (Trend)":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_col = st.selectbox(f"X-axis (Time/Category) for {chart_name}", all_cols, key=f"multi_{chart_name}_x_vis")
                        with col2:
                            y_col = st.selectbox(f"Y-axis (Value) for {chart_name}", numeric_cols, key=f"multi_{chart_name}_y_vis")
                        with col3:
                            color_by = st.selectbox(f"Color by (Optional) for {chart_name}", ["None"] + all_cols, key=f"multi_{chart_name}_color_vis")
                        
                        if x_col and y_col:
                            fig = px.line(df, x=x_col, y=y_col,
                                          color=color_by if color_by != "None" else None,
                                          title=f'{y_col} Trend over {x_col}')
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Shows the trend of **{y_col}** over **{x_col}**.")
                        else:
                            st.info(f"Select X and Y axes for {chart_name}.")

                    elif chart_name == "Bar Chart (Comparison)":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            x_col = st.selectbox(f"X-axis (Category) for {chart_name}", all_cols, key=f"multi_{chart_name}_x_vis")
                        with col2:
                            y_col = st.selectbox(f"Y-axis (Value) for {chart_name}", numeric_cols, key=f"multi_{chart_name}_y_vis")
                        with col3:
                            color_by = st.selectbox(f"Color by (Optional) for {chart_name}", ["None"] + all_cols, key=f"multi_{chart_name}_color_vis")
                        
                        if x_col and y_col:
                            # For bar chart, if x is categorical and y is numerical, average y by x
                            if df[x_col].dtype == 'object' or df[x_col].nunique() < 50:
                                plot_df_bar = df.groupby(x_col)[y_col].mean().reset_index()
                                fig = px.bar(plot_df_bar, x=x_col, y=y_col,
                                             color=color_by if color_by != "None" else None,
                                             title=f'Average {y_col} by {x_col}')
                            else:
                                fig = px.bar(df, x=x_col, y=y_col,
                                             color=color_by if color_by != "None" else None,
                                             title=f'{y_col} by {x_col}')
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Compares **{y_col}** across different categories of **{x_col}**.")
                        else:
                            st.info(f"Select X and Y axes for {chart_name}.")

                    elif chart_name == "Pie Chart (Proportion)":
                        col1, col2 = st.columns(2)
                        with col1:
                            names_col = st.selectbox(f"Names Column for {chart_name}", all_cols, key=f"multi_{chart_name}_names_vis")
                        with col2:
                            values_col = st.selectbox(f"Values Column for {chart_name}", numeric_cols, key=f"multi_{chart_name}_values_vis")
                        
                        if names_col and values_col:
                            plot_df_pie = df.groupby(names_col)[values_col].sum().reset_index()
                            fig = px.pie(plot_df_pie, names=names_col, values=values_col,
                                         title=f'Proportion of {values_col} by {names_col}')
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Shows the proportional distribution of **{values_col}** across categories in **{names_col}**.")
                        else:
                            st.info(f"Select names and values columns for {chart_name}.")

                    elif chart_name == "Density Heatmap":
                        col1, col2 = st.columns(2)
                        with col1:
                            x_col = st.selectbox(f"X-axis for {chart_name}", numeric_cols, key=f"multi_{chart_name}_x_heatmap_vis")
                        with col2:
                            y_col = st.selectbox(f"Y-axis for {chart_name}", numeric_cols, key=f"multi_{chart_name}_y_heatmap_vis")
                        
                        if x_col and y_col:
                            fig = px.density_heatmap(df, x=x_col, y=y_col,
                                                     title=f'Density Heatmap of {x_col} vs. {y_col}')
                            st.plotly_chart(fig, use_container_width=True)
                            st.caption(f"Visualizes the concentration of data points across two numerical variables, **{x_col}** and **{y_col}**. Darker areas indicate higher density.")
                        else:
                            st.info(f"Select X and Y axes for {chart_name}.")
                    
                    elif chart_name == "Geographic Map (Scattermapbox)":
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            lon_col = st.selectbox(f"Longitude Column for {chart_name}", possible_lon_cols, key=f"multi_{chart_name}_lon_vis")
                        with col2:
                            lat_col = st.selectbox(f"Latitude Column for {chart_name}", possible_lat_cols, key=f"multi_{chart_name}_lat_vis")
                        with col3:
                            color_by = st.selectbox(f"Color by (Optional) for {chart_name}", ["None"] + all_cols, key=f"multi_{chart_name}_color_map_vis")
                            size_by = st.selectbox(f"Size by (Optional) for {chart_name}", ["None"] + numeric_cols, key=f"multi_{chart_name}_size_map_vis")
                        map_style = st.selectbox(f"Map Style for {chart_name}", ["carto-positron", "open-street-map", "mapbox://styles/mapbox/light-v10", "mapbox://styles/mapbox/dark-v10"], key=f"multi_{chart_name}_map_style_vis")

                        if lon_col and lat_col:
                            if pd.api.types.is_numeric_dtype(df[lat_col]) and pd.api.types.is_numeric_dtype(df[lon_col]):
                                fig = px.scatter_mapbox(df, lat=lat_col, lon=lon_col,
                                                        color=color_by if color_by != "None" else None,
                                                        size=size_by if size_by != "None" else None,
                                                        zoom=1, height=400,
                                                        mapbox_style=map_style,
                                                        title=f'Geographic Distribution of Data Points')
                                st.plotly_chart(fig, use_container_width=True)
                                st.caption(f"Plots data points on a map based on **Latitude ({lat_col})** and **Longitude ({lon_col})**. Color and size can represent additional data attributes, helping to visualize geographical patterns.")
                            else:
                                st.warning(f"Skipping '{chart_name}': Latitude and Longitude columns must be numeric.")
                        else:
                            st.info(f"Please select valid Latitude and Longitude columns for {chart_name}.")

                    elif chart_name == "Word Cloud (Text Analysis)":
                        text_col = st.selectbox(f"Select Text Column for {chart_name}", text_cols, key=f"multi_{chart_name}_text_col_vis")
                        if text_col and text_col in df.columns:
                            text_data = " ".join(df[text_col].dropna().astype(str))
                            if text_data:
                                wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='cividis').generate(text_data)
                                plt.figure(figsize=(10, 5))
                                plt.imshow(wordcloud, interpolation='bilinear')
                                plt.axis('off')
                                st.pyplot(plt)
                                st.caption(f"Highlights frequently appearing terms in **'{text_col}'**.")
                                plt.close()
                            else:
                                st.warning(f"Skipping '{chart_name}': No text data found in column '{text_col}'.")
                        else:
                            st.info(f"Please select a valid text column for {chart_name}.")
                except KeyError as ke:
                    st.warning(f"Missing column for '{chart_name}': {ke}. Please ensure all required columns are selected and exist in the data.")
                except Exception as e:
                    st.error(f"Error generating {chart_name}: {e}")
                    st.info(f"Please check column data types and selections for {chart_name}.")
                
                st.markdown("---") # Separator between charts
    elif file_content_text.strip():
        st.subheader("Extracted Document Content & AI Summary:")
        col_doc_preview, col_doc_summary = st.columns(2)

        with col_doc_preview:
            st.markdown("#### Document Content Preview")
            st.text_area("Full Document Content", file_content_text[:5000] + ("..." if len(file_content_text) > 5000 else ""), height=600, help="This is the extracted text content from your document.", label_visibility="collapsed") # Added label_visibility

        with col_doc_summary:
            st.markdown("#### AI-Generated Summary (10 Key Points)")
            if gemini_model:
                if st.button("Generate 10-Point Summary", key="generate_doc_summary_visualizer"):
                    with st.spinner("Generating summary..."):
                        try:
                            # Using a larger chunk of text for summarization to allow more comprehensive summaries
                            # Note: Very long documents may still be truncated by the model's token limit.
                            text_to_summarize = file_content_text
                            
                            prompt = f"Summarize the following document into exactly 10 concise key bullet points:\n\n{text_to_summarize}"
                            response = gemini_model.generate_content(prompt)
                            summary = response.text
                            st.success("Summary generated!")
                            st.markdown(summary)

                            st.markdown("---")
                            st.info("💡 You can now chat with this document! Go to the **💬 AI Chat Assistant** tab to ask specific questions about its content.")

                        except Exception as e:
                            st.error(f"Failed to generate summary: {e}. This might be due to document length or API issues. Please try again.")
            else:
                st.warning("AI model not available for summarization. Please check API key.")
        
        st.markdown("---")
        st.info("The chart builder is designed for structured data (CSV, JSON, XLSX). Please upload a structured file to use charting features.")
    else:
        st.warning("No data loaded. Upload a file or select a pre-loaded dataset.")

# --- Document Summarizer View ---
elif app_mode == "📄 Document Summarizer":
    st.header("AI Governance Document Summarizer")
    st.markdown("---")
    if gemini_model is None:
        st.error("AI model not initialized. Please check your API key and internet connection.")
        st.stop()

    st.info("Upload a PDF, Text (.txt), or Word (.docx) document to get an AI-generated summary. Alternatively, use the sample policy text or select a pre-loaded document below for quick testing.")

    # Initialize session state for document source type if not present
    if 'summarizer_source_type' not in st.session_state:
        st.session_state.summarizer_source_type = "None"
    if 'summarizer_active_preloaded_doc_name' not in st.session_state:
        st.session_state.summarizer_active_preloaded_doc_name = "None"
    if 'summarizer_uploaded_file' not in st.session_state:
        st.session_state.summarizer_uploaded_file = None
    if 'summarizer_use_sample' not in st.session_state:
        st.session_state.summarizer_use_sample = False


    selected_text_input = ""
    
    col_upload_file, col_preloaded_doc, col_sample_text_toggle = st.columns([1, 1, 0.8])

    with col_upload_file:
        uploaded_file_current = st.file_uploader("Upload Document (.pdf, .txt, .docx)", type=["pdf", "txt", "docx"], key="summarizer_uploader", label_visibility="collapsed")
    
    with col_preloaded_doc:
        # Filter preloaded_datasets to only show PDF/TXT/DOCX files for this section
        document_datasets = {name: path for name, path in preloaded_datasets.items() if path.endswith(('.pdf', '.txt', '.docx'))}
        preloaded_options = ["None"] + list(document_datasets.keys())
        
        # Determine initial index for the selectbox based on current state
        preloaded_default_index = preloaded_options.index(st.session_state.summarizer_active_preloaded_doc_name) if st.session_state.summarizer_active_preloaded_doc_name in preloaded_options else 0
        selected_preloaded_doc_name_current = st.selectbox("Or Select a Pre-loaded Document",
                                                   preloaded_options,
                                                   index=preloaded_default_index,
                                                   key="summarizer_preloaded_doc_select",
                                                   help="Select a document from the pre-loaded examples.",
                                                   label_visibility="collapsed")

    with col_sample_text_toggle:
        use_sample_current = st.checkbox("Use Sample Policy Text",
                                 value=st.session_state.summarizer_use_sample,
                                 key="use_sample_doc_checkbox_summarizer",
                                 help="Use a built-in sample document for demonstration.")

    # --- Logic to update session state based on user interaction ---
    # Priority: Uploaded file > Pre-loaded > Sample > None
    if uploaded_file_current is not None:
        if st.session_state.summarizer_uploaded_file != uploaded_file_current:
            st.session_state.summarizer_source_type = "upload"
            st.session_state.summarizer_uploaded_file = uploaded_file_current
            st.session_state.summarizer_active_preloaded_doc_name = "None"
            st.session_state.summarizer_use_sample = False
    elif selected_preloaded_doc_name_current != "None":
        if st.session_state.summarizer_active_preloaded_doc_name != selected_preloaded_doc_name_current:
            st.session_state.summarizer_source_type = "preloaded"
            st.session_state.summarizer_active_preloaded_doc_name = selected_preloaded_doc_name_current
            st.session_state.summarizer_uploaded_file = None
            st.session_state.summarizer_use_sample = False
    elif use_sample_current:
        if not st.session_state.summarizer_use_sample:
            st.session_state.summarizer_source_type = "sample"
            st.session_state.summarizer_uploaded_file = None
            st.session_state.summarizer_active_preloaded_doc_name = "None"
            st.session_state.summarizer_use_sample = True
    else: # If none are explicitly selected, and a source was previously active, reset
        if st.session_state.summarizer_source_type != "None" and \
           st.session_state.summarizer_uploaded_file is None and \
           st.session_state.summarizer_active_preloaded_doc_name == "None" and \
           not st.session_state.summarizer_use_sample: # Check previous state to avoid constant reset
            st.session_state.summarizer_source_type = "None"
            st.session_state.summarizer_uploaded_file = None
            st.session_state.summarizer_active_preloaded_doc_name = "None"
            st.session_state.summarizer_use_sample = False

    # --- Document Loading based on active source type ---
    if st.session_state.summarizer_source_type == "upload" and st.session_state.summarizer_uploaded_file is not None:
        with st.spinner(f"Extracting text from uploaded {st.session_state.summarizer_uploaded_file.type.split('/')[-1]}..."):
            try:
                st.session_state.summarizer_uploaded_file.seek(0) 
                if st.session_state.summarizer_uploaded_file.type == "application/pdf":
                    selected_text_input = extract_text_from_pdf(st.session_state.summarizer_uploaded_file)
                elif st.session_state.summarizer_uploaded_file.type == "text/plain":
                    selected_text_input = extract_text_from_txt(st.session_state.summarizer_uploaded_file)
                elif st.session_state.summarizer_uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    selected_text_input = extract_text_from_docx(st.session_state.summarizer_uploaded_file)
                
                if not selected_text_input.strip():
                    st.warning(f"Could not extract text from the uploaded {st.session_state.summarizer_uploaded_file.type.split('/')[-1]}. It might be empty, encrypted, or image-based.")
                    selected_text_input = ""
                else:
                    st.subheader("Extracted Text Preview:")
                    st.text_area("Document Content", selected_text_input[:2000] + ("..." if len(selected_text_input) > 2000 else ""), height=300, label_visibility="collapsed")
            except Exception as e:
                st.error(f"An error occurred during file processing: {e}")
                selected_text_input = ""

    elif st.session_state.summarizer_source_type == "preloaded" and st.session_state.summarizer_active_preloaded_doc_name != "None":
        doc_path = document_datasets[st.session_state.summarizer_active_preloaded_doc_name]
        with st.spinner(f"Loading '{st.session_state.summarizer_active_preloaded_doc_name}'..."):
            try:
                if doc_path.endswith('.pdf'):
                    with open(doc_path, "rb") as f:
                        selected_text_input = extract_text_from_pdf(f)
                elif doc_path.endswith('.txt'):
                    with open(doc_path, "r", encoding="utf-8") as f:
                        selected_text_input = extract_text_from_txt(f)
                elif doc_path.endswith('.docx'):
                    with open(doc_path, "rb") as f:
                        selected_text_input = extract_text_from_docx(f)
                
                if not selected_text_input.strip():
                    st.warning(f"Could not load text from pre-loaded document '{st.session_state.summarizer_active_preloaded_doc_name}'. It might be empty or unreadable.")
                    selected_text_input = ""
                else:
                    st.subheader(f"Pre-loaded Document Preview ({st.session_state.summarizer_active_preloaded_doc_name}):")
                    st.text_area("Document Content", selected_text_input[:2000] + ("..." if len(selected_text_input) > 2000 else ""), height=300, label_visibility="collapsed")

            except FileNotFoundError:
                st.error(f"Error: Pre-loaded document file '{doc_path}' not found. Please ensure it's in your project's 'data/' folder.")
                selected_text_input = ""
            except Exception as e:
                st.error(f"An unexpected error occurred while loading pre-loaded document: {e}")
                selected_text_input = ""
    
    elif st.session_state.summarizer_source_type == "sample":
        selected_text_input = sample_policy_text
        st.subheader("Sample Policy Text Preview:")
        st.text_area("Sample Content", selected_text_input[:2000] + ("..." if len(selected_text_input) > 2000 else ""), height=300, label_visibility="collapsed")
    else:
        st.info("Upload a document, select a pre-loaded document, or check 'Use Sample Policy Text' to enable summarization.")

    # Store the content for the chat assistant
    st.session_state.chat_document_content = selected_text_input

    # Summarization button and logic
    if selected_text_input.strip(): # Only show button if there's content to summarize
        if st.button("Generate Summary"):
            if gemini_model:
                with st.spinner("Summarizing your document with Gemini... This may take a moment."):
                    try:
                        # Attempt to summarize the full text input. The model has its own internal token limits.
                        text_to_summarize = selected_text_input
                        
                        prompt = f"Summarize this policy document concisely, focusing on key takeaways, main principles, challenges, and proposed solutions for AI governance. Format as bullet points or a clear paragraph structure:\n\n{text_to_summarize}"
                        response = gemini_model.generate_content(prompt)
                        summary = response.text
                        st.subheader("AI-Generated Summary:")
                        st.success("Summary generated successfully!")
                        st.write(summary)

                        st.markdown("---")
                        st.info("💡 You can now chat with this document! Go to the **💬 AI Chat Assistant** tab to ask specific questions about its content.")

                    except Exception as e:
                        st.error(f"Failed to generate summary: {e}. This might be due to document length or API issues. Please try again.")
            else:
                st.warning("AI Model not ready. Cannot generate summary.")

        # --- Flashcards Feature (New) ---
        if st.button("Generate Flashcards for Interactive Session"):
            if gemini_model:
                with st.spinner("Generating flashcards..."):
                    try:
                        # Prompt for flashcards: asking for Q&A pairs
                        flashcard_prompt = f"""
                        From the following document, extract 5-10 key terms, concepts, or questions and their corresponding concise answers/definitions.
                        Format them as a JSON array of objects, where each object has "question" and "answer" keys.
                        Example:
                        [
                            {{ "question": "What is AI Governance?", "answer": "The framework for responsible development and deployment of AI." }},
                            {{ "question": "Key principle of Fairness?", "answer": "Minimizing bias and promoting equitable outcomes." }}
                        ]

                        Document:
                        {selected_text_input[:30000]}
                        """ # Limit input for flashcard generation for efficiency

                        flashcard_response = gemini_model.generate_content(
                            flashcard_prompt,
                            generation_config={
                                "response_mime_type": "application/json",
                                "response_schema": {
                                    "type": "ARRAY",
                                    "items": {
                                        "type": "OBJECT",
                                        "properties": {
                                            "question": {"type": "STRING"},
                                            "answer": {"type": "STRING"}
                                        },
                                        "propertyOrdering": ["question", "answer"]
                                    }
                                }
                            }
                        )
                        
                        flashcards_data = json.loads(flashcard_response.text)
                        
                        if flashcards_data:
                            st.subheader("Interactive Flashcards:")
                            st.info("Click on a flashcard to reveal the answer!")
                            for i, card in enumerate(flashcards_data):
                                with st.expander(f"**Question {i+1}: {card.get('question', 'N/A')}**"):
                                    st.markdown(card.get('answer', 'N/A'))
                        else:
                            st.warning("Could not generate flashcards. No data extracted.")

                    except json.JSONDecodeError:
                        st.error("Failed to parse flashcard data. AI response might not be in expected JSON format.")
                        st.code(flashcard_response.text) # Show raw response for debugging
                    except Exception as e:
                        st.error(f"Error generating flashcards: {e}. Please try again.")
            else:
                st.warning("AI model not available for flashcard generation.")


# --- Ethical and Compliance Audit (Prototype) View ---
elif app_mode == "⚖️ Compliance Audit (Prototype)":
    st.header("Ethical and Compliance Audit Indicator (Prototype)")
    st.markdown("---")
    st.info("Upload a policy document (PDF, TXT, or DOCX) to run a rule-based audit. This prototype checks for the presence of key AI governance concepts.")

    # Initialize session state for audit document source type
    if 'audit_source_type' not in st.session_state:
        st.session_state.audit_source_type = "None"
    if 'audit_active_preloaded_doc_name' not in st.session_state:
        st.session_state.audit_active_preloaded_doc_name = "None"
    if 'audit_uploaded_file' not in st.session_state:
        st.session_state.audit_uploaded_file = None


    selected_audit_doc_text = ""
    col_audit_upload, col_audit_preloaded = st.columns([1,1])

    with col_audit_upload:
        audit_uploaded_file_current = st.file_uploader("Upload Policy Document (.pdf, .txt, .docx)", type=["pdf", "txt", "docx"], key="audit_uploader", label_visibility="collapsed")
    
    with col_audit_preloaded:
        audit_document_datasets = {name: path for name, path in preloaded_datasets.items() if path.endswith(('.pdf', '.txt', '.docx'))}
        audit_preloaded_options = ["None"] + list(audit_document_datasets.keys())

        audit_preloaded_default_index = audit_preloaded_options.index(st.session_state.audit_active_preloaded_doc_name) if st.session_state.audit_active_preloaded_doc_name in audit_preloaded_options else 0
        selected_audit_preloaded_doc_name_current = st.selectbox("Or Select a Pre-loaded Policy Document",
                                                         audit_preloaded_options,
                                                         index=audit_preloaded_default_index,
                                                         key="audit_preloaded_doc_select",
                                                         help="Select a document from the pre-loaded examples for audit.",
                                                         label_visibility="collapsed")
    
    # --- Logic to update session state based on user interaction for Audit ---
    # Priority: Uploaded file > Pre-loaded > None
    if audit_uploaded_file_current is not None:
        if st.session_state.audit_uploaded_file != audit_uploaded_file_current:
            st.session_state.audit_source_type = "upload"
            st.session_state.audit_uploaded_file = audit_uploaded_file_current
            st.session_state.audit_active_preloaded_doc_name = "None"
    elif selected_audit_preloaded_doc_name_current != "None":
        if st.session_state.audit_active_preloaded_doc_name != selected_audit_preloaded_doc_name_current:
            st.session_state.audit_source_type = "preloaded"
            st.session_state.audit_active_preloaded_doc_name = selected_audit_preloaded_doc_name_current
            st.session_state.audit_uploaded_file = None
    else: # If none are explicitly selected, and a source was previously active, reset
        if st.session_state.audit_source_type != "None" and \
           st.session_state.audit_uploaded_file is None and \
           st.session_state.audit_active_preloaded_doc_name == "None":
            st.session_state.audit_source_type = "None"
            st.session_state.audit_uploaded_file = None
            st.session_state.audit_active_preloaded_doc_name = "None"


    # --- Document Loading based on active source type for Audit ---
    if st.session_state.audit_source_type == "upload" and st.session_state.audit_uploaded_file is not None:
        with st.spinner(f"Extracting text from uploaded {st.session_state.audit_uploaded_file.type.split('/')[-1]} for audit..."):
            try:
                st.session_state.audit_uploaded_file.seek(0)
                if st.session_state.audit_uploaded_file.type == "application/pdf":
                    selected_audit_doc_text = extract_text_from_pdf(st.session_state.audit_uploaded_file)
                elif st.session_state.audit_uploaded_file.type == "text/plain":
                    selected_audit_doc_text = extract_text_from_txt(st.session_state.audit_uploaded_file)
                elif st.session_state.audit_uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                    selected_audit_doc_text = extract_text_from_docx(st.session_state.audit_uploaded_file)
                
                if not selected_audit_doc_text.strip():
                    st.warning("Could not extract text from the uploaded document for audit.")
                    selected_audit_doc_text = ""
            except Exception as e:
                st.error(f"Error processing uploaded file for audit: {e}")
                selected_audit_doc_text = ""

    elif st.session_state.audit_source_type == "preloaded" and st.session_state.audit_active_preloaded_doc_name != "None":
        doc_path = audit_document_datasets[st.session_state.audit_active_preloaded_doc_name]
        with st.spinner(f"Loading '{st.session_state.audit_active_preloaded_doc_name}' for audit..."):
            try:
                if doc_path.endswith('.pdf'):
                    with open(doc_path, "rb") as f:
                        selected_audit_doc_text = extract_text_from_pdf(f)
                elif doc_path.endswith('.txt'):
                    with open(doc_path, "r", encoding="utf-8") as f:
                        selected_audit_doc_text = extract_text_from_txt(f)
                elif doc_path.endswith('.docx'):
                    with open(doc_path, "rb") as f:
                        selected_audit_doc_text = extract_text_from_docx(f)

                if not selected_audit_doc_text.strip():
                    st.warning(f"Could not load text from pre-loaded document '{selected_audit_preloaded_doc_name}' for audit.")
                    selected_audit_doc_text = ""
            except FileNotFoundError:
                st.error(f"Error: Pre-loaded document file '{doc_path}' not found.")
                selected_audit_doc_text = ""
            except Exception as e:
                st.error(f"An unexpected error occurred while loading pre-loaded document for audit: {e}")
                selected_audit_doc_text = ""
    else:
        st.info("Upload a document or select a pre-loaded policy document to run the compliance audit.")

    if selected_audit_doc_text.strip():
        st.subheader("Document Content for Audit Preview:")
        st.text_area("Audit Document Content", selected_audit_doc_text[:2000] + ("..." if len(selected_audit_doc_text) > 2000 else ""), height=200, label_visibility="collapsed")

        if st.button("Run Compliance Audit"):
            st.markdown("---")
            st.subheader("Audit Results:")
            with st.spinner("Analyzing document for compliance indicators..."):
                audit_results = conduct_compliance_audit(selected_audit_doc_text)
                
                total_rules = len(compliance_rules)
                found_count = sum(1 for status, _ in audit_results.values() if status)
                compliance_percentage = (found_count / total_rules) * 100 if total_rules > 0 else 0

                st.markdown(f"**Overall Compliance Score (Prototype):** {compliance_percentage:.1f}% ({found_count}/{total_rules} indicators found)")
                st.markdown("---")

                audit_report_content = f"# AI Governance Audit Report\n\n"
                audit_report_content += f"**Overall Compliance Score (Prototype):** {compliance_percentage:.1f}% ({found_count}/{total_rules} indicators found)\n\n"
                audit_report_content += "## Individual Rule Findings:\n\n"

                for rule, (found, snippet) in audit_results.items():
                    status_icon = "✅" if found else "❌"
                    status_text = "Found" if found else "Not Found"
                    st.markdown(f"**{status_icon} {rule}**")
                    if found:
                        st.success(f"Found! Snippet: `{snippet}`")
                    else:
                        st.warning(f"Not explicitly found. Consider adding specific statements about '{rule}'.")
                    st.markdown("---")

                    audit_report_content += f"- {status_icon} **{rule}**: {status_text}\n"
                    audit_report_content += f"  - Snippet: `{snippet}`\n\n"
                
                # --- Download Button for Audit Report (New) ---
                st.download_button(
                    label="Download Audit Report (Markdown)",
                    data=audit_report_content,
                    file_name="ai_governance_audit_report.md",
                    mime="text/markdown"
                )
    else:
        st.info("Upload a document or select a pre-loaded policy document to run the compliance audit.")

# --- AI Chat Assistant View (with RAG) ---
elif app_mode == "💬 AI Chat Assistant":
    st.header("AI Governance Chat Assistant (with RAG)")
    st.markdown("---")
    if gemini_model is None or embedding_model is None:
        st.error("AI models not initialized. Please check your API key and internet connection.")
        st.stop()
    else:
        st.info("Ask the AI Assistant questions. If you have loaded a document in the 'Document Summarizer' section, the AI will use its content for Retrieval-Augmented Generation (RAG) to provide more specific answers.")

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        # --- RAG Setup ---
        # Get the document content stored from the Summarizer page
        current_document_content = st.session_state.get('chat_document_content', '')
        
        document_chunks = []
        faiss_index = None
        
        if current_document_content.strip():
            with st.spinner("Preparing document for RAG chat..."):
                document_chunks = get_text_chunks(current_document_content)
                if document_chunks:
                    embeddings = generate_embeddings(document_chunks, embedding_model)
                    if embeddings.size > 0:
                        faiss_index = get_faiss_index(embeddings)
                        st.success(f"Document indexed for RAG ({len(document_chunks)} chunks).")
                    else:
                        st.warning("Could not generate embeddings for the document. RAG will not be active.")
                else:
                    st.warning("No valid text chunks generated from the document. RAG will not be active.")
        else:
            st.info("No document loaded for RAG. Chatbot will use general knowledge.")


        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt := st.chat_input("Ask about AI governance..."):
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "content": prompt})

            with st.spinner("Thinking..."):
                try:
                    ai_response = ""
                    if faiss_index and document_chunks:
                        # Perform RAG: retrieve relevant chunks
                        retrieved_info = retrieve_info(prompt, document_chunks, faiss_index, embedding_model, top_k=2)
                        
                        if retrieved_info:
                            context = "\n\n".join(retrieved_info)
                            rag_prompt = f"Based on the following document excerpts and your general knowledge, answer the question. If the document doesn't contain the answer, use your general knowledge. \n\nDocument Excerpts:\n{context}\n\nQuestion: {prompt}"
                        else:
                            rag_prompt = f"Question: {prompt}" # Fallback to general knowledge if no relevant info found
                            st.info("No highly relevant information found in the document for RAG. Using general knowledge.")
                    else:
                        rag_prompt = f"Question: {prompt}" # Use general knowledge if no document or index
                        if not current_document_content.strip():
                            st.info("No document loaded for RAG. Using general knowledge.")
                        else:
                            st.warning("RAG setup failed. Using general knowledge.")

                    # Generate content with Gemini
                    response = gemini_model.generate_content(rag_prompt)
                    ai_response = response.text

                    with st.chat_message("assistant"):
                        st.markdown(ai_response)
                    st.session_state.messages.append({"role": "assistant", "content": ai_response})
                except Exception as e:
                    st.error(f"Failed to get response from AI: {e}")
                    st.warning("There might be an issue with the AI service or the prompt. Please try again or rephrase your question.")
