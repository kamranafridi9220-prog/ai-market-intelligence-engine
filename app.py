import streamlit as st
import pandas as pd
import plotly.express as px
from openai import OpenAI

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(
    page_title="AI Market Intelligence Engine",
    page_icon="📊",
    layout="wide"
)

# ----------------------------
# OpenAI Client
# ----------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# ----------------------------
# Header
# ----------------------------
st.markdown(
    """
    <h1 style='text-align: center;'>AI Market Intelligence Engine</h1>
    <p style='text-align: center; font-size:18px; color: gray;'>
    Turning Data into Market Decisions with AI
    </p>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    A portfolio-level AI system for market analysis, customer pattern discovery,
    business risk detection, and decision intelligence.
    """
)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Control Panel")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

# ----------------------------
# Main App
# ----------------------------
if uploaded_file is not None:

    # Load Data
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.success("Dataset uploaded successfully.")

    # Column Detection
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    meaningful_numeric_columns = [
        col for col in numeric_columns
        if "id" not in col.lower()
    ]

    meaningful_categorical_columns = [
        col for col in categorical_columns
        if df[col].nunique() <= 30
    ]

    # Sidebar Filter
    st.sidebar.subheader("Optional Data Filter")

    use_filter = st.sidebar.checkbox("Apply filter to AI analysis only")

    filtered_df = df.copy()

    if use_filter:
        selected_column = st.sidebar.selectbox(
            "Filter by column",
            df.columns
        )

        unique_values = df[selected_column].dropna().unique()

        selected_value = st.sidebar.selectbox(
            "Filter value",
            unique_values
        )

        filtered_df = df[df[selected_column] == selected_value]
        st.sidebar.info(f"Filtered rows: {filtered_df.shape[0]}")

    else:
        st.sidebar.info("Charts are using full dataset.")

    # ----------------------------
    # KPI Dashboard
    # ----------------------------
    st.markdown("## 📊 Executive Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Records", df.shape[0])
    col2.metric("AI Analysis Rows", filtered_df.shape[0])
    col3.metric("Total Features", df.shape[1])
    col4.metric("Missing Values", int(df.isnull().sum().sum()))

    st.markdown("---")

    # ----------------------------
    # Dataset Preview
    # ----------------------------
    with st.expander("Dataset Preview", expanded=False):
        st.dataframe(df.head(20), use_container_width=True)

    # ----------------------------
    # Market Intelligence Visuals
    # ----------------------------
    st.markdown("## 📈 Market Intelligence Visuals")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        if meaningful_categorical_columns:
            pie_column = st.selectbox(
                "Select category for market distribution",
                meaningful_categorical_columns
            )

            pie_data = df[pie_column].value_counts().reset_index()
            pie_data.columns = [pie_column, "Count"]

            fig_pie = px.pie(
                pie_data,
                names=pie_column,
                values="Count",
                title=f"Market Distribution by {pie_column}",
                hole=0.35
            )

            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.warning("No suitable categorical column available for pie chart.")

    with chart_col2:
        if meaningful_categorical_columns:
            bar_column = st.selectbox(
                "Select category for top segment analysis",
                meaningful_categorical_columns,
                key="bar_column"
            )

            bar_data = df[bar_column].value_counts().head(10).reset_index()
            bar_data.columns = [bar_column, "Count"]

            fig_bar = px.bar(
                bar_data,
                x=bar_column,
                y="Count",
                title=f"Top Segments by {bar_column}",
                text="Count"
            )

            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.warning("No suitable categorical column available for bar chart.")

    chart_col3, chart_col4 = st.columns(2)

    with chart_col3:
        if meaningful_numeric_columns:
            hist_column = st.selectbox(
                "Select numeric metric for distribution",
                meaningful_numeric_columns
            )

            fig_hist = px.histogram(
                df,
                x=hist_column,
                title=f"Distribution of {hist_column}",
                nbins=30
            )

            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.warning("No suitable numeric column available for histogram.")

    with chart_col4:
        if len(meaningful_numeric_columns) >= 2:
            x_axis = st.selectbox(
                "Select X-axis metric",
                meaningful_numeric_columns,
                key="scatter_x"
            )

            y_axis = st.selectbox(
                "Select Y-axis metric",
                meaningful_numeric_columns,
                key="scatter_y"
            )

            fig_scatter = px.scatter(
                df,
                x=x_axis,
                y=y_axis,
                title=f"{x_axis} vs {y_axis}",
                opacity=0.7
            )

            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("Need at least two meaningful numeric columns for scatter chart.")

    st.markdown("---")

    # ----------------------------
    # Segment Performance Analysis
    # ----------------------------
    st.markdown("## 🎯 Segment Performance Analysis")

    if meaningful_categorical_columns and meaningful_numeric_columns:
        segment_col = st.selectbox(
            "Select segment column",
            meaningful_categorical_columns,
            key="segment_col"
        )

        metric_col = st.selectbox(
            "Select business metric",
            meaningful_numeric_columns,
            key="metric_col"
        )

        segment_data = (
            df.groupby(segment_col, as_index=False)[metric_col]
            .mean()
            .sort_values(metric_col, ascending=False)
            .head(10)
        )

        fig_segment = px.bar(
            segment_data,
            x=segment_col,
            y=metric_col,
            title=f"Average {metric_col} by {segment_col}",
            text=metric_col
        )

        st.plotly_chart(fig_segment, use_container_width=True)

    else:
        st.info("Need one categorical and one numeric column for segment performance chart.")

    st.markdown("---")

    # ----------------------------
    # Strategic Decision Recommendations
    # ----------------------------
    st.markdown("## 🧭 Strategic Decision Recommendations")

    if meaningful_categorical_columns:

        recommendation_col = st.selectbox(
            "Select segment for strategic analysis",
            meaningful_categorical_columns,
            key="recommendation_col"
        )

        top_segment = df[recommendation_col].value_counts().idxmax()
        top_count = df[recommendation_col].value_counts().max()

        st.success(
            f"Top Segment Identified: {top_segment} ({top_count} records)"
        )

        st.markdown("### Executive Insights")

        st.write(
            f"The segment **{top_segment}** currently represents the strongest group in the selected category and may offer the highest opportunity for growth, customer engagement, and strategic focus."
        )

        st.markdown("### Recommended Business Actions")

        st.write(
            "- Focus marketing resources on the highest-performing segment."
        )

        st.write(
            "- Analyse customer behaviour within this segment."
        )

        st.write(
            "- Develop retention and upsell strategies."
        )

        st.write(
            "- Compare this segment against revenue and profitability metrics."
        )

        st.markdown("### Risk Assessment")

        st.warning(
            "Heavy dependence on one segment may create concentration risk. Diversification should also be considered."
        )

    st.markdown("---")
    
    # ----------------------------
    # Dataset Summary
    # ----------------------------
    with st.expander("Dataset Summary", expanded=False):
        st.dataframe(df.describe(include="all"), use_container_width=True)

    # ----------------------------
    # AI Section
    # ----------------------------
    st.markdown("## 🤖 AI Intelligence Engine")

    user_question = st.text_area(
        "Ask a business question",
        placeholder="Example: What are the key market trends, risks, and growth opportunities in this dataset?"
    )

    analysis_mode = st.radio(
        "Choose analysis mode",
        ["Multi-Agent Market Intelligence", "Decision Mode"],
        horizontal=True
    )

    if st.button("Generate AI Market Intelligence"):

        if user_question.strip() == "":
            st.warning("Please type a question first.")

        else:
            with st.spinner("Generating AI market intelligence..."):

                sample_data = filtered_df.head(25).to_string()

                if analysis_mode == "Multi-Agent Market Intelligence":
                    prompt = f"""
You are a team of AI experts working together.

There are 3 roles:
1. Market Analyst
2. Risk Analyst
3. Strategy Advisor

Analyze the dataset below and answer the user's question.

Dataset sample:
{sample_data}

User question:
{user_question}

Each role must respond separately in this structure:

--- Market Analyst ---
- Key Trends
- Customer Behavior
- Revenue Insights

--- Risk Analyst ---
- Key Risks
- Warning Signals
- Data Limitations

--- Strategy Advisor ---
- Growth Opportunities
- Strategic Recommendations
- Next Best Actions

Keep answers clear, practical, and business-focused.
"""
                else:
                    prompt = f"""
You are an AI Decision Intelligence System.

Analyze the dataset sample below and answer the user's question by making a clear business decision.

Dataset sample:
{sample_data}

User question:
{user_question}

Respond in this exact structure:

--- Executive Decision ---
- Final Decision:
- Decision Rationale:
- Confidence Score: Give a percentage from 0% to 100%
- Priority Level: Low / Medium / High / Critical

--- Business Impact ---
- Revenue Impact:
- Customer Impact:
- Market Impact:
- Operational Impact:

--- Risk Check ---
- Main Risk:
- Warning Signal:
- Data Limitation:
- Mitigation:

--- Next Best Action ---
- Immediate Action:
- 30-Day Action:
- 90-Day Action:
- KPI to Track:

Keep the response practical, commercial, and suitable for senior business users.
"""

                response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a professional AI market intelligence and decision intelligence system."
                        },
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=0.3
                )

                ai_output = response.choices[0].message.content

                if analysis_mode == "Multi-Agent Market Intelligence":
                    st.markdown("## 📋 Executive Market Intelligence Report")
                else:
                    st.markdown("## 🧠 Decision Intelligence Report")

                sections = ai_output.split("---")

                for section in sections:
                    section = section.strip()
                    if section:
                        lines = section.split("\n")
                        title = lines[0].strip()
                        content = "\n".join(lines[1:])

                        with st.expander(title, expanded=True):
                            st.markdown(content)

else:
    st.info("Upload a CSV or Excel file from the sidebar to begin.")
