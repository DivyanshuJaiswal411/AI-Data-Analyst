import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import google.generativeai as genai
import io
import contextlib
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# -----------------------------
# 🔑 Configure Gemini API
# -----------------------------
def setup_gemini_api():
    """Setup Gemini API with error handling"""
    try:
        # Direct API key integration
        api_key = "AIzaSyCLQ5jq-gVUyNYW6eXcMqy9r3viGBU225E"
        genai.configure(api_key=api_key)
        return genai.GenerativeModel("gemini-1.5-flash")
    except Exception as e:
        st.error(f"Failed to setup Gemini API: {e}")
        return None

# -----------------------------
# 🧹 Column Name Sanitization
# -----------------------------
def sanitize_columns(df):
    """Sanitize column names for Python compatibility"""
    new_cols = []
    for col in df.columns:
        col_clean = str(col).strip().replace(" ", "_").replace("-", "_")
        col_clean = ''.join(c for c in col_clean if c.isalnum() or c == "_")
        if col_clean and not col_clean[0].isalpha():
            col_clean = "col_" + col_clean
        elif not col_clean:
            col_clean = f"unnamed_col_{len(new_cols)}"
        new_cols.append(col_clean)
    
    # Handle duplicate column names
    seen = {}
    final_cols = []
    for col in new_cols:
        if col in seen:
            seen[col] += 1
            final_cols.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            final_cols.append(col)
    
    df.columns = final_cols
    return df

# -----------------------------
# 🧹 Enhanced Data Cleaning
# -----------------------------
def clean_dataset(df):
    """Enhanced data cleaning with progress tracking"""
    original_shape = df.shape
    
    # Remove duplicates
    df = df.drop_duplicates()
    duplicates_removed = original_shape[0] - df.shape[0]
    
    # Remove columns with too many missing values
    thresh = len(df) * 0.5
    cols_before = df.shape[1]
    df = df.dropna(axis=1, thresh=thresh)
    cols_removed = cols_before - df.shape[1]
    
    # Fill missing values intelligently
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            # Use median for numeric columns
            df[col] = df[col].fillna(df[col].median())
        elif df[col].dtype == 'object':
            # Use mode for categorical columns, fallback to "Unknown"
            mode_val = df[col].mode()
            fill_val = mode_val[0] if not mode_val.empty else "Unknown"
            df[col] = df[col].fillna(fill_val)
    
    # Store cleaning stats
    cleaning_stats = {
        'duplicates_removed': duplicates_removed,
        'columns_removed': cols_removed,
        'original_shape': original_shape,
        'final_shape': df.shape
    }
    
    return df, cleaning_stats

def auto_clean_numeric_like(df):
    """Auto-detect and clean numeric-like columns"""
    df_cleaned = df.copy()
    converted_cols = []
    
    for col in df_cleaned.select_dtypes(include='object'):
        sample = df_cleaned[col].dropna().astype(str).head(20)
        if any(any(c.isdigit() for c in val) for val in sample if val):
            original_type = df_cleaned[col].dtype
            # Clean common numeric patterns
            df_cleaned[col] = (df_cleaned[col].astype(str)
                             .str.replace(",", "")
                             .str.replace("$", "")
                             .str.replace("%", "")
                             .str.replace("€", "")
                             .str.replace("£", "")
                             .str.strip())
            
            # Try to convert to numeric
            numeric_col = pd.to_numeric(df_cleaned[col], errors='ignore')
            if not numeric_col.equals(df_cleaned[col]):
                df_cleaned[col] = numeric_col
                converted_cols.append(col)
    
    return df_cleaned, converted_cols

# -----------------------------
# ⚡ Enhanced Code Execution
# -----------------------------
def run_generated_code(code, df):
    """Run AI-generated code with enhanced error handling and visualization capture"""
    # Create safe execution environment
    safe_globals = {
        '__builtins__': {
            'len': len, 'str': str, 'int': int, 'float': float, 'bool': bool,
            'list': list, 'dict': dict, 'set': set, 'tuple': tuple,
            'min': min, 'max': max, 'sum': sum, 'abs': abs, 'round': round,
            'sorted': sorted, 'reversed': reversed, 'enumerate': enumerate,
            'zip': zip, 'range': range, 'print': print
        }
    }
    
    local_vars = {
        "df": df.copy(), 
        "plt": plt, 
        "sns": sns, 
        "pd": pd,
        "px": px,
        "go": go
    }
    
    figures = []
    output_text = ""
    
    try:
        # Ensure result variable exists
        if "result" not in code:
            code += "\nresult = df"
        
        # Capture stdout
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            exec(code, safe_globals, local_vars)
        
        output_text = f.getvalue()
        result = local_vars.get("result", df)
        
        # Handle different result types
        if isinstance(result, pd.Series):
            result = result.to_frame().T
        elif not isinstance(result, (pd.DataFrame, str, int, float, list, dict)):
            result = pd.DataFrame({"Result": [str(result)]})
        
        # Capture matplotlib figures
        for fig_num in plt.get_fignums():
            fig = plt.figure(fig_num)
            figures.append(('matplotlib', fig))
        plt.close('all')
        
        return result, figures, output_text, None
        
    except Exception as e:
        error_msg = f"⚠️ Error executing code: {str(e)}"
        return error_msg, figures, output_text, str(e)

# -----------------------------
# 🤖 Enhanced AI Query Processing
# -----------------------------
def nl_to_pandas_gemini(question, df, model):
    """Convert natural language to pandas code using Gemini"""
    if not model:
        return "# Error: Gemini API not available\nresult = df.head()"
    
    # Get sample data for context
    sample_data = df.head(3).to_string()
    
    prompt = f"""
You are an expert Python data analyst. Generate clean, executable Python code to answer the user's question.

Dataset Info:
- Columns: {list(df.columns)}
- Shape: {df.shape}
- Sample data:
{sample_data}

Question: "{question}"

Requirements:
1. Use only pandas, matplotlib, seaborn, plotly.express (as px), plotly.graph_objects (as go)
2. DataFrame variable is named 'df'
3. Always assign final result to variable 'result' 
4. Use double quotes for column names
5. Handle missing values appropriately
6. For visualizations, create clear titles and labels
7. Return clean, executable code only (no markdown, no explanations)

Code:
"""
    
    try:
        response = model.generate_content(prompt)
        code = response.text.strip()
        
        # Clean the response
        if '```python' in code:
            code = code.split('```python')[1].split('```')[0]
        elif '```' in code:
            code = code.split('```')[1]
        
        return code.strip()
        
    except Exception as e:
        return f"# API Error: {e}\nresult = df.head()"

# -----------------------------
# 📊 Enhanced Dataset Profiling
# -----------------------------
def dataset_summary(df):
    """Generate comprehensive dataset summary with enhanced statistics"""
    summary = {}
    summary["rows"], summary["columns"] = df.shape
    summary["memory_usage"] = f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    
    # Column-wise analysis
    col_info = []
    for col in df.columns:
        info = {
            "Column": col,
            "Type": str(df[col].dtype),
            "Non-Null": f"{df[col].count()}/{len(df)} ({df[col].count()/len(df)*100:.1f}%)",
            "Unique": df[col].nunique(),
            "Top Value": str(df[col].mode().iloc[0]) if not df[col].mode().empty else "N/A"
        }
        
        if pd.api.types.is_numeric_dtype(df[col]):
            info.update({
                "Mean": f"{df[col].mean():.2f}",
                "Std": f"{df[col].std():.2f}",
                "Min": df[col].min(),
                "Max": df[col].max(),
                "Median": f"{df[col].median():.2f}"
            })
            
            # Outlier detection using IQR
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))]
            info["Outliers"] = f"{len(outliers)} ({len(outliers)/len(df)*100:.1f}%)"
        else:
            # For categorical columns
            info.update({
                "Mode": str(df[col].mode().iloc[0]) if not df[col].mode().empty else "N/A",
                "Most Frequent": f"{df[col].value_counts().iloc[0]} times" if len(df[col].value_counts()) > 0 else "N/A"
            })
        
        col_info.append(info)
    
    summary_df = pd.DataFrame(col_info)
    return summary, summary_df

# -----------------------------
# 🎨 Enhanced Visualization Dashboard  
# -----------------------------
def create_enhanced_dashboard(df):
    """Create an enhanced Power BI style dashboard"""
    
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    # Key Metrics
    st.subheader("📈 Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Rows", f"{len(df):,}")
    with col2:
        st.metric("Total Columns", len(df.columns))
    with col3:
        st.metric("Numeric Features", len(num_cols))
    with col4:
        st.metric("Categorical Features", len(cat_cols))
    
    # Interactive visualizations
    if len(num_cols) > 0:
        st.subheader("📊 Distribution Analysis")
        
        # Allow user to select columns for visualization
        selected_num_col = st.selectbox("Select numeric column for distribution:", num_cols)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Interactive histogram
            fig = px.histogram(df, x=selected_num_col, nbins=30, 
                             title=f"Distribution of {selected_num_col}",
                             marginal="box")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot for outlier detection
            fig = px.box(df, y=selected_num_col, 
                        title=f"Box Plot - {selected_num_col}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        if len(num_cols) > 1:
            st.subheader("🔗 Correlation Analysis")
            corr_matrix = df[num_cols].corr()
            
            fig = px.imshow(corr_matrix, 
                           text_auto=True, 
                           title="Correlation Heatmap",
                           color_continuous_scale='RdBu_r')
            st.plotly_chart(fig, use_container_width=True)
    
    # Categorical analysis
    if len(cat_cols) > 0:
        st.subheader("📋 Categorical Analysis")
        selected_cat_col = st.selectbox("Select categorical column:", cat_cols)
        
        # Top categories
        top_categories = df[selected_cat_col].value_counts().head(10)
        fig = px.bar(x=top_categories.values, 
                     y=top_categories.index,
                     orientation='h',
                     title=f"Top 10 Categories in {selected_cat_col}",
                     labels={'x': 'Count', 'y': selected_cat_col})
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 💬 Enhanced Chat Interface
# -----------------------------
def enhanced_chat_container(chat_history):
    """Enhanced chat interface with better styling and functionality"""
    
    chat_container_style = """
    <style>
    .chat-container {
        max-height: 400px;
        overflow-y: auto;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        padding: 15px;
        background-color: #fafafa;
    }
    
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 0 5px 50px;
        text-align: right;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .ai-message {
        background: white;
        color: #333;
        padding: 10px 15px;
        border-radius: 18px;
        margin: 5px 50px 5px 0;
        border-left: 4px solid #4CAF50;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .timestamp {
        font-size: 0.8em;
        color: #888;
        margin-top: 5px;
    }
    </style>
    """
    
    st.markdown(chat_container_style, unsafe_allow_html=True)
    
    chat_html = "<div class='chat-container'>"
    
    for entry in chat_history:
        timestamp = datetime.now().strftime("%H:%M")
        
        # User message
        chat_html += f"""
        <div class='user-message'>
            {entry['query']}
            <div class='timestamp'>{timestamp}</div>
        </div>
        """
        
        # AI response
        if isinstance(entry["result"], pd.DataFrame):
            if len(entry["result"]) <= 10:
                preview = entry["result"].to_html(index=False, classes='table table-striped')
            else:
                preview = entry["result"].head(5).to_html(index=False, classes='table table-striped')
                preview += f"<p><i>... and {len(entry['result']) - 5} more rows</i></p>"
            
            chat_html += f"""
            <div class='ai-message'>
                <strong>📊 Analysis Result:</strong><br>
                {preview}
                <div class='timestamp'>Generated at {timestamp}</div>
            </div>
            """
        else:
            chat_html += f"""
            <div class='ai-message'>
                <strong>🤖 AI Response:</strong><br>
                {entry['result']}
                <div class='timestamp'>{timestamp}</div>
            </div>
            """
    
    chat_html += "</div>"
    
    # Auto-scroll to bottom
    chat_html += """
    <script>
        var chatContainer = document.querySelector('.chat-container');
        if (chatContainer) {
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }
    </script>
    """
    
    components.html(chat_html, height=420, scrolling=False)

# -----------------------------
# 🚀 Main Streamlit Application
# -----------------------------
def main():
    st.set_page_config(
        page_title="AI Dataset Explorer", 
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="main-header">📊 AI-Powered Dataset Explorer</h1>', unsafe_allow_html=True)
    
    # Initialize Gemini model
    model = setup_gemini_api()
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Settings")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload Dataset", 
            type=['csv', 'xlsx', 'json'],
            help="Supported formats: CSV, Excel, JSON"
        )
        
        if uploaded_file:
            st.success(f"✅ Loaded: {uploaded_file.name}")
            
            # Display file info
            file_size = uploaded_file.size / 1024**2  # MB
            st.info(f"📁 Size: {file_size:.2f} MB")
    
    # Main content
    if uploaded_file is not None:
        # Initialize session state
        if "df_original" not in st.session_state:
            try:
                # Load data based on file type
                if uploaded_file.name.endswith('.csv'):
                    raw_df = pd.read_csv(uploaded_file)
                elif uploaded_file.name.endswith(('.xlsx', '.xls')):
                    raw_df = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.json'):
                    raw_df = pd.read_json(uploaded_file)
                
                # Process data
                with st.spinner("🔄 Processing dataset..."):
                    raw_df = sanitize_columns(raw_df)
                    cleaned_df, cleaning_stats = clean_dataset(raw_df)
                    cleaned_df, converted_cols = auto_clean_numeric_like(cleaned_df)
                
                # Store in session state
                st.session_state.df_original = cleaned_df.copy()
                st.session_state.df_current = cleaned_df.copy()
                st.session_state.chat_history = []
                st.session_state.cleaning_stats = cleaning_stats
                st.session_state.converted_cols = converted_cols
                
                # Show cleaning summary
                with st.expander("🧹 Data Cleaning Summary", expanded=False):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Duplicates Removed", cleaning_stats['duplicates_removed'])
                    with col2:
                        st.metric("Columns Removed", cleaning_stats['columns_removed'])  
                    with col3:
                        st.metric("Columns Converted", len(converted_cols))
                        
                    if converted_cols:
                        st.info(f"🔢 Auto-converted to numeric: {', '.join(converted_cols)}")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {e}")
                return
        
        df_current = st.session_state.df_current
        
        # Tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📊 Dashboard", "💬 AI Chat", "🔍 Data View"])
        
        with tab1:
            st.header("📋 Dataset Overview")
            summary, summary_df = dataset_summary(df_current)
            
            # Key metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", f"{summary['rows']:,}")
            with col2:
                st.metric("Columns", summary['columns'])
            with col3:
                st.metric("Memory Usage", summary['memory_usage'])
            
            # Column details
            st.subheader("📊 Column Analysis")
            st.dataframe(summary_df, use_container_width=True)
        
        with tab2:
            st.header("📊 Interactive Dashboard")
            create_enhanced_dashboard(df_current)
        
        with tab3:
            st.header("💬 Chat with Your Dataset")
            
            # Display chat history
            enhanced_chat_container(st.session_state.chat_history)
            
            # Query input
            col1, col2 = st.columns([4, 1])
            with col1:
                query = st.text_input(
                    "Ask me anything about your dataset:",
                    placeholder="e.g., Show me the correlation between sales and profit",
                    key="query_input"
                )
            with col2:
                send_query = st.button("🚀 Send", type="primary")
            
            # Sample questions
            with st.expander("💡 Sample Questions"):
                sample_questions = [
                    "Show me the top 10 rows with highest values",
                    "Create a correlation matrix",
                    "Find outliers in the numeric columns", 
                    "Group data by category and show average",
                    "Create a visualization showing trends"
                ]
                for q in sample_questions:
                    if st.button(q, key=f"sample_{q}"):
                        query = q
                        send_query = True
            
            # Process query
            if send_query and query.strip():
                with st.spinner("🤖 Processing your query..."):
                    # Generate and execute code
                    code = nl_to_pandas_gemini(query, df_current, model)
                    result, figures, output_text, error = run_generated_code(code, df_current)
                    
                    # Update session state
                    if error is None:
                        if isinstance(result, pd.DataFrame):
                            st.session_state.df_current = result
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            "query": query,
                            "result": result,
                            "code": code,
                            "output_text": output_text,
                            "figures": figures
                        })
                        
                        st.rerun()
                    else:
                        st.error(f"❌ {error}")
                        st.code(code, language='python')
        
        with tab4:
            st.header("🔍 Data Viewer")
            
            # Data filtering options
            col1, col2, col3 = st.columns(3)
            with col1:
                show_rows = st.selectbox("Show rows:", [10, 25, 50, 100, "All"], index=0)
            with col2:
                if st.button("🔄 Reset to Original"):
                    st.session_state.df_current = st.session_state.df_original.copy()
                    st.rerun()
            with col3:
                if st.button("📥 Download Current Data"):
                    csv = df_current.to_csv(index=False)
                    st.download_button(
                        "💾 Download CSV",
                        csv,
                        f"processed_{uploaded_file.name}",
                        "text/csv"
                    )
            
            # Display data
            if show_rows == "All":
                st.dataframe(df_current, use_container_width=True)
            else:
                st.dataframe(df_current.head(show_rows), use_container_width=True)
            
            st.info(f"Showing {min(show_rows if show_rows != 'All' else len(df_current), len(df_current))} of {len(df_current)} rows")
        
        # Generate downloadable script
        if st.session_state.chat_history:
            with st.expander("📄 Generate Analysis Script"):
                full_script = f"""# Generated Analysis Script
# Dataset: {uploaded_file.name}
# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

# Load the dataset
df = pd.read_csv('{uploaded_file.name}')

"""
                
                for i, item in enumerate(st.session_state.chat_history, 1):
                    full_script += f"""
# Query {i}: {item['query']}
{item['code']}

"""
                
                st.code(full_script, language='python')
                st.download_button(
                    "📄 Download Analysis Script",
                    full_script,
                    f"analysis_{uploaded_file.name.split('.')[0]}.py",
                    "text/plain"
                )
    
    else:
        # Welcome screen
        st.markdown("""
        ## Welcome to AI Dataset Explorer! 🎉
        
        Upload your dataset to get started with powerful AI-driven analysis:
        
        ### ✨ Features:
        - **Smart Data Cleaning**: Automatic preprocessing and column sanitization
        - **AI-Powered Queries**: Ask questions in natural language
        - **Interactive Dashboards**: Power BI style visualizations
        - **Advanced Analytics**: Correlation analysis, outlier detection, and more
        - **Code Generation**: Download Python scripts for your analysis
        
        ### 📁 Supported Formats:
        - CSV files
        - Excel files (.xlsx, .xls)  
        - JSON files
        
        **👆 Upload a file using the sidebar to begin!**
        """)

if __name__ == "__main__":
    main()