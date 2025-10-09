import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import traceback
import sys
from io import BytesIO, StringIO

st.set_page_config(
    page_title="Ask Your CSV",
    page_icon="📊",
    layout="wide"
)

# ==================== ERROR HANDLING UTILITIES ====================

class DataAnalysisError(Exception):
    """Custom exception for data analysis errors"""
    pass

def safe_api_init():
    """Safely initialize OpenAI client with error handling"""
    try:
        if 'OPENAI_API_KEY' not in st.secrets:
            st.error("⚠️ OpenAI API key not found in secrets. Please configure it in your Streamlit secrets.")
            st.stop()
        return openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {str(e)}")
        st.stop()

def safe_file_read(uploaded_file):
    """Safely read CSV file with comprehensive error handling"""
    try:
        # Try reading with default settings
        df = pd.read_csv(uploaded_file)
        return df, None
    except pd.errors.EmptyDataError:
        return None, "The uploaded file is empty. Please upload a file with data."
    except pd.errors.ParserError as e:
        return None, f"Unable to parse the CSV file. It may be corrupted or incorrectly formatted. Error: {str(e)}"
    except UnicodeDecodeError:
        # Try different encodings
        try:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, encoding='latin-1')
            return df, "Note: File was read using 'latin-1' encoding."
        except Exception:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='iso-8859-1')
                return df, "Note: File was read using 'iso-8859-1' encoding."
            except Exception as e:
                return None, f"Encoding error: Unable to read file with standard encodings. Error: {str(e)}"
    except MemoryError:
        return None, "The file is too large to process. Please try a smaller file or a subset of your data."
    except Exception as e:
        return None, f"Unexpected error reading file: {str(e)}"

def safe_data_summary(df):
    """Safely create data summary with error handling"""
    try:
        summary = {
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.to_dict().items()},
            "sample": df.head(3).to_dict(),
        }
        
        # Safely get statistics
        try:
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                summary["stats"] = numeric_df.describe().to_dict()
            else:
                summary["stats"] = {}
        except Exception:
            summary["stats"] = {}
        
        return summary, None
    except Exception as e:
        return None, f"Error creating data summary: {str(e)}"

def execute_code_safely(code, df):
    """
    Safely execute Python code with comprehensive error handling
    Returns: (success, result_message, figure, error_details)
    """
    if not code or not code.strip():
        return False, "No code to execute", None, None
    
    # Create a string buffer to capture print statements
    output_buffer = StringIO()
    old_stdout = sys.stdout
    
    try:
        # Redirect stdout to capture prints
        sys.stdout = output_buffer
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create figure for potential plots
            plt.figure(figsize=(10, 6))
            
            # Create controlled execution environment
            exec_globals = {
                "df": df.copy(),  # Use a copy to prevent accidental modifications
                "pd": pd,
                "plt": plt,
                "sns": sns,
                "st": st,
                "__builtins__": __builtins__
            }
            
            # Execute the code
            exec(code.strip(), exec_globals)
            
            # Get any printed output
            printed_output = output_buffer.getvalue()
            
            # Collect warnings
            warning_messages = []
            if w:
                for warning in w:
                    warning_messages.append(str(warning.message))
            
            # Check if a plot was created
            fig = plt.gcf()
            has_plot = len(fig.get_axes()) > 0
            
            # Prepare result message
            result_parts = []
            if printed_output:
                result_parts.append(f"Output:\n{printed_output}")
            if warning_messages:
                result_parts.append(f"Warnings:\n" + "\n".join(warning_messages))
            
            result_message = "\n\n".join(result_parts) if result_parts else None
            
            return True, result_message, fig if has_plot else None, None
            
    except SyntaxError as e:
        error_detail = f"Syntax Error on line {e.lineno}: {e.msg}"
        return False, None, None, {
            "type": "SyntaxError",
            "message": error_detail,
            "suggestion": "The generated code has a syntax error. Try rephrasing your question."
        }
    
    except NameError as e:
        return False, None, None, {
            "type": "NameError",
            "message": str(e),
            "suggestion": "A column name might be misspelled or doesn't exist in your dataset. Check your column names."
        }
    
    except KeyError as e:
        return False, None, None, {
            "type": "KeyError",
            "message": str(e),
            "suggestion": f"The column {e} doesn't exist. Available columns: {', '.join(df.columns.tolist())}"
        }
    
    except TypeError as e:
        return False, None, None, {
            "type": "TypeError",
            "message": str(e),
            "suggestion": "This often happens when trying to perform numeric operations on non-numeric data, or vice versa."
        }
    
    except ValueError as e:
        return False, None, None, {
            "type": "ValueError",
            "message": str(e),
            "suggestion": "There might be an issue with data types or values. Check if your data matches the operation requirements."
        }
    
    except MemoryError:
        return False, None, None, {
            "type": "MemoryError",
            "message": "Not enough memory to complete this operation",
            "suggestion": "Try working with a subset of your data or simplify the operation."
        }
    
    except Exception as e:
        # Catch-all for unexpected errors
        error_trace = traceback.format_exc()
        return False, None, None, {
            "type": type(e).__name__,
            "message": str(e),
            "suggestion": "An unexpected error occurred. Try rephrasing your question or check your data format.",
            "trace": error_trace
        }
    
    finally:
        # Always restore stdout and close plot
        sys.stdout = old_stdout
        plt.close('all')

# ==================== EXPORT FUNCTIONALITY ====================

def export_conversation():
    """Export conversation history as HTML"""
    if not st.session_state.messages:
        return None
    
    try:
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                .question {{ background-color: #f0f0f0; padding: 10px; border-radius: 5px; margin: 10px 0; }}
                .answer {{ padding: 10px; margin: 10px 0; }}
                .metadata {{ color: #999; font-size: 14px; }}
                code {{ background-color: #f5f5f5; padding: 2px 4px; border-radius: 3px; }}
                pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 5px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <h1>Data Analysis Report</h1>
            <p class="metadata">Generated on: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
        """
        
        if st.session_state.df is not None:
            html_content += f"""
            <h2>Dataset Information</h2>
            <ul>
                <li>Total Rows: {st.session_state.df.shape[0]}</li>
                <li>Total Columns: {st.session_state.df.shape[1]}</li>
                <li>Column Names: {', '.join(st.session_state.df.columns)}</li>
            </ul>
            """
        
        html_content += "<h2>Analysis Conversation</h2>"
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                html_content += f'<div class="question"><strong>Question:</strong> {msg["content"]}</div>'
            else:
                content = msg["content"].replace("```python", "<pre><code>").replace("```", "</code></pre>")
                html_content += f'<div class="answer"><strong>Analysis:</strong><br>{content}</div>'
                if "figure" in msg:
                    html_content += '<p><em>[Visualization generated - see application for details]</em></p>'
        
        html_content += """
        </body>
        </html>
        """
        return html_content
    except Exception as e:
        st.error(f"Error generating export: {str(e)}")
        return None

# ==================== SESSION STATE INITIALIZATION ====================

if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None
if "client" not in st.session_state:
    st.session_state.client = safe_api_init()

# ==================== MAIN APP ====================

st.title("📊 Ask Your CSV")
st.markdown("Upload your data and ask questions in plain English!")

# ==================== SIDEBAR ====================

with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        df, error = safe_file_read(uploaded_file)
        
        if error:
            st.error(f"❌ {error}")
            if "encoding" in error.lower():
                st.info("💡 Try saving your CSV with UTF-8 encoding or check for special characters.")
        elif df is not None:
            st.session_state.df = df
            
            # Create data summary
            summary, summary_error = safe_data_summary(df)
            if summary_error:
                st.warning(f"⚠️ {summary_error}")
                st.session_state.data_summary = None
            else:
                st.session_state.data_summary = summary
            
            st.success(f"✅ Loaded {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Data preview
            with st.expander("Preview Data"):
                st.dataframe(df.head())
            
            # Basic stats
            with st.expander("Data Summary"):
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Rows", df.shape[0])
                    st.metric("Total Columns", df.shape[1])
                with col2:
                    try:
                        memory_kb = df.memory_usage().sum() / 1024
                        st.metric("Memory Usage", f"{memory_kb:.1f} KB")
                    except Exception:
                        st.metric("Memory Usage", "N/A")
                    
                    try:
                        missing = df.isnull().sum().sum()
                        st.metric("Missing Values", missing)
                    except Exception:
                        st.metric("Missing Values", "N/A")
    else:
        st.info("👆 Upload a CSV file to start analyzing!")
    
    # Export options
    if st.session_state.messages:
        st.markdown("---")
        st.header("💾 Export Options")
        if st.sidebar.button("Generate Report"):
            export_html = export_conversation()
            if export_html:
                st.sidebar.download_button(
                    label="📥 Download Report (HTML)",
                    data=export_html,
                    file_name=f"data_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.html",
                    mime="text/html"
                )
                st.sidebar.success("Report ready for download!")
            else:
                st.sidebar.error("Failed to generate report")

# ==================== CHAT INTERFACE ====================

if st.session_state.df is not None:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "figure" in msg and msg["figure"] is not None:
                try:
                    st.pyplot(msg["figure"])
                except Exception:
                    st.info("(Visualization no longer available)")
    
    # Chat input
    user_input = st.chat_input("Ask a question about your data")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Prepare data context
        df = st.session_state.df
        
        try:
            if len(df) > 100:
                data_context = f"""
Dataset shape: {st.session_state.data_summary['shape']}
Columns: {', '.join(st.session_state.data_summary['columns'])}
Data types: {st.session_state.data_summary['dtypes']}
Sample rows: {st.session_state.data_summary['sample']}
Basic statistics: {st.session_state.data_summary['stats']}
"""
            else:
                data_context = f"Full dataset:\n{df.to_string()}"
        except Exception as e:
            data_context = f"Dataset with {df.shape[0]} rows and {df.shape[1]} columns. Columns: {', '.join(df.columns.tolist())}"
        
        # System prompt
        system_prompt = f"""You are a helpful data analyst assistant. The user has uploaded a CSV file with the following information:

{data_context}

The data is loaded in a pandas DataFrame called df.

Guidelines:
- Answer the user's question clearly and concisely
- If the question requires analysis, write Python code using pandas, matplotlib, or seaborn
- For visualizations, always use plt.figure() before plotting and include plt.tight_layout()
- Always validate data before operations (check for nulls, data types, etc.)
- Keep responses focused on the data and question asked
- Summarize your findings, insights, and any relevant statistics or visual trends
- If a chart or visualization would help, create it using matplotlib or seaborn

When writing code:
- Import statements are already done (pandas as pd, matplotlib.pyplot as plt, seaborn as sns)
- The dataframe is available as 'df'
- For plots, use plt.figure(figsize=(10, 6)) for better display
- Always add titles and labels to plots
- Handle potential errors gracefully (check for null values, correct data types, etc.)
"""
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Analyzing your data..."):
                try:
                    # Prepare messages for API
                    messages = [{"role": "system", "content": system_prompt}]
                    
                    # Include conversation history (last 6 messages)
                    for msg in st.session_state.messages[-6:]:
                        content = msg["content"]
                        if len(content) > 500:
                            content = content[:500] + "..."
                        messages.append({"role": msg["role"], "content": content})
                    
                    # API call with error handling
                    try:
                        response = st.session_state.client.chat.completions.create(
                            model="gpt-4-turbo-preview",
                            messages=messages,
                            temperature=0.1,
                            max_tokens=1500
                        )
                        reply = response.choices[0].message.content
                    except openai.APIError as e:
                        st.error(f"❌ OpenAI API Error: {str(e)}")
                        st.info("Please check your API key and quota, then try again.")
                        st.stop()
                    except openai.RateLimitError:
                        st.error("❌ Rate limit reached. Please wait a moment and try again.")
                        st.stop()
                    except Exception as e:
                        st.error(f"❌ Error communicating with OpenAI: {str(e)}")
                        st.stop()
                    
                    message_placeholder.markdown(reply)
                    
                    # Execute code if present
                    if "```python" in reply:
                        code_blocks = reply.split("```python")
                        
                        for i in range(1, len(code_blocks)):
                            code = code_blocks[i].split("```")[0]
                            
                            success, output, fig, error_details = execute_code_safely(code, df)
                            
                            if success:
                                # Show any output
                                if output:
                                    st.info(output)
                                
                                # Show plot
                                if fig:
                                    st.pyplot(fig)
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": reply,
                                        "figure": fig
                                    })
                                else:
                                    st.session_state.messages.append({
                                        "role": "assistant",
                                        "content": reply
                                    })
                            else:
                                # Handle execution error
                                if error_details:
                                    st.error(f"❌ {error_details['type']}: {error_details['message']}")
                                    st.info(f"💡 {error_details['suggestion']}")
                                    
                                    with st.expander("Show code that failed"):
                                        st.code(code, language="python")
                                    
                                    if "trace" in error_details:
                                        with st.expander("Show detailed error trace"):
                                            st.code(error_details['trace'])
                                
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": reply
                                })
                    else:
                        # No code in response
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reply
                        })
                
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    st.info("Please try again or rephrase your question.")
                    with st.expander("Show error details"):
                        st.code(traceback.format_exc())

else:
    # No data uploaded state
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.info("👈 Please upload a CSV file to start")
        
        st.markdown("### 💡 Example questions you can ask:")
        st.markdown("""
        - What are the main trends in my data?
        - Show me a correlation matrix
        - Create a bar chart of the top 10 categories
        - What's the average value by month?
        - Are there any outliers in the price column?
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
💡 Tip: Be specific with your questions for better results | 
🔒 Your data stays private and is not stored
</div>
""", unsafe_allow_html=True)
