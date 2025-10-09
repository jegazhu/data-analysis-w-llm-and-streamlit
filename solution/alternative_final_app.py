import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime
import traceback
from io import BytesIO

st.set_page_config(
    page_title="Ask Your CSV",
    page_icon="📊",
    layout="wide"
)

# Initialize OpenAI client
try:
    client = openai.OpenAI(api_key=st.secrets['OPENAI_API_KEY'])
except Exception as e:
    st.error(f"⚠️ Failed to initialize OpenAI: {str(e)}")
    st.stop()

# Helper function for export
def export_conversation():
    """Export conversation history as HTML (works like PDF when printed)"""
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
                h3 {{ color: #888; margin-top: 20px; }}
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

# Session state initialization
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "data_summary" not in st.session_state:
    st.session_state.data_summary = None

st.title("📊 Ask Your CSV")
st.markdown("Upload your data and ask questions in plain English!")

# Sidebar for file upload
with st.sidebar:
    st.header("📁 Data Upload")
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            # Try reading with default encoding
            df = pd.read_csv(uploaded_file)
            st.session_state.df = df
            
            # Create data summary for token optimization
            try:
                st.session_state.data_summary = {
                    "shape": df.shape,
                    "columns": df.columns.tolist(),
                    "dtypes": df.dtypes.to_dict(),
                    "sample": df.head(3).to_dict(),
                    "stats": df.describe().to_dict() if not df.empty else {}
                }
            except Exception as e:
                st.warning(f"Could not generate full summary: {str(e)}")
                st.session_state.data_summary = {
                    "shape": df.shape,
                    "columns": df.columns.tolist()
                }
            
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
                        st.metric("Memory Usage", f"{df.memory_usage().sum() / 1024:.1f} KB")
                        st.metric("Missing Values", df.isnull().sum().sum())
                    except Exception:
                        st.metric("Memory Usage", "N/A")
                        st.metric("Missing Values", "N/A")
        
        except UnicodeDecodeError:
            # Try alternative encodings
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding='latin-1')
                st.session_state.df = df
                st.warning("⚠️ File read with 'latin-1' encoding. Some characters may appear differently.")
            except Exception as e:
                st.error(f"❌ Encoding error: {str(e)}")
                st.info("💡 Try saving your CSV with UTF-8 encoding.")
        
        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded file is empty.")
        
        except pd.errors.ParserError as e:
            st.error(f"❌ Unable to parse CSV: {str(e)}")
            st.info("💡 Check if your file is properly formatted.")
        
        except Exception as e:
            st.error(f"❌ Error reading file: {str(e)}")
            st.info("Please make sure your file is a valid CSV format.")
    else:
        st.info("👆 Upload a CSV file to start analyzing!")
    
    # Export options (only show if there are messages)
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
            st.sidebar.info("💡 Tip: Open the HTML file and print to PDF for best results")

# Main chat interface
if st.session_state.df is not None:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "figure" in msg:
                try:
                    st.pyplot(msg["figure"])
                except Exception:
                    pass  # Figure no longer available
    
    # Chat input
    user_input = st.chat_input("Ask a question about your data")
    
    if user_input:
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Prepare data context with token optimization
        df = st.session_state.df
        if len(df) > 100:
            data_context = f"""
Dataset shape: {st.session_state.data_summary['shape']}
Columns: {', '.join(st.session_state.data_summary['columns'])}
Data types: {st.session_state.data_summary.get('dtypes', {})}
Sample rows: {st.session_state.data_summary.get('sample', {})}
Basic statistics: {st.session_state.data_summary.get('stats', {})}
"""
        else:
            data_context = f"Full dataset:\n{df.to_string()}"
        
        # Enhanced system prompt
        system_prompt = f"""You are a helpful data analyst assistant. The user has uploaded a CSV file with the following information:

{data_context}

The data is loaded in a pandas DataFrame called df.

Guidelines:
- Answer the user's question clearly and concisely
- If the question requires analysis, write Python code using pandas, matplotlib, or seaborn
- For visualizations, always use plt.figure() before plotting and include plt.tight_layout()
- Always validate data before operations (check for nulls, data types, etc.)
- If you can't answer due to data limitations, explain why
- Keep responses focused on the data and question asked
- Summarize your findings, insights, and any relevant statistics or visual trends
- Focus on delivering the results and what they mean, not on how to get them
- If a chart or visualization would help, display the chart in the response using matplotlib or seaborn
- If a user asks for a specific visualization, display the chart in the response using matplotlib or seaborn

When writing code:
- Import statements are already done (pandas as pd, matplotlib.pyplot as plt, seaborn as sns)
- The dataframe is available as 'df'
- For plots, use plt.figure(figsize=(10, 6)) for better display
- Always add titles and labels to plots
"""
        
        # Generate response
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            
            with st.spinner("Analyzing your data..."):
                try:
                    # Get conversation history for context
                    messages = [{"role": "system", "content": system_prompt}]
                    
                    # Include last 3 exchanges for context
                    for msg in st.session_state.messages[-6:]:
                        content = msg["content"]
                        if len(content) > 500:
                            content = content[:500] + "..."
                        messages.append({"role": msg["role"], "content": content})
                    
                    # API call
                    response = client.chat.completions.create(
                        model="gpt-4-turbo-preview",
                        messages=messages,
                        temperature=0.1,
                        max_tokens=1500
                    )
                    reply = response.choices[0].message.content
                    message_placeholder.markdown(reply)
                    
                    # Try to execute any code in the response
                    if "```python" in reply:
                        code_blocks = reply.split("```python")
                        for i in range(1, len(code_blocks)):
                            code = code_blocks[i].split("```")[0]
                            
                            try:
                                with warnings.catch_warnings(record=True) as w:
                                    warnings.simplefilter("always")
                                    
                                    plt.figure(figsize=(10, 6))
                                    
                                    exec_globals = {
                                        "df": df,
                                        "pd": pd,
                                        "plt": plt,
                                        "sns": sns,
                                        "st": st
                                    }
                                    
                                    exec(code.strip(), exec_globals)
                                    
                                    if w:
                                        for warning in w:
                                            st.info(f"Note: {warning.message}")
                                    
                                    fig = plt.gcf()
                                    if fig.get_axes():
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
                                    
                                    plt.close()
                            
                            except NameError as e:
                                st.error(f"❌ Column or variable error: {str(e)}")
                                st.info("💡 This might mean a column name is misspelled or doesn't exist.")
                                with st.expander("Show code"):
                                    st.code(code, language="python")
                            
                            except KeyError as e:
                                st.error(f"❌ Column not found: {str(e)}")
                                st.info(f"💡 Available columns: {', '.join(df.columns.tolist())}")
                                with st.expander("Show code"):
                                    st.code(code, language="python")
                            
                            except TypeError as e:
                                st.error(f"❌ Data type error: {str(e)}")
                                st.info("💡 This often happens when trying to plot non-numeric data.")
                                with st.expander("Show code"):
                                    st.code(code, language="python")
                            
                            except ValueError as e:
                                st.error(f"❌ Value error: {str(e)}")
                                st.info("💡 Check if your data values match the operation requirements.")
                                with st.expander("Show code"):
                                    st.code(code, language="python")
                            
                            except Exception as e:
                                st.error(f"❌ Execution error ({type(e).__name__}): {str(e)}")
                                st.info("💡 Try rephrasing your question or check your data format.")
                                with st.expander("Show code"):
                                    st.code(code, language="python")
                                with st.expander("Show detailed error"):
                                    st.code(traceback.format_exc())
                    else:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": reply
                        })
                
                except openai.APIError as e:
                    st.error(f"❌ OpenAI API Error: {str(e)}")
                    st.info("Please check your API key and quota, then try again.")
                
                except openai.RateLimitError:
                    st.error("❌ Rate limit reached. Please wait a moment and try again.")
                
                except Exception as e:
                    st.error(f"❌ Error generating response: {str(e)}")
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

# Footer with tips
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 12px;'>
💡 Tip: Be specific with your questions for better results | 
🔒 Your data stays private and is not stored
</div>
""", unsafe_allow_html=True)
