import streamlit as st
import base64
from io import BytesIO
import pandas as pd
from classifier import RealFakeClassifier
from chatbot import AgenticLLMSystem

# Set page config
st.set_page_config(
    page_title="Real vs Fake Data Classifier + Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .stButton button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
    }
    .chat-container {
        border: 1px solid #ddd;
        border-radius: 10px;
        padding: 20px;
        height: 500px;
        overflow-y: auto;
        margin-bottom: 20px;
    }
    .user-message {
        background-color: #e6f7ff;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
        text-align: right;
    }
    .bot-message {
        background-color: #f0f0f0;
        padding: 10px;
        border-radius: 10px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'classifier' not in st.session_state:
    st.session_state.classifier = None
if 'chatbot' not in st.session_state:
    st.session_state.chatbot = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'api_key' not in st.session_state:
    st.session_state.api_key = ""

# App title
st.markdown('<h1 class="main-header">🤖 Real vs Fake Data Classifier + Chatbot</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Settings")
    
    # API Key for chatbot
    api_key = st.text_input("OpenAI API Key (optional)", type="password", 
                           help="Enter your OpenAI API key to enable the chatbot's full capabilities")
    
    if api_key:
        st.session_state.api_key = api_key
        if st.session_state.chatbot is None:
            st.session_state.chatbot = AgenticLLMSystem(api_key=api_key)
    
    st.header("Navigation")
    app_mode = st.radio("Choose App Mode", 
                       ["Real vs Fake Classifier", "Chatbot", "About"])
    
    if app_mode == "Real vs Fake Classifier":
        st.info("Generate and classify real vs fake data patterns")
    elif app_mode == "Chatbot":
        st.info("Ask questions or request creative content")
    else:
        st.info("Learn about this application")

# Main content
if app_mode == "Real vs Fake Classifier":
    st.markdown('<h2 class="sub-header">Real vs Fake Data Classification</h2>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Configuration")
        real_type = st.selectbox("Real Data Type", 
                                ["blobs", "moons", "multivariate_normal"],
                                help="Choose the pattern for real data")
        fake_type = st.selectbox("Fake Data Type",
                                ["uniform", "different_gaussian", "noise"],
                                help="Choose the pattern for fake data")
        dimensions = st.slider("Dimensions", 2, 128, 2, 
                              help="Number of features (2 for visualization)")
        samples = st.slider("Samples per Class", 100, 5000, 1000,
                           help="Number of samples for each class")
        
        if st.button("Generate and Train Models", type="primary"):
            with st.spinner("Generating data and training models..."):
                st.session_state.classifier = RealFakeClassifier(
                    dimensions=dimensions, 
                    n_samples=samples,
                    random_state=42
                )
                
                # Prepare dataset
                X, y = st.session_state.classifier.prepare_dataset(real_type, fake_type)
                
                # Train models
                st.session_state.classifier.train_models(X, y)
                
                # Evaluate models
                st.session_state.classifier.evaluate_models()
                
                st.success("Models trained successfully!")
    
    with col2:
        if st.session_state.classifier is not None:
            st.subheader("Results")
            
            # Display summary
            st.markdown(st.session_state.classifier.get_summary(), unsafe_allow_html=True)
            
            # Display plots
            st.subheader("Visualizations")
            plots = st.session_state.classifier.get_plots()
            
            if st.session_state.classifier.dimensions == 2:
                st.image(BytesIO(base64.b64decode(plots['2d']['data_distribution'])), 
                        caption="Data Distribution")
                
                for name in list(st.session_state.classifier.models.keys())[:3]:
                    if f'decision_boundary_{name}' in plots['2d']:
                        st.image(BytesIO(base64.b64decode(plots['2d'][f'decision_boundary_{name}'])), 
                                caption=f"{name} Decision Boundary")
            else:
                st.image(BytesIO(base64.b64decode(plots['high_dim']['high_dim'])), 
                        caption="High Dimensional Data Visualization")
            
            # Model comparison
            st.image(BytesIO(base64.b64decode(plots['model_comparison'])), 
                    caption="Model Performance Comparison")
            
            # Confusion matrices
            st.image(BytesIO(base64.b64decode(plots['confusion_matrices'])), 
                    caption="Confusion Matrices")
            
            # ROC curves
            st.image(BytesIO(base64.b64decode(plots['roc_curves'])), 
                    caption="ROC Curves")
        else:
            st.info("Configure the settings and click 'Generate and Train Models' to see results")

elif app_mode == "Chatbot":
    st.markdown('<h2 class="sub-header">🤖 Intelligent Chatbot</h2>', unsafe_allow_html=True)
    
    # Initialize chatbot if not already done
    if st.session_state.chatbot is None:
        st.session_state.chatbot = AgenticLLMSystem(api_key=st.session_state.api_key)
    
    # Chat container
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat history
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.markdown(f'<div class="user-message"><b>You:</b> {message["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="bot-message"><b>Bot:</b> {message["content"]}</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Chat input
    user_input = st.text_input("Type your message here...", key="chat_input")
    
    if st.button("Send") and user_input:
        # Add user message to chat history
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        # Get bot response
        with st.spinner("Thinking..."):
            response = st.session_state.chatbot.process_input(user_input)
        
        # Add bot response to chat history
        st.session_state.chat_history.append({"role": "bot", "content": response})
        
        # Rerun to update the chat display
        st.rerun()
    
    # Clear chat button
    if st.button("Clear Chat"):
        st.session_state.chat_history = []
        if st.session_state.chatbot:
            st.session_state.chatbot.conversation_history = []
        st.rerun()

else:  # About page
    st.markdown('<h2 class="sub-header">About This Application</h2>', unsafe_allow_html=True)
    
    st.write("""
    ## Real vs Fake Data Classifier + Chatbot
    
    This application combines two powerful AI capabilities:
    
    ### 1. Real vs Fake Data Classifier
    - Generates synthetic datasets with different patterns
    - Trains multiple machine learning models to distinguish between real and fake data
    - Provides comprehensive visualizations and performance metrics
    - Supports 2D (visualizable) and high-dimensional data
    
    ### 2. Intelligent Chatbot
    - Uses OpenAI's GPT models for natural language processing
    - Detects intent (factual vs creative queries)
    - Maintains conversation memory
    - Works in both API mode and local fallback mode
    
    ### How to Use
    1. **Classifier**: Select data types, dimensions, and sample size, then click "Generate and Train Models"
    2. **Chatbot**: Enter your OpenAI API key for full functionality or use the local fallback mode
    3. **Navigation**: Use the sidebar to switch between different modes
    
    ### Technical Details
    - Built with Streamlit for the web interface
    - Uses scikit-learn for machine learning models
    - Integrates with OpenAI API for chatbot capabilities
    - Generates interactive visualizations with Matplotlib and Seaborn
    """)
    
    st.info("""
    Note: For the full chatbot experience, you'll need an OpenAI API key. 
    Without it, the chatbot will use a limited local knowledge base.
    """)