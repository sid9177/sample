# chat_app.py
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

# --- Environment and LLM Setup ---
# Load environment variables from .env file
load_dotenv()

# Check for Azure OpenAI credentials
if not all(os.getenv(var) for var in [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
]):
    st.error("Azure OpenAI environment variables not set. Please configure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME.")
    st.stop() # Stop execution if credentials are missing

# Import LangChain and mcp-use components *after* checking env vars
try:
    from langchain_openai import AzureChatOpenAI
    from mcp_use import MCPAgent, MCPClient
except ImportError:
    st.error("Required libraries (langchain-openai, mcp-use) not found. Please install them: pip install langchain-openai mcp-use")
    st.stop()

# --- Configuration ---
MCP_CONFIG_FILE = "excel_mcp_config.json" # Assumes config file is in the same directory

# --- Initialization (Cached to avoid reloading on every interaction) ---

@st.cache_resource # Cache the client and agent for the session duration
def initialize_agent():
    """Initializes the MCPClient and MCPAgent."""
    print("Initializing MCP Client and Agent...") # Add print statement for debugging
    try:
        # 1. Create LLM (Azure OpenAI)
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            temperature=0.1, # Adjust temperature as needed
            # model_kwargs={"response_format": {"type": "json_object"}} # If needed for structured output
        )

        # 2. Create MCPClient from the config file
        # Ensure the path in MCP_CONFIG_FILE is correct relative to chat_app.py
        if not os.path.exists(MCP_CONFIG_FILE):
            st.error(f"MCP configuration file not found: {MCP_CONFIG_FILE}")
            print(f"Error: MCP configuration file not found at {os.path.abspath(MCP_CONFIG_FILE)}")
            return None # Return None to indicate failure

        print(f"Loading MCP config from: {os.path.abspath(MCP_CONFIG_FILE)}")
        client = MCPClient.from_config_file(MCP_CONFIG_FILE)
        print("MCP Client created.")

        # 3. Create the MCPAgent
        # Enable memory to remember conversation history
        # Increase max_steps if complex analysis requires more interactions
        agent = MCPAgent(
            llm=llm,
            client=client,
            memory_enabled=True,
            max_steps=15, # Allow more steps for analysis
            verbose=True, # Set to True for detailed agent logging in the console
            # We want the agent to know about *all* tools by default
            # Server manager might be useful if you add many more servers later
            use_server_manager=False
        )
        print("MCPAgent created.")
        return agent

    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        print(f"Error during agent initialization: {e}") # Print error to console
        import traceback
        traceback.print_exc() # Print full traceback to console
        return None

# --- Streamlit App UI ---

st.title("📊 Excel Analyzer Chat")
st.write("Chat with an AI that can analyze your Excel files using MCP.")
st.warning("⚠️ **Security Note:** The backend server currently accepts arbitrary file paths. Ensure the configured path in `excel_mcp_config.json` points to a safe directory and the server runs with appropriate permissions.", icon="🚨")


# Initialize Agent using the cached function
# This will only run once per session unless the function code changes
agent = initialize_agent()

# Stop if agent initialization failed
if agent is None:
    st.stop()

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about your Excel data... (e.g., 'Load sample_data.xlsx', 'Describe the data', 'Show sales by region')"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")

        try:
            # --- IMPORTANT: Run the agent asynchronously ---
            # Streamlit runs synchronously, but agent.run is async.
            # We use asyncio.run() to execute the async function.
            # Note: Running nested asyncio loops can be tricky in some environments,
            # but it's often necessary for integrating async libraries with Streamlit.
            print(f"Running agent with query: {prompt}") # Debug print
            response = asyncio.run(agent.run(query=prompt))
            print(f"Agent response: {response}") # Debug print
            # ----------------------------------------------

            message_placeholder.markdown(response)
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            error_msg = f"An error occurred: {e}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {e}"})
            print(f"Error during agent run: {e}") # Print error to console
            import traceback
            traceback.print_exc() # Print full traceback
