# chat_app.py
import streamlit as st
import asyncio
import os
import io
import json
from dotenv import load_dotenv
import traceback

# --- Environment and LLM Setup ---
load_dotenv()

# Check for Azure OpenAI credentials
if not all(os.getenv(var) for var in [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
]):
    st.error("Azure OpenAI environment variables not set.")
    st.stop()

# Import LangChain and mcp-use components
try:
    from langchain_openai import AzureChatOpenAI
    # Use mcp_use.client directly now
    from mcp_use import MCPAgent, MCPClient
    # --- FIX: Remove MCPClientError import ---
    # from mcp_use.client import MCPClientError # REMOVE THIS LINE
    # --- END FIX ---
    from mcp.types import TextContent # To check response type from direct call
except ImportError as e:
    st.error(f"Required libraries not found. Please install them. Error: {e}")
    st.stop()

# --- Configuration ---
MCP_CONFIG_FILE = "excel_mcp_config.json"
MCP_SERVER_NAME = "excel_analyzer"

# --- Initialization (Cached) ---

@st.cache_resource
def initialize_mcp_client():
    # ... (keep implementation as before) ...
    print("Initializing MCP Client...")
    try:
        config_path = os.path.abspath(MCP_CONFIG_FILE)
        print(f"DEBUG: Loading MCP config from: {config_path}")
        if not os.path.exists(config_path):
            st.error(f"MCP configuration file not found: {config_path}")
            print(f"Error: MCP configuration file not found at {config_path}")
            return None

        client = MCPClient.from_config_file(config_path)
        print("MCP Client created successfully.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize MCP Client: {e}")
        print(f"Error during MCP Client initialization: {e}")
        traceback.print_exc()
        return None

@st.cache_resource
def initialize_llm():
    # ... (keep implementation as before) ...
    print("Initializing LLM...")
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            temperature=0.1,
        )
        print("LLM Initialized successfully.")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        print(f"Error during LLM initialization: {e}")
        traceback.print_exc()
        return None

# --- Agent Initialization ---
def get_agent():
    # ... (keep implementation as before) ...
    if "agent" not in st.session_state:
        client = st.session_state.get("mcp_client")
        llm = st.session_state.get("llm")
        if client and llm:
            print("Creating MCPAgent instance...")
            st.session_state.agent = MCPAgent(
                llm=llm,
                client=client,
                memory_enabled=True,
                max_steps=15,
                verbose=True,
                use_server_manager=False
            )
            print("MCPAgent created and stored in session state.")
        else:
            st.session_state.agent = None
            print("Agent creation skipped: MCP Client or LLM not initialized.")
    return st.session_state.agent

# --- Session State Management ---
# ... (keep implementation as before) ...
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = initialize_mcp_client()
if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = None
if "data_loaded_on_server" not in st.session_state:
    st.session_state.data_loaded_on_server = False


# --- Helper Function to Call Load Tool ---
async def call_load_tool_directly(client: MCPClient, file_name: str, file_content: bytes) -> tuple[bool, str | None]:
    """Calls the load_excel_content tool on the server directly."""
    session = None
    try:
        # Get or create the specific session for the excel server
        try:
            session = client.get_session(MCP_SERVER_NAME)
            print(f"DEBUG: Using existing session for '{MCP_SERVER_NAME}'.")
        except ValueError:
            print(f"DEBUG: Creating new session for '{MCP_SERVER_NAME}'.")
            session = await client.create_session(MCP_SERVER_NAME, auto_initialize=True)
            print(f"DEBUG: New session created and initialized for '{MCP_SERVER_NAME}'.")

        if not session:
             raise ConnectionError(f"Could not get or create session for '{MCP_SERVER_NAME}'") # Use standard ConnectionError

        load_args = {
            "file_name": file_name,
            "file_content": file_content,
        }
        print(f"DEBUG: Calling MCP tool 'load_excel_content' with file_name: {file_name}")
        load_result = await session.call_tool("load_excel_content", load_args)
        print(f"DEBUG: Raw result from 'load_excel_content': {load_result}")

        # --- Handle Result (same as before) ---
        if load_result and not load_result.isError:
            content = load_result.content[0] if load_result.content else None
            if isinstance(content, TextContent):
                try:
                    response_data = json.loads(content.text)
                    message = response_data.get("message", f"File '{file_name}' loaded successfully.")
                    print(f"DEBUG: Successfully parsed load response: {message}")
                    st.session_state.data_loaded_on_server = True
                    st.session_state.uploaded_file_name = file_name
                    return True, message
                except json.JSONDecodeError:
                     err_msg = "File loaded, but response from server was not valid JSON."
                     print(f"WARN: {err_msg} Raw: {content.text}")
                     st.session_state.data_loaded_on_server = False
                     return False, err_msg
            else:
                err_msg = f"File loaded, but received unexpected content type: {type(content)}"
                print(f"WARN: {err_msg}")
                st.session_state.data_loaded_on_server = False
                return False, err_msg
        else:
            error_text = "Unknown error"
            if load_result and load_result.content and isinstance(load_result.content[0], TextContent):
                error_text = load_result.content[0].text
            err_msg = f"MCP server returned error during load: {error_text}"
            print(f"ERROR: {err_msg}")
            st.session_state.data_loaded_on_server = False
            return False, err_msg
        # --- End Handle Result ---

    # --- FIX: Adjusted Exception Handling ---
    except (ConnectionError, Exception) as e: # Catch standard ConnectionError and base Exception
        err_type = type(e).__name__
        # Adjust message slightly as we don't know if it's specifically a client error
        err_msg = f"{err_type}: Error communicating with MCP server during load. Details: {e}"
        print(f"ERROR: {err_msg}")
        traceback.print_exc()
        st.session_state.data_loaded_on_server = False
        return False, err_msg
    # --- END FIX ---

# --- Streamlit App UI ---
# ... (UI setup, file uploader, message display - keep as before) ...
st.title("📊 Excel Analyzer Chat (Upload)")
st.write("Upload an Excel file (.xlsx or .xls), ask the AI to load it, then ask for analysis.")

# File Uploader
uploaded_file = st.file_uploader(
    "Upload your Excel file",
    type=["xlsx", "xls"],
    key="excel_file_widget", # Unique key for this widget
    # Reset relevant state when a new file is chosen via the widget
    on_change=lambda: st.session_state.update({
        "uploaded_file_name": None,
        "uploaded_file_content": None,
        "data_loaded_on_server": False,
        "messages": [] # Optionally clear messages on new upload
        })
)

# Process the uploaded file *after* the widget interaction
if uploaded_file is not None:
    if uploaded_file.name != st.session_state.get("processed_upload_name"):
        print(f"DEBUG: Processing newly uploaded file: {uploaded_file.name}")
        st.session_state.uploaded_file_name = uploaded_file.name
        st.session_state.uploaded_file_content = uploaded_file.getvalue()
        st.session_state.data_loaded_on_server = False # New file, needs loading
        st.session_state.processed_upload_name = uploaded_file.name # Mark as processed for this run
        st.info(f"File **'{uploaded_file.name}'** ready. Ask the chat to 'load the uploaded file'.")

# Display status based on session state
elif st.session_state.get("uploaded_file_name"):
    status = "Data loaded on server." if st.session_state.data_loaded_on_server else "Ready to load (ask the chat)."
    st.info(f"Current file: **{st.session_state.uploaded_file_name}**. Status: {status}")
else:
    st.info("Please upload an Excel file to begin.")

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# --- Chat Input Logic ---
if prompt := st.chat_input("Ask about the uploaded Excel data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = get_agent()
    if agent is None or st.session_state.mcp_client is None:
        st.error("Agent or MCP Client could not be initialized. Cannot process request.")
        st.stop()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        full_response = ""

        is_load_request = ("load" in prompt.lower() and ("upload" in prompt.lower() or "file" in prompt.lower()))

        if is_load_request:
            # ... (logic for direct load call - keep as before) ...
            if st.session_state.uploaded_file_content is not None:
                message_placeholder.markdown("Loading data onto server...")
                success, response_msg = asyncio.run(
                    call_load_tool_directly(
                        st.session_state.mcp_client,
                        st.session_state.uploaded_file_name,
                        st.session_state.uploaded_file_content,
                    )
                )
                full_response = response_msg or "Load operation finished."
                if not success:
                    message_placeholder.error(full_response)
                else:
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": f"System Note: Successfully loaded '{st.session_state.uploaded_file_name}'. You can now analyze it."})
            else:
                full_response = "No file uploaded or file content is missing. Please upload a file first."
                message_placeholder.warning(full_response)

        elif not st.session_state.data_loaded_on_server:
             # ... (logic for asking user to load first - keep as before) ...
             full_response = "The data hasn't been loaded onto the server yet. Please ask me to 'load the uploaded file' first."
             message_placeholder.warning(full_response)

        else:
            # This is an analysis request - run the agent
            message_placeholder.markdown("Analyzing...")
            try:
                contextual_prompt = f"Using the data from the loaded file identified as '{st.session_state.uploaded_file_name}', please {prompt}"
                print(f"DEBUG: Running agent with contextual prompt: {contextual_prompt}")

                response = asyncio.run(agent.run(query=contextual_prompt))

                print(f"DEBUG: Agent response: {response}")
                full_response = response
                message_placeholder.markdown(full_response)

            # --- FIX: Adjusted Exception Handling for agent run ---
            except (ConnectionError, Exception) as e: # Catch broader errors
                error_msg = f"An error occurred during analysis: {type(e).__name__} - {e}"
                message_placeholder.error(error_msg)
                full_response = f"Error during analysis: {e}"
                print(f"Error during agent run: {e}")
                traceback.print_exc()
            # --- END FIX ---

        # Add response to history (Keep this as is)
        if not full_response.startswith("System Note:"):
             st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- End of File ---
