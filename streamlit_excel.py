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

# Check for Azure OpenAI credentials (keep this check)
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
    # REMOVED specific MCPConnectionError import
    # Instead, we'll catch broader errors or rely on general Exception
    from mcp_use.client import MCPClientError # Import a general client error if available
    from mcp.types import TextContent # To check response type from direct call
except ImportError as e:
    st.error(f"Required libraries not found. Please install them. Error: {e}")
    st.stop()

# --- Configuration ---
MCP_CONFIG_FILE = "excel_mcp_config.json" # Assumes config file is in the same directory
MCP_SERVER_NAME = "excel_analyzer" # Must match the key in your excel_mcp_config.json
# UPLOADED_DATA_KEY_SERVER = "uploaded_data" # The key the server uses internally (optional for client)

# --- Initialization (Cached) ---

@st.cache_resource
def initialize_mcp_client():
    """Initializes only the MCPClient."""
    print("Initializing MCP Client...")
    try:
        config_path = os.path.abspath(MCP_CONFIG_FILE)
        print(f"DEBUG: Loading MCP config from: {config_path}")
        if not os.path.exists(config_path):
            st.error(f"MCP configuration file not found: {config_path}")
            print(f"Error: MCP configuration file not found at {config_path}")
            return None

        client = MCPClient.from_config_file(config_path)
        print("MCP Client created.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize MCP Client: {e}")
        print(f"Error during MCP Client initialization: {e}")
        traceback.print_exc()
        return None

@st.cache_resource
def initialize_llm():
    """Initializes the AzureChatOpenAI LLM."""
    print("Initializing LLM...")
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            temperature=0.1,
        )
        print("LLM Initialized.")
        return llm
    except Exception as e:
        st.error(f"Failed to initialize LLM: {e}")
        print(f"Error during LLM initialization: {e}")
        traceback.print_exc()
        return None

# --- Agent Initialization ---
def get_agent():
    if "agent" not in st.session_state:
        client = st.session_state.mcp_client
        llm = st.session_state.llm
        if client and llm:
            print("Creating MCPAgent...")
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
            st.error("MCP Client or LLM failed to initialize. Agent cannot be created.")
    return st.session_state.agent

# --- Session State Management ---
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
        try:
            session = client.get_session(MCP_SERVER_NAME)
            print(f"DEBUG: Using existing session for '{MCP_SERVER_NAME}'.")
        except ValueError:
            print(f"DEBUG: Creating new session for '{MCP_SERVER_NAME}'.")
            session = await client.create_session(MCP_SERVER_NAME, auto_initialize=True)
            print(f"DEBUG: New session created and initialized.")

        if not session:
             # This case might not be strictly necessary if create_session raises on failure, but good for safety
             raise ConnectionError(f"Could not get or create session for '{MCP_SERVER_NAME}'")

        load_args = {
            "file_name": file_name,
            "file_content": file_content,
        }
        print(f"DEBUG: Calling 'load_excel_content' tool with file_name: {file_name}")
        load_result = await session.call_tool("load_excel_content", load_args)
        print(f"DEBUG: Raw result from 'load_excel_content': {load_result}")

        if load_result and not load_result.isError:
            content = load_result.content[0] if load_result.content else None
            if isinstance(content, TextContent):
                try:
                    response_data = json.loads(content.text)
                    message = response_data.get("message", f"File '{file_name}' loaded successfully.")
                    print(f"DEBUG: Successfully parsed load response: {message}")
                    st.session_state.data_loaded_on_server = True
                    st.session_state.uploaded_file_name = file_name # Update the canonical loaded name
                    return True, message
                except json.JSONDecodeError:
                     err_msg = "File loaded, but response from server was not valid JSON."
                     print(f"WARN: {err_msg} Raw: {content.text}")
                     return False, err_msg
            else:
                err_msg = f"File loaded, but received unexpected content type: {type(content)}"
                print(f"WARN: {err_msg}")
                return False, err_msg
        else:
            error_text = "Unknown error"
            if load_result and load_result.content and isinstance(load_result.content[0], TextContent):
                error_text = load_result.content[0].text
            err_msg = f"MCP server returned error during load: {error_text}"
            print(f"ERROR: {err_msg}")
            st.session_state.data_loaded_on_server = False
            return False, err_msg

    # --- CATCH BROADER ERRORS ---
    # Catching ConnectionError for network/subprocess issues
    # Catching MCPClientError if that's the intended general error from mcp-use client operations
    # Catching base Exception as a fallback
    except (ConnectionError, MCPClientError, Exception) as e:
        err_type = type(e).__name__
        err_msg = f"{err_type}: Could not communicate with MCP server for loading. Is it running? Details: {e}"
        print(f"ERROR: {err_msg}")
        traceback.print_exc()
        st.session_state.data_loaded_on_server = False
        return False, err_msg
    # --- END CATCH BROADER ERRORS ---

# --- Streamlit App UI ---

st.title("📊 Excel Analyzer Chat (Upload)")
st.write("Upload an Excel file (.xlsx or .xls) and then ask the AI to load and analyze it.")

# File Uploader
uploaded_file = st.file_uploader(
    "Upload your Excel file (.xlsx or .xls)", type=["xlsx", "xls"],
    key="excel_uploader",
    on_change=lambda: st.session_state.update(
        uploaded_file_name=None,
        uploaded_file_content=None,
        data_loaded_on_server=False
    )
)

# Store uploaded file content if a new file is uploaded
if uploaded_file is not None and uploaded_file.name != st.session_state.get("uploaded_file_name"):
     print(f"DEBUG: New file uploaded: {uploaded_file.name}")
     st.session_state.uploaded_file_name = uploaded_file.name # Store only the name initially
     st.session_state.uploaded_file_content = uploaded_file.getvalue() # Store the content
     st.session_state.data_loaded_on_server = False # Mark as not loaded on server yet
     st.info(f"File **'{uploaded_file.name}'** ready. Ask the chat to 'load the uploaded file'.")
elif st.session_state.uploaded_file_name:
    status = "Data loaded on server." if st.session_state.data_loaded_on_server else "Ready to load."
    st.info(f"Current file: **{st.session_state.uploaded_file_name}**. Status: {status}")
else:
    st.info("Please upload an Excel file to begin.")


# --- Chat Interface ---

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("Ask about the uploaded Excel data..."):
    # Add user message to history and display
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if agent is available
    agent = get_agent() # Ensure agent is created if client/llm are ready
    if agent is None:
        st.error("Agent could not be initialized. Cannot process request.")
        st.stop()

    # Process the prompt
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        full_response = ""

        # --- Logic to handle load vs analysis ---
        is_load_request = ("load" in prompt.lower() and "upload" in prompt.lower()) or \
                          ("use" in prompt.lower() and "upload" in prompt.lower())

        if is_load_request:
            if st.session_state.uploaded_file_content is not None:
                message_placeholder.markdown("Loading data onto server...")
                # Call the load tool directly using the helper
                success, response_msg = asyncio.run(
                    call_load_tool_directly(
                        st.session_state.mcp_client, # Pass the MCPClient instance
                        st.session_state.uploaded_file_name,
                        st.session_state.uploaded_file_content,
                    )
                )
                full_response = response_msg or "Load operation completed."
                if not success:
                    message_placeholder.error(full_response)
                else:
                    message_placeholder.markdown(full_response)
                    # Add system note to history for LLM context AFTER successful load
                    st.session_state.messages.append({"role": "assistant", "content": f"System Note: Successfully loaded '{st.session_state.uploaded_file_name}'. You can now analyze it."})
            else:
                full_response = "No file has been uploaded yet. Please upload a file first."
                message_placeholder.warning(full_response)

        elif not st.session_state.data_loaded_on_server:
             # If asking for analysis but data isn't loaded
             full_response = "Data from the uploaded file has not been loaded onto the server yet. Please ask me to 'load the uploaded file' first."
             message_placeholder.warning(full_response)

        else:
            # Assume it's an analysis request - run the agent
            message_placeholder.markdown("Analyzing...")
            try:
                # Add context about the loaded file for the agent
                # IMPORTANT: Use the file name stored in session state *after* successful loading
                contextual_prompt = f"Using the data from the loaded file identified as '{st.session_state.uploaded_file_name}', please {prompt}"
                print(f"DEBUG: Running agent with contextual prompt: {contextual_prompt}")

                # Run the agent asynchronously
                # Make sure agent initialization happened correctly before this point
                if agent:
                     response = await agent.run(query=contextual_prompt)
                     print(f"DEBUG: Agent response: {response}")
                     full_response = response
                else:
                     full_response = "Error: Agent is not available."
                     print("ERROR: Agent was None when trying to run.")


                message_placeholder.markdown(full_response)

            except (MCPClientError, ConnectionError, Exception) as e: # Catch broader errors
                error_msg = f"An error occurred during analysis: {type(e).__name__} - {e}"
                message_placeholder.error(error_msg)
                full_response = f"Error: {e}"
                print(f"Error during agent run: {e}")
                traceback.print_exc()

        # Add the final response to chat history only if it's not a system note
        if not full_response.startswith("System Note:"):
             st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Cleanup ---
# Still tricky in Streamlit, best effort is process exit.
