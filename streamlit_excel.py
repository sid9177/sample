# chat_app.py
import streamlit as st
import asyncio
import os
import io
import json # Import json
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
    st.error("Azure OpenAI environment variables not set. Please configure AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION, and AZURE_OPENAI_CHAT_DEPLOYMENT_NAME.")
    st.stop()

# Import LangChain and mcp-use components
try:
    from langchain_openai import AzureChatOpenAI
    from mcp_use import MCPAgent, MCPClient
    from mcp_use.exceptions import MCPConnectionError # Base error for connection issues
    from mcp_use.client import MCPClientError # General client errors
    from mcp.types import TextContent # To check response type from direct call
except ImportError:
    st.error("Required libraries (streamlit, langchain-openai, mcp-use, python-dotenv, pandas, openpyxl, matplotlib, pillow) not found. Please install them.")
    st.stop()

# --- Configuration ---
MCP_CONFIG_FILE = "excel_mcp_config.json" # Assumes config file is in the same directory
MCP_SERVER_NAME = "excel_analyzer" # Must match the key in your excel_mcp_config.json
UPLOADED_DATA_KEY_SERVER = "uploaded_data" # The key the server uses internally (optional for client)

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
# We initialize the agent only once needed and store in session state
def get_agent():
    if "agent" not in st.session_state:
        client = st.session_state.mcp_client
        llm = st.session_state.llm
        if client and llm:
            print("Creating MCPAgent...")
            st.session_state.agent = MCPAgent(
                llm=llm,
                client=client,
                memory_enabled=True, # Important for chat context
                max_steps=15,
                verbose=True, # Set True for console debugging
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
    session = None # Initialize session to None
    try:
        # Get or create the specific session for the excel server
        try:
            session = client.get_session(MCP_SERVER_NAME)
            print(f"DEBUG: Using existing session for '{MCP_SERVER_NAME}'.")
        except ValueError:
            print(f"DEBUG: Creating new session for '{MCP_SERVER_NAME}'.")
            session = await client.create_session(MCP_SERVER_NAME, auto_initialize=True) # Ensure it's initialized
            print(f"DEBUG: New session created and initialized.")

        if not session:
            raise MCPConnectionError(f"Could not get or create session for '{MCP_SERVER_NAME}'")

        load_args = {
            "file_name": file_name,
            "file_content": file_content,
            # sheet_name defaults to 0 on the server
        }
        print(f"DEBUG: Calling 'load_excel_content' tool with file_name: {file_name}")
        # Note: session.call_tool returns CallToolResult, not just content
        load_result = await session.call_tool("load_excel_content", load_args)
        print(f"DEBUG: Raw result from 'load_excel_content': {load_result}")

        if load_result and not load_result.isError:
            content = load_result.content[0] if load_result.content else None
            if isinstance(content, TextContent):
                try:
                    response_data = json.loads(content.text)
                    message = response_data.get("message", f"File '{file_name}' loaded successfully.")
                    print(f"DEBUG: Successfully parsed load response: {message}")
                    # Store confirmation in session state
                    st.session_state.data_loaded_on_server = True
                    st.session_state.uploaded_file_name = file_name # Track the loaded file name
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
            # Handle MCP error result
            error_text = "Unknown error"
            if load_result and load_result.content and isinstance(load_result.content[0], TextContent):
                error_text = load_result.content[0].text
            err_msg = f"MCP server returned error during load: {error_text}"
            print(f"ERROR: {err_msg}")
            st.session_state.data_loaded_on_server = False # Ensure status is False on error
            return False, err_msg

    except MCPConnectionError as e:
        err_msg = f"Connection Error: Could not communicate with MCP server for loading. Details: {e}"
        print(f"ERROR: {err_msg}")
        traceback.print_exc()
        st.session_state.data_loaded_on_server = False
        return False, err_msg
    except Exception as e:
        err_msg = f"Unexpected error calling load tool: {type(e).__name__} - {e}"
        print(f"ERROR: {err_msg}")
        traceback.print_exc()
        st.session_state.data_loaded_on_server = False
        return False, err_msg

# --- Streamlit App UI ---

st.title("📊 Excel Analyzer Chat (Upload)")
st.write("Upload an Excel file (.xlsx or .xls) and then ask the AI to load and analyze it.")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Upload Excel File",
    type=["xlsx", "xls"],
    key="excel_uploader",
    # Clear previous state if a new file is uploaded
    on_change=lambda: st.session_state.update(
        uploaded_file_name=None,
        uploaded_file_content=None,
        data_loaded_on_server=False
    )
)

# Store uploaded file content if a new file is uploaded
if uploaded_file is not None and uploaded_file.name != st.session_state.uploaded_file_name:
     print(f"DEBUG: New file uploaded: {uploaded_file.name}")
     st.session_state.uploaded_file_name = uploaded_file.name
     st.session_state.uploaded_file_content = uploaded_file.getvalue()
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
        # Simple check: does the prompt ask to load the uploaded file?
        is_load_request = ("load" in prompt.lower() and "upload" in prompt.lower()) or \
                          ("use" in prompt.lower() and "upload" in prompt.lower())

        if is_load_request:
            if st.session_state.uploaded_file_content is not None:
                message_placeholder.markdown("Loading data onto server...")
                # Call the load tool directly using the helper
                success, response_msg = asyncio.run(
                    call_load_tool_directly(
                        st.session_state.mcp_client,
                        st.session_state.uploaded_file_name,
                        st.session_state.uploaded_file_content,
                    )
                )
                full_response = response_msg or "Load operation completed."
                if not success:
                    message_placeholder.error(full_response)
                else:
                    message_placeholder.markdown(full_response)
                    # Add system note to history for LLM context
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
                contextual_prompt = f"Using the data from the loaded file identified as '{st.session_state.uploaded_file_name}', please {prompt}"
                print(f"DEBUG: Running agent with contextual prompt: {contextual_prompt}")

                # Run the agent asynchronously
                response = await agent.run(query=contextual_prompt)
                print(f"DEBUG: Agent response: {response}")
                full_response = response
                message_placeholder.markdown(full_response)

            except (MCPClientError, MCPConnectionError, Exception) as e:
                error_msg = f"An error occurred during analysis: {type(e).__name__} - {e}"
                message_placeholder.error(error_msg)
                full_response = f"Error: {e}"
                print(f"Error during agent run: {e}")
                traceback.print_exc()

        # Add the final response to chat history ONLY IF IT WASN'T A SYSTEM NOTE
        # (or handle system notes differently if needed)
        if not full_response.startswith("System Note:"):
             st.session_state.messages.append({"role": "assistant", "content": full_response})

# --- Cleanup ---
# Session cleanup is harder in Streamlit. The mcp_use client might need
# explicit closing if the app were long-running in a different framework.
# For Streamlit, rely on process termination or add a manual disconnect button.
