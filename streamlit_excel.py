# chat_app.py
import streamlit as st
import asyncio
import os
from dotenv import load_dotenv
import traceback

# --- Environment and LLM Setup ---
load_dotenv()

if not all(os.getenv(var) for var in [
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_API_VERSION",
    "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"
]):
    st.error("Azure OpenAI environment variables not set.")
    st.stop()

try:
    from langchain_openai import AzureChatOpenAI
    from mcp_use import MCPAgent, MCPClient
    from mcp_use.exceptions import MCPClientError # Import base MCP error if needed
except ImportError:
    st.error("Required libraries not found. Please install them: pip install streamlit langchain-openai mcp-use python-dotenv pandas openpyxl matplotlib pillow")
    st.stop()

# --- Configuration ---
MCP_CONFIG_FILE = "excel_mcp_config.json"

# --- Initialization ---

@st.cache_resource
def initialize_agent():
    """Initializes the MCPClient and MCPAgent."""
    print("Initializing MCP Client and Agent...")
    try:
        llm = AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
            temperature=0.1,
        )

        config_path = os.path.abspath(MCP_CONFIG_FILE)
        print(f"DEBUG: Loading MCP config from: {config_path}")
        if not os.path.exists(config_path):
            st.error(f"MCP configuration file not found: {config_path}")
            print(f"Error: MCP configuration file not found at {config_path}")
            return None

        client = MCPClient.from_config_file(config_path)
        print("MCP Client created.")

        agent = MCPAgent(
            llm=llm,
            client=client,
            memory_enabled=True,
            max_steps=15,
            verbose=True,
            use_server_manager=False
        )
        print("MCPAgent created.")
        return agent

    except Exception as e:
        st.error(f"Failed to initialize agent: {e}")
        print(f"Error during agent initialization: {e}")
        traceback.print_exc()
        return None

# --- Streamlit App UI ---

st.title("📊 Excel Analyzer Chat (Upload)")
st.write("Upload an Excel file and chat with an AI to analyze it using MCP.")

# --- File Upload ---
uploaded_file = st.file_uploader(
    "Choose an Excel file (.xlsx, .xls)",
    type=["xlsx", "xls"],
    key="excel_uploader" # Add key for stability
)

agent = initialize_agent()

# --- Session State Management ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "loaded_file_info" not in st.session_state:
    # Stores {'name': original_filename, 'content': bytes}
    st.session_state.loaded_file_info = None
if "current_upload_key" not in st.session_state:
    st.session_state.current_upload_key = None # To track if file changed

# --- Handle File Upload ---
if uploaded_file is not None:
    # Check if this is a new file upload
    new_upload_key = f"{uploaded_file.name}_{uploaded_file.size}"
    if new_upload_key != st.session_state.current_upload_key:
        st.session_state.current_.
*   **Fixed Data Key:** Stores the DataFrame in `loaded_data[UPLOADED_DATA_KEY]`.
*   **Other Tools:** Changed the `file_path` parameter to `data_key: str` with a default value of `UPLOADED_DATA_KEY`. The descriptions are updated to mention the key.
*   **Helper `_get_loaded_df`:** Updated to accept `data_key`.
*   **Error Messages:** Updated to refer to `data_key` or `file_name`.

**Step 2: Modify the Streamlit App (`chat_app.py`)**

This requires more significant changes to handle the upload and trigger the new `load_excel_content` tool.

```python
# chat_app.py
import streamlit as st
import asyncio
import os
import io # Import io
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
    from mcp_use import MCPAgent, MCPClient, session as mcp_session
    from mcp_use.exceptions import MCPConnectionError
except ImportError:
    st.error("Required libraries not found. Please install them.")
    st.stop()

# --- Configuration ---
MCP_CONFIG_FILE = "excel_mcp_config.json"
UPLOADED_DATA_KEY = "uploaded_data" # The key used by the server

# --- Initialization ---

@st.cache_resource # Cache the client, agent initialization might need rethink
def initialize_mcp_client():
    """Initializes only the MCPClient."""
    print("Initializing MCP Client...")
    try:
        if not os.path.exists(MCP_CONFIG_FILE):
            st.error(f"MCP configuration file not found: {MCP_CONFIG_FILE}")
            print(f"Error: MCP configuration file not found at {os.path.abspath(MCP_CONFIG_FILE)}")
            return None

        print(f"Loading MCP config from: {os.path.abspath(MCP_CONFIG_FILE)}")
        client = MCPClient.from_config_file(MCP_CONFIG_FILE)
        print("MCP Client created.")
        return client
    except Exception as e:
        st.error(f"Failed to initialize MCP Client: {e}")
        print(f"Error during MCP Client initialization: {e}")
        traceback.print_exc()
        return None

@st.cache_resource # Cache the LLM
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

# --- Session State Management ---
# Use session state to store upload status and potentially the MCP session if needed across runs
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = initialize_mcp_client()

if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()

if "agent" not in st.session_state:
    if st.session_state.mcp_client and st.session_state.llm:
        st.session_state.agent = MCPAgent(
            llm=st.session_state.llm,
            client=st.session_state.mcp_client,
            memory_enabled=True,
            max_steps=15,
            verbose=True,
            use_server_manager=False # Keep simple for now
        )
        print("MCPAgent created and stored in session state.")
    else:
        st.session_state.agent = None

if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "data_loaded_on_server" not in st.session_state:
    st.session_state.data_loaded_on_server = False
if "data_key" not in st.session_state:
    st.session_state.data_key = None # Store the key ('uploaded_data') once loaded


# --- Helper Function to Call Load Tool ---
async def call_load_tool(client: MCPClient, file_name: str, file_content: bytes):
    """Calls the load_excel_content tool on the server."""
    try:
        # Ensure session exists - create if necessary
        # MCPClient now handles session creation implicitly via agent or directly
        # We might need to get/create the session manually if calling tool outside agent
        print(f"Calling 'load_excel_content' for file: {file_name}")

        # If using MCPAgent, it handles sessions. If calling client directly:
        session_name = "excel_analyzer" # Must match the key in your JSON config
        try:
            session = client.get_session(sessionupload_key = new_upload_key
        with st.spinner(f"Reading '{uploaded_file.name}'..."):
            try:
                file_content = uploaded_file.getvalue() # Read content into bytes
                st.session_state.loaded_file_info = {
                    "name": uploaded_file.name,
                    "content": file_content
                }
                st.success(f"File '{uploaded_file.name}' ready for analysis. You can now ask the agent to load and analyze it.")
                # Clear chat history when a new file is uploaded/processed
                st.session_state.messages = []
                # Auto-load the data via the agent? Optional, could be a button too.
                # For now, let the user explicitly ask to load it.
            except Exception as e:
                st.error(f"Error reading uploaded file: {e}")
                st.session_state.loaded_file_info = None # Clear if error
                st.session_state.current_upload_key = None
else:
    # If no file is uploaded in the current run, reset state
    st.session_state.loaded_file_info = None
    st.session_state.current_upload_key = None

# Display info about the loaded file
if st.session_state.loaded_file_info:
    st.info(f"Current file: **{st.session_state.loaded_file_info['name']}**. Ask the agent to 'load the uploaded file' to begin analysis.")
else:
    st.info("Please upload an Excel file to start.")

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

    # Check if agent is initialized
    if agent is None:
        st.error("Agent could not be initialized. Cannot process request.")
        st.stop()

    # Process the prompt with the agent
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        full_response = ""

        # --- Agent Interaction ---
        # Check if the user prompt likely implies loading the uploaded file
        should_load_file = ("load" in prompt.lower() and "upload" in prompt.lower()) or \
                           ("use" in prompt.lower() and "upload" in prompt.lower()) or \
                           ("analyze" in prompt.lower() and "upload" in prompt.lower() and not any("describe" in m["content"].lower() for m in st.session_state.messages if m["role"]=="assistant")) # crude check if already loaded

        try:
            print(f"DEBUG: Running agent with query: {prompt}")

            if should_load_file and st.session_state.loaded_file_info:
                # If the prompt is likely a load request, construct the tool call directly
                # This bypasses the LLM for the *initial load* but uses it for subsequent steps
                print(f"DEBUG: Detected load request. Calling load_excel_data directly.")
                load_args = {
                    "file_name": st.session_state.loaded_file_info["name"],
                    "file_content": st.session_state.loaded_file_info["content"],
                    # Add sheet_name if you want to parse it from prompt or use default
                }
                # Directly call the tool via the client/session
                # We need to ensure the server is connected first
                if not agent.client.active_sessions.get("excel_analyzer"):
                     print("DEBUG: Connecting to excel_analyzer for direct load call.")
                     await agent.client.create_session("excel_analyzer") # Assumes server name is 'excel_analyzer'

                session = agent.client.get_session("excel_analyzer")
                load_result = await session.call_tool("load_excel_data", load_args)

                # Parse the result to display to the user
                if load_result and not load_result.isError:
                    content = load_result.content[0]
                    if isinstance(content, TextContent):
                         response_data = json.loads(content.text)
                         response = response_data.get("message", "File loaded successfully.")
                         # Add message about successful load to history for LLM context
                         st.session_state.messages.append({"role": "assistant", "content": f"System Note: File '{response_data.get('file_name')}' loaded. Columns: {response_data.get('columns')}, Rows: {response_data.get('rows')}."})
                    else:
                         response = "File loaded, but received unexpected response format from tool."
                else:
                     response = f"Error loading file: {load_result.content[0].text if load_result else 'Unknown error'}"

                print(f"DEBUG: Direct load result: {response}")

            else:
                # For other requests, let the agent handle it
                # Include the filename in the context for the LLM if a file is loaded
                contextual_prompt = prompt
                if st.session_state.loaded_file_info:
                    fname = st.session_state.loaded_file_info['name']
                    contextual_prompt = f"Using the loaded data identified as '{fname}', please {prompt}"
                    print(f_name)
            print("Using existing session.")
        except ValueError:
            print("Creating new session...")
            session = await client.create_session(session_name)
            print("New session created.")

        # Now call the tool using the session's connector directly or via agent.run
        # Using agent.run is simpler if agent knows the tool
        # Here we call directly for clarity of this specific step
        tool_result = await session.call_tool(
            "load_excel_content",
            {
                "file_name": file_name,
                "file_content": file_content, # Pass bytes directly
                # sheet_name defaults to 0 on server
            },
        )
        print(f"Tool 'load_excel_content' result: {tool_result}")

        # Assuming result is like {'message': '...', 'data_key': 'uploaded_data', ...}
        # Extract the data_key if needed, although we know it's fixed
        data_key = tool_result.get("data_key", UPLOADED_DATA_KEY)
        return True, data_key, tool_result.get("message", "Data loaded.")

    except MCPConnectionError as e:
        st.error(f"Connection Error: Failed to connect/communicate with the MCP server. Is it running? Details: {e}")
        print(f"MCPConnectionError: {e}")
        traceback.print_exc()
        return False, None, f"Connection Error: Could not reach MCP server."
    except Exception as e:
        st.error(f"Error loading data via MCP: {e}")
        print(f"DEBUG: Modified prompt with filename: {contextual_prompt}")

                # Run the agent asynchronously
                response = await agent.run(query=contextual_prompt)
                print(f"DEBUG: Agent response: {response}")


            full_response = response
            message_placeholder.markdown(full_response)

        except (MCPClientError, Exception) as e: # Catch specific MCP errors if available
            error_msg = f"An error occurred: {e}"
            message_placeholder.error(error_msg)
            full_response = f"Error: {e}"
            print(f"Error during agent run: {e}")
            traceback.print_exc()

        # Add assistant response (or error) to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response})


# --- Cleanup ---
# Streamlit doesn't have a clean shutdown hook easily accessible
# The MCPAgent/Client should ideally be cleaned up, but closing sessions
# here might be tricky with caching. For simple cases, rely on process exit.
# If needed, you could add a "Disconnect" button to call agent.close().
