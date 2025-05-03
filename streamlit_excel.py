# chat_app.py
# ... (keep imports and initialization functions as they were in the previous corrected version) ...

# --- Streamlit App UI ---

st.title("📊 Excel Analyzer Chat (Upload)")
st.write("Upload an Excel file (.xlsx or .xls) and then ask the AI to load and analyze it.")

# File Uploader (Keep this section as is)
uploaded_file = st.file_uploader(
    "Upload your Excel file (.xlsx or .xls)", type=["xlsx", "xls"],
    key="excel_uploader",
    on_change=lambda: st.session_state.update(
        uploaded_file_name=None,
        uploaded_file_content=None,
        data_loaded_on_server=False
    )
)

# Initialize client, llm, agent via session state (Keep this section as is)
if "mcp_client" not in st.session_state:
    st.session_state.mcp_client = initialize_mcp_client()
if "llm" not in st.session_state:
    st.session_state.llm = initialize_llm()
if "agent" not in st.session_state:
    # Agent initialization logic (Keep this as is)
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
        # Error messages handled within initialize functions

# Session state for messages and file status (Keep this as is)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "uploaded_file_name" not in st.session_state:
    st.session_state.uploaded_file_name = None
if "uploaded_file_content" not in st.session_state:
    st.session_state.uploaded_file_content = None
if "data_loaded_on_server" not in st.session_state:
    st.session_state.data_loaded_on_server = False


# call_load_tool_directly function (Keep this async def as is)
async def call_load_tool_directly(client: MCPClient, file_name: str, file_content: bytes) -> tuple[bool, str | None]:
    # ... (implementation from the previous correct version) ...
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
                    st.session_state.uploaded_file_name = file_name
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

    except (ConnectionError, MCPClientError, Exception) as e:
        err_type = type(e).__name__
        err_msg = f"{err_type}: Could not communicate with MCP server for loading. Is it running? Details: {e}"
        print(f"ERROR: {err_msg}")
        traceback.print_exc()
        st.session_state.data_loaded_on_server = False
        return False, err_msg


# Display file status (Keep this section as is)
if st.session_state.uploaded_file_name:
    status = "Data loaded on server." if st.session_state.data_loaded_on_server else "Ready to load."
    st.info(f"Current file: **{st.session_state.uploaded_file_name}**. Status: {status}")
else:
    st.info("Please upload an Excel file to begin.")

# Display chat messages (Keep this section as is)
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Chat Input Logic (Corrected Part) ---
if prompt := st.chat_input("Ask about the uploaded Excel data..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    agent = get_agent()
    if agent is None:
        st.error("Agent could not be initialized. Cannot process request.")
        st.stop()

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        message_placeholder.markdown("Thinking...")
        full_response = ""

        is_load_request = ("load" in prompt.lower() and "upload" in prompt.lower()) or \
                          ("use" in prompt.lower() and "upload" in prompt.lower())

        if is_load_request:
            if st.session_state.uploaded_file_content is not None:
                message_placeholder.markdown("Loading data onto server...")
                success, response_msg = asyncio.run( # Uses asyncio.run
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
                    st.session_state.messages.append({"role": "assistant", "content": f"System Note: Successfully loaded '{st.session_state.uploaded_file_name}'. You can now analyze it."})
            else:
                full_response = "No file has been uploaded yet. Please upload a file first."
                message_placeholder.warning(full_response)

        elif not st.session_state.data_loaded_on_server:
             full_response = "Data from the uploaded file has not been loaded onto the server yet. Please ask me to 'load the uploaded file' first."
             message_placeholder.warning(full_response)

        else:
            message_placeholder.markdown("Analyzing...")
            try:
                contextual_prompt = f"Using the data from the loaded file identified as '{st.session_state.uploaded_file_name}', please {prompt}"
                print(f"DEBUG: Running agent with contextual prompt: {contextual_prompt}")

                # --- FIX: Use asyncio.run() for the agent call ---
                response = asyncio.run(agent.run(query=contextual_prompt))
                # --- END FIX ---

                print(f"DEBUG: Agent response: {response}")
                full_response = response
                message_placeholder.markdown(full_response)

            except (MCPClientError, ConnectionError, Exception) as e:
                error_msg = f"An error occurred during analysis: {type(e).__name__} - {e}"
                message_placeholder.error(error_msg)
                full_response = f"Error: {e}"
                print(f"Error during agent run: {e}")
                traceback.print_exc()

        # Add response to history (Keep this as is)
        if not full_response.startswith("System Note:"):
             st.session_state.messages.append({"role": "assistant", "content": full_response})
