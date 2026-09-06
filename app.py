import streamlit as st
from rag_pipeline import answer_query

st.set_page_config(page_title="BIS Chatbot", layout="wide")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Sidebar clear chat button
if st.sidebar.button("🗑️ Clear Chat"):
    st.session_state["messages"] = []

st.title("💬 BIS AI-Assistant")

# Sidebar file upload
uploaded_file = st.sidebar.file_uploader("📎 Upload a file", type=["pdf", "docx", "txt", "csv", "png", "jpg", "jpeg"])
if uploaded_file is not None:
    st.sidebar.success(f"File uploaded: {uploaded_file.name}")

# Sidebar download chat history
if st.sidebar.download_button(
    label="⬇️ Download Chat",
    data="\n".join([f"{msg['role']}: {msg['content']}" for msg in st.session_state["messages"]]),
    file_name="chat_history.txt",
    mime="text/plain"
):
    st.sidebar.success("Chat history downloaded!")



# Display all past messages
for msg in st.session_state["messages"]:
    if msg["role"] == "user":
        st.write("👤 You:", msg["content"])
    else:
        st.write("🤖 Bot:", msg["content"])


# Use Streamlit's chat_input (auto-clears after sending)
user_input = st.chat_input("Type your message...")

if user_input:
    st.session_state["messages"].append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        response = answer_query(user_input)

    # Debug: show raw response in the app (optional)
    st.write("Raw response:", response)

    # Extract the answer text
    if isinstance(response, dict):
        answer_text = response.get("result", "")
    else:
        answer_text = str(response)

    st.session_state["messages"].append({"role": "assistant", "content": answer_text})

