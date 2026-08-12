#Streaming 
import re
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from record import texttospeech
end = re.compile(r'(?<=[.!?])\s+')

def stream_response(agent, messages, on_token=None):
    buffer = ""
    full_response = ""
    print("Agent: ", end="", flush=True)

    for chunk, metadata in agent.stream({"messages": messages}, stream_mode="messages"):
        if metadata.get("langgraph_node") != "model":
            continue
        if isinstance(chunk.content, str):
            token = chunk.content
        else:
            token = ""
        if not token:
            continue

        print(token, end="", flush=True)
        buffer += token
        full_response += token

        if on_token:
            on_token(full_response)

        line = end.split(buffer)
        if len(line) > 1:
            for j in line[:-1]:
                texttospeech(j)
            buffer = line[-1]

    print()
    if buffer.strip():
        texttospeech(buffer)

    new = messages + [AIMessage(content=full_response)]
    return full_response, new
