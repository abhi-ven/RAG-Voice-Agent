#Agent
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from rag import retrieve_notes #change this 
model = init_chat_model("ollama:qwen3.5", temperature=0, streaming=True)
"""
def call():
    # Tagged with its own tracer name so keepalive pings are easy to
    keepalive_tracer = otel_trace.get_tracer("keepalive")
    while True:
        time.sleep(30)
        try:
            with keepalive_tracer.start_as_current_span("keepalive-ping"):
                model.invoke("hi")
        except Exception as e:
            print(f"[call] {e}") 
            

threading.Thread(target=call, daemon=True).start()"""

tools = [retrieve_notes]
prompt = """You are an only an assistant for a quizzing task. You have access to a tool called retrieve_notes 
"that searches Assignment 2, 3, and 4. Always call retrieve_notes first to fetch relevant content 
"before generating any question and never ask the user what the documents are about. 
"Answer only the questions regarding the assignments. Do not answer unrelated questions and from general knowledge.
"Use only the retrieved content to generate questions for the user. 
"When the user says the answer back for a question verify the validity of the answer and identify any possible improvements to be made 
"to the answer and prompt them with the next question keeping in mind the chain of thought, and remember to track all the questions asked 
"so that feedback can be given to the user if asked either about the question or the entire quiz. 
"When speaking do not read symbols out loud. 
When evaluating the user's answer,
ALWAYS begin your response with exactly one of these words:

Correct
Partially Correct
Incorrect

Then explain why.

Finally ask the next question."""

agent = create_agent(model=model, tools=tools, system_prompt=prompt)