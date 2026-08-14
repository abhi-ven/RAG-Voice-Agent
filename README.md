# Voice Quiz Agent

An interactive voice-driven quiz agent which uses RAG (Retrieval-Augmented Generation) over uploaded documents (PDFs) to ask questions, evaluate spoken answers and adapt to difficulty based on performance with speech-to-text input and text-to-speech output. 

## Features

**Voice Interface**: Responses can be given by the user via microphone (SST: Whisper) and agent's response will be heard through TTS: Kokoro.
  
**Relevant Questions**: The use of RAG ensures that the questions and latter responses by the agent are specific to a certain topic without any attempt to consult any general knowledge. Data is specific to the content in the uploaded PDFs (via Chroma + Ollama Embeddings).

**Adaptive Difficulty**: A score tracker records the accuracy of the user's responses and adapts the difficulty level of the questions as necessary after every 5 questions. 

**Streaming Response**: The agent's response is streamed simultaneously as it starts speaking in order to reduce latency and increase efficiency. 



## Architecture

1. rag.py: Loads/Embeds (using OllamaEmbeddings) PDFs into Chroma Vector Store after creating chunks and creates retrieve_notes tool.

2. agent.py: Defines the LangChain agent (model: ollama:qwen3.5), tools, system prompt and model configurations. 

3. record.py: Defines functions for recording audio data, transcribing audio to text via Whisper(base) and Kokoro TTS response. 

4. streaming.py: Streams agent output token-by-token and simultaneously invokes TTS per sentence. 

5. main.py: Contains the user interface made using Flet and score tracking. 



## Prerequisites
  
  - Ollama (with necessary models pulled)
  - Python 3.9 to Python 3.12 (recommend a dedicated conda environment if version is not compatible)
    

## Setup

1. **Clone and create environment:**
  
```bash
   git clone https://github.com/abhi-ven/RAG-Voice-Agent.git
   cd RAG-Voice-Agent
   conda create -n kokoro python=3.12
   conda activate kokoro
   pip install -r requirements.txt
```
2. **Upload PDFs:** Update the 'files' list in the 'rag.py' to point to necessary documents in a pdf format and update os.chdir(...) to the folder containing them. 
3. **Run the main.py:**
   
```bash
   python main.py
```


## Usage


  Press "Speak" to starting asking/answering. Press "Stop" when done. 
  
  The agent transcribes your speech, retrieves relevant notes and responds both in the chat and out loud. 
  
  In order to stop the program, say "Stop now".
  
  Score will be displayed at the top in format: correct answers/questions answered.
 
