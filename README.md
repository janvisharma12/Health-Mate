# MediAssist-Medical-Diagnosis-Chatbot-LLama2-RAG

# How to run?
### STEPS:

Clone the repository

```bash
Project repo: https://github.com/Mihir7b311/MediAssist-Medical-Diagnosis-Chatbot-LLama2-RAG
```

### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n mychat python=3.11 -y
```

```bash
conda activate mychat
```

### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


### Create a `.env` file in the root directory and add your Pinecone credentials as follows:

```
## I am attaching my API key
PINECONE_API_KEY = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

```


### Download the quantize model from the link provided in model folder & keep the model in the model directory:

```ini
## Download the Llama 2 Model:

llama-2-7b-chat.ggmlv3.q4_0.bin


## From the following link download the model name given above:
https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main
```

```bash
# run the following command
python store_index.py
```

```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up localhost:
```


### Techstack Used:

- Python
- LangChain
- Flask
- Meta Llama2
- Pinecone


"# Health-Mate" 
