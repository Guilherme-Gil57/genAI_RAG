from langchain_chroma.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
import warnings

warnings.filterwarnings("ignore")

Data_base_path = 'database'

prompt_template = """Você é um agente interno que auxilia colaboradores a decidirem sobre reembolsos e cancelamentos. 
Sempre consulte a base de conhecimento:
{base}
Antes de responder a pergunta do usuário:
{question} 
Se não houver confiança suficiente, sugira validação manual ou abertura de ticket interno, em vez de gerar uma resposta incerta.
"""
def question():

    ask = input("Escreva sua pergunta: ")
    db = Chroma(persist_directory = Data_base_path, embedding_function = OllamaEmbeddings(model="nomic-embed-text"))

    results = db.similarity_search_with_relevance_scores(ask, k=4)

    if len(results) == 0 or results[0][1] < -250:
        print('Não consigo informar. Faça a abertura de um ticket interno')
        return
    
    text_result = []
    for result in results: text_result.append(result[0].page_content)
        
    base = "\n\n------\n\n".join(text_result)
    prompt = PromptTemplate.from_template(prompt_template).invoke({"base":base, "question":ask})

    print("Resposta da IA:", ChatOllama(model="llama3.2:1b").invoke(prompt).content)

question()