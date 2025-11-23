Protótipo RAG (Retrieval-Augmented Generation) desenvolvido em Python como estudo prático durante a trilha de IA Generativa da plataforma Networkme, utilizando o case de políticas internas disponibilizado pelo iFood.

O projeto implementa um pipeline completo de GenAI:

  Embeddings e Vetorização: divisão dos documentos em chunks e conversão para vetores semânticos.

  Recuperação de Contexto: utilização de um retriever para busca vetorial baseada na similaridade da consulta.

  Geração Aumentada: integração do contexto retornado com o modelo Llama via LangChain, produzindo respostas contextualizadas ao conteúdo do case.

Tecnologias Utilizadas:

Python · LangChain · Llama · Embeddings · Vetorização Semântica · RAG Pipeline
