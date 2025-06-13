from langchain import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import ChatGoogleGenerativeAI
import pandas as pd
import os
import asyncio
import json
from dotenv import load_dotenv
from prompt import poem_extraction
from clases import Poem,PoemExtractionOutput
#langchain-google-genai == 2.1.3
#google-ai-genarativelanguage == 0.6.17
#Cargamos las variables de entornop
load_dotenv()

api_key = os.getenv("API_KEY")

#Loader de los documentos

path = "./Poemas"



def loader_docs(path: str) -> list:
    texts = []

    for root, _, files in os.walk(path):
        for file in files:
            if file.lower().endswith(".pdf"):
                ruta = os.path.join(root,file)
                try:
                    loader= PyPDFLoader(ruta)
                    pages = loader.load()
                    texts.append(pages)
                except Exception as e:
                    print(f"Fail to load {ruta} : c")
    return texts


def llm(texts: list,poem_extraction: PromptTemplate)-> list:
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro",
                 temperature=0, top_p=0, google_api_key=api_key)
    output_parser = JsonOutputParser(pydantic_object=PoemExtractionOutput)
    poem_chain = (
            poem_extraction
            | llm
            | output_parser
    )

    all_extracted_poems = []

    for text in texts:
        combi = "\n\n".join([page.page_content for page in text])
        imprime(text)
        try:
                extract_result = poem_chain.invoke({"page_content": combi})
                imprime(extract_result)
                if isinstance(extract_result, PoemExtractionOutput):
                    if extract_result.poemas_encontrados:
                        print(f"Successfully extracted {len(extract_result.poemas_encontrados)} poems (Pydantic object).")
                        all_extracted_poems.extend(extract_result.poemas_encontrados)
                    else:
                        print("No poems found in the Pydantic object (poemas_encontrados list is empty).")
                elif isinstance(extract_result, dict):
                    print("Warning: JsonOutputParser returned a dictionary instead of a Pydantic object.")
                    if "poemas_encontrados" in extract_result and isinstance(extract_result["poemas_encontrados"], list):
                            print(f"Found 'poemas_encontrados' key in dictionary with {len(extract_result['poemas_encontrados'])} items.")
                            for poem_data in extract_result["poemas_encontrados"]:
                                try:
                                    all_extracted_poems.append(Poem(**poem_data))
                                except Exception as parse_error:
                                    print(f"Error parsing dictionary item to Poem: {parse_error}. Data: {poem_data}")
                    else:
                            print("Dictionary does not contain 'poemas_encontrados' key or it's not a list.")
        except Exception as e:
            print(f"FAIL loading {e}")
    return all_extracted_poems


def save(poems: list[Poem], filename: str = "Extracted_poems.json"):
    if not poems:
        print(f"No poems to save to {filename}.")
        return

    poems_data = []
    for poem in poems:
        
        poems_data.append(poem.model_dump())

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(poems_data, f, indent=4, ensure_ascii=False)
        print(f"Successfully saved {len(poems)} poems to {filename}")
    except IOError as e:
        print(f"Error saving poems to {filename}: {e}")



def imprime(texts):
    return print(f"Extraido el texto: {texts} ")

def main():
    texts = loader_docs(path)
    imprime(texts)
    extracted_poems = llm(texts,poem_extraction)
    #imprime(extracted_poems)
    save(extracted_poems)


if __name__ == "__main__":
    main()






