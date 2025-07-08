import spacy
from textblob import TextBlob
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
import numpy as np
import wikipedia
import wikipediaapi
import requests

# Load spaCy model for Named Entity Recognition
# nlp = spacy.load("en_core_web_sm")

# from textblob import TextBlob


# def correct_text(text):
#     return TextBlob(text).correct()

# def analyze_text(text):
#     doc = nlp(text)
#     desired_labels = {"ORG", "PERSON", "GPE", "LOC"}  # Specify the labels you want
#     entities = [(ent.text, ent.label_) for ent in doc.ents if ent.label_ in desired_labels]

#     # Sentiment Analysis
#     sentiment = TextBlob(text).sentiment.polarity
#     sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"

#     # Text Summarization
#     parser = PlaintextParser.from_string(text, Tokenizer("english"))
#     summarizer = LsaSummarizer()
#     summary_sentences = summarizer(parser.document, 2)  # Get top 2 sentences
#     summary = " ".join(str(sentence) for sentence in summary_sentences)

#     return {
#         "Named Entities": entities,
#         "Sentiment": sentiment_label,
#         "Summary": summary
#     }

# def fetch_dbpedia_info(entity):
#     dbpedia_url = f"https://dbpedia.org/data/{entity.replace(' ', '_')}.json"
#     response = requests.get(dbpedia_url, headers={"Accept": "application/json"})
    
#     if response.status_code == 200:
#         data = response.json()
#         entity_url = f"http://dbpedia.org/resource/{entity.replace(' ', '_')}"
#         if entity_url in data and "http://www.w3.org/2000/01/rdf-schema#comment" in data[entity_url]:
#             return data[entity_url]["http://www.w3.org/2000/01/rdf-schema#comment"][0]["value"]
    
#     return "No information found"



# def fetch_wikipedia_info(analysis_result):
#     wiki_wiki = wikipediaapi.Wikipedia(user_agent='your-user-agent', language='en')

#     named_entities = analysis_result.get("Named Entities", [])
    
#     entity_info = {}

#     for entity_tuple in named_entities:
#         entity = entity_tuple[0] if isinstance(entity_tuple, tuple) else entity_tuple
        
#         page = wiki_wiki.page(entity)
#         entity_info[entity] = page.summary[:1200] if page.exists() else fetch_dbpedia_info(entity)

#     return {"data": entity_info}



# new model
from transformers import pipeline
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import faiss

# Load models
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Load open fallback dataset (e.g., SQuAD)
qa_dataset = load_dataset("squad", split="train[:2000]")  # keep it small for now
questions = [item["question"] for item in qa_dataset]
answers = [item["answers"]["text"][0] if item["answers"]["text"] else "No answer" for item in qa_dataset]

# Create embedding index
question_embeddings = embedder.encode(questions, convert_to_tensor=False)
index = faiss.IndexFlatL2(len(question_embeddings[0]))
index.add(np.array(question_embeddings))

def smart_summary_with_open_fallback(query: str) -> str:
    try:
        wikipedia.set_lang("en")
        try:
            page = wikipedia.page(query)
        except wikipedia.DisambiguationError as e:
            page = wikipedia.page(e.options[0])
        except wikipedia.PageError:
            search_results = wikipedia.search(query)
            if not search_results:
                raise ValueError("No Wikipedia match")
            page = wikipedia.page(search_results[0])

        content = page.content
        paragraphs = content.split('\n')
        filtered = "\n".join([p for p in paragraphs if len(p.strip()) > 50])
        trimmed = filtered[:2000]
        summary = summarizer(trimmed, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
        return { 'data':summary}

    except Exception:
        # Fallback to SQuAD
        query_vec = embedder.encode([query])[0]
        _, I = index.search(np.array([query_vec]), k=1)
        best_match_q = questions[I[0][0]]
        best_match_a = answers[I[0][0]]
        return { 'data':best_match_a }


#print(smart_summary_with_open_fallback('econimic capital of india'))
                                       
                                    