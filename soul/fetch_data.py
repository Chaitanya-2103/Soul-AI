import wikipediaapi
import requests



def fetch_dbpedia_info(entity):
    dbpedia_url = f"https://dbpedia.org/data/{entity.replace(' ', '_')}.json"
    response = requests.get(dbpedia_url, headers={"Accept": "application/json"})
    
    if response.status_code == 200:
        data = response.json()
        entity_url = f"http://dbpedia.org/resource/{entity.replace(' ', '_')}"
        if entity_url in data and "http://www.w3.org/2000/01/rdf-schema#comment" in data[entity_url]:
            return data[entity_url]["http://www.w3.org/2000/01/rdf-schema#comment"][0]["value"]
    
    return "No information found in DBpedia."

def fetch_wikipedia_info(analysis_result):
    wiki_wiki = wikipediaapi.Wikipedia(user_agent='your-user-agent', language='en')
    entity_info = {}

    for entity, entity_type in analysis_result.get("Named Entities", []):
        page = wiki_wiki.page(entity)
        if page.exists():
            entity_info[entity] = page.summary[:500]  # First 500 characters
        else:
            entity_info[entity] = fetch_dbpedia_info(entity)  # Try DBpedia

    return {
        "Entities": entity_info,
        "Sentiment": analysis_result["Sentiment"],
        "Summary": analysis_result["Summary"]
    }


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Example predictions and ground truths (you should replace these)
ground_truths = ["Paris is the capital of France.", "1889", "Isaac Newton discovered gravity."]
predictions = ["Paris is France's capital.", "Built in 1889", "Newton discovered gravity."]

# Compare using cosine similarity
threshold = 0.8
matches = []

for pred, true in zip(predictions, ground_truths):
    sim = cosine_similarity([embedder.encode(pred)], [embedder.encode(true)])[0][0]
    matches.append(sim > threshold)

# Labels for metrics
y_true = [1] * len(ground_truths)
y_pred = [1 if match else 0 for match in matches]

# Compute scores
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred)
rec = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# Plot
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
scores = [acc, prec, rec, f1]

plt.figure(figsize=(8, 5))
bars = plt.bar(metrics, scores, color=['skyblue', 'lightgreen', 'gold', 'salmon'])
plt.ylim(0, 1)
plt.title('Model Evaluation Metrics')
plt.ylabel('Score')

# Annotate each bar
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2.0, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

plt.tight_layout()
plt.show()
