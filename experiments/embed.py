import xml.etree.ElementTree as ET
from collections import Counter

import torch
from sklearn.cluster import KMeans
from transformers import AutoModel, AutoTokenizer

tree = ET.parse("../data/The_Times_in_Plain_English_complete_v4.xml")
root = tree.getroot()

texts = []
for article in list(root):
    content = article.find("content")
    texts.append(content.find("text").text)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True, verbose=False)

tokenized_texts = [tokenizer.tokenize(text) for text in texts]
tok_counter = Counter([token for text in tokenized_texts for token in text])
tok_freqs = sorted(dict(tok_counter), key=tok_counter.get, reverse=True)

model = AutoModel.from_pretrained(model_name)

word = "states"  # picked this somewhat arbitrarily -- amongst top non-trivial tokens
contexts = []
indices = []
for tokens in tokenized_texts:
    if word in tokens:
        idx = tokens.index(word)
        context = tokens[max(0, idx - 256) : idx + 256]
        idx = context.index(word)
        contexts.append(tokenizer.convert_tokens_to_string(context))
        indices.append(idx)
        # TODO: remove this restriction
        if len(contexts) == 10:
            break

inputs = tokenizer(contexts, return_tensors="pt", padding=True, truncation=True)
outputs = model(**inputs)
embeds = [outputs.last_hidden_state[i, idx, :].detach() for i, idx in enumerate(indices)]


# TODO: visualize with PCA, t-SNE


def cluster_tensors(tensors, n_clusters):
    # Stack tensors into a single 2D tensor
    stacked = torch.stack(tensors).view(len(tensors), -1)

    # Convert to numpy array for scikit-learn
    data = stacked.numpy()

    # Perform K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    labels = kmeans.fit_predict(data)

    return labels


cluster_labels = cluster_tensors(embeds, 4)  # 4 found using elbow method

# for each cluster, find the corresponding context
clustered_contexts = {i: [] for i in range(4)}
for i, context in enumerate(contexts):
    clustered_contexts[cluster_labels[i]].append(context)
