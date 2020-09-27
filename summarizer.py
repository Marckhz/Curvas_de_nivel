import spacy
import numpy as np
import networkx as nx

def sim_matrix(x):
    matrix = np.zeros([len(x), len(x)])
    return matrix

def build_matrix(matrix, sentences, r, docs):
    for col in range(r):
        for row in range(r):
            if col != row:
                matrix[col][row] = docs.sents[col].similarity(docs.sents[row])
    return matrix

def rank_sentences(sentences_raw, scores):
    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences_raw)), reverse=True)
    return ranked_sentences

if __name__ == "__main__":
    def summarizer(text):
        nlp = spacy.load("en_web_lg")
        sen = nlp.create_pipe("sentencizer")
        nlp.add_pipe(sen)
        #return sentences obj
        docs = nlp.make_doc(text)
        num_of_sentences = len(list(docs.sents))
        zeroes_matrix  = sim_matrix(num_of_sentences)
        emb_matrix = build_matrix(zeroes_matrix, text, num_of_sentences, docs)
    
        nx_graph = nx.from_numpy_array(emb_matrix)
        scores = nx.pagerank(nx_graph)
        summary = rank_sentences(text,scores )
        
        for i in range(10):
            print(summary[i][1])
    
