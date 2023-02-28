from src.contriever import Contriever
from transformers import AutoTokenizer

contriever = Contriever.from_pretrained("facebook/contriever") 
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever") #Load the associated tokenizer

sentences = [
    "Where was Marie Curie born?",
    "Maria Sklodowska, later known as Marie Curie, was born on November 7, 1867.",
    "Born in Paris on 15 May 1859, Pierre Curie was the son of Eugène Curie, a doctor of French Catholic origin from Alsace."
]

sentences2 = [
    "eli lilly founder president of pharmaceutical company eli lilly and company",
    "Eli is the founder of Voices4, a nonviolent direct-action activist group committed to advancing global queer liberation.",
    "Before beginning in activism, Eli worked in real estate."
]

inputs = tokenizer(sentences2, padding=True, truncation=True, return_tensors="pt")
embeddings = contriever(**inputs)

score01 = embeddings[0] @ embeddings[1] #1.0473
score02 = embeddings[0] @ embeddings[2] #1.0095
print(score01)
print(score02)