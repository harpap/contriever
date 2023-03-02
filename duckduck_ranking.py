from src.contriever import Contriever
from transformers import AutoTokenizer
import json

titleFP = open('/home/chapapadopoulos/github/NER/SemEval2023/M4D-SemEval2023_Task2/data/v1_with_entity/en_dev.txt', 'r', encoding="UTF-8")
duckduck_retrieveFP = open('/home/chapapadopoulos/github/NER/SemEval2023/M4D-SemEval2023_Task2/kb/en-dev_retrieved_data.json', 'r', encoding="UTF-8")
#bertscoresFP = open('/home/chapapadopoulos/github/NER/SemEval2023/M4D-SemEval2023_Task2/data/bertScore/bertScores_en_dev.txt', 'r', encoding="UTF-8")

exportFP = open('duckduckContrieverScores_en_dev.txt', 'w', encoding="UTF-8")

contriever = Contriever.from_pretrained("facebook/contriever") 
tokenizer = AutoTokenizer.from_pretrained("facebook/contriever") #Load the associated tokenizer

def get_title(fp):
    for line in fp:
        if line.startswith('id:'):
            title = line.split('	# ')[1]
            title = title.split('\t')[0]
            return title

duckduck_data = json.load(duckduck_retrieveFP)
print(duckduck_data[1][0]['results']['results'][1]['snippet'])
print(duckduck_data[1][0]['id'])
i=0
while i < len(duckduck_data):
    title = get_title(titleFP)
    exportFP.write(duckduck_data[i][0]['id'] + '\n' + title + '\n')
    j=0
    scores = []
    snips = []
    try:
        while j < len(duckduck_data[i][0]['results']['results']):
            snip = duckduck_data[i][0]['results']['results'][j]['snippet']
            snips.append(snip)
            inputs = tokenizer([title, snip], padding=True, truncation=True, return_tensors="pt")
            embeddings = contriever(**inputs)
            score = embeddings[0] @ embeddings[1]
            scores.append(score.item())
            j+=1
        context_dict = dict(zip(scores, snips))
        ranked_context_dict = dict(sorted(context_dict.items(), reverse=True))
        for (key, value) in ranked_context_dict.items():
                exportFP.write(str(key) + '\t' + value + '\n')
    except KeyError:
         print('KeyError')
         exportFP.write('KeyError\n')
    exportFP.write('\n')
    i+=1