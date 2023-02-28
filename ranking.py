from src.contriever import Contriever
from transformers import AutoTokenizer

titleFP = open('/home/chapapadopoulos/github/NER/SemEval2023/M4D-SemEval2023_Task2/data/v1_with_entity/en_dev.txt', 'r', encoding="UTF-8")
bertscoresFP = open('/home/chapapadopoulos/github/NER/SemEval2023/M4D-SemEval2023_Task2/data/bertScore/bertScores_en_dev.txt', 'r', encoding="UTF-8")
exportFP = open('contrieverScores_en_dev.txt', 'w', encoding="UTF-8")
