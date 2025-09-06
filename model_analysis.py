from typing import List, Dict, Optional, Literal, Tuple
import re
import logging

# For preprocessing social media text
import emoji

# Hugging Face transformers
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

classifier = pipeline(task="sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment-latest")
data1 = ["happy happy happy", "I hate you"]
res = classifier(data1)
print(res)
