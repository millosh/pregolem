import spacy, nltk
from spacy_wordnet.wordnet_annotator import WordnetAnnotator 

# Load an spacy model (supported models are "es" and "en") 
nlp = spacy.load('en_core_web_lg')
nltk.download('wordnet')
nlp.add_pipe(WordnetAnnotator(nlp.lang), after='tagger')
token = nlp('prices')[0]

# wordnet object link spacy token with nltk wordnet interface by giving acces to
# synsets and lemmas 
print(token._.wordnet.synsets())
print(token._.wordnet.lemmas())

# And automatically tags with wordnet domains
print(token._.wordnet.wordnet_domains())
