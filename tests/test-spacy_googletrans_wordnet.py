import spacy_udpipe, spacy, nltk, re, sys
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from googletrans import Translator

text = "Pregolem je previše velik!"
#fd = sys.argv[1]
#text = open(fd).read()
input_language = 'sr'
#input_language = sys.argv[2]
working_language = 'en'
#working_language = sys.argv[3]
spacy_udpipe.download(input_language)
spacy_udpipe.download(working_language)
nlp = spacy_udpipe.load(input_language)
trans = Translator()

pos_spacy_wordnet = {
    'NOUN': 'n',
    'VERB': 'v',
    'ADJ': 'a',
    'ADV': 'a',
}

doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)
    lemma = token.lemma_
    pos = token.pos_
    if pos in pos_spacy_wordnet:
        translation = trans.translate(lemma,src=input_language,dest=working_language).text
        nwords = len(translation.split(" "))
        if nwords == 1:
            netword = translation + "." + pos_spacy_wordnet[pos] + ".01"
            synword = swn.senti_synset(netword)
            print(synword.pos_score(),synword.neg_score(),synword.obj_score())
        elif nwords > 1:
            translations = re.split("\s+",translation)
            tmin = 0
            tmax = len(translations)
            for t in range(tmin,tmax):
                translation = translations[t]
                synword = swn.senti_synset(wn.synsets(translation)[0].name())
                print(synword.pos_score(),synword.neg_score(),synword.obj_score())
