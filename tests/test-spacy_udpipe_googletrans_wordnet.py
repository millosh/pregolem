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
nlp_input = spacy_udpipe.load(input_language)
nlp_working = spacy_udpipe.load(working_language)
trans = Translator()

pos_spacy_wordnet = {
    'NOUN': 'n',
    'VERB': 'v',
    'ADJ': 'a',
    'ADV': 'r',
}

def add_word(language,text,lemma,pos,dep):
    word = {
        'working language': language,
        'text': text,
        'lemma': lemma,
        'pos': pos,
        'dep': dep,
    }
    return word

def add_synset(word_input,text_working,lemma_working,pos_working,dep_working):
    word_working = add_word(working_language,text_working,lemma_working,pos_working,dep_working)
    synset = lemma_working + "." + pos_spacy_wordnet[pos_working] + ".01"
    synword = swn.senti_synset(synset)
    word_working['synset'] = synset
    word_working['positivity score'] = synword.pos_score()
    word_working['negativity score'] = synword.neg_score()
    word_working['objectivity score'] = synword.obj_score()
    if 'list' not in word_input['translation']:
        word_input['translation']['list'] = []
    word_input['translation']['list'].append(word_working)
    return word_input

doc_input = nlp_input(text)
sentence = []
for token_input in doc_input:
    text_input = token_input.text
    lemma_input = token_input.lemma_
    pos_input = token_input.pos_
    dep_input = token_input.dep_
    word_input = add_word(input_language,text_input,lemma_input,pos_input,dep_input)
    input_translation = trans.translate(lemma_input,src=input_language,dest=working_language)
    print(input_translation)
    print(dir(input_translation))
    translation = input_translation.text
    word_input['translation'] = {
        'primary translation': translation,
    }
    nwords = len(translation.split(" "))
    if nwords == 1:
        if pos_input in pos_spacy_wordnet:
            text_working = translation
            lemma_working = translation
            pos_working = token_input.pos_
            dep_working = token_input.dep_
            word_input = add_synset(word_input,text_working,lemma_working,pos_working,dep_working)
    elif nwords > 1:
        doc_working = nlp_working(translation)
        for token_working in doc_working:
            pos_working = token_working.pos_
            if pos_working in pos_spacy_wordnet:
                text_working = token_working.text
                lemma_working = token_working.lemma_
                pos_working = token_working.pos_
                dep_working = token_working.dep_
                word_input = add_synset(word_input,text_working,lemma_working,pos_working,dep_working)
    sentence.append(word_input)

print(sentence)
