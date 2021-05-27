import spacy_udpipe, sys

fd = sys.argv[1]
text = open(fd).read()
spacy_udpipe.download("en")
spacy_udpipe.download("sr")
#text = "Pregolem je previše velik!"
nlp = spacy_udpipe.load("sr")

doc = nlp(text)
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.dep_)
