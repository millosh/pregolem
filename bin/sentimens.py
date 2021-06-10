import spacy_udpipe, spacy_wordnet, spacy, nltk, re, sys, time, os
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from spacy_wordnet.wordnet_annotator import WordnetAnnotator
#from googletrans import Translator
from translate import Translator
from os.path import *

def exec_cmd(cmd,torun,toprint,togo):
    ret = False
    if toprint:
        print(cmd)
    if torun:
        if togo:
            ret = go(cmd)
        else:
            ret = os.system(cmd)
    return ret

def get_arg(arg,args):
    value = False
    if arg in sys.argv:
        index = sys.argv.index(arg)
        if len(sys.argv) > index+1:
            value = sys.argv[index+1]
            arg = re.sub("^\-+","",arg)
            args[arg] = value
        else:
            sys.exit("No value for argument '" + arg + "', no fun!")
    return args

def get_args():
    args = {}
    # timestamps
    args['start time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    default_timestamp = '2020-08-24 12:00'
    # general
    args = get_arg('--command',args)
    ## --email: optional and used for translation provider MyMemory
    ##          cf. https://translate-python.readthedocs.io/en/latest/providers.html
    ##              https://mymemory.translated.net/doc/usagelimits.php
    args = get_arg('--email',args)
    args = get_arg('--input',args)
    args = get_arg('--input-text',args)
    if 'input' in args:
        args['input-text'] = open(args['input']).read()
    elif 'input-text' not in args:
        args['input-text'] = "Pregolem je previše velik!"
    args = get_arg('--input-language',args)
    if 'input-language' not in args:
        args['input-language'] = 'sr'
    args = get_arg('--working-language',args)
    if 'working-language' not in args:
        args['working-language'] = 'en'
    ## TODO: specify different input and working languages for:
    ##       - spacy
    ##       - spacy_udpipie
    ##       - translate module
    ##       - spacy/nltk interface to wordnet
    # download models
    spacy_udpipe.download(args['input-language'])
    spacy_udpipe.download(args['working-language'])
    nltk.download('wordnet')
    nltk.download('sentiwordnet')
    # pointers
    args['nlp-input'] = spacy_udpipe.load(args['input-language'])
    #args['nlp-working'] = spacy_udpipe.load(args['working-language'])
    args['nlp-working'] = spacy.load("en")
    args['nlp-working'].add_pipe(WordnetAnnotator(args['nlp-working'].lang), after='tagger')
    #args['translator'] = Translator()
    if 'email' in args:
        args['translator'] = Translator(from_lang=args['input-language'],to_lang=args['working-language'],email=args['email'])
    else:
        args['translator'] = Translator(from_lang=args['input-language'],to_lang=args['working-language'])
    # meta translation
    args['relevant parts of speech'] = {
        'NOUN': 'n',
        'VERB': 'v',
        'ADJ': 'a',
        'ADV': 'r',
    }
    #
    if 'root-dir' not in args:
        args['root-dir'] = "."
    #if not exists(args['root-dir']):
    #    os.mkdir(args['root-dir'])
    args = get_arg('--error-file',args)
    if 'error-file' not in args:
        args['error-file'] = args['root-dir'] + '/errors.log'
    args = get_arg('--data-dir',args)
    if 'data-dir' not in args:
        args['data-dir'] = args['root-dir'] + "/data"
    #if not exists(args['data-dir']):
    #    os.mkdir(args['data-dir'])
    return args

def add_word(language,token,lemma,pos,args):
    word = {
        'language': language,
        'token': token,
        'lemma': lemma,
        'pos': pos,
    }
    return word

def add_synset(language,lemma,pos,args):
    working_token = add_word(language,None,lemma,pos,args)
    wordnet_token = args['nlp-working'](lemma)[0]
    working_token['wordnet token'] = wordnet_token
    try:
        # Getting WordNet part of speech not so nice as I want to avoid
        #   creating a dictionary :/
        wordnet_pos = str(wordnet_token._.wordnet.lemmas()[0]).split(".")[1]
        synset = lemma + "." + wordnet_pos + ".01"
        working_token['wordnet enetity'] = synset
        try:
            working_token['sentiment token'] = swn.senti_synset(synset)
        except nltk.corpus.reader.wordnet.WordNetError:
            working_token['sentiment token'] = None
    except IndexError:
        working_token['sentiment token'] = None
    return working_token

def process_text(args):
    paragraphs = {}
    protopars = re.split("\n",args['input-text'])
    pmin = 0
    pmax = len(protopars)
    for p in range(pmin,pmax):
        par = protopars[p].strip()
        if par != '':
            paragraphs[p] = {
                'text': par,
                'doc': args['nlp-input'](par),
                'sentences': {},
            }
    return paragraphs
    
def get_sentences(paragraphs,args):
    pmin = 0
    pmax = len(list(paragraphs.keys()))
    prev = ''
    for p in range(pmin,pmax):
        pkey = list(paragraphs.keys())[p]
        doc = paragraphs[pkey]['doc']
        sn = 0
        for token in doc:
            if token.is_sent_start:
                sentence = str(token.sent)
                sdoc = args['nlp-input'](sentence)
                paragraphs[pkey]['sentences'][sn] = {
                    'text': sentence,
                    'doc': sdoc,
                    'tokens': {},
                }
                sn += 1
    return paragraphs

def count_transenti(token,args):
    ### TODO: develop a method to translate words separately from the rest
    ###       of the engine
    if token.pos_ in args['relevant parts of speech']:
        tw = 1
    else:
        tw = 0
    return tw

def get_transenti(token,args):
    working_entity = {
        'positivity score': None,
        'negativity score': None,
        'objectivity score': None,
        'working lemma': None,
        'children': [],
        'working tokens': [],
    }
    for child in token.subtree:
        working_entity['children'].append(child)
    if token.pos_ in args['relevant parts of speech']:
        translation = args['translator'].translate(token.lemma_)
        print(token.lemma_,translation,args['input-language'],args['working-language'])
        working_entity['primary translation'] = translation
        nwords = len(translation.split(" "))
        if nwords == 1:
            working_lemma = translation
            working_token = add_synset(args['working-language'],working_lemma,token.pos_,args)
            # TODO: token._.wordnet.wordnet_domains()
            working_entity['working tokens'].append(working_token)
        else:
            wdoc = args['nlp-working'](translation)
            working_entity['sentiment words'] = []
            for wtoken in wdoc:
                working_token = add_synset(args['working-language'],wtoken.lemma_,wtoken.pos_,args)
                working_entity['working tokens'].append(working_token)
    return working_entity

def process_tokens(paragraphs,what,args):
    tw = 0
    pmin = 0
    pmax = len(list(paragraphs.keys()))
    for p in range(pmin,pmax):
        pkey = list(paragraphs.keys())[p]
        smin = 0
        smax = len(list(paragraphs[pkey]['sentences'].keys()))
        for s in range(smin,smax):
            skey = list(paragraphs[pkey]['sentences'].keys())[s]
            doc = paragraphs[pkey]['sentences'][skey]['doc']
            tn = 0
            for token in doc:
                if what == 'process':
                    working_entity = get_transenti(token,args)
                    print(pkey, ":::", skey, ":::", tn, ":::", token.text, ":::",)
                    paragraphs[pkey]['sentences'][skey]['tokens'][tn] = {
                        'text': str(token.text),
                        'token': token,
                        'working entity': working_entity,
                    }
                elif what == 'count-translate':
                    tw += count_transenti(token,args)
                tn += 1
    if what == 'count-translate':
        print(tw)
    return paragraphs

def main():
    args, data = get_args()
    if args['command'] == 'process':
        # python sentimens.py --email your@email --command process --input input/file
        # python sentimens.py --email your@email --command process --input-text "input text"
        ## --email: optional and used for translation provider MyMemory
        ##          cf. https://translate-python.readthedocs.io/en/latest/providers.html
        ##              https://mymemory.translated.net/doc/usagelimits.php
        paragraphs = process_text(args)
        paragraphs = get_sentences(paragraphs,args)
        paragraphs = process_tokens(paragraphs,"process",args)
    elif args['command'] == 'count-translate':
        # python sentimens.py --command count-translate --input input/file
        # python sentimens.py --command count-translate --input-text "input text"
        paragraphs = process_text(args)
        paragraphs = get_sentences(paragraphs,args)
        paragraphs = process_tokens(paragraphs,"count-translate",args)

if __name__ == "__main__":
    main()
