import spacy_udpipe, spacy_wordnet, spacy, nltk, re, csv, time, pickle, sys, os
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
    data = {
        'domains': {},
        'sentiments': {},
        'paragraphs': {},
    }
    # timestamps
    args['start time'] = time.strftime("%Y-%m-%d %H:%M:%S")
    default_timestamp = '2020-08-24 12:00'
    
    # directories
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
    args = get_arg('--input-pickle',args)
    args = get_arg('--output-pickle',args)
    args = get_arg('--output-csv',args)
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

    # dictionary
    args = get_arg('--input-dictionary',args)
    if 'input-dictionary' in args:
        args['dict'] = pickle.load(open(args['input-dictionary'],'rb'))
    else:
        args['dict'] = {}
    args = get_arg('--output-dictionary',args)
    if 'outpu-dictionary' in args:
        args['output-dictionary'] = 'dictionary-output.pickle'
    
    # download models
    spacy_udpipe.download(args['input-language'])
    spacy_udpipe.download(args['working-language'])
    nltk.download('wordnet')
    nltk.download('sentiwordnet')
    
    # pointers
    args['nlp-input'] = spacy_udpipe.load(args['input-language'])
    ## for working language -- i.e. English -- main spacy model is more useful
    args['nlp-working'] = spacy.load(args['working-language'])
    args['nlp-working'].add_pipe(WordnetAnnotator(args['nlp-working'].lang), after='tagger')
    #args['nlp-working'] = spacy_udpipe.load(args['working-language'])
    ## Google Translate is not reliable, we are using MyMemory
    ##    through the translate module
    if 'email' in args:
        args['translator'] = Translator(from_lang=args['input-language'],to_lang=args['working-language'],email=args['email'])
    else:
        args['translator'] = Translator(from_lang=args['input-language'],to_lang=args['working-language'])
    #args['translator'] = Translator()
    
    # relevant parts of speech
    args['relevant parts of speech'] = [
        'NOUN', 'VERB', 'ADJ', 'ADV',
    ]

    return args, data

def add_word(language,lemma,pos,args):
    word = {
        'language': language,
        'lemma': lemma,
        'pos': pos,
    }
    return word

def add_synset(language,lemma,pos,args):
    working_word = add_word(language,lemma,pos,args)
    wordnet_token = args['nlp-working'](lemma)[0]
    working_word['domains'] = wordnet_token._.wordnet.wordnet_domains()
    try:
        # Getting WordNet part of speech not so nice as I want to avoid
        #   creating a dictionary :/
        wordnet_pos = str(wordnet_token._.wordnet.lemmas()[0]).split(".")[1]
        synset = lemma + "." + wordnet_pos + ".01"
        working_word['wordnet enetity'] = synset
        try:
            sn = swn.senti_synset(synset)
            working_word['sentiments'] = {
                'negativity score': sn.neg_score(),
                'positivity score': sn.pos_score(),
                'objectivity score': sn.obj_score(),
            }
        except nltk.corpus.reader.wordnet.WordNetError:
            working_word['sentiments'] = None
    except IndexError:
        working_word['sentiments'] = None
    return working_word

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
                'sentences': {},
            }
    return paragraphs
    
def get_sentences(paragraphs,args):
    pmin = 0
    pmax = len(list(paragraphs.keys()))
    prev = ''
    for p in range(pmin,pmax):
        pkey = list(paragraphs.keys())[p]
        doc = args['nlp-input'](paragraphs[pkey]['text'])
        sn = 0
        for token in doc:
            if token.is_sent_start:
                sentence = str(token.sent)
                paragraphs[pkey]['sentences'][sn] = {
                    'text': sentence,
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


def set_working_entity(token,args):
    working_entity = {
        'positivity score': None, # construct score!
        'negativity score': None, # construct score!
        'objectivity score': None, # construct score!
        'token id': token.i,
        'lemma': token.lemma_,
        'relevant': False,
        'children': [],
        'working doc': [],
        'working words': [],
    }
    for child in token.subtree:
        working_entity['children'].append(child.i)
    if token.pos_ in args['relevant parts of speech']:
        working_entity['relevant'] = True
    return working_entity

def create_structure(paragraphs,args):
    tw = 0
    pmin = 0
    pmax = len(list(paragraphs.keys()))
    for p in range(pmin,pmax):
        pkey = list(paragraphs.keys())[p]
        smin = 0
        smax = len(list(paragraphs[pkey]['sentences'].keys()))
        for s in range(smin,smax):
            skey = list(paragraphs[pkey]['sentences'].keys())[s]
            doc = args['nlp-input'](paragraphs[pkey]['sentences'][skey]['text'])
            tn = 0
            for token in doc:
                print(pkey, ":::", skey, ":::", tn, ":::", token.text, ":::",)
                working_entity = set_working_entity(token,args)
                paragraphs[pkey]['sentences'][skey]['tokens'][tn] = {
                    'text': str(token.text),
                    'working entity': working_entity,
                }
                tn += 1
    return paragraphs

def get_translation(token,working_entity,nrel,args):
    #translation = ''
    if token.lemma_ in args['dict']:
        translation = args['dict'][token.lemma_]
    else:
        translation = args['translator'].translate(token.lemma_)
        args['dict'][token.lemma_] = translation
    print("nrel=" + str(nrel), '::: ntrans=' + str(len(list(args['dict'].keys()))), ':::', token.lemma_, ':::', translation)
    working_entity['primary translation'] = translation
    return working_entity, args

def get_sentiment(token,working_entity,nrel,args):
    translation = working_entity['primary translation']
    working_doc = args['nlp-working'](working_entity['primary translation'])
    for working_token in working_doc:
        working_word = add_synset(args['working-language'],working_token.lemma_,working_token.pos_,args)
        working_entity['working words'].append(working_word)
        print(nrel, ':::', token.lemma_, ':::', working_word)
    return working_entity

def make_domains(working_entity,args):
    domains = {}
    working_words = working_entity['working words']
    wmin = 0
    wmax = len(working_words)
    for w in range(wmin,wmax):
        working_word = working_words[w]
        working_domains = working_word['domains']
        for domain in working_domains:
            if domain not in domains:
                domains[domain] = 0
            domains[domain] += 1
    return domains

def make_sentiments(working_entity,others,nrel,data):
    lemma = working_entity['lemma']
    token_id = working_entity['token id']
    if lemma not in data['sentiments']:
        data['sentiments'][lemma] = {
            'negativity score': 0,
            'positivity score': 0,
            'objectivity score': 0,
            'sentiment addition frequency': 0,
        }
    working_words = working_entity['working words']
    wmin = 0
    wmax = len(working_words)
    for w in range(wmin,wmax):
        working_word = working_words[w]
        if working_word['sentiments'] != None:
            data['sentiments'][lemma]['negativity score'] += working_word['sentiments']['negativity score'] * float(1)/float(wmax)
            data['sentiments'][lemma]['positivity score'] += working_word['sentiments']['positivity score'] * float(1)/float(wmax)
            data['sentiments'][lemma]['objectivity score'] += working_word['sentiments']['objectivity score'] * float(1)/float(wmax)
            data['sentiments'][lemma]['sentiment addition frequency'] += 1
            for other in others:
                if others[other]['working entity']['token id'] != token_id:
                    other_lemma = others[other]['working entity']['lemma']
                    if other_lemma not in data['sentiments']:
                        data['sentiments'][other_lemma] = {
                            'negativity score': 0,
                            'positivity score': 0,
                            'objectivity score': 0,
                            'sentiment addition frequency': 0,
                        }
                        data['sentiments'][other_lemma]['negativity score'] += working_word['sentiments']['negativity score'] * float(1)/float(len(others))
                        data['sentiments'][other_lemma]['positivity score'] += working_word['sentiments']['positivity score'] * float(1)/float(len(others))
                        data['sentiments'][other_lemma]['objectivity score'] += working_word['sentiments']['objectivity score'] * float(1)/float(len(others))
                        data['sentiments'][other_lemma]['sentiment addition frequency'] += 1
    return data

def update_paragraphs(paragraphs,args,data):
    plist = list(paragraphs.keys())
    pmin = 0
    pmax = max(plist)
    nrel = 0
    for p in range(pmin,pmax):
        try:
            pkey = plist[p]
            data['paragraphs'][pkey] = {
                'domains': {},
                'sentences': {},
            }
            slist = list(paragraphs[pkey]['sentences'].keys())
            smin = 0
            smax = max(slist)
            for s in range(smin,smax):
                skey = slist[s]
                data['paragraphs'][pkey]['sentences'][skey] = {
                    'domains': {},
                    'tokens': {},
                }
                doc = args['nlp-input'](paragraphs[pkey]['sentences'][skey]['text'])
                tlist = list(paragraphs[pkey]['sentences'][skey]['tokens'].keys())
                tmin = 0
                tmax = max(tlist)
                for t in range(tmin,tmax):
                    tkey = tlist[t]
                    token = doc[tkey]
                    working_entity = paragraphs[pkey]['sentences'][skey]['tokens'][tkey]['working entity']
                    if working_entity['relevant']:
                        #print(pkey,skey,tkey)
                        nrel += 1
                        if args['command'] == "get-translations":
                            working_entity, args = get_translation(token,working_entity,nrel,args)
                        elif args['command'] == "get-sentiments":
                            working_entity = get_sentiment(token,working_entity,nrel,args)
                        elif args['command'] == "make-domains":
                            if pkey not in data['paragraphs']:
                                data['paragraphs'][pkey] = {
                                    'domains': {},
                                    'sentences': {},
                                }
                            if skey not in data['paragraphs'][pkey]['sentences']:
                                data['paragraphs'][pkey]['sentences'][skey] = {
                                    'domains': {},
                                    'tokens': {},
                                }
                            if tkey not in data['paragraphs'][pkey]['sentences'][skey]['tokens']:
                                data['paragraphs'][pkey]['sentences'][skey]['tokens'][tkey] = {
                                    'domains': {},
                                    'tokens': {},
                                }
                            data['paragraphs'][pkey]['sentences'][skey]['tokens'][tkey]['domains'] = make_domains(working_entity,args)
                            for domain in data['paragraphs'][pkey]['sentences'][skey]['tokens'][tkey]['domains']:
                                if domain not in data['paragraphs'][pkey]['sentences'][skey]['domains']:
                                    data['paragraphs'][pkey]['sentences'][skey]['domains'][domain] = 0
                                data['paragraphs'][pkey]['sentences'][skey]['domains'][domain] += data['paragraphs'][pkey]['sentences'][skey]['tokens'][tkey]['domains'][domain]
                        elif args['command'] == "make-sentiments":
                            tokens = paragraphs[pkey]['sentences'][skey]['tokens']
                            data = make_sentiments(working_entity,tokens,args,data)
                        paragraphs[pkey]['sentences'][skey]['tokens'][tkey]['working entity'] = working_entity
                    if args['command'] == "make-domains":
                        for domain in data['paragraphs'][pkey]['sentences'][skey]['domains']:
                            if domain not in data['paragraphs'][pkey]['domains']:
                                data['paragraphs'][pkey]['domains'][domain] = 0
                            data['paragraphs'][pkey]['domains'][domain] += data['paragraphs'][pkey]['sentences'][skey]['domains'][domain]
            if args['command'] == "make-domains":
                for domain in data['paragraphs'][pkey]['domains']:
                    if domain not in data['domains']:
                        data['domains'][domain] = 0
                    data['domains'][domain] += data['paragraphs'][pkey]['domains'][domain]
        except IndexError:
            pass
    return paragraphs, args, data

def write_csv(args,data):
    csv_fd = open(args['output-csv'],'w')
    csv_writer = csv.writer(csv_fd)
    row = [
        'lemma',
        'sentiment addition frequency',
        'normalized linguistic negativity score',
        'normalized linguistic positivity score',
        'normalized linguistic objectivity score',
        'linguistic negativity score',
        'linguistic positivity score',
        'linguistic objectivity score',
    ]
    csv_writer.writerow(row)
    for lemma in data['sentiments']:
        if data['sentiments'][lemma]['sentiment addition frequency'] > 0:
            row = [
                lemma,
                data['sentiments'][lemma]['sentiment addition frequency'],
                float(data['sentiments'][lemma]['negativity score']) / float(data['sentiments'][lemma]['sentiment addition frequency']),
                float(data['sentiments'][lemma]['positivity score']) / float(data['sentiments'][lemma]['sentiment addition frequency']),
                float(data['sentiments'][lemma]['objectivity score']) / float(data['sentiments'][lemma]['sentiment addition frequency']),
                data['sentiments'][lemma]['negativity score'],
                data['sentiments'][lemma]['positivity score'],
                data['sentiments'][lemma]['objectivity score'],
            ]
            csv_writer.writerow(row)
    csv_fd.close()

def fix_dict(args,data):
    newdict = {}
    for inword in args['dict']:
        outword = args['dict'][inword]
        if not re.search("^MYMEMORY WARNING: YOU USED ALL AVAILABLE FREE TRANSLATIONS FOR TODAY",outword):
            print(inword,outword)
            newdict[inword] = outword
    old_number = len(list(args['dict'].keys()))
    new_number = len(list(newdict.keys()))
    print("old number:", old_number, ":::", "new number:", new_number)
    args['dict'] = newdict
    return args, data
    
def main():
    args, data = get_args()
    # 1. Create structure
    # 2. Get translations
    # 3. Get sentiments
    # 4. Make wordnet domains
    if args['command'] == 'create-structure':
        # python sentiments.py --command create-structure --input input/file --output-pickle structure.pickle --input-language <ISO 639-1 code>
        # python sentiments.py --command create-structure --input-text "input text" --output-pickle structure.pickle --input-language <ISO 639-1 code>
        ## --email: optional and used for translation provider MyMemory
        ##          cf. https://translate-python.readthedocs.io/en/latest/providers.html
        ##              https://mymemory.translated.net/doc/usagelimits.php
        paragraphs = process_text(args)
        paragraphs = get_sentences(paragraphs,args)
        paragraphs = create_structure(paragraphs,args)
        pickle.dump(paragraphs,open(args['output-pickle'],'wb'))
    elif args['command'] == 'get-translations':
        # python sentiments.py --command get-translations --email your@email --input-pickle structure.pickle --output-pickle translated.pickle --input-language <ISO 639-1 code> --working-language <iso 639-1 code> --output-dictionary output-dict.pickle
        # It's useful second time to use already generated dicitonary :)
        # python sentiments.py --command get-translations --email your@email --input-pickle structure.pickle --output-pickle translated.pickle --input-language <ISO 639-1 code> --working-language <iso 639-1 code> --input-dictionary input-dict.pickle --output-dictionary output-dict.pickle
        paragraphs = pickle.load(open(args['input-pickle'],'rb'))
        paragraphs, args, data = update_paragraphs(paragraphs,args,data)
        pickle.dump(args['dict'],open(args['output-dictionary'],'wb'))
        pickle.dump(paragraphs,open(args['output-pickle'],'wb'))
    elif args['command'] == 'get-sentiments':
        # python sentiments.py --command get-sentiments --input-pickle translated.pickle --output-pickle sentiments.pickle --input-language <ISO 639-1 code> --working-language <iso 639-1 code>
        paragraphs = pickle.load(open(args['input-pickle'],'rb'))
        paragraphs, args, data = update_paragraphs(paragraphs,args,data)
        pickle.dump(paragraphs,open(args['output-pickle'],'wb'))
    elif args['command'] == 'make-domains':
        # python sentiments.py --command make-domains --input-pickle sentiments.pickle
        paragraphs = pickle.load(open(args['input-pickle'],'rb'))
        paragraphs, args, data = update_paragraphs(paragraphs,args,data)
        for domain in data['domains']:
            print(data['domains'][domain], domain)
    elif args['command'] == 'make-sentiments':
        # python sentiments.py --command make-sentiments --input-pickle sentiments.pickle --output-csv sentiments.csv
        paragraphs = pickle.load(open(args['input-pickle'],'rb'))
        paragraphs, args, data = update_paragraphs(paragraphs,args,data)
        write_csv(args,data)
    elif args['command'] == 'fix-dict':
        # Check the function "fix_dict" and make your own rules for cleaning the dictionary, depending of how it's been created.
        # 
        # python sentiments.py --command fix-dict --input-dictionary dict-in.pickle --output-dictionary dict-out.pickle
        args, data = fix_dict(args,data)
        pickle.dump(args['dict'],open(args['output-dictionary'],'wb'))
if __name__ == "__main__":
    main()