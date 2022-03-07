import spacy, spacy_udpipe, spacy_wordnet, spacy, nltk, re, csv, time, pickle, sys, os
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
        'specific domains': {},
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
    args = get_arg('--domain-name',args)
    ## TODO: specify different input and working languages for:
    ##       - spacy
    ##       - spacy_udpipie
    ##       - translate module
    ##       - spacy/nltk interface to wordnet

    # dictionaries
    args = get_arg('--input-dictionary',args)
    if 'input-dictionary' in args:
        args['dict'] = pickle.load(open(args['input-dictionary'],'rb'))
    else:
        args['dict'] = {}
    args = get_arg('--output-dictionary',args)
    if 'output-dictionary' not in args:
        args['output-dictionary'] = 'dictionary-output.pickle'
    args = get_arg('--input-domain-file-type',args)
    args = get_arg('--input-file',args)
    args = get_arg('--domain-dictionary',args)
    
    # download models
    spacy_udpipe.download(args['input-language'])
    spacy_udpipe.download(args['working-language'])
    nltk.download('wordnet')
    nltk.download('sentiwordnet')
    
    # pointers
    args['nlp-input'] = spacy_udpipe.load(args['input-language'])
    ## for working language -- i.e. English -- main spacy model is more useful
    ### Spacy 2.x
    #args['nlp-working'] = spacy.load(args['working-language'])
    #args['nlp-working'].add_pipe(WordnetAnnotator(args['nlp-working'].lang), after='tagger')
    #args['nlp-working'] = spacy_udpipe.load(args['working-language'])
    ## Spacy 3.x
    args['nlp-working'] = spacy.load("en_core_web_lg")
    args['nlp-working'].add_pipe("spacy_wordnet", after='tagger', config={'lang': args['nlp-working'].lang})
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
        print("get sentences:", p, '/', pmax)
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

def get_sentences(paragraphs,args):
    pmin = 0
    pmax = len(list(paragraphs.keys()))
    prev = ''
    for p in range(pmin,pmax):
        pkey = list(paragraphs.keys())[p]
        doc = args['nlp-input'](paragraphs[pkey]['text'])
        sn = 0
        print("get sentences, paragraphs:", p, '/', pmax)
        for token in doc:
            if token.is_sent_start:
                sentence = str(token.sent)
                paragraphs[pkey]['sentences'][sn] = {
                    'text': sentence,
                    'tokens': {},
                }
                sn += 1
    return paragraphs

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

def update_paragraphs(paragraphs,args,data):
    p = 0
    nrel = 0
    start = 386
    stop = len(paragraphs)
    stop = 387
    for pkey in paragraphs:
        if p <= start:
            next
        if p >= stop:
            break
        print(pkey, len(paragraphs))
        data['paragraphs'][pkey] = {
            'domains': {},
            'sentences': {},
        }
        print(paragraphs[pkey]['sentences'])
        slist = list(paragraphs[pkey]['sentences'].keys())
        print(slist)
        smin = 0
        smax = max(slist)
        for s in range(smin,smax):
            skey = slist[s]
            data['paragraphs'][pkey]['sentences'][skey] = {
                'domains': {},
                'tokens': {},
            }
            doc = args['nlp-input'](paragraphs[pkey]['sentences'][skey]['text'])
            print(doc)
            tlist = list(paragraphs[pkey]['sentences'][skey]['tokens'].keys())
            tmin = 0
            tmax = max(tlist)
            for t in range(tmin,tmax):
                tkey = tlist[t]
                token = doc[tkey]
                working_entity = paragraphs[pkey]['sentences'][skey]['tokens'][tkey]['working entity']
                
        p += 1
    return paragraphs,args,data

def main():
    args, data = get_args()
    if args['command'] == 'create-structure':
        # 1. Create structure
        # 2. Get translations
        # 3. Get sentiments
        # 4. Make wordnet domains
        # python semantic_analysis.py --command create-structure --input input/file --output-pickle structure.pickle --input-language <ISO 639-1 code>
        # python semantic_analysis.py --command create-structure --input-text "input text" --output-pickle structure.pickle --input-language <ISO 639-1 code>
        paragraphs = process_text(args)
        paragraphs = get_sentences(paragraphs,args)
        paragraphs = create_structure(paragraphs,args)
        pickle.dump(paragraphs,open(args['output-pickle'],'wb'))
    elif args['command'] == 'analyze-sentences':
        # python semantic_analysis.py --command analyze-sentences --input-pickle sa-structure.pickle --output-pickle sa-sentences.pickle --input-language <ISO 639-1 code> --working-language <iso 639-1 code>
        paragraphs = pickle.load(open(args['input-pickle'],'rb'))
        paragraphs, args, data = update_paragraphs(paragraphs,args,data)
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
        # python sentiments.py --command make-domains --input-pickle psycho.pickle
        paragraphs = pickle.load(open(args['input-pickle'],'rb'))
        paragraphs, args, data = update_paragraphs(paragraphs,args,data)
        for domain in data['domains']:
            print(data['domains'][domain] + "," + domain)
    elif args['command'] == 'make-sentiments':
        # python sentiments.py --command make-sentiments --input-pickle sentiments.pickle --output-csv sentimens.csv
        paragraphs = pickle.load(open(args['input-pickle'],'rb'))
        paragraphs, args, data = update_paragraphs(paragraphs,args,data)
        write_csv(args,data)
    elif args['command'] == 'make-domain':
        # python sentiments.py --command make-domain --input-pickle sentiments.pickle --output-pickle domain.pickle --input-language <ISO 639-1 code> --working-language <iso 639-1 code> --domain-dict domain-dictionary.pickle --domain-language <input|output> --domain-grammar-type <form|lemma> --domain-name <domain_name>
        # IMPORTANT: You have to create your own domain dictionary, check "create-domain-dict".
        paragraphs = pickle.load(open(args['input-pickle'],'rb'))
        paragraphs, args, data = update_paragraphs(paragraphs,args,data)
        pickle.dump(paragraphs,open(args['output-pickle'],'wb'))
    elif args['command'] == 'fix-dict':
        # Check the function "fix_dict" and make your own rules for cleaning the dictionary, depending of how it's been created.
        # 
        # python sentiments.py --command fix-dict --input-dictionary dict-in.pickle --output-dictionary dict-out.pickle
        args, data = fix_dict(args,data)
        pickle.dump(args['dict'],open(args['output-dictionary'],'wb'))
    elif args['command'] == 'create-domain-dict':
        # python sentiments.py --command create-domain-dict --input-domain-file-type <input-type> --input-file <input-file> --output-dictionary domain-dict.pickle
        # * input-type: particular type, described inside of particular function starting with the name "parse_domain_<type>"
        domain_dict, args, data = create_domain_dict(args,data)
        pickle.dump(domain_dict,open(args['output-dictionary'],'wb'))
        

if __name__ == "__main__":
    main()
