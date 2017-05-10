# Michelle Morales
# Script performs linguistic analysis (support included for English, Spanish, German)

from __future__ import division
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import pandas, numpy, os, subprocess, os.path, string


def bag_of_words(document):
    all_words = []
    words = document.lower().strip().split()
    for w in words:
         all_words.append(w)
    print('Done processing words!')
    bag = []
    for w in all_words:
        count = all_words.count(w)
        if count > 5 and w not in bag:
            bag.append(w)
    bag = sorted(bag)
    print('Done creating bag!')
    return bag


def english_parse(transcript, parser_dir):
    os.chdir(r"%s" % parser_dir)
    command = "echo '%s' | syntaxnet/demo.sh" % transcript
    try:
        output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
    except:
        output = False
    return output


def spanish_parse(transcript, parser_dir):
    os.chdir(r"%s" % parser_dir)
    command = "echo '%s' | syntaxnet/models/parsey_universal/parse.sh /Users/morales/GitHub/models/Spanish" % transcript
    try:
        output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
    except:
        output = False
    return output


def german_parse(transcript, parser_dir):
    os.chdir(r"%s" % parser_dir)
    command = "echo '%s' | syntaxnet/models/parsey_universal/parse.sh /Users/morales/GitHub/models/German" % transcript
    try:
        output = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE).stdout.read()
    except:
        output = False
    return output


def dependency_distance(conll_df):
    """ Computes dependency distance for dependency tree. Based off of:
    Pakhomov, Serguei, et al. "Computerized assessment of syntactic complexity
    in Alzheimers disease: a case study of Iris Murdochs writing."
    Behavior research methods 43.1 (2011): 136-144."""
    ID = numpy.array([int(x) for x in conll_df['ID']])
    HEAD = numpy.array([int(x) for x in conll_df['HEAD']])
    diff = abs(ID - HEAD)
    total_distance = sum(diff)
    return total_distance


def load_tags():
    # Load universal POS tag set - http://universaldependencies.org/u/pos/all.html
    tags = "ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".strip().split()
    return tags


def tag_count(df):
    tag_count = []
    tags = load_tags()
    for tag in sorted(tags):
        df_tags = df['UPOS'].values.tolist()
        count = df_tags.count(tag)
        tag_count.append(count)
    return tag_count


def vader(sentence):
    # VADER (Valence Aware Dictionary and sEntiment Reasoner) - https://github.com/cjhutto/vaderSentiment
    # Hutto, C.J. & Gilbert, E.E. (2014). VADER: A Parsimonious Rule-based Model for Sentiment Analysis of Social Media
    # Text. Eighth International Conference on Weblogs and Social Media (ICWSM-14). Ann Arbor, MI, June 2014.
    sid = SIA()
    ss = sid.polarity_scores(sentence)
    return [ss['neg'], ss['neu'], ss['pos'], ss['compound']]

# def anger():
#
# def hurt():
#
# def trauma():


def tree_feats(conll):
    conll_lines = conll.strip().split('\n')
    conll_table = [line.split('\t') for line in conll_lines]
    df = pandas.DataFrame(conll_table,
                          columns=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS',
                                   'MISC'])
    # POS tag count
    pos_feats = tag_count(df)
    # Unique number of pos tags
    univ_tag = len(set(df['UPOS'].values))
    # Dependency distance
    distance = dependency_distance(df)
    heads = df['HEAD'].values
    # Tree depth
    levels = len(set(heads))
    return [levels, distance, univ_tag] + pos_feats


def get_syntax(document, lang, parser_dir, output):
    openF = open(os.path.join(output), 'w')
    syntax_header = 'word_count,avg_wordlen,levels,distance,univ_tag,%s' % (','.join(load_tags()))
    sentiment_header = 'neg,neu,pos,compound'
    header = '%s,%s\n' % (syntax_header, sentiment_header)
    openF.write(header)
    feature_list = []
    sentiment_feats = vader(document)
    words = document.lower().strip().split()
    word_count = len(words)
    avg_wordlen = sum([len(w) for w in words]) / len(words)
    if word_count > 3:
        if lang == 'german':
            try:
                conll = german_parse(sentence, parser_dir)
                syntax_feats = tree_feats(conll)
            except:
                syntax_feats = 20 * [0]
        elif lang == 'spanish':
            try:
                conll = spanish_parse(sentence, parser_dir)
                syntax_feats = tree_feats(conll)
            except:
                syntax_feats = 20 * [0]
        elif lang == 'english':
            try:
                conll = english_parse(sentence, parser_dir)
                syntax_feats = tree_feats(conll)
            except:
                syntax_feats = 20 * [0]
    else:
        syntax_feats = 20 * [0]
    feats = [word_count, avg_wordlen] + syntax_feats + sentiment_feats
    features = ','.join([str(f) for f in feats])
    feature_list.append(features)

    for s in feature_list:
        openF.write(s + '\n')
    print ('Done processing document. Linguistic features extracted!')


def get_feats(document, bag, lang, parser_dir, output):
    openF = open(os.path.join(output), 'w')
    syntax_header = 'word_count,avg_wordlen,levels,distance,univ_tag,%s' % (','.join(load_tags()))
    sentiment_header = 'neg,neu,pos,compound'
    bag_header = ','.join(bag).encode('ascii', 'ignore')
    header = '%s,%s,%s\n' % (bag_header, syntax_header, sentiment_header)
    openF.write(header)
    feature_list = []
    for sentence in document:
        words = sentence.strip().split()
        word_count = len(words)
        sentiment_feats = vader(sentence)
        feats = []
        if word_count > 0:
            for w in bag:
                count = words.count(w)
                feats.append(float(count) / word_count)
            if word_count > 3:
                if lang == 'german':
                    conll = german_parse(sentence, parser_dir)
                elif lang == 'spanish':
                    conll = spanish_parse(sentence, parser_dir)
                elif lang == 'english':
                    conll = english_parse(sentence, parser_dir)
            else:
                syntax_feats = 22 * [0]
            if 'conll' in locals():
                conll_lines = conll.strip().split('\n')
                conll_table = [line.split('\t') for line in conll_lines]
                df = pandas.DataFrame(conll_table,
                                      columns=['ID', 'FORM', 'LEMMA', 'UPOS', 'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS',
                                               'MISC'])
                # POS tag count
                pos_feats = tag_count(df)
                # Unique number of pos tags
                univ_tag = len(set(df['UPOS'].values))
                # Dependency distance
                distance = dependency_distance(df)
                heads = df['HEAD'].values
                # Tree depth
                levels = len(set(heads))
                # Average word length
                avg_wordlen = sum([len(w) for w in words]) / len(words)
                syntax_feats = [word_count, avg_wordlen, levels, distance, univ_tag] + pos_feats
            else:
                syntax_feats = 22 * [0]
        else:
            feats = len(bag) * [0]

        feats = feats + syntax_feats + sentiment_feats
        features = ','.join([str(f) for f in feats]).encode('ascii', 'ignore')
        feature_list.append(features)

    for s in feature_list:
        openF.write(s + '\n')
    print ('Done processing document. Linguistic features extracted!')

