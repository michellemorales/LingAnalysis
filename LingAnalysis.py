# Michelle Morales 2017
# Script performs linguistic analysis (support included for English, Spanish, German)
# Linguistic analysis outpugts a set of features, including:
# word count, average word length, xx syntactic features
# 4 sentiment features, and crisis language frequency


from __future__ import division
from nltk.tokenize import TweetTokenizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import numpy as np
import pandas, os, subprocess, os.path, scipy


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
    return total_distance1


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


def crisis_words(document):
    crisis_language = [u'i', u'to', u'and', u'the', u':', u'my', u'it', u'a', u"i'm", u'of', u'me', u'so', u'that',
                       u'in', u'just', u'is',
                       u'for', u'but', u'this', u'have', u'with', u"don't", u'like', u'know', u'do', u'on', u'not',
                       u'be', u'feel', u'at',
                       u're', u'can', u'was', u'feeling', u'am', u'myself', u'all', u'up', u'?', u'now', u'want',
                       u'get', u'what', u'her',
                       u'because', u'really', u'or', u'out', u'pos', u"it's", u'neg', u'time', u'about', u'into',
                       u'even', u"i've",
                       u"can't", u'how', u'when', u'if', u'things', u'she', u'right', u'positive', u'been', u'much',
                       u'negative', u'you',
                       u'need', u'they', u'go', u'why', u'from', u'people', u'going', u'are', u'as', u'!', u'anymore',
                       u'help', u'no',
                       u'some', u'turning', u'through', u'work', u'positives', u'negatives', u'one', u'too', u'today',
                       u'had', u'try',
                       u'them', u'anything', u'will', u'hard', u'there', u'day', u'back', u'scared', u'life', u'good',
                       u'trying', u'has',
                       u'any', u'being', u'more', u'keep', u'over', u'down', u'away', u'everything', u'getting',
                       u'psych', u'make',
                       u'only', u'bad', u'where', u'"', u'find', u'last', u')', u'its', u'something', u'(', u'an',
                       u'would', u'by',
                       u'take', u'see', u'then', u'talk', u'very', u'think', u'dont', u'hate', u'here', u'got', u'days',
                       u'these',
                       u'home', u'again', u'sick', u'still', u'feels', u'doing', u'cant', u"didn't", u'after',
                       u'better', u'best',
                       u'always', u'which', u'off', u'never', u'way', u'maybe', u'nothing', u'could', u'felt',
                       u'around', u'having',
                       u'prac', u'he', u'tired', u'give', u'-', u'thoughts', u'someone', u'done', u'twittro', u'long',
                       u'did', u'tried',
                       u'first', u'sure', u'self', u'also', u'told', u'little', u'thing', u"i'll", u'point', u'made',
                       u'hospital',
                       u'before', u'cope', u'said', u'worse', u'family', u'few', u'friends', u'year', u'anyone',
                       u'world', u'gp', u'let',
                       u'apt', u'who', u'end', u'tomorrow', u'enough', u'week', u'while', u'other', u'sleep', u'yet',
                       u'/', u'head',
                       u'leave', u'say', u'most', u'each', u'wanted', u'another', u'anxiety', u'im', u'alone', u'come',
                       u'without',
                       u"that's", u'part', u'during', u'their', u'safe', u'mind', u'love', u'place', u'than', u'pretty',
                       u'person',
                       u'able', u'well', u'talking', u'care', u'2', u'morning', u'stupid', u'lot', u'strong', u'since',
                       u'ever', u'tell',
                       u"doesn't", u'fight', u'working', u'years', u'looking', u"couldn't", u'same', u'either',
                       u"wasn't", u'advice',
                       u'mental', u'guess', u'least', u'else', u'understand', u'look', u'bit', u"'", u'tonight',
                       u'should', u'hide',
                       u'we', u'thanks', u'happy', u'inside', u'making', u'course', u'were', u'start', u'stop',
                       u'times', u'thought',
                       u'does', u'sorry', u'worth', u'school', u'new', u'feelings', u'went', u'seem', u'though',
                       u'depressed', u'left',
                       u'change', u'write', u'those', u'different', u'doctor', u'seems', u'ok', u'your', u'lost',
                       u'use', u'next',
                       u'calm', u'already', u'mum', u'asked', u'night', u'managed', u'friend', u'wanting',
                       u'struggling', u'money',
                       u'broken', u'low', u'past', u'tafe', u'ask', u'him', u'moment', u'couple', u'ago', u'depression',
                       u'started',
                       u'redhead', u'job', u'become', u'stay', u'room', u'turn', u'wrong', u'weeks', u'normal',
                       u'months', u'support',
                       u'call', u'run', u'live', u'figure', u'may', u'thinking', u'read', u'flat', u'found',
                       u'everyone', u'anxious',
                       u'health', u'eheadspace', u'spent', u'body', u'wish', u'kinda', u"i'd", u'break', u'matter',
                       u'old', u'deal',
                       u'seeing', u'angry', u'confused', u'such', u'guys', u'remember', u'happened', u'half',
                       u'fucking', u'pain',
                       u'many', u'running', u'deep', u'put', u'actually', u'late', u'energy', u'beyond', u'motivation',
                       u'set',
                       u'recently', u'whole', u'crying', u'meds', u'others', u'control', u'instead', u'move',
                       u'problem', u'every',
                       u'mess', u'mean', u'needed', u"there's", u'hopefully', u'kept', u'later', u'shit', u'test',
                       u'happen', u'mood',
                       u'two', u'none', u'rather', u'coming', u'house', u'sometimes', u'overwhelmed', u'yesterday',
                       u'helping', u'once',
                       u'giving', u'thats', u'manage', u'university', u'face', u'longer', u'choices', u'almost',
                       u'gets', u'emotionally',
                       u'pressure', u'stories', u'book', u'study', u'ro', u'hopeless', u'onto', u'annoyed', u'wasting',
                       u'stuck',
                       u'until', u'lonely', u'worried', u'yourself', u'between', u'finish', u'comes', u'real', u'hey',
                       u'important',
                       u'taken', u'might', u"isn't", u'harm', u'kind', u'close', u'continue', u'psychosis', u'great',
                       u'makes', u'sort',
                       u"won't", u'cry', u'fact', u'words', u'stuff', u'walk', u'difficult', u'*', u'monday', u"aren't",
                       u'stressed',
                       u'writing', u'easier', u'therapy', u'struggle', u'nobody', u'saying', u'busy', u'explain',
                       u'despite', u'contact',
                       u'afternoon', u'true', u'focus', u'guy', u'entirely', u'voices', u'especially', u'outside',
                       u'parent', u'wants',
                       u'wake', u'buying', u'gone', u'follow', u'hours', u'sense', u'isnt', u'coping', u'reason',
                       u'terrible', u'story',
                       u'failed', u'probably', u"wouldn't", u'straight', u'often', u'phone', u'question', u'nice',
                       u'helps', u'breathing',
                       u'meant', u'high', u'1', u'shut', u'bed', u'anyway', u'bring', u'hope', u'means', u'email',
                       u'ball', u'fighting',
                       u'happens', u'own', u'hi', u'suicidal', u'music', u'keeps', u'fix', u'silly', u'blergh', u'men',
                       u'harming',
                       u'mother', u'easy', u'hear', u'father', u'die', u'fragile', u'reply', u'unsure', u'shift',
                       u'weird', u'counsellor',
                       u'wont', u'living', u'seriously', u'quite', u'open', u'given', u'10', u'3', u'soon', u'wonder',
                       u'rest', u'4',
                       u'early', u'starting', u'decisions', u'list', u'small', u'experience', u'social', u'6', u'hour',
                       u'strategies',
                       u'everyday', u'okay', u'messed', u'hurting', u'idea', u'miss', u'relationship', u'doctors',
                       u'useless',
                       u'registers', u'fine', u'#', u'please', u'attention', u'expensive', u'due', u'teacher', u'damn',
                       u'treatment',
                       u'used', u'spend', u'big', u'crap', u'intense', u'repair', u'forward', u'supervisor', u'curl',
                       u'helpful',
                       u"she's", u'send', u'psychiatrist', u'doubt', u'women', u'completely', u"you're", u'absolutely',
                       u'ones',
                       u'bought', u'harder', u'uni', u'identity', u'empty', u'second', u'+', u'path', u'spoke',
                       u'aware', u'returned',
                       u'answer', u'badly', u'lay', u'fit', u'took', u'ran', u'seen', u'societyim', u'appreciated',
                       u'ill', u'disappear',
                       u'possible', u'asking', u'post', u'khl', u'reality', u'happening', u'push', u':/', u"it'll",
                       u'falling',
                       u'usually', u'extra', u'crisis', u'club', u'following', u'awesome', u'crashed', u'breaking',
                       u'whats', u'no2',
                       u'ring', u'boyfriend', u'sophie', u'knew', u'forever', u'cannot', u'nearly', u'hurts',
                       u'article', u'situation',
                       u'check', u'mentally', u'listening', u'thinks', u'terrified', u'oh', u'weekend', u'exhausted',
                       u'5', u'worst',
                       u'reminding', u'full', u'door', u'broke', u'numb', u'playing', u'hell', u'food', u'rant',
                       u'friendship',
                       u'alcohol', u'behind', u'believe', u'gotten', u'finding', u'towards', u'extremely', u'lying',
                       u'ugh', u'shits',
                       u'paranoid', u'hit', u'sex', u'currently', u'case', u'both', u'gay', u"what's", u'ready',
                       u'older', u'questions',
                       u'using', u'parents', u'ignore', u'patient', u'called', u'age', u'putting', u'telling',
                       u'pushing', u'sister',
                       u'fuck', u'sore', u'process', u'struggled', u'panic', u'cutting', u'books', u'reached',
                       u'patients', u'fear',
                       u'calling', u'certain', u'assessments', u'site', u'cried', u'less', u'lose', u'pointless',
                       u'eat', u'girl',
                       u'chance', u'hating', u'enjoy', u'forget', u'bipolar', u'above', u'tips', u'brought', u'type',
                       u'breathe', u'hold',
                       u'damage', u'faulty', u'third', u'side', u'mine', u'chest', u'\ud83d', u'letter', u'doc',
                       u'ending', u'eventually',
                       u'release', u'fair', u'depressive', u'continually', u'tough', u'speak', u'expected', u'doubts',
                       u'worked',
                       u'voice', u'clear', u'hand', u'tho', u'grades', u'gave', u'judge', u'apart', u'dead',
                       u'becoming', u'failure',
                       u'scream', u'unwell', u'hurt', u'together', u'cares', u'battle', u'answered', u'finally',
                       u'keeping', u'choice',
                       u'trouble', u'cool', u'brother', u'walked', u'says', u'insignificant', u'dealing', u'rough',
                       u'stressful',
                       u'ground', u'understanding', u'sending', u'guilty', u'diagnosed', u'studies', u'nowhere',
                       u'tears', u'heart',
                       u'allowed', u'date', u'man', u'stress', u'thank', u'troubles', u'policy', u'emailed', u'nor',
                       u'square', u'dunno',
                       u'frequent', u'emotion', u'12', u'gotta', u'cards', u'illness', u'saw', u'online', u'managing',
                       u'crashing',
                       u'horrible', u'regarding', u'attitude', u'worries', u'physically', u'us', u'ways', u'notice',
                       u'afterwards',
                       u'cancer', u'medication', u'cycling', u'carer', u'capable', u'changing', u'aspect', u'throw',
                       u'preference',
                       u'everytime', u'mixed', u'cared', u'complete', u'ended', u'practically', u'expect', u'selfish',
                       u'chemo',
                       u'children', u'enjoying', u'emotions', u'amount', u'semester', u'takes', u'basically', u'sucks',
                       u'sent', u'share',
                       u'accept', u'drained', u'blood', u'short', u'pay', u'smiling', u"haven't", u"you've", u'foot',
                       u'stopped',
                       u'connect', u'belief', u'beginning', u'thread', u'number', u'feet', u'lie', u'reach', u'plan',
                       u'sector', u'his',
                       u'likes', u'whom', u'meeting', u'woke', u'honest', u'purpose', u'dark', u'edge', u'letting',
                       u'internal',
                       u'easter', u'suffering', u'knock', u'apathetic', u'step', u'within', u'properly', u'chat', u'%',
                       u'journalling',
                       u'far', u'hello', u'stable', u'drinking', u'uncomfortable', u'leaving', u'stand', u'choose',
                       u'truth', u'our',
                       u'math', u'bee', u'apparently', u'afraid', u'centre', u'upset', u'son', u'planning',
                       u'knowledge', u'written',
                       u'progress', u'frustrated', u'tuesday', u'slowly', u'waste', u'headspace', u'fault',
                       u'frustrating', u'ringing',
                       u'helped', u'muscle', u'usual', u'logged', u'increased', u'whatever', u'taking', u'alive',
                       u'screwed', u'reaching',
                       u'45', u'lately', u'incredibly', u'class', u'risk', u'screaming', u'street', u'resilient',
                       u'kids', u'changed',
                       u'changes', u'forums', u'unit', u'survive', u'warm', u'movies', u'amazing', u'bringing',
                       u'downs', u'effects',
                       u'series', u'wednesday', u'forgot', u'god', u'adapt', u'threatened', u'free', u'appointments',
                       u'atm', u'shush',
                       u'hearing', u'top', u'cycle', u'serve', u'speaking', u'seek', u'car', u'strength', u'isolated',
                       u'regular',
                       u'medical', u'breather', u'yeh', u'judged', u'rice', u'professionals', u'watch', u'coast',
                       u'respond', u'unworthy',
                       u'decided', u'result', u'holding', u'fail', u'disturbing', u'lots', u'indecisive',
                       u'rollercoaster', u'however',
                       u'news', u'picking', u'behaviour', u'30mins', u'country', u'gathering', u'trust', u'bathroom',
                       u'three', u'deeper',
                       u'drugs', u'sunshine', u'writting', u'sad', u'save', u'air', u'mistake', u'emergency',
                       u'potentially', u'hahah',
                       u'babies']
    tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)
    words = tknzr.tokenize(document.lower().replace('.', '').replace(',', ''))
    count = 0
    for w in crisis_language:
        w_count = words.count(w)
        count += w_count
    try:
        freq = count / len(words)
    except:
        freq = 0
    return freq


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


def write_header(csv):
    openF = open(csv, 'a')
    syntax_header = 'word_count,avg_wordlen,levels,distance,univ_tag,%s' % (','.join(load_tags()))
    sentiment_header = 'neg,neu,pos,compound,crisis_language'
    header = '%s,%s\n' % (syntax_header, sentiment_header)
    openF.write(header)
    openF.close()


def get_feats(sentence, lang, parser_dir, output):
    openF = open(os.path.join(output), 'a')
    # syntax_header = 'word_count,avg_wordlen,levels,distance,univ_tag,%s' % (','.join(load_tags()))
    # sentiment_header = 'neg,neu,pos,compound,crisis_language'
    # header = '%s,%s\n' % (syntax_header, sentiment_header)
    # openF.write(header)
    feature_list = []
    sentiment_feats = vader(sentence)
    crisis_score = crisis_words(sentence)
    words = sentence.lower().strip().split()
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
    feats = [word_count, avg_wordlen] + syntax_feats + sentiment_feats + [crisis_score]
    features = ','.join([str(f) for f in feats])
    feature_list.append(features)

    for s in feature_list:
        openF.write(s + '\n')
        # print ('Done processing document. Linguistic features extracted!')


def fusion(file_name):
    combo_file = file_name.replace('.csv', '_combined.csv')
    stats_names = ['max', 'min', 'mean', 'median', 'std', 'var', 'kurt', 'skew', 'percentile25', 'percentile50',
                   'percentile75']
    all_feats = []
    all_names = []
    df = pandas.read_csv(file_name, header='infer')
    feature_names = df.columns.values
    for feat in feature_names:
        # Feature vector
        vals = df[feat].values
        # Run statistics
        maximum = np.nanmax(vals)
        minimum = np.nanmin(vals)
        mean = np.nanmean(vals)
        median = np.nanmedian(vals)
        std = np.nanstd(vals)
        var = np.nanvar(vals)
        kurt = scipy.stats.kurtosis(vals)
        skew = scipy.stats.skew(vals)
        percentile25 = np.nanpercentile(vals, 25)
        percentile50 = np.nanpercentile(vals, 50)
        percentile75 = np.nanpercentile(vals, 75)
        names = [feat.strip() + "_" + stat for stat in stats_names]
        feats = [maximum, minimum, mean, median, std, var, kurt, skew, percentile25, percentile50, percentile75]
        for n in names:
            all_names.append(n)
        for f in feats:
            all_feats.append(f)
    new_file = open(combo_file, 'w')
    new_file.write(','.join(all_names) + '\n')
    new_file.write(','.join([str(mm) for mm in all_feats]))
    new_file.close()
    print 'Done combining sentences, new csv saved: %s' % combo_file
