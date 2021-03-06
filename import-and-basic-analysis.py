
"""
input:
directory which contains documents of various file formats. supports languages German, English as well as the latin languages

output:
pandas dataframe with basic analysis and linguistic features for each document (or each language of each document)
html graphical summary of the dataframe

The script does basically the following:
1: let the user decide from which directory documents should be imported
2: transform pdfs, docx and some other formated documents into txt
3: let the user specify if single documents in the directory are written in more than one language
4: depending on the answer to 3, the language detection is applied on the entire document or on each sentence of the doc individually (= considerably more costly and not always accurate, especially if the document is not well parsed)
5: apply some cleaning functions, adapted from the package nlpre
6: define textname, language, cleaned text (= subselect certain linguistic elements), nouns, verbs, adjectives and named entities
7: store this data along with the original texts in a pandas dataframe in the directory from which the documents are parsed
8: output a graphical summary of the documents with the package scattertext (does not work currently if only one language is predicted!)

parameters:
supported_languages # set to = ["English", "German",
					   "Spanish", "Portuguese", "French", "Italian"] 
default_language # set to = "English"
useful_characters # set to = string.printable + \
	'äöüÄÖÜéÉèÈáÁàÀóÓòÒúÚùÙíÍìÌñÑãÃõÕêÊâÂîÎôÔûÛ' 
parsable_extensions # set to ['.csv', '.doc', '.docx', '.eml', '.epub', '.json',
					   '.msg', '.odt', '.ogg', '.pdf', '.pptx', '.rtf', '.xlsx', '.xls']
# set to = 2000000  # default would be 1m
minlength_of_text # set to = 100  # if textlen is lower, we ignore this text
POS_blacklist # set to = ["PUNCT", "PART", "SYM", "SPACE",
				 "DET", "CONJ", "CCONJ", "ADP", "INTJ", "X", ""] 
path # defined by function get_path

"""

import string
import re
import sys
import collections
import os
from os import system, listdir
import numpy as np
import pandas as pd
import pycountry
from pycountry import languages
import pprint
from pprint import pprint
import textract
import fitz
import scattertext as st
import spacy
import spacy_langdetect
from spacy import matcher
from spacy import tokens
from spacy.matcher import Matcher
from spacy.tokens import Token
from spacy_langdetect import LanguageDetector


# check if python and pip as well as the executable are in line
os.system('which python')
os.system('which pip')
sys.executable


def main():
    ######## defining the parameters ##############
    supported_languages = ["English", "German",
                           "Spanish", "Portuguese", "French", "Italian"]
    # this is required, otherwise we get weird languages for long and untidy documents
    
    default_language = "English"
	# making English the default, which is used when no language is detected
	
    useful_characters = string.printable + \
        'äöüÄÖÜéÉèÈáÁàÀóÓòÒúÚùÙíÍìÌñÑãÃõÕêÊâÂîÎôÔûÛ'  # filtering the characters of the texts
	
    parsable_extensions = ['.csv', '.doc', '.docx', '.eml', '.epub', '.json',
                           '.msg', '.odt', '.ogg', '.pdf', '.pptx', '.rtf', '.xlsx', '.xls']
    """ '.gif', '.jpg', '.mp3', '.tiff', '.wav', '.ps', '.html' """
    # the extensions which we try to parse to text
	
    doc_maxlength = 2000000  # default would be 1m which is the maximum length of a document in spacy
	
    minlength_of_text = 100  # if textlen is lower, we ignore this text
	
    POS_blacklist = ["PUNCT", "PART", "SYM", "SPACE",
                     "DET", "CONJ", "CCONJ", "ADP", "INTJ", "X", ""]  # we filter out these token-types
    
    parsers = [titlecaps, token_replacement, url_replacement] # the parsing functions used
	
    path = get_path(parsable_extensions) # Determining the directory from which to import documents

	
    ######## initiating the pipelines ##############
    multilanguage, nlp = decide_language_detection(
        path, supported_languages, default_language)
	# let the user determine if he wants to use the sentence-wise
    # language detection or the document-wise. The sentence-wise allows
    # to ignore parts of docs that contain text not of interest, such
    # as metadata in english for a german document

    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
	# add the language detector to the spacy nlp pipeline

    pdf_to_text(path, parsable_extensions)
	# safe all non-text-documents with parsable extensions to txt-file

    doc_list = documents_dataframe(path, minlength_of_text, doc_maxlength,
                                   nlp, multilanguage, default_language, supported_languages, parsers, useful_characters)
	# create a document list with detected language, filename, textname and text
	
    df_doclist = get_all_text_info(
        doc_list, supported_languages, POS_blacklist, doc_maxlength)
	# use the document list to retrieve various basic information from the text 

    print(df_doclist.shape)
    df_doclist.to_pickle(path+"/df_doclist.pkl") # saving the data frame to path
	df_doclist = pd.read_pickle("./df_doclist.pkl") # and opening it

    ######## an example what we can do with df_doclist: create scattertext html graph ##############
    nlp = get_spacy_tokenizer("German", supported_languages, bigmodel_required=False)
    try:
        corpus = st.CorpusFromPandas(
            df_doclist, category_col='Sprache', text_col='bereinigter Text', nlp=nlp).build()
        # actually, the category should be sth else than the language, because if we take the language
        # we don't get a big overlap between the categories! so ideally we would train a textcat model
        # BEFORE and use the category-column inside pd_doclist as the category

        term_freq_df = corpus.get_term_freq_df()
        term_freq_df['German words'] = corpus.get_scaled_f_scores('German')
        pprint("Most unique German words are" + list(term_freq_df.sort_values(
            by='German words', ascending=False).index[:20]))

        html = st.produce_scattertext_explorer(
            corpus,
            category='German',  # not_category_name='English',
            minimum_term_frequency=5, metadata=corpus.get_df()['Textname'])

        fn = path+path.split("/")[-2]+'-Auswertung.html'
        open(fn, 'wb').write(html.encode('utf-8'))
        print('Open ' + fn + ' in Chrome or Firefox.')
    except:
        print("Only one language was given. Hence, scatterplot does not work a.t.m.")


######## Define Functions ##############


def documents_dataframe(path, minlength_of_text, doc_maxlength, nlp, multilanguage, default_language, supported_languages, parsers, useful_characters):
    '''outputs an array which contains rows of the documents' texts, corresponding filenames, textnames and languages'''

    filenames_list_beforeprocess = [x for x in os.listdir(
        path) if x.endswith(".txt")]

    filenames_list_final = []
    textnames_list_final = []
    filenames_list = []
    textnames_list = []
    languagelist = []
    textslist = []

    for i in range(len(filenames_list_beforeprocess)):
        filename = str(filenames_list_beforeprocess[i])
        textname = get_textname(filenames_list_beforeprocess[i])
        doc_text = import_doc(path, filename, useful_characters)
        if len(doc_text) > minlength_of_text:
            textslist.append(doc_text)
        filenames_list.append(filename)
        textnames_list.append(textname)

    with nlp.disable_pipes("ner", "tagger"):
        nlp.max_length = doc_maxlength
        docs = list(nlp.pipe(textslist))

    counter = 0
    texts = []

    for doc in docs:
        for funct in parsers:
            text_of_doc = funct(doc)
        language, text, = get_language(multilanguage, doc,
                                       text_of_doc, default_language,
                                       supported_languages)
        if type(language) == list:
            '''textnew = [] ???'''
            for j in range(len(language)):
                filenames_list_final.append(filenames_list[counter])
                textnames_list_final.append(textnames_list[counter])
            languagelist += language
            texts += text
        elif type(language) == str:
            filenames_list_final = filenames_list
            textnames_list_final = textnames_list
            languagelist.append(language)
            for funct in parsers:
                text = funct(doc)
            texts.append(text)
        else:
            print("something went wrong")
        counter += 1

    doc_list = list(
        zip(filenames_list_final, textnames_list_final, languagelist, texts))

    doc_list.sort(key=lambda doc_list: doc_list[2])
    return doc_list


def get_all_text_info(doc_list, supported_languages, POS_blacklist, doc_maxlength):
    '''applies all functions that retrieve info from the documents texts
    doc_list is the dataframe with texts, textnames and so on
    supported_languages is the list of supported languages
    POS_blacklist is a list of POS-tags that are omitted in the function pos_tokenizer
    doc_maxlength is the maximum number of tokens within a document that spacy processes 
    '''

    length = []
    filteredtxts = []
    filteredADJss = []
    filteredNOUNss = []
    filteredVERBss = []
    uniquelst = []
    poslex = []
    neglex = []
    polarscores = []

    for lin in sorted(list(set([row[2] for row in doc_list]))):
        nlp = get_spacy_tokenizer(
            lin, supported_languages, bigmodel_required=False)
        postxt, negtxt = get_polarlex(lin)
        texte = []

        for i in range(len([row[2] for row in doc_list])):
            if [row[2] for row in doc_list][i] == lin:
                texte.append([row[3] for row in doc_list][i])

        docs = list(nlp.pipe(texte))
        filteredtexts = []
        filteredNOUNs = []
        filteredVERBs = []
        filteredADJs = []
        unique = []
        polarscoree = []
        poslexdoce = []
        neglexdoce = []
        length_of_docs_per_language = []
        for doc in docs:
            length_of_doc = len(doc)
            length_of_docs_per_language.append(str(length_of_doc))
            filteredtxt, filteredNOUN, filteredVERB, filteredADJ = pos_tokenizer(
                doc, nlp, POS_blacklist, doc_maxlength)
            filteredtexts.append(filteredtxt)
            filteredNOUNs.append(filteredNOUN)
            filteredVERBs.append(filteredVERB)
            filteredADJs.append(filteredADJ)
            uniques = []
            poslexdoc = ""
            neglexdoc = ""
            for item in doc.ents:
                if item.text not in uniques:
                    uniques.append(item.text)
            unique.append(uniques)
            for token in doc:
                if token.text in postxt:
                    # if token.text not in poslexdoc:
                    poslexdoc += " "+token.text
                if token.text in negtxt:
                    # if token.text not in neglexdoc:
                    # not used, otherwise polarscore is inaccurate
                    neglexdoc += " "+token.text

            polarscore = str(
                round(((len(poslexdoc)+len(neglexdoc))/len(doc)), 2))
            polarscoree.append(polarscore)
            poslexdoce.append(poslexdoc)
            neglexdoce.append(neglexdoc)
            # maybe I should only add the ents if they belong to the nlp.vocab, because a.t.m
            # I get wild ents for texts with wild strings.
            # downside would be that then I have to use the "md"-spacy-model for this function
        length += length_of_docs_per_language
        filteredtxts += filteredtexts
        filteredNOUNss += filteredNOUNs
        uniquelst += unique
        filteredVERBss += filteredVERBs
        filteredADJss += filteredADJs
        poslex += poslexdoce
        neglex += neglexdoce
        polarscores += polarscoree

    df_doclist = pd.DataFrame(doc_list, columns=[
        'File', 'Textname', 'Sprache', 'Text'])

    # now we put all the lists we created into a dataframe
    df_doclist['Textlänge'] = length
    df_doclist['bereinigter Text'] = filteredtxts
    df_doclist['Substantive'] = filteredNOUNss
    df_doclist['Verben'] = filteredVERBss
    df_doclist['Adjektive'] = filteredADJss
    df_doclist['Entitäten'] = uniquelst
    df_doclist['Positive Wörter'] = poslex
    df_doclist['Negative Wörter'] = neglex
    df_doclist['Polarität'] = polarscores

    # now we add a categorical variable, that is set to "emotional" if pocarscores for this
    # text is >.1
    polarcat = []
    for score in polarscores:
        if float(score) < 0.4:
            polarcat.append("neutral")
        else:
            polarcat.append("emotional")
    df_doclist['Polaritätskategorie'] = polarcat

    return (df_doclist)


def pdf_to_text(path, extensions):
    """transforms pdf and other files to txt format and safes it in the path folder
    adopted from PyMuPDF Tutorial"""
    for x in os.listdir(path):
        # Wenn es sich um ein pdf-File handelt, nehmen wir das paket PyMuPDF via fitz
        if x.endswith('.pdf'):
            #"""for i in range(0, len(filenames_list)): fname = filenames_list[i]"""
            doc = fitz.open(path+x)  # open document
            out = open(path+x[:-4] + ".txt", "wb")  # open text output
            for page in doc:  # iterate the document pages
                text = page.getText().encode("utf8")  # get plain text (is in UTF-8)
                out.write(text)  # write text of page
                # write page delimiter (form feed 0x0C)
                out.write(bytes((12,)))
            out.close()
        # Ansonsten probieren wir es mit textract, einem wrapper für verschiedene
        # Pakete, die verschiedene Dateien in Text-Dateien transformieren können.
        # Jedoch kann es hier auch unsauberen Output geben.
        elif x.endswith(tuple(extensions)) and not x.endswith('.pdf') and not x.endswith('.txt'):
            print(
                "Will try to parse file {file} with other extension.".format(file=x))
            try:
                text = textract.process(path+x)
                # xxxx could make -extensions
                #xclean = x.replace(extensions, "")
                out = open(path+x[:-4] + ".txt", "wb")
                out.write(text)
                out.close()
                print("Sucessfully parsed")
            except:
                print("Failed to parse document")


def get_textname(filename):
    textname = filename.split(".txt")[0]
    textname = textname.replace('_', ' ').replace('-', ' ')
    return textname


def import_doc(path, filename, useful_characters):
    """simply importing the txtfiles"""
    title_path = (path+filename)
    with open(title_path, "r", encoding="utf8", errors='ignore') as current_file:
        text = current_file.read()  # get it directly clean. optional if using nlpre!

    # WHY NOT USED???
    text = clean_words(text, useful_characters)

    return text


def clean_words(text, useful_characters):
    """makes some cleaning, especially excluding characters that should not
    appear in text of the languages used in this script. Must be silence
    for Chinese, Russian and other languages """
    text = re.sub("ß", "ss", text)
    # remove weird characters
    text = ''.join(filter(lambda x: x in useful_characters, text))
    return text


def is_any_lowercase(sentence):
    """
    Checks if any letter in a sentence is lowercase, return False if there
    are no alpha characters.
    Args:
        tokens: A list of string
    Returns:
        boolean: True if any letter in any token is lowercase
    """
    if any(x.isalpha() & (x == x.lower()) for x in sentence):
        return True
    return not any(x.isalpha() for x in sentence)


def titlecaps(doc):
    '''lowercases any sentence, where the entire sentence is written uppercase
    source: https://github.com/NIHOPA/NLPre/blob/master/nlpre/titlecaps.py'''
    # Need to keep the parser for sentence detection
    sents = doc.sents
    doc2 = []
    for sent in sents:
        line = sent.text
        if not is_any_lowercase(line):
            if len(line) > 4:
                line = line.lower()
        doc2.append(line + sent[-1].whitespace_)
    doc2 = "".join(doc2)
    return doc2


def token_replacement(doc):
    """replaces tokens some common symbols and characters with corresponding letters
    source: https://github.com/NIHOPA/NLPre/blob/master/nlpre/token_replacement.py"""
    replace_dict = {
        # these should not be used since they only work for english...
        # "&": " and ",
        # "%": " percent ",
        # ">": " greater-than ",
        # "<": " less-than ",
        # "=": " equals ",
        "#": " ",
        "~": " ",
        "/": " ",
        "\\": " ",
        "|": " ",
        "$": "",
        "€": " Euro",
        # Remove empty :
        " : ": " ",
        # Remove double dashes
        "--": " ",
        # Remove possesive splits
        " 's ": " ",
        # Remove quotes
        "'": "",
        '"': "",
    }
    # for key in replace_dict:
    #    replace_dict[key] = " " #was macht das?????
    text = doc.text
    for key, val in replace_dict.items():
        text = text.replace(key, val)
        # Remove blank tokens, but keep line breaks
        text = [
            " ".join([token for token in line.split()])
            for line in text.split("\n")
        ]
        # Remove blank lines
        text = "\n".join(filter(None, text))
    return text


def url_replacement(doc):
    """replaces url text with blank
    source: https://github.com/NIHOPA/NLPre/blob/master/nlpre/url_replacement.py"""
    text = []
    for token in doc:
        if token.like_url:
            text.append("")
            text.append(token.whitespace_)
        elif token.like_email:
            text.append("")
            text.append(token.whitespace_)
        else:
            text.append(token.text_with_ws)
    return "".join(text)


def keep_root(token, word_order=0):
    # If it's the first word, keep any other letters are capped
    # Otherwise, keep if any letters are capped.
    shape = token.shape_
    if word_order == 0:
        shape = shape[1:]
    return "X" in shape


def pos_tokenizer(doc, nlp, POS_blacklist, doc_maxlength, use_base=True):
    ''' Diese Funktion setzen wir erst ein, wenn wir bereits die Sprache kennen.
    Daher benutzen wir hier nicht das default englische nlp, sondern die Sprachenspezifische.
    Gerade für diese Funktion ist dies wichtig, da wir hier ja auf die token.pos_
    Variable gehen, die von der Sprache abhängt...
    source: https://github.com/NIHOPA/NLPre/blob/master/nlpre/pos_tokenizer.py'''
    #text = " ".join(text.strip().split())
    special_words = set(["_"])
    doc = doc
    txt = []
    nouns = []
    verbs = []
    adjs = []
    with nlp.disable_pipes("ner"):
        nlp.max_length = doc_maxlength
        for sent in doc.sents:
            sent_tokens = []
            sent_nouns = []
            sent_verbs = []
            sent_adjs = []
            for k, token in enumerate(sent):
                # If we have a special word, add it without modification
                if any(sw in token.text for sw in special_words):
                    sent_tokens.append(token.text)
                    continue
                if token.pos_ in POS_blacklist:
                    continue
                # this condition can be deactivated if preferred:
                if token.is_stop == True or token.is_alpha == False:
                    continue
                word = token.text
                if (
                    use_base
                    and not keep_root(token, k)
                    and token.pos_ != "PRON"
                ):
                    word = token.lemma_
                # If the word is a pronoun, we need to use the base form, see
                # https://github.com/explosion/spaCy/issues/962
                if token.lemma_ == "-PRON-":
                    word = token.text.lower()
                sent_tokens.append(word)
                if token.pos_ in ["NOUN"]:
                    noun = token.text
                    sent_nouns.append(noun)
                if token.pos_ in ["VERB"]:
                    verb = token.text.lower()
                    sent_verbs.append(verb)
                if token.pos_ in ["ADJ", "ADV"]:
                    adj = token.text.lower()
                    sent_adjs.append(adj)
            # xxxx replace lists ev. with a dictionary
            txt.append(" ".join(sent_tokens))
            for nn in sent_nouns:
                if nn not in nouns:
                    nouns.append("".join(nn))
            for vrb in sent_verbs:
                if vrb not in verbs:
                    verbs.append("".join(vrb))
            for dj in sent_adjs:
                if dj not in adjs:
                    adjs.append("".join(dj))
            #verbs.append(" ".join(sent_verbs))
            #adjs.append(" ".join(sent_adjs))
        txt = " ".join(txt)
        nouns = " ".join(nouns)
        verbs = " ".join(verbs)
        adjs = " ".join(adjs)
        return txt, nouns, verbs, adjs


def get_polarlex(input_language):
    language = pycountry.languages.get(name=input_language)
    langcode = language.alpha_2
    while True:
        if os.path.exists("lexika/"):
            lexikapath = "lexika/"
            break
        else:
            print("Input directory of lexica:")
            npt = input()
            if os.path.exists(npt):
                if npt.endswith("/"):
                    lexikapath = npt
                else:
                    lexikapath = npt+"/"
                break
            else:
                continue
    poslexname = "positive_words_"+langcode+".txt"
    neglexname = "negative_words_"+langcode+".txt"
    with open(lexikapath+poslexname, "r", encoding="utf8", errors='ignore') as current_file:
        postxt = current_file.read()
        postxt = postxt.split("\n")
    with open(lexikapath+neglexname, "r", encoding="utf8", errors='ignore') as current_file:
        negtxt = current_file.read()
        negtxt = negtxt.split("\n")
    return postxt, negtxt


def get_spacy_tokenizer(default_lingo, supported_languages, bigmodel_required):
    '''returns the spacy nlp function corresponding to the language of a document'''
    if default_lingo in supported_languages:
        if bigmodel_required == False:
            if default_lingo == "German":
                import de_core_news_sm
                nlp = de_core_news_sm.load()
            elif default_lingo == "English":
                import en_core_web_sm
                nlp = en_core_web_sm.load()
            elif default_lingo == "Spanish":
                import es_core_news_sm
                nlp = es_core_news_sm.load()
            elif default_lingo == "French":
                import fr_core_news_sm
                nlp = fr_core_news_sm.load()
            elif default_lingo == "Portuguese":
                import pt_core_news_sm
                nlp = pt_core_news_sm.load()
            else:
                import it_core_news_sm
                nlp = it_core_news_sm.load()
        else:
            if default_lingo == "German":
                import de_core_news_md
                nlp = de_core_news_md.load()
            elif default_lingo == "English":
                import en_core_web_md
                nlp = en_core_web_md.load()
            elif default_lingo == "Spanish":
                import es_core_news_md
                nlp = es_core_news_md.load()
            elif default_lingo == "French":
                import fr_core_news_md
                nlp = fr_core_news_md.load()
            elif default_lingo == "Portuguese":
                # there is no pt_md model
                import pt_core_news_sm
                nlp = pt_core_news_sm.load()
            else:
                # there is no it_md model
                import it_core_news_sm
                nlp = it_core_news_sm.load()
    else:
        print("NOT A SUPPORTED LANGUAGE!")
    return nlp


def decide_language_detection(path, supported_languages, default_lingo):
    from pycountry import languages
    print("Does your directory with the name \n {path} \n contain single documents written in more than one language?\n Enter y for yes or n for no".format(
        path=path))
    while True:
        try:
            multilanguage = input()
        except ValueError:
            print("Sorry, I didn't understand that.")
            continue
        if multilanguage == "exit":
            break
        elif multilanguage not in ["y", "n"]:
            print("You gave a wrong input. Try again")
        else:
            print("Thank you for your input. Language Detection starting now")
            break
    nlp = get_spacy_tokenizer(default_lingo, supported_languages, bigmodel_required=False)
    return multilanguage, nlp


def get_language(multilanguage, doc, text, default_lingo, supported_lingos):
    '''returns a list or string of language(s) and text(s) per input text'''

    '''if len(doc.text) >= 50 and doc.text.isdigit() == False:'''
    if multilanguage == "n":
        textlist = text
        langcode = doc._.language['language']
        if langcode != "UNKNOWN":
            langlist = languages.get(alpha_2=langcode).name
            if langlist not in supported_lingos:
                langlist = default_lingo
                print(
                    "Language detection probably not successfull, using default language.")
        else:
            langlist = default_lingo
            print(
                "Language detection not successfull, using default language.")
    else:
        langlist = []  # the list of all unique detected languages
        textlist = []  # the list which contains the string of text for each unique languages
        removerow = []  # just an index of rows
        count_sents = 0
        '''doc = nlp(text)'''
        for sent in doc.sents:
            count_sents += 1
            langcode = sent._.language['language']
            if langcode != "UNKNOWN":
                langname = languages.get(alpha_2=langcode).name
                if langname in supported_lingos:
                    if langname not in set(langlist):
                        langlist.append(langname)
                        textlist.append(sent.text)
                    else:
                        for k in range(0, len(langlist)):
                            if langname == langlist[k]:
                                textlist[k] += " "+sent.text
        for i in range(len(langlist)):
            percentageoflingo = int(
                100*len(textlist[i])/len(''.join(textlist)))
            if percentageoflingo < 2 and not len(textlist[i]) > 500:
                removerow.append(i)
                # or len(textlist[i]) < 50:
        # wir löschen also jenen Teil des Textes, der eine der obrigen Bedingungen erfüllt
        # reverse the order so that not the first elements
        # are deleted before the latter ones, which would cause errors
        for j in sorted(removerow, reverse=True):
            del langlist[j]
            del textlist[j]
    return langlist, textlist


def get_path(parsable_extensions):
    '''returns the path depending on the working directory and user input'''
    filenames_list = []
    workingdir = os.getcwd()
    workingdir = ''.join(workingdir)
    if not workingdir.endswith("/"):
        workingdir = workingdir + "/"
    elif not workingdir.startswith("/"):
        workingdir = "/" + workingdir
    print("Current working directory is {}. Do you want to analyse documents from your working directory? Press Enter if so. Otherwise indicate the desired subfolder from working directory. Input exit to leave the module".format(os.getcwd()))
    while True:
        datenablage = input()
        """datenablage = "daten/neu"""
        # 1. zuerst den richtigen Pfad bestimmen
        if datenablage == "exit":
            break
        if datenablage == "":
            path = workingdir
        else:
            if not datenablage.endswith("/"):
                datenablage = datenablage + "/"
            path = datenablage
        try:
            filenames_list = [x for x in os.listdir(
                path) if x.endswith(".txt") or x.endswith(tuple(parsable_extensions))]
        except:  # falls Fehler kommt weil Nutzer sub-path eingegeben hat
            path = workingdir+datenablage
            try:
                filenames_list = [x for x in os.listdir(
                    path) if x.endswith(".txt") or x.endswith(tuple(parsable_extensions))]
            # 2. Einen Fehler werfen wenn kein (sub)directory vorliegt. Neu starten
            except:
                print("Error occured! Probably not a directory. Try again")
                continue
        # 3. den Prozess erneut starten, falls zu viele Dateien im directory sind
        if len(filenames_list) > 30:
                # including a performance upper limit
            print("Too many files: use directory with less documents!")
            continue
        elif filenames_list == []:
            print("No documents detected. Try again")
            continue
        else:
            print("Using path ", path)
            break
    return path


if __name__ == '__main__':
    main()
