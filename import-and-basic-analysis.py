
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
"""

import textract
import fitz
import string
import re
import sys
import os
from os import system, listdir
import numpy as np
import pandas as pd
import collections
import pycountry
import scattertext as st
import spacy
import spacy_langdetect
from pycountry import languages
from spacy import matcher
from spacy import tokens
from spacy.matcher import Matcher
from spacy.tokens import Token

from spacy_langdetect import LanguageDetector


# check if python and pip as well as the executable are in line
os.system('which python')
os.system('which pip')
sys.executable

######## Define Functions ##############


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
    textname = textname.replace('_', ' ')
    return textname


def import_doc(path, filename):
    """simply importing the txtfiles"""
    title_path = (path+filename)
    with open(title_path, "r", encoding="utf8", errors='ignore') as current_file:
        text = current_file.read()  # get it directly clean. optional if using nlpre!
        # text = clean_words(text)
    return text


def clean_words(text, useful_characters):
    """makes some cleaning, especially excluding characters that should not
    appear in text of the languages used in this script. Must be silence
    for Chinese, Russian and other languages """
    text = re.sub("ß", "ss", text)
    # remove weird characters
    text = ''.join(filter(lambda x: x in useful_characters, text))
    # remove double slashes and thus http//
    #text = re.sub("//|[\\\]", " ", text)
    #text = re.sub(" n", " ", text)
    # text = re.sub("|\d","",text)#remove digits
    # text = re.sub("[\\\]", "", text)#remove backslashes
    # text = re.sub("\s+", " ", text)  # remove empty parts like "aa     bb."
    return text
    # see https://www.w3schools.com/python/python_regex.asp


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


def pos_tokenizer(doc, nlp, POS_blacklist, maxlength, use_base=True):
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
        nlp.max_length = maxlength
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
                word = token.text
                if (
                    use_base
                    # vllt muss "k" noch removed werden
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


def get_spacy_tokenizer(default_lingo, supported_languages, higher):
    '''returns the nlp function corresponding to the language of a doc/corpus'''
    if default_lingo in supported_languages:
        if higher == False:
            if default_lingo == "German":
                import de_core_news_sm
                nlp = de_core_news_sm.load()
            elif default_lingo == "English":
                import en_core_web_sm
                nlp = en_core_web_sm.load()
            elif default_lingo == "Spanish":
                import es_core_news_sm
                nlp = es_core_news_sm.load()
            elif default_lingo == "Frensh":
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
            elif default_lingo == "Frensh":
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
        if multilanguage not in ["y", "n"]:
            print("You gave a wrong input. Try again")
        else:
            print("Thank you for your input. Language Detection starting now")
            break
    nlp = get_spacy_tokenizer(default_lingo, supported_languages, higher=False)
    return multilanguage, nlp


def get_language(multilanguage, doc, text, default_lingo, supported_lingos):
    if len(doc.text) >= 6 and doc.text.isdigit() == False:

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
            doc = nlp(text)
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
            print("sum of all sentences in this doc is", count_sents)
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


######## start the pipeline :) ##############
if __name__ == '__main__':
    # this is required, otherwise we get weird languages for long and untidy documents
    supported_languages = ["English", "German",
                           "Spanish", "Portuguese", "French", "Italian"]
    default_language = "English"  # making English the default...
    useful_characters = string.printable + \
        'äöüÄÖÜéÉèÈáÁàÀóÓòÒúÚùÙíÍìÌñÑãÃõÕêÊâÂîÎôÔûÛ'  # filtering the characters of the texts
    parsable_extensions = ['.csv', '.doc', '.docx', '.eml', '.epub', '.json',
                           '.msg', '.odt', '.ogg', '.pdf', '.pptx', '.rtf', '.xlsx', '.xls']
    """ '.gif', '.jpg', '.mp3', '.tiff', '.wav', '.ps', '.html' """
    # the extensions which we try to parse to text
    maxlength = 2000000  # default would be 1m
    minlength = 100
    POS_blacklist = ["PUNCT", "PART", "SYM", "SPACE",
                     "DET", "CONJ", "CCONJ", "ADP", "INTJ", "X", ""]
    # the parsing functions in use
    parsers = [titlecaps, token_replacement, url_replacement]
    ######### Determining the directory from which to import documents #########
    workingdir = os.getcwd()
    workingdir = ''.join(workingdir)
    if not workingdir.endswith("/"):
        workingdir = workingdir + "/"
    elif not workingdir.startswith("/"):
        workingdir = "/" + workingdir
    print("Current working directory is {}. Do you want to analyse documents from your working directory? Press Enter if so. Otherwise indicate the desired subfolder from working directory".format(os.getcwd()))
    while True:
        datenablage = input()
        """datenablage = "daten/neu"""
        # 1. zuerst den richtigen Pfad bestimmen
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
    # now we let the user determine if he wants to use the sentence-wise
    # language detection or the document-wise. The sentence-wise allows
    # to ignore parts of docs that contain text not of interest, such
    # as metadata in english for a german document or so
    multilanguage, nlp = decide_language_detection(
        path, supported_languages, default_language)
    nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)

    pdf_to_text(path, parsable_extensions)
    filenames_lst = [x for x in os.listdir(
        path) if x.endswith(".txt")]

    filenames = []
    textnames = []
    filenames_list = []
    textnames_list = []
    languagelist = []
    textspre = []

    for i in range(len(filenames_lst)):
        filename = str(filenames_lst[i])
        textname = get_textname(filenames_lst[i])
        text = import_doc(path, filename)
        if len(text) > minlength:
            textspre.append(text)
        filenames_list.append(filename)
        textnames_list.append(textname)

    with nlp.disable_pipes("ner", "tagger"):
        nlp.max_length = maxlength
        docs = list(nlp.pipe(textspre))

    counter = 0
    texts = []

    for doc in docs:
        for funct in parsers:
            txt = funct(doc)
        language, text, = get_language(multilanguage, doc,
                                       txt, default_language,
                                       supported_languages)
        if type(language) == list:
            textnew = []
            for j in range(len(language)):
                filenames.append(filenames_list[counter])
                textnames.append(textnames_list[counter])
            languagelist += language
            texts += text
        elif type(language) == str:
            filenames = filenames_list
            textnames = textnames_list
            languagelist.append(language)
            for funct in parsers:
                text = funct(doc)
            texts.append(text)
        else:
            print("something went wrong")
        counter += 1

    doc_list = list(
        zip(filenames, textnames, languagelist, texts))

    doc_list.sort(key=lambda doc_list: doc_list[2])

    filteredtxts = []
    filteredADJss = []
    filteredNOUNss = []
    filteredVERBss = []
    uniquelst = []

    for lin in sorted(list(set([row[2] for row in doc_list]))):
        nlp = get_spacy_tokenizer(
            lin, supported_languages, higher=False)
        texte = []

        for i in range(len(languagelist)):
            if languagelist[i] == lin:
                texte.append(texts[i])

        docs = list(nlp.pipe(texte))
        filteredtexts = []
        filteredNOUNs = []
        filteredVERBs = []
        filteredADJs = []
        unique = []

        for doc in docs:
            filteredtxt, filteredNOUN, filteredVERB, filteredADJ = pos_tokenizer(
                doc, nlp, POS_blacklist, maxlength)
            filteredtexts.append(filteredtxt)
            filteredNOUNs.append(filteredNOUN)
            filteredVERBs.append(filteredVERB)
            filteredADJs.append(filteredADJ)
            uniques = []
            for item in doc.ents:
                if item.text not in uniques:
                    uniques.append(item.text)
            unique.append(uniques)
            # maybe I should only add the ents if they belong to the nlp.vocab, because a.t.m
            # I get wild ents for texts with wild strings.
            # downside would be that then I have to use the "md"-spacy-model for this function

        filteredtxts += filteredtexts
        filteredNOUNss += filteredNOUNs
        uniquelst += unique
        filteredVERBss += filteredVERBs
        filteredADJss += filteredADJs

    df_doclist = pd.DataFrame(doc_list, columns=[
        'File', 'Textname', 'Sprache', 'Text'])
    df_doclist['bereinigter Text'] = filteredtxts
    df_doclist['Substantive'] = filteredNOUNss
    df_doclist['Verben'] = filteredVERBss
    df_doclist['Adjektive'] = filteredADJss
    df_doclist['Entitäten'] = uniquelst

    print(df_doclist.shape)
    df_doclist.to_pickle(path+"/df_doclist.pkl")

    import de_core_news_sm
    nlp = de_core_news_sm.load()
    corpus = st.CorpusFromPandas(
        df_doclist, category_col='Sprache', text_col='bereinigter Text', nlp=nlp).build()
    # actually, the category should be sth else than the language, because if we take the language
    # we don't get a big overlap between the categories! so ideally we would train a textcat model
    # BEFORE and use the category-column inside pd_doclist as the category

    import pprint
    from pprint import pprint
    term_freq_df = corpus.get_term_freq_df()
    term_freq_df['German words'] = corpus.get_scaled_f_scores('German')
    pprint(list(term_freq_df.sort_values(
        by='German words', ascending=False).index[:20]))

    html = st.produce_scattertext_explorer(
        corpus,
        category='German',  # not_category_name='English',
        minimum_term_frequency=5, metadata=corpus.get_df()['Textname'])

    fn = path+path.split("/")[-2]+'-Auswertung.html'
    open(fn, 'wb').write(html.encode('utf-8'))
    print('Open ' + fn + ' in Chrome or Firefox.')
