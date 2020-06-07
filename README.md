# nlp-repo


The code in this repo allows you to import, clean, analyse, sort and illustrate texts of several data types from a directory of choice.
- 

As this is my first nlp-project and I am a beginner in coding, I am extremely grateful for feedback!
In fact, professional developers may laugth at some parts of the code :) Well, it was just more fun to code something useful than to read through long books on programming...


input:
- 

directory which contains documents of various file formats. supports languages German, English as well as the latin languages

output:
- 

pandas dataframe with basic analysis and linguistic features for each document (or each language of each document)

html graphical summary of the dataframe



The script does basically the following:
- 

1: transform pdfs, docx and some other formated documents into txt

2: let the user decide from which directory documents should be imported

3: let the user specify if single documents in the directory are written in more than one language

4: depending on the answer to 3, the language detection is applied on the entire document or on each sentence of the doc individually (= considerably more costly and not always accurate, especially if the document is not well parsed)

5: apply some cleaning functions, adapted from the package nlpre

6: define textname, language, cleaned text (= subselect certain linguistic elements), nouns, verbs, adjectives and named entities

7: story all these data along with the original text in a pandas dataframe in the directory from which the documents are parsed

8: output a graphical summary of the documents in a html with the package scattertext (does not work currently if only one language is predicted!)



further development path:
- 

1: train a text classifier with spacy-transformers and textcat and add to the second nlp-pipeline

2: use the predicted category of the document as the category variable in the scattertext-corpus

3: allow parsing from web ressources (see spacy-holmes code and beautifulsoup)

4: find a way to eliminate the first spacy pipeline with the language detection

5: make the code more efficient
