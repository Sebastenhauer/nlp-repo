# nlp-repo


allows you to import, clean, analyse, sort and illustrate texts of several data types from a directory of choice. based on spacy.
- 

Still work in progress, I am extremely grateful for feedback!



input:
- 

directory which contains documents of various file formats. supports languages German, English as well as the latin languages Italian, Frensh, Portuguese, Spanish

output:
- 

pandas dataframe with basic analysis and linguistic features for each document (or each language of each document)

html graphical summary of the dataframe



The script does basically the following:
- 

1: let the user decide from which directory documents should be imported

2: transform pdfs, docx and some other formated documents into txt

3: let the user specify if single documents in the directory are written in more than one language

4: depending on the answer to 3, the language detection is applied on the entire document or on each sentence of the doc individually (= considerably more costly and not always accurate, especially if the document is not well parsed)

5: apply some cleaning functions, adapted from the package nlpre

6: define textname, language, cleaned text (= subselect certain linguistic elements), nouns, verbs, adjectives and named entities

7: store this data along with the original texts in a pandas dataframe in the directory from which the documents are parsed

8: output a graphical summary of the documents with the package scattertext (does not work currently if only one language is predicted!)



further development path:
- 
1: find a way to eliminate the first spacy pipeline with the language detection

2: make the code more efficient

3: train a text classifier with spacy-transformers and textcat and add to the second nlp-pipeline

4: use the predicted category of the document as the category variable in the scattertext-corpus

5: allow parsing from web ressources (see spacy-holmes code and beautifulsoup)

6: add further useful functions to the pipeline such as a specific spacy-matcher, train an ner ... 


lexika are from:
- 
https://sites.google.com/site/datascienceslab/projects/multilingualsentiment 
