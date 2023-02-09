# sentiment-Detection
This end to end solution for text Sentiment detection contains the following sections
1. Querriying the required datasets from the db
2. Preprocessing of the text data
3. Scoring the words using tfidf vetorizer
4. Model training 
5. Saving the model as .sav file for future use
6. Getting the emotions(joy/Sad/Happy/trust/fear/anger) from the document -implemented using the rule based approach
7. grouping the documents based on predefined Keywords
8. Creating a new colum for storign the lemmatized text inorder build a wordcloud
9. Writing back the results back to database 
which can be used for detecting the sentiment (positive/negative/neutral) of a text document.
The entire solution consist of pulling the required datasets from an SQL server writing the results back to a database from where results can be used to build power Bi dash board to visualise the final results.
