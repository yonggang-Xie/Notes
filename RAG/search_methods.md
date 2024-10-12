## BM 25
BM25 (Best Match 25) is a keyword search algorithm because it relies on the presence and frequency of query terms (keywords) in documents to determine their relevance. Here's a detailed explanation of how BM25 works as a keyword search:

1. Determining keywords in an input query:
   - When a user enters a query, BM25 treats each individual word in the query as a keyword.
   - The query is tokenized, meaning it is split into individual words or terms.
   - Stop words (common words like "the," "and," "is") are usually removed from the query, as they are not considered informative keywords.
   - The remaining terms in the query are considered the keywords that will be used to search for relevant documents.

2. How keywords determine which documents to retrieve:
   - BM25 calculates a relevance score for each document in the collection based on the presence and frequency of the query keywords.
   - The relevance score is determined by considering two main factors:
     a. Term Frequency (TF): It measures how frequently a keyword appears in a document. The more times a keyword occurs in a document, the higher its TF score.
     b. Inverse Document Frequency (IDF): It measures the importance of a keyword across the entire document collection. Keywords that appear in fewer documents are considered more informative and receive a higher IDF score.
   - BM25 combines the TF and IDF scores using a specific formula to calculate the relevance score for each document with respect to the query keywords.
   - The formula also includes two parameters: k1 and b, which control the impact of term frequency and document length normalization, respectively.
   - Documents with higher relevance scores are considered more relevant to the query and are ranked higher in the search results.
   - BM25 retrieves the top-ranked documents based on their relevance scores, presenting them to the user as the search results.

To summarize, BM25 is a keyword search algorithm because:
1. It determines the keywords by tokenizing the input query and treating each term as a keyword.
2. It uses the presence and frequency of these keywords in documents to calculate relevance scores.
   - The relevance scores are based on the Term Frequency (TF) and Inverse Document Frequency (IDF) of the keywords.
   - Documents with higher relevance scores, indicating a stronger match with the query keywords, are considered more relevant.
3. BM25 retrieves and ranks the documents based on their relevance scores, presenting the top-ranked documents as the search results.

The effectiveness of BM25 lies in its ability to consider both the frequency of keywords within documents (TF) and the importance of those keywords across the document collection (IDF). By combining these factors, BM25 can retrieve documents that are most relevant to the user's query based on the presence and significance of the keywords.

## Vectore embedding
   Use a embedding model to convert query and documents to embeddings and perform cosine similarity search (with threshold)
