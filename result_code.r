library(textmineR)

# Select a document
fileName <- '1984-George_Orwell_chap1.txt'
theName <- unlist(strsplit(fileName,'[.]'))[1]

myText <- readChar(fileName, file.info(fileName)$size)

# Preprocessing
cleanText <- function(thisText, wthres = 3,rmtilt = TRUE){
  thisText <- gsub("[\n*+]", ".", thisText)               # rm text scaped symbols
  thisText <- gsub("[\r\t*+]", "", thisText)              # rm text scaped symbols
  thisText <- gsub('([[:punct:]])\\1+', '\\1 ', thisText) # rm duplicated punctuation symbols
  thisText <- gsub('([[:space:]])\\1+', '\\1', thisText)  # rm duplicated spaces
  thisText <- unlist(strsplit(thisText, "[.]"))
  thisText <- gsub(" $", "", thisText)                    # rm final spaces
  thisText <- gsub("(^[[:space:]])", "", thisText)        # rm initial spaces
  thisText <- thisText[!grepl("^\\s*$", thisText)]          # take non empty sentences
  thisText <- unique(thisText)                            # rm duplicated sentences 
  chosenElement = sapply(strsplit(thisText, "\\s+"), length) > wthres
  thisText <- thisText[chosenElement]
  
  return(thisText)
}

doc <- cleanText(myText, 4)
names(doc) <- seq(along = doc)
  
# Turn those sentences into a DTM, use stemming & bi-grams
dtm <- CreateDtm(doc, 
                 ngram_window = c(1, 2),
                 stopword_vec = c(tm::stopwords("english"), # stopwords from tm
                                  tm::stopwords("SMART")), # this is the default value
                 stem_lemma_function = function(x) SnowballC::wordStem(x, "porter"))

# TF-IDF Frequency re-weighting
idf <- log(nrow(dtm) / colSums(dtm > 0))

tfidf <- t(dtm) * idf

tfidf <- t(tfidf)
tfidf <- tfidf[-which(apply(tfidf,1,sum) <= 0),]

# Calculate document-to-document cosine similarity
csim <- tfidf / sqrt(rowSums(tfidf * tfidf))

csim <- csim %*% t(csim)

cdist <- as.dist(1 - csim)

# word/sentence weights
p_words <- colSums(dtm) / sum(dtm)  #
p_sent <- rowSums(dtm)

# clustering

hc <- hclust(cdist, "ward.D")

nsentences <- 15
clustering <- cutree(hc, nsentences)

# make list of sentences - clustered
summary_sentences <- unlist(lapply(unique(clustering), function(x, document,weights){    # lista con los terminos segun el cluster
  selected <- document[ clustering == x ]
  names(selected) <- names(document[ clustering == x ])
  chosename = names(sort(weights[names(selected)], decreasing = TRUE)[1])
  
  selected[chosename]
}, doc, p_sent))

sortindex = as.character(sort(as.integer(names(summary_sentences))))
summary_sentences <- summary_sentences[sortindex]

finalsum = paste(summary_sentences,collapse = '. ')
finalsum <- paste0(finalsum,'.') 

## save file
if (!file.exists('outputs')){
  dir.create('outputs')}

savFile = paste0('outputs/','summary',nsentences,'sentences_',theName,'_hc.txt')
fileConn<-file(savFile)
writeLines(c(paste0(rep('#', 100), collapse = ''),finalsum, paste0(rep('#', 100), collapse = '')) , fileConn)
close(fileConn)

# Plot clustering tree

plot(hc, main = "Hierarchical clustering",
     ylab = "", xlab = "", yaxt = "n")
rect.hclust(hc, nsentences, border = "red")

### Post -- clustering word extraction -- tag cloud

# Summary table

cluster_words <- lapply(unique(clustering), function(x){
  rows <- dtm[ clustering == x , ]
  
  # for memory's sake, drop all words that don't appear in the cluster
  rows <- rows[ , colSums(rows) > 0 ]
  
  colSums(rows) / sum(rows) - p_words[ colnames(rows) ]
})

# create a summary table of the top words defining each cluster
cluster_summary <- data.frame(cluster = unique(clustering),
                              size = as.numeric(table(clustering)),
                              top_words = sapply(cluster_words, function(d){
                                chosennames = names(d)[ order(d, decreasing = TRUE) ][ 1:5 ]
                                chosennames <- chosennames[!is.na(chosennames)]
                                paste(chosennames,
                                      collapse = ", ")
                              }),
                              stringsAsFactors = FALSE)

cluster_summary <- cluster_summary[order(cluster_summary$size, decreasing = TRUE),]



# Tag cloud

# plot a word cloud of one cluster as an example
whichclust = cluster_summary[1,1]
wccluster <- wordcloud::wordcloud(words = names(cluster_words[[ whichclust ]]),
                                  freq = cluster_words[[ whichclust ]],
                                  max.words = 50,
                                  random.order = FALSE,
                                  colors = c("mediumseagreen",  "dodgerblue3", "firebrick3"),
                                  main = paste("Top words in",theName,sep=' ' ))

# plot a word cloud of most important words under the tfidf metric 
wcall <-wordcloud::wordcloud(words = names(colSums(tfidf)),
                             freq = colSums(tfidf),
                             max.words = 70,
                             random.order = TRUE,
                             colors = c("mediumseagreen",  "dodgerblue3", "firebrick3"),
                             main = paste("Top words in",theName,sep=' ' ))


