library(textmineR)

# Select a document
fileName <- '1984-George_Orwell_chap1.txt'
theName <- unlist(strsplit(fileName,'[.]'))[1]

myText <- readChar(fileName, file.info(fileName)$size)

# Preprocessing
cleanText <- function(thisText, wthres = 3,rmtilt = TRUE){
  thisText <- gsub("[\n*+]", ".", thisText)               # rm text scaped symbols
  thisText <- gsub("[\r\t*+]", "", thisText)              # rm text scaped symbols
  thisText <- gsub('([[:punct:]])\\1+', '\\1 ', thisText) # rm remove duplicated punctuation symbols
  thisText <- gsub('([[:space:]])\\1+', '\\1', thisText)  # rm remove duplicated spaces
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

# Turn those sentences into a DTM, use stemming
dtm <- CreateDtm(doc, 
                 ngram_window = c(1, 2),
                 stopword_vec = c(tm::stopwords("english"), 
                                  tm::stopwords("SMART")),  # this is the default value
                 stem_lemma_function = function(x) SnowballC::wordStem(x, "porter"))

# TF-IDF Frequency re-weighting
idf <- log(nrow(dtm) / colSums(dtm > 0))

tfidf <- t(dtm) * idf

tfidf <- t(tfidf)

# Calculate document-to-document cosine similarity
csim <- tfidf / sqrt(rowSums(tfidf * tfidf))

csim <- csim %*% t(csim)


p_words <- colSums(dtm) / sum(dtm)  #
p_sent <- rowSums(dtm)

# Turn that cosine similarity matrix into a nearest-neighbor network
nn <- csim

diag(nn) <- 0

nn <- apply(nn, 1, function(x){
  x[ x < sort(x, decreasing = TRUE)[ 2 ] ] <- 0
  x
})

nn <- nn * 100

g <- igraph::graph_from_adjacency_matrix(nn, mode = "directed", weighted = TRUE)

# Calculate eigenvalue centrality
ec <- igraph::eigen_centrality(g)

# Return top  central sentences as the summary
nsentences <- 15
summary <- doc[ names(ec[[ 1 ]])[ order(ec[[ 1 ]], decreasing = T) ][ 1:nsentences ] ]

summary <- summary[ order(as.numeric(names(summary))) ]

# finaldoc <- paste(doc, collapse = ". ")
finalsum <- paste(c(summary,''), collapse = ". ")
finalsum

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

if (!file.exists('outputs')){
  dir.create('outputs')}

savFile = paste0('outputs/','summary',nsentences,'sentences_',theName,'_ec.txt')
fileConn<-file(savFile)
# writeLines(c(finaldoc, '###########', finalsum ) , fileConn)
writeLines(c('###########', finalsum, '###########' ) , fileConn)
close(fileConn)
