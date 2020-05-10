library(httr)
library(data.table)
library(Rtsne)
library(ggplot2)
library(ClusterR)
library(caret)
library(Metrics)
library(tidyverse)

#Read in the data
raw_train <- fread('./project/volume/data/raw/training_data.csv')
train_emb <- fread('./project/volume/data/raw/train_emb.csv')

raw_test <- fread('./project/volume/data/raw/test_file.csv')
test_emb <- fread('./project/volume/data/raw/test_emb.csv')


#take the 10 different 'y' (subreddit) columns and condense them into 1
#assign them each a number to do this. (assign 0-9)

#subredditcars = 0
#subredditCooking = 1 
#subredditMachineLearning = 2
#subredditmagicTCG = 3
#subredditpolitics = 4
#subredditReal_Estate = 5
#subredditscience = 6
#subredditStockMarket = 7
#subreddittravel = 8
#subredditvideogames = 9

#Rename the columns
colnames(raw_train) <- c('id','text',0,1,2,3,4,5,6,7,8,9)

#remove the text from the raw training data
raw_train$text <- NULL
raw_train$id <- seq(1,200)


#This is all done in order to put the classes into one column
#Melt did not work for me and was hurting my score
class_0 <- raw_train %>% select(id, '0') %>% filter(raw_train$`0` == 1)
colnames(class_0) <- c('id','class')
class_0$class <- 0

class_1 <- raw_train %>% select(id, '1') %>% filter(raw_train$`1` == 1)
colnames(class_1) <- c('id','class')
class_1$class <- 1

class_2 <- raw_train %>% select(id, '2') %>% filter(raw_train$`2` == 1)
colnames(class_2) <- c('id','class')
class_2$class <- 2

class_3 <- raw_train %>% select(id, '3') %>% filter(raw_train$`3` == 1)
colnames(class_3) <- c('id','class')
class_3$class <- 3

class_4 <- raw_train %>% select(id, '4') %>% filter(raw_train$`4` == 1)
colnames(class_4) <- c('id','class')
class_4$class <- 4

class_5 <- raw_train %>% select(id, '5') %>% filter(raw_train$`5` == 1)
colnames(class_5) <- c('id','class')
class_5$class <- 5

class_6 <- raw_train %>% select(id, '6') %>% filter(raw_train$`6` == 1)
colnames(class_6) <- c('id','class')
class_6$class <- 6

class_7 <- raw_train %>% select(id, '7') %>% filter(raw_train$`7` == 1)
colnames(class_7) <- c('id','class')
class_7$class <- 7

class_8 <- raw_train %>% select(id, '8') %>% filter(raw_train$`8` == 1)
colnames(class_8) <- c('id','class')
class_8$class <- 8

class_9 <- raw_train %>% select(id, '9') %>% filter(raw_train$`9` == 1)
colnames(class_9) <- c('id','class')
class_9$class <- 9

train <- rbind(class_0, class_1)
train <- rbind(train, class_2)
train <- rbind(train, class_3)
train <- rbind(train, class_4)
train <- rbind(train, class_5)
train <- rbind(train, class_6)
train <- rbind(train, class_7)
train <- rbind(train, class_8)
train <- rbind(train, class_9)

train <- train %>% arrange(id)

#Create numerical ids for the data
test_ids <- seq(1, 20555)

#Create the final training data table
final_train <- data.table()
final_train$id <- train$id
final_train$class <- train$class

#Add the embedded values to each row
final_train <- cbind(final_train, train_emb)#, fill = TRUE)

#Create the final test data
final_test <- data.table()
final_test$id <- test_ids
final_test$class <- 11

final_test <- cbind(final_test, test_emb)#, fill = TRUE)


#GET TSNE DATA AND ADD IT TO THE FINAL SET
train_pca <- prcomp(train_emb)

test_pca <- prcomp(test_emb)

train_pca_dt <- data.table(unclass(train_pca)$x)

test_pca_dt <- data.table(unclass(test_pca)$x)

screeplot(train_pca)
biplot(train_pca)

train_tsne <- Rtsne(train_pca_dt, pca = F, perplexity = 30)

test_tsne <- Rtsne(test_pca_dt, pca = F, check_duplicates = F, perplexity = 30)

train_tsne_dt <- data.table(train_tsne$Y)

test_tsne_dt <- data.table(test_tsne$Y)

ggplot(train_tsne_dt,aes(x=V1,y=V2))+geom_point()
#ggplot(test_tsne_dt,aes(x=V1,y=V2))+geom_point()

final_train$tsne1 <- train_tsne_dt$V1
final_train$tsne2 <- train_tsne_dt$V2

final_test$tsne1 <- test_tsne_dt$V1
final_test$tsne2 <- test_tsne_dt$V2

########################
# write out to interim #
########################
fwrite(final_train,"./project/volume/data/interim/training_set.csv")
fwrite(final_test,"./project/volume/data/interim/testing_set.csv")


