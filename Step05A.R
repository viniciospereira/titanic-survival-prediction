

#################### STEP 5  : add some information about cabin names and numbers and use functionalised code

summary(testtrain3)  # let's look at the data again

testtrain3$Cabin %>% table %>% sort


CabinIndex <- as.integer(testtrain3$Cabin=="")  #is there a nonempty cabin for thsi passanger ?
cablist <- testtrain3$Cabin %>% as.character()  %>% strsplit(" ") #what cabins does the passenger occupy ?
NumCabins <- sapply(cablist,length) #how many cabins does the passanger occupy ?

testtrain4 <- cbind(testtrain3,CabinIndex,NumCabins)
train4 <- testtrain4[testtrain4$Survived !=-2000,-1]

#trainnames3a <- trainnames3
trainnames4 <- c(trainnames3,"CabinIndex","NumCabins")
X4a <- train4[,trainnames4]

X4 <- ageimputeMedian(X4a)
Run4 <- runmodels(X4,Y,runinteractions=F)

Run4i <- runmodels(X4,Y,runinteractions=T)

save.image("step5save.RData")


###### END STEP 5 ################################################################
