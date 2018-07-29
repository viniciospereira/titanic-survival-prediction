
###### STEP 6 : MORE ABOUT CABINS: CABIN NAMES AND NUMBERS #######################

#get the letter of the first cabin, and its number (yes, we could do better, but this might be plenty)
cabcode <- cablist %>% sapply(function(x) x[1]) #get the cabin codes of the first cabins
cabletter <- cabcode %>% sapply(substr,1,1) %>% as.character() #get the letter of the cabin code
cabletter[is.na(cabletter)] <- "Missing" #set missing letters to "Missing"
cabletter <- as.factor(cabletter) # make the letter a factor
cabletter %>% table %>% sort


cabnum <- cabcode %>% sapply(substr,2,5) %>% as.numeric #get the number of the first cabin code
cabnum[is.na(cabnum)] <- 0 #set missing cabin numbers to 0
cabnum %>% table %>% sort

testtrain5 <- cbind(testtrain4,cabnum,cabletter) #add cabin letters and numbers to previous data set

train5 <- testtrain5[testtrain5$Survived !=-2000,-1] #extract the training set as before
trainnames5 <- c(trainnames4,"cabnum","cabletter")
X5a <- train5[,trainnames5]
X5 <- ageimputeMedian(X5a)
Run5 <- runmodels(X5,Y,runinteractions=T)



save.image("step6save.RData")

########## END STEP 6 #############################################################
