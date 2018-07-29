

########## STEP 7: Ticket codes #####################################

table(testtrain5$Ticket) %>% sort  #what do ticket codes look like ?

ticketlist <- testtrain5$Ticket %>% as.character()  %>% strsplit(" ")
ticketlength <- ticketlist %>% sapply(length) #get the length of the code - how many entires seperated by blanks ?

ticketlength %>% table %>% sort

ticketcode <- ticketlist %>% 
  sapply(function(x) if(length(x)>1) x[1] else "BLANK") %>%  unname %>%     #get the text codes at the start of tickets
  sapply(function(x) gsub("\\.","",x))  %>% unname %>%                  #get rid of "."
  sapply(function(x) gsub("/","",x)) %>%  unname %>%                     #get rid of "/"
  sapply(function(x) gsub("CASOTON","STON",x)) %>%             #combine all "STON" and "SOTON" tickets - "Southhampton"
  sapply(function(x) gsub("SOTON","STON",x)) %>%
  sapply(function(x) gsub("Paris","PARIS",x)) 

ticketcode %>% table %>% sort
######ticket codes are too rare

##### take first letter and second letter
ticketletter1 <- ticketcode %>% sapply(substr,1,1)
ticketletter2 <- ticketcode %>% sapply(substr,1,2)

ticketletter1  %>% table %>% sort
ticketletter2  %>% table %>% sort

ticketnumber <- ticketlist %>% 
  sapply(function(x) if(length(x)>1) x[2] else x[1])  #%>% 
  #as.numeric
ticketnumber[ticketnumber=="LINE"] <- 0
ticketnumber[ticketnumber=="Basle"] <- -1
ticketnumber <- ticketnumber %>% as.numeric

testtrain6 <- cbind(testtrain5,ticketletter1,ticketletter2,ticketnumber,ticketcode)
trainnames6 <- c(trainnames5,"ticketletter1","ticketletter2","ticketnumber","ticketcode")
train6 <- testtrain6[testtrain6$Survived !=-2000,-1]
X6a <- train6[,trainnames6]
X6 <- ageimputeMedian(X6a)
#Run6 <- runmodels(X6,Y,runinteractions=F)
Run6a <- runmodels(X6,Y,runinteractions=T)

xyhat <- cbind(X6,yhat=Run6a$RF$votes[,2])

save.image("step7save.RData")


####################### END STEP 7 ######################################
