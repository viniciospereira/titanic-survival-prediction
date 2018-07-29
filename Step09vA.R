############## STEP 9  skip missing value imputation for xgboost - keep missing values - because you can !###########

elem <- function(x,xname=NULL,drop=T)
{
  el <- data.frame(1:length(x))
  uvals <- sort(unique(x))
    for(i in 1:length(uvals))
  {  
    el[i] <- as.numeric(x==uvals[i])
    names(el)[i] <- paste(xname,uvals[i],sep="")
  }
  if(drop)
    el <- el[-ncol(el)]
  return(as.matrix(el))
}  

# function convelem converts all non-numeric fields to dummy fields in data frame X
convelem <- function(X)
{ 
  catfields <- which(sapply(X,function(x) is.character(x) | is.factor(x) | is.logical(x)))  
  if(length(catfields)==0)
    return(X)
  Xnum <- X[,-catfields]
  for(i in seq(catfields))
    Xnum <- cbind(Xnum,elem(X[,catfields[i]],xname=names(X)[catfields[i]]))
  Xnum
}  


#convelem <- function(X)  model.matrix( ~ .,data=model.frame(X,na.action="na.pass"))[,-1]


XGBmissing <- buildXGBOOST(convelem(X7keepmissing),Y)


XGBmissing <- X7keepmissing %>% convelem %>% buildXGBOOST(Y)

save.image("step9save.RData")




############### END STEP 9 ##################################
