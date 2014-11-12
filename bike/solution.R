set.seed(1000)

loadDataset <- function(fname){
  dr <- read.csv(fname)
  dr$datetime <- as.POSIXct(dr$datetime,tz='GMT',format="%Y-%m-%d %H:%M:%S")
  dr$hour <- as.factor(as.integer(format(dr$datetime, "%H")))
  dr$weekday <- as.factor(weekdays(dr$datetime))
  for(catVar in catVars){
    dr[,catVar] = as.factor(dr[,catVar])
  }
  dr
}

correctOutput <- function(d){
  func <- function(x){
    if(x>=0){
      x <- round(x)
    }else{
      x <- 0
    }
    x
  }
  sapply(d, func)
}

predictModel <- function(model, data){
  out <- predict(model, newdata=data)
  correctOutput(out)
}

rmsle <- function (y, y_p){
  if(length(y) != length(y_p)){
    stop('y and y_p should have the same length')
  }
  epsl <- 1e-9
  N <- length(y)
  ((1/N) * sum((log(y+1) - log(y_p+1))^2))^0.5
}

evaluateModel <- function(model, data, var){
  dataPred <- predictModel(model, data=data)
  print(sprintf("rmsle for %s: %f", var, rmsle(data[,outVar], dataPred)))
}

d <- loadDataset('train.csv')
dt <- loadDataset('test.csv')
catVars <- c('hour',
             'weekday', 
             'season', 
             'holiday', 
             'workingday',
             'weather')
numVars <- c('temp',
             'atemp',
             'humidity',
             'windspeed')
outVar <- 'count'
fitVars <- paste(c(catVars, numVars), sep='')

# split
d$randn = runif(dim(d)[1])
train <- subset(d, randn < 0.7)
test <- subset(d, randn >= 0.7)

l_fml <- formula(paste(outVar, 
               ' ~ ', 
               paste(c(catVars, numVars), collapse=' + ')))
l_model <- lm(l_fml, data=train)
evaluateModel(l_model, train, outVar)
evaluateModel(l_model, test, outVar)
# [1] "rmsle for count: 1.047211"
# [1] "rmsle for count: 1.086207"

library(rpart)
t_fml <- l_fml
t_model <- rpart(t_fml, data=train)
evaluateModel(t_model, train, outVar)
evaluateModel(t_model, test, outVar)
# [1] "rmsle for count: 0.824456"
# [1] "rmsle for count: 0.828872"

library('party')
ct_fml <- l_fml
ct_model <- ctree(as.formula(ct_fml), data=train)
evaluateModel(ct_model, train, outVar)
evaluateModel(ct_model, test, outVar)
# [1] "rmsle for count: 0.464043"
# [1] "rmsle for count: 0.497510"

# ctree scored best, use it as submission
ct_model <- ctree(as.formula(ct_fml), data=d)
dt$count <- predict(ct_model, newdata=dt)
write.csv(dt[,c('datetime','count')], 
          file='submission.csv',
          row.names=F)
