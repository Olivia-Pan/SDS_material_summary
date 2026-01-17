#---------------------------#
#---------- SDS 313 --------#
#------ User-Written -------#
#-------- Functions --------#
#------- Nov. 2, 2022 ------#
#---------------------------#


#A simple example:

hello <- function(){
  print('Hello World!')
}

hello()



#Including function arguments:

mysum <- function(x,y){
  x + y
}

mysum(5,6)
mysum(1:10,1)


#Default values for arguments:

mysum <- function(x,y=0){
  x + y
}

mysum(5)
mysum(5,y=1)

#Think about the values you want to return:



#Explicit return()

mymode <- function(x){
  if (length(x)==1){
    return(x)
  }
  else {
  xtable <- table(x)
  maxcount <- max(xtable)
  xmode <- xtable[xtable==maxcount]
  as.numeric(xmode)
  }
}

x2 <- c(5,6,6,6,7)
mymode(x2)
mymode(x1)



#
#Exercises: 
#
#1. Write a function that returns the mean 
#   of a vector (without using mean()).  
#   Confirm your function works by feeding
#   it a numeric vector and comparing it
#   to the output from mean().  
#
#2. Update your function so it rounds to
#   3 decimal places in the return by default,
#   but this can be changed with an
#   additional argument.
#
#3. What happens when you feed it a 
#   character vector?
#



#Including stops for condition checks:

mymode <- function(x){
  stopifnot(is.numeric(x))
  xtable <- table(x)
  maxcount <- max(xtable)
  xmode <- names(xtable[xtable==maxcount])
  as.numeric(xmode)
}

x <- c(4,2,5,5,7,1,9,2,12)
mymode(x)
y <- c('yes','no')
mymode(y)


#Writing your own warning messages:

mymode <- function(x){
  if (is.numeric(x)==FALSE){
    stop('Input must be numeric!')
  }
  xtable <- table(x)
  maxcount <- max(xtable)
  xmode <- names(xtable[xtable==maxcount])
  as.numeric(xmode)
}

x <- c(4,2,5,5,7,1,9,2,12)
mymode(x)
y <- c('yes','no')
mymode(y)



#Import cars dataset:
cars <- read.csv('cars.csv') 

#For loop within a function:

my3table <- function(x,y,z){
  zlevels <- unique(z)
  n <- length(zlevels)
  for (i in 1:n){
    print(zlevels[i])
    print(table(x[z==zlevels[i]],y[z==zlevels[i]]))
  }
}

cars$Passengers <- as.factor(cars$Passengers)
my3table(cars$Type,cars$Passengers,cars$Origin)



#
#Exercises: 
#
#1. Write a function that takes three numeric
#   vectors (1 outcome, 2 predictors) and plots 
#   a scatterplot for each predictor vs. the 
#   outcome. Test it out with horsepower vs. price
#   and Highway MPG vs. prices for the cars dataset. 
#
#2. Make the default color of your plots your favorite
#   color, but allow this to be changed with another
#   argument (or two arguments, one for each plot).
#
#3. Add a check in your function that ensures
#   that all three variables are numeric.
#
#4. Update your function so that it returns the 
#   correlation between the each pair of variables
#   to the console.
#

