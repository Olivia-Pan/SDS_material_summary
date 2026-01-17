#---------------------------#
#---------- SDS 313 --------#
#-------- For Loops --------#
#------ Oct. 31, 2022 ------#
#---------------------------#


#
#for loops
#

#Example 1:

n <- 5

for (i in 1:n){
  print(i)
}


#Example 2: if statement

for (i in 1:10) { 	
  if (i > 5) print(i)
} 					


#Example 3: else statements

for (i in 1:10) { 	
  if (i > 5) print(i)
  else print('Less than 6')
} 					


#Example 4: else if statements

for (i in 1:10) { 	
  if (i > 5) print(i)
  else if (i > 2) print('Between 3 and 5')
  else print('Less than 3')
} 	


#Example 5: Storing output

myoutput <- numeric(0)  

for (i in 1:10) { 	
  myoutput[i] <- i			
} 					

myoutput



#
#Exercises:  Import the 'cars' dataset
#

cars <- read.csv('cars.csv')

#
#1. Using a loop, calculate the mean Highway
#   MPG of each "type" of car.  Print to the 
#   screen the name of each car type next to 
#   its mean.
#
#2. Try accomplishing the same thing as above
#   without using a loop.
#



#
#Some For Loop Applications
#


#Create a subset for each level of a factor

mytypes <- unique(cars$Type)
n <- length(mytypes)

for (i in 1:n){
  assign(paste(mytypes[i],'_data',sep=''),cars[cars$Type==mytypes[i],])
}



#Create separate plots for each level of a factor

mytypes <- unique(cars$Type)
n <- length(mytypes)

for (i in 1:n){
  X <- cars[cars$Type==mytypes[i],]
  plot(X$Horsepower,X$Price,xlim=c(50,300),ylim=c(8,50),xlab='Horsepower',ylab='Price', pch=20, main=paste('Horsepower and Price for',mytypes[i],'Cars'))
}




#
#Exercises:
#
#1. Pick a variable that you want to investigate
#   across the different car manufacturers
#   in this dataset.
#
#2. Using a loop, export a univariate plot of
#   your chosen variable for each manufacturer
#   separately.  Have the files be named so that
#   it is clear which plot belongs to which
#   manufacturer.
#
#   Hint: remember the pdf() and dev.off() 
#         functions.
#


