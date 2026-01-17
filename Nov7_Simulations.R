#---------------------------#
#---------- SDS 313 --------#
#------- Simulations -------#
#------- Nov. 7, 2022 ------#
#---------------------------#


#
#Generating random numbers
#


#Uniform distribution:
runif(1,0,1)

#Set starting point within pseudo random number generator:
set.seed(5)
runif(1,0,1)

#Simulating a probability distribution
x <- runif(10000,0,1)
hist(x, main='Uniform (0,1) Distribution')

y <- rnorm(10000,0,1)
hist(y, main='Normal (0,1) Distribution')

#
#Exercises
#
#1. Randomly generate an integer between
#   0 and 10 (inclusive).
#
#2. Simulate randomly drawing 10,000
#   integers between 0 and 10 and plot
#   your results with a histogram.
#
#3. What percentage of your random 
#   numbers in #2 above were 5's?
#   Is this close to what you expected?
#


#Sampling from a vector

x <- c(4,6,9,10,15,20)
sample(x,1)
sample(x,3,replace=TRUE)




#
#Simulations using for loops
#


#Rolling a dice:

myrolls <- numeric(0)

for (i in 1:5000) {
  x <- sample(1:6,1)
  myrolls[i] <- x
}

hist(myrolls)


#Calculate the probability of rolling a "4"

myrolls <- numeric(0)
my4prob <- numeric(0)

for (i in 1:5000) {
  x <- sample(1:6,1)
  myrolls[i] <- x
  my4prob[i] <- sum(myrolls==4)/i
}

plot(1:5000, my4prob, type='l')
abline(h=1/6, lty=2, col='red')


#
#Exercises:
#
#1. Update this dice roll simulation to 
#   account for rolling two dice at once.
#
#2. Plot the probability of rolling a 4
#   in your updated simulation for #1. Does
#   it converge to what you expected?
#


#Sample from a dataset

#Import genes dataset
genes <- read.csv('genes.csv')

#Randomly select 10 genes
sample(genes$nucleotides, 10)

#
#Demonstration of the Central Limit Theorem
#

#Population distribution
hist(genes$nucleotides, main='Distribution of Gene Lengths', xlab='Gene Length (nucleotides)', right=F, col='grey')

# Generate 5,000 random samples of n=10 and store means
mymeans <- numeric(0)

for (i in 1:5000) {
  x <- sample(genes$nucleotides, size=10)
  mymeans <- c(mymeans, mean(x))
}

hist(mymeans, main='Sampling Distribution for n=10', xlab='Mean Gene Length (nucleotides)', col='grey')



#
#Exercise:
#
# What is the minimum sample size needed to 
# make this sampling distribution "normal?"
#


