#---------------------------#
#---------- SDS 313 --------#
#--- String Manipulation ---#
#------- Oct. 26, 2022 -----#
#---------------------------#



#
#--- String Basics ---#
#

#You can use " " or ' ' to make a string

string1a <- "This is a string"
string1a

string1b <- 'This is also a string'
string1b

#Use \' or \" to include those within your string
string3 <- "It\'s been a great day!"
string3

#\n is a line break:
string4a <- "Go over there...\n NOW!"
string4a
#use cat() to print actual text:
cat(string4a)

#\t is a tab:
string4b <- "Go over there...\t NOW!"
cat(string4b)

#\ can be used to denote symbols like:
string5 <- "My favorite letter is \u00b5"
cat(string5)



#
#--- stringr Text Functions ---#
#

library(tidyverse)
library(stringr)

#String length:
string3
str_length(string3)

#String subsetting
myfruit <- c("Apple", "Banana", "Pear") 
str_sub(myfruit, 1, 3)

#Negative indices count backwards from end
str_sub(myfruit, -3, -1)

#Upper vs. lower case
str_to_upper("my favorite food is pizza")
str_to_lower("my FAVORITE food is pizza")
str_to_title("my favorite food is pizza")
str_to_sentence("my favorite food is pizza")


#
#--- Finding Text Matches ---#
#

install.packages("htmlwidgets")
library(htmlwidgets)

x <- c("bear.", "fox", "tiger.", "koala", "arctic fox")

#Finding specific character(s) within a string:
str_detect(x, "a")
str_count(x, "a")
str_subset(x, "a")

#Highlight matches in viewer:
str_view(x, "a")
str_view(x, "ar")

#Use . as a wildcard:
str_view(x, ".a.")
str_view(x, ".e.")
str_view(x, ".e")

#Matching a period with \\.
str_view(x, "\\.")

#^ matches the start of a string, $ matches the end
str_view(x, "^b")
str_view(x, "a$")

#Exact matches
str_view(x, "fox")
str_view(x, "^fox$")

#Using (|) or [] for 'or' matches
str_view(x, "ti(g|c)")
str_view(x, "[abc]")


#Cheat Sheet: https://github.com/rstudio/cheatsheets/blob/main/strings.pdf


#
#Exercises using the built-in words dataset:
#

words

#
#1. What is the average length of the words in
#   this vector?
#
#2. How many of these words start with q?
#   What are they?
#
#3. What proportion of these words end in a vowel? 
#
#4. Find all words that consist only of consonants
#   (no a, e, i, o, or u). 
#
#5. Make a tibble that contains these words as well
#   as a variable of the count of vowels in each word.



#
#--- Extracting and Replacing ---#
#

sentences

#Find all the sentences that mention a color:
mycolors <- str_subset(sentences, "blue|green|purple")
mycolors

#Extract the colors that were matched:
str_extract(mycolors, "blue|green|purple")
table(str_extract(mycolors, "blue|green|purple"))

#Replace text with other text:
str_replace_all(mycolors, "blue", "-")
str_replace_all(mycolors, c("blue" = "-", "green" = "+", "purple" = "*"))



#
#Exercises: Use the web scraped Martha Stewart 
#           recipes from last time:
#

library(rvest)
martha_link <- "https://www.marthastewart.com/1505788/recipes"
martha_page <- read_html(martha_link)
martha_recipes <- html_text(html_elements(martha_page,".elementFont__resetHeading"))
martha_recipes

#
#1. Confirm (with code) that there are spaces at the 
#   start and end of each recipe.
#
#2. Remove the start and end spaces from each recipe.
#
#3. How many of these are chicken recipes?
#
#4. Which recipes have apple or cake mentioned? What
#   about both apple and cake?
#
#5. Change cookies into cupcakes.
#
