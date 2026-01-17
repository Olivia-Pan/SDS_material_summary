#---------------------------#
#---------- SDS 313 --------#
#------- Working with ------#
#------- Online Data -------#
#------- Oct. 24, 2022 -----#
#---------------------------#



#
#--- Importing data that was exported
#    using other software ---#
#

#Example: SAS file from Canvas

install.packages('sas7bdat')
library(sas7bdat)
texas <- read.sas7bdat('texas.sas7bdat')


#
#--- Downloading Online Data ---#
#

#Example 1: https://data.cms.gov/provider-data/topics/hospitals

#Example 2: https://repository.niddk.nih.gov/studies/dpp/
  

#
#--- Web Scraping ---#
#

#Manual example:
# https://www.floridamuseum.ufl.edu/shark-attacks/maps/world/ 


#Using code:

#install.packages('rvest')
library(rvest)

#Step 1: Store HTML source code as a list with read_html()
IMDb_link <- "https://mycurlyadventures.com/fun-austin-date-night-ideas/"
IMDb_page <- read_html(IMDb_link)
IMDb_page

#Step 2: Use Chrome's "Selector Gadget" to find the HTML tag
# of the elements you want to scrape (movie titles)
movie_titles <- html_text(html_elements(IMDb_page, ".wp-block-heading:nth-child(63) , .wp-block-heading:nth-child(65) , .mv-ad-box+ .wp-block-heading , .wp-block-heading:nth-child(28) , .wp-block-heading:nth-child(34) , .wp-block-heading:nth-child(36) , .wp-block-heading:nth-child(38) , .wp-block-heading:nth-child(40) , .size-large+ .wp-block-heading , .wp-block-heading:nth-child(21)"))
movie_titles


#Exercises: 
#  1. Scrape the year and IMDb Average Rating of these movies.
#  2. Create a tibble that contains all three variables.



#More examples:
snl_link <- "https://www.nbc.com/saturday-night-live" 
snl_page <- read_html(snl_link)
snl_titles <- html_text(html_elements(snl_page,".tile__title"))
snl_titles

martha_link <- "https://www.marthastewart.com/1505788/recipes"
martha_page <- read_html(martha_link)
martha_recipes <- html_text(html_elements(martha_page,".elementFont__resetHeading"))
martha_recipes

amazon_link <- "https://www.amazon.com/Gifts-College-Student/s?k=Gifts+for+The+College+Student"
amazon_page <- read_html(amazon_link)
amazon_names <- html_text(html_elements(amazon_page,".a-size-base-plus"))
amazon_names
amazon_stars <- html_text(html_elements(amazon_page,".aok-align-bottom"))
amazon_stars


#
#Exercise:  Try to scrape data from one or more of these:
# 1. Austin Date Ideas: https://mycurlyadventures.com/2020/08/16/fun-austin-date-night-ideas/ 
# 2. Tweets by your favorite celebrity in last 30 days.
# 3. Monkeypox cases: https://www.cdc.gov/poxvirus/monkeypox/response/2022/us-map.html 
# 4.Travel destinations: https://www.forbes.com/sites/laurabegleybloom/2019/09/04/bucket-list-travel-the-top-50-places-in-the-world/?sh=248d064820cf 
#

