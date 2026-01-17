#course 1
import streamlit as st
from graphviz import Digraph
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



def show():
    st.markdown("### Unit 1: Intro to Data Science")

    st.markdown("#### 1. Visualizing and Describing Data")

    st.markdown("##### 1a. Variable Types")
    #st.image("1.png", caption="Variable Types", use_column_width=True)

    st.graphviz_chart("""
    digraph VariableTypes {
    rankdir=LR;
    node [shape=box, style=rounded, fontname="Helvetica", fontsize=10];

    A [label=" "];
    
    A -> B [label=""];
    B -> C;
    C -> D;
    C -> E;

    A -> F;
    F -> G;
    G -> H;
    G -> I;

    A -> J;
    J -> K;
    J -> L;

    A -> M;
    M -> N;
    M -> O;

    A -> P;
    P -> Q;
    P -> R;

    B [label="Single Categorical Variable"];
    C [label="defines membership in a group"];
    D [label="describes with frequencies or proportions"];
    E [label="displays with bar chart"];

    F [label="Single Numeric Variable"];
    G [label="quantitative measurement"];
    H [label="describes with\\n1. mean/median\\n2. standard deviation/quartiles"];
    I [label="displays with histogram or boxplot"];

    J [label="2 Numeric Variables"];
    K [label="describes with correlation: -1 ≤ r ≤ 1"];
    L [label="displays with scatterplot"];

    M [label="2 Categorical Variables"];
    N [label="describes with row/column percentages"];
    O [label="displays with grouped bar chart"];

    P [label="One Categorical + One Numeric Variable"];
    Q [label="compare average/spread for each group"];
    R [label="grouped histogram or boxplot"];
    }
    """)

    st.markdown("##### 1b. Data Types")

    st.graphviz_chart("""
    digraph DataTypes {
    rankdir=LR;
    node [shape=box, style=rounded, fontname="Helvetica", fontsize=10];

    A1 [label=" "];

    A1 -> B1;
    B1 [label="Numeric Data Types"];
    B1 -> F1 [label=" "];
    F1 [label="int = integers (discrete)"];
    B1 -> G1;
    G1 [label="dbl = double (real numbers, continuous)"];

    A1 -> C1;
    C1 [label="Categorical Data Types"];
    C1 -> H1;
    H1 [label="chr = character vectors (e.g., 'apple')"];
    C1 -> I1;
    I1 [label="fctr = factors (categorical with fixed levels)"];
    H1 -> T1;
    I1 -> T1;
    T1 [label="Nominal: not ranked\\nOrdinal: ranked or ordered"];

    A1 -> D1;
    D1 [label="Date/Time Data Types"];
    D1 -> J1;
    J1 [label="dttm = date and time"];
    D1 -> K1;
    K1 [label="date = date only"];
    D1 -> M1;
    M1 [label="time = time only"];

    J1 -> N1;
    K1 -> N1;
    M1 -> N1;
    N1 [label="format codes:\\n%d, %m, %b, %B, %y, %Y"];

    J1 -> O1;
    K1 -> O1;
    M1 -> O1;
    O1 [label="functions:\\nas.Date(), ymd(), dmy(),\\nmake_date(), ymd_hms()"];

    M1 -> P1;
    P1 [label="hms package:\\nhms(), as_hms()"];

    A1 -> E1;
    E1 [label="Logical Data Types"];
    E1 -> L1;
    L1 [label="lgl = TRUE or FALSE"];

    A1 -> Y1;
    Y1 [label="NA Values"];
    Y1 -> Z1;
    Z1 [label="is.na(), na.rm = TRUE"];
    }
    """)


    #st.image("2.png", caption="Data Types", use_column_width=True)

    st.markdown("**Example of make_date:**")
    st.code("""
    library(dplyr)
    library(lubridate)
    library(hms)

    df <- tibble(
    year = c(2021, 2022),
    month = c(1, 12),
    day = c(5, 31)
    )

    df <- df %>%
    mutate(date = make_date(year, month, day))

    df
    """, language="r")

    st.markdown("**hms example:**")
    st.code("""
    x <- hms::hms(hours = 2, minutes = 30, seconds = 15)
    x
    """, language="r")

    st.markdown("##### 1c. Matrices, DataFrames, Tibbles")
    #st.image("images/mermaid_matrix_dataframe_tibble.png", caption="Matrix, DataFrame, Tibble", use_column_width=True)

    st.graphviz_chart("""
    digraph DataStructures {
    rankdir=LR;
    node [shape=box, style=rounded, fontname="Helvetica", fontsize=10];

    A2 [label=" "];

    A2 -> B2;
    B2 [label="Matrix"];
    B2 -> E2;
    E2 [label="functions:\\nmatrix()\\nis.matrix()\\nas.matrix()"];
    B2 -> F2;
    F2 [label="definition:\\nhomogeneous 2D dataset"];
    B2 -> G2;
    G2 [label="performs arithmetic\\nitem-by-item"];

    A2 -> C2;
    C2 [label="DataFrame"];
    C2 -> H2;
    H2 [label="functions:\\ndata.frame()\\nis.data.frame()\\nas.data.frame()"];
    C2 -> I2;
    I2 [label="definition:\\nheterogeneous 2D dataset"];
    C2 -> J2;
    J2 [label="has named columns and rows\\neach row = case"];

    A2 -> D2;
    D2 [label="Tibble"];
    D2 -> K2;
    K2 [label="functions:\\ntibble()\\nis.tibble()\\nas_tibble()"];
    D2 -> L2;
    L2 [label="like a dataframe\\nbut easier for big data"];
    D2 -> M2;
    M2 [label="preserves types and variable names"];
    D2 -> N2;
    N2 [label="prints only first 10 rows\\nand columns that fit"];
    }
    """)


    st.markdown("**Example:**")
    st.code("""
    # 1. Matrix: homogeneous data (all numeric here)
    mat <- matrix(1:6, nrow = 2, ncol = 3)
    print("Matrix:")
    print(mat)

    # 2. Data frame: heterogeneous data types allowed
    df <- data.frame(
    id = c(1, 2),
    name = c("Alice", "Bob"),
    score = c(88.5, 92.3)
    )
    print("Data Frame:")
    print(df)

    # 3. Tibble: modern, tidyverse-friendly data frame
    tb <- tibble(
    id = c(1, 2),
    name = c("Alice", "Bob"),
    score = c(88.5, 92.3)
    )
    print("Tibble:")
    print(tb)
    """, language="r")

    st.markdown("##### 1d. Merging Data")
    #st.image("images/mermaid_joins.png", caption="Join Types in dplyr", use_column_width=True)

    st.graphviz_chart("""
    digraph JoinTypes {
    rankdir=LR;
    node [shape=box, style=rounded, fontname="Helvetica", fontsize=10];

    A5 [label="Join Types"];

    A5 -> B5;
    B5 [label="inner_join(x, y)"];

    A5 -> C5;
    C5 [label="left_join(x, y)"];

    A5 -> D5;
    D5 [label="right_join(x, y)"];

    A5 -> E5;
    E5 [label="full_join(x, y)"];

    A5 -> F5;
    F5 [label="semi_join(x, y)"];

    A5 -> G5;
    G5 [label="anti_join(x, y)"];
    }
    """)

    st.image("inner.png", width=600)
    st.image("outer.png", width=600)
    st.image("filter.png", width=600)
    st.write("")
    st.markdown("#### 2. dplyr Package Functions")
    #st.image("images/mermaid_dplyr.png", caption="dplyr Functions", use_column_width=True)

    st.graphviz_chart("""
    digraph DplyrFunctions {
    rankdir=LR;
    node [shape=box, style=rounded, fontname="Helvetica", fontsize=11];

    A3 [label=" "];

    A3 -> B3;
    B3 [label="filter"];
    B3 -> G3;
    G3 [label="select observations\\nby criteria"];

    A3 -> C3;
    C3 [label="arrange"];
    C3 -> H3;
    H3 [label="reorder rows"];

    A3 -> D3;
    D3 [label="select"];
    D3 -> I3;
    I3 [label="pick variables\\nby name"];

    A3 -> E3;
    E3 [label="mutate"];
    E3 -> J3;
    J3 [label="create new variables\\nfrom existing ones"];

    A3 -> F3;
    F3 [label="summarize"];
    F3 -> K3;
    K3 [label="collapse many values\\nto single summary"];
    }
    """)

    st.code("""
    df <- tibble(
    name = c("Alice", "Bob", "Charlie", "David"),
    age = c(25, 30, 22, 28),
    score = c(88, 95, 77, 84)
    )

    df %>% filter(age > 25)

    df %>% arrange(score)

    df %>% select(name, score)

    df %>% mutate(passed = score >= 80)

    df %>% summarize(avg_score = mean(score))
    """, language="r")

    st.markdown("#### 3. tidyr Package Functions")
    #st.image("images/mermaid_tidyr.png", caption="tidyr Functions", use_column_width=True)

    st.graphviz_chart("""
    digraph DplyrFunctions {
    rankdir=LR;
    node [shape=box, style=rounded, fontname="Helvetica", fontsize=10];

    A3 [label=" "];

    A3 -> B3;
    B3 [label="filter"];
    B3 -> G3;
    G3 [label="select observations\\nby criteria"];

    A3 -> C3;
    C3 [label="arrange"];
    C3 -> H3;
    H3 [label="reorder rows"];

    A3 -> D3;
    D3 [label="select"];
    D3 -> I3;
    I3 [label="pick variables\\nby their names"];

    A3 -> E3;
    E3 [label="mutate"];
    E3 -> J3;
    J3 [label="create new variables\\nfrom existing ones"];

    A3 -> F3;
    F3 [label="summarize"];
    F3 -> K3;
    K3 [label="collapse values\\nto single summary"];
    }
    """)


    st.code("""
    library(tidyr)
    library(tibble)

    wide_data <- tibble(
    name = c("Alice", "Bob"),
    math = c(90, 85),
    english = c(88, 92)
    )

    long_data <- wide_data %>%
    gather(key = "subject", value = "score", math:english)

    united_data <- long_data %>%
    unite("name_subject", name, subject, sep = "_")

    separated_data <- united_data %>%
    separate(name_subject, into = c("name", "subject"), sep = "_")

    final_data <- separated_data %>%
    spread(key = subject, value = score)
    """, language="r")

    # --- Section 4: Web Scraping ---

    st.markdown("#### 4. Web Scraping")
    st.code('''
    library(rvest)

    #Step 1: Store HTML source code as a list with read_html()
    IMDb_link <- "https://mycurlyadventures.com/fun-austin-date-night-ideas/"
    IMDb_page <- read_html(IMDb_link)

    #Step 2: Use Chrome's "Selector Gadget" to find the HTML tag
    # of the elements you want to scrape (movie titles)
    movie_titles <- html_text(html_elements(IMDb_page, ".wp-block-heading:nth-child(63) ,\
    .wp-block-heading:nth-child(65) , .mv-ad-box+ .wp-block-heading , .wp-block-heading:nth-child(28) , .wp-block-heading:nth-child(34) , .wp-block-heading:nth-child(36) , .wp-block-heading:nth-child(38) , .wp-block-heading:nth-child(40) , .size-large+ .wp-block-heading , .wp-block-heading:nth-child(21)"))
    movie_titles
    ''', language='r')

    st.write("")
    # --- Section 5: Stringr package ---

    st.markdown("#### 5. Stringr Package")

    st.code('''
    #--- String Basics ---#

    string1a <- "This is a string"
    string1b <- 'This is also a string'
    string3 <- "It\\'s been a great day!"
    string4a <- "Go over there...\\n NOW!"
    cat(string4a)
    string4b <- "Go over there...\\t NOW!"
    cat(string4b)
    string5 <- "My favorite letter is \\u00b5"
    cat(string5)
    ''')

    string1a = "This is a string"
    string1b = 'This is also a string'
    string3 = "It's been a great day!"
    string4a = "Go over there...\n NOW!"
    string4b = "Go over there...\t NOW!"
    string5 = "My favorite letter is µ"

    st.write("", string3)
    st.text(string4a)
    st.text(string4b)
    st.text(string5)

    st.code('''
    #--- Stringr Text Functions ---#
    string3
    str_length(string3)
    myfruit <- c("Apple", "Banana", "Pear") 
    str_sub(myfruit, 1, 3)
    str_sub(myfruit, -3, -1)
    str_to_upper("my favorite food is pizza")
    str_to_lower("my FAVORITE food is pizza")
    str_to_title("my favorite food is pizza")
    str_to_sentence("my favorite food is pizza")
    ''')

    myfruit = ["Apple", "Banana", "Pear"]
    subset1 = [f[:3] for f in myfruit]
    subset2 = [f[-3:] for f in myfruit]
    st.write("str_length of \"It\'s been a great day! \" is 22")
    st.write("Subsets (first 3 letters):", subset1)
    st.write("Subsets (last 3 letters):", subset2)

    text = "my FAVORITE food is pizza"
    st.write("Upper:", text.upper())
    st.write("Lower:", text.lower())
    st.write("Title:", text.title())
    st.write("Sentence:", text.capitalize())

    st.code('''
    #--- Finding Text Matches ---#
    x <- c("bear.", "fox", "tiger.", "koala", "arctic fox")
    str_detect(x, "a")
    str_count(x, "a")
    str_subset(x, "a")
    str_view(x, "a")
    str_view(x, "ar")
    str_view(x, ".a.")
    str_view(x, ".e.")
    str_view(x, ".e")
    str_view(x, "\\\.")
    str_view(x, "^b")
    str_view(x, "a$")
    str_view(x, "fox")
    str_view(x, "^fox$")
    str_view(x, "ti(g|c)")
    str_view(x, "[abc]")
    ''', language='r')

    x = ["bear.", "fox", "tiger.", "koala", "arctic fox"]
    matches_a = [s for s in x if 'a' in s]
    count_a = [s.count('a') for s in x]
    st.write("Matches 'a':", matches_a)
    st.write("Count of 'a':", count_a)

    # --- Section 6: Loops and Functions ---
    st.markdown("#### 6. Loops and Functions")
    code = '''
    def hello():
        return "Hello World!"

    def mymode(x):
        if len(x) == 1:
            return x[0]
        vals, counts = np.unique(x, return_counts=True)
        return vals[np.argmax(counts)]

    hello()
    mymode([1, 2, 2, 3, 4])
    '''

    st.write("Code:")
    st.code(code, language='python')

    # Now display the results of running the functions
    st.write("Results:")

    # Run the actual functions
    def hello():
        return "Hello World!"

    def mymode(x):
        if len(x) == 1:
            return x[0]
        vals, counts = np.unique(x, return_counts=True)
        return vals[np.argmax(counts)]

    st.write("hello() →", hello())
    st.write("mymode([1, 2, 2, 3, 4]) →", mymode([1, 2, 2, 3, 4]))

    st.header("For Loop with Conditions")

    # Show the raw code
    code = '''
    input 1,2,2,3,4

    for i in range(1, 11):
        if i > 5:
            st.write(i)
        elif i > 2:
            st.write("Between 3 and 5")
        else:
            st.write("Less than 3")
    '''
    st.subheader("Code")
    st.code(code, language='python')

    # Show the output of the loop
    st.subheader("Results")
    for i in range(1, 11):
        if i > 5:
            st.write(i)
        elif i > 2:
            st.write("Between 3 and 5")
        else:
            st.write("Less than 3")


    # --- Section 7: Simulations ---
    st.header("7. Simulations")

    rcode = '''
    x <- runif(10000,0,1)
    hist(x, main='Uniform (0,1) Distribution')
    '''
    st.code(rcode, language='r')

    x = np.random.uniform(0, 1, 10000)
    fig1, ax1 = plt.subplots()
    ax1.hist(x, bins=30)
    ax1.set_title("Uniform (0,1) Distribution")
    st.pyplot(fig1)
    r_code = '''
    y <- rnorm(10000,0,1)
    hist(y, main='Normal (0,1) Distribution')
    '''
    st.code(r_code, language='r')

    y = np.random.normal(0, 1, 10000)
    fig2, ax2 = plt.subplots()
    ax2.hist(y, bins=30)
    ax2.set_title("Normal (0,1) Distribution")
    st.pyplot(fig2)

    # --- Simulated Dice Rolls for P(4) ---
    rcode_dice = '''
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
    '''
    st.subheader("Dice Roll Simulation for P(4)")
    st.code(rcode_dice, language='r')

    n = 5000
    rolls = np.random.randint(1, 7, size=n)
    is_4 = (rolls == 4)
    cum_prob = np.cumsum(is_4) / np.arange(1, n + 1)

    fig3, ax3 = plt.subplots()
    ax3.plot(range(1, n + 1), cum_prob, label='Empirical P(4)')
    ax3.axhline(1/6, color='red', linestyle='--', label='Theoretical P(4)')
    ax3.set_title("Probability of Rolling a 4 Over Time")
    ax3.set_xlabel("Number of Rolls")
    ax3.set_ylabel("Estimated Probability")
    ax3.legend()
    st.pyplot(fig3)

    # --- Section 8: Shiny Equivalent ---
    st.header("8. Shinny App")

    shiny_code = '''
    library(shiny)

    films <- read.csv('films.csv')

    # Define UI for application that draws a histogram
    ui <- fluidPage(

        # Application title
        titlePanel("Films Data"),

        # Sidebar with a slider input for number of bins 
        sidebarLayout(
            sidebarPanel(
                sliderInput("bins",
                            "Number of bins:",
                            min = 1,
                            max = 50,
                            value = 30),
                
            #option to show mean
            checkboxInput("checkbox", label="Display mean", value=FALSE),
            ),

            # Show a plot of the generated distribution
            mainPanel(
            plotOutput("distPlot"),
            hr(),
            fluidRow(column(5, verbatimTextOutput("mean"))),
            
            )
        )
    )

    # Define server logic required to draw a histogram
    server <- function(input, output) {

        output$distPlot <- renderPlot({
            
            # draw the histogram based on input$bins from ui.R
            hist(films$Days, breaks = input$bins, main='Distribution of Days Spent in Theaters',xlab='Days Spent in Theaters',col = 'purple', border = 'darkgrey')
        })
        
        #Display mean if selected
        output$mean <- renderPrint({ 
            if(input$checkbox == TRUE){
                mean(films$Days)
            }
            })
    }

    # Run the application 
    #shinyApp(ui = ui, server = server)
    '''
    st.code(shiny_code, language='r')
    films = pd.DataFrame({
        'Days': np.random.randint(1, 100, 200)
    })

    bins = st.slider("Number of bins:", min_value=1, max_value=50, value=30)
    display_mean = st.checkbox("Display mean")

    fig4, ax4 = plt.subplots()
    ax4.hist(films['Days'], bins=bins, color='purple', edgecolor='grey')
    ax4.set_title("Distribution of Days Spent in Theaters")
    ax4.set_xlabel("Days")

    if display_mean:
        mean_days = films['Days'].mean()
        st.write("Mean Days:", mean_days)

    st.pyplot(fig4)