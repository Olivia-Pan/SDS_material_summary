#
# SDS 313 Shiny App Example - Hello Shiny!
#

library(shiny)

# Define user interface
ui <- fluidPage(

    # App title
    titlePanel('Hello Shiny!'),

    # Sidebar and main panels 
    sidebarLayout(
        sidebarPanel('This is a sidebar panel'),
        mainPanel(
            h2('This is a main panel', align='center'),
            p(strong('bolded text')),
            div('We can style text here', style='color:purple')
        )
        
    )
)

# Define server function (blank for now)
server <- function(input, output) {
}

# Run the app
shinyApp(ui = ui, server = server)
