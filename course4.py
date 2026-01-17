#course4

import streamlit as st
from graphviz import Digraph
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show():
    st.markdown("### Unit 4 Intermediary Statistical Methods")

    st.write("")


    st.markdown("#### 1. Introduction and Basics")

    st.markdown("""
    One of the key goals in statistical analysis is to **understand and describe relationships between variables**.  
    **Regression analysis** is a foundational method used to explore quantitative, predictive relationships between a **response variable** and one or more **predictor variables**. It plays a crucial role across disciplines:

    - Data science and machine learning  
    - Social and natural sciences  
    - Medicine and healthcare  
    - Finance and industry  
    - Public policy and beyond
    """)
    st.markdown(r"""
**Leverage** measures how far an observation's predictor value \( Xi \) is from the mean predictor value in regression.  
It reflects the potential influence of that observation on the fitted regression line.

- High leverage: when \( Xi \) is far from the mean predictor value → the observation can strongly influence the line.
- Low leverage: when \( Xi \) is close to the mean predictor value → little influence on the line.

Leverage depends only on the predictor values \( X \), not on the outcome \( Y \).
""")

    st.write("")
    st.markdown("##### 🔍 Setup: Response and Predictors")


    #st.markdown("###### 🔍 Setup: Response and Predictors")

    st.markdown(r"""
    In this course, we'll work with scenarios where:

    - One variable serves as the **response** (the outcome we're trying to predict)  
    - One or more variables serve as **predictors** (also called covariates, features, or independent variables)

    We denote:
    - Response variable: \( Y \)  
    - Predictor(s): \( X \) (can be a scalar or a vector)

    We observe data:
    """)

    st.latex(r"""
    (X_1, Y_1),\ (X_2, Y_2),\ \ldots,\ (X_n, Y_n)
    """)


    st.latex(r"""
    \text{For } i = 1, \ldots, n \text{ units (e.g., people, cities, patients):}
    """)

    st.latex(r"""
    X_i: \text{ predictor(s) for unit } i
    """)

    st.latex(r"""
    Y_i: \text{ outcome of interest for unit } i
    """)

    st.write("")
    st.markdown("**Law of Iterated Expectations**")
    st.latex(r"E[Y] = E\bigl[E[Y \mid X]\bigr]")
    st.markdown("#### 2. Regression")
    st.markdown("##### 2.1 What is Regression?")

    st.markdown(r"""
    The key object of interest is the **regression function** — the **conditional expectation** of the response given the predictor(s):
    """)

    st.latex(r"""
    x \mapsto \mathbb{E}[Y \mid X = x]
    """)

    st.markdown(r"""
    This function tells us how the *average* value of \( Y \) changes with \( X \).

    """)


    st.markdown(r"""
    We begin with the case of **univariate regression** (one predictor).  
    In real applications, we often deal with **multiple predictors**, possibly thousands.

    In **linear regression**, we assume a linear form of the regression function:
    """)

    st.latex(r"""
    \mathbb{E}[Y \mid X = x] = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p
    """)

    st.latex(r"""
    \text{Where:}
    """)

    st.latex(r"""
    x = (x_1, x_2, \dots, x_p): \text{ a vector of covariates}
    """)

    st.latex(r"""
    \beta_0: \text{ intercept}
    """)

    st.latex(r"""
    \beta_1, \dots, \beta_p: \text{ slope coefficients (regression parameters)}
    """)

    st.markdown(r"""
    For the **univariate case**, we simplify:
    """)



    st.latex(r"""
    \mathbb{E}[Y \mid X = x] = \alpha + \beta x
    """)


    st.latex(r"""
    \text{Here, } \alpha \text{ is the intercept and } \beta \text{ is the slope — this notation is commonly used for the single-variable case.}
    """)
    st.write("")
    st.markdown("##### 2.2 Two Main Goals of Regression")

    st.markdown("##### Prediction")

    st.latex(r"""
    \text{We aim to quantify how well the linear regression model fits the data,}
    """)

    st.latex(r"""
    \text{specifically in terms of how well it predicts a new, unseen data point } (X_{\text{new}}, Y_{\text{new}}).
    """)

    st.latex(r"""
    \text{The model performs well if the predictive mean squared error (PMSE) is small:}
    """)

    st.latex(r"""
    \mathbb{E} \left[ \left( Y_{\text{new}} - \hat{f}_{\boldsymbol{\beta}}(X_{\text{new}}) \right)^2 \right]
    """)




    st.markdown("##### Inference")

    st.markdown(r"""
    We also want to assess the **accuracy of our estimates** for the regression coefficients and construct proper **inference tools**, such as:

    - **Hypothesis tests**  
    - **Confidence intervals**

    These allow us to evaluate the **statistical significance** and **uncertainty** around the regression parameters.
    """)


    st.write("")
    st.markdown("##### 2.3 Regression for Prediction vs Causal Inference")

    st.write(r"""
     Example: Spurious Association

    Suppose we observe that: People who take **multivitamins** have fewer **heart attacks**.

    Should we conclude that multivitamins **prevent** heart attacks?

    > Not necessarily — it could be that multivitamin users also **exercise more** or **smoke less**.  
    > These **confounding variables** can create an association that is not causal.
    """)

    st.markdown(r"""
    Even when we include multiple predictors and "control for" (adjust for) confounders, **observational studies generally only support association**, not causation.
    """)
    st.write("")

    st.markdown(r"""
    - If your **goal is prediction**, causality may be less critical — it's about **accuracy**, not interpretation.
    - If you want to predict the **impact of an intervention**, you're asking a **causal question**, which requires more careful design (often an experiment or advanced causal modeling).

    📌 **Key point**:  
    Regression can make **statistical claims about associations** — but **not causal claims** — unless you are working within a properly designed **experimental framework**.
    """)
    st.write("")
    st.markdown("###### 💹Application: The Capital Asset Pricing Model (CAPM)")


    st.latex(r"""
    \text{The Capital Asset Pricing Model (CAPM) is a widely-used financial model that assesses the profitability of an asset through a simple univariate linear regression.}
    """)

    st.latex(r"""
    \text{It relates the excess return of an individual asset (over the risk-free rate) to the excess return of the market portfolio.}
    """)

    st.latex(r"""
    \text{Let:}
    """)

    st.latex(r"""
    P_t: \text{ price of the asset at time } t
    """)

    st.latex(r"""
    R_t: \text{ return of the asset at time } t, \text{ defined as:}
    """)


    st.latex(r"""
    R_t = \frac{P_t - P_{t-1}}{P_{t-1}}
    """)

    st.markdown(r"""
    Alternatively, we often use the **log-return**, especially for small returns:
    """)

    st.latex(r"""
    \log\left( \frac{P_t}{P_{t-1}} \right)
    """)
    st.latex(r"""
    \text{Let:}
    """)

    st.latex(r"""
    M_t: \text{ return of the market portfolio at time } t
    """)

    st.latex(r"""
    \nu_t: \text{ risk-free rate at time } t \ (\text{e.g., 10Y Treasury bond rate})
    """)

    st.latex(r"""
    \text{The CAPM formula is:}
    """)




    st.latex(r"""
    R_t - \nu_t = \alpha + \beta (M_t - \nu_t) + \varepsilon_t
    """)

    st.latex(r"\text{Where:}")

    st.latex(r"R_t - \nu_t: \text{ excess return of the asset}")

    st.latex(r"M_t - \nu_t: \text{ excess return of the market}")

    st.latex(r"\alpha: \text{ intercept (abnormal return)}")

    st.latex(r"\beta: \text{ sensitivity to the market (market risk exposure)}")

    st.latex(r"\varepsilon_t: \text{ idiosyncratic noise (unexplained return)}")

   
    st.markdown("###### 📌 Interpretation")

    st.latex(r"\beta < 1 \Rightarrow \text{ asset is conservative (less volatile than the market)}")

    st.latex(r"\beta > 1 \Rightarrow \text{ asset is aggressive (more volatile than the market)}")

    st.latex(r"\text{Beyond estimating } \beta, \text{ it's important to:}")

    st.latex(r"\text{- Estimate } \alpha")

    st.latex(r"\text{- Test whether } \alpha = 0")

    st.latex(r"\alpha > 0 \Rightarrow \text{ asset has excess return above the benchmark — highly desirable!}")

    st.latex(r"\text{💼 Hedge funds dedicate significant resources to estimating } \alpha.")

    st.write("")
   

    st.markdown("#### 2.4 MSE")
    st.write("Suppose we want to predict the value of a random variable \( Y \).")

    st.write("Instead of just 'guessing,' we define a **prediction** \( m \), and evaluate its quality using the squared error:")

    st.latex(r"""
    \text{MSE}(m) = \mathbb{E}[(Y - m)^2] \tag{1}
    """)

    st.write("This is called the **mean squared error (MSE)** of the prediction \( m \).")

  


    st.latex(r"""
    \mathbb{E}[(Y - m)^2] = (\mathbb{E}[Y] - m)^2 + \operatorname{Var}(Y) \tag{2}
    """)

    st.write("Thus:")

    st.latex(r"""
    \text{MSE}(m) = (\mathbb{E}[Y] - m)^2 + \operatorname{Var}(Y) \tag{3}
    """)

    st.write("The variance term doesn't depend on \( m \), so to minimize MSE, we only need to minimize the squared bias term.")




    st.write("To minimize MSE with respect to \( m \), we take the derivative and set it equal to zero:")

    st.latex(r"""
    \frac{d}{dm} \text{MSE}(m) = \frac{d}{dm} \left[ (\mathbb{E}[Y] - m)^2 + \operatorname{Var}(Y) \right] \tag{4}
    """)


    st.write("Since Var(Y) is constant with respect to m, we differentiate only the squared term:")

    st.latex(r"""
    (\mathbb{E}[Y] - m)^2
    """)



    st.latex(r"""
    \frac{d}{dm} \left( (\mathbb{E}[Y] - m)^2 \right) = -2 (\mathbb{E}[Y] - m) \tag{5}
    """)




    st.write("Set the derivative equal to zero:")

    st.latex(r"""
    -2(\mathbb{E}[Y] - m) = 0 \quad \Rightarrow \quad m = \mathbb{E}[Y]
    """)

    st.write("So the value of \( m \) that minimizes MSE is:")

    st.latex(r"""
    \boxed{m = \mathbb{E}[Y]}
    """)

    
    # Transition to linear predictor
    st.markdown("---")

    st.write("Unfortunately, mu(x) can be complex or unknown.")

    st.write("To make things simpler, we restrict ourselves to **linear** (or technically, affine) prediction functions:")

    st.latex(r"""
    m(x) = b_0 + b_1 x
    """)

    st.write("We now ask: What are the values of the coefficients that minimize the mean squared error?")


    st.latex(r"""
    \mathbb{E}[(Y - (b_0 + b_1 X))^2]
    """)




    st.write("This is the mean squared error of the linear model — now a function of two variables:")
    st.latex(r"b_0 \quad \text{and} \quad b_1")

    st.write("To proceed, we'll separate what we control (the coefficients) from what we can't (the joint distribution of \( X \) and \( Y \)).")



    st.write("We want to minimize the mean squared error:")

    st.latex(r"""
    \text{MSE}(b_0, b_1) = \mathbb{E}[(Y - (b_0 + b_1 X))^2] \tag{13}
    """)

    st.write("Expanding this expression:")

    st.latex(r"""
    = \mathbb{E}[Y^2] - 2b_0 \mathbb{E}[Y] - 2b_1 \mathbb{E}[X Y] + \mathbb{E}[(b_0 + b_1 X)^2] \tag{14}
    """)

    st.latex(r"""
    = \mathbb{E}[Y^2] - 2b_0 \mathbb{E}[Y] - 2b_1 \left( \operatorname{Cov}(X, Y) + \mathbb{E}[X]\mathbb{E}[Y] \right) 
    + b_0^2 + 2b_0 b_1 \mathbb{E}[X] + b_1^2 \mathbb{E}[X^2] \tag{15}
    """)

    st.latex(r"""
    = \mathbb{E}[Y^2] - 2b_0 \mathbb{E}[Y] - 2b_1 \operatorname{Cov}(X, Y) - 2b_1 \mathbb{E}[X]\mathbb{E}[Y] 
    + b_0^2 + 2b_0 b_1 \mathbb{E}[X] + b_1^2 \operatorname{Var}(X) + b_1^2 (\mathbb{E}[X])^2 \tag{16}
    """)

    # 2) As a LaTeX block
    st.latex(
    r"\frac{\partial}{\partial b_0}\sum_{i=1}^n\bigl(y_i - b_0 - b_1 x_i\bigr)^2 \;=\; 0"
)



    st.latex(
    r"\frac{\partial}{\partial b_1}\sum_{i=1}^n\bigl(y_i - b_0 - b_1 x_i\bigr)^2 \;=\; 0"
)




    st.latex(r"""
    \frac{\partial \text{MSE}}{\partial b_0} = -2 \mathbb{E}[Y] + 2b_0 + 2b_1 \mathbb{E}[X] \tag{17}
    """)

    

    st.latex(r"""
    \frac{\partial \text{MSE}}{\partial b_1} = 
    -2 \operatorname{Cov}(X, Y) - 2 \mathbb{E}[X] \mathbb{E}[Y] + 2b_0 \mathbb{E}[X] 
    + 2b_1 \operatorname{Var}(X) + 2b_1 (\mathbb{E}[X])^2 \tag{18}
    """)




    st.markdown(
    r"Set the derivatives to zero and solve for $b_0$, $b_1$. Let the optimal values be $\beta_0$ and $\beta_1$."
)


    st.write("From the first equation (17):")

    st.latex(r"""
    0 = -2 \mathbb{E}[Y] + 2\beta_0 + 2\beta_1 \mathbb{E}[X]
    """)

    st.latex(r"""
    \Rightarrow \quad \beta_0 = \mathbb{E}[Y] - \beta_1 \mathbb{E}[X] \tag{19}
    """)

    st.write("This tells us the best-fit line passes through the mean point:")
    st.latex(r"(\mathbb{E}[X], \mathbb{E}[Y])")

    st.write("If the variables were centered so that the means are both zero:")
    st.latex(r"\mathbb{E}[X] = \mathbb{E}[Y] = 0")
    st.write("then we would get:")
    st.latex(r"\beta_0 = 0")

    st.write("Now substitute equation (19) into equation (18) to solve for:")
    st.latex(r"\beta_1")


    st.write("After substituting the expression for the intercept into the second derivative condition, we get:")

    st.latex(r"""
    \beta_0 = \mathbb{E}[Y] - \beta_1 \mathbb{E}[X]
    """)


    st.latex(r"""
    0 = -\operatorname{Cov}(X, Y) - \mathbb{E}[X] \mathbb{E}[Y] 
    + \beta_0 \mathbb{E}[X] + \beta_1 \operatorname{Var}(X) + \beta_1 (\mathbb{E}[X])^2 \tag{20}
    """)

    st.write("Now substitute in the following expression for the intercept:")

    st.latex(r"\beta_0 = \mathbb{E}[Y] - \beta_1 \mathbb{E}[X]")

    st.latex(r"""
    = -\operatorname{Cov}(X, Y) - \mathbb{E}[X] \mathbb{E}[Y] 
    + (\mathbb{E}[Y] - \beta_1 \mathbb{E}[X]) \mathbb{E}[X] + \beta_1 \operatorname{Var}(X) + \beta_1 (\mathbb{E}[X])^2 \tag{21}
    """)

    st.write("Simplifying:")

    st.latex(r"""
    = -\operatorname{Cov}(X, Y) + \beta_1 \operatorname{Var}(X) \tag{22}
    """)



    st.write("Solving for the slope:")


    st.latex(r"""
    \beta_1 = \frac{\operatorname{Cov}(X, Y)}{\operatorname{Var}(X)} \tag{23}
    """)


    st.write("")
    st.markdown("#### 2.5 OLS")

    st.latex(r"""
    \hat{\beta}_1 = \beta_1 + \frac{1}{\sum_{i=1}^n (X_i - \bar{X})^2} \sum_{i=1}^n (X_i - \bar{X}) \varepsilon_i
    """)
    st.latex(r"""
\hat{\beta}_0 \;=\;\frac{\sum_{i=1}^n y_i \;-\;\hat{\beta}_1 \sum_{i=1}^n x_i}{n}
""")

    st.markdown("Apply the variance of a linear combination of i.i.d. random variables:")
    st.latex(r"""
\mathrm{Var}(\hat{\beta}_0) \;=\; \sigma^2 \left(\frac{1}{n} \;+\; \frac{\bar x^2}{\sum_{i=1}^n (x_i - \bar x)^2}\right)
""")
    st.latex(r"""
    \text{Var}(\hat{\beta}_1 \mid X^n) 
    = \text{Var} \left( \frac{1}{\sum (X_i - \bar{X})^2} \sum (X_i - \bar{X}) \varepsilon_i \right)
    """)

    st.latex(r"""
    = \frac{1}{\left[\sum (X_i - \bar{X})^2\right]^2} \cdot \text{Var}\left( \sum (X_i - \bar{X}) \varepsilon_i \right)
    """)

    st.latex(r"""
    = \frac{1}{\left[\sum (X_i - \bar{X})^2\right]^2} \cdot \sum (X_i - \bar{X})^2 \cdot \sigma^2
    """)

    st.latex(r"""
    = \frac{\sigma^2}{\sum (X_i - \bar{X})^2}
    """)

    st.latex(r"""
    \text{se}(\hat{\beta}_1 \mid X^n) = \sqrt{ \text{Var}(\hat{\beta}_1 \mid X^n) } 
    = \sqrt{ \frac{\sigma^2}{\sum_{i=1}^n (X_i - \bar{X})^2} } 
    = \frac{\sigma}{\sqrt{\text{SXX}}}
    """)

    st.latex(r"""
    \text{Alternative form:} \quad 
    \text{SXX} = n \cdot \text{Var}(\tilde{X}) 
    \Rightarrow 
    \text{se}(\hat{\beta}_1 \mid X^n) = \frac{\sigma}{\sqrt{n \cdot \text{Var}(\tilde{X})}}
    """)

    st.write("")
    st.markdown("#### 2.6 Law of Total Variance")

    st.latex(r"""
    \text{Law of Total Variance:}
    \quad \text{Var}(\hat{\beta}_1) = \mathbb{E}[\text{Var}(\hat{\beta}_1 \mid X^n)] 
    + \text{Var}(\mathbb{E}[\hat{\beta}_1 \mid X^n])
    """)

    st.latex(r"""
    \mathbb{E}[\hat{\beta}_1] = \mathbb{E}[\mathbb{E}[\hat{\beta}_1 \mid X^n]] = \mathbb{E}[\beta_1] = \beta_1
    """)

    st.latex(r"""
    \text{So } \hat{\beta}_1 \text{ is unconditionally unbiased.}
    """)


    st.latex(r"""
    \text{Since } \mathbb{E}[\hat{\beta}_1 \mid X^n] = \beta_1 \text{ (a constant), the second term is } 0
    """)

    st.latex(r"""
    \text{And } \text{Var}(\hat{\beta}_1 \mid X^n) = \frac{\sigma^2}{n \cdot \text{Var}(\tilde{X})}
    """)

    st.latex(r"""
    \boxed{
    \text{Var}(\hat{\beta}_1) = \frac{\sigma^2}{n} \cdot \mathbb{E} \left[ \frac{1}{\text{Var}(\tilde{X})} \right]
    }
    """)
    st.write("")
    st.write("")
    st.write("#### 2.7 Linear Function of the Response Variables")
    st.write("##### Fitted Value")

    st.latex(r"""
    \hat{Y}_i = \hat{\beta}_0 + \hat{\beta}_1 X_i
    """)

    st.latex(r"""
    = \bar{Y} + \hat{\beta}_1 (X_i - \bar{X})
    """)

    st.latex(r"""
    = \frac{1}{n} \sum_{j=1}^n Y_j + \left( \sum_{j=1}^n (X_j - \bar{X}) Y_j \right) \cdot \frac{X_i - \bar{X}}{\sum_{j=1}^n (X_j - \bar{X})^2}
    """)

    st.latex(r"""
    = \sum_{j=1}^n Y_j \left( \frac{1}{n} + \frac{(X_j - \bar{X})(X_i - \bar{X})}{\sum_{j=1}^n (X_j - \bar{X})^2} \right)
    """)

    st.latex(r"""
    = \sum_{j=1}^n h_{ij} Y_j
    """)

    st.write("where the hat matrix entries are:")

    st.latex(r"""
    h_{ij} = h_{ji} = \frac{1}{n} + \frac{(X_j - \bar{X})(X_i - \bar{X})}{\sum_{j=1}^n (X_j - \bar{X})^2}
    """)

    st.write("##### Variance and Covariance of Fitted Values")

    st.latex(r"""
    \text{Var}(\hat{Y}_i \mid X^n) = \sigma^2 h_{ii}
    """)

    st.latex(r"""
    \text{Cov}(\hat{Y}_i, \hat{Y}_j \mid X^n) = \sigma^2 h_{ij}
    """)

    st.write("These follow from the fact that:")

    st.latex(r"""
    \hat{Y} = HY, \quad \text{and} \quad \text{Var}(Y \mid X) = \sigma^2 I
    """)

    st.latex(r"""
    \Rightarrow \text{Var}(\hat{Y} \mid X) = \text{Var}(HY) = \sigma^2 H
    """)



    st.write("##### Comparing Residuals vs. Noise in Simple Linear Regression")

    st.latex(r"""
    Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i
    \quad \Rightarrow \quad
    \varepsilon_i = Y_i - (\beta_0 + \beta_1 X_i)
    """)

    st.latex(r"""
    \hat{\varepsilon}_i = Y_i - (\hat{\beta}_0 + \hat{\beta}_1 X_i)
    """)

    st.latex(r"""
    \hat{\varepsilon}_i = \varepsilon_i + (\beta_0 - \hat{\beta}_0) + (\beta_1 - \hat{\beta}_1) X_i
    """)

    st.write("So residuals are close to the noise terms, but not exactly equal — they include estimation error.")

    st.write("##### Residuals as Weighted Sums of the Noise")

    st.latex(r"""
    \hat{Y}_i = \sum_{j=1}^n h_{ij} Y_j \quad \Rightarrow \quad e_i = Y_i - \hat{Y}_i
    """)

    st.latex(r"""
    e_i = \sum_{j=1}^n (\delta_{ij} - h_{ij}) Y_j
    = \sum_{j=1}^n (\delta_{ij} - h_{ij}) (\beta_0 + \beta_1 X_j + \varepsilon_j)
    """)



    st.latex(r"""
    e_i = \sum_{j=1}^n (\delta_{ij} - h_{ij}) \varepsilon_j
    """)

    st.write("✅ Each residual is a linear combination of the noise terms.")


    st.write("##### Variance and Covariance of Residuals using Kronecker Delta")

    st.write("Let the Kronecker delta be defined as:")

    st.latex(r"""
    \delta_{ij} =
    \begin{cases}
    1 & \text{if } i = j \\
    0 & \text{if } i \ne j
    \end{cases}
    """)

    st.write("Then the variance and covariance of residuals in linear regression are:")

    st.latex(r"""
    \text{Var}(e_i \mid X^n) = \sigma^2 (1 - h_{ii})
    """)

    st.latex(r"""
    \text{Cov}(e_i, e_j \mid X^n) = \sigma^2 (\delta_{ij} - h_{ij})
    """)

    st.write("This follows from the fact that the covariance matrix of residuals is:")

    st.latex(r"""
    \text{Var}(e \mid X^n) = \sigma^2 (I - H)
    """)
    st.latex(r"""
    \begin{aligned}
    &\text{In simple linear regression:} \\
    &\quad Y_i = \beta_0 + \beta_1 X_i + \varepsilon_i, \quad \varepsilon_i \sim \mathcal{N}(0, \sigma^2) \\
    &\text{We estimate 2 parameters: } \beta_0 \text{ and } \beta_1 \Rightarrow \text{Degrees of freedom} = n - 2 \\
    &\text{Then the expected residual sum of squares is:} \\
    &\quad \mathbb{E} \left( \sum_{i=1}^n \varepsilon_i^2 \mid X^n \right) = (n - 2)\sigma^2
    \end{aligned}
    """)

    st.latex(r"""
    \begin{aligned}
    \sigma^2 &= \operatorname{Var}[Y_i \mid X^n] \\
            &= \operatorname{Var}[\hat{Y}_i \mid X^n] + \operatorname{Var}[\varepsilon_i \mid X^n] \\
            &= \sigma^2 h_{ii} + \sigma^2(1 - h_{ii}) \\
            &= \sigma^2
    \end{aligned}
    """)

    st.latex(r"""
    \begin{aligned}
    \text{Under linearity and constant variance:} \quad & \\
    \mathbb{E}[\hat{\beta}_0 \mid X^n] &= \beta_0, \quad
    \operatorname{Var}[\hat{\beta}_0 \mid X^n] = \sigma^2 \left( \frac{1}{n} + \frac{\bar{X}^2}{\sum_{i=1}^n (X_i - \bar{X})^2} \right) \\
    \mathbb{E}[\hat{\beta}_1 \mid X^n] &= \beta_1, \quad
    \operatorname{Var}[\hat{\beta}_1 \mid X^n] = \frac{\sigma^2}{\sum_{i=1}^n (X_i - \bar{X})^2} \\
    \operatorname{Cov}[\hat{\beta}_0, \hat{\beta}_1 \mid X^n] &= -\sigma^2 \cdot \frac{\bar{X}}{\sum_{i=1}^n (X_i - \bar{X})^2} \\
    \mathbb{E}[\hat{m}(x) \mid X^n] &= \beta_0 + \beta_1 x = m(x), \\
    \operatorname{Var}[\hat{m}(x) \mid X^n] &= \sigma^2 \left( \frac{1}{n} + \frac{(x - \bar{X})^2}{\sum_{i=1}^n (X_i - \bar{X})^2} \right) \\
    \\
    \text{With Gaussian errors:} \quad & \\
    \hat{\beta}_0 &\sim \mathcal{N} \left( \beta_0, \sigma^2 \left( \frac{1}{n} + \frac{\bar{X}^2}{\sum_{i=1}^n (X_i - \bar{X})^2} \right) \right) \\
    \hat{\beta}_1 &\sim \mathcal{N} \left( \beta_1, \frac{\sigma^2}{\sum_{i=1}^n (X_i - \bar{X})^2} \right) \\
    \hat{m}(x) &\sim \mathcal{N} \left( \beta_0 + \beta_1 x, \sigma^2 \left( \frac{1}{n} + \frac{(x - \bar{X})^2}{\sum_{i=1}^n (X_i - \bar{X})^2} \right) \right)
    \end{aligned}
    """)
    st.write("---")
    st.markdown("#### Summary: Properties on Residuals")



    st.write("1. **Zero Conditional Mean:**")
    st.latex(r"\mathbb{E}[\varepsilon_i \mid X = x_i] = 0")
    st.latex(r"\frac{1}{n} \sum_{i=1}^n e_i = 0")

    st.write("2. **Homoskedasticity:**")
    st.latex(r"\operatorname{Var}[\varepsilon_i \mid X = x_i] = \sigma^2 \quad \text{(independent of } x_i \text{)}")

    st.write("3. **(Near) Uncorrelated Residuals:**")
    st.write("Residuals may not be perfectly uncorrelated, but correlations should weaken as:")
    st.latex(r"n \to \infty")


    st.write("4. **No Pattern in Residual Plots:**")
    st.write("Residual plots vs. fitted values or covariates should show no systematic pattern.")

    st.write("5. **If the noise if Gaussian, the residuals should also be Gaussian:**")

    st.write("Residuals should be approximately normally distributed as:")
    st.latex(r"e_i \sim \mathcal{N}(0, \sigma^2)")

    st.markdown( "#### 2.8 Simple Linear Regression Model Assumptions")

    st.latex(r"""
    \textbf{1. IID Sample:} \quad \{(X_1, Y_1), \dots, (X_n, Y_n)\} \overset{iid}{\sim} P
    """)

    st.latex(r"""
    \textbf{2. Linearity:} \quad Y = \beta_0 + \beta_1 X + \varepsilon
    """)

    st.latex(r"""
    \textbf{3. Zero-mean error:} \quad \mathbb{E}[\varepsilon \mid X] = 0
    """)

    st.latex(r"""
    \textbf{4. Constant variance (Homoskedasticity):} \quad \text{Var}(\varepsilon \mid X) = \sigma^2
    """)
    st.write("")
    st.write("")
    st.markdown("#### 2.9 Gaussian-Noise Simple Linear Regression Model Assumptions")

    st.latex(r"""
    \textbf{1. IID Sample:} \quad \{(X_1, Y_1), \dots, (X_n, Y_n)\} \overset{iid}{\sim} P
    """)

    st.latex(r"""
    \textbf{2. Linearity:} \quad Y = \beta_0 + \beta_1 X + \varepsilon
    """)

    st.latex(r"""
    \textbf{3. Zero Mean and Constant Variance:} \quad \mathbb{E}[\varepsilon \mid X] = 0, \quad \text{Var}(\varepsilon \mid X) = \sigma^2
    """)

    st.latex(r"""
    \textbf{4. Gaussian Noise:} \quad \varepsilon \mid X = x \sim \mathcal{N}(0, \sigma^2)
    """)


   
   

    st.markdown(" #### 2.10 Coefficient of Determination: R squared")
    st.latex(r"""
R^2 =\frac{1}{n} \sum_{i=1}^n \left( Y_i - \bar{Y} \right)^2 = 
\frac{1}{n} \sum_{i=1}^n \left( \hat{\beta}_1 (X_i - \bar{X}) + \hat{\varepsilon}_i \right)^2
""")
    st.latex(r"""
= \frac{1}{n} \sum_{i=1}^{n} \left( \hat{\beta}_1 (X_i - \bar{X}) + \hat{\varepsilon}_i \right)^2
""")

    st.latex(r"""
= \hat{\beta}_1^2 \cdot \frac{1}{n} \sum_{i=1}^n (X_i - \bar{X})^2 
+ \frac{1}{n} \sum_{i=1}^n \hat{\varepsilon}_i^2
""")
    st.write("also:")
    st.latex(r"""
R^2 = \frac{\hat{\beta}_1^2 \sum_{i=1}^n (X_i - \bar{X})^2}{\sum_{i=1}^n (Y_i - \bar{Y})^2}
""")
    
    st.write("")
    st.markdown("#### 2.11 Confidence Intervals vs. Prediction Intervals")
    st.markdown("##### Confidence Intervals")
    st.markdown("""
It is common for people to **misinterpret confidence intervals (CIs)** by saying that,  
after data has been collected and the confidence interval has been constructed,  
the probability that the true parameter value lies within the interval is 95%.

This is **not quite correct** because:
- Once the data have been collected and the CI has been constructed,  
the **endpoints are no longer random**—we have seen them and we know their exact values.
- At that point, the true parameter is either **inside** the interval or **not inside**.  
There is no remaining randomness.
""")
    st.markdown("""
The **correct interpretation** is:
- We are **95% confident** that the true value is contained within the interval.  
- This means that **over repeated sampling and repeated construction of confidence intervals**,  
the procedure will capture the true parameter in **95% of cases**.
""")
    st.write(" A **confidence interval (CI)** aims to cover a **fixed but unknown population parameter** (such as the true mean mu or slope).")
    st.latex(r"""
\mathbb{P}(\theta \in \text{CI}) = 0.95
""")
    st.markdown("""
This probability refers to the **long-run success rate of the method**,  
not the probability that a specific already-computed interval contains the true value.
""")
    st.markdown(r"""
Recall we showed in previous sections that:
""")

    st.latex(r"""
\hat{\beta}_1 \mid X_n \sim N\left( \beta_1, \frac{\sigma^2}{n \, \operatorname{Var}(X)} \right)
""")

    st.latex(r"""
\hat{m}(x) \mid X_n \sim N\left( m(x), \frac{\sigma^2}{n} \left( 1 + \frac{(x - \bar{X})^2}{\operatorname{Var}(X)} \right) \right)
""")

    st.markdown(r"""
These results can equivalently be written as:
""")

    st.latex(r"""
\frac{\hat{\beta}_1 - \beta_1}{\sigma / \sqrt{n \, \operatorname{Var}(X)}} \mid X_n \sim N(0, 1)
""")

    st.latex(r"""
\frac{\hat{m}(x) - m(x)}{\left( \frac{\sigma}{\sqrt{n}} \right) \sqrt{1 + \frac{(x - \bar{X})^2}{\operatorname{Var}(X)}} } \mid X_n \sim N(0, 1)
""")

   

    st.latex(r"""
\text{Also note that if the conditional distribution of a random variable } T \text{ given } X_n \text{ is Gaussian and does not depend on } X_n, \text{ then the marginal distribution is also Gaussian, since:}
""")


    st.latex(r"""
P(T \leq t) = E \left[ P(T \leq t \mid X_n) \right] = E \left[ \Phi(t) \right] = \Phi(t)
""")

    st.latex(r"""
\text{The first equality follows from the \textbf{law of iterated expectation}, and the last since } \Phi(t) \, (\text{the Gaussian CDF evaluated at } t) \text{ is not random.}
""")
    st.latex(r"""
\text{Also note that if the conditional distribution of a random variable } T \text{ given } X_n \text{ is Gaussian and does not depend on } X_n, \text{ then the marginal distribution is also Gaussian, since:}
""")

    st.latex(r"""
    P(T \leq t) = E\left[ P(T \leq t \mid X_n) \right] = E\left[ \Phi(t) \right] = \Phi(t)
    """)

    st.latex(r"""
    \text{where the first equality follows by the law of iterated expectation, and the last since } \Phi(t) \text{ (the Gaussian CDF evaluated at } t \text{) is not random.}
    """)

    st.latex(r"""
    \text{Therefore, we also get confidence intervals for free, based on our result from the previous section.}
    """)

    st.latex(r"""
    \text{In particular, 95\% confidence intervals for } \beta_1 \text{ and } m(x) \text{ are given by:}
    """)

    st.latex(r"""
    \hat{\beta}_1 \pm 1.96 \times \frac{\sigma}{\sqrt{n \, \operatorname{Var}(X)}}
    """)

    st.latex(r"""
    \hat{m}(x) \pm 1.96 \times \frac{\sigma}{\sqrt{n}} \sqrt{1 + \frac{(x - \bar{X})^2}{\operatorname{Var}(X)}}
    """)

    st.latex(r"""
    \text{and in general, to construct } 100(1 - \alpha)\% \text{ confidence intervals, we would replace 1.96 with } z_{1 - \alpha/2}.
    """)

    st.latex(r"""
\frac{\hat{\beta}_1 - \beta_1}{\dfrac{\sigma}{\sqrt{n \, \operatorname{Var}(X)}}} \sim N(0, 1)
""")

    st.latex(r"""
\frac{\hat{m}(x) - m(x)}{\left( \dfrac{\sigma}{\sqrt{n}} \right) \sqrt{1 + \dfrac{(x - \bar{X})^2}{\operatorname{Var}(X)}}} \sim N(0, 1)
""")

    st.write("---")
    st.markdown("##### Prediction Intervals")
    st.write("A **prediction interval** aims to cover a **random but observable future value** of the outcome.")
    st.latex(r"""
\text{In other words, the prediction interval covers the future value we are trying to predict with 95\% probability,} \\
\text{or more generally, with probability } 1 - a.
""")


    st.latex(r"""
\mathbb{P}(\ell \leq \beta \leq u) = \mathbb{P}\left( \hat{\beta} - 1.96 \tau \leq \beta \leq \hat{\beta} + 1.96 \tau \right)
""")

    st.latex(r"""
= \mathbb{P}\left( -1.96 \leq \frac{\beta - \hat{\beta}}{\tau} \leq 1.96 \right)
""")

    st.latex(r"""
= \mathbb{P}\left( -1.96 \leq \frac{\hat{\beta} - \beta}{\tau} \leq 1.96 \right)
""")

    st.latex(r"""
= \mathbb{P}\left( -1.96 \leq N(0, 1) \leq 1.96 \right) = 95\%
""")

    st.latex(r"""
\text{Often, for the estimators we are considering, we have approximately:} \quad
\tau \approx \frac{\sigma}{\sqrt{n}}
""")

    st.latex(r"""
\text{In other words, the } \textbf{prediction interval} \text{ covers the outcome we are trying to predict with } 95\% \text{ probability} \left( \text{or more generally, with probability } 1 - \alpha \right).
""")
    st.write("---")
    st.markdown("#### 2.12 Hypothesis testing")
    st.write("##### Ingredients of a test")
    st.latex(r"""
\text{Choose a test statistic } T \text{ such that } T \text{ tends to be large when } H_0 \text{ is false.}
""")

    st.latex(r"""
\text{Find a critical value } c = c_\alpha \text{ such that:}
""")

    st.latex(r"""
\mathbb{P}(T \geq c_\alpha \mid H_0) \leq \alpha
""")

    st.latex(r"""
\text{Reject } H_0 \text{ if } T \geq c_\alpha
""")

    st.write("##### P values")
    st.latex(r"""
\text{The } p\text{-value is the smallest } \alpha \text{ level at which we would reject the null hypothesis.}
""")

    st.latex(r"""
p = \min \{ \alpha : T \geq c_\alpha \}
""")

    st.latex(r"""
\text{Smaller } p\text{-values indicate stronger evidence against the null hypothesis,} 
""")

    st.latex(r"""
\text{since we would have rejected even at a smaller Type I error level.}
""")
    st.latex(r"""
\text{The } p\text{-value is also the probability, under } H_0, \text{ of observing a test statistic as large or larger than the observed value.}
""")

    st.latex(r"""
\text{Let } t_{\text{obs}} \text{ denote the observed value of the test statistic. Then:}
""")

    st.latex(r"""
p = \min \left\{ \alpha : t_{\text{obs}} \geq c_\alpha \right\}
""")

    st.latex(r"""
= \min \left\{ \mathbb{P}(T \geq c_\alpha \mid H_0) : c_\alpha \leq t_{\text{obs}} \right\}
""")

    st.latex(r"""
= \mathbb{P}(T \geq t_{\text{obs}} \mid H_0)
            
""")
    st.markdown(r"""
- The **p-value** is the probability, **assuming the null hypothesis \( H0 \) is true**, of obtaining a test statistic **as extreme or more extreme** than the one observed.
""")



    st.markdown(r"""
- A **small p-value** (e.g., less than 0.05) suggests that the observed data is **unlikely under \( H0 \)**, providing **evidence against the null hypothesis**.
- A **large p-value** (e.g., greater than 0.05) suggests that the observed data is **plausible under \( H0 \)**, so we **fail to reject the null hypothesis**.
""")
    st.write("---")
    st.latex(r"""
\textbf{Duality Between Confidence Intervals and Hypothesis Tests}
""")

    st.latex(r"""
\text{Let } [\ell, u] \text{ be any valid confidence interval for a parameter } \beta,
""")

    st.latex(r"""
\text{with coverage level } 1 - \alpha \text{ such that:}
""")

    st.latex(r"""
\mathbb{P}(\ell \leq \beta \leq u) = 1 - \alpha
""")

    st.latex(r"""
\text{To test the null hypothesis } H_0: \beta = t,
""")

    st.latex(r"""
\text{Reject } H_0 \text{ if } t \notin [\ell, u]
""")

    st.latex(r"""
\text{The test is valid because:}
""")

    st.latex(r"""
\mathbb{P}(\text{Reject } H_0 \mid H_0 \text{ true}) = \mathbb{P}(t \notin [\ell, u] \mid \beta = t)
""")

    st.latex(r"""
= \mathbb{P}(\beta \notin [\ell, u]) = 1 - \mathbb{P}(\beta \in [\ell, u]) = 1 - (1 - \alpha) = \alpha
""")

    st.latex(r"""
\text{Thus: once we have a valid confidence interval, we automatically get a valid hypothesis test.}
""")

    st.latex(r"""
\text{Conversely: once we have a valid hypothesis test, we can invert it to create a valid confidence interval.}
""")

    st.write("---")

    st.markdown("##### Statistical Significance")
    st.latex(r"""
\text{If we test the hypothesis } H_0: \beta_1 = b \text{ and reject it,}
""")

    st.latex(r"""
\text{it is common to say that the difference between } \beta_1 \text{ and } b \text{ is statistically significant.}
""")
    st.write("---")
    st.markdown("#### Matrix")
    #st.markdown("###### **Expectations and Variances with Vectors and Matrices**")

    st.markdown("###### **Expectations and Variances with Vectors and Matrices**")

    st.markdown("""
    If we have $p$ random variables, $Z_1, Z_2, \\dots, Z_p$, we can put them into a random vector:
    """)

    st.latex(r"""
    \mathbf{Z} =
    \begin{bmatrix}
    Z_1 \\
    Z_2 \\
    \vdots \\
    Z_p
    \end{bmatrix}
    """)

    st.markdown("""
    This random vector can be thought of as a $p \\times 1$ matrix of random variables.

    The expected value of $\\mathbf{Z}$ is defined to be the vector:
    """)

    st.latex(r"""
    \boldsymbol{\mu} \equiv \mathbb{E}[\mathbf{Z}] =
    \begin{bmatrix}
    \mathbb{E}[Z_1] \\
    \mathbb{E}[Z_2] \\
    \vdots \\
    \mathbb{E}[Z_p]
    \end{bmatrix}
    \tag{1}""")

    st.markdown("If $a$ and $b$ are non-random scalars, then:")

    st.latex(r"""
    \mathbb{E}[a\mathbf{Z} + b\mathbf{W}] = a\mathbb{E}[\mathbf{Z}] + b\mathbb{E}[\mathbf{W}]
    \tag{2}
    """)

    st.markdown("If $\\mathbf{a}$ is a non-random vector, then:")

    st.latex(r"""
    \mathbb{E}[\mathbf{a}^T \mathbf{Z}] = \mathbf{a}^T \mathbb{E}[\mathbf{Z}]
    """)

    st.markdown("If $\\mathbf{A}$ is a non-random matrix, then:")

    st.latex(r"""
    \mathbb{E}[\mathbf{A}\mathbf{Z}] = \mathbf{A} \mathbb{E}[\mathbf{Z}]
    """)
    st.markdown("###### **Variance of a Random Vector**")

    st.markdown("""
    The variance of a random vector $\\mathbf{Z}$ is defined as:
    """)

    st.latex(r"""
    \text{Var}[\mathbf{Z}] = \mathbb{E}[\mathbf{Z} \mathbf{Z}^T] - \mathbb{E}[\mathbf{Z}] \, \mathbb{E}[\mathbf{Z}]^T
    \tag{5}
    """)

    st.markdown("""
    For a non-random vector $\\mathbf{a}$ and a non-random scalar $b$:
    """)

    st.latex(r"""
    \text{Var}[\mathbf{a} + b\mathbf{Z}] = b^2 \, \text{Var}[\mathbf{Z}]
    """)
    st.markdown("""
For a non-random matrix $\\mathbf{C}$:
""")

    st.latex(r"""
\text{Var}[\mathbf{C}\mathbf{Z}] = \mathbf{C} \, \text{Var}[\mathbf{Z}] \, \mathbf{C}^T
""")
    st.markdown("###### **Trace and Quadratic Forms**")

    st.markdown("""
    If $\\mathbf{A}$ is a square matrix, then the **trace** of $\\mathbf{A}$ — denoted by $\\text{tr}(\\mathbf{A})$ — is defined to be the **sum of the diagonal elements**. In other words:
    """)

    st.latex(r"""
    \text{tr}(\mathbf{A}) = \sum_j A_{jj}
    """)

    st.markdown("The trace satisfies the following properties:")

    st.latex(r"""
    \text{tr}(\mathbf{A} + \mathbf{B}) = \text{tr}(\mathbf{A}) + \text{tr}(\mathbf{B})
    """)

    st.latex(r"""
    \text{tr}(c\mathbf{A}) = c \, \text{tr}(\mathbf{A})
    """)

    st.latex(r"""
    \text{tr}(\mathbf{A}^T) = \text{tr}(\mathbf{A})
    """)

    st.markdown("It also has the **cyclic property** for matrix products:")

    st.latex(r"""
    \text{tr}(\mathbf{A}\mathbf{B}\mathbf{C}) = \text{tr}(\mathbf{B}\mathbf{C}\mathbf{A}) = \text{tr}(\mathbf{C}\mathbf{A}\mathbf{B})
    """)

    st.markdown("""
    If $\\mathbf{C}$ is a **non-random matrix**, then $\\mathbf{Z}^T \\mathbf{C} \\mathbf{Z}$ is called a **quadratic form**. Its expected value is:
    """)

    st.latex(r"""
    \mathbb{E}[\mathbf{Z}^T \mathbf{C} \mathbf{Z}] = \mathbb{E}[\mathbf{Z}]^T \mathbf{C} \mathbb{E}[\mathbf{Z}] + \text{tr}(\mathbf{C} \, \text{Var}[\mathbf{Z}])
    \tag{8}
    """)

    st.markdown("To see this, recall that:")

    st.latex(r"""
    \mathbf{Z}^T \mathbf{C} \mathbf{Z} = \text{tr}(\mathbf{Z}^T \mathbf{C} \mathbf{Z})
    """)
    st.markdown("###### **Mean Squared Error (MSE)**")

    st.markdown("""
    Let the residual vector be defined as:
    """)

    st.latex(r"""
    \mathbf{e} \equiv \mathbf{e}(\boldsymbol{\beta}) = \mathbf{Y} - \mathbf{X} \boldsymbol{\beta}
    \tag{21}
    """)

    st.markdown("The training error (or observed mean squared error) is given by:")

    st.latex(r"""
    \text{MSE}(\boldsymbol{\beta}) = \frac{1}{n} \sum_{i=1}^n e_i^2(\boldsymbol{\beta}) = \frac{1}{n} \, \mathbf{e}^T \mathbf{e}
    \tag{22}
    """)

    st.markdown("Expanding this expression, we get:")

    st.latex(r"""
    \text{MSE}(\boldsymbol{\beta}) = \frac{1}{n} \, \mathbf{e}^T \mathbf{e}
    \tag{23}
    """)

    st.latex(r"""
    = \frac{1}{n} (\mathbf{Y} - \mathbf{X} \boldsymbol{\beta})^T (\mathbf{Y} - \mathbf{X} \boldsymbol{\beta})
    \tag{24}
    """)

    st.latex(r"""
    = \frac{1}{n} (\mathbf{Y}^T - \boldsymbol{\beta}^T \mathbf{X}^T)(\mathbf{Y} - \mathbf{X} \boldsymbol{\beta})
    \tag{25}
    """)

    st.latex(r"""
    = \frac{1}{n} \left(
    \mathbf{Y}^T \mathbf{Y} - \mathbf{Y}^T \mathbf{X} \boldsymbol{\beta}
    - \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{Y}
    + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
    \right)
    \tag{26}
    """)

    st.latex(r"""
    = \frac{1}{n} \left(
    \mathbf{Y}^T \mathbf{Y}
    - 2 \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{Y}
    + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
    \right)
    """)
    st.markdown("###### **Minimizing the Mean Squared Error (MSE)**")

    st.markdown("We start by computing the gradient of the in-sample MSE with respect to $\\boldsymbol{\\beta}$:")

    st.latex(r"""
    \nabla \text{MSE}(\boldsymbol{\beta}) = \frac{1}{n} \left(
    \nabla \mathbf{Y}^T \mathbf{Y}
    - 2 \nabla \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{Y}
    + \nabla \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{X} \boldsymbol{\beta}
    \right)
    \tag{28}
    """)

    st.latex(r"""
    = \frac{1}{n} \left( 0 - 2 \mathbf{X}^T \mathbf{Y} + 2 \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} \right)
    \tag{29}
    """)

    st.latex(r"""
    = \frac{2}{n} \left( \mathbf{X}^T \mathbf{X} \boldsymbol{\beta} - \mathbf{X}^T \mathbf{Y} \right)
    \tag{30}
    """)

    st.markdown("Setting this gradient equal to zero at the optimum $\\hat{\\boldsymbol{\\beta}}$ gives:")

    st.latex(r"""
    \mathbf{X}^T \mathbf{X} \hat{\boldsymbol{\beta}} = \mathbf{X}^T \mathbf{Y}
    \tag{31}
    """)

    st.markdown("This is called the **normal equation** or **estimating equation**. Solving it gives:")

    st.latex(r"""
    \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
    \tag{32}
    """)

    st.markdown("""
    This single matrix equation yields both coefficient estimates in simple linear regression.

    If this is correct, it should reproduce the standard closed-form expressions for $\\hat{\\beta}_0$ and $\\hat{\\beta}_1$:
    """)

    st.latex(r"""
    \hat{\beta}_1 = \frac{\text{Cov}(X, Y)}{\text{Var}(X)}, \quad
    \hat{\beta}_0 = \bar{Y} - \hat{\beta}_1 \bar{X}
    \tag{33}
    """)

    st.markdown("To make this connection clearer, we normalize both matrix products by $\\frac{1}{n}$:")

    st.latex(r"""
    \hat{\boldsymbol{\beta}} = \left( \frac{1}{n} \mathbf{X}^T \mathbf{X} \right)^{-1} \left( \frac{1}{n} \mathbf{X}^T \mathbf{Y} \right)
    """)

    st.markdown("##### **Expected Value and Variance of Residuals in Linear Regression**")

    st.markdown("Let residuals be defined as:")
    st.latex(r"\mathbf{e} = \mathbf{Y} - \hat{\mathbf{Y}} = \mathbf{Y} - \mathbf{H} \mathbf{Y} = (\mathbf{I} - \mathbf{H}) \mathbf{Y}")

    st.latex(r" \mathbf{Y} = \mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon} , \text{ so we substitute:}")

    st.latex(r"\mathbf{e} = (\mathbf{I} - \mathbf{H})(\mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon})")

    st.markdown("Distribute the matrix product:")
    st.latex(r"""
    \mathbb{E}[\mathbf{e} \mid \mathbf{X}]
    = (\mathbf{I} - \mathbf{H}) \mathbb{E}[\mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon} \mid \mathbf{X}]
    = (\mathbf{I} - \mathbf{H}) (\mathbf{X} \boldsymbol{\beta} + \mathbb{E}[\boldsymbol{\varepsilon} \mid \mathbf{X}])
    """)

    st.latex(r"\text{Assuming the errors have zero conditional mean, } \mathbb{E}[\boldsymbol{\varepsilon} \mid \mathbf{X}] = 0")

    st.latex(r"""
    \mathbb{E}[\mathbf{e} \mid \mathbf{X}]
    = (\mathbf{I} - \mathbf{H}) \mathbf{X} \boldsymbol{\beta}
    = \mathbf{X} \boldsymbol{\beta} - \mathbf{H} \mathbf{X} \boldsymbol{\beta}
    = \mathbf{X} \boldsymbol{\beta} - \mathbf{X} \boldsymbol{\beta}
    = 0
    \tag{50}
    """)

    st.markdown("### Variance of the residuals")

    st.markdown("Now compute the variance-covariance matrix of the residuals:")

    st.latex(r"""
    \text{Var}(\mathbf{e} \mid \mathbf{X}) =
    \text{Var}((\mathbf{I} - \mathbf{H})(\mathbf{X} \boldsymbol{\beta} + \boldsymbol{\varepsilon}) \mid \mathbf{X})
    = \text{Var}((\mathbf{I} - \mathbf{H}) \boldsymbol{\varepsilon} \mid \mathbf{X})
    \tag{52}
    """)

    st.latex(r"\text{Since } \boldsymbol{\varepsilon} \mid \mathbf{X} \sim \mathcal{N}(0, \sigma^2 \mathbf{I}) \text{:}")


    st.latex(r"""
    \text{Var}(\mathbf{e} \mid \mathbf{X}) =
    (\mathbf{I} - \mathbf{H}) \, \text{Var}(\boldsymbol{\varepsilon} \mid \mathbf{X}) \, (\mathbf{I} - \mathbf{H})^T
    = \sigma^2 (\mathbf{I} - \mathbf{H})(\mathbf{I} - \mathbf{H})^T
    \tag{54}
    """)

    st.latex(r"\text{Because the hat matrix } \mathbf{H} \text{ is symmetric and idempotent:}")

    st.latex(r"""
\mathbf{H} = \mathbf{H}^T \\
\mathbf{H}^2 = \mathbf{H} \\
(\mathbf{I} - \mathbf{H})^T = \mathbf{I} - \mathbf{H}
""")

    st.latex(r"""
    \Rightarrow \text{Var}(\mathbf{e} \mid \mathbf{X}) = \sigma^2 (\mathbf{I} - \mathbf{H})
    \tag{Final Result}
    """)
