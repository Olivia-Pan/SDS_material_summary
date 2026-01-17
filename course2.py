#course2
import streamlit as st
from graphviz import Digraph
import requests
from bs4 import BeautifulSoup
import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

def show():
    st.markdown("### Unit 2: Statistical Thinking")

    st.markdown("#### 1. Studies")
    st.markdown("""
    data set = data frame(s) + code book(s)<br>\
    sample = a specific selection of cases from the population <br>\
    The unit of analysis = the type of entity you choose to focus on\
    bias: a systematic discrepancy between the population and a sample
    """, unsafe_allow_html=True)


    from streamlit_mermaid import st_mermaid

    st.markdown("##### 1a. Sampling Methods")
    mermaid_code = '''
    %%{init: {'themeVariables': {'fontSize': '20px'}}}%%
    flowchart LR
    A6[ ] --> B6["Simple Random Sampling:<br>Every member of the population has an equal chance of getting selected"]
    B6 --> C6["But you do NOT have access to the full registry"]
    A6 --> D6["Stratified Random Sampling:<br>Respondents are split into sub groups and<br>then randomly selected from each group"]
    A6 --> E6["Multi-Stage Sampling:<br>several sampling methods across different levels, applied in stages"]
    A6 --> F6["Cluster Sampling:<br>one divides the population into different clusters and then randomly selects some clusters (and all the units in those clusters)"]
    A6 --> G6["Systematic Sampling:<br>select every kth unit from a list"]
    A6 --> H6["Convenience Sampling:<br>one selects units that are convenient to access, like people walking on the street"]
    A6 --> I6["Snowball Sampling:<br>one selects individuals that are convenient to access, those individuals then recruit people they can access, and so on"]
    A6 --> J6["Quota Sampling:<br>one selects individuals (non-randomly) from some particular sections of the population that are of special interest in the study"]
    A6 --> K6["Judgement Sampling:<br>one selects individuals based on one’s judgment"]
    '''

    st_mermaid(mermaid_code)

    st.markdown("##### 1b. Types of Biases")
    mermaid_code2 = """
    %%{init: {'themeVariables': {'fontSize': '15px'}}}%%
    flowchart LR
    A7[ ] --> B7["Sampling bias:<br>Due to the faulty sampling scheme and analyses, e.g., failing to correct for
    the imbalance when selecting units with equal probabilities"]
    A7 --> C7["Non-Response Bias:<br> People not responding to questions"]
    A7 --> D7("Response Bias:<br> People giving incorrect/distorted/less truthful responses due to
    psychological, social reasons")
    A7 -->F7["Recall Bias:Compared to people not affected by a condition, people with the
    condition may be more likely to recall or exaggerate a certain risk
    factor, even if it did not exist, because of a temptation to tell
    themselves a coherent story about their own misfortune."]
    A7 --> E7("Biases arising from erroneous study designs:<br>1. Not properly defining the research
    problem and/or the target population,<br>2. Not properly framing the survey questions, etc.")
    """
    st_mermaid(mermaid_code2)
    st.code("""
        x<-c(1,2,3,4,2,3,4,2)
    sample(x, 3, replace=FALSE, prob=NULL)
    #prob = NULL means can define the weights of each
    """, language="r")

    st.markdown("##### 1c. Experiments, Observational Studies, Cohort Studies")
    mermaid_code3 = '''
    %%{init: {'themeVariables': {'fontSize': '35px'}}}%%
    flowchart LR
    R8["Quasi/Natural Experiment"] --> Z8["designs that are like an experiment, but without an actual experimental intervention"]
    BB8["Experiment"] --> FF8["Placebo Effect"]
    BB8 --> GG8["Blocking"]
    GG8 --> HH8["reduces random imbalances between<br> treatment and control groups"]
    C8["Observational Studies"] --> J8[Matching]
    GG8 --> ZZ8["Blocking vs Matching:<br>Matching is similar in spirit to<br> blocking.\
        But matching happens in the analysis phase, in the absence of randomization. We already know who's been<br>\
            treated and who hasn’t, and this treatment decision was out of our control. Blocking happens in the design\
            phase, prior to randomization.<br>When we create blocks, we haven’t yet decided who gets the treatment and who gets the control"]
    J8 -->ZZ8
    J8 -->K8["Pros<br>It’s transparent. Even if there are lots of details about<br>how matching is actually done, pretty<br>much anyone can understand the idea."]
    J8 --> YY8["Cons<br>You can only match on what you can observe.
    <br>Rejoinder: with unobserved confounders, <br>nothing works except randomization"]
    W8["Repeated Measures"] --> D8["Cohort Studies"]
    D8 --> M8["Defined group followed over time"]
    D8 --> N8["Prospective vs. Retrospective"]
    D8 --> P8["Pros & Cons:<br>Multiple outcomes,<br>Rare exposures,<br>Long, costly,<br>Confounding"]
    X8["One-Time Comparison"] --> S8["Cross-Sectional Studies"]
    S8 --> T8["Compare groups at one point in time"]
    S8 --> V8["Descriptive vs. Analytical"]
    S8 --> U8["Pros:<br>Quick, broad data<br>Cons:<br>No causality,<br>Recall bias,<br>Interpretation issues"]
    A9["Experiment (Randomized Controlled Trials)"] --> B9["Quasi-Experiments<br>with Randomization"]
    B9 --> C9["Prospective Cohort Studies"]
    C9 --> D9["Retrospective Cohort &<br>Case-Control Studies"]
    '''

    st_mermaid(mermaid_code3)




    st.write("<br>Experiment: Use a control group, block what you can, randomize what you cannot<br><br>\
    Placebo effect: any effect produced by a treatment that cannot be attributed to the properties<br>\
    of the treatment itself, and must therefore be due to the subject’s belief in the effect of that treatment.<br><br>\
        Blocking: In the context of an experiment, the best-case scenario is to randomize subjects to treatment and <br>\
        control within a bunch of near-twin pairs<br><br>\
            Observational Studies: merely observe the effect of some treatment or risk factor, without having the ability to change who is or\
            isn’t exposed to it<br><br>Matching: For every treated case, find control cases that are as similar as possible. \
            Discard cases without a good match.<br><br>Cohort Studies: similar in concept to an experiment, but involves no randomization.\
            Example: You start with a cohort of people who share some particular defining characteristic. You follow those people over time. \
            During this follow-up, some of the cohort will be exposed to a specific risk factor (pollution, sugary foods, etc.), and some won’t.\
            These form your treatment and control groups, respectively \
    <br>- Descriptive: Assess the prevalence of a condition <br>vs. <br> - Analytical: Evaluate the association of an outcome with other characteristics \
            of the population (i.e., percentage of Texans with lung cancer who also smoked)", unsafe_allow_html=True)

    st.write("A diff-in-diff design shares some things in common with RD design: <br>- It’s used in situations where a policy affects one group, but not the other.\
            There’s a “running variable” that determines the treatment, almost always time. <br>- A diff-in-diff design also shares something in common with a block\
            design: Each observational unit is a “block” that serves as its own control. Here, the control is in a “before and after” sense, as we’ll see.", unsafe_allow_html=True)

    st.write("")
    st.markdown("#### 2. Statistical Uncertainty")

    mermaid_code4 = '''
    %%{init: {'themeVariables': {'fontSize': '25px'}}}%%
    flowchart LR
    A10[difference] --> B10["Estimand :<br>some feature of the population of interest<br>(e.g. proportion of Democratic voters among all U.S. voters).<br>Often called a parameter."]
    A10 --> C10["Sample estimate :<br>that same feature, but of a sample<br>rather than the whole population"]
    '''

    st_mermaid(mermaid_code4)

    st.markdown("##### 2a. Bootstrap")

    st.write("In most situations, we can’t repeatedly sample from the population and track how much <br>\
            the answer changes from one sample to the next.", unsafe_allow_html=True)

    mermaid_code5 = '''
    %%{init: {'themeVariables': {'fontSize': '25px'}}}%%
    flowchart LR
    A11[Bootstrap] --> C11["Bootstrap Sample: A sample from the sample with replacement"]
    A11 -->D11["Core assumption of the bootstrap:<br>The randomness in your data"]
    '''

    st_mermaid(mermaid_code5)

    st.write("Bootstrap Definition: repeatedly sample (with replacement) from the sample itself, and we track how much the estimate changes from one such sample to the next.<br>\
            Coverage Principle: X% confidence interval for every estimate you made, those intervals should cover their corresponding values at least X% of the time. <br> <br>Sample code for bootstrapping:", unsafe_allow_html=True)

    st.code("""
    # Data Distribution
    ggplot(NHANES_sleep) + geom_histogram(aes(x = SleepHrsNight), binwidth=1)
    mean(~SleepHrsNight, data=NHANES_sleep)
    # Bootstrap Samples
    # one sample
    NHANES_sleep_bootstrap = mosaic::resample(NHANES_sleep)
    mean(~SleepHrsNight, data=NHANES_sleep_bootstrap)
    # many samples
    rep_B <- 10000
    boot_sleep <- do(rep_B)*mean(~SleepHrsNight, data=mosaic::resample(NHANES_sleep))
    head(boot_sleep)

    # Bootstrap Distribution
    gg_boot_sleep <- ggplot(boot_sleep) + geom_histogram(aes(x=mean),bins=30)
    gg_boot_sleep
    sd(boot_sleep$mean)

    # Approx 68% Confidence Interval
    lwr <- mean(boot_sleep$mean) - sd(boot_sleep$mean)
    upr <- mean(boot_sleep$mean) + sd(boot_sleep$mean)
    cbind(lwr,upr)
    gg_boot_sleep + geom_vline(xintercept=c(lwr,upr))
    sum(boot_sleep$mean > lwr & boot_sleep$mean < upr)/rep_B

    """, language="r")

    st.markdown("##### 2b. Hypothesis Testing")

    mermaid_code5 = '''
    %%{init: {'themeVariables': {'fontSize': '20px'}}}%%
    flowchart LR
    A11["Null Hypothesis (H₀)"] --> B11["Test Statistic (T)<br>Summarizes data<br>Measures evidence"]
    B11 --> C11["Sampling Distribution<br>P(T | H₀)"]
    C11 --> D11["Assess if H₀ is believable<br>Or indicates an anomaly"]
    
    E11["p-value:<br>Probability of seeing T or<br>more extreme if H₀ is true"] --> D11
    F11["Smaller p-value<br>→less likely under H₀"] --> D11
    G11["Interpretation:<br>Probability of observed or<br>more extreme result<br>given H₀ is true"] --> D11
    H11["Test Statistic:<br>Numeric summary for assessing H₀"] --> B11
    '''
    st_mermaid(mermaid_code5)

    st.write("Test statistic:")
    st.latex(r"\chi^2 = \sum_{i=1}^{k} \frac{(O_i - E_i)^2}{E_i}")

    st.write("Error Types:")
    mermaid_code6='''
    %%{init: {'themeVariables': {'fontSize': '15px'}}}%%
    flowchart TB
    A12["Error Types"] --> B12["Type I Error:<br>H₀ is true,<br>but we reject it<br>(False Positive)"]
    A12 --> C12["Type II Error:<br>H₀ is false,<br>but we fail to reject it<br>(False Negative)"]
    '''
    st_mermaid(mermaid_code6)

    st.write('''
    Permutation Tests: <br>- A statistical hypothesis \
    test motivated by the proof of contradiction <br>-\
    Suppose that we have a treatment and control group, \
    and we want to test whether the treatment affects the\
    outcome of interest. We can compare the average response \
    variable across groups to obtain our test statistic. Then, \
    we randomly re-assign thbe responses to the treatment and control\
    groups to construct the approximate test statistic distribution.
    ''', unsafe_allow_html=True)

    st.image("t-tes.png", caption="Test Statistic Formula")

    st.write('''
            P-hacking is the misuse of data analysis to find patterns in \
            data that can be presented as statistically significant, thus \
            dramatically increasing and understating the risk of false \
            positives.<br><br>
    ''', unsafe_allow_html=True)

    st.markdown("##### 2c. Goodness of Fit")

    st.write('''
            A goodness of fit test is a statistical hypothesis test used\
            to determine whether a variable is likely to come from a \
            specified distribution or not.<br><br>
    ''', unsafe_allow_html=True)

    st.markdown("##### 2d. Large Sample Inference")
    st.write("Do Moivre Equation:")
    st.latex(r"\text{SE}(\bar{x}) = \frac{\sigma}{\sqrt{n}}")

    st.write("Where:")
    st.latex(r"""
    \begin{aligned}
    & \bar{x} = \text{sample mean} \\
    & \sigma = \text{population standard deviation} \\
    & n = \text{sample size}
    \end{aligned}
    """)


    st.write("**Central Limit Theorem**")

    st.write("""
    - Suppose we take a sample of size N from some wider population, and compute the sample average.
    - Let μ be the population mean, and σ be the population standard deviation.
    - If N is sufficiently large (e.g., ≥ 30), then the sampling distribution of the sample mean can be approximated by:
    """)

    st.latex(r"\bar{x} \sim \mathcal{N} \left( \mu,\ \frac{\sigma}{\sqrt{N}} \right)")


    st.write("**Where:**")


    st.latex(r"""
    \begin{aligned}
    & \bar{x}: \ \text{sample mean} \\
    & \mu: \ \text{population mean} \\
    & \sigma: \ \text{population standard deviation} \\
    & N: \ \text{sample size}
    \end{aligned}
    """)

    st.latex(r"\bar{X}_N \sim \mathcal{N} \left( \mu, \frac{\sigma}{\sqrt{N}} \right)")

    st.write("N: sample size <br>bar x: sample mean <br>σ: use the sample standard deviation\
    <br><br> CLT FOR PROPORTIONS <br>- Let be the proportion of an event A in a population. \
        <br>- Let be the proportion of the event A in a random sample of size N from \
        this population. <br>-CLT: If N is sufficiently large, then the statistical\
            fluctuations in from one sample to the other can be well approximated by\
            a normal distribution with mean and standard deviation", unsafe_allow_html=True)

    st.latex(r"\hat{p}_n \sim \mathcal{N} \left( p_0,\ \frac{p_0 (1 - p_0)}{n} \right)")
    st.latex(r"\text{SD}(\hat{p}) = \sqrt{ \frac{p_0(1 - p_0)}{n} }")

    st.write("**Definitions:**")
    st.latex(r"""
    \begin{aligned}
    & n: \ \text{sample size} \\
    & \bar{x}: \ \text{sample mean} \\
    & \sigma: \ \text{population standard deviation} \\
    & \hat{p}: \ \text{sample proportion} \\
    & p_0: \ \text{population proportion (under the null hypothesis)} \\[1em]
    & \text{For proportions, the standard deviation (standard error) of } \hat{p} \text{ is:} \\
    & \text{SD}(\hat{p}) = \sqrt{ \frac{p_0(1 - p_0)}{n} }
    \end{aligned}
    """)

    st.write("As N gets larger, the sampling distribution of the mean\
    starts to look: <br>More normal, narrower<br> Standard Error is \
    the standard deviation of a sampling distribution.", unsafe_allow_html=True)


    st.markdown("**CLT-based Intervals:**")
    st.latex(r"\hat{\theta} \pm z^* \cdot \text{SE}(\hat{\theta})")

    st.image("dis.png", caption="Test Statistic Formula", use_container_width=True)

    st.markdown("##### 2e. Regression")

    st.write("Linear Regression Model:")
    st.latex(r"y_i = \beta_0 + \beta_1 x_i + \varepsilon_i")

    st.write("1. RMSE")
    st.latex(r"\text{RMSE} = \sqrt{ \frac{1}{N - 1} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2 } = \sqrt{ \frac{1}{N - 1} \sum_{i=1}^{N} \hat{e}_i^2 }")
    st.write("**Where:**")
    st.latex(r"\hat{y}_i = \beta_0 + \beta_1 x_i:\ \text{predicted value}")
    st.latex(r"\hat{e}_i = y_i - \hat{y}_i:\ \text{residual for observation } i")
    st.latex(r"N:\ \text{number of observations}")

    st.write("Small RMSE:<br>more variation in y is systematic (predictable by x)<br>vs.<br>Large RMSE:<br>\
            more variation in y is individual (not predictable by x)", unsafe_allow_html=True)

    st.write("2. R-squared")

    st.write("R-squared answers the question: <br>what fraction of variation in y is predictable in terms of x? It’s always between 0 and 1. <br>0 means no relationship: all variation in y is individual. <br>1 means that y and x are perfectly related: all variation in y is predictable\
    <br>Ex: Here, about 51% of the variation is predictable, and 49% unpredictable. So ( R\^2 ) ≈ 0.51", unsafe_allow_html=True)

    st.latex(r"""
    \begin{aligned}
    & R^2 \text{ closer to } 1: \text{ more variation in } y \text{ is systematic (predictable by } x\text{)} \\
    & R^2 \text{ closer to } 0: \text{ more variation in } y \text{ is individual (not predictable by } x\text{)} \\
    & R^2 \text{ measures strength of association, not causation}
    \end{aligned}
    """)

    st.write("Statistical significance just means whether the confidence interval for some estimate contains zero—nothing more, nothing less.")
    st.write("----------------------------------------------------")

    st.write("Baseline: <br>pick a baseline and compare", unsafe_allow_html=True)

    st.write("2 Major advantages of baseline/offset form:<br> 1. Convenience. Expressing numbers in baseline/offset form puts the focus on differences between situations, which is often what we can about. <br>2. Generalizability. We can write down equations involving multiple variables, and everything hangs together easily.<br><br>", unsafe_allow_html=True)

    st.markdown("##### 2f. Regression with Interactions")

    st.write("Definition:<br>- We use the term interaction in statistical modeling to describe situations where the effect of\
            some feature x on the outcome y is context-specific.<br> <br>Do we expect the effect of some predictor (x) on the response (y) to be basically the\
    same, regardless of context? ->Main effect only\
    <br>Is that predictor’s effect on y contingent on some other variable? -> Interaction.", unsafe_allow_html=True)

    st.write("---------------------------------------")

    st.write("ANOVA:<br>is a way of partitioning the variability in the outcome variable into different categorical predictor sources", unsafe_allow_html=True)

    st.write("Total Variation = Between + Within Variation")

    st.write("-   **Total Sum of Squares (SST):**")

    st.latex(r"\text{SST} = \sum_{k=1}^{K} \sum_{i=1}^{n_k} \left( Y_{k,i} - \bar{Y} \right)^2")
    st.write("-   **Between-Group Sum of Squares (SSB):**")
    st.latex(r"\text{SSB} = \sum_{k=1}^{K} \sum_{i=1}^{n_k} \left( \bar{Y}_k - \bar{Y} \right)^2")

    st.write("-  **Mean Square Between (MSB):**")

    st.latex(r"\text{MSB} = \frac{\text{SSB}}{K - 1}")

    st.write("- **Within-Group Sum of Squares (SSW) or SSE:**")

    st.latex(r"\text{SSE} = \sum_{k=1}^{K} \sum_{i=1}^{n_k} \left( Y_{k,i} - \bar{Y}_k \right)^2")

    st.write("ANOVA uses **Mean Square Error (MSE):**")

    st.latex(r"\text{MSE} = \frac{\text{SSE}}{N - K}")

    st.write("Sample code in R:")

    st.code("""
    games_model4 = lm(PictureTarget.RT ~ FarAway + Littered + Subject +
                    FarAway:Littered, data = rxntime)
    eta_squared(games_model4, partial = FALSE)
    """, language="r")

    st.latex(r"F = \frac{\text{MSB}}{\text{MSE}} \sim F_{K - 1,\ N - K}")
    st.write("F score statistic used in ANOVA is to test whether a group of variables significantly\
            explains variability in the response variable: ratio of explained variance to unexplained\
            variance")
    st.write("**P score means how likely youa are to get your observed F-score\
            or more extreme just by chance.**<br><br>", unsafe_allow_html=True)

    st.markdown("##### 2g. Multiple Regression")

    st.write("Partial relationship: how changes in x predict changes \
            in y, holding other variables constant (i.e., conditioning\
            on fixed values of the other variables). <br>Overall relationship: \
            how changes in x predict changes in y, letting the other variables \
            change as they might.<br><br>Correlated predictors lead to causal confusion.\
    <br>An interaction means “it depends.” You can’t summarize the x-y relationship in a single number,\
            because the strength of that relationship depends on other factors.<br><br>", unsafe_allow_html=True)
