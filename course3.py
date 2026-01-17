#course3
import streamlit as st
from graphviz import Digraph
import requests
from bs4 import BeautifulSoup
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def show():
    st.markdown("### Unit 3 Probability/Statistical Inference")
    st.markdown("#### 1. Basic")
    st.write("**Definition:**")
    st.write("Outcomes: $\\omega_1, \\omega_2, \\ldots, \\omega_n$")
    st.write("Sample space: the set of all possible outcomes $\\Omega$")
    st.write("Event: a subset of $\\Omega$, e.g., $A \\subseteq \\Omega$")



    st.write("<br>**Example:**", unsafe_allow_html=True)
    st.write("Pick a number at random from 1 to 100:")
    st.write(" Outcomes: $1, 2, \\ldots, 100$")
    st.write(" Sample space: $\\{1, 2, \\ldots, 100\\}$")
    st.write(" Set notation (unordered): $\\{a, b\\} = \\{b, a\\}$")
    st.write(" Event $A$: <br>number drawn has 1 digit $\\Rightarrow \\{1, 2, \\ldots, 9\\}$", unsafe_allow_html=True)
    st.write(" $|A| = 9$")



    st.write("Roll a 6-sided die twice:")
    st.write(" Outcomes: $(a, b) \\ne (b, a)$ (order matters)")
    st.write(" $|\\Omega| = 36$")
    st.write("")




    st.latex(r"""
    \begin{aligned}
    &\text{The union of two events } A \text{ and } B \text{ is the event } C \text{ that either } A \text{ occurs or } B \text{ occurs or both occur:} \\
    &\quad A \cup B \\[1em]

    &\text{The intersection of two events } C = A \cap B \text{ is the event that both } A \text{ and } B \text{ occur.} \\[1em]

    &\text{The complement of an event } A^c \text{ is the event that } A \text{ does not occur:} \\
    &\quad (A \cup C)^c = A^c \cap C^c \\[1em]

    &\text{The empty set is the set with no elements:} \\
    &\quad A \cap C = \emptyset \Rightarrow \text{A and C are disjoint.} \\[1em]

    &\text{If } A \subset B, \text{ then if A happens, B must also happen.} \\[1em]

    &\textbf{Commutative Laws:} \\
    &\quad A \cup B = B \cup A \\
    &\quad A \cap B = B \cap A \\[1em]

    &\textbf{Associative Laws:} \\
    &\quad (A \cup B) \cup C = A \cup (B \cup C) \\
    &\quad (A \cap B) \cap C = A \cap (B \cap C) \\[1em]

    &\textbf{Distributive Laws:} \\
    &\quad (A \cup B) \cap C = (A \cap C) \cup (B \cap C) \\
    &\quad (A \cap B) \cup C = (A \cup C) \cap (B \cup C)
    \end{aligned}
    """)

    st.image("samp.png", caption="", use_container_width=True)

    st.markdown(r"""
    <div style="text-align: left">

    $$\textbf{Probability Measure Axioms:}$$

    1. $$ P(\Omega) = 1 \quad \text{(Normalization)} $$  
    2. $$ P(A) \geq 0 \quad \text{for any } A \subseteq \Omega \quad \text{(Non-negativity)} $$  
    3. $$ P(A_1 \cup A_2) = P(A_1) + P(A_2) \quad \text{if } A_1 \cap A_2 = \emptyset \quad \text{(Additivity)} $$

    $$ \textbf{Addition Law of Probability:} \quad P(A \cup B) = P(A) + P(B) - P(A \cap B) $$

    $$ \textbf{Multiplication Principle: } \text{If one experiment has } m \text{ outcomes and another has } n, \text{ there are } mn \text{ outcomes.} $$

    $$ \binom{12}{3} = \frac{12!}{3!(12 - 3)!} = \frac{12!}{3! \cdot 9!} = \frac{12 \times 11 \times 10}{6} $$

    $$ \binom{11}{9} = \binom{11}{2} $$

    $$ (a + b)^n = \sum_{k=0}^{n} \binom{n}{k} a^{n-k} b^k \quad \textbf{(Binomial Coefficients)} $$

    $$ \binom{n}{n_1, n_2, \ldots, n_r} = \frac{n!}{n_1! \cdot n_2! \cdots n_r!} \quad \textbf{(Multinomial Coefficient)} $$

    $$ \binom{7}{3, 2, 2} = \frac{7!}{3! \times 2! \times 2!} $$

    $$ \binom{n - k}{r - m} \quad \text{ways to choose } r - m \text{ non-defectives} $$

    $$ p = \frac{\binom{k}{m} \binom{n - k}{r - m}}{\binom{n}{r}} $$

    $$ \textbf{Conditional Probability:} \quad P(A \mid B) = \frac{P(A \cap B)}{P(B)}, \quad \text{if } P(B) > 0 $$

    $$ \textbf{Independence: } P(A \mid B) = P(A),\quad P(B \mid A) = P(B),\quad P(A \cap B) = P(A) \cdot P(B) $$

                
    $$ \textbf{Multiplication Law: } P(A \cap B) = P(A \mid B) \cdot P(B) $$

    $$ \textbf{Law of Total Probability: } P(A) = P(A \mid B) \cdot P(B) + P(A \mid B^c) \cdot P(B^c) $$

    $$ \textbf{Bayes' Rule: } P(B \mid A) = \frac{P(A \mid B) \cdot P(B)}{P(A)} $$

    </div>
    """, unsafe_allow_html=True)

    st.write("Example:<br>about 10% employyes are thieves, lie detectors are 80% accurate<br>\
            meaning P(+|thief)=0.8, P(-|honest)=0.8", unsafe_allow_html=True)

    st.latex(r"P(\text{thief} \mid +) = \frac{P(+ \mid \text{thief}) \cdot P(\text{thief})}{P(+ \mid \text{thief}) \cdot P(\text{thief}) + P(+ \mid \text{honest}) \cdot P(\text{honest})}=\frac{0.8 \cdot 0.1}{0.8 \cdot 0.1 + 0.2 \cdot 0.9}")

    st.markdown("""
    ---  
    ##### 2a. Discrete vs Continuous Random Variable:
    """, unsafe_allow_html=True)

    st.write("random variable that can take on only a finite or at most a countably infinite \
            number of values<br>cdf: F(x)=P(X<=x)<br>pmf: p(x)=P(X=x)<br><br>\
            Indicator random variable(r.v.): let A be a set: ", unsafe_allow_html=True)

    st.latex(r"f(x) = \begin{cases} P(A) & \text{if } x = 1 \\ 1 - P(A) & \text{if } x = 0 \\ 0 & \text{otherwise} \end{cases}")

    st.latex(r"f(x) = p^x (1 - p)^{1 - x}, \quad x \in \{0, 1\}")

    st.markdown("##### For Discrete random variables:")
    st.write("PMF: $f(x) = P(X = x)$: gives probability at exact points")
    st.write("$\\mathcal{X} = \\{x_1, x_2, x_3, \\dots\\}$")
    st.write("$F(x) = P(X \\le x) = \\sum_{t \\le x} f(t)$")
    st.write("$f(x) = P(X = x) = F(x) - F(x^-)$")
    st.write("$F(x^-)$ means everything before $f(x)$")
    st.write("$P(\\Omega) = \\sum_{x \\in \\mathcal{X}} f(x) = 1$")
    st.write("")

    import pandas as pd

    # Title
    st.markdown("##### Joint PMF and Marginal PMF")

    st.markdown("""
    **Joint PMF (Probability Mass Function):**  
    The joint PMF of two discrete random variables \( X \) and \( Y \) gives the probability that both occur together.
    """)
    st.latex(r"P(X = x, Y = y)")

    st.markdown("""
    **Marginal PMFs:**  
    These are obtained by summing over the joint PMF.
    """)

    st.latex(r"""
    \begin{aligned}
    P(X = x) &= \sum_y P(X = x, Y = y) \\
    P(Y = y) &= \sum_x P(X = x, Y = y)
    \end{aligned}
    """)




    # Section: Example Table
    st.markdown("###### Example: Joint PMF Table")

    data = {
        "Y=1": [0.1, 0.2],
        "Y=2": [0.15, 0.25],
        "Y=3": [0.05, 0.25],
    }
    joint_pmf = pd.DataFrame(data, index=["X=1", "X=2"])
    joint_pmf.index.name = "X \\ Y"

    # Display joint PMF table
    st.dataframe(joint_pmf.style.format("{:.2f}"))

    # Marginal PMFs


    # Marginal P(X)
    st.markdown("###### Marginal PMF of X")
    marginal_x = joint_pmf.sum(axis=1)
    st.dataframe(marginal_x.to_frame(name="P(X=x)").style.format("{:.2f}"))
    st.latex(r"P(X = x) = \sum_y P(X = x, Y = y)")

    # Marginal P(Y)
    st.markdown("###### Marginal PMF of Y")
    marginal_y = joint_pmf.sum(axis=0)
    st.dataframe(marginal_y.to_frame(name="P(Y=y)").style.format("{:.2f}"))
    st.latex(r"P(Y = y) = \sum_x P(X = x, Y = y)")



    st.markdown("##### For Continuous random variables:")
    st.write("$f(x)$ is the probability density function (PDF) which is relative density around a point, not a probability itself.")
    st.write("$f(x) \\ge 0$ for all $x$")

    st.write("$P(a \\le X \\le b) = \\int_a^b f(x)\\,dx$")
    st.write("$F(x) = P(X \\le x) = \\int_{-\\infty}^{x} f(t)\\,dt$")
    st.write("$f(x) = \\frac{d}{dx} F(x)$")

    st.write("$P(X = x) = 0$, because the area under a single point is zero")
    st.markdown("###### Definition for real pdf:")


    st.latex(r"""
    f_X(x) \geq 0 \quad \text{(non-negative)}
    """)

    st.latex(r"""
    \iint_{\mathbb{R}^2} f_{X,Y}(x, y) \, dx \, dy = 1
    """)

    st.markdown("###### Reminder:")
    st.markdown("Unlike discrete probabilities, for a continuous variable, the probability at a single point is zero:")
    st.latex(r"""
    P(X = x) = 0
    """)
    st.markdown("To get **1-dimensional marginal distribution** of a  X , we integrate over the other two variables:")

    st.latex(r"""
    f_X(x) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} f_{X,Y,Z}(x, y, z) \, dy \, dz
    """)

    st.markdown("To get a **2-dimensional marginal distribution**, say for X, Y, from a joint distribution over \( X, Y, Z \), we integrate out the third variable \( Z \):")

    st.latex(r"""
    f_{X,Y}(x, y) = \int_{-\infty}^{\infty} f_{X,Y,Z}(x, y, z) \, dz
    """)
    st.image("fig.jpg", caption="Variable Types", use_container_width=True)
    st.latex(r"""
    \text{Conditional} = \frac{\text{Joint}}{\text{Marginal}}
    """)


    st.markdown("""-------------------""")
    st.markdown("##### 2b. Binomial Distribution")
    st.write("""<br>The **Binomial distribution** models\
            n trials, n is fixed;<br>\
            Each trial can be considered "success" or "failure";<br>\
            Trials are independent<br>\
            Each Trial has the same probability p of success

    It answers questions like:  
    *“What is the probability of getting exactly 3 heads in 5 coin tosses?”*


    ##### Parameters:
    - \( n \): number of trials  
    - \( p \): probability of success in a single trial  
    - \( X \): number of successes out of \( n \) trials, ~Binom(n which stands for number of trials,p=win probability)

    """, unsafe_allow_html=True)

    st.latex(r"P(X = x) = \binom{n}{x} p^x (1 - p)^{n - x}, \quad x = 0, 1, 2, \dots, n")

    st.write("**Mean:** $\\mu = np$")
    st.write("**Variance:** $\\sigma^2 = np(1 - p)$")


    st.markdown("""
    This distribution is widely used in quality control, clinical trials, and modeling binary outcomes.
    """)

    st.latex(r"P(X = x) = \binom{n}{x} p^x (1 - p)^{n - x}, \quad x = 0, 1, 2, \dots, n")

    st.write("In R: _binom()<br>\
            d=density (pmf)<br>p=cdf<br>r=random number generator<br>q=quantile function<br><br>", unsafe_allow_html=True)

    st.markdown("""
    **Special case 1**: The **Poisson distribution** models the number of times an event occurs in a fixed interval of time or space, \
                given a constant average rate of occurrence and independence between events.

    Typical use cases include:
    - Number of emails received per hour
    - Number of cars passing a checkpoint per minute
    - Number of typos on a printed page
    """)

    st.latex(r"P(X = x) = \frac{\lambda^x e^{-\lambda}}{x!}, \quad x = 0, 1, 2, \dots")

    st.markdown("###### Parameters:")
    st.write("$\\lambda$: average number of occurrences in the interval")

    st.markdown("###### Properties:")
    st.write("**Mean:** $\\mu = \\lambda$")
    st.write("**Variance:** $\\sigma^2 = \\lambda$")
    st.write("The Poisson distribution is a limit case of the Binomial distribution when the number of trials is large and the probability of success is small.")

    st.write("")

    st.markdown("**Special Case 2**: The **Multinomial distribution** generalizes the Binomial distribution to more than two categories. It models the probability of counts for each category after a fixed number of independent trials.")

    # LaTeX: PMF formula
    st.latex(r"""
    P(X_1 = x_1, X_2 = x_2, \dots, X_k = x_k) = 
    \frac{n!}{x_1! \, x_2! \, \dots \, x_k!} \cdot p_1^{x_1} \, p_2^{x_2} \, \dots \, p_k^{x_k}
    """)

    # Parameter definitions
    st.latex(r"""
    \text{where:}
    """)
    st.latex(r"""
    \begin{aligned}
    &n = \text{total number of trials} \\
    &k = \text{number of possible categories} \\
    &x_i = \text{number of times outcome } i \text{ occurs (for } i = 1, \dots, k) \\
    &p_i = \text{probability of outcome } i \text{ on a single trial} \\
    &\sum_{i=1}^{k} x_i = n, \quad \sum_{i=1}^{k} p_i = 1
    \end{aligned}
    """)

    # Example (optional)
    st.write("Example:")
    st.markdown("You roll a 6-sided die 10 times. What is the probability of getting:")
    st.markdown("- 2 ones, 2 twos, 2 threes, 1 four, 2 fives, and 1 six?")
    st.latex(r"""
    P(2, 2, 2, 1, 2, 1) = 
    \frac{10!}{2! \cdot 2! \cdot 2! \cdot 1! \cdot 2! \cdot 1!} \cdot \left(\frac{1}{6}\right)^{10}
    """)

    st.write("In r, dmultinom(x=(2,2,2,1,2,1), prob=rep(1/6,6))")
    st.write("-----")



    import matplotlib.pyplot as plt
    import numpy as np
    import scipy.stats as stats
    st.markdown("##### 2c. Geometric Distribution")
    st.markdown("""
    The **Geometric distribution** models the number of **Bernoulli trials** needed to get the **first success**.

    - It is discrete.
    - Each trial is independent.
    - Each trial has a constant success probability \( p \).
    """)

    st.latex(r"""
    f(x) = P(X = k) = (1 - p)^{k - 1} \cdot p, \quad k = 1, 2, 3, \dots
    """)
    st.latex(r"""
    1-F(x) = P(X > k) = (1-p)^k
            """)
    st.latex(r"""
    F(x) = 1- (1-p)^k
            """)

    st.markdown("##### 📌 Parameters:")
    st.write("**Success probability:** $p \\in (0, 1]$")
    st.write("in which $k \\in \\{1, 2, 3, \\dots\\}$")


    st.markdown("##### 📈 Properties:")
    st.write("**Mean:** $\\mu = \\frac{1}{p}$")
    st.write("**Variance:** $\\sigma^2 = \\frac{1 - p}{p^2}$")

    st.markdown("""
    ##### 💡 Example Use Cases:
    - How many coin flips until you get the first heads?
    - How many customer calls until one results in a sale?
    """)

    # Plot the PMF
    p = 0.3
    x = np.arange(1, 15)
    pmf = stats.geom.pmf(x, p)

    fig, ax = plt.subplots()
    ax.bar(x, pmf)
    ax.set_title(f"Geometric PMF (p = {p})")
    ax.set_xlabel("x (Trial of first success)")
    ax.set_ylabel("P(X = x)")

    st.pyplot(fig)

    st.write("Example:<br>" \
    "You roll a pair of 6 sided fair dice until you get double 6s.<br>\
    What is the probability that it will take 20 rolls?<br><br>\
        P(X = 20)= 1-P(X<=19), in r it will be pgeom(19, 1/36)<br><br>", unsafe_allow_html=True)
    st.write("---")
    st.markdown("##### 2d. Negative Binomial Distribution")
    st.write("X is the number of flips until a person gets r heads<br>\
            P(X=k)= P(_ _ _ _ _ _ _ 1)<br>\
            within the _ there are r-1 heads and k-r tails<br><br>\
            In r it is nbinom(k-1, p)", unsafe_allow_html=True)

    st.latex(r"""
    P(X = k) = \binom{k - 1}{r - 1} p^r (1 - p)^{k - r}, \quad k = r, r+1, r+2, \dots
    """)


    st.markdown("### `dbinom(x, size, prob)`")
    st.write("What it means:<br>Gives the probability of getting exactly x successes in size independent trials, where each trial has a prob probability of success.", unsafe_allow_html=True)
    st.write("- `x`: number of successes")
    st.write("- `size`: number of trials")
    st.write("- `prob`: success probability")

    st.markdown("### `pbinom(q, size, prob, lower.tail = TRUE)`")
    st.write("What it means:<br>Returns the cumulative probability of getting at most q successes in size trials.<br>\
            Think of it as:\
    What is the probability I get 4 or fewer heads out of 10 tosses?", unsafe_allow_html=True)
    st.write("- `q`: number of successes (quantile)")
    st.write("- `lower.tail = TRUE`: computes P(X ≤ q)")

    st.markdown("### `qbinom(p, size, prob, lower.tail = TRUE)`")
    st.write("What it means:<br>\
    Returns the smallest number of successes x such that the cumulative probability is at least p.<br> \
    Think of it as: What is the smallest number of heads I can expect such that there's a 90% chance I get that many or fewer?", unsafe_allow_html=True)
    st.write("- `p`: cumulative probability (between 0 and 1)")
    st.write("- returns the smallest `x` such that P(X ≤ x) ≥ p")

    st.markdown("### `rbinom(n, size, prob)`")
    st.write("What it means:<br>\
    Generates n random numbers from a binomial distribution — each representing the number of successes in size trials.\
    Think of it as: Simulate tossing a coin 10 times, and count how many heads you get — do this 100 times.", unsafe_allow_html=True)
    st.write("- `n`: number of random values to generate")
    st.write("- returns binomial random values<br><br>")

    st.write("---")
    st.markdown("##### 2e. **Hypergeometric Distribution**")


    st.latex(r"""
    P(X = x) = \frac{\binom{K}{x} \binom{N - K}{n - x}}{\binom{N}{n}}
    """)
    st.write("In r, ~hyper(r which is defective items in total, n-r which is total non-defective items, m meaning the sample size)")
    # Problem setup
    N = 20  # total items
    K = 6   # defective
    n = 5   # number drawn
    x = 2   # want exactly 2 defectives

    st.markdown(r"""
    **Scenario1:**

    A box contains 20 items, of which 6 are defective.  
    If 5 items are selected at random **without replacement**,  
    what is the probability that exactly 2 are defective?

    """)

    # LaTeX formula
    st.latex(r"""
    P(X = x) = \frac{\binom{K}{x} \binom{N - K}{n - x}}{\binom{N}{n}}
    """)

    st.write(f"With N = {N}, K = {K}, n = {n}, x = {x}")
    from scipy.stats import hypergeom
    # Compute using scipy
    prob = hypergeom.pmf(x, N, K, n)

    st.write(f"**P(X = {x}) = {prob:.5f}**")

    # Optional bar chart of probabilities
    import numpy as np
    import matplotlib.pyplot as plt

    x_vals = np.arange(0, min(K, n) + 1)
    pmf_vals = hypergeom.pmf(x_vals, N, K, n)

    fig, ax = plt.subplots()
    ax.bar(x_vals, pmf_vals, color='slateblue')
    ax.set_title("PMF: Hypergeometric (N=20, K=6, n=5)")
    ax.set_xlabel("x = # of defective items")
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    st.write("**Scenario 2**:")
    st.write("6 students in 2 classes, 5 are friends, they ended up in one class, what is the probability?")
    st.write("r code:<br>\
            dhyper(5,5,55,30)+dhyper(0,5,55,30)", unsafe_allow_html=True)
    st.write("**dhyper(x:# of success drawn, m:# of successes in the population, n:# of failure in the population, k:# number of items drawn)")

    st.write("---")

    st.markdown("##### 2f. Uniform Distribution")
    st.write("In r, ~dunif(X=x, min, max)")

    st.write("---")
    st.markdown("##### 2g. Normal/Gaussian Distribution")
    from scipy.stats import norm

    st.markdown("###### Equations:")
    st.latex(r"""
    f(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left( -\frac{(x - \mu)^2}{2\sigma^2} \right)
    """)

    st.latex(r"""
    \text{If } X \sim \mathcal{N}(\mu, \sigma^2) \text{ and } Y = aX + b,
    \text{ then } Y \sim \mathcal{N}(a\mu + b,\ a^2\sigma^2)
    """)



    st.latex(r"P(X \le x) = \int_{-\infty}^x \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(t - \mu)^2}{2\sigma^2}} \, dt")



    z = 1.5

    # Section 1: LaTeX Notation
    st.write("###### for standard normal PDF and CDF")
    st.latex(r'''
    \phi(z) = \frac{1}{\sqrt{2\pi}} e^{-z^2/2}, \quad
    \Phi(z) = \int_{-\infty}^{z} \phi(t) \, dt
    ''')


    st.write("###### Step-by-step: Standardizing X ~ N(2, 4)")

    st.latex(r'''
    Z = \frac{X - \mu}{\sigma} = \frac{X - 2}{2}
    ''')


    st.write("Gaussian PDF for X ~ N(2, 4)")

    st.latex(r'''
    f_X(x) = \frac{1}{\sqrt{2\pi \cdot 4}} \exp\left( -\frac{(x - 2)^2}{2 \cdot 4} \right)
    = \frac{1}{2\sqrt{2\pi}} \exp\left( -\frac{(x - 2)^2}{8} \right)
    ''')

    st.write("<br><br>**Note**", unsafe_allow_html=True)
    st.latex(r'''
    f_X(y) = \frac{1}{\sqrt{2\pi}} e^{-y^2/2}
    ''')
    st.write(r"The function $f_X(y)$ represents the PDF of $X$ evaluated at $y$.")

    st.write("in r: **pnorm(q, mean = μ, sd = σ, lower.tail = TRUE, log.p = FALSE)**")

    st.write("- $q$: value to evaluate at")
    st.write("- $\\mu$: mean of the distribution (default 0)")
    st.write("- $\\sigma$: standard deviation (default 1)")
    st.write("- `lower.tail = TRUE`: computes $P(X \\le x)$")
    st.write("- `log.p = TRUE`: returns log of the probability")
   
    st.write("Another similar shape: **Cauchy Distribution**")
    from scipy.stats import norm, cauchy
    st.write("The standard **Cauchy distribution** has the PDF:")
    st.latex(r'''
    f(x) = \frac{1}{\pi(1 + x^2)}
    ''')
    st.write("For comparison, the **standard normal distribution** has the (unnormalized) shape:")
    st.latex(r'''
    f(x) \propto e^{-x^2/2}
    ''')

    # x values
    x = np.linspace(-10, 10, 1000)
    cauchy_pdf = cauchy.pdf(x)
    normal_pdf = norm.pdf(x)

    # Plot both distributions
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(x, cauchy_pdf, label="Cauchy PDF", color='red')
    ax.plot(x, normal_pdf, label="Normal PDF", color='blue', linestyle='--')
    ax.set_title("Cauchy vs. Normal Distribution")
    ax.set_xlabel("x")
    ax.set_ylabel("Density")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)

    # Explain the key difference
    st.write("###### Key Difference:")
    st.write(
        "- The **Cauchy distribution has heavy tails**, meaning more probability in the extremes.\n"
        "- The **mean and variance of the Cauchy distribution are undefined**.\n"
        "- Unlike the normal distribution, the **sample mean of Cauchy values does not converge** as the sample size increases."
    )
    st.write("---")
    st.write("**Standardization as a Linear Transformation**:")
    st.write("$Z = \\alpha X + b$")

    # Definitions of variables
    st.write("$X \\sim \\mathcal{N}(\\mu, \\sigma^2)$")
    st.write("$\\alpha = \\frac{1}{\\sigma}$")
    st.write("$b = -\\frac{\\mu}{\\sigma}$")

    # Substitution result
    st.write("$Z = \\frac{1}{\\sigma} X - \\frac{\\mu}{\\sigma} = \\frac{X - \\mu}{\\sigma}$")

    # Conclusion
    st.write("This transformation converts any normal variable $X$ into a standard normal variable $Z \\sim \\mathcal{N}(0, 1)$.")

    st.write("**De-standardization**")
    st.write("$X = \\sigma Z + \\mu$")

    # Definitions
    st.write("$Z \\sim \\mathcal{N}(0, 1)$")
    st.write("$X \\sim \\mathcal{N}(\\mu, \\sigma^2)$")
    st.write("This reverses the standardization: $Z = \\frac{X - \\mu}{\\sigma}$")
    st.write("The Z score: $Z = \\frac{X - \\mu}{\\sigma}$")
            
    st.write("---")
    st.markdown("###### **Bivariate normal density**")

    st.markdown("###### Definition")

    st.markdown("The bivariate normal distribution describes the joint behavior of two continuous variables \( X \) and \( Y \), where each marginal distribution is normal and the joint distribution accounts for correlation.")

    st.latex(r"""
    f_{X,Y}(x, y) =
    \frac{1}{2\pi\sigma_X\sigma_Y\sqrt{1 - \rho^2}} \exp\left(
    -\frac{1}{2(1 - \rho^2)} \left[
    \left(\frac{x - \mu_X}{\sigma_X}\right)^2
    - 2\rho \left(\frac{x - \mu_X}{\sigma_X}\right) \left(\frac{y - \mu_Y}{\sigma_Y}\right)
    + \left(\frac{y - \mu_Y}{\sigma_Y}\right)^2
    \right] \right)
    """)
    st.write("When mean of x and y is 0 and sd of x and y=1:")
    st.latex(r"""
    f_{X,Y}(x, y) = \frac{1}{2\pi \sqrt{1 - \rho^2}} \cdot 
    \exp\left( -\frac{1}{2(1 - \rho^2)} 
    \left( x^2 - 2\rho x y + y^2 \right) \right)
    """)
    st.latex(r"""
    \rho = \text{Cor}(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)} \cdot \sqrt{\text{Var}(Y)}}
    """)
    st.markdown("###### Parameters")

    st.latex(r"\mu_X, \mu_Y : \text{Means of } X \text{ and } Y")
    st.latex(r"\sigma_X, \sigma_Y : \text{Standard deviations of } X \text{ and } Y")
    st.latex(r"\rho : \text{Correlation coefficient between } X \text{ and } Y")
    st.latex(r"-1 < \rho < 1")


    st.markdown("###### Key Properties")

    st.latex(r"""
    f_{X,Y}(x, y) \geq 0 \quad \text{and} \quad \iint_{\mathbb{R}^2} f_{X,Y}(x, y) \, dx \, dy = 1
    """)

    st.latex(r"""
    f_X(x) \sim \mathcal{N}(\mu_X, \sigma_X^2), \quad f_Y(y) \sim \mathcal{N}(\mu_Y, \sigma_Y^2)
    """)


    st.latex(r"""
    \text{If } \rho = 0, \text{ then } X \text{ and } Y \text{ are independent, and } 
    f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y)
    """)

    st.latex(r"""
    f_{X,Y}(x, y) = f_X(x) \cdot f_Y(y)
    """)

    st.markdown("""
    Two random variables can each follow a **normal distribution individually** (marginally), but their **joint distribution** might not be a valid **bivariate normal**.
    (bivariate normal means that Any linear combination 
    𝑎
    𝑋
    +
    𝑏
    𝑌
    is also normally distributed.
    )""")

    st.write("---")
    st.markdown("##### 2h. Gamma Density")
    st.markdown("###### Probability Density Function (PDF):")
    st.latex(r"""
    f(x; \alpha, \beta) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha - 1} e^{-\beta x}, \quad x > 0
    """)
    st.write(r"with $\Gamma(x) = \int_0^{\infty} u^{x - 1} e^{-u} \, du$")

    st.markdown("""
    The **Gamma distribution** is defined by two positive parameters:

    - $\\alpha$ (**shape parameter**):  
    Determines the shape of the distribution.  
    - If $\\alpha < 1$: the distribution is **heavily right-skewed**
    - If $\\alpha = 1$: the distribution becomes **Exponential**
    - If $\\alpha > 1$: the distribution becomes **more symmetric**

    - $\\beta$ (**rate parameter**):  
    Controls the rate at which events occur (inverse of scale).  
    Higher $\\beta$ compresses the distribution toward 0.

    > Alternatively, the **scale parameter** $\\theta = 1/\\beta$ is sometimes used instead of rate.

    ###### Key Properties:
    - **Mean:** $\\mu = \\frac{\\alpha}{\\beta}$
    - **Variance:** $\\sigma^2 = \\frac{\\alpha}{\\beta^2}$
    """)


  
    st.write("""
    **Example:**  
    Time until 5 light bulbs fail → Gamma with shape α = 5
    """, unsafe_allow_html=True)

    st.write("""
    Used for modeling **lifetimes** of components or systems with non-constant failure rates.

    **Example:**  
    Lifespan of a machine that becomes more failure-prone with age.
    """)
    st.write("""
    Models **interarrival times** or **service durations** in queuing theory.

    **Example:**  
    Time until the 10th customer arrives at a service desk.
    """)

    st.write("---")
    st.markdown("##### 2i. Exponential Density")

    st.markdown("""
    The **Exponential distribution** models the time until the **first event** in a Poisson process — that is, a process where events happen continuously and independently at a constant average rate.

    ###### Probability Density Function (PDF):

    For rate parameter lambda > 0:

    """)

    st.latex(r"""
    f(x; \lambda) = \lambda e^{-\lambda x}, \quad x \ge 0
    """)
    st.write("**Where:**")
    st.write("- $x$: time until the event")
    st.write("- $\\lambda$: rate of occurrence (events per unit time)")
    st.write("- $\\frac{1}{\\lambda}$: mean time between events")

    st.write("**Key Properties:**")
    st.write("- Mean: $\\mu = \\frac{1}{\\lambda}$")
    st.write("- Variance: $\\sigma^2 = \\frac{1}{\\lambda^2}$")
    st.write("- Memoryless property: $P(X > s + t \\mid X > s) = P(X > t)$")
    st.markdown("""
    ###### Common Use Cases:
    - Time until next phone call
    - Time between radioactive decays
    - Lifespan of a component with constant failure rate
    """)

    st.markdown("""
                **Compare with Poisson Distribution**
    | Property               | Poisson Distribution              | Exponential Distribution            |
    |------------------------|-----------------------------------|-------------------------------------|
    | Type                   | Discrete                          | Continuous                          |
    | Models                 | Number of events in an interval   | Time between events                 |
    | Parameter              | lambda: rate per interval     | lambda: rate per time unit         |
    | Example                | Emails received per hour          | Time until next email               |
    """)

    st.write("---")


    st.markdown("##### 2j. Expected Value")

    # General definition
    st.write("###### General Definition of Expected Value")
    st.latex(r'''
    \mathbb{E}[g(X)] = \begin{cases}
    \sum_x g(x) \cdot P(X = x), & \text{(discrete)} \\
    \int_{-\infty}^{\infty} g(x) \cdot f(x)\, dx, & \text{(continuous)}
    \end{cases}
    ''')

    # Convergence condition
    st.write("###### ⚠️ When Is the Expected Value Defined?")
    st.write("We must check whether the expectation **converges absolutely**.")

    st.latex(r'''
    \text{Discrete: } \sum_x |g(x)| \cdot P(X = x) < \infty \quad \Rightarrow \mathbb{E}[g(X)] \text{ exists}
    ''')

    st.latex(r'''
    \text{Continuous: } \int_{-\infty}^{\infty} |g(x)| \cdot f(x)\, dx < \infty \quad \Rightarrow \mathbb{E}[g(X)] \text{ exists}
    ''')

    st.write("If the sum or integral **diverges**, then:")
    st.latex(r'''
    \boxed{\mathbb{E}[g(X)] \text{ is undefined}}
    ''')
    st.write("For Discrete case:")
    st.latex(r"""
    \text{Var}(X) = \sum (x - \mu)^2 \cdot P(x)
    """)
    st.write("For Continuous case:")
    st.latex(r"""
    \text{Var}(X) = \int_{-\infty}^{\infty} (x - \mu)^2 \cdot f(x) \, dx
    """)
    # Example of divergence: Cauchy
    st.write("###### ❌ Example: Undefined Expectation (Cauchy Distribution)")
    st.latex(r'''
    f(x) = \frac{1}{\pi(1 + x^2)} \quad \text{(Cauchy)} \\
    \mathbb{E}[X] = \int_{-\infty}^{\infty} x \cdot f(x)\, dx \quad \text{diverges}
    ''')

    st.write("Even though the Cauchy PDF is valid (area = 1), the expectation diverges due to the heavy tails.")

    # Example of proper convergence
    st.write("###### ✅ Example: Defined Expectation (Uniform Distribution)")
    st.latex(r'''
    X \sim \text{Uniform}(0,1) \Rightarrow f(x) = 1 \\
    \mathbb{E}[X] = \int_0^1 x \cdot 1 \, dx = \frac{1}{2}
    ''')

    st.write("###### Problem Statement")
    st.write(
        "A stick of unit length is broken at two points, each chosen independently and uniformly at random along the stick.\n"
        "What is the **expected length of the middle piece**?"
    )



    # Integral expression
    st.write("###### Step 1: Set up the expected value as a double integral")
    st.latex(r'''
    \mathbb{E}[|U_1 - U_2|] = \int_0^1 \int_0^1 |u_1 - u_2| \, du_1 \, du_2
    ''')

    # Splitting the domain
    st.write("###### Step 2: Split the domain based on absolute value")
    st.latex(r'''
    |u_1 - u_2| =
    \begin{cases}
    u_1 - u_2, & \text{if } u_1 \geq u_2 \\
    u_2 - u_1, & \text{if } u_1 < u_2
    \end{cases}
    ''')

    st.write("By symmetry, we double the integral over the triangle")
    st.latex(r'''
    \mathbb{E}[|U_1 - U_2|] = 2 \int_0^1 \int_0^{u_1} (u_1 - u_2) \, du_2 \, du_1
    ''')

    # Inner integral
    st.write("###### Step 3: Evaluate the inner integral")
    st.latex(r'''
    \int_0^{u_1} (u_1 - u_2) \, du_2 
    = \left[ u_1 u_2 - \frac{u_2^2}{2} \right]_0^{u_1} 
    = u_1^2 - \frac{u_1^2}{2} 
    = \frac{u_1^2}{2}
    ''')

    # Outer integral
    st.write("###### Step 4: Evaluate the outer integral")
    st.latex(r'''
    \mathbb{E}[|U_1 - U_2|] = 2 \int_0^1 \frac{u_1^2}{2} \, du_1 
    = \int_0^1 u_1^2 \, du_1 
    = \left[ \frac{u_1^3}{3} \right]_0^1 = \frac{1}{3}
    ''')
    st.write("")

    # Main principle
    st.write("###### 📘 General Rule")
    st.latex(r'''
    \mathbb{E}[g(X)] \neq g(\mathbb{E}[X]) \quad \text{in general}
    ''')

    st.write(
        "The equality holds **only if** \( g(x) \) is a **linear function**, i.e., of the form \( g(x) = ax + b \)."
    )

    # Linear case
    st.write("###### ✅ Case 1: Linear Function")
    st.write("If \( g(x) = ax + b \), then:")
    st.latex(r'''
    \mathbb{E}[g(X)] = \mathbb{E}[aX + b] = a \mathbb{E}[X] + b = g(\mathbb{E}[X])
    ''')

    # Nonlinear counterexample
    st.write("###### ❌ Case 2: Nonlinear Function (Counterexample)")
    st.write("Let \( X \sim \text{Uniform}(-1, 1) \) and let \( g(x) = x^2 \). Then:")

    st.latex(r'''\mathbb{E}[X] = 0''')
    st.latex(r'''g(\mathbb{E}[X]) = (0)^2 = 0''')

    st.write("But:")
    st.latex(r'''
    \mathbb{E}[X^2] = \int_{-1}^1 x^2 \cdot \frac{1}{2} \, dx = \frac{1}{3}
    ''')

    st.write("So we clearly have:")
    st.latex(r'''
    \mathbb{E}[X^2] \neq (\mathbb{E}[X])^2
    ''')

    # Summary
    st.write("###### 📦 Summary")
    st.markdown("- If \( g \) is nonlinear, then:")
    st.latex(r'''\mathbb{E}[g(X)] \neq g(\mathbb{E}[X])''')
    st.markdown("- The expectation operator does distribute through linear functions.")
    st.markdown("- Always be careful when applying functions to expected values!")

    st.write("Side note: Variance Calculation")
    st.latex(r'''
    \mathbb{E}[(X - \mu)^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2
    ''')
    st.write("")
    st.write("")
    st.markdown("##### 2k. Markov's Inequality")



    st.write("###### 💡 Key Idea")
    st.write(
        "Even if you don't know the full distribution of a random variable, "
        "Markov's inequality lets you bound how likely it is that a **non-negative** variable exceeds a threshold, "
        "based only on its expected value."
    )
    st.latex(r'''
    P(X \geq a) \leq \frac{\mathbb{E}[X]}{a}
    ''')

    # Example
    st.write("###### 🎲 Example")
    st.write("Let a non-negative random variable have expected value:")
    st.latex(r'''\mathbb{E}[X] = 10''')
    st.write("Then the probability that x>=20 is bounded by:")
    st.latex(r'''
    P(X \geq 20) \leq \frac{10}{20} = 0.5
    ''')

    # Condition for validity
    st.write("###### ✅ When Can You Use It?")
    st.write("Markov’s inequality holds under these conditions:")
    st.latex(r'''
    X \geq 0 \quad \text{and} \quad \mathbb{E}[X] < \infty
    ''')

    # Summary
    st.write("###### 📦 Summary")
    st.markdown("""
    - Works for **any non-negative** random variable  
    - Useful when you know the **mean but not the full distribution**  
    - Gives a **conservative bound** on the upper tail probability  
    """)

    ##class 18
    st.write("##### 2l. Chebyshev's Inequality")

    st.markdown("""
    Chebyshev's Inequality provides a lower bound on the proportion of values that lie within a certain number of standard deviations from the mean, **regardless of the distribution shape** (as long as the variance is finite).  
    """)


    st.latex(r"""
    P(|X - \mu| \geq t) \leq \frac{\sigma^2}{t^2}
    """)


    st.markdown("""
    ✅ This inequality holds for **all** distributions with a finite mean and variance.  
    🎯 It's especially useful when the distribution is unknown or non-normal.
    """)
    mu = 10  # mean height in inches
    sigma = 2  # assumed standard deviation
    x1 = 15
    x2 = 2

    st.markdown("###### 🦝 Chebyshev's Inequality: Raccoon Height Example")

    st.markdown(f"""
    **Given**:  
    - Average (mean) raccoon height: \( {mu} \) inches  
    - Assumed standard deviation: \( {sigma} \) inches  
    """)

    st.markdown(f"""
    ###### ❓ Question :
    **At least ___% of raccoons are between 5 and 15 inches?**
    """)

    mu = 10
    t = 5
    var = 4
    sigma = var ** 0.5

    upper_bound = var / (t**2)
    lower_bound = 1 - upper_bound



    st.latex(r"""
    P(|X - \mu| \geq t) \leq \frac{\sigma^2}{t^2}
    """)

    st.markdown(f"""
    **Given**:
    - Mean \( {mu} \)
    - Standard deviation \( ={sigma} \), variance = ({var})


    ### Result:
    """)

    st.latex(rf"""
    P(|X - {mu}| \geq {t}) \leq \frac{{{var}}}{{{t}^2}} = \frac{{{var}}}{{{t**2}}} = {upper_bound:.2f}
    """)

    st.markdown(f"""
    ✅ **At least {lower_bound:.0%}** of the values lie between {mu - t} and {mu + t}.
    """)

    st.write("")
    st.markdown("##### 2m. Covariance")

    st.markdown("""
    Covariance measures how two random variables vary **together**.  
    If \( X \) and \( Y \) tend to be above or below their means at the same time, covariance is positive.
    """)

    st.latex(r"""
    \text{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)]
    """)

    st.markdown("An equivalent form useful for calculations is:")

    st.latex(r"""
    \text{Cov}(X, Y) = \mathbb{E}[XY] - \mathbb{E}[X] \cdot \mathbb{E}[Y]
    """)

    st.markdown("""
    ✅ If Cov(X, Y) > 0 : they tend to increase together  
    ✅ If Cov(X, Y) < 0 : one increases while the other decreases  
    ✅ If Cov(X, Y) = 0: no linear relationship
    """)
    st.write("###### 📘 Summary of E(·), Var(·), Cov(·), Cor(·) Rules")

    # Expectation
    st.markdown("#### 💡 Expectation")
    st.latex(r"E(a + bX) = a + b \cdot \mathbb{E}[X]")
    st.latex(r"\mathbb{E}[X + Y] = \mathbb{E}[X] + \mathbb{E}[Y] \quad \text{(always)}")

    # Variance
    st.markdown("#### 📏 Variance")
    st.latex(r"\text{Var}(a + bX) = b^2 \cdot \text{Var}(X)")
    st.latex(r"\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) \quad \text{if } X \perp Y")
    st.latex(r"\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\cdot \text{Cov}(X, Y) \quad \text{(general)}")

    # Standard Deviation
    st.markdown("#### 📐 Standard Deviation")
    st.latex(r"\text{SD}(a + bX) = |b| \cdot \text{SD}(X)")
    st.markdown("To get SD of a sum: take the square root of the variance formula above.")

    # Covariance
    st.markdown("#### 🔗 Covariance")
    st.latex(r"\text{Cov}(a + bX, c + dY) = bd \cdot \text{Cov}(X, Y)")
    st.latex(r"""
    \text{Cov}(W + X, Y + Z) = \text{Cov}(W, Y) + \text{Cov}(W, Z) + 
    \text{Cov}(X, Y) + \text{Cov}(X, Z)
    """)

    # Correlation
    st.markdown("#### 📉 Correlation")
    st.latex(r"""
    \text{Cor}(X, Y) = \rho_{X,Y} = \frac{\text{Cov}(X, Y)}{\text{SD}(X) \cdot \text{SD}(Y)}
    """)
    st.latex(r"\text{Cor}(a + bX, c + dY) = \text{sign}(b) \cdot \text{sign}(d) \cdot \text{Cor}(X, Y)")
    st.latex(r"""
    \text{Cor}(W + X, Y + Z) = \frac{\text{Cov}(W + X, Y + Z)}{\text{SD}(W + X) \cdot \text{SD}(Y + Z)}
    """)

    # Tips
    st.markdown("#### 🛠️ Tips")
    st.markdown("""
    - These formulas extend to **sums of more than 2 variables** — see your book for the generalized versions.
    - You can **combine rules**. For example:
    """)

    st.latex(r"""
    \text{Cov}(a + bX, Y + Z) = b \cdot \text{Cov}(X, Y + Z) = b[\text{Cov}(X, Y) + \text{Cov}(X, Z)]
    """)


    st.latex(r"""
    \text{Cov}(U, V) = \mathbb{E}[(U - \mathbb{E}[U])(V - \mathbb{E}[V])]
    """)

    st.write("Solve this:")
    st.latex(r"""
    \text{Cov}(a + X, Y) = \mathbb{E}[(a + X - \mathbb{E}[a + X])(Y - \mathbb{E}[Y])]
    """)



    st.latex(r"""
    \mathbb{E}[a + X] = a + \mathbb{E}[X] \Rightarrow 
    \text{Cov}(a + X, Y) = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])]
    """)


    st.latex(r"""
    \text{Cov}(a + X, Y) = \text{Cov}(X, Y)
    """)

    st.markdown("Solve it:")

    st.latex(r"""
    \text{Cov}(X, Y + Z)
    """)


    st.latex(r"""
    \text{Cov}(X, Y + Z) = \mathbb{E}[(X - \mathbb{E}[X])((Y + Z) - \mathbb{E}[Y + Z])]
    """)


    st.latex(r"""
    = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y] + Z - \mathbb{E}[Z])]
    """)


    st.latex(r"""
    = \mathbb{E}[(X - \mathbb{E}[X])(Y - \mathbb{E}[Y])] + \mathbb{E}[(X - \mathbb{E}[X])(Z - \mathbb{E}[Z])]
    """)


    st.latex(r"""
    = \text{Cov}(X, Y) + \text{Cov}(X, Z)
    """)

    st.write("Theorem 1:")
    st.latex(r"""
    \text{Var}\left(\sum_{i=1}^{n} X_i\right) = \sum_{i=1}^{n} \text{Var}(X_i) \quad \text{if the } X_i \text{ are independent}
    """)
    st.markdown("""
    If \( X \) and \( Y \) are **jointly distributed random variables**, and their **variances and covariance exist**, then the correlation is defined as:
    """)

    st.latex(r"""
    \text{Cor}(X, Y) = \frac{\text{Cov}(X, Y)}{\sqrt{\text{Var}(X)} \cdot \sqrt{\text{Var}(Y)}}
    """)
    st.write("Theorem 2:")
    st.latex(r"""
    -1 \leq \rho \leq 1
    """)

    st.markdown("###### ✅ Furthermore:")

    st.latex(r"""
    \rho = \pm 1 \quad \text{if and only if} \quad P(Y = a + bX) = 1
    """)

    st.markdown("""
    That is, \( Y \) is a **perfect linear function** of \( X \) with probability 1.  
    - \( \rho = 1 \): Perfect positive linear relationship  
    - \( \rho = -1 \): Perfect negative linear relationship
    """)
    st.write("")
    st.markdown("##### 2n. Limit Theorems")

    st.write("Theorem A: Law of Large Numbers")
    # Parameters

    st.latex(r"""
    \text{Let } X_1, X_2, \dots, X_n \text{ be i.i.d. random variables with } \mathbb{E}[X_i] = \mu.
    """)
    st.latex(r"""
    \bar{X}_n = \frac{1}{n} \sum_{i=1}^{n} X_i \xrightarrow{P} \mu \quad \text{as } n \to \infty
    """)
    st.latex(r"""
    \text{This means: For any } \epsilon > 0, \quad 
    \mathbb{P}\left( \left| \bar{X}_n - \mu \right| > \epsilon \right) \to 0 \quad \text{as } n \to \infty
    """)

    st.write("")
    st.write("Theorem B: Central Limit Theorem")
    import seaborn as sns
    st.latex(r"""
    \text{Let } X_1, X_2, \dots, X_n \text{ be i.i.d. random variables with mean } \mu \text{ and variance } \sigma^2.
    """)

    st.latex(r"""
    \text{Then the standardized sample mean:}
    \quad
    Z_n = \frac{\bar{X}_n - \mu}{\sigma / \sqrt{n}} 
    \quad \xrightarrow{d} \quad \mathcal{N}(0, 1) \quad \text{as } n \to \infty
    """)

    st.latex(r"""
    \text{This means the distribution of the sample mean } \bar{X}_n \text{ approaches:}
    \quad
    \bar{X}_n \sim \mathcal{N}\left(\mu, \frac{\sigma^2}{n} \right)
    """)

    st.latex(r"""
    \text{Regardless of the original distribution of } X_i, \text{ the sample mean } \bar{X}_n \text{ becomes approximately normal as } n \text{ increases.}
    """)

    st.latex(r"""
    \text{The accuracy of the normal approximation improves as } n \to \infty.
    """)
    st.write("")


   

   

    st.markdown("###### ✅ Classic CLT for Sample Mean")
    st.latex(r"""
    \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \approx \mathcal{N}(0, 1)
    """)
    st.markdown("- Applies to any i.i.d. data with finite variance")

    # Divider
    st.markdown("---")

    st.markdown("###### 🔢 CLT Formulas for Different Distributions")

    st.markdown("###### 📊 Binomial(n, p) – Count")
    st.latex(r"""
    \frac{X - np}{\sqrt{np(1 - p)}} \approx \mathcal{N}(0, 1)
    """)
    st.markdown("- Use when approximating the number of successes in n trials")

    st.markdown("###### 📈 Binomial(n, p) – Proportion")
    st.latex(r"""
    \frac{\hat{p} - p}{\sqrt{p(1 - p) / n}} \approx \mathcal{N}(0, 1)
    """)
    st.markdown("- Useful for confidence intervals and hypothesis tests for proportions")

    st.markdown("###### 🧮 Poisson(λ)")
    st.latex(r"""
    \frac{N - \lambda}{\sqrt{\lambda}} \approx \mathcal{N}(0, 1)
    """)
    st.markdown("- Use for large λ when approximating event counts")

    st.markdown("###### 🎯 Geometric(p) and Negative Binomial(r, p)")
    st.latex(r"""
    \text{Mean: } \mu = \frac{r(1-p)}{p}, \quad \text{Variance: } \sigma^2 = \frac{r(1-p)}{p^2}
    """)
    st.latex(r"""
    \frac{X - \mu}{\sigma} \approx \mathcal{N}(0, 1)
    """)
    st.markdown("- Apply when summing many trials to success")

    st.markdown("###### 📦 Hypergeometric(N, K, n)")
    st.latex(r"""
    \mu = n \cdot \frac{K}{N}, \quad \sigma^2 = n \cdot \frac{K}{N} \cdot \left(1 - \frac{K}{N}\right) \cdot \left(\frac{N - n}{N - 1}\right)
    """)
    st.latex(r"""
    \frac{X - \mu}{\sigma} \approx \mathcal{N}(0, 1)
    """)
    st.markdown("- Approximate sampling without replacement (finite population)")

    st.markdown("###### 📉 Exponential(λ)")
    st.latex(r"""
    \text{Mean: } \mu = \frac{1}{\lambda}, \quad \text{SD: } \sigma = \frac{1}{\lambda}
    """)
    st.latex(r"""
    \frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \approx \mathcal{N}(0, 1)
    """)
    st.markdown("- CLT still applies even though Exponential is skewed")

    # End
    st.markdown("---")
    st.markdown("✅ These approximations become better as sample size or parameters increase.")

    st.markdown("##### 2o. Chi Square Distribution")

    st.latex(r"""
    \text{If } Z_1, Z_2, \dots, Z_k \text{ are i.i.d. standard normal variables, i.e., } Z_i \sim \mathcal{N}(0, 1),
    """)

    st.latex(r"""
    \text{then the random variable: } 
    \quad
    \chi^2_k = \sum_{i=1}^{k} Z_i^2 
    \quad \text{follows a Chi-Square distribution with } k \text{ degrees of freedom}.
    """)

    st.latex(r"""
    \begin{aligned}
    &\bullet\ \text{Mean of } \chi^2_k \text{ is } k \\
    &\bullet\ \text{Variance of } \chi^2_k \text{ is } 2k \\
    &\bullet\ \chi^2_k \text{ is right-skewed, but becomes more symmetric as } k \text{ increases} \\
    &\bullet\ \text{Applications:} \\
    &\quad \circ\ \text{Goodness-of-fit tests} \\
    &\quad \circ\ \text{Tests of independence (e.g., contingency tables)} \\
    &\quad \circ\ \text{Confidence intervals for variance}
    \end{aligned}
    """)

    st.write("Besides:")
    st.markdown("Let:")
    st.latex(r"X_i \sim \mathcal{N}(\mu, \sigma^2)")

    st.markdown("Then the standardized form is:")
    st.latex(r"Z_i = \frac{X_i - \mu}{\sigma} \sim \mathcal{N}(0, 1)")

    st.markdown("Squaring both sides gives:")
    st.latex(r"\left( \frac{X_i - \mu}{\sigma} \right)^2 \sim \chi^2_1")

    st.markdown("✅ This is because the **square of a standard normal variable** follows a chi-square distribution with 1 degree of freedom.")


    st.markdown("###### ❓ Question")


    st.latex(r"""
    \text{Let } X_1, X_2, \dots, X_n \text{ be i.i.d. random variables from } \mathcal{N}(0, \sigma^2). \\
    \text{Show that the statistic } \sum_{i=1}^{n} X_i^2 \text{ follows a scaled chi-square distribution:} \\
    \sum_{i=1}^n X_i^2 \sim c \cdot \chi^2_k \\
    \text{What are the values of } c \text{ and } k\text{?}
    """)

    st.latex(r"""
    \sum_{i=1}^n X_i^2 \sim c \cdot \chi^2_k
    """)

    st.write("What are the values of \( c \) and \( k \)?")

    # Step-by-step derivation
    #st.markdown("### ✅ Step-by-Step Derivation")

    st.markdown("**Step 1: Given**")
    st.latex(r"X_i \sim \mathcal{N}(0, \sigma^2), \quad \text{i.i.d. for } i = 1, \dots, n")

    st.markdown("**Step 2: Standardize Each \( X_i \)**")
    st.latex(r"Z_i = \frac{X_i}{\sigma} \quad \Rightarrow \quad Z_i \sim \mathcal{N}(0, 1)")

    st.markdown("**Step 3: Square and Sum**")
    st.latex(r"""
    \sum_{i=1}^n X_i^2 = \sum_{i=1}^n (\sigma Z_i)^2 = \sigma^2 \sum_{i=1}^n Z_i^2
    """)

    st.markdown("**Step 4: Use Definition of Chi-Square Distribution**")
    st.latex(r"\sum_{i=1}^n Z_i^2 \sim \chi^2_n")

    #st.markdown("### 🟩 Final Result")
    st.write("Definition to Chi Square:")
    st.latex(r"\sum_{i=1}^n X_i^2 \sim \sigma^2 \chi^2_n")

    st.latex(r"""
    \begin{aligned}
    \text{Thus:} \\
    c &= \sigma^2 \\
    k &= n
    \end{aligned}
    """)

    st.write("The distribution of the scaled sample variance is the chi-square distribution with \( n - 1 \) degrees of freedom:")
    st.latex(r"\frac{(n - 1) S^2}{\sigma^2} \sim \chi^2_{n - 1}")

    st.markdown("##### 2p. T distribution")
    st.write("Let:")
    st.latex(r"Z \sim \mathcal{N}(0, 1)")
    st.latex(r"U \sim \chi_n^2")
    st.write("where \( Z \) and \( U \) are independent.")

    st.write("Then the Student’s *t*-distribution is defined as:")
    st.latex(r"""
    T = \frac{Z}{\sqrt{U / n}} \sim t_n
    """)
    st.write("t distribution with n degrees of freedom")
    st.write("###### 🎯 Properties:")
    import scipy.stats as stats
    import matplotlib.pyplot as plt

    st.markdown("""
    - Bell-shaped, symmetric like the normal distribution  
    - Heavier tails (more prone to outliers)  
    """)
    st.latex(r"\text{As } n \to \infty, \quad t_n \to \mathcal{N}(0, 1)")
    st.latex(r"\mathbb{E}[T] = 0 \quad \text{for } n > 1")

    st.latex(r"\mathrm{Var}(T) = \frac{n}{n - 2} \quad \text{for } n > 2")
    st.markdown("""
    - Used when estimating population mean with unknown variance  
    - Appears naturally in small-sample inference
    """)

    st.markdown("##### Sample Mean and Variance of Student's t-Distribution")
    st.latex(r"\bar{T} = \frac{1}{m} \sum_{i=1}^{m} T_i")
    st.latex(r"S^2 = \frac{1}{m - 1} \sum_{i=1}^{m} (T_i - \bar{T})^2")

    # User inputs
    df = st.slider("Degrees of freedom (n)", min_value=1, max_value=100, value=3)
    sample_size = st.slider("Sample size (m)", min_value=10, max_value=1000, value=100, step=10)
    seed = st.number_input("Random seed (optional)", value=42)

    # Generate sample
    np.random.seed(seed)
    sample = np.random.standard_t(df, size=sample_size)

    # Compute sample statistics
    sample_mean = np.mean(sample)
    sample_var = np.var(sample, ddof=1)

    st.write("###### 📊 Results")
    st.latex(f"\\text{{Sample Mean: }} \\quad \\bar{{T}} = {sample_mean:.4f}")

    st.latex(f"\\text{{Sample Variance: }} \\quad S^2 = {sample_var:.4f}")


    # Optional: Plot histogram


    fig, ax = plt.subplots()
    ax.hist(sample, bins=30, density=True, alpha=0.6, label="Sample")
    x = np.linspace(min(sample), max(sample), 500)
    ax.plot(x, stats.t.pdf(x, df), 'r-', lw=2, label=f"t-distribution (df={df})")
    ax.axvline(sample_mean, color='k', linestyle='dashed', linewidth=1, label="Sample Mean")
    ax.set_title("Histogram of Sample from t-Distribution")
    ax.legend()
    st.pyplot(fig)

    st.write("---")

    st.markdown("###### ✅ When population standard deviation is **known**:")
    st.write("The standardized sample mean follows a **standard normal** distribution:")
    st.latex(r"\frac{\bar{X} - \mu}{\sigma / \sqrt{n}} \sim \mathcal{N}(0, 1)")

    st.markdown("###### ❓ When population standard deviation is **unknown**:")
    st.write("We estimate it with the sample standard deviation \( S \), and the statistic follows a **t-distribution**:")
    st.latex(r"\frac{\bar{X} - \mu}{S / \sqrt{n}} \sim t_{n - 1}")

    st.write("")

    st.markdown("##### 2q. Confidence Interval")

    st.markdown("###### ✅ When population standard deviation is **known** (Z-distribution):")
    st.latex(r"\bar{X} \pm z^* \cdot \frac{\sigma}{\sqrt{n}}")

    st.markdown("###### ❓ When is **unknown** and estimated by sample SD \\( S \\) (t-distribution):")
    st.latex(r"\bar{X} \pm t^* \cdot \frac{S}{\sqrt{n}}")

    st.latex(r"\bar{X} \quad \text{: sample mean}")
    st.latex(r"\sigma \quad \text{: population standard deviation}")
    st.latex(r"S \quad \text{: sample standard deviation}")
    st.latex(r"z^* \quad \text{: Z-critical value from standard normal (e.g., 1.96 for 95\%)}")
    st.latex(r"t^* \quad \text{: t-critical value from t-distribution with } n - 1 \text{ degrees of freedom}")
    st.latex(r"n \quad \text{: sample size}")


    # Inputs
    p_hat = 0.12
    n = 100
    z_star = 1.28
    import math
    # Standard Error
    SE = math.sqrt(p_hat * (1 - p_hat) / n)
    margin_of_error = z_star * SE
    lower = p_hat - margin_of_error
    upper = p_hat + margin_of_error
    # Output
    st.markdown("###### 80% Confidence Interval for a Proportion")

    st.markdown("###### ❓ Question:")
    st.write("""
    In a sample of 100 condo owners, 12% said they plan to try to sell their condo unit in the next year.  
    Find an 80% confidence interval for the proportion of **all** Austin condo owners who plan to sell.
    """)

    st.markdown("###### ✅ Solution:")
    st.latex(r"\hat{p} = 0.12,\quad n = 100,\quad z^* = 1.28")

    st.markdown("**Standard Error:**")
    st.latex(r"\text{SE} = \sqrt{ \frac{0.12 \cdot 0.88}{100} } = 0.0325")

    st.markdown("**Confidence Interval:**")
    st.latex(r"0.12 \pm 1.28 \cdot 0.0325 = 0.12 \pm 0.0416")
    st.latex(r"(0.0784,\ 0.1616)")

    st.success("We are 80% confident that between 7.84% and 16.16% of all Austin condo owners plan to sell their unit in the next year.")


    st.markdown("###### ❓ When is the normal approximation invalid?")

    st.markdown("""
    You **cannot** use the formula:
    """)

    st.latex(r"\hat{p} \pm z^* \cdot \sqrt{ \frac{ \hat{p}(1 - \hat{p}) }{n} }")

    st.markdown("when these conditions are not met:")

    st.latex(r"np \ge 10 \quad \text{and} \quad n(1 - p) \ge 10")
    st.write("")
    st.markdown("###### ⚠️ Why?")
    st.write("""
    - The normal approximation assumes the sampling distribution of \( \hat{p} \) is bell-shaped.
    - If there are too few successes or failures, the binomial distribution becomes skewed.
    """)
    st.write("")
    st.write("Another setup:")
    # Given values
    x_bar = 1.6
    s = 0.8
    n = 100
    N = 8000
    df = n - 1
    t_star = 1.984  # from t-table for 95% confidence, df = 99

    # Standard Error with finite population correction
    SE_fpc = (s / math.sqrt(n)) * math.sqrt((N - n) / (N - 1))
    ME = t_star * SE_fpc
    lower = x_bar - ME
    upper = x_bar + ME

    # Output
    st.markdown("###### 95% Confidence Interval with Finite Population Correction")

    st.markdown("###### ❓ Question:")
    st.write("""
    A region contains 8,000 condominium units.  
    A survey of 100 owners shows a **mean of 1.6** and a **sample standard deviation of 0.8**.  
    Find the **95% confidence interval** for the **true average number of units per owner**, using finite population correction.
    """)

    st.markdown("###### ✅ Solution:")
    st.latex(r"\text{SE} = \frac{0.8}{\sqrt{100}} \cdot \sqrt{1 - \frac{100}{8000}} = 0.0795")
    st.latex(r"\text{Margin of Error} = 1.984 \cdot 0.0795 = 0.1576")
    st.latex(r"1.6 \pm 0.1576 = (1.4424,\ 1.7576)")

    st.success("We are 95% confident that the true average number of units per condo owner is between 1.44 and 1.76.")

    st.markdown("###### 🧠 What is FPC?")
    st.write("""
    The **Finite Population Correction (FPC)** adjusts the standard error when you're sampling **without replacement** from a **finite population**.
    """)

    st.latex(r"\text{FPC} = \sqrt{1 - \frac{n}{N}}")

    st.markdown("This reduces the standard error:")
    st.latex(r"\text{SE}_{\text{adj}} = \frac{s}{\sqrt{n}} \cdot \sqrt{1 - \frac{n}{N}}")

    # Divider


    st.markdown("###### ✅ When to Use It")

    st.markdown("""
    | Condition                               | Use FPC? |
    |----------------------------------------|:--------:|
    | Population size N not known            | ❌ No    |
    | Sample is less than 5% of population   | ❌ No    |
    | Sample is 5% or more of population     | ✅ Yes   |
    | Sampling with replacement              | ❌ No    |
    """)



    st.markdown("###### 🎯 Rule of Thumb:")
    st.markdown("""
    If your sample is **at least 5%** of the population **and** you're sampling **without replacement**,  
    then apply the finite population correction.
    """)
    st.write("")
    st.markdown("##### 2r. Method of Moments (MoM)")
    st.markdown("###### 🧠 What is the Method of Moments?")
    st.markdown(r"""
    The **Method of Moments (MoM)** is a technique to estimate parameters of a distribution by:
    1. Computing **sample moments** (like the sample mean, variance, etc.)
    2. Matching them to the **theoretical moments** (expressed in terms of the unknown parameters)
    3. Solving for the parameters

    """)

    st.latex(r"""
    \mu_k = \mathbb{E}[X^k] \quad \text{is the } k\text{-th theoretical moment}
    """)
    st.latex(r"""
    m_k = \frac{1}{n} \sum_{i=1}^n X_i^k \quad \text{is the } k\text{-th sample moment}
    """)

    st.latex(r"""
    \text{Method of Moments:} \quad m_k = \mu_k(\text{parameters})
    """)

    st.write("Example practice:")
    st.latex(r"""
    \text{Suppose } X_1, X_2, \dots, X_n \sim \text{Exponential}(\lambda)
    """)

    st.latex(r"""
    \text{The population (theoretical) moment is: } \mu_1 = \mathbb{E}[X] = \frac{1}{\lambda}
    """)
    st.latex(r"""
    \mathbb{E}[X] = \int_0^\infty x \cdot \lambda e^{-\lambda x} \, dx 
    = \lambda \int_0^\infty x e^{-\lambda x} \, dx 
    = \lambda \cdot \frac{1}{\lambda^2} 
    = \frac{1}{\lambda}
    """)

    st.latex(r"""
    \text{The sample moment is: } m_1 = \bar{X} = \frac{1}{n} \sum_{i=1}^n X_i
    """)

    st.latex(r"""
    \text{Set } \mu_1 = m_1 \Rightarrow \hat{\lambda}_{\text{MoM}} = \frac{1}{\bar{X}}
    """)
    st.latex(r"""
    \text{Let } \hat{\lambda} = \frac{1}{\bar{X}}, \quad
    \text{then } \bar{X} \sim \mathcal{N}\left( \frac{1}{\lambda}, \frac{1}{n\lambda^2} \right)
    """)

    st.markdown("Calculate confidence interval:")

    st.latex(r"""
    Z = \frac{\bar{X} - \frac{1}{\lambda}}{1 / (\lambda \sqrt{n})}
    = \sqrt{n} \left( \lambda \bar{X} - 1 \right)
    = \sqrt{n} \left( \frac{\lambda}{\hat{\lambda}} - 1 \right)
    """)

    #st.markdown("### ✅ Solve for confidence interval:")

    st.latex(r"""
    P\left( -z < \sqrt{n} \left( \frac{\lambda}{\hat{\lambda}} - 1 \right) < z \right)
    """)

    st.latex(r"""
    \Rightarrow 
    - \frac{z}{\sqrt{n}} < \frac{\lambda}{\hat{\lambda}} - 1 < \frac{z}{\sqrt{n}}
    \quad \Rightarrow \quad
    1 - \frac{z}{\sqrt{n}} < \frac{\lambda}{\hat{\lambda}} < 1 + \frac{z}{\sqrt{n}}
    """)

    st.latex(r"""
    \Rightarrow 
    \hat{\lambda} \left(1 - \frac{z}{\sqrt{n}}\right)
    < \lambda <
    \hat{\lambda} \left(1 + \frac{z}{\sqrt{n}}\right)
    """)

    st.write("Consistency in Probability:")
    st.latex(r"""
    \begin{aligned}
    \text{Let } \hat{\theta}_n &\text{ be an estimator of a parameter } \theta \text{ based on a sample of size } n. \\
    \text{We say that } \hat{\theta}_n &\text{ is consistent in probability if:} \\
    \hat{\theta}_n &\xrightarrow{p} \theta \quad \text{as } n \to \infty \\
    \text{That is, for any } c > 0, \quad &\lim_{n \to \infty} \mathbb{P} \left( \left| \hat{\theta}_n - \theta \right| > c \right) = 0
    \end{aligned}
    """)

    st.markdown("##### 2s. Hypothesis Testing")
    st.markdown("- A **simple hypothesis** specifies **one exact distribution** (all parameters known).")
    st.markdown("- A **composite hypothesis** includes **more than one possible distribution** (some parameters vary).")

    st.latex(r"""
    \begin{aligned}
    H_0 &: \text{Null Hypothesis (e.g., } \mu = \mu_0 \text{)} \\
    H_A &: \text{Alternative Hypothesis (e.g., } \mu \neq \mu_0 \text{, } \mu > \mu_0 \text{, or } \mu < \mu_0 \text{)}
    \end{aligned}
    """)


    st.latex(r"""
    \begin{aligned}
    \text{Type I Error } (\alpha) &: \text{Reject } H_0 \text{ when } H_0 \text{ is true} \\
    \text{Type II Error } (\beta) &: \text{Fail to reject } H_0 \text{ when } H_A \text{ is true}
    \end{aligned}
    """)

    st.latex(r"""
    \begin{aligned}
    \mathbb{P}(\text{Reject } H_0 \mid H_0 \text{ true}) &= \alpha \quad \text{(Type I error)} \\
    \mathbb{P}(\text{Accept } H_0 \mid H_A \text{ true}) &= \beta \quad \text{(Type II error)} \\
    \mathbb{P}(\text{Reject } H_0 \mid H_A \text{ true}) &= 1 - \beta \quad \text{(Power of the test)} \\
    \mathbb{P}(\text{Accept } H_0 \mid H_0 \text{ true}) &= 1 - \alpha
    \end{aligned}
    """)

    st.markdown("###### Example 1:")
    st.latex(r"""
    H_0: \mu = 100 \quad \text{vs} \quad H_A: \mu = 105
    """)

    # Parameters
    mu_0 = 100
    mu_a = 105
    sigma = 15
    n = 36
    alpha = 0.05

    # Calculate standard error
    se = sigma / np.sqrt(n)

    # Critical value under H0
    z_crit = norm.ppf(1 - alpha)
    x_crit = mu_0 + z_crit * se  # rejection region threshold under H0

    # Power calculation: probability of rejecting H0 when mu = 105
    z_power = (x_crit - mu_a) / se
    power = 1 - norm.cdf(z_power)

    #st.markdown("### 📊 Calculation")

    st.latex(fr"""
    \text{{Standard Error: }} \quad SE = \frac{{\sigma}}{{\sqrt{{n}}}} = \frac{{15}}{{\sqrt{{36}}}} = {se:.2f}
    """)

    st.latex(fr"""
    \text{{Critical value for rejection: }} \quad x_{{\text{{crit}}}} = 100 + z_{{0.95}} \cdot SE = {x_crit:.2f}
    """)

    st.latex(fr"""
    \text{{Power}} = \mathbb{{P}}(X > {x_crit:.2f} \mid \mu = 105) = 1 - \Phi\left(\frac{{{x_crit:.2f} - 105}}{{{se:.2f}}}\right)
    = {power:.4f}
    """)

    st.success(fr"✅ Power of the test = {power:.4f}")
    st.markdown("""
    ###### 🔎 Interpretation Guidelines:
    - **0.80** → Standard goal: 80% chance of detecting a true effect  
    - **< 0.5** → Low power: likely to miss the effect  
    - **> 0.9** → High power, but may require more data

    ###### 🧠 Example Interpretation:
    If your test has **power = 0.80**, then **80% of the time** you will detect a real effect when there is one.
    """)
    from scipy.stats import norm
    import numpy as np
    st.write("")
    st.markdown("###### Example 2:")
    st.markdown("###### 🔍 Type I Error")

    st.latex(r"""
    \alpha = \mathbb{P}_{\mu = \mu_0} \left( \bar{X}_n > C \right)
    = \mathbb{P} \left( \frac{\bar{X}_n - \mu_0}{\sigma / \sqrt{n}} > \frac{C - \mu_0}{\sigma / \sqrt{n}} \right)
    = \mathbb{P}(Z > z_\alpha)
    """)
    st.write("")
    st.markdown("##### 2t. Likelihood Function")

    st.latex(r"""
    X_1, X_2, \dots, X_n \overset{iid}{\sim} f_\theta(x)
    """)

    st.latex(r"""
    \text{We test:} \quad
    H_0: \theta = \theta_0 \quad \text{vs.} \quad H_1: \theta = \theta_1
    """)

    st.latex(r"""
    \text{Under } H_0, \text{ the data follows } f_{\theta_0}(x), \quad
    \text{Under } H_1, \text{ the data follows } f_{\theta_1}(x)
    """)


    st.markdown("###### 🔧 Definition")

    st.latex(r"""
    L(\theta) = f(X_1, X_2, \dots, X_n \mid \theta) = \prod_{i=1}^n f(X_i \mid \theta)
    """)


    st.latex(r"""
    \text{The symbol } \prod \text{ means "take the product" of a sequence of values. It's commonly used in likelihood functions when multiplying over observations.}
    """)

    st.latex(r"""
    \text{This expresses the likelihood of observing the sample given a parameter } \theta.
    """)
    st.latex(r"""
    \Lambda(X) = \frac{L(\theta_0)}{L(\theta_1)} 
    = \frac{\prod_{i=1}^n f_{\theta_0}(X_i)}{\prod_{i=1}^n f_{\theta_1}(X_i)}
    """)
    st.latex(r"""
    \text{Reject } H_0 \quad \text{if} \quad \Lambda(X) \leq C
    """)


    st.write("Example:")
    st.latex(r"""
    \text{We observe data: } X_1, \dots, X_n \overset{iid}{\sim} \mathcal{N}(\mu, \sigma^2),\ 
    \text{ where } \sigma^2 \text{ is known and } \mu \text{ is unknown.}
    """)


    #st.markdown("### 🔢 Step 1: Likelihood Function")

    st.latex(r"""
    L(\mu) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi \sigma^2}} 
    \exp\left( -\frac{(X_i - \mu)^2}{2\sigma^2} \right)
    """)

    #st.markdown("### 📉 Step 2: Log-Likelihood")

    st.latex(r"""
    \log L(\mu) = -\frac{n}{2} \log(2\pi \sigma^2) - 
    \frac{1}{2\sigma^2} \sum_{i=1}^n (X_i - \mu)^2
    """)

    #st.markdown("### ✅ Step 3: Maximize")

    st.latex(r"""
    \frac{d}{d\mu} \log L(\mu) = \frac{1}{\sigma^2} \sum_{i=1}^n (X_i - \mu)
    = 0 \quad \Rightarrow \quad \hat{\mu}_{\text{MLE}} = \bar{X}
    """)

    st.latex(r"""
    \text{The MLE of } \mu \text{ is the sample mean: } \hat{\mu} = \bar{X}
    """)
    st.write("MLE is Maximum Likelihood Estimate.")