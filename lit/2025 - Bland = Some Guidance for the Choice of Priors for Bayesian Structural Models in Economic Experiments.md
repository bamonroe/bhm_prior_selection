_Journal of the Economic Science Association_ (2025), **11**, 117–126
doi:10.1017/esa.2025.6


**ORIGINAL** **PAPER**

# **Some guidance for the choice of priors for Bayesian** **structural models in economic experiments**


James R. Bland


Department of Economics, The University of Toledo, 2801 Bancroft St, Toledo, Ohio 43606, USA
[Email: james.bland@utoledo.edu](mailto:james.bland@utoledo.edu)


(Received 13 January 2025; revised 30 January 2025; accepted 30 January 2025; first published online 18 July 2025)


**Abstract**
Bayesian estimates from experimental data can be influenced by highly diffuse or “uninformative” priors.
This paper gives examples of how diffuse priors can affect estimates, and discusses how practitioners can
use their expertise to critique and select a prior. [1]


**Keywords:** Bayesian estimation; Prior; Structural estimation; Economic Experiment;


**JEL Codes:** C11; C18; C51; C90


**1.** **Introduction**
It is becoming more common for Bayesian techniques to be used to analyze data from economic
experiments. [2] While their applications often aim to estimate similar parameters to their maximumlikelihood counterparts, such as parameters in utility functions, to estimate a Bayesian model the
practitioner has one additional modeling choice: the prior. Here, we express our beliefs about the
values of our model’s parameters before we observe any data. For Bayesian models, the prior is as
much a part of the model as the likelihood: the likelihood is our formal statement of how our model’s
parameters generate data conditional on parameter values, and the prior is our formal statement of
our beliefs about the values of the model’s parameters before we observe new data.
Although we may have a good idea about what are likely and unlikely values for our model’s parameters, it is tempting to assign highly diffuse priors. If we want our estimates to be “driven mostly by
our data”, and not be too influenced by our priors, then we may be tempted to choose priors that do
not place too much mass on any small region of the parameter’s support. In the limit, this could be a
uniform prior, but in practice this inclination typically manifests itself in the choice of an astronomically large prior variance. In addition to the obvious conflict between such diffuse priors and our
domain expertise (e.g., our understanding of typical parameter values for our subject pool), these
priors can also imply implausible distributions of economically meaningful quantities that we intend
to calculate from our estimates.


1This paper benefited greatly from comments by Glenn Harrison, Lionel Page, and three anonymous referees.
2See for example Alam et al. (2022); Baillon et al. (2020); Balcombe & Fraser 2015; Bland (2019a, 2019b, 2023); Bland &
Rosokha (2021); Gao et al. (2020, 2023; Nilsson et al. (2011); Pachur et al. (2018).


© The Author(s), 2025. Published by Cambridge University Press on behalf of the Economic Science Association. This is an Open Access
[article, distributed under the terms of the Creative Commons Attribution licence (http://creativecommons.org/licenses/by/4.0), which permits](http://creativecommons.org/licenses/by/4.0)
unrestricted re-use, distribution and reproduction, provided the original article is properly cited.


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


118 James R. Bland


**Fig. 1** Draws from the prior distribution of Φ(𝜃), varying the standard deviation of 𝜃


In order to understand the implications of assigning highly diffuse priors, consider estimating a
constant-only probit model:


likelihood: _y_ ∣𝜃∼ Bernoulli (Φ(𝜃))
prior: 𝜃∼ _N_ ( _m_ 𝜃, _s_ 𝜃 [2][)]


where Φ(⋅) is the standard normal cumulative density function, and _m_ 𝜃 and _s_ 𝜃 are the prior mean
and standard deviation of the model’s parameter 𝜃, respectively. Since our data are binary, we might
be interested in reporting the posterior distribution of Φ(𝜃), which is the probability of success in the
Bernoulli process. Suppose that we are happy setting _m_ 𝜃 = 0, which sets the median of Φ(𝜃) to 0.5.
However since we do not want our prior for 𝜃 to influence our estimates too much, we decide to set
_s_ 𝜃 to a large number. True, this will mean that the distribution of 𝜃 will be nicely spread out, but what
does this say about Φ(𝜃), which is the quantity we actually care about? Figure 1 shows 10,000 draws
from this distribution, varying the prior standard deviation _s_ 𝜃. Note that even for modest choices of
this variable, there are large accumulations of probability mass around 0 and 1. A highly diffuse prior
for 𝜃 actually provides _a lot_ of information about Φ(𝜃): it is unlikely that the probability of success
is anything other than 0 or 1. Put differently, if our data-generating process was a series of coin flips,
our prior belief with _s_ 𝜃 = 16 means that we believe we are either flipping a coin with two heads, or a
coin with two tails! [3]

This may seem like a rather convoluted example, however note that many of the structural models
we take to our data are also models of coin-flipping. Our “coin” is a participant’s choice between two
options, and our model describes how this coin is weighted based on conditions of the experiment
(e.g. two lotteries that a participant is choosing between) and parameters that we aim to estimate


3If we in fact suspect that numbers close to zero or one are very likely, then a large value for _s_ 𝜃 may be an appropriate prior.


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


_Journal of the Economic Science Association_ 119


(e.g. a parameter in a utility function). These models often involve non-linear transformations of data
and parameters in ways which, without careful consideration, can make predictions as implausible
as the above example.
Even if we are comfortable with dissonance between our actual priors as experts in the field, and
the priors we use in estimation, highly diffuse priors can greatly influence posterior estimates of
economically meaningful quantities, such as welfare measures (e.g. Monroe, 2023). These quantities are typically not fundamental parameters of our model, but can be easily computed once we have
estimated a model’s parameters. As such, it is unlikely that _any_ prior we choose will have negligible
effect on our posterior estimates.
This calls for the use of the “Principled Bayesian Workflow”, as described by Betancourt (2020)
and demonstrated in Schad et al. (2021). This means, among other things, careful introspection and
testing of our priors, not only as to what they imply about the distribution of our model’s fundamental
parameters, but also what they imply about other, perhaps more important, quantities that we intend
to investigate. [4] For example, our model’s fundamental parameter may describe risk preferences, but
our aim might be to use this parameter to compute a certainty equivalent. We should then investigate
our prior’s implications for both (i) the distribution of risk preferences, and (ii) the distribution of the
certainty equivalents of interest. The former is important because calibrating our prior to plausible
properties of our participants’ preferences will provide us with an appropriate sense of scale for the
distribution of a parameter. The latter is important because it is the inferential goal: we may even be
_more_ interested in this distribution.
Using the task of estimating a popular model of choice under risk as an example, I demonstrate
through simulation that assigning highly diffuse priors to parameters can imply prior distributions
on economically meaningful quantities that all but rule out plausible values. Even moderately diffuse
priors can introduce substantial noise in the point estimates of a model’s structural parameters. Put
differently, for a typical number of decisions made by a participant in an economic experiment, priors
are _always_ going to weigh somewhat in the posterior estimates. It is therefore imperative that we (i)
check that priors over our model’s parameters imply reasonable distributions over the quantities we
care about, and (ii) check that our models, which include the prior, are going to recover the desired
parameters or quantities of interest.


**2.** **Example experiment and model**
For the purposes of this exercise, I will use Harrison & Ng (2016) as an example experiment. In this
experiment, 111 participants made 80 choices over pairs of lotteries. Each lottery is characterized by
three monetary prizes ($10, $30, and $50 for all decisions), and a probability distribution over these
prizes. I use data from the first 80 decisions made in this experiment, which are representative of
many binary choice experiments on decision-making under risk.
Perhaps the most common application of structural models to economic experiments is in estimating risk preferences assuming Expected Utility Theory (EUT). A simple one-parameter specification
for the utility function over certain amounts of money is the constant relative risk-aversion (CRRA)
utility function:

_ui_ ( _x_ ) = 1 _[x]_                             - [1][−] _[r]_ _r_ _[i]_ _i_


Participant _i_ ’s expected utility over lottery ℒ is then:

_Ui_ ( _L_ ) = _E_ [ _ui_ ( _xu_ ) − _i_ ( _Xu_ ) _i_ ( _x_ ) [∣] [ℒ][]]


4See Gelman et al. (2017) for a general discussion of considerations for choosing a prior, and the importance of choosing a
prior.


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


120 James R. Bland


where _X_ ∣ ℒ is the distribution of monetary prizes conditional on lottery ℒ being chosen, and _x_ and
_x_ are the maximum and minimum prizes possible. The denominator of this expression makes the
“contextual utility” (Wilcox, 2011) normalization. [5]

We then combine this with a probabilistic choice rule, which assigns a probability distribution
over the lotteries that participant _i_ has to choose from. A popular choice for this is the (Fechner)
logistic choice rule, or “softmax”, which for a binary choice between lotteries _L_ (eft) and _R_ (ight)
is:

_p_ ( _L_ ∣ _ri_, 𝜆 _i_ ) = exp(𝜆 _iUi_ exp(ℛ)) +(𝜆 _iU_ exp _i_ (ℒ))(𝜆 _iUi_ (ℒ))


Note that the above expression is the likelihood if we were to just observe one decision. So for a
dataset of _T_ decisions made by participant _i_, the likelihood of all of their choices { _yi_, _t_ } _[T]_ _t_ =1 [over lottery]
pairs {ℒ _i_, _t_, ℛ _i_, _t_ } _[T]_ _t_ =1 [is (assuming that each choice is independently distributed):]



_p_ ( _yi_ ∣ _ri_, 𝜆 _i_ ) =



_T_
∏ _p_ ( _yi_, _t_ ∣ _ri_, 𝜆 _i_ )

_t_ =1



The above model and parametric functions, coupled with decisions from an appropriatelydesigned experiment, is all the information we need to estimate _ri_ and 𝜆 _i_ if we were using maximum
likelihood estimation. Bayesian estimation also requires a prior for _ri_ and 𝜆 _i_ . That is, instead of
maximizing _p_ ( _yi_ ∣ _ri_, 𝜆 _i_ ), we wish to determine the distribution:


_p_ ( _ri_, 𝜆 _i_ ∣ _yi_ ) ∝ _p_ ( _yi_ ∣ _ri_, 𝜆 _i_ ) _p_ ( _ri_, 𝜆 _i_ )


and so we must also specify _p_ ( _ri_, 𝜆 _i_ ).
In the next section, I focus on the task of estimating one model for each participant separately.
This is in contrast to focusing on hierarchical models, which are where Bayesian techniques seem
to be making the most headway into analysis of experimental data. This is not because hierarchical
models are impervious to this prior selection problem. [6] Rather, I focus on participant-level estimation
to demonstrate the techniques in the simplest situation where one needs to carefully scrutinize and
calibrate their priors. Furthermore, given the computational burden of hierarchical models, this focus
on individual-level estimation permits a deeper look into the sampling properties of the estimators
under consideration.


**3.** **Considerations for choosing a prior**
_**3.1.**_ _**Priors informed by our understanding of the parameters**_
Our expert understanding of our model’s parameters should help us to form our priors. This
could be possible because the theory behind our model can help us interpret these parameters, or
because we have some knowledge of reasonable values for this parameter from past experiments.
In order to choose our prior, it is useful to break down the decision-making problem into two
parts:


5Aside from being empirically relevant, this normalization also permits thinking about the choice precision parameter 𝜆 _i_,
discussed below, independently of the CRRA parameter _ri_ . This is because normalized utility differences are scaled so they
must be between zero and one in absolute value.
6Consider, for example, assigning a prior to a “hyperparameter” in a hierarchical model that governs the mean of risk
aversion in a CRRA utility function. Choosing a highly diffuse prior for this parameter will imply a prior belief that the mean
level of risk aversion in the population will be well outside the values that we would generally expect with probability close to
one. This problem arises for exactly the same reasons as discussed in the example related to Table 1.


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


_Journal of the Economic Science Association_ 121


1. _What is an appropriate family of distribution for the prior?_ This usually comes down to choosing
a family of distributions with an appropriate support. In this part, we should rule out nonsensical realizations of our parameters, but we should also be careful not to rule out realizations
that are possible, however unlikely.
2. _Within this family of distributions, what distribution should we choose?_


To answer (1), note that any _ri_ ∈ℝ could constitute a valid utility function, so it is important
that we choose a family of distributions that covers the real number line. [7] When this is the case, it is
common to choose the normal family of distributions, so assume a prior as follows:

_ri_ ∼ _N_ ( _mr_, _sr_ [2][)]

For 𝜆 _i_, which measures the choice precision of the probabilistic decision rule, negative values are
nonsensical: this would mean that the participant will choose the lottery that yields _less_ utility more
than 50% of the time. Additionally, we do _not_ want to completely rule out very large positive values of
𝜆 _i_, as these approximate the special case where the participant always chooses the action with greatest
utility. Therefore we seek a family of distributions that covers the positive real numbers. [8] A common
choice in these situations is the log-normal distribution:

log 𝜆 _i_ ∼ _N_ ( _m_ 𝜆, _s_ 𝜆 [2] [)]

At this point, we can move to the second question: choosing the prior parameters _mr_, _sr_, _m_ 𝜆, and
_s_ 𝜆. Start with priors for the risk-aversion parameter _r_ . Since we are using a normal prior for this
parameter, _mr_ is its mean and median, and _sr_ is its standard deviation. We can use our understanding
of previous economic experiments to guide our choices of these parameters. For example Holt &
Laury (2002) (see their Table 3), for their “Low [incentives] real” treatment, place 75% of participants
in the range −0.15 _< r_ _<_ 0.68. Setting _mr_ = 0.27, the midpoint of this region, seems as good a place
as any to start with.
Now consider the scale parameter _sr_ . If we believed that the participants in Holt & Laury (2002) and
our experiment are drawn from a similar population, and that the experiments are similar enough,
we could choose _sr_ to match the 75% of participants who fall in this range, that is: [9]



0.75 = Φ ( [0][.][68][ −] [0][.][27]




[0][.][27]

) −Φ ( [−][0][.][15][ −] [0][.][27]
_sr_ _sr_



)
_sr_



which yields _sr_ = 0.36. However one may be worried that this prior is “too informative”, and hence
wish to choose a larger value of _sr_ . The flip-side of this, choosing _sr_ to be smaller than this, understandably may also lead to problems with our estimator: in this case we would be presuming too
much information about _r_ before observing the data. Taken to its extreme, setting _sr_ → 0 would
negate any reason for running an experiment: we already know a lot about _ri_, and so should spend
our experimental budget elsewhere! But there is also a problem with setting _sr_ too large. To see this,
Table 1 shows the distribution of risk preferences implied by three different prior scale parameters
_sr_, holding location _mr_ constant. Holt & Laury (2002) classify participants into nine groups based on
their risk preferences. This table shows the prior distribution of these categories implied by different
prior standard deviations. Note that even for _sr_ = 3.6, a ten-fold increase from the suggested value,
almost all of the prior is classified into the most extreme categories of “highly risk loving” and “stay
in bed”. That is, by assuming these highly diffuse priors, we are assuming that we will observe very
few participants in the range of risk preferences that experimental economists would typically expect.
Therefore, _ri_ ∼ _N_ (0.27, 0.36 [2] ) is a reasonable choice for the risk aversion parameter.


7This is the case for Harrison & Ng (2016), where the smallest lottery prize is strictly greater than zero. In other experiments
where prizes can be zero, one can either restrict the support of the prior or use a different utility function.
8Bayesians often refer to this as a “weakly informative” prior.
9Balcombe & Fraser (2015) calibrate a prior for a similar utility function parameter so that 75% of the probability mass lies
in the concave, risk-averse range.


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


122 James R. Bland


**Table** **1** Distribution across risk preference categories. The rightmost three columns show the implied prior distribution
accorss these categories for three different prior standard deviations. The “Holt and Laury (2002)” column shows the
estimated distribution from the “Low real” treatment of this experiment



_**3.2.**_ _**Priors informed by our understanding of transformations of the parameters**_
While we may have some good intuition to help us formalize our prior for some parameters, for
others we may need to look elsewhere for guidance. In our example model, one might have trouble
expressing prior uncertainty over the choice precision parameter 𝜆. This parameter may be difficult to
interpret because it is a nuisance parameter (for particular inferential goals): we may not be interested
in estimating it, but it is necessary to include it in the model in order to estimate the parameter(s) we
_do_ care about. Additionally, the units (inverse utility) may be difficult to interpret. As such, we may
know less about what constitutes “reasonable”, “likely” and “unreasonable” values for 𝜆 than we do
for _r_ . Fortunately all structural models with a likelihood necessarily make _predictions_ about behavior,
and we can explore how different values of our parameters affect our model’s predictions.
For 𝜆, the probability of choosing the utility-maximizing is equal to (1 + exp(−𝜆|Δ|)) [−][1], where
Δ is the difference in expected utility between the two lotteries. We can use this to explore the implications of different values of 𝜆 for a given lottery pair. In Figure 2 I plot this probability as a function
of 𝜆 for a choice between $30 for sure, and a 50-50 mix between $10 and $50. This Figure shows how
the prediction changes with 𝜆 for three different values of _r_ . Note that based on our calibrated prior
for _r_, _r_ = 0.27 (dotted line) is the prior mean, and _r_ = −0.09 (solid line) and _r_ = 0.63 (dashed line)
are one prior standard deviation either side of this. Looking at the dotted line, which corresponds
to our calibrated prior mean of _r_ = 0.27, most of the plausible choice probabilities lie in the region
𝜆∈(10, 100). That is for 𝜆 _>_ 100 choices seem very precise, and for 𝜆 _<_ 10 choices seem very noisy.
Given our choice of a log-normal prior for 𝜆, we can calibrate this prior to these endpoints so that
the prior probability that 𝜆∈(10, 100) is (say) 95% as follows:


log 100 = _m_ 𝜆 + 1.96 _s_ 𝜆
log 10 = _m_ 𝜆                        - 1.96 _s_ 𝜆
⟹ _m_ 𝜆 ≈ 3.45, _s_ 𝜆 ≈ 0.59


For perspective, this implies a prior mean of _E_ (𝜆) = exp(3.45 + 0.5 × 0.59 [2] ) ≈ 37, which is shown
as the vertical red line in Figure 2.
Even if we have a good understanding of how to interpret our parameters without transforming
them into predictions or other quantities, investigating the implications of our prior on quantities of
interest can be helpful. For example with an experiment investigating risk preferences like Harrison &
Ng (2016), we may be more interested in what our parameters imply about a certainty equivalent than


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


_Journal of the Economic Science Association_ 123


**Fig. 2** Probability of choosing the utility-maximizing lottery between (i) $30 for sure and (ii) a 50-50 mix of $10 and $50


on the parameters themselves. To this end, in Figure 3 I plot the prior distribution of the certainty
equivalent of a lottery that is a 50-50 mix of $10 and $50. I do this for the calibrated prior (left panel)
and an “inflated” prior (right panel), where I increase the prior standard deviations by a factor of 10
to demonstrate the effect of a highly diffuse prior. For the inflated prior, the certainty equivalent’s
prior distribution has modes close to the endpoints of their allowable values ($10 and $50). This may
not be a desirable implication of our prior, so checking for these implications is important.


_**3.3.**_ _**Priors guided by the performance of our estimator**_
We should also evaluate our prior, coupled with the rest of our model and experiment design, on
how well it achieves our inferential goal. Since our structural model includes a likelihood, we can
simulate data from our experiment for known parameter values, and since we also have a prior, we
can also simulate data from our experiment _consistent with our prior beliefs_ . It is therefore possible,
_before_ running the experiment, to investigate the sampling properties of our estimator. This might
not only identify problems with our prior if they exist, but also provides us with an opportunity to
re-design the experiment if this exercise reveals that we are not estimating our desired quantities as
well as we would like.
To demonstrate this in a Monte Carlo exercise, I draw 1,000 sets of parameters from the calibrated
prior. For each of these draws, I then simulate a dataset using these parameters and estimate the model
four times: (i) with the calibrated prior, (ii) with both prior standard deviations inflated by a factor of
2, (iii) with both prior standard deviations inflated by a factor of 8, and (iv) using maximum likelihood estimation. The root mean squared error of posterior means of the model’s parameters and the


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


124 James R. Bland


**Fig. 3** Prior distributions of a certainty equivalent for the calibrated prior and a prior inflating the standard deviations by a
factor of 10


**Table 2** Root mean squared error of posterior mean estimates and maximum likelihood estimates. The certainty equivalent
is for a lottery that mixes 50-50 over prizes $10 and $50


calibrated ×2 ×8 MLE

Certainty equivalent 0.93 0.99 1.42 12.63

𝜆 12.44 14.17 16.64 36.90

_r_ 0.13 0.14 0.28 56.81


certainty equivalent are shown in Table 2. [10] The maximum likelihood estimator (right-most column)
performs by far the worst of the five estimators under consideration, followed by the estimator using
the most inflated prior. The Bayesian estimator with the calibrated prior estimates these quantities
with smaller error.

Figure 4 shows the estimated certainty equivalents against their true values. A model that recovers
its parameters well will have posterior means (dots) close to the 45 [∘] line (red dashed line). A good
example of this is the top-left panel of this Figure, which shows the results when using the calibrated
prior. Estimates are slightly pulled toward to prior mean, which is to be expected for a Bayesian estimator. Looking at the bottom two panels, we can see that posterior estimates are more noisy (posterior
means are more spread out vertically), less precise (credible regions are larger), and exhibit more bias
(the smoothed means are further away from the 45 [∘] line).


10Further details of the simulation process can be found in Section 1 of the Online Appendix. Section 2 of the Online
Appendix reports the results of a second simulation where simulated datasets are drawn from an inflated prior. This serves
to demonstrate sampling properties of the estimators when the prior is too informative relative to the true distribution of
[parameters. Code to replicate these simulations can be found at https://github.com/JamesBlandEcon/ChoiceOfPriors.](https://github.com/JamesBlandEcon/ChoiceOfPriors)


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


_Journal of the Economic Science Association_ 125


**Fig. 4** Simulated estimates of a certainty equivalent (vertical axis) against their true values (horizontal axis) for various
prior specifications and maximum likelihood estimates. Dots show posterior means, vertical lines show a 50% Bayesian
credible region (25th-75th percentile). No expression of uncertainty is shown for the maximum likelihood estimates. The
red dashed line is a 45 [∘] line. The blue curve shows a smoothed mean of the posterior means


**4.** **Conclusion**
This paper provides guidance for choosing priors when estimating structural models in economic
experiments using Bayesian techniques. The choice of prior is not inconsequential for sample sizes
used in typical economic experiments, and so care must be taken to calibrate this prior so that (i) it
reflects our underlying knowledge of our parameters and other quantities before we observe the data,
and (ii) we are satisfied with our model’s ability to estimate these parameters from our experiment.
However one should not see this as always being in favor of highly informative priors: the exercise of prior calibration should help us with expressing our increased uncertainty without implying
implausible predictions.
Prior calibration plays (at least) two useful functions. Firstly, we formally state and justify what we
believe is the economically meaningful range of our models’ parameters. Secondly, through simulating parameters, then data from the calibrated prior, we explore the implications of our estimators for
datasets which are hopefully similar to those we will see in our experiment. In doing so, we are able
to assess our experiment’s ability to provide us with data that will answer our research question, and
whether we should modify our design before collecting data.
This paper focuses on estimating models of behavior at the individual participant level. However
the techniques it develops need not be limited to these applications. In particular, when estimating models that aggregate information from more than one individual, such as hierarchical models,
the econometrician must choose priors over the model’s population-level “hyperparameters”, rather
than the individual-level parameters. [11] Highly diffuse priors on these hyperparameters can also imply


11For example, see Gao et al. (2023); Gao et al. (2020).


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


126 James R. Bland


implausible prior distributions of individual-level parameters and the quantities of interest that can be
derived from them. Hierarchical models estimated in such studies provide us with suggestions about
what reasonable priors are for individual-level parameters. Using posterior mean estimates from Gao
et al. (2020) suggests that _r_ ∼ _N_ (0.338, 0.268 [2] ) could be an appropriate prior for the CRRA parameter _r_ . [12] This is not too dissimilar from the prior _r_ ∼ _N_ (0.27, 0.36 [2] ) I calibrated to Holt & Laury
(2002), and is in fact _more_ “informative”.


**Supplementary material.** [The supplementary material for this article can be found at https://doi.org/10.1017/esa.2025.6.](https://doi.org/10.1017/esa.2025.6)


**Competing interests.** The author declares no conflicts of interest.


**References**


Alam, J., Georgalos, K., & Rolls, H. (2022). Risk preferences, gender effects and bayesian econometrics. _Journal of Economic_
_Behavior & Organization_, _202_, 168–183.
Baillon, A., Bleichrodt, H., & Spinu, V. (2020). Searching for the reference point. _Management Science_, _66_ (1), 93–112.
Balcombe, K., & Fraser, I. (2015). Parametric preference functionals under risk in the gain domain: a bayesian analysis. _Journal_
_of Risk and Uncertainty_, _50_ (2), 161–87.
[Betancourt, M. 2020. https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html.](https://betanalpha.github.io/assets/case_studies/principled_bayesian_workflow.html)
Bland, J. R. (2019a). How many games are we playing? An experimental analysis of choice bracketing in games. _Journal_ _of_
_Behavioral and Experimental Economics_, _80_ [, 80–91. https://doi.org/10.1016/j.socec.2019.03.011.](https://doi.org/10.1016/j.socec.2019.03.011)
Bland, J. R. (2019b). Measuring and comparing two kinds of rationalizable opportunity cost in mixture models. _Games_,
_11_ (1), 1.
Bland, J. R. (2023). Bayesian inference for quantal response equilibrium in normal-form games. _Games and Economic Behavior._ .
Bland, J. R., & Rosokha, Y. (2021). Learning under uncertainty with multiple priors: experimental investigation. _Journal of Risk_
_and Uncertainty_, _62_ (2), 157–76.
Gao, X. S., Harrison, G. W. and Tchernis, R. (2020). Estimating risk preferences for individuals: a bayesian analysis. _CEAR_
_Working Paper 2020-15_ . Center for the Economic Analysis of Risk, Robinson College of Business, Georgia State University,
[https://cear.gsu.edu/wp-2020-15-estimating-risk-preferences-for-individuals-a-bayesian-approach/).](https://cear.gsu.edu/wp-2020-15-estimating-risk-preferences-for-individuals-a-bayesian-approach/)
Gao, X. S., Harrison, G. W., & Tchernis, R. (2023). Behavioral welfare economics and risk preferences: a bayesian approach.
_Experimental Economics_, _26_ [(2), 273–303. https://doi.org/10.1007/s10683-022-09751-0.](https://doi.org/10.1007/s10683-022-09751-0)
Gelman, A., Simpson, D., & Betancourt, M. (2017). The prior can often only be understood in the context of the likelihood.
_Entropy_, _19_ (10), 555.
Harrison, G. W., & Ng, J. M. (2016). Evaluating the expected welfare gain from insurance. _Journal of Risk and Insurance_, _83_ (1),
91–120.
Holt, C. A., & Laury, S. K. (2002). Risk aversion and incentive effects. _American Economic Review_, _92_ (5), 1644–55.
Monroe, B. A. (2023). The welfare consequences of individual-level risk preference estimation. In G. W. Harrison, & D. Ross
(eds), _Models of Risk Preferences: Descriptive and Normative Challenges_ . Bingley, UK, Emerald.
Nilsson, H., Rieskamp, J., & Wagenmakers, E. -J. (2011). Hierarchical bayesian parameter estimation for cumulative prospect
theory. _Journal of Mathematical Psychology_, _55_ [(1), 84–93. http://dx.doi.org/10.1016/j.jmp.2010.08.006.](http://dx.doi.org/10.1016/j.jmp.2010.08.006)
Pachur, T., Schulte-Mecklenbeck, M., Murphy, R. O., & Hertwig, R. (2018). Prospect theory reflects selective allocation of
attention. _Journal of Experimental Psychology: General_, _147_ (2), 147.
Schad, D. J., Betancourt, M., & Vasishth, S. (2021). Toward a principled bayesian workflow in cognitive science. _Psychological_
_Methods_, _26_ (1), 103.
Wilcox, N. T. (2011). Stochastically more risk averse: a contextual theory of stochastic discrete choice under risk. _Journal of_
_Econometrics_, _162_ (1), 89–104.


12See their Table 5, Specification V.


**Cite** **this** **article:** Bland, J. R. (2025). Some guidance for the choice of priors for Bayesian structural models in economic
experiments. _Journal of the Economic Science Association_, _11_ [(1), 117–126. https://doi.org/10.1017/esa.2025.6](https://doi.org/10.1017/esa.2025.6)


[https://doi.org/10.1017/esa.2025.6 Published online by Cambridge University Press](https://doi.org/10.1017/esa.2025.6)


