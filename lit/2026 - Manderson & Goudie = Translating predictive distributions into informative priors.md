#### **Journal of the American Statistical Association**

**[ISSN: 0162-1459 (Print) 1537-274X (Online) Journal homepage: www.tandfonline.com/journals/uasa20](https://www.tandfonline.com/journals/uasa20?src=pdf)**

## **Translating predictive distributions into** **informative priors**


**Andrew A. Manderson & Robert J. B. Goudie**


**To cite this article:** Andrew A. Manderson & Robert J. B. Goudie (26 Jan 2026): Translating
predictive distributions into informative priors, Journal of the American Statistical Association,
[DOI: 10.1080/01621459.2026.2614034](https://www.tandfonline.com/action/showCitFormats?doi=10.1080/01621459.2026.2614034)


**To link to this article:** [https://doi.org/10.1080/01621459.2026.2614034](https://doi.org/10.1080/01621459.2026.2614034)


[View supplementary material](https://www.tandfonline.com/doi/suppl/10.1080/01621459.2026.2614034)


Accepted author version posted online: 26
Jan 2026.


[Submit your article to this journal](https://www.tandfonline.com/action/authorSubmission?journalCode=uasa20&show=instructions&src=pdf)


Article views: 162


[View related articles](https://www.tandfonline.com/doi/mlt/10.1080/01621459.2026.2614034?src=pdf)


[View Crossmark data](http://crossmark.crossref.org/dialog/?doi=10.1080/01621459.2026.2614034&domain=pdf&date_stamp=26%20Jan%202026)


Full Terms & Conditions of access and use can be found at
[https://www.tandfonline.com/action/journalInformation?journalCode=uasa20](https://www.tandfonline.com/action/journalInformation?journalCode=uasa20)


**Translating predictive distributions into informative priors**


Andrew A. Manderson* and Robert J. B. Goudie


MRC Biostatistics Unit, University of Cambridge


*andrew.manderson@mrc-bsu.cam.ac.uk, a.manderson@live.co.uk

## Abstract


“When complex Bayesian models exhibit implausible behaviour, one solution is to assemble
available information into an informative prior. Challenges arise as prior information is often
only available for the observable quantity, or some model-derived marginal quantity, rather
than directly pertaining to the (usually latent) parameters in our model. We propose a method
for translating available prior information, in the form of an elicited distribution for the
observable or model-derived marginal quantity, into an informative joint prior. Our approach
proceeds given a parametric class of prior distributions with as yet undetermined
hyperparameters, and minimises the difference between the supplied elicited distribution and
corresponding prior predictive distribution. We employ a global, multi-stage Bayesian
optimisation procedure to locate optimal values for the hyperparameters. Three examples
illustrate our approach: a cure-fraction survival model, where censoring implies that the
observable quantity is _a priori_ a mixed discrete/continuous quantity; a setting in which prior
information pertains to _R_ 2 - a model-derived quantity; and a nonlinear regression model.”


_Keywords:_ prior specification, multi-objective optimisation, prior predictive checks


### 1 Introduction

Incorporating prior information in Bayesian models is conceptually easy, but in practice
constructing an informative prior is not easy. Formulating priors in accordance with
predictive information obtained via predictive elicitation (O’Hagan et al., 2006) is attractive
due to the widespread availability, reliability (Kadane and Wolfson, 1998) and modelagnostic nature of such information. However it is often unclear how to implement this
approach, particularly for complex, nonlinear, or overparameterised models, for which
informative priors can be essential to exclude model behaviours that conflict with known
properties of the world. In this paper we suppose predictive information is available in the
form of a _target_ prior predictive distribution, and consider how to _translate_ this into a prior
distribution for model parameters, a step that has heretofore received relatively little
attention.


One simple approach to this task is to directly model the elicited quantity. This requires no
translation step. For example, Perepolkin et al. (2024) directly updated elicited information in
light of observations using a Bayesian quantile-parameterised likelihood, but extensions for
models with latent structure are not obvious. For models with simple latent structure,
analytically translating elicited information about an invertible function of the parameters
into a prior may be possible (e.g. Chaloner et al., 1993). Translation is also clear for
conjugate distributions, since the prior predictive distribution determines the prior
hyperparameter values (Percy, 2002). Translation, however, is challenging for arbitrary
(nonconjugate) models (Gribok et al., 2004) due to the analytic intractability of the prior
predictive distribution. Techniques exist for specific models with specific latent structures,
including for logistic regression (Chen et al., 1999), contingency table analyses (Good, 1967)
and hierarchical models (Hem, 2021), but a model-agnostic approach is needed for models
outwith these classes, as noted by Mikkola et al. (2023).


Our approach to translation builds on the idea of predictive checks _Gabry et al., 2019; the_
_“hypothetical future samples” of Winkler, 1967_ and the Bayesian workflow (Gelman et al.,
2020), in which the prior is repeatedly adjusted until there is concordance between the prior
predictive distribution and the elicited predictive information. However, this manual
approach is impractical whenever the relationship between the prior and the distribution of
the observables is muddied by the complexity of the intervening model. A more automated
method is required. Wang et al. (2018) and Thomas et al. (2020) have proposed approaches in
which either regions of observable space or specific realisations are labelled as plausible or
implausible by experts, and then a prior accounting for this information is formed via either
history matching or a “human in the loop” process driven by a Gaussian process model.
Albert et al. (2012) propose a supra-Bayesian approach intended for multiple experts, in
which a Bayesian model is formed for quantiles or probabilities elicited from the experts.
Another approach, and the closest in motivation and methodology to ours, is _Hartmann et al._
_2020; which is partly inspired by da Silva et al., 2019_, which employs a Dirichlet likelihood
for elicited predictive quantiles to handle both elicitation and translation, as well as
uncertainty in elicited quantiles. Our approach is model-agnostic and is fully based around
distributions, meaning uncertainty is directly and intuitively represented. We specify a
suitable, generic loss function between the prior predictive distribution and this target
distribution, and minimise this loss function via a generic, multi-objective global optimisation
process. We implement our methodology in an `R` package `pbbo`
(https://github.com/hhau/pbbo; Supplement S1).


### 2 Methodology

We postulate three properties that we would like our method to satisfy:


_Faithfulness_ A prior is faithful if it accurately encodes the target data distribution provided by
the elicitation subject. Faithfulness is a property of both the procedure employed to obtain the
prior and the model itself, since not all target prior predictive distributions can be encoded by
simple models and prior structures. Faithfulness is related to the definition of _validity_ in
Johnson et al. (2010) and O’Hagan et al. (2006)’s use of _faithful_, but their concerns are
specific to the elicitation process and not to the translational step.


_Uniqueness_ Multiple equally faithful prior distributions may exist in complex models,
meaning we must distinguish between such priors based on properties other than just
faithfulness if a unique prior is desired. For example, if maximising prior uncertainty whilst
retaining faithfulness is desired, then we could choose the prior with the largest marginal
standard deviations (see Section 2.3). Other properties could be used similarly. The challenge
of uniqueness has been noted by Stefan et al. (2022).


_Replicability_ A procedure is replicable if, given the same target, it constructs identical priors
across independent replications. This is unlikely to hold exactly with stochastic algorithms,
meaning it is important to assess.

#### 2.1 Setup


Consider a joint probability distribution for an observable _Y_   and parameters


######  Q, given hyperparameters  L . This distribution has cumulative

distribution function (CDF) P( _Y_,  ∣ ) and prior predictive CDF P( _Y_ ∣  ) for _Y_ . We suppose
a target predictive distribution, with CDF T( _Y_ ), for the observable quantity _Y_ has been
chosen, and that this encapsulates our prior knowledge about _Y_ . Our primary aim is to choose
######  so that the prior predictive P( Y ∣  ) is faithful to the target T( Y ) .


We assume that the target T( _Y_ ) can be described by a (mixture of) standard distributions and
that samples can be drawn from it; but we do not require T( _Y_ ) to be in the same parametric
family as the prior predictive P( _Y_ ∣  ), since this is often unavailable in closed-form. We
recommend choosing T( _Y_ ) using predictive elicitation (Kadane and Wolfson, 1998), in
which an appropriate parametric distribution is fitted to a small number of quantiles (of the
observable quantity) elicited from experts (O’Hagan et al., 2006, chap. 6).


We describe our methodology in a slightly more general setting in which the observable
###### quantity Y is conditional on a covariate X   C, with joint probability distribution CDF

P( _Y_,  ∣, _X_ ) and prior predictive CDF P( _Y_ ∣ , _X_ ) . We assume information has been
elicited about _Y_ at a fixed set of values for _X_ . Specifically we suppose the target CDF

T( _Y_ ∣ _X_ _r_ ) has been elicited at _R_ values of the covariate vector denoted { _X_ _r_ } _r_ _[R]_  1 [, which we ]

stack in the matrix _**X**_   _X_ 1 _X_ _R_    _R_  _C_ . We assume that each target T( _Y_ ∣ _X_ _r_ ) has


identical support to P( _Y_ ∣ , _X_ _r_ ) . We denote


P(  ∣, _**X**_ ) defined analogously.


## ∣ X )   T( Y ∣ X r ), with P( Y ∣ , X ) and

_r_  1



_R_



T( _Y_ ∣ _**X**_ )  T( _Y_ ∣ _X_ _r_ )

_r_  1



_Y_ ∣ _**X**_ )  T( _Y_ ∣ _X_ _r_
_r_  1


#### 2.2 Predictive discrepancy (primary objective)

We quantify the difference between the prior predictive and target by the _covariate-specific_
_predictive discrepancy_, which we define to be



_R_



1 _R_
(  ∣ _**X**_ )  _d_ (P( _Y_ ∣ , _X_ ),T( _Y_ ∣ _X_ ))dT( _Y_ ∣ _X_ ),



∣ _**X**_ )  _d_ (P( _Y_ ∣ , _X_ _r_ ),T( _Y_ ∣ _X_ _r_ ))dT( _Y_ ∣ _X_ _r_ ), (1)
## 

_R_ _r_  1



_D_ (  ∣ _**X**_ )  _d_ (P( _Y_ ∣ , _X_ _r_ ),T( _Y_ ∣ _X_ _r_ ))dT( _Y_ ∣ _X_ _r_
## 
_R_ _r_  1



1



for some discrepancy function _d_ (·,·) . Minimising (1) admits the optimal hyperparameter

 -  min  _D_ (  ∣ _**X**_ ) . The covariate-independent equivalent _D_ (  ) is obtained by setting

_R_  1 and ignoring conditioning on _X_ _r_ .


Many forms of discrepancy function _d_ (·,·) could be adopted, but restricting to proper
scoring rules (Gneiting and Raftery, 2007), which are minimised iff P( _Y_ ∣ , _X_ _r_ )  T( _Y_ ∣  )

for all _Y_ , is intuitive. In this case, if P( _Y_ ∣ , _X_ _r_ ) is flexible enough to exactly match

T( _Y_ ∣ _X_ _r_ ) for some  -, then any such discrepancy will yield the same  - . Differences arise

when P( _Y_ ∣ , _X_ _r_ ) is insufficiently flexible, with discrepancy functions differing in placement
of emphasis.


CDF-based discrepancies are appealing because they are widely-applicable, so inspired by
the Cramér-von Mises (von Mises, 1947) and Anderson-Darling (Anderson and Darling,
1952) distributional tests we define, for CDFs M( _Y_ ) and P( _Y_ ) :


#####  M( Y )  P( Y  )2, d AD (M( Y ),P( Y ))   M( Y )  P( Y 


##### d CvM (M( Y ),P( Y ))   M( Y )  P( Y  )2, d AD (M( Y ),P( Y ))   M( Y )  P( Y  )2 . (2)

P( _Y_ )(1  P( _Y_ ))


##### CvM (M( Y ),P( Y ))   M( Y )  P( Y  )2, d AD (M( Y ),P( Y ))   M( Y )  P( Y  )2 .

P( _Y_ )(1  P( _Y_ ))



The Anderson-Darling (AD) discrepancy _d_ AD places more emphasis than Cramér-von Mises
(CvM) on matching the tails of two CDFs.


Another option is either direction of Kullback-Leibler (KL) divergence,

###### d KL-fwd  KL(M( Y ) ‖ P( Y )), d KL- rev  KL(P( Y ) ‖ M( Y )). (3)


We consider the form of KL divergences detailed in Supplement S4.

#### 2.3 Regularising estimates of  * (secondary objective)


There often are many optimal values  - that yield values of _D_ (  *∣ _**X**_ ) that are practically
indistinguishable (noted by da Silva et al., 2019) but with immensly differing prior
distributions P(  ∣ *, _**X**_ ) . That is, there are many equally faithful priors. This is not


surprising because we are providing information only on _Y_, which is typically of lower
dimension than  . Uniqueness is particularly challenging for models with additive noise,
such as (13); using knowledge of the measurement process to fix the noise prior will usually
be necessary.


To handle more general cases of lack of uniqueness, we define a secondary objective

_N_ (  ∣ _**X**_ ), typically via a function _n_  ( ) with


_N_ (  ∣ _**X**_ )  _n_ (  )dP(  ∣, _**X**_ ). (4)
## 


This objective can be chosen by practitioners to promote or inhibit properties of the prior
predictive P( _Y_ ∣ , _X_ _r_ ) as desired.


As an example, we demonstrate how to encode a preference for maximising prior uncertainty,
as is commonly desired in the absence of contrary prior knowledge. Specifically, given two
estimates of  - which have equivalent values of _D_ (  *∣ _**X**_ ), we prefer the one with the larger

variance for P(  ∣ *, _**X**_ ) . This preference could be encoded in several ways: a simple option
is the (negative) mean of the marginal log standard deviations across the _Q_ components of
######  Q .


### log  SD P(  q ∣, X )  q 

1



_Q_



1 _Q_
## (  ∣ X )   log  SD P(  q ∣, X )  q ,


## ∣ X )   log  SD P(  q ∣, X )  q , (5)



 ∣ _**X**_ )  log SD P(  ∣, _**X**_ )  _q_



_N_ (  ∣ _**X**_ )  log SD P(  _q_ ∣, _**X**_ )  _q_
_Q_ _q_  1







where SDP( _Z_ )[ _[Z]_ ] [ is the standard deviation of ] _[Z]_ [ under distribution ] [P(] _Z_ [)] . Analytic expressions

for SDP(  ∣, _**X**_ )[  _q_ ] can be used if available; or Monte Carlo estimates otherwise.

#### 2.4 Algorithm and optimisation



We jointly minimise (1) and (5) using a multi-objective optimisation algorithm, and obtain a
set of possible  values which comprise the Pareto frontier  {  _l_ }| _l_  1| . This is the set of all
###### “non-dominated” choices for , meaning that no point in  is preferable in both objectives

to any of the remaining points in (Deb, 2001, chap. 2). For each  in we compute the
loss


_L_ (  )  log( _D_ (  ∣ _**X**_ ))  _N_ (  ∣ _**X**_ ), (6)

###### where the value of  0 expresses our relative belief in the importance of the secondary

objective. The optimal value is then  - :  min _L_ (  ) .



This optimum depends on , which will usually be difficult to assess. However, using multiobjective optimisation we can evaluate (6) for any  without needing to redo the
###### optimisation step, and thus plot Pareto frontiers for a wide range of values  coloured by

loss, with the minimum loss point indicated. These can guide our choice of  : we can seek a
value of  with the minimum loss point not on the extreme of the Pareto frontier, since we
would like to balance the two objectives. This approach is particularly useful in settings


where the scales of the two optima differ markedly, which we further discuss in Supplement
S2. Where it is feasible to replicate the optimisation procedure, we can additionally seek a
choice of  that leads to Pareto frontiers with minimal variability across replicates, since this
suggests the optimal solution can be estimated reliably.


We use a two-stage global optimisation process. Our algorithm requires: a method for
sampling P( _Y_ ∣ , _**X**_ ) ; upper and lower limits that render  a compact subset of _L_, due to
our use of global optimisation; and methods to evaluate the log-target CDF log(T( _Y_ ∣ _**X**_ ))
and for drawing samples according to T( _Y_ ∣ _**X**_ ) . The first optimisation stage in our algorithm

considers only _D_ (  ∣ _**X**_ ) to focus on faithfulness, whereas the second stage also considers
_N_ (  ∣ _**X**_ ) to account for uniqueness and replicability. We adopt this approach because

minimising _D_ (  ∣ _**X**_ ) is considerably more challenging than minimising _N_ (  ∣ _**X**_ ) . An
idealised form of this process is illustrated in Figure 1. We briefly describe the algorithm
below; full details are in Supplement S3.


In stage one we minimise _D_ (  ∣ _**X**_ ) using controlled random search 2 (CRS2) with local
mutation (Kaelo and Ali, 2006), which we run for _N_ CRS2 iterations. We use the final optimum

value  -, as well as the _N_ CRS2 trial points, to obtain a design for the next stage. The
###### design comprises values of , and their corresponding values of log( D (  ∣ X )) . A (small)

number of padding points _N_ pad are added to for numerical robustness in stage 2. The


### result is the design   i,log( D (  i ∣ X )) 



_N_  _N_



  _i_,log( _D_ (  _i_ ∣ _**X**_ )) design pad . Whilst CRS2 was not designed to

_i_  1



 _i_,log( _D_ (  _i_ ∣ _**X**_ ))
_i_



design pad



,log( _D_ (  _i_ ∣ _**X**_ ))

_i_  1







minimise noisy functions, empirically it appears robust to small quantities of noise.


Stage one output is then used to initialise stage two, which additionally focuses on
uniqueness and replicability by employing multi-objective Bayesian optimisation (Frazier,
2018) via MSPOT (Zaefferer et al., 2012) to jointly minimise _D_ (  ∣ _**X**_ ) and _N_ (  ∣ _**X**_ ) .
MSPOT uses a separate Gaussian process (GP) approximation to each of the objectives, and
evaluates these approximations at many points from a Latin hypercube design. At each
iteration the best points under the current GP approximations are evaluated using the actual
objectives and used to iteratively improve the approximations. The noisy (in practice) and
computationally expensive nature of our objectives, particularly _D_ (  ∣ _**X**_ ), necessitates an
approach such as MSPOT. Employing GP models for the objectives enables inexpensive
###### screening of values of  that are far from optimal. Moreover, the GP is a flexible yet

data efficient model to use as an approximation and can, through appropriate choice of
kernel, capture correlation or other complex relationships between components of  and the
objective. We use an optional batching technique in stage two because the computational cost
of evaluating the GP grows cubically in the number of points _N_ BO used in its construction.

After _N_ BO iterations, the evaluated points are reduced to their Pareto frontier (Kung et al.,
1975). Note finding the global optimum is not guaranteed by our optimisation strategy
(Mullen, 2014).



To approximate _D_ (  ∣ _**X**_ ) we first approximate the prior predictive CDF P( _Y_ ∣ , _**X**_ ) by



(P) (P)

_**y**_ _r_  ( _ys_, _r_ ) _s_ _[S]_  _[r]_ 1 with _**y**_ _r_ ~ P( _Y_ ∣ , _X_ _r_ ) to form the ECDF



drawing _[S]_ _r_ [ samples ] _**y**_ _r_ (P)


P( _Y_ ∣ , _X_ _r_, _**y**_ _r_ (P) ), given values of  and _X_ _r_ . We then approximate (1), denoted _D_ (  ∣ _**X**_ ),
###### using [I] r [ samples ] ( yi, r ) i [I]  r 1 ~ Q( Y ∣ X r ) drawn from an importance distribution Q( Y ∣ X r ) with


## D (  ∣ X )  1  R 1 [I] r d (P( yi, r ∣ , X r ),T( yi, r ∣ X r )) dT( yi, r ∣ X

_R_ _r_  1 _I_ _i_  1 dQ( _y_ ∣ _X_


## (  ∣ X )  1  R 1 [I] r d (P( yi, r ∣ , X r ),T( yi, r ∣ X r )) dT( yi, r ∣ X r ) .

_R_ _r_  1 _I_ _i_  1 dQ( _y_ ∣ _X_ )



_R_ _[I]_ _r_


## ∣ X )  1  R 1 [I] r d (P( yi, r ∣ , X r ),T( yi, r ∣ X r )) dT( yi, r ∣ X r ) . (7)

_R_ _r_  1 _I_ _r_ _i_  1 dQ( _yi_, _r_ ∣ _X_ _r_ )


##  d (P( yi, r ∣ , X r ),T( yi, r ∣ X r )) i, r r

_r_  1 _I_ _r_ _i_  1 dQ( _yi_, _r_ ∣ _X_ _r_



1 _I_ _r_ _i_  1 _d_ (P( _yi_, _r_ ∣ , _X_ _r_ ),T( _yi_, _r_ ∣ _X_ _r_ )) dQ( _yii_,, _r_



We select Q( _Y_ ∣ _X_ _r_ ) to match the support, using samples from P( _Y_ ∣ , _X_ _r_ ) and T( _Y_ ∣ _X_ _r_ )
so that the numerical evaluation of (7) is reliable and accurate. Supplement S3.4.1 contains
further information on our importance sampling approach. Approximating _N_ (  ∣ _**X**_ ) is
usually straightforward via Monte Carlo, and we denote the corresponding estimate (or
analytic form, if available) by _N_ (  ∣ _**X**_ ) .

#### 2.5 Benchmarking and other empirical considerations


We show results for both the multi-objective approach and a single-objective approach,
which optimises only (1) even in Stage 2. Given  -, we empirically assess faithfulness by
comparing the target distribution T( _Y_ ∣ _X_ _r_ ) and the estimated optimal prior predictive

distribution P( _Y_ ∣  *, _X_ _r_ ) . Replicability and uniqueness are more challenging to disentangle
empirically: without replicability we are unable to conclude whether the multi-objective
optimisation problem admits a unique solution. We will first assess replicability by
examining the stability of the components of the loss in (6) across independent replications of
the optimisation procedure. When the loss is stable across replicates, we will assess
uniqueness by examining whether the optimal prior predictive distribution P( _Y_ ∣  *, _X_ _r_ ) and

prior P(  ∣ *, _**X**_ ) are stable across replicates; stability of both is good evidence of
uniqueness.

### 3 Examples

#### 3.1 Calibrating a cure fraction survival model


Cure models (Amico and Van Keilegom, 2018) are models for survival in which long-term
survival (or “cure”) is plausible, such as when a large fraction of the censored observations
occur after the last observed event time. However, we cannot distinguish in the right tail of
the survival time distribution between censored uncured individuals and genuinely cured
individuals. We suppose here that we possess prior knowledge on the fraction of individuals
likely to be cured, and the distribution of event times amongst the uncured, and seek to
translate this information into a prior. We consider the CDF-based CvM and AD
discrepancies in this example because the target distribution is of mixed discrete/continuous
type (due to censoring). Additionally, we specify a model with a nontrivial correlation
structure, about which we wish to specify an informative prior, which is known to be
challenging.


Target survival time distribution and covariate generation
Suppose that individuals are followed up for an average (but arbitrary) of 21 units of time,
with those who experience the event doing so a long time before the end of follow up.
###### Furthermore, suppose we believe that, a priori, 5% of the patients will be cured, with 0.2%

of events censored due to insufficient follow up.


Consider individuals _n_  1, , _N_ with event times _[Y]_ _n_ [ and censoring times ] _Cn_, such that
_Yn_  (0, _Cn_ ] . A target distribution that is consistent with our beliefs comprises a point mass of

0.05 at _Cn_, and a lognormal distribution with location  LN  log(3) and scale  LN  2 / 3 for
###### Yn  Cn . This choice of lognormal has 99.8% of its mass residing below 21, and thus

produces event times that are “well separated” from the censoring time, as required by a cure
fraction model. Denoting the lognormal CDF with LogNormal( _Y_ ; , 2), we define the target
CDF



_Yn_ ∣ _Cn_ )  0.95 FLN ( _Yn_ ;  _Z_ LN,(  LN )2 )  0.051  _Yn_  _Cn_ , _Yn_  (0, _C_



T( _Yn_ ∣ _Cn_ )  0.95 FLN ( _Yn_ ;  _Z_ LN _n_,(  LN )2 )  0.051  _Yn_  _Cn_ , _Yn_  (0, _Cn_ ],



∣ _Cn_ )  0.95 _n_  0.051  _Y_  _C_ , _Yn_  (0, _Cn_ ], (8)



 _Yn_  _Cn_ 



2
#### where Zn  LogNormal( Cn ;  LN,  LN  ) is the required normalising constant.

###### We simulate data for this example with N  50 individuals, each with 4 correlated

covariates. When we consider the censoring time _Cn_, which also functions as a covariate, we
###### have B  5 covariates (we use B instead of C as in Section 2 for clarity). In line with our

target distribution, simulated censoring times are distributed such that _Cn_ ~ 20  Exp(1) . We
sample a single correlation matrix _**Q**_ ~ LKJ(5) (Lewandowski et al., 2009) and subsequently

covariates **x** _n_ ~ MultiNormal( **0**, _**Q**_ ) . This results in marginally-standardised yet correlated
covariates.

##### 3.1.1 Model


A cure model for survival data, expressed in terms of its survival function, is


_S_ ( _Y_ ∣ _X_,  )  (1  ) _S_ ( _Y_ ∣ _**X**_,  ), (9)


where a proportion  (0,1) of the population are _cured_ and never experience the event of
###### interest. The survival times for the remaining 1  proportion of the population are

distributed according to the _uncured_ survival function _S_ ( _Y_ ∣ _**X**_,  ) . We use the tilde in _**X**_

and  to denote quantities specific to the uncured survival distribution, and denote

 ( , ) to align with our general notation.


Right censoring results in _Yn_  _Cn_ . The censoring indicator  _n_  1  _Yn_  _Cn_  is 0 for right

censored events, and is 1 otherwise. We denote with **x** _n_ the _n_ [th] row of the _N_  ( _B_  1)


covariate matrix _**X**_, which we assume is column-wise standardised. We assume a Weibull
regression model for the uncured event times, with survival function



### S ( Yn ∣ , x n, Cn )  exp  Yn exp  0  x n β , Yn  (0, Cn ] (10)

###### with  ( , 0, β ) . The likelihood, with hazard h Y ( n ∣ , x n, Cn ), for the n [th] individual is


####  (1  ) S ( Yn ∣ , x n, Cn ) h ( Yn ∣ , x n, Cn ) 



(11)



, **x** _n_, _Cn_ )  (1  ) _S_ ( _Yn_ ∣ , **x** _n_, _Cn_ ) _h_ ( _Yn_ ∣ ,


1 



_n_



p( _Yn_ ∣ , **x** _n_, _Cn_ )  (1  ) _S_ ( _Yn_ ∣ , **x** _n_, _Cn_ ) _h_ ( _Yn_ ∣ , **x** _n_, _Cn_ )

1 



_Y_ ∣ , **x** _n_, _C_ )  (1  ) _S_ ( _Y_ ∣ , **x** _n_, _C_ ) _h_ ( _Y_ ∣ , **x** _n_, _C_



**x** _n_, _C_ )  (1  ) _S_ ( _Y_ ∣ , **x** _n_, _C_ ) _h_ ( _Y_ ∣ , **x**







_n_



∣ , **x** _n_, _C_ )  (1  ) _S_ ( _Y_ ∣ , **x** _n_, _C_ ) _h_ ( _Y_ ∣



_n_ _n_ _n_ _n_ _n_ _n_ _n_ _n_ _n_


####  (1  ) S ( Yn ∣ , x n, Cn ) 



  (1 



 (1  ) _S_ ( _Y_ ∣ ,



_S_ ( _Y_ ∣ , **x** _n_, _C_



 (1  ) _S_ ( _Yn_ ∣


1 



_n_ _n_ _n_



∣



**x**



(1  ) _S_ ( _Y_ ∣ , **x** _n_, _C_ ) .


###### In the notation of Section 2, we have Y  ( Yn ) n [N]  1 and X  ( Cn, x n ) n [N]  1, with X including

censoring times because the support of _Y_ ∣ _X_ _r_ depends on _X_ _r_ .


We will seek to identify optimal values of the hyperparamers
######  ( ,, 0, 02, s , ω η,, a , b  ), with  ~ Beta( a , b  ),  ~ Gamma( , ),

 0 ~ Normal(  0, 02 ) and _**β**_ ~ MVSkewNormal( **0**, _**S**_, _**η**_ ), with _**S**_  diag( _s_  ) **Ω** diag( _s_  ) where

_s_  is the prior marginal scale of _**β**_ and **Ω** is parameterised by _**ω**_  (  1, ,  6 )  [ 1,1]6 that
uniquely determine its Cholesky factor. The skewness is necessary to incorporate the
nonlinear relationship between the hazard and the effect of the covariates, and a covariance
structure is used to account for fact that not all the elements of _**β**_ can be large
simultaneously. Further details are in Supplement S5.

##### 3.1.2 Results


We use _[S]_ _r_ [] 104 prior predictive samples and _[I]_ _r_ [] 5 103 importance samples to estimate the

discrepancy; and optimise using _N_ CRS2  2000 CRS2 iterations, followed by _N_ batch  3



batches of Bayesian optimisation with _N_ BO  200 iterations per batch, carrying _N_ design  60
points between batches. We select  by inspecting the Pareto frontiers for

 {0.1,0.2,0.3,0.5,1,2} (Figure 2 and Supplement S5.2). Except for the maximum and
minimum values, which yield minimum loss points on the extremes of the Pareto frontier, the
###### minimum loss point is insensitive to a wide range of  values. We select  0.3 which

simultaneously minimises variability in loss and both objectives.


The values of the loss and discrepancy functions at  - across replicas are tightly distributed
(see Supplement S5.3), which indicates replicability. Across all replicates and discrepancies
the estimated optimal prior predictive distribution is highly faithful to the target, as illustrated
###### for individual n  9 in Figure 3 (other individuals are visually indistinguishable).


Figure 4 displays the marginals of  for each independent replicate. The single objective
approach (i.e. only optimising predictive discrepancy, shown in blue) consistently locates the
degenerate, non-unique solution where all the variation in the uncensored event times is
attributed to the baseline hazard shape  and the intercept  0 : i.e. all the mass for _**β**_ (the


regression coefficients) is close to 0. The combination of  and  0 is far from unique, and

further calculation reveals that only the derived product  exp  0 is uniquely determined.
Given the inter-replicate consistency previously observed in Supplement S5.3 we infer that
the solution is not unique. In the multi-objective approach, there is a preference for the
optima surrounding  7.5,  0  10 ; an improvement in uniqueness over the single
objective approach, but imperfect.


Figure 5 displays the bivariate prior marginal densities for  3 and  4, two representative
elements of _**β**_ . Nonuniqueness is clearly apparent, with both positive and negative marginal
skewness possible. The multi-objective approach suggests a wider distribution for (  3, 4),
as does the CvM discrepancy relative to the AD discrepancy.


Overall, the procedure produces priors that faithfully reflect the target, in a replicable manner.
However, neither multi- or single-objective solutions are unique, particularly for the
covariance structure, with the former closer to unique for  .

#### 3.2 Priors from model-derived quantities


Consider the linear model _Y_  _**Xβ**_  for _n_  _p_ design matrix _**X**_ and _p_ -vector of
coefficients _**β**_ indexed by _j_  1, , _p_, and where the noise  has zero mean and variance

 2 . Suppose information about the fraction of variance explained by the model is available –
from previous similar experiments, or from knowledge of the measurement process – in the
form of a plausible distribution for the coefficient of determination



 2



2  2



 1 _n_  1 _**β**_ _**X**_ _**Xβ**_  2 . (12)



_R_ 2  1 _n_  1 _**β**_ _**X**_ _**Xβ**_  2 .



assuming that the columns of _**X**_ have been centred. Our aim is to use our knowledge of _R_ 2
to set suitable priors for the regression coefficients  conditional on _**X**_ . This idea was the
inspiration for a class of shrinkage priors (Zhang and Bondell, 2018; Zhang et al., 2022), but
we would like to make this idea applicable to a wider selection of prior structures.



We consider three priors for the regression coefficients: a Gaussian prior and two shrinkage
priors. To demonstrate the challenge that noise parameters pose for uniqueness we will
assume  ~ _N_ (0,  2) and seek to select the hyperparameters for  2 ~ InverseGamma( _a_ 1, _b_ 1) .



The Gaussian prior



  2 
 _j_ ~ N  0, 

 



has only one hyperparameter , which controls the ratio



of prior variability due to _**β**_ to that of  . Hence, we denote parameters _**θ**_ GA  ( _**β**_,  2 ), and

seek optimum values for hyperparameters _**λ**_ GA  ( , _a_ 1, _b_ 1) .


The Dirichlet-Laplace prior ( _Dir. Lap._ ) is defined (Bhattacharya et al., 2015) for the _j_ [th]
#### coefficient such that  j ~ Laplace  0,  j , (  1, ,  p ) ~ Dirichlet( , ,  ),


 ~ Gamma( _p_ ,1/ 2) . Smaller values of the single hyperparameter  yield more sparsity in
###### β . Thus we denote λ DL  ( , a 1, b 1) and θ DL  ( β,  2, 1, ,  p, ) .


The regularised horseshoe prior ( _Reg. Horse._ ) (Piironen and Vehtari, 2017) has more
intermediary quantities and less linearity, increasing its flexiblity but making finding optimal
hyperparameter values more challenging. The prior is



















 _j_ ~ Cauchy  (0,1),  2 _j_  _c_ 2 _c_  2  2 _j_ 2 2,  _j_ ~ N(0,  2 2 _j_



,



_j_ ~ Cauchy  (0,1),  2 _j_  _c_ 2  _j_ 2 2 _j_,  _j_ ~ N(0,  2 2 _j_ )







2 _j_  2 _c_ 2  2 _j_ 2 2,  _j_ ~ N(0,  2 2 _j_



 _c_ 2 



_c_



_c_



with Cauchy [] denoting a Cauchy distribution truncated to [0,  ) . Whilst the regularised
horseshoe is carefully designed to make ( _p_ 0, , _s_ 2 ) interpretable and easy to choose, here we

aim to see if we can choose _**λ**_ HS  ( _p_ 0, , _s_ 2, _a_ 1, _b_ 1) to match an informative prior for _R_ 2 . We
###### denote θ HS  ( β,  2, c 2, , 1,  p ) .

##### 3.2.1 Evaluation setup and tuning parameters


To assess each prior’s ability to faithfully encode the information present across a wide
variety of target distribution and assess the uniqueness and replicability of the optimisation
process, we consider sixteen different Beta( _s_ 1, _s_ 2) distributions as our target T( _R_ 2 ), with

{ _s_ 1, _s_ 2}   and chosen to be four exponentially-spaced values between and including
###### 1/ 3 and 3 (i.e. equally-spaced between log(1/ 3) and log(3) ). These values represent a

variety of potential forms of the supplied target predictive distribution for _R_ 2 .

###### We fix n  50 and p  80 with entries in X drawn from a standard Gaussian distribution,

and assess replicability using 10 independent runs for each prior and target. The support 
for the hyperparameters is defined in Supplement S6.1. We use _S_  104 prior predictive
samples, _I_  5 103 importance samples from a Uniform(0,1), and use both _d_ AD and _d_ CvM
as discrepancy functions. We employ _N_ CRS2  1000 iterations, and subsequently perform

both single and multi-objective Bayesian optimisation for _N_ batch  1 batch of _N_ BO  150
iterations, using _N_ design  50 points from the first stage. The single objective approach
illustrates that differences in flexibility between priors also induce differences in uniqueness,
and highlights issues in choosing a prior for the additive noise parameter  2 . Choosing  is
challenging in the multi-objective approach, as its value should depend on the target, the
discrepancy function, and the prior. These dependencies result in 96 possible choices of ,
###### which is an infeasible number of choices to make in this example. Instead we fix  0.5 for

all multi-objective settings. We use the secondary objective (5), except for quantities where
###### the standard deviation is undefined for some , for which we use a robust scale estimator

(Rousseeuw and Croux, 1993).


##### 3.2.2 Results

We first assess replicability. It appears from Figure 6 that both discrepancies are replicable
for the both Gaussian and Dirichlet-Laplace priors. In contrast, the results for the Regularised
Horseshoe prior appear to replicate poorly under the AD discrepancy, but reasonably under
the CvM discrepancy.


We evaluate faithfulness by inspecting the densities p( _R_ 2∣  *) and t( _R_ 2 ) for the various
targets (all distributions in this example have corresponding densities). A selected subset of
the pairs of ( _s_ 1, _s_ 2) values are displayed in Figure 7 (complete results are in Supplement
S6.2). The Gaussian prior is universally poorly faithful. Both shrinkage priors perform better
in cases where one of _[s]_ 1 [ or ] _[s]_ 2 [ is less than 1, with the regularised horseshoe performing better ]

for the _s_ 1  _s_ 2  1 cases. Interestingly, the results are not symmetric in _[s]_ 1 [ and ] _[s]_ 2 [; the ]

Dirichlet-Laplace prior is able to match the _s_ 1  3, _s_ 2  0.69 target well, with many of
regularised horseshoe replicates performing poorly; whilst the relative performance is
reversed for _s_ 1  0.69, _s_ 2  3 (see Supplement Figure 3). There is also perceptibly more
variability in the regularised horseshoe replicates, which suggests the optimisation problem is
more challenging and the predictive discrepancy objective is noisier. The multi-objective
approach generally produces more variable sets of optima, which is expected, as it is a more
difficult optimisation problem but we do not allow it additional computational resources.
There is little visible difference between the CvM and AD discrepancy functions. Finally, as
the values of _[s]_ 1 [ and ] _[s]_ 2 [ increase, the faithfulness of the shrinkage priors generally decreases. ]
Across the full set of simulations, the regularised horseshoe is evidently the most flexible.


To assess uniqueness, we consider estimated optimal hyperparameter values  - in each
replicate. Figure 8 displays the estimates for _s_ 1  3 and _s_ 2  {0.33,0.69,1.44,3}, which
corresponds to the targets in Figure 7. The estimates for  and , for the Gaussian and
Dirichlet-Laplace priors respectively, are consistent across replicates, which suggests the
optima may be unique. This remains true even for targets where the prior is not faithful to the
target, e.g. the Beta(3,3) target. There is more variability in the hyperparameters of the
regularised horseshoe prior. There does appear to be unique solution for  for the

Beta(3,0.33) and Beta(3,0.69) targets, whereas _[p]_ 0 [ and ] _s_ 2 are highly variable across
replicates, which may reflect nonuniqueness or may be due to the lack of replicability
(discussed above) of this optimisation for the regularised horseshoe.


The hyperparameters ( _a_ 1, _b_ 1) for the additive noise variance  2 are highly variable across
replications for almost all prior/target combinations. This reflects the anticipated lack of
uniqueness when incorporating such hyperparameters. It is particularly striking for the
Dirichlet-Laplace prior when _s_ 2  {0.33,0.69}, where we consistently attain faithfulness but

no replicability in estimates for ( _a_ 1, _b_ 1) . These settings are also interesting as the choice of

single or multi-objective approach greatly impacts the optimum values of _[a]_ 1 [ and ] _[b]_ 1 [. ]
Faithfulness of the multi-objective optima, illustrated in Figure 7, are not appreciably worse
than the single objective approach, but the inclusion of  2 into the secondary objective has
resulted in optimal values of _[a]_ 1 [ and ] _[b]_ 1 [ that maximise the dispersion of the marginal prior (i.e. ]


small _[a]_ 1 [ and large ] _[b]_ 1 [). Asymptotic results are also known for the Gaussian prior, and in ]
Supplement S6.3 we further assess replicability by benchmarking against (asymptotically)
‘true’ values.


Our optimisation procedure has minimised log( _D_ (  )) using both the AD and CvM
discrepancy functions. The former places extra emphasis on matching the tails of the target,
and thus the Regularised Horseshoe values in the top row of Figure 6 differ from our
expectations given the results in the top two rows of Figure 7. Take, for example, the

_s_ 1  3, _s_ 2  0.69 case. It is plainly evident from Figure 7 that the regularised horseshoe prior

provides a better fit to the target distribution at _**λ**_ HS*, and yet the corresponding log( _D_ ( _**λ**_ HS* ))
values in the top row of Figure 6 suggest that it is considerably worse that the Gaussian prior
at _**λ**_ GA* . To reconcile this apparent contradiction, we inspect log( _D_ (  )) at the optima
computed using the CvM discrepancy function. These values are displayed in the bottom row
of Figure 6, whose values closely match our expectations given Figure 7. Given the range of
behaviours of p( _R_ 2∣  *) for all the optima, we can conclude that AD more heavily penalises

over-estimation of the tails of p( _R_ 2∣  *) than under-estimation. This does not discount it as
an optimisation objective, but does complicate comparisons between competing priors.


Overall, this example illustrates how information about a model-derived, nonobservable
quantity can be used to form an informative prior. The most flexible shrinkage model (the
regularised horseshoe prior) was almost always the most faithful to the supplied information.
Conversely, the Gaussian prior is the most replicable and unique, but the lack of faithfulness
means it is unsuitable in combination with a Beta prior on _R_ 2 .

#### 3.3 A human-aware prior for a human growth model


Suppose an individual has their height measured at age _tm_ (in years) for _m_  1, , _M_, with

corresponding measurement _ym_ (in centimetres). The first Preece-Baines model (Preece and
Baines, 1978) for human height is the nonlinear regression model


_ym_  _h_ ( _tm_ ;  )  _m_ (13)



 _h_ 1  exp{ _s_ ( _t_  2(  )} _h_ 1  exp{ _h_ 0 ) _s_ ( _t_  )}  _m_, (14)



exp{ _s_ ( _t_  2(  )} _h_ 1  exp{ _h_ 0 ) _s_ ( _t_  )}  _m_,



1  1 0



_m_  )}  exp{ _s_ 1( _tm_  )} _m_



0 _m_ 1



with  _m_ ~ N(0,  _y_ 2) . Some constraints are required to identify this model and ensure its

physical plausibility: specifically, we require 0  _h_ 0  _h_ 1 and 0  _s_ 0  _s_ 1 . To satisfy these

constraints, we parameterise in terms of  _h_ _h_ 1  _h_ 0 and  _s_ _s_ 1  _s_ 0, which results in

( _h_ 0,  _h_, _s_ 0,  _s_ ) all sharing the same positivity constraint. We also constrain  such that
 (min _m_ ( _tm_ ),max _m_ ( _tm_ )) . Even with these constraints the denominator of the fraction can be
very small, yielding negative heights, meaning the model is not plausible for all parameter
values. Furthermore, the model is poorly behaved under a flat prior, so prior information is
required to stabilise and/or regularise the posterior.


We thus seek in this example to specify priors congruent with two specific target prior
###### predictive distributions. We choose LogNormal(  q, [s] q 2) priors for each of the q  1, ,5


#### elements of  ( h 0,  h, s 0,  s, ), and seek optimal values of   q, sq 2 



5
2



  _q_, _sq_ 2
_q_  1



  _q_, _sq_ 2 (see Supplement

_q_  1


###### S7.1 for  ). We fix the prior  y ~ LogNormal(0,0.22) to avoid uniqueness problems

(Section 3.2). We suppose both sex and age (between ages 2 and 18) are uniformly
distributed in our data. We first consider a _covariate-independent_ prior predictive density

t( _Y_ ) with corresponding CDF T( _Y_ ) for height across the entire age-range, derived by
summarising external data. This target (Figure 10) is a mixture of 3 gamma densities
specified to approximate the external data, which is multimodal due to the fact that humans
grow in spurts. We also consider a _covariate-specific_ T( _Y_ ∣ _X_ _r_ ), specifying Gaussian height

distributions at ages _X_ _r_  (2,8,13,18) (see Figure 9, and Supplement S7.2 for details).

##### 3.3.1 Comparison with Hartmann et. al. and tuning parameters


Hartmann et al. (2020) also considered this example, but elicited 6 predictive quantiles at
ages _t_  (0,2.5,10,17.5), as opposed to entire predictive distributions at ages _t_  (2,8,13,18)
as in our covariate-specific approach. We use different ages because the model is stated to be
accurate for ages  2 (Preece and Baines, 1978). Hartmann et al. (2020) also include in their
definition of  a noise parameter; the distribution of this depends on the conditional mean of
the model, that being the likelihood derived from the Weibull distribution adopted by
Hartmann et al. (2020). Finally, Hartmann et al. (2020) elicit quantiles from 5 different users
and report an estimated  - for each user. These estimates (reproduced in Supplement S7.3)
allow us to compare optimal the selected priors p(  ∣ *) .


We obtain  - for both targets using both single- and multi-objective optimisation processes.
We use the CvM discrepancy, and both forward and reverse KL discrepancies (numerical
instability prevented use of the AD discrepancy). We use _S_  5  104 samples from p( _Y_ ∣  )
and likewise _[S]_ _r_ [] 5 104 samples from p( _Y_ ∣ , _X_ _r_ ) for each of the 4 values of _X_ _r_ . We use

_I_  5 103 and _[I]_ _r_ [] 5 103 importance samples for the CvM discrepancy, and the same
number of samples for estimating the relevant Gaussian parameters in the KL approximation.
Lastly, all settings use _N_ CRS2  2000 CRS2 iterations, _N_ batch  5 Bayesian optimisation

batches each of _N_ BO  250 iterations, and carry forward _N_ design  50 points per batch. We
assess replicability using 30 independent runs of each objective/target pair.


For this example, we also assess whether sampling from the posterior distribution associated
with each prior leads to either ‘divergent Hamiltonian Monte Carlo (HMC) transition’ or
‘maximum treedepth’ warnings, when using the default HMC sampler implemented in `Stan`
(Stan Development Team, 2021). We set `adapt_delta = 0.95` and `max_treedepth`
`= 12` to minimise false positives. The frequent occurrence, across replications, of such
warnings under a specific prior is suggestive of either near-degeneracy in the associated
posterior distribution, or a posterior distribution that is too irregular to sample reliably using
the standard HMC algorithm in `Stan` . In both cases, the corresponding prior distribution is
unappealing for practitioners: an alternative prior that induces a more regular posterior that is
easier to sample is likely to be preferred. To heighten the importance of the prior, we


separately consider each of the 93 individuals in the `growth` data in R-package `fda`
(Ramsay et al., 2022). We also include the flat, improper prior as a benchmark.

##### 3.3.2 Results

###### We consider target- and discrepancy-specific ranges  for the multi-objective settings,

and follow our ‘minimum variability across replicates’ heuristic (Section 2.4) to select
optimum  - values (listed in Supplement S7.4). There is notable inter-replicate variability in
the Pareto frontiers at the optimal values  - (Figure 9), due to the stochasticity of our twostage optimisation approach, with some replicates totally dominated by other replicates. The
predictive discrepancies for the corresponding optimal  - values are reasonably, but not
entirely, consistent across replicates (see Supplement S7.6).


Figure 10 displays the target and prior predictive density estimates in the covariateindependent case. The multi-objective replicates are obtained after  - is chosen. The
estimates of  - after introducing the secondary objective are congruent with the single
objective case, but more variable. Both single and multi-objective approaches result in
reasonably, but not entirely, faithful densities for p( _Y_ ∣  *), though the KL-based
discrepancies are notably less faithful. However, most optimum priors seem to accumulate
###### additional probability surrounding Y  h 1  155 (for the CvM discrepancy) or  125 (for the

KL discrepancies), resulting in individual trajectories attaining their adult height _[h]_ 1 [ for ]
younger than expected ages _t_, as shown in Figure 11. Supplement S7.7 assesses faithfulness
to the covariate-specific target: in this case, the reverse-KL exhibits over-concentration
compared to the other discrepancies, which is to be expected Minka (2005).


Figure 11 shows that both the covariate-independent and covariate-specific targets yield
plausible mean growth trajectories for the CvM discrepancy, however only the covariatespecific target does so for the KL-based discrepancies. The covariate-independent priors are
more uncertain, resulting in implausible heights having _a priori_ support. The covariatespecific priors have similar levels of uncertainty across all ages, further suggesting that the
model is too inflexible to simultaneously match all the covariate-specific targets, which have
varying variance. All 5 of the priors from Hartmann et al. (2020), for a narrower uncertainty
interval, are implausible in both shape and width when viewed on this scale. It also seems
unlikely that these priors accurately reflect the information provided by the experts in
Hartmann et al. (2020), but this information is not reported.


The different priors produce a widely ranging proportion of warning messages in `Stan`
(Figure 12). The flat prior produced more warnings than the informative priors, suggesting
the posterior is more challenging to estimate using HMC under a flat prior than under the
informative priors. Within the informative prior approaches, the covariate-specific
approaches (both single and multi-objective) produce fewer warnings than the Hartmann
priors and covariate-independent (CvM discrepancy) approaches, which produce a moderate
number of warnings. Since the covariate-specific priors are the most informative
(Figure∼S7.7), it seems likely that the additional regularisation induced makes the posterior
easier to explore with HMC.

###### Under the flat prior, individual n  26 produced the most warnings, suggesting this case is

the most challenging for the HMC sampler, so we consider the results for this individual in


more detail. For this individual, the priors for _[h]_ 0 [ exhibit substantial variability across ]
replicates (Figure 13). Under both covariate-independent and covariate-specific approaches,
there are two distinct unimodal priors for _[h]_ 0 [ with similar loss, suggesting that ] [T(] _Y_ ∣ _**X**_ ) does
not provided enough information to uniquely determine a prior distribution (see Supplement
S7.9 for all  ; and Supplement S7.8 for a comparison to the KL discrepancy). However both
priors are significantly broader than the Hartmann et. al. priors. The estimated posterior
distributions for _[h]_ 0 [ are highly variable across replicates under the flat prior, suggesting ]
degeneracy or a sufficiently challenging/irregular posterior for the sampler in `Stan` .
Conversely, any single replicate from the informative prior approaches provides enough
regularization to enable reliable posterior sampling, although the posteriors strongly depend
on the prior distribution used, which as noted is not stable under any method here. However,
our priors produce posteriors for [] _s_ [ with almost all mass below 2 (Supplement S7.9); this is ]

desirable, because [] _s_ 2 corresponds to physiologically implausible growth spurts that are
unsupported by the data.


In summary, the priors estimated by our procedure in this example are broadly faithful to the
supplied information, except in the covariate-specific case where model inflexibility prevents
###### matching both t  2 and t  18 targets simultaneously, and in the covariate-independent case

when either KL discrepancy is used. The covariate-specific, multi-objective method appears
the most useful prior, but is arguably over concentrated, which occasionally prevents the
model from fitting the data well, although all our priors successfully regularise the posterior
sufficiently to enable accurate posterior sampling. Our approach does not produce a unique
prior, although the secondary objective leads to a small improvement in uniqueness (see
Supplement S7.9). However, some of this non-uniqueness may be attributable to imperfect
replicability of the optimisation.


### 4 Conclusion

Setting priors for models congruent with our knowledge is often difficult without a method
for translation such as we have proposed. The Preece-Baines model is a typical example, in
which the observable is well understood but the model parameters are not. Similarly we
anticipate our approach will be valuable for model-derived quantities (such as _R_ 2 ), which are
often readily reasoned about but difficult to set priors for.


One limitation of the current work is that we only partly address non-uniqueness, but we
emphasise that our methodology remains valuable in such settings. Specifically, our approach
provides insight into consequences of certain T( _Y_ ∣ _**X**_ ) : it facilitates discovering which
components of  are uniquely determined, and consideration of whether any differences
between T( _Y_ ∣ _X_ _r_ ) and P( _Y_ ∣  *, _X_ _r_ ) are attributable to model inflexibility or an implausible
target. We also have the opportunity to re-assess whether we have information that we could
employ to fix certain components within  (e.g. the fixed prior for the noise in the human
height example). Another limitation of our current work is that global optimisation methods
lack guarantees of finding the global optimum in finite time: results for CRS2 are largely
empirical and results for multi-objective Bayesian optimisation remain a topic of research
(e.g. Chowdhury and Gopalan, 2021). The generalisability of our optimisation process thus
requires further investigation. Finally, the choice of secondary objective also invites future
investigation into alternatives: practitioners may have other principles they wish to encode
into the prior-setting process. Alternative objectives that instead minimise the variation in
only a subset of  whilst maximising the remaining parameters, or objectives that are
functions of the joint distribution of , are avenues for further research.

### Acknowledgments and data availability


We thank Daniela De Angelis and Mevin Hooten for their feedback; and the The Alan Turing
Institute under the UK Engineering and Physical Sciences Research Council (EPSRC)

[EP/N510129/1] and the UK Medical Research Council [MC_UU_00002/2,
MC_UU_00002/20 and MC_UU_00040/04] for support. No original data were generated; the
`fda` package for `R` contains the `growth` data. `R` code for the examples is at
https://gitlab.com/andrew-manderson/pbbo-paper. For the purpose of open access, the author
has applied a Creative Commons Attribution (CC BY) licence to any Author Accepted
Manuscript version arising.

### Disclosure Statement


The authors report there are no competing interests to declare.


### References

Albert, I., Donnet, S., Guihenneuc-Jouyaux, C., Low-Choy, S., Mengersen, K. and Rousseau,
J. (2012), ‘Combining expert opinions in prior elicitation’, _Bayesian Analysis_ **7** (3), 503–532.


Amico, M. and Van Keilegom, I. (2018), ‘Cure models in survival analysis’, _Annual Review_
_of Statistics and Its Application_ **5** (1), 311–342.


Anderson, T. W. and Darling, D. A. (1952), ‘Asymptotic theory of certain “Goodness of Fit”
criteria based on stochastic processes’, _The Annals of Mathematical Statistics_ **23** (2), 193–212.


Bhattacharya, A., Pati, D., Pillai, N. S. and Dunson, D. B. (2015), ‘Dirichlet–Laplace priors
for optimal shrinkage’, _Journal of the American Statistical Association_ **110** (512), 1479–1490.


Chaloner, K., Church, T., Louis, T. A. and Matts, J. P. (1993), ‘Graphical elicitation of a
prior distribution for a clinical trial’, _Journal of the Royal Statistical Society. Series D (The_
_Statistician)_ **42** (4), 341–353.


Chen, M.-H., Ibrahim, J. G. and Yiannoutsos, C. (1999), ‘Prior elicitation, variable selection
and Bayesian computation for logistic regression models’, _Journal of the Royal Statistical_
_Society. Series B (Statistical Methodology)_ **61** (1), 223–242.


Chowdhury, S. R. and Gopalan, A. (2021), No-regret Algorithms for Multi-task Bayesian
Optimization, _in_ ‘Proceedings of The 24th International Conference on Artificial Intelligence
and Statistics’, PMLR, pp. 1873–1881.


da Silva, E. d. S., Kuśmierczyk, T., Hartmann, M. and Klami, A. (2019), ‘Prior specification
via prior predictive matching: Poisson matrix factorization and beyond’, _arXiv:1910.12263_

_[cs, stat]_ .


Deb, K. (2001), _Multi-Objective Optimization Using Evolutionary Algorithms_, 1st ed edn,
John Wiley & Sons, Chichester; New York.


Frazier, P. I. (2018), ‘A tutorial on Bayesian optimization’, _arXiv:1807.02811_ .


Gabry, J., Simpson, D., Vehtari, A., Betancourt, M. and Gelman, A. (2019), ‘Visualization in
Bayesian workflow’, _Journal of the Royal Statistical Society: Series A (Statistics in Society)_
**182** (2), 389–402.


Gelman, A., Vehtari, A., Simpson, D., Margossian, C. C., Carpenter, B., Yao, Y., Kennedy,
L., Gabry, J., Bürkner, P.-C. and Modrák, M. (2020), ‘Bayesian workflow’,
_arXiv:2011.01808 [stat]_ .


Gneiting, T. and Raftery, A. E. (2007), ‘Strictly proper scoring rules, prediction, and
estimation’, _Journal of the American Statistical Association_ **102** (477), 359–378.


Good, I. J. (1967), ‘A Bayesian significance test for multinomial distributions’, _Journal of the_
_Royal Statistical Society: Series B (Methodological)_ **29** (3), 399–418.


Gribok, A. V., Urmanov, A. M., Wesley Hines, J. and Uhrig, R. E. (2004), ‘Backward
specification of prior in Bayesian inference as an inverse problem’, _Inverse Problems in_
_Science and Engineering_ **12** (3), 263–278.


Hartmann, M., Agiashvili, G., Bürkner, P. and Klami, A. (2020), Flexible prior elicitation via
the prior predictive distribution, _in_ ‘Proceedings of the 36th Conference on Uncertainty in
Artificial Intelligence (UAI)’, PMLR, pp. 1129–1138.


Hem, I. G. (2021), Robustifying Bayesian Hierarchical Models Using Intuitive Prior
Elicitation, PhD thesis, Norwegian University of Science and Technology.


Johnson, S. R., Tomlinson, G. A., Hawker, G. A., Granton, J. T., Grosbein, H. A. and
Feldman, B. M. (2010), ‘A valid and reliable belief elicitation method for Bayesian priors’,
_Journal of Clinical Epidemiology_ **63** (4), 370–383.


Kadane, J. and Wolfson, L. J. (1998), ‘Experiences in elicitation’, _Journal of the Royal_
_Statistical Society: Series D (The Statistician)_ **47** (1), 3–19.


Kaelo, P. and Ali, M. M. (2006), ‘Some variants of the controlled random search algorithm
for global optimization’, _Journal of Optimization Theory and Applications_ **130** (2), 253–264.


Kung, H. T., Luccio, F. and Preparata, F. P. (1975), ‘On finding the maxima of a set of
vectors’, _Journal of the ACM_ **22** (4), 469–476.


Lewandowski, D., Kurowicka, D. and Joe, H. (2009), ‘Generating random correlation
matrices based on vines and extended onion method’, _Journal of Multivariate Analysis_
**100** (9), 1989–2001.


Mikkola, P., Martin, O. A., Chandramouli, S., Hartmann, M., Pla, O. A., Thomas, O.,
Pesonen, H., Corander, J., Vehtari, A., Kaski, S., Bürkner, P.-C. and Klami, A. (2023), ‘Prior
knowledge elicitation: The past, present, and future’, _Bayesian Analysis_ pp. 1–33.


Minka, T. (2005), Divergence measures and message passing, Technical Report MSR-TR2005-173, Microsoft Research.


Mullen, K. M. (2014), ‘Continuous global optimization in R’, _Journal of Statistical Software_
**60**, 1–45.


O’Hagan, A., Buck, C., Daneshkhah, A., Eiser, J., Garthwaite, P., Jenkinson, D., Oakley, J.
and Rakow, T. (2006), _Uncertain Judgements: Eliciting Experts’ Probabilities_, Wiley.


Percy, D. F. (2002), ‘Bayesian enhanced strategic decision making for reliability’, _European_
_Journal of Operational Research_ **139** (1), 133–145.


Perepolkin, D., Goodrich, B. and Sahlin, U. (2024), ‘Hybrid elicitation and quantileparametrized likelihood’, _Statistics and Computing_ **34** (1), 11.


Piironen, J. and Vehtari, A. (2017), ‘Sparsity information and regularization in the horseshoe
and other shrinkage priors’, _Electronic Journal of Statistics_ **11** (2), 5018–5051.


Preece, M. and Baines, M. (1978), ‘A new family of mathematical models describing the
human growth curve’, _Annals of Human Biology_ **5** (1), 1–24.


Ramsay, J. O., Graves, S. and Hooker, G. (2022), ‘FDA: Functional data analysis’.


Rousseeuw, P. J. and Croux, C. (1993), ‘Alternatives to the median absolute deviation’,
_Journal of the American Statistical Association_ **88** (424), 1273–1283.


Stan Development Team (2021), ‘RSTAN: The R interface to Stan’.


Stefan, A. M., Evans, N. J. and Wagenmakers, E.-J. (2022), ‘Practical challenges and
methodological flexibility in prior elicitation’, _Psychological Methods_ **27** (2), 177–197.


Thomas, O., Pesonen, H. and Corander, J. (2020), ‘Probabilistic elicitation of expert
knowledge through assessment of computer simulations’, _arXiv:2002.10902 [stat]_ .


von Mises, R. (1947), ‘On the asymptotic distribution of differentiable statistical functions’,
_The Annals of Mathematical Statistics_ **18** (3), 309–348.


Wang, X., Nott, D. J., Drovandi, C. C., Mengersen, K. and Evans, M. (2018), ‘Using history
matching for prior choice’, _Technometrics_ **60** (4), 445–460.


Winkler, R. L. (1967), ‘The assessment of prior distributions in Bayesian analysis’, _Journal_
_of the American Statistical Association_ **62** (319), 776–800.


Zaefferer, M., Bartz-Beielstein, T., Friese, M., Naujoks, B. and Flasch, O. (2012), ‘MSPOT:
Multi-criteria sequential parameter optimization’.


Zhang, Y. and Bondell, H. D. (2018), ‘Variable selection via penalized credible regions with
Dirichlet–Laplace global-local shrinkage priors’, _Bayesian Analysis_ **13** (3).


Zhang, Y. D., Naughton, B. P., Bondell, H. D. and Reich, B. J. (2022), ‘Bayesian regression
using a prior on the model fit: The R2-D2 shrinkage prior’, _Journal of the American_
_Statistical Association_ **117** (538), 862–874.


Figure 1: Illustration of the algorithm, which seeks to match the prior (blue) and target
distribution (red) by optimising  . The initial value  produces a poor match. Stage 1
minimises (1); stage 2 then minimises (6), which increases the variance of _p_  ( ∣ *) .


Figure 2: Pareto frontiers for the cure model with AD discrepancy. Each panel displays the
replicates for each , with minimum loss point marked (  ).


Figure 3: Estimated optimal prior predictive densities p( _Yn_ ∣  *) (red/blue lines and dots) and

target densities t( _Yn_ ∣ _CN_ ) (black lines and crosses).


Figure 4: Estimated optimal prior marginal densities of p(  ∣ *) for each component of  (

 2, ,  4, not shown, are near identical to  1 ). Both axes are truncated for readability.


Figure 5: Contours of the log prior density log(p(  3, 4∣  *)) at the optima. For clarity we
plot only the final 12 replicates, each in a unique colour.


Figure 6: Discrepancy at the optima log( _D_ (  *)) for four target distributions across 10

replicates, with each the mean of 10 evaluations of log( _D_ (  *)) for the same  - .


Figure 7: Optimal prior predictive densities p( _R_ 2∣  *) for the three priors considered, for
selected target densities (black lines). Density values are truncated to [0, 10] for readability.


Figure 8: Optimal values  - for each of the three priors considered. Columns contain
(possibly prior-specific) hyperparameters, with the point colour corresponding to a specific
prior. Each point’s shape corresponds to the combination of discrepancy function and single
or multi-objective approach. The target beta densities (denoted by the row panel titles)
correspond to Figure 7.


Figure 9: Pareto frontiers for the optimum  - for each discrepancy in the **covariate-**
**specific** example. The minimum loss point for each replicate is plotted with  .


Figure 10: The covariate-independent marginal target density t( _Y_ ) (red) and prior predictive

densities p( _Y_ ∣  *) for each of the 30 replicates (blue).


Figure 11: Mean (solid lines) and 95% intervals (grey regions) for the prior mean

p( _h_ ( ; _t_  )∣  *), for covariate-independent and covariate-specific targets in the multi- and
single-objective settings, for all discrepancies; and for the Hartmann priors (with 75%
intervals). The y-axis is truncated to (50,200) . The red lines are 95% intervals for t( _Y_ ∣ _X_ _r_ ) .


Figure 12: Presence/absence of `Stan` warnings for all individuals (columns) in the `growth`
data and replicate prior estimates (rows). Each replicate corresponds to a run of the
optimisation process (CvM discrepancy) and thus a different prior (except for flat).

###### Figure 13: Priors (blue) for [h] 0 [ and corresponding posteriors (red) for individual ] n  26 using

either covariate-independent and covariate-specific targets, for the CvM discrepancy; a flat
prior scenario (prior not shown); and Hartmann et al. (2020).


