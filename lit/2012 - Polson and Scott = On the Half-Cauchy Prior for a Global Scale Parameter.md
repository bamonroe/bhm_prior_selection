Bayesian Analysis (2012) **7**, Number 4, pp. 887–902

# **On the Half-Cauchy Prior for a Global Scale** **Parameter**


Nicholas G. Polson _[∗]_ and James G. Scott _[†]_


**Abstract.** This paper argues that the half-Cauchy distribution should replace the
inverse-Gamma distribution as a default prior for a top-level scale parameter in
Bayesian hierarchical models, at least for cases where a proper prior is necessary.
Our arguments involve a blend of Bayesian and frequentist reasoning, and are
intended to complement the case made by Gelman (2006) in support of folded- _t_
priors. First, we generalize the half-Cauchy prior to the wider class of hypergeometric inverted-beta priors. We derive expressions for posterior moments and
marginal densities when these priors are used for a top-level normal variance in a
Bayesian hierarchical model. We go on to prove a proposition that, together with
the results for moments and marginals, allows us to characterize the frequentist
risk of the Bayes estimators under all global-shrinkage priors in the class. These
results, in turn, allow us to study the frequentist properties of the half-Cauchy
prior versus a wide class of alternatives. The half-Cauchy occupies a sensible middle ground within this class: it performs well near the origin, but does not lead
to drastic compromises in other parts of the parameter space. This provides an
alternative, classical justification for the routine use of this prior. We also consider
situations where the underlying mean vector is sparse, where we argue that the
usual conjugate choice of an inverse-gamma prior is particularly inappropriate,
and can severely distort inference. Finally, we summarize some open issues in the
specification of default priors for scale terms in hierarchical models.


**Keywords:** hierarchical models; normal scale mixtures; shrinkage

## **1 Introduction**


Consider a normal hierarchical model where, for _i_ = 1 _, . . ., p_,


( _yi_ _| βi, σ_ [2] ) _∼_ N( _βi, σ_ [2] )

( _βi_ _| λ_ [2] _, σ_ [2] ) _∼_ N(0 _, λ_ [2] _σ_ [2] )

_p_ ( _λ_ [2] _, σ_ [2] ) = _p_ ( _λ_ [2] ) _p_ ( _σ_ [2] ) _._


This prototype case embodies a general problem in Bayesian inference: how to choose
default priors _p_ ( _λ_ [2] ) and _p_ ( _σ_ [2] ) for top-level variances in a hierarchical model.


The routine use of Jeffreys’ prior for the error variance, _p_ ( _σ_ [2] ) _∝_ _σ_ _[−]_ [2], poses no
practical issues. This is not the case for _p_ ( _λ_ [2] ), however, as the improper prior _p_ ( _λ_ [2] ) _∝_


_∗_ Booth School of Business, Chicago, IL, [nicholas.polson@chicagobooth.edu](mailto:nicholas.polson@chicagobooth.edu)

_†_ McCombs School of Business, University of Texas at Austin, Austin, TX,
[james.scott@mccombs.utexas.edu](mailto:james.scott@mccombs.utexas.edu)


© 2012 International Society for Bayesian Analysis DOI:10.1214/12-BA730


888 _**The**_ _**Half-Cauchy**_ _**Prior**_


_λ_ _[−]_ [2] leads to an improper posterior. This can be seen from the marginal likelihood:



2 exp











_p_ ( _y_ _| λ_ [2] ) _∝_



_p_




�(1 + _λ_ [2] ) _[−]_ [1] 2


_i_ =1




_−_ [1]



2



_p_



_i_ =1



_yi_ [2]
1 + _λ_ [2]



_,_



where we have taken _σ_ [2] = 1 for convenience. This is positive at _λ_ [2] = 0; therefore,
whenever the prior _p_ ( _λ_ [2] ) fails to be integrable at the origin, so too will the posterior. A
number of default choices have been proposed to overcome this issue. A classic reference
is Tiao and Tan (1965); a very recent one is Morris and Tang (2011), who use a flat
prior _p_ ( _λ_ [2] ) _∝_ 1.


We focus on a proposal by Gelman (2006), who studies the class of half- _t_ priors for
the scale parameter _λ_ :




   _p_ ( _λ | d_ ) _∝_ 1 + _[λ]_ [2]

_d_




- _−_ ( _d_ +1) _/_ 2



for a given _d_ . The half- _t_ prior has the appealing property that its density evaluates to
a nonzero constant at _λ_ = 0. This distinguishes it from the usual conjugate choice of
an inverse-gamma prior for _λ_ [2], whose density vanishes at _λ_ = 0. As Gelman (2006)
points out, posterior inference under these priors is no more difficult than it is under an
inverse-gamma prior, using the simple trick of parameter expansion.


These facts lead to a simple, compelling argument against the use of the inversegamma prior for variance terms in models such as that above. Since the marginal
likelihood of the data, considered as a function of _λ_, does not vanish when _λ_ = 0,
neither should the prior density _p_ ( _λ_ ). Otherwise, the posterior distribution for _λ_ will
be inappropriately biased away from zero. This bias, moreover, is most severe near the
origin, precisely in the region of parameter space where the benefits of shrinkage become
most pronounced.


This paper studies the special case of a half-Cauchy prior for _λ_ with three goals in
mind. First, we embed it in the wider class of hypergeometric inverted-beta priors for
_λ_ [2], and derive expressions for the resulting posterior moments and marginal densities.
Second, we derive expressions for the classical risk of Bayes estimators arising from
this class of priors. In particular, we prove a result that allows us to characterize the
improvements in risk near the origin (that is, when _**β**_ is nearly 0 in Euclidean norm)
that are possible using the wider class. Having proven our risk results for all members
of this wider class, we then return to the special case of the half-Cauchy. We find
that the frequentist risk profile of the resulting Bayes estimator is quite favorable: it
is admissible, but otherwise similar to the James–Stein estimator. Therefore Bayesians
can be comfortable using the prior on purely frequentist grounds.


Third, we attempt to provide insight about the use of such priors in situations
where _**β**_ is expected to be sparse. We find that the arguments of Gelman (2006) in
favor of the half-Cauchy are, if anything, amplified in the presence of sparsity, and that
the inverse-gamma prior can have an especially distorting effect on inference for sparse
signals.


_N._ _G._ _Polson_ _and_ _J._ _G._ _Scott_ 889


Overall, our results provide a complementary set of arguments to those of Gelman
(2006) that support the routine use of the half-Cauchy prior: its excellent frequentist
risk properties, and its sensible behavior in the presence of sparsity compared to the
usual conjugate alternative. Bringing all these arguments together, we contend that
the half-Cauchy prior is a sensible default choice for a top-level variance in Gaussian
hierarchical models. We echo the call for it to replace inverse-gamma priors in routine
use (e.g. Spiegelhalter et al. 1994, 2003; Park and Casella 2008), particularly given the
availability of a simple parameter-expanded Gibbs sampler for posterior computation
(Gelman 2006).

## **2 Inverted-beta priors and their generalizations**


Consider the family of inverted-beta priors for _λ_ [2] :


[(1 +] _[ λ]_ [2][)] _[−]_ [(] _[a]_ [+] _[b]_ [)]
_p_ ( _λ_ [2] ) = [(] _[λ]_ [2][)] _[b][−]_ [1] _,_

Be( _a, b_ )


where Be( _a, b_ ) denotes the beta function, and where _a_ and _b_ are positive reals. A
half-Cauchy prior for _λ_ corresponds to an inverted-beta prior for _λ_ [2] with _a_ = _b_ = 1 _/_ 2.
This family also generalizes the robust priors of Strawderman (1971) and Berger (1980);
the normal-exponential-gamma prior of Griffin and Brown (2012); and the horseshoe
prior of Carvalho et al. (2010). The inverted-beta distribution is also known as the
beta-prime or Pearson Type VI distribution. An inverted-beta random variable is equal
in distribution to the ratio of two gamma-distributed random variables having shape
parameters _a_ and _b_, respectively, along with a common scale parameter.


The inverted-beta family is itself a special case of a new, wider class of hypergeometric inverted-beta distributions having the following probability density function:




- _−_ 1
_,_ (1)




            - _s_
_p_ ( _λ_ [2] ) = _C_ _[−]_ [1] ( _λ_ [2] ) _[b][−]_ [1] ( _λ_ [2] + 1) _[−]_ [(] _[a]_ [+] _[b]_ [)] exp _−_
1 + _λ_ [2]



��
_τ_ [2] + [1] _[ −]_ _[τ]_ [ 2]

1 + _λ_ [2]



for _a_ _>_ 0, _b_ _>_ 0, _τ_ [2] _>_ 0, and _s_ _∈R_ . This comprises a wide class of priors leading to
posterior moments and marginals that can be expressed using confluent hypergeometric
functions. In Appendix 5 we give details of these computations, which yield


_C_ = _e_ _[−][s]_ Be( _a, b_ ) Φ1( _b,_ 1 _, a_ + _b, s,_ 1 _−_ 1 _/τ_ [2] ) _,_ (2)


where Φ1 is the degenerate hypergeometric function of two variables (Gradshteyn and
Ryzhik 1965, 9.261). This function can be calculated accurately and rapidly by transforming it into a convergent series of 2 _F_ 1 functions ( § 9.2 of Gradshteyn and Ryzhik 1965;
Gordy 1998), making evaluation of (2) quite fast for most choices of the parameters.


Both _τ_ and _s_ are global scale parameters, and do not control the behavior of _p_ ( _λ_ )
at 0 or _∞_ . The parameters _a_ and _b_ are analogous to those of the beta distribution.
Smaller values of _a_ encourage heavier tails in _p_ ( _β_ ), with _a_ = 1 _/_ 2, for example, yielding
Cauchy-like tails. Smaller values of _b_ encourage _p_ ( _β_ ) to have more mass near the origin,


890 _**The**_ _**Half-Cauchy**_ _**Prior**_


and eventually to become unbounded; _b_ = 1 _/_ 2 yields, for example, _p_ ( _β_ ) _≈_ log(1+1 _/β_ [2] )
near 0.


We now derive expressions for the moments of _p_ ( _**β**_ _|_ **y** _, σ_ [2] ) and the marginal likelihood _p_ ( **y** _| σ_ [2] ) for priors in this family. As a special case, we obtain the posterior mean
for _**β**_ under a half-Cauchy prior on _λ_ .


Given _λ_ [2] and _σ_ [2], the posterior distribution of _**β**_ is multivariate normal, with mean
_m_ and variance _V_ given by




  - 1
_m_ = 1 _−_
1 + _λ_ [2]




- - 1
**y** _,_ _V_ = 1 _−_
1 + _λ_ [2]




_σ_ [2] _._



Define _κ_ = 1 _/_ (1 + _λ_ [2] ). By Fubini’s theorem, the posterior mean and variance of _**β**_ are


_E_ ( _**β**_ _|_ **y** _, σ_ [2] ) = _{_ 1 _−_ _E_ ( _κ |_ **y** _, σ_ [2] ) _}_ **y** (3)

var( _**β**_ _|_ **y** _, σ_ [2] ) = _{_ 1 _−_ _E_ ( _κ |_ **y** _, σ_ [2] ) _}σ_ [2] _,_ (4)


now conditioning only on _σ_ [2] .


It is most convenient to work with _p_ ( _κ_ ) instead:




       - 1        _p_ ( _κ_ ) _∝_ _κ_ _[a][−]_ [1] (1 _−_ _κ_ ) _[b][−]_ [1] _τ_ [2] [+] 1 _−_ _τ_ [1][2]




- - _−_ 1
_κ_ _e_ _[−][κs]_ _._ (5)



The joint density for _κ_ and **y** takes the same functional form:




           - 1           _p_ ( _y_ 1 _, . . ., yp, κ_ ) _∝_ _κ_ _[a][′][−]_ [1] (1 _−_ _κ_ ) _[b][−]_ [1] 1 _−_ [1]
_τ_ [2] [+] _τ_ [2]




- - _−_ 1
_κ_ _e_ _[−][κs][′]_ _,_



with _a_ _[′]_ = _a_ + _p/_ 2, and _s_ _[′]_ = _s_ + _Z/_ 2 _σ_ [2] for _Z_ = [�] _i_ _[p]_ =1 _[y]_ _i_ [2][.] [Hence] [the] [posterior] [for] _[λ]_ [2] [is]
also a hypergeometric inverted-beta distribution, with parameters ( _a_ _[′]_ _, b, τ_ [2] _, s_ _[′]_ ).


Next, the moment-generating function of (5) is easily shown to be

[Φ][1][(] _[b,]_ [ 1] _[, a]_ [ +] _[ b, s][ −]_ _[t,]_ [ 1] _[ −]_ [1] _[/τ]_ [ 2][)]
_M_ ( _t_ ) = _e_ _[t]_ _._

Φ1( _b,_ 1 _, a_ + _b, s,_ 1 _−_ 1 _/τ_ [2] )


See, for example, Gordy (1998). Expanding Φ1 as a sum of 1 _F_ 1 functions (where 1 _F_ 1
denotes the Kummer confluent hypergeometric function) and using the differentiation
rules given in Chapter 15 of Abramowitz and Stegun (1964) yields



( _a_ _[′]_ ) _n_
_E_ ( _κ_ _[n]_ _|_ **y** _, σ_ [2] ) =
( _a_ _[′]_ + _b_ ) _n_



Φ1( _b,_ 1 _, a_ _[′]_ + _b_ + _n, s_ _[′]_ _,_ 1 _−_ 1 _/τ_ [2] )

_._ (6)
Φ1( _b,_ 1 _, a_ _[′]_ + _b, s_ _[′]_ _,_ 1 _−_ 1 _/τ_ [2] )



Combining the above expression with (3) and (4) yields the conditional posterior
mean and variance for _**β**_, given **y** and _σ_ [2] . Similarly, the marginal density _p_ ( **y** _|_ _σ_ [2] )
involves the ratio of prior to posterior normalizing constants:




         _p_ ( **y** _| σ_ [2] ) = (2 _πσ_ [2] ) _[−][p/]_ [2] exp _−_ _[Z]_

2 _σ_ [2]




- Be( _a′, b_ ) Φ1( _b,_ 1 _, a_ _[′]_ + _b, s_ _[′]_ _,_ 1 _−_ 1 _/τ_ [2] )
Be( _a, b_ ) Φ1( _b,_ 1 _, a_ + _b, s,_ 1 _−_ 1 _/τ_ [2] ) _[.]_



Figure 1 shows a simple example of the posterior mean under the half-Cauchy prior
for _λ_ when _p_ = 10, calculated for fixed _σ_ using the results of this section.


_N._ _G._ _Polson_ _and_ _J._ _G._ _Scott_ 891


Truth


Data


Estimate


-2 -1 0 1 2


Figure 1: Ten true means drawn from a standard normal distribution; data from these
means under standard normal noise; posterior means under the half-Cauchy prior for _λ_ .

## **3 Classical risk results**


These priors are useful in situations where standard priors like the inverse-gamma or
Jeffreys’ are inappropriate or ill-behaved. Non-Bayesians will find them useful for generating easily computable shrinkage estimators that have known risk properties. Bayesians
will find them useful for generating computationally tractable priors for a variance parameter. We argue that these complementary but overlapping goals can both be satisfied
for the special case of the half-Cauchy. To show this, we first characterize the risk properties of the Bayes estimators that result from the wider family of priors used for a
normal mean under a quadratic loss. Our analysis indicates that noticeable improvements over the James–Stein estimator are possible near the origin. As Figure 2 shows,
this can be done in several ways: by choosing _a_ large relative to _b_, by choosing _a_ and _b_
both less than 1, by choosing _s_ negative, or by choosing _τ_ _<_ 1. Each of these choices
involves a compromise somewhere else in the parameter space.


We now derive expressions for the classical risk, as a function of _∥_ _**β**_ _∥_, for the resulting
Bayes estimators under hypergeometric inverted-beta priors. Assume without loss of
generality that _σ_ [2] = 1, and let _p_ ( **y** ) = - _p_ ( **y** _|_ _**β**_ ) _p_ ( _**β**_ ) _d_ _**β**_ denote the marginal density
of the data. Following Stein (1981), write the the mean-squared error of the posterior
mean _**β**_ [ˆ] as



_p_



_i_ =1




  
_∂_
_g_ ( **y** )
_∂yi_



_,_



_E_ ( _∥_ _**β**_ [ˆ] _−_ _**β**_ _∥_ [2] ) = _p_ + _E_ **y** _|_ _**β**_





_∥∇_ log _p_ ( **y** ) _∥_ [2] + 2



where _∇_ denotes the gradient operator function and _∥· ∥_ is Euclidean norm. In turn
this can be written as







_E_ ( _∥_ _**β**_ [ˆ] _−_ _**β**_ _∥_ [2] ) = _p_ + 4 _E_ **y** _|_ _**β**_




_∇_ [2][�] _p_ ( **y** )

~~�~~ _p_ ( **y** )



_._



We now state our main result concerning computation of this quantity.


892 _**The**_ _**Half-Cauchy**_ _**Prior**_


**Proposition** **3.1.** _Suppose_ _that_ _**β**_ _∼_ _N_ (0 _, λ_ [2] _I_ ) _,_ _that_ _κ_ = 1 _/_ (1 + _λ_ [2] ) _,_ _and_ _that_ _the_ _prior_
_p_ ( _κ_ ) _is_ _such_ _that_ lim _κ→_ 0 _,_ 1 _κ_ (1 _−_ _κ_ ) _p_ ( _κ_ ) = 0 _._ _Define_




    - 1
_mp_ ( _Z_ ) =



_p_
_κ_ 2 _e_ _[−]_ _[Z]_ 2
0



2 _[κ]_ _p_ ( _κ_ ) _dκ_ (7)



_for_ _Z_ = [�] _i_ _[p]_ =1 _[y]_ _i_ [2] _[,]_ _[and]_ _[let]_ _[g]_ [(] _[Z]_ [) =] _[ E]_ [(] _[κ][ |][ Z]_ [)] _[.]_ _[Then]_ _[as]_ _[a]_ _[function]_ _[of]_ _**[β]**_ _[,]_ _[the]_ _[quadratic]_ _[risk]_
_of_ _the_ _posterior_ _mean_ _under_ _p_ ( _κ_ ) _is_



_E_ ( _∥_ _**β**_ [ˆ] _−_ _**β**_ _∥_ [2] ) = _p_ + 2 _E_ ( _Z|_ _**β**_ )




- - _mp_ +4( _Z_ )
_Z_
_mp_ ( _Z_ )




- 
_−_ _pg_ ( _Z_ ) _−_ _[Z]_ _,_ (8)

2 _[g]_ [(] _[Z]_ [)][2]



_with_ _the_ _expectation_ _taken_ _over_ _the_ _noncentral_ _chi-squared_ _distribution_ _Z_ _∼_ _χ_ [2] ( _∥_ _**β**_ _∥_ [2] ) _,_
_given_ _**β**_ _;_ _and_ _with_




 - _mp_ +4( _Z_ )
_Z_
_mp_ ( _Z_ )




= ( _p_ + _Z_ + 4) _g_ ( _Z_ ) _−_ ( _p_ + 2) _−_ _Eκ|Z_




2 _κ_ (1 _−_ _κ_ ) _[p][′]_ [(] _[κ]_ [)]

_p_ ( _κ_ )




_._ (9)



Proposition 3.1 is useful because it characterizes the risk conditional on _Z_ in terms of
known quantities: the integral _mp_ ( _Z_ ), and the posterior expectation _g_ ( _Z_ ) = _E_ ( _κ_ _|_ _Z_ ).
This reduces the problem to one of evaluating the outer expectation over _Z_, given
_**β**_ . One straightforward way to do so is to simulate draws of _Z_ from a non-central
chi-squared distribution, allowing the inner terms to be evaluated as a function of _Z_ .
An interesting comparison is with George et al. (2006), who consider Kullback–Leibler
predictive risk for similar priors.


Our interest is in the special case _a_ = _b_ = 1 _/_ 2, _τ_ = 1, and _s_ = 0, corresponding
to a half-Cauchy prior for the global scale _λ_ . Figure 3 shows the classical risk of the
Bayes estimator under this prior for _p_ = 7 and _p_ = 15. The risk of the James-Stein
estimator is shown for comparison. These pictures look similar for other values of _p_,
and show overall that the half-Cauchy prior for _λ_ leads to a Bayes estimator that is
competitive with the James–Stein estimator, while retaining admissibility and a fully
Bayesian interpretation.


A natural question is: of all the hypergeometric inverted-beta priors, why choose
the half-Cauchy? There is no iron-clad reason to do so, of course, and we can imagine
many situations where subjective information would support a different choice. But in
examining many other members of the class, we have observed that the half-Cauchy
seems to occupy a sensible middle ground in terms of frequentist risk. To study this, we
are able to appeal to the theory of the previous section. See, for example, Figure 2, which
compares several members of the class for the case _p_ = 7. Observe that large gains over
James–Stein near the origin are possible, but only at the expense of minimaxity. The
half-Cauchy, meanwhile, still improves upon the James–Stein estimator near the origin,
but does not sacrifice good risk performance in other parts of the parameter space.
From a purely classical perspective, it looks like a sensible default choice, appropriate
for repeated general use.


_N._ _G._ _Polson_ _and_ _J._ _G._ _Scott_ 893


**Changing a, p = 7**

|Col1|Col2|Col3|Col4|Col5|Col6|Col7|Col8|Col9|
|---|---|---|---|---|---|---|---|---|
||||||||||
||MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|
||||||||||
||||||||||
|||||||MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|MLE<br>James−Stein<br>a = 3, b = 1, tau = 1, s = 0<br>a = 1, b = 1, tau = 1, s = 0<br>a = 1/8, b = 1, tau = 1, s = 0|
||||||||||
||||||||||



0 2 4 6 8 10 12 14


**Changing s, p = 7**

|Col1|Col2|Col3|
|---|---|---|
||||
||MLE<br>James−Stein<br>a = b = 1/2, tau = 1, s = −5<br>a = b = 1/2, tau = 1, s = 0<br>a = b = 1/2, tau = 1, s = 5|MLE<br>James−Stein<br>a = b = 1/2, tau = 1, s = −5<br>a = b = 1/2, tau = 1, s = 0<br>a = b = 1/2, tau = 1, s = 5|
||||
||||
|||MLE<br>James−Stein<br>a = b = 1/2, tau = 1, s = −5<br>a = b = 1/2, tau = 1, s = 0<br>a = b = 1/2, tau = 1, s = 5|
||||



0 2 4 6 8 10 12 14


Figure 2: Mean-squared error as a function of _∥_ _**β**_ _∥_ [2] for _p_ = 7 and various cases of the
hypergeometric inverted-beta hyperparameters.


894 _**The**_ _**Half-Cauchy**_ _**Prior**_



**P = 7**


0 5 10 15


||Beta||^2



**P = 15**


0 5 10 15 20


||Beta||^2



Figure 3: Mean-squared error as a function of _∥_ _**β**_ _∥_ [2] for _p_ = 7 and _p_ = 15. Solid line:
James–Stein estimator. Dotted line: Bayes estimator under a half-Cauchy prior for _λ_ .

## **4 Global scale parameters in local-shrinkage models**


A now-canonical modification of the basic hierarchical model from the introduction
involves the use of local shrinkage parameters:


( _yi_ _| βi, σ_ [2] ) _∼_ N( _βi, σ_ [2] )

( _βi_ _| λ_ [2] _, u_ [2] _i_ _[, σ]_ [2][)] _∼_ N(0 _, λ_ [2] _σ_ [2] _u_ [2] _i_ [)]

_u_ [2] _i_ _∼_ _f_ ( _u_ [2] _i_ [)]
_λ_ [2] _∼_ _g_ ( _λ_ [2] ) _._


Mixing over _ui_ leads to a non-Gaussian marginal for _βi_, given the global parameter
_λ_ [2] . For example, choosing an exponential prior for each _u_ [2] _i_ [results] [in] [a] [Laplace] [prior,]
used in the Bayesian lasso model (Park and Casella 2008; Hans 2009). This class
of models provides a Bayesian alternative to penalized-likelihood estimation. When
the underlying vector of means is sparse, these global-local shrinkage models can lead
to large improvements in both estimation and prediction compared with pure global
shrinkage rules. There is a large literature on the choice of _p_ ( _u_ [2] _i_ [), with][ Polson and Scott]
(2011) providing a recent review.


As many authors have documented, strong global shrinkage combined with heavytailed local shrinkage is why these sparse Bayes estimators work so well at sifting signals
from noise. Intuitively, the idea is that _λ_ acts as a global parameter that adapts to the
underlying sparsity of the signal. When few signals are present, it is quite common for
the marginal likelihood of **y** as a function of _λ_ to concentrate near 0, and for the signals


_N._ _G._ _Polson_ _and_ _J._ _G._ _Scott_ 895



1.0


0.8


0.6


0.4


0.2


0.0



0.0 0.2 0.4 0.6 0.8 1.0


Lambda



Figure 4: The black line shows the marginal likelihood of the data as a function of _λ_
under a horseshoe prior for each _βi_ . (The likelihood has been renormalized to obtain a
maximum of 1.) The other lines show two priors for _λ_ : the half-Cauchy (dashed line),
and that induced by an inverse-Gamma prior on _λ_ [2] (dotted).


to be flagged via very large values of the local shrinkage parameters _u_ [2] _i_ [.] [Indeed, in some]
cases the marginal maximum-likelihood solution can be the degenerate _λ_ [ˆ] = 0 (Tiao and
Tan 1965).


The classical risk results of the previous section no longer apply to a model with these
extra local-shrinkage parameters, since the marginal distribution of _**β**_, given _λ_, is not
multivariate normal. Nonetheless, the case of sparsity serves only to amplify the purely
Bayesian argument in favor of the half-Cauchy prior for a global scale parameter—
namely, the argument that _p_ ( _λ |_ **y** ) should not be artificially pulled away from zero by
an inverse-gamma prior.


Figure 4 vividly demonstrates this point. We simulated data from a sparse model
where _**β**_ contained the entries (5 _,_ 4 _,_ 3 _,_ 2 _,_ 1) along with 45 zeroes, and where _yij_ _∼_ N(0 _,_ 1)
for _j_ = 1 _,_ 2 _,_ 3. We then used Markov Chain Monte Carlo to compute the marginal likelihood of the data as a function of _λ_, assuming that each _βi_ has a horseshoe prior (Carvalho et al. 2010). This can be approximated by assuming a flat prior for _λ_ (here truncated above at 10), and then computing the conditional likelihood _p_ ( **y** _| λ, σ, u_ [2] 1 _[, . . ., u]_ _p_ [2][)]
over a discrete grid of _λ_ values at each step of the Markov chain. The marginal likelihood function can then be approximated as the pointwise average of the conditional
likelihood over the samples from the joint posterior.


This marginal likelihood has been renormalized to obtain a maximum of 1 and then
plotted alongside two alternatives: a half-Cauchy prior for _λ_, and the prior induced by
assuming that _λ_ [2] _∼_ IG(1 _/_ 2 _,_ 1 _/_ 2). Under the inverse-gamma prior, there will clearly
be an inappropriate biasing of _p_ ( _λ_ ) away from zero, which will negatively affect the


896 _**The**_ _**Half-Cauchy**_ _**Prior**_


ability of the model to handle sparsity. For data sets with even sparse signals, the
distorting effect of a supposedly non-informative inverse-gamma prior will be even more
pronounced, as the marginal likelihood will favor values of _λ_ very near zero (along with
a small handful of very large _u_ [2] _i_ [terms).]

## **5 Discussion**


On strictly Bayesian grounds, the half-Cauchy is a sensible default prior for scale parameters in hierarchical models: it tends to a constant at _λ_ = 0; it is quite heavy-tailed;
and it leads to simple conditionally conjugate MCMC routines, even in complex settings. All these desirable features are summarized by Gelman (2006). Our results give
a quite different, classical justification for this prior in high-dimensional settings: its
excellent quadratic risk properties. The fact that two independent lines of reasoning
both lead to the same prior is a strong argument in its favor as a default proper prior
for a shared variance component. We also recommend scaling the _βi_ ’s by _σ_, as reflected
in the hierachical model from the introduction. This is the approach taken by Jeffreys
(1961, Section 5.2), and we cannot improve upon his arguments.


In addition, our hypergeometric inverted-beta class provides a useful generalization
of the half-Cauchy prior, in that it allows for greater control over global shrinkage
through _τ_ and _s_ . It leads to a large family of estimators with a wide range of possible
behavior, and generalizes the form noted by Maruyama (1999), which contains the
positive-part James–Stein estimator as a limiting, improper case. Further study of this
class may yield interesting frequentist results, quite apart from the Bayesian implications
considered here. The expressions for marginal likelihoods also have connections with
recent work on generalized _g_ -priors (Maruyama and George 2010; Polson and Scott
2012). Finally, all estimators arise from proper priors on _λ_ [2], and will therefore be
admissible.


There are still many open issues in default Bayes analysis for hierarchical models that
are not addressed by our results. One issue is whether to mix further over the scale in
the half-Cauchy prior, _λ ∼_ C [+] (0 _, τ_ ). One possibility here is simply to let _τ_ _∼_ C [+] (0 _,_ 1).
We then get the following “double” half-Cauchy prior for _λ_ :



_dτ_ = [ln] _[ |][λ][|]_

_[λ]_ _τ_ [2][2] [)] _λ_ [2] _−_



_p_ ( _λ_ ) = [2]

_π_ [2]




- _∞_


0



1 1
1 + _τ_ [2] _τ_ (1 + _[λ]_ [2][2]



_λ_ [2] _−_ 1 _[.]_



Admittedly, it is difficult to know where to stop in this “turtles all the way down”
approach to mixing over hyperparameters. (Why not, for example, mix still further
over a scale parameter for _τ_ ?) Even so, this prior has a number of appealing properties.
It is proper, and therefore leads to a proper posterior; it is similar in overall shape to
Jeffreys’ prior; and it is unbounded at the origin, and will therefore not down-weight
the marginal likelihood as much as the half-Cauchy for near-sparse configurations of _**β**_ .
The implied prior on the shrinkage weight _κ_ for the double half-Cauchy is

_p_ ( _κ_ ) _∝_ [ln]             - 1 _−κκ_             - 1 _._

1 _−_ 2 _κ_                 - _κ_ (1 _−_ _κ_ )


_N._ _G._ _Polson_ _and_ _J._ _G._ _Scott_ 897


This is like the horseshoe prior on the shrinkage weight (Carvalho et al. 2010), but with
an extra factor that comes from the fact that one is letting the scale itself be random
with a C [+] (0 _,_ 1) prior.


We can also transform to the real line by letting _ψ_ = log _λ_ [2] . For the half-Cauchy
prior _p_ ( _λ_ ) _∝_ 1 _/_ (1 + _λ_ [2] ) this transformation leads to



_ψ_
_e_ 2
_p_ ( _ψ_ ) _∝_



_ψ_

2 + _e_ _[−]_ _[ψ]_ 2




_._



2   - _ψ_
1 + _e_ _[ψ]_ [=] _e_ 2




_[ψ]_ 2 - _−_ 1 = sech - _ψ_

2



This is the hyperbolic secant distribution, which may provide fertile ground for further
generalizations or arguments involving sensible choices for a default prior.


A more difficult issue concerns the prior scaling for _λ_ in the presence of unbalanced
designs—that is, when _yij_ _∼_ N( _βi, σ_ [2] ) for _j_ = 1 _, . . ., ni_, and the _ni_ ’s are not necessarily
equal. In this case most formal non-informative priors for _λ_, such as the reference priors
for particular parameter orderings, involve complicated functions of the _ni_ ’s (Yang and
Berger 1997). These expressions emerge from the reference-prior formalism that, in
turn, embodies a particular operational definition of “non-informative.”


We have focused on default priors that occupy a middle ground between formal noninformative analysis and pure subjective Bayes. This is clearly an important situation
for the many practicing Bayesians who do not wish to use noninformative priors, whether
for practical, mathematical, or philosophical reasons. An example of a situation in which
formal noninformative priors for _λ_ should not be used on mathematical grounds is when
_**β**_ is expected to be sparse; see Scott and Berger (2006) for a discussion of this issue
in the context of multiple-testing. It is by no means obvious how, or even whether,
the _ni_ ’s should play a role in scaling _λ_ within this (admittedly ill-defined) paradigm of
“default” Bayes.


Finally, another open issue is the specification of default priors for scale parameters
in non-Gaussian models. For example, in logistic regression, the likelihood is highly
sensitive to large values of the underyling linear predictor. It is therefore not clear
whether something so heavy-tailed as the half-Cauchy is an appropriate prior for the
global scale term for logistic regression coefficients. All of these issues merit further
research.

## **References**


Abramowitz, M. and Stegun, I. A. (eds.) (1964). _Handbook_ _of_ _Mathematical_ _Functions_
_With Formulas,_ _Graphs,_ _and Mathematical Tables_, volume 55 of _Applied Mathemat-_
_ics_ _Series_ . Washington, DC: National Bureau of Standards. Reprinted in paperback
by Dover (1974); on-line at `[http://www.math.sfu.ca/~cbm/aands/](http://www.math.sfu.ca/~cbm/aands/)` . 890, 899


Berger, J. O. (1980). “A robust generalized Bayes estimator and confidence region for
a multivariate normal mean.” _The_ _Annals_ _of_ _Statistics_, 8(4): 716–761. 889


Carvalho, C. M., Polson, N. G., and Scott, J. G. (2010). “The horseshoe estimator for
sparse signals.” _Biometrika_, 97(2): 465–80. 889, 895, 897


898 _**The**_ _**Half-Cauchy**_ _**Prior**_


Fourdrinier, D., Strawderman, W., and Wells, M. T. (1998). “On the construction of
Bayes minimax estimators.” _The_ _Annals_ _of_ _Statistics_, 26(2): 660–71. 900


Gelman, A. (2006). “Prior distributions for variance parameters in hierarchical models.”
_Bayesian_ _Analysis_, 1(3): 515–33. 887, 888, 889, 896


George, E. I., Liang, F., and Xu, X. (2006). “Improved minimax predictive densities
under Kullback-Leibler loss.” _The_ _Annals_ _of_ _Statistics_, 34(1): 78–91. 892


Gordy, M. B. (1998). “A generalization of generalized beta distributions.” Finance
and Economics Discussion Series 1998-18, Board of Governors of the Federal Reserve
System (U.S.). 889, 890, 899, 900


Gradshteyn, I. and Ryzhik, I. (1965). _Table of Integrals, Series, and Products_ . Academic
Press. 889, 899


Griffin, J. and Brown, P. (2012). “Alternative prior distributions for variable selection
with very many more variables than observations.” _Australian_ _and_ _New_ _Zealand_
_Journal_ _of_ _Statistics_ . (to appear). 889


Hans, C. M. (2009). “Bayesian Lasso Regression.” _Biometrika_, 96(4): 835–45. 894


Jeffreys, H. (1961). _Theory_ _of_ _Probability_ . Oxford University Press, 3rd edition. 896


Maruyama, Y. (1999). “Improving on the James–Stein estimator.” _Statistics_ _and_ _Deci-_
_sions_, 14: 137–40. 896


Maruyama, Y. and George, E. I. (2010). “ _g_ BF: A Fully Bayes Factor with a Generalized
g-prior.” Technical report, University of Tokyo, arXiv:0801.4410v2. 896


Morris, C. and Tang, R. (2011). “Estimating Random Effects via Adjustment for Density
Maximization.” _Statistical_ _Science_, 26(2): 271–87. 888


Park, T. and Casella, G. (2008). “The Bayesian Lasso.” _Journal_ _of_ _the_ _American_
_Statistical_ _Association_, 103(482): 681–6. 889, 894


Polson, N. G. and Scott, J. G. (2011). “Shrink globally, act locally: sparse Bayesian
regularization and prediction.” In Bernardo, J., Bayarri, M., Berger, J. O., Dawid,
A., Heckerman, D., Smith, A., and West, M. (eds.), _Proceedings_ _of_ _the_ _9th_ _Valencia_
_World_ _Meeting_ _on_ _Bayesian_ _Statistics_ . Oxford University Press. 894


- (2012). “Local shrinkage rules, L´evy processes, and regularized regression.” _Journal_
_of_ _the_ _Royal_ _Statistical_ _Society_ _(Series_ _B)_ . (to appear). 896


Scott, J. G. and Berger, J. O. (2006). “An exploration of aspects of Bayesian multiple
testing.” _Journal_ _of_ _Statistical_ _Planning_ _and_ _Inference_, 136(7): 2144–2162. 897


Spiegelhalter, D. J., Thomas, A., Best, N. G., Gilks, W. R., and Lunn, D. (1994,
2003). _BUGS:_ _Bayesian_ _inference_ _using_ _Gibbs_ _sampling_ . MRC Biostatistics Unit,
Cambridge, England. 889


_N._ _G._ _Polson_ _and_ _J._ _G._ _Scott_ 899


Stein, C. (1981). “Estimation of the mean of a multivariate normal distribution.” _The_
_Annals_ _of_ _Statistics_, 9: 1135–51. 891


Strawderman, W. (1971). “Proper Bayes minimax estimators of the multivariate normal
mean.” _The_ _Annals_ _of_ _Statistics_, 42: 385–8. 889


Tiao, G. C. and Tan, W. (1965). “Bayesian analysis of random-effect models in the
analysis of variance. I. Posterior distribution of variance components.” _Biometrika_,
51: 37–53. 888, 895


Yang, R. and Berger, J. O. (1997). “A Catalog of Noninformative Priors.” Technical
Report 42, Duke University Department of Statistical Science. 897

## **Appendix 1**


In this appendix we present the details for computings moments and marginals of the
hypergeometric-beta distribution. The normalizing constant in (2) is




  - 1
_C_ =



_τ_ [2]




      - 1       0 _κ_ _[α][−]_ [1] (1 _−_ _κ_ ) _[β][−]_ [1] _τ_ [2] [+] 1 _−_ _τ_ [1][2]




- - _−_ 1
_κ_ exp( _−sκ_ ) d _κ ._ (10)



Let _η_ = 1 _−_ _κ_ . Using the identity that _e_ _[x]_ = [�] _m_ _[∞]_ =0 _[x][m][/m]_ [!,] [we] [obtain]



_∞_
_C_ = _e_ _[−][s]_ 

_m_ =0




- _sm_


_m_ !




- 1 
_η_ _[β]_ [+] _[m][−]_ [1] (1 _−_ _η_ ) _[α][−]_ [1] _{_ 1 _−_ (1 _−_ 1 _/τ_ [2] ) _η}_ _[−]_ [1] d _η_ _._
0



Using properties of the hypergeometric function 2 _F_ 1 (Abramowitz and Stegun 1964,
§ 15.1.1 and § 15.3.1), this becomes, after some straightforward algebra,



_∞_



_n_ =0



_C_ = _e_ _[−][s]_ Be( _α, β_ )



_∞_



_m_ =0



( _β_ ) _m_ + _n_

[(1] _[ −]_ [1] _[/τ]_ [ 2][)] _[m][,]_ (11)
( _α_ + _β_ ) _m_ + _n_ _m_ ! _n_ ! _[s][m]_



where ( _a_ ) _n_ is the rising factorial. Appendix C of Gordy (1998) proves that, for all
_α_ _>_ 0, _β_ _>_ 0, and 1 _/τ_ [2] _>_ 0, the nested series in (11) converges to a positive real
number, yielding


_C_ = _e_ _[−][s]_ Be( _α, β_ ) Φ1( _β,_ 1 _, α_ + _β, s,_ 1 _−_ 1 _/τ_ [2] ) _,_ (12)


where Φ1 is the degenerate hypergeometric function of two variables (Gradshteyn and
Ryzhik 1965, 9.261).


The Φ1 function can be written as a double hypergeometric series,



_∞_



_n_ =0



Φ1( _α, β_ ; _γ_ ; _x, y_ ) =



_∞_



_m_ =0



( _α_ ) _m_ + _n_ ( _β_ ) _n_

_[x][m][ .]_ (13)
( _γ_ ) _m_ + _nm_ ! _n_ ! _[y][n]_


900 _**The**_ _**Half-Cauchy**_ _**Prior**_


We use three different representations of Φ1( _α, β, γ, x, y_ ) for handling different combinations of arguments, all from Gordy (1998). When 0 _≤_ _y_ _<_ 1 and _x ≥_ 0,



_x_ _[n]_

2 _[F]_ 1 [(] _[β, α]_ [ +] _[ n]_ [;] _[ γ]_ [ +] _[ n]_ [;] _[ y]_ [)] _[ .]_ (14)
_n_ !



Φ1( _α, β, γ, x, y_ ) =


When 0 _≤_ _y_ _<_ 1 and _x <_ 0,



_∞_



_n_ =0



( _α_ ) _n_
( _γ_ ) _n_



_∞_
Φ1( _α, β, γ, x, y_ ) = _e_ _[x]_      

_n_ =0


Finally, when _y_ _<_ 0,



( _γ −_ _α_ ) _n_

( _γ_ ) _n_



( _−x_ ) _[n]_

2 _[F]_ 1 [(] _[β, α]_ [;] _[ γ]_ [ +] _[ n]_ [;] _[ y]_ [)] _[ .]_ (15)
_n_ !



Φ1( _α, β, γ, x, y_ ) = _e_ _[x]_ (1 _−_ _y_ ) _[−][β]_ Φ1(˜ _α, β, γ, −x,_ ˜ _y_ ) _,_ (16)


where _α_ ˜ = _γ −_ _α_ and _y_ ˜ = _y/_ ( _y −_ 1). Then either (14) or (15) may be used to evaluate
the righthand side of (16), depending on the sign of _x_ .

## **Appendix 2**


To prove Proposition 3.1, we begin with Stein’s decomposition of risk. Following Equation (10) of Fourdrinier et al. (1998), we have




       - 1
_∥∇p_ ( **y** ) _∥_ = _∥_ **y** _∥_




_[Z]_ 2 _[κ]_ d _κ ._



_p_
_κ_ 2 [+1] _p_ ( _κ_ ) _e_ _[−]_ _[Z]_ 2
0



The score can be written as



_∥∇p_ ( **y** ) _∥_



_p_ ( **y** ) _∥_

= _∥_ **y** _∥_ _[m][p]_ [+2][(] _[∥]_ **[y]** _[∥]_ [)]
_p_ ( **y** ) _mp_ ( _∥_ **y** _∥_ )



= _∥_ **y** _∥_ _E_ ( _κ | Z_ ) _,_
_mp_ ( _∥_ **y** _∥_ )



with _mp_ ( _·_ ) defined as in Equation (7). We may also write the Laplacian term in Stein’s
decomposition as




    - 1
∆ _p_ ( **y** ) =




_[Z]_ 2 _[κ]_ d _κ ._



_p_
( _Zκ −_ _p_ ) _κ_ 2 [+1] _p_ ( _κ_ ) _e_ _[−]_ _[Z]_ 2
0



Combining these terms, we have,


∆ _p_ ( **y** )

=
_p_ ( **y** )



_p_

_[ −]_ _[p]_ [)] _[ κ]_ 2 [+1] _p_ ( _κ_ ) _e_ _[−]_ _[Z]_ 2 _[κ]_ d _κ_

�01 _[κ]_ _p_ 2 _p_ ( _κ_ ) _e_ _[−]_ _[Z]_ 2 _[κ]_ d _κ_




_[Z]_ 2 _[κ]_ d _κ_



�01 [(] _[Zκ][ −]_ _[p]_ [)] _[ κ]_ _p_ 2 [+1] _p_ ( _κ_ ) _e_ _[−]_ _[Z]_ 2




_[p]_ [+4][(] _[Z]_ [)]

_−_ _p_ _[m][p]_ [+2][(] _[Z]_ [)]
_mp_ ( _Z_ ) _mp_ ( _Z_ )



= _Z_ _[m][p]_ [+4][(] _[Z]_ [)]



_mp_ ( _Z_ ) _[.]_



The risk term ∆� _p_ ( **y** ) _/_ - _p_ ( **y** ) is then computed using the identity



_∇_ [2][�]



�2 [�]



_p_ ( **y** )
= [1]

- _p_ ( **y** ) 2




∆ _p_ ( **y** )



2



_p_ ( **y** )

_p_ ( **y** ) _[−]_ [1] 2



2




- _∥∇p_ ( **y** ) _∥_

_p_ ( **y** )



_,_


_N._ _G._ _Polson_ _and_ _J._ _G._ _Scott_ 901



which reduces to
1

2


for _g_ ( _Z_ ) = _E_ ( _κ | Z_ ).


Secondly, note that




- _Z_ _[m][p]_ [+4][(] _[Z]_ [)]




_[Z]_ 
2 _[g]_ [(] _[Z]_ [)][2]




_[p]_ [+4][(] _[Z]_ [)]

_−_ _pg_ ( _Z_ ) _−_ _[Z]_
_mp_ ( _Z_ ) 2




            - 1
_Z{mp_ +2( _Z_ ) _−_ _mp_ +4( _Z_ ) _}_ = 2



_p_   _κ_ 2 [+1] (1 _−_ _κ_ ) _p_ ( _κ_ ) _d_ _−e_ _[−]_ _[Z]_ 2
0




_[Z]_ 2 _[κ]_ [�] _._ (17)



Therefore,


 - _mp_ +2( _Z_ ) _−_ _mp_ +4( _Z_ )
_Z_
_mp_ ( _Z_ )




- - 1
=

0




( _p_ + 2)(1 _−_ _κ_ ) _−_ 2 _κ_ + 2 _κ_ (1 _−_ _κ_ ) _[p][′]_ [(] _[κ]_ [)]

_p_ ( _κ_ )




_h_ ( _κ_ ) d _κ_



_p_
_κ_ 2 _e_ _[−]_ _[Z]_ 2 _[κ]_ _p_ ( _κ_ )
_h_ ( _κ_ ) = _._

_mp_ ( _Z_ )



Then under the assumption that lim _κ→_ 0 _,_ 1 _κ_ (1 _−_ _κ_ ) _p_ ( _κ_ ) = 0, integration by parts
gives (9). Hence




2 _κ_ (1 _−_ _κ_ ) _[p][′]_ [(] _[κ]_ [)]

_p_ ( _κ_ )



_E_ ( _∥_ _**β**_ [ˆ] _−_ _**β**_ _∥_ [2] ) = _p_ +2 _EZ|_ _**β**_


**Acknowledgments**




( _Z_ + 4) _g_ ( _Z_ ) _−_ ( _p_ + 2) _−_ _[Z]_

2 _[g]_ [(] _[Z]_ [)][2] _[ −]_ _[E][κ][|][Z]_



��
_._



The authors would like to thank an anonymous referee, the editor, the associate editor, and the

production editor, whose many useful suggestions have helped us to improve the manuscript.


902 _**The**_ _**Half-Cauchy**_ _**Prior**_


