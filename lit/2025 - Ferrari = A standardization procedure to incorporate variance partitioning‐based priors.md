Received: 28 April 2025 Revised: 26 September 2025 Accepted: 24 November 2025


DOI: 10.1111/sjos.70042


**O R I G I N A L** **A R T I C L E**

# **A standardization procedure to incorporate** **variance partitioning-based priors in latent** **Gaussian models**



1Department of Economics, University of
Modena & Reggio Emilia, Modena, Italy


2Department of Statistical Sciences,
University of Bologna, Bologna, Italy


**Correspondence**
Luisa Ferrari, Department of Economics,
University of Modena & Reggio Emilia,
Modena, Italy.
Email: luisa.ferrari@unimore.it


**Funding information**
European Union, Grant/Award Number:
2022PA3BS2 (CUP E53D23007580006)



**ABSTRACT**

Latent Gaussian models (LGMs) are a subset of Bayesian


Hierarchical models where Gaussian priors, conditional


on variance parameters, are assigned to all effects in

the model. LGMs are employed in many fields for


their flexibility and computational efficiency. How

ever, practitioners find prior elicitation on the variance

parameters challenging because of a lack of intuitive


interpretation for them. Recently, several papers have


tackled this issue by representing the model in terms

of variance partitioning (VP) and assigning priors to


parameters reflecting the relative contribution of each


effect to the total variance. So far, the class of priors based

on VP has been mainly applied to random effects and


fixed effects separately. This work presents a novel stan

dardization procedure that expands the applicability of

VP priors to a broader class of LGMs, including both


fixed and random effects. The practical advantages of


standardization are demonstrated with simulated data

and a real dataset on survival analysis.


**K E Y W O R D S**


Gaussian Markov Random Fields, hierarchical decomposition priors,

P-splines, PC priors, R2D2



Luisa Ferrari and Massimo Ventrucci equally contributing authors.


[This is an open access article under the terms of the Creative Commons Attribution License, which permits use, distribution and](http://creativecommons.org/licenses/by/4.0/)
reproduction in any medium, provided the original work is properly cited.
© 2025 The Author(s). _Scandinavian Journal of Statistics_ published by John Wiley & Sons Ltd on behalf of The Board of the Foundation of
the Scandinavian Journal of Statistics.


**364** [wileyonlinelibrary.com/journal/sjos](http://wileyonlinelibrary.com/journal/SJOS) _Scand J Statist_ . 2026;53:364–394.


FERRARI and VENTRUCCI **365**


**1** **INTRODUCTION**


Latent Gaussian models (LGMs, Rue et al., 2009) are a popular class of Bayesian Hierarchical
models used by applied scientists in many fields for their flexibility and ability to include linear
and nonlinear effects of continuous covariates, spatial and temporal random effects, and more
complex structures such as interactions. A popular subclass of LGMs assumes that the response
is linked to the latent parameters with a univariate link function, and that the linear predictor for
observations _i_ = 1 _,_ … _, N_ is structured as:



∑ _R_

_**D**_ _[T]_ _r_ [(] _[z]_ _ir_ [)] _**[u]**_ _r_ _[,]_ (1)
_r_ =1



_𝜂i_ = _𝜇_ +



∑ _P_

_xip𝛽p_ +
_p_ =1



where _𝛽p_ ∼ _N_ (0 _, 𝜏p_ [2][)] _[,][ p]_ [ =][ 1] _[,]_ […] _[,][ P]_ [are] [linear] [effects] [of] _[P]_ [covariates] _[x]_ [1] _[,]_ […] _[,][ x][P]_ [(usually] [called] [fixed]
effects), and _**u**_ _r_ ∼ _NKr_ ( **0** _, 𝜎r_ [2] _**[Q]**_ _r_ [−][)] _[,][ r]_ [=][ 1] _[,]_ […] _[ R]_ [ (] _**[Q]**_ [−] _r_ [denotes here the generalized inverse of] _**[ Q]**_ _r_ [) repre-]
sent other types of effects (usually called random) for additional covariates _z_ 1 _,_ … _, zR_ . The vector
_**D**_ _r_ (⋅) = [ _Dr,_ 1(⋅) _,_ … _, Dr,Kr_ (⋅)] _[T]_ contains _Kr_ known basis functions. _𝜏p_ [2] [and] _[𝜎]_ _r_ [2] [are] [typically] [called]
variance or scale parameters, while _**Q**_ _r_ is usually called the structure matrix, and it is either
fully known or controlled by additional correlation parameters. The model in Equation (1) can
embrace many types of random effects, including group effects or splines. Along with proper
Gaussian processes, LGMs can include Intrinsic Gaussian Markov Random Fields (IGMRFs, Rue
& Held, 2005; Fahrmeir et al., 2004), which are characterized by a rank-deficient _**Q**_ _r_, such as the
random walk of order 1 (RW1) and 2 (RW2), and the intrinsic conditional autoregressive process
(ICAR, Besag & Kooperberg, 1995), often employed as priors for spatial and temporal random
effects.
The traditional approach to prior specification of the variance parameters sets the _𝜏p_ [2][s] [to]
large _fixed_ values, to ensure flat priors on the linear coefficients, while the _𝜎r_ [2] [are] [usually]
treated as _random_ and assigned independent priors. Prior specification on variance parameters
is a long-discussed problem due to their poor interpretability, especially for IGMRFs (Sørbye
& Rue, 2014). The conjugate Inverse Gamma priors have been found to lead to overfitting
(Gelman, 2006; Lunn et al., 2009; Simpson et al., 2017) and are now being replaced as the default
choice by Penalized Complexity (PC) priors (Simpson et al., 2017) in the R package INLA (Rue
et al., 2009), which provides fast approximate inference for LGMs.


**1.1** **Variance partitioning-based priors**


Domain experts are often more comfortable reasoning about the relative importance of model
components (e.g., “the spatial effect should account for about half the variability”) than specifying absolute magnitudes of variance parameters. Recently, multiple works have tried to leverage
this underlying information by specifying priors on proportions of variance, rather than absolute
variance parameters.
Such priors can be specified considering a _variance partitioning_ (VP) reparametrization: the
original variance parameters are transformed into an overall total variance _V_, given by the sum
of all or some variance parameters, and a set of variance proportions _**𝝎**_ = [ _𝜔_ 1 _, 𝜔_ 2 _,_ …] _[T]_, given by
the ratio between the variance parameters and _V_ . The reparametrization maps the original, often
nonintuitive, variance scale to a more interpretable scale of proportions constrained to the (0,1)
interval, plus a single total variance. This shift facilitates prior specification as experts usually have


**366** FERRARI and VENTRUCCI


more insight about the relative importance of the different effects, rather than their absolute one.
Two main lines of research have so far used the VP reparametrization, namely, the Hierarchical
Decomposition (HD) priors (Fuglstad et al., 2020) and the R2D2 priors (Zhang et al., 2022).
The HD approach by Fuglstad et al. (2020) provides a structured and interpretable way to build
a joint prior for the random effects’ variance parameters, _𝜎r_ [2][s.] [It] [does] [so] [by] [assigning] [indepen-]
dent priors on the total variance _V_ and the vector of variance proportions _**𝝎**_, where _V_ = [∑] _[R]_ _r_ =1 _[𝜎]_ _r_ [2]
includes only the random effects, and the proportion parameters are defined as _𝜔r_ = _𝜎r_ [2][∕] _[V]_ [, for]
_r_ = 1 _,_ … _, R_ . One key feature of the HD approach is the possibility of a further reparametrization
of _**𝝎**_ into nested sets of proportions through a hierarchical decomposition of the total variance
in the form of a tree. This decomposition involves a sequence of splits, structured to reflect the
available prior information. The final prior on _**𝝎**_ is then constructed by assigning independent
priors (either Dirichlet or PC priors) to the sets of proportions generated at each split. As a prior
on _V_, Fuglstad et al. (2020) suggests a Jeffreys or a PC prior, depending on the response distribution in the observation model. This structured approach improves interpretability and flexibility,
allowing users to encode their prior assumptions and knowledge in an intuitive way. Applications
of HD priors can be found in Hem et al. (2021) and Franco-Villoria et al. (2022). The R package makemyprior (Hem et al., 2024) provides user-friendly tools to design HD priors in a wide
range of models. In the same spirit of HD priors, Wakefield (2007) and Riebler et al. (2016) have
proposed priors on the proportion of variance explained by spatially structured and unstructured
random effects in disease mapping.
The second line of research developed in the context of sparse regression, with the aim
of designing a novel shrinkage prior. Following the Dirichlet-Laplace prior of Bhattacharya
et al. (2015), the R2D2 prior by Zhang et al. (2022) proposes the use of a VP reparametrization to
induce sparsity in multiple linear regression models via prior specification. Considering a model
with only linear effects, the total variance is here defined as _V_ = [∑] _[P]_ _p_ =1 _[𝜏]_ _p_ [2] [and] [the] [proportions]
as _𝜔p_ = _𝜏p_ [2][∕] _[V]_ [for] _[ p]_ [ =][ 1] _[,]_ […] _[,][ P]_ [. The set of proportions is assigned a symmetric Dirichlet distribu-]
tion with a sparsity-inducing hyperparameter choice. On the other hand, the prior on _V_ is not
directly specified, but rather implied by imposing a Beta prior on the measure of goodness-of-fit
_R_ [2] (Yanchenko et al., 2024a), defined as a function of _V_ and the additional likelihood parameters. The R2D2 priors have recently been extended to mixed model cases where only a limited
class of random effects is allowed, such as group effects (Aguilar & Bürkner, 2023; Yanchenko
et al., 2024a, 2024b).
We have seen how both HD and R2D2 approaches make use of a variance partitioning representation of the model. For this reason, we consider them under a common class of priors that,
in this paper, we refer to as _VP priors_ .


**1.2** **The notion of variance contribution in HD and R2D2**


The appeal of VP priors lies in the possibility of obtaining parameters that are more easily interpretable than the original ones. Specifically, _V_ can be understood as the _a priori_ total variance in
the linear predictor due to the summed effects, while the elements in _**𝝎**_ represent the proportional
contributions of the individual effects to this total variance. Importantly, this decomposition holds
_a priori_ : once the model is fitted, different effects will generally be dependent _a posteriori_, and the
variance attributable to each term will no longer sum exactly to _V_ . The clear interpretation of _V_
and _**𝝎**_, however, only holds if all effects lie on a comparable scale, so that each variance parameter (either the _𝜏p_ [2][s] [or] [the] _[𝜎]_ _r_ [2][s)] [correctly] [reflects] [the] [variance] [contribution] [of] [its] [corresponding]


FERRARI and VENTRUCCI **367**


**F I G U R E** **1** Conditional variances over the support of the covariate _zr_ ∈[0 _,_ 1] when _𝜎_ RW1 [2] [=][ 1,] _[ 𝜎]_ IID [2] [=][ 1: (a)]
RW1 effects observed at 20, 25, or 30 equally spaced points; (b) RW1 effect (green) an IID effect (grey) on 25 equally
spaced points; (c) RW1 effect and IID effect after geometric mean scaling. The dashed line indicates variance 1.


effect. We stress that VP priors lose their intuitive advantage if variance parameters are unable to
measure variance contributions on a standardized scale.
The literature on VP priors has so far defined the concept of variance contribution of an effect
in different ways, either directly or implicitly. In what follows, we compare HD and R2D2 priors regarding the adopted notion of variance contribution and the implied definition of intuitive
interpretation of a variance parameter. (Hereafter, E _Z_ [⋅] and Var _Z_ [⋅] will denote, respectively, the
expectation and variance taken only with respect to _Z_, treating all other variables as fixed).
The HD framework explicitly requires that each _𝜎r_ [2] [must] [be] [equal] [to] [the] _[conditional]_ _[vari-]_
_ance_ of the effect given the covariate value _zir_, i.e., Var _**u**_ _r_ [ _**D**_ _[T]_ _r_ [(] _[z]_ _ir_ [)] _**[u]**_ _r_ [|] _[𝜎]_ _r_ [2] _[,][ z][ir]_ []][. Many popular effects]
(called heterogeneous by Fuglstad et al., 2020) have a conditional variance that is nonconstant with respect to the covariate support, and potentially not well approximated by the
value of _𝜎r_ [2][.] [Notable] [examples] [include] [the] [popular] [IGMRFs] [(RW1,] [R2,] [ICAR,] [etc.),] [which]
are also affected by the so-called _scaling_ _issue_ noted by Sørbye and Rue (2014): the conditional variance of an IGMRF (defined only under appropriate linear constraints) depends on
the dimension, i.e., the number of observational points over the domain (see Figure 1, panel
a). Sørbye and Rue (2014) proposed a simple procedure to neutralize the effect of the dimension in an IGMRF, which consists of replacing the structure matrix _**Q**_ with its scaled version
_𝜎_ [2] [where] _[𝜎]_ [2] [a] _[reference]_ _[variance]_ [computed] [as] [the] [geometric] [mean] [of] [the] [conditional]
ref _**[Q]**_ [,] ref [is] {

variance, i.e., _𝜎_ ref [2] [=][ exp] _n_ 1 ∑ _ni_ =1 [log Var] _**[u]**_ _r_ [ _**D**_ _Tr_ [(] _[z]_ _ir_ [)] _**[u]**_ _r_ [|] _[𝜎]_ _r_ [2] [=][ 1][]}] . Fuglstad et al. (2020) propose to
apply this scaling procedure to each heterogeneous random effect modelled, thus implicitly
defining the concept of intuitive interpretation of _𝜎r_ [2] [as] [the] [geometric] [mean] [of] [the] [conditional]
variance.
As an example of how scaling issues can limit the use of VP priors, consider a model where
the linear predictor is the sum of two random effects, a random walk of order 1 (RW1 effect) and
random noise (IID effect); this model will be used in Section 5.1 to demonstrate the benefit of
using the standardization procedure proposed in this paper. The user may have prior information on the proportion of variance contributed by the RW1 component, i.e., _𝜔_ = _𝜎_ RW1 [2] [∕] _[V]_ [, where]
_V_ = _𝜎_ RW1 [2] [+] _[ 𝜎]_ IID [2] [, and hence decide to set priors directly on] _[ 𝜔]_ [and] _[ V]_ [instead of on] _[ 𝜎]_ RW1 [2] [and] _[ 𝜎]_ IID [2] [.]
Unfortunately, this model is affected by a scaling issue, shown in Figure 1 panel b: the conditional variances for RW1 and IID for _𝜎r_ [2] [=][ 1 are clearly on different scales (and for the RW1, it]


**368** FERRARI and VENTRUCCI


is not constant over the covariate domain [0 _,_ 1] and it also depends on the number of observational points). If this mismatch is not fixed, the prior assigned to _𝜔_ will fail in providing a correct
summary of the user’s prior information; for instance, a uniform prior on _𝜔_ will not mean equal
importance _a priori_ between the RW1 and the IID terms. By applying the geometric mean-based
scaling to the RW1, the conditional variance of the RW1 effect gets aligned with the conditional
variance of the IID effect (Figure 1 panel c). While the geometric mean-based scaling by Sørbye
and Rue (2014) provides an effective solution in this particular example, it exhibits some important limitations in the class of models considered in this paper; notably, it fails for linear effects,
which therefore remain outside the scope of application of HD priors.
Alternatively, the R2D2 framework mostly focuses on linear effects, which are also heterogeneous. The original work of Zhang et al. (2022) considers the covariates as random variables
and simply requires that they are standardized (i.e., E _Xp_ [ _Xp_ ] = 0 _,_ Var _Xp_ [ _Xp_ ] = 1 for _p_ = 1 _,_ … _, P_ ), so
that _𝜏p_ [2] [=][ E] _[X]_ _p_ [{][Var] _[𝛽]_ _p_ [[] _[X][p][𝛽][p]_ [|] _[X][p][, 𝜏]_ _p_ [2][]}][ for] _[ p]_ [ =][ 1] _[,]_ […] _[,][ P]_ [. This expression is proven in Zhang et al. (][2022][)]
to correspond to the _marginal variance_, i.e., Var _Xp,𝛽p_ [ _Xp𝛽p_ | _𝜎p_ [2][]][: hence, we consider this to be the]
implicit definition of the intuitive interpretation of _𝜏p_ [2] [from the R2D2 framework. While standard-]
ization is sufficient for linear effects, it is not straightforward to generalize it to random effects
with general structures, such as those from Equation (1).
In conclusion, the notion of variance contribution has been defined in different ways in the
VP literature. In models with both fixed and random effects, a unique definition of what we mean
by variance contribution is currently missing. Scaling methods, to align variance parameters to
their intuitive interpretation, have been proposed in specific models, but a general procedure valid
for a large class of LGMs is currently unavailable. We believe that filling these gaps may greatly
increase the use of HD priors and R2D2 priors in LGM contexts.


**1.3** **Contribution & outline**


This paper proposes a standardization procedure that enables the consistent and interpretable
application of existing VP priors (such as HD and R2D2) within a broad class of LGMs, defined
to include both fixed and random effects, including IGMRFs. We start by defining the concept of
_expected variance of interest_ based on the inferential goal of the user. We argue that the expected
variance of interest formalizes a notion of variance contribution that reflects the intuitive interpretation of the user about the variance parameters for both fixed and random effects. We then
derive the conditions under which the variance parameters match the expected variance of interest, which leads to a general standardization procedure that guarantees an exact match between
variance parameters and variance contributions. We support our methodology with a theoretical
result, proving that the proportions of variance parameters are correctly interpreted after standardization (Remark 1), and simulations. On the practical side, our primary interest is to enable
the use of VP priors in realistic settings. For this reason, we illustrate the proposed standardization
through various examples frequently encountered in LGMs, such as linear and nonlinear effects
of continuous covariates, group effects (i.e., effects of categorical covariates), and spatial and temporal random effects. We put emphasis on the challenges behind standardization of IGMRFs, as
these are frequently used not only to model spatial and temporal dependence but also to construct
nonlinear effects via P-splines (Eilers & Marx, 1996). We argue that a preliminary adjustment is
necessary for P-splines before standardization. For a user-friendly, practical implementation of
standardization, the scaleGMRF R package has been developed and made publicly available at
[https://github.com/LFerrariIt/scaleGMRF.](https://github.com/LFerrariIt/scaleGMRF)


FERRARI and VENTRUCCI **369**


The remainder of the paper is organized as follows. Section 2 introduces the class of LGMs
under study, our definitions of fixed/random effects, and variance contributions. Section 3
explains in great detail the standardization procedure and its applications to linear effects and
group effects. Section 4 discusses standardization for IGMRFs, focusing on the P-spline case.
Section 5 reports simulation studies and a real-data application, while Section 6 concludes.


**2** **DEFINING** **VARIANCE** **CONTRIBUTION** **FOR** **FIXED**
**AND** **RANDOM** **EFFECTS**


In this section, we first define the class of LGMs considered in this paper, we then formally adopt
a modern perspective on the fixed/random effects classification, and finally, we discuss how the
intuitive interpretation of variance contribution varies between the two categories. In contrast to
the common formulation of LGMs, our definition treats the covariates as random variables, as in
the R2D2 proposal of Zhang et al. (2022). This choice is not intended to model the true generative
distribution of the covariates, but rather to allow the user to formalize how variability in each
covariate must be weighted when computing the variance of its associated effect.


**Definition 1** (Latent Gaussian Model). Let _**X**_ = [ _X_ 1 _,_ … _, XJ_ ] _[T]_ be a random vector
of dimension _J_, with marginal distributions _Xj_ ∼ _𝜋j_ ( _xj_ ) on support  _j_ _⊆_ R for _j_ =
1 _,_ … _, J_ . Let a response _Y_ ∼ Dist( _𝜂,_ _**𝝍**_ ) have an exponential family distribution, where
_𝜂_ corresponds to a transformation of the expected value of _Y_ given _**𝝍**_ and:



_𝜂_ = _𝜇_ +



∑ _J_

_fj_ ( _Xj_ ) _._
_j_ =1



Each _fj_ ( _Xj_ ) for _j_ = 1 _,_ … _, J_ is defined as: _fj_ ( _Xj_ ) = _**D**_ _[T]_ _j_ [(] _[X][j]_ [)] _**[u]**_ _[j]_ [where] _**[D]**_ _[j]_ [(] _[X][j]_ [)] [is] [a] [column]
vector containing _Kj_ known functions evaluated at _Xj_, and _**u**_ _j_ = [ _uj,_ 1 _,_ … _, uj,Kj_ ] _[T]_ is a random vector of dimension _Kj_ distributed as: _**u**_ _j_ | _𝜎j_ [2] [∼] _[N][K][j]_ [(] **[0]** _[, 𝜎]_ _j_ [2] _**[Q]**_ _j_ [−][)][ where] _**[ Q]**_ _[j]_ [ is a known]
matrix of dimension _Kj_ × _Kj_, and _𝜎j_ [2] [is a scalar parameter. We shall call] _**[ D]**_ _[j]_ [(] _[X][j]_ [)][ the basis,]
_**u**_ _j_ the coefficients, _𝜎j_ [2] [the variance parameter,] _**[ Q]**_ _[j]_ [ the] _[ structure matrix]_ [ for] _[ effect f][j]_ [(] _[X][j]_ [)][.]


Implementing VP priors on models from Definition 1 consists in specifying a joint prior on
_**𝝈**_ [2] = [ _𝜎_ 1 [2] _[,]_ […] _[, 𝜎]_ _J_ [2][]] _[T]_ [.] [Specification] [is] [then] [completed] [with] [priors] [on] _[𝜇]_ [and] _**[𝝍]**_ [.] [The] [class] [of] [models]
from Definition 1 is very flexible as it can accommodate a wide variety of effects. For instance, linear effects can be specified by setting _Kj_ = 1 and the basis to a single identity function, i.e., _Dj_ ( _Xj_ ) =
_Xj_, hence _uj_ ∼ _N_ 1(0 _, 𝜎j_ [2][)][ is the slope coefficient. Various IGMRFs can be accommodated by mak-]
ing precise assumptions on the basis and precision matrix _**D**_ _j_ (⋅) and _**Q**_ _j_ . Effects whose _**Q**_ _j_ depends
on additional correlation parameters (e.g., Matern) can still be included after conditioning on
reasonable values, as suggested in Fuglstad et al. (2020).


**2.1** **Redefinition of fixed and random effects**


In a Bayesian LGM, common practice is to label the effects as fixed or random according to
whether _𝜎_ [2]
_j_ [is treated as fixed (typically, a large value) or random (hence, assigned a prior). Fol-]
lowing this perspective, all the effects in our LGM would be classified as random. In this paper,
we focus on a different categorization of the effects, which recognizes the different inferential


**370** FERRARI and VENTRUCCI


goals of the user or, in other words, the different ways in which the model estimates can be used
by practitioners. This classification is not novel, and it is inspired by the discussion in Gelman
et al. (2013) and Hodges (2013).
Gelman et al. (2013) in Section 15.6 states that: “[…] …much of the statistical literature
on fixed and random effects can be fruitfully re-expressed in terms of finite-population and
super-population inferences […]. The difference between fixed and random effects is thus not a
difference in inference or computation but in the ways that these inferences will be used”. Similarly, Hodges (2013) proposes to categorize the random effects (i.e., effects where _𝜎j_ [2] [is treated as]
random) between old-style random effect, i.e., when the levels “[…] are draws from a population,
and the draws are not of interest in themselves […]”, and new-style random effects, i.e., when the
“[…] levels themselves are of interest”. Following these ideas, we find it convenient to define a
simple rule that classifies each effect as fixed or random on the basis of the concept of inferential
interest.


**Definition 2** (Fixed and random effects). Consider each effect _fj_ ( _Xj_ ) from the model
in Definition 1. _fj_ ( _Xj_ ) is called “fixed” if the inferential interest is on _**u**_ _j_ (or on some
transformation of them). _fj_ ( _Xj_ ) is called “random” if the inferential interest is on _𝜎j_ [2]
(or some transformation of it).


Hereafter, we use the terms ‘fixed’ and ‘random’ according to Definition 2. As an example of
the application of the above definition, linear effects will typically be considered fixed because the
inferential focus is always on the slope coefficients _u_ (i.e., the effect estimate). Other effects, such
as nonlinear, spatial, temporal, interactions, or group effects, may be classified as fixed or random according to the researcher’s specific inferential objectives. Consider an interaction between
a continuous ( _X_ 1) and a categorical ( _X_ 2) covariate, where the linear effect of _X_ 1 is let vary across
the levels of _X_ 2, i.e., _f_ ( _X_ 1 _, X_ 2) = _u_ ( _X_ 2) _X_ 1. If the interest is not on the slope coefficients themselves
but on their variation across the different levels, then the parameter of interest is _𝜎j_ [2] [and] [not]
_u_ (⋅), hence the effect will be classified as random. On the contrary, if the interest is in the slope
coefficient estimates, the interaction effect will be classified as fixed. As an example of a group
effect, consider a medical study in which we want to account for the impact of the hospital where
patients receive treatment. If the goal is to characterize the variability across hospitals, then the
hospital effect must be considered random. In contrast, if interest lies in comparing specific hospitals, for example, to evaluate performance, the hospital effect should be considered fixed. The
group effect will be examined further in Example 2.
In summary, we find it reasonable to let the user specify each effect as fixed or random according to their specific inferential goal. For convenience, we assume that the effects from Definition 1
are ordered such that the first _L_ effects are fixed, and the last _J_ - _L_ are random. Hence, we can
denote with:


_**𝜽**_ = [ _**u**_ 1 _,_ … _,_ _**u**_ _L, 𝜎L_ [2] +1 _[,]_ […] _[, 𝜎]_ _J_ [2][]] _[T]_ (2)


the vector containing all the parameters of interest for the user.


**2.2** **Intuitive interpretation of** _**𝝈**_ **[2]**


We now formally define the concept of intuitive interpretation of _𝜎_ [2], based on the consideration that the user has a different intuition of variance contribution for fixed and random effects.
For example, the variance contribution of a fixed linear effect is typically measured conditioning


FERRARI and VENTRUCCI **371**


on the slope coefficient to obtain the so-called _explained_ _variance_ Var _X_ [ _Xu_ ], which quantifies
the variability in the response due to the variation in the covariate, while the coefficient is
fixed. Instead, the amount of variability imputed to random group effects is usually estimated
directly through the variance parameter _𝜎_ [2], which captures the heterogeneity across the groups
or levels themselves, considered as draws from a larger population distribution. More generally, we adopt the terminology of Gelman et al. (2013) to argue that the user intends variance
contribution as the:


**•** _finite-population variance_ for fixed effects, i.e., Var _Xj_ [ _fj_ ( _Xj_ )| _**u**_ _j_ ];

**•** _super-population variance_ for random effects, i.e., Var _Xj,_ _**u**_ _j_ [ _fj_ ( _Xj_ )| _𝜎j_ [2][]][.]


This distinction can be neatly summarized using the concept of _variance of interest_, defined
as the variance of the effect conditional on the parameters of interest, i.e., Var _Xj,_ _**u**_ _j_ [ _fj_ ( _Xj_ )| _**𝜽**_ ] with
_**𝜽**_ as in Equation (2). Note, the variance of interest of a random effect (i.e., _𝜃j_ = _𝜎j_ [2][) corresponds]
to the super-population variance, and the variance of interest of a fixed effect (i.e., _𝜃j_ = _uj_ ) equals
the finite-population variance. However, the variance of interest cannot be directly used for the
definition of the intuitive interpretation of _𝜎j_ [2][, as it is not a function of] _[ 𝜎]_ _j_ [2] [for fixed effects. Hence,]
we propose to use instead the _expected variance of interest_, defined as: E _**𝜽**_ {Var _Xj,_ _**u**_ _j_ [ _fj_ ( _Xj_ )| _**𝜽**_ ]| _**𝝈**_ [2] }.


**Definition 3** (Intuitive interpretation of _**𝝈**_ [2] parameters). Consider the model from
Definition 1 with parameters of interest _**𝜽**_ = [ _**u**_ 1 _,_ … _,_ _**u**_ _L, 𝜎L_ [2] +1 _[,]_ […] _[, 𝜎]_ _J_ [2][]] _[T]_ [.] [We] [say] [that]
the variance parameter _𝜎j_ [2] _[,][ j]_ [ =][ 1] _[,]_ […] _[,][ J]_ [,] [has] [an] [intuitive] [interpretation] [if] [it] [is] [equal]
to the expected variance of interest of the corresponding effect, defined as _𝜎j_ [2] [=]
E _𝜃j_ {Var _Xj,_ _**u**_ _j_ [ _fj_ ( _Xj_ )| _𝜃j_ ]| _**𝝈**_ [2] }.


Deriving the expected variance of interest separately for fixed and random effects, we find
out the conditions under which we obtain an intuitive interpretation of the _𝜎j_ [2] [parameters for all]
effects in our model. For fixed effects ( _j_ = 1 _,_ … _, L_ ), the condition for an intuitive interpretation
of _𝜎j_ [2] [is]


_𝜎j_ [2] [=][ E] _**[u]**_ _j_ [{][Var] _[X]_ _j_ [[] _[f][j]_ [(] _[X][j]_ [)][|] _**[u]**_ _[j]_ []][|] _[𝜎]_ _j_ [2][}] _[,]_ (3)


which is the expected finite-population variance marginalizing with respect to _**uj**_ . For random
effects ( _j_ = _L_ + 1 _,_ … _, J_ ), the condition for an intuitive interpretation of _𝜎_ [2]
_j_ [is]


_𝜎j_ [2] [=][ Var] _[X]_ _j_ _[,]_ _**[u]**_ _j_ [[] _[f][j]_ [(] _[X][j]_ [)][|] _[𝜎]_ _j_ [2][]] _[,]_ (4)


which is the super-population variance.
Definition 3 is consistent with the suggestion in Fuglstad et al. (2020) of considering the
explained variance for the introduction of fixed effects in the HD framework, since Equation (3)
corresponds to the _expected_ _explained_ _variance_ . The definition is also coherent with the R2D2
approach since Equation (4) is equivalent to the (implicit) definition of variance contribution in
the R2D2 literature (see Section 1).
Note that Definition 3 relies on the probability assumption made in Definition 1 about the
covariates’ marginal distributions _𝜋j_ ( _xj_ ). By explicitly specifying the distribution of the covariates,
the user can express their view about what constitutes the contribution of each effect. For multiple reasons, a convenient default option consists of assuming marginal uniform distributions


**372** FERRARI and VENTRUCCI


(either discrete or continuous) over sensible supports. First, a Uniform distribution treats all values within the support as equally important in the variance computation: this property ensures
that the variance contributions are not biased toward specific regions of the support space. Secondly, it leads to analytically tractable expressions for the variance contributions of popular effects
(see Examples 2 to 4).


**3** **STANDARDIZATION** **PROCEDURE**


Based on the considerations from the previous section and inspired by Sørbye and Rue (2014), we
propose a standardization procedure that ensures that the condition of Definition 3 is satisfied for
all effects that can be specified in the model class from Definition 1.


**Proposition 1** (Standardization procedure). _Consider_ _the_ _class_ _of_ _models_ _from_
_Definition 1 with parameters of interest_ _**𝜽**_ = [ _**u**_ 1 _,_ … _,_ _**u**_ _L, 𝜎L_ [2] +1 _[,]_ […] _[, 𝜎]_ _J_ [2][]] _[T]_ [.] _[ The condition in]_
_Definition 3 is satisfied after the implementation of the following two steps_ .


_1_ . 0 _-mean constraint on fixed effects. Each fixed effect_ _j_ = 1 _,_ … _, L is redesigned, either_
_by appropriately modifying the basis or constraining the coefficients, so that_ :


E _Xj_ [ _fj_ ( _Xj_ )| _**u**_ _j_ ] = 0 _._



_2_ . _Scaling. Each effect_ _j_ = 1 _,_ … _, J_ _is replaced by_ _[̃]_ _fj_ ( _Xj_ ) = _[f]_ ~~√~~ _[j]_ [(] _[X][j]_ [)]



, _where_ _Cj_ _is a constant_
_Cj_



~~√~~



_that depends on the specific choice of_ _𝜋j_ ( _xj_ ) _and is given by_ :


_Cj_ = Var _Xj,_ _**u**_ _j_ [ _fj_ ( _Xj_ )| _𝜎j_ [2] [=][ 1][]]

[ ]
= E _Xj_ _**D**_ _[T]_ _j_ [(] _[X][j]_ [)] _**[Q]**_ _j_ [−] _**[D]**_ _[j]_ [(] _[X][j]_ [)] _._ (5)


_See the proof in Section A1 of the Appendix_ .
In other words, Proposition 1 states that scaling by the appropriate constant _Cj_ is sufficient
for random effects, as this guarantees the equality between _𝜎_ [2]
_j_ [and the super-population variance]
(Equation 4). On the other hand, both steps of Proposition 1 are necessary for fixed effects to
satisfy Equation (3). The procedure is called _standardization_ as it can be proven that classical
standardization is a special case that arises for linear effects.


**Example 1** (Linear effects). Consider a linear effect _f_ ( _X_ ) = _Xu_ where _u_ | _𝜎_ [2] ∼
_N_ (0 _, 𝜎_ [2] ). Assuming the effect is fixed, Proposition 1 requires the application of both
steps of the standardization procedure. The 0-mean constraint can be simply imposed
by redefining the single basis function as _X_   - E _X_ [ _X_ ], while the scaling constant is
found to be _C_ = Var _X_ [ _X_ ]. As a consequence, a linear effect after the application of
Proposition 1 becomes:


_̃f_ ( _X_ ) = _[X]_ ~~√~~ [−] [E] _[X]_ [[] _[X]_ []] _u_

Var _X_ [ _X_ ]


which corresponds to a linear effect on the standardized version of covariate _X_ . This
result shows the coherence of Proposition 1 with the approach taken in the R2D2
literature.


FERRARI and VENTRUCCI **373**


Linear effects represent a special case in which the standardization is straightforward. The
correct application of more complex effects requires special care in considering some technical
aspects affecting both steps of the procedure.


**3.1** **0-mean constraint**


The 0-mean constraint can be imposed in two ways. One method consists in redefining the
basis as _**D**_ _j_ ( _Xj_ ) − E _Xj_ [ _**D**_ _j_ ( _Xj_ )]: this is the only option when the process has a single basis function
_Dj_ ( _Xj_ ), as in Example 1. For a general process with more than one basis function, the constraint
can also be imposed by replacing the prior on _**u**_ _j_ with its distribution conditional on the linear
constraint _**a**_ _[T]_ _j_ _**[u]**_ _[j]_ [=][ 0, where] _**[ a]**_ _[j]_ [=][ E] _[X][j]_ [[] _**[D]**_ _[j]_ [(] _[X][j]_ [)]][. This latter solution is preferable since it avoids modi-]
fying the basis and therefore preserves the original interpretation of the coefficients. Additionally,
popular IGMRFs inherently require the imposition of linear constraints to ensure propriety
(see Section 4): it is therefore convenient to directly exploit this natural property to impose the
0-mean constraint whenever possible, rather than introducing unnecessary transformations of
the basis.


**Example 2** (Group effect, i.e., effect of a categorical variable). Consider a categorical
variable _X_ with support {1 _,_ … _, K_ } and _𝜋_ ( _X_ = _k_ ) = _pk_ for _k_ = 1 _,_ … _, K_ . An effect for _X_
can be expressed using a vector of _K_ coefficients _**u**_ = [ _u_ 1 _,_ … _, uK_ ] _[T]_, each of them being
linked to a value of the support through basis functions _Dk_ ( _X_ ) = I[ _X_ = _k_ ]:



_f_ ( _X_ ) =



∑ _K_

I( _X_ = _k_ ) _uk_ ; _**u**_ | _𝜎_ [2] ∼ _N_ ( **0** _, 𝜎_ [2] _**I**_ ) _._
_k_ =1



If the effect is considered random, the standardization procedure has no impact
as _C_ = 1. However, there might be cases where _u_ 1 _,_ … _, uK_ are directly of interest, i.e., the effect should be treated as fixed. In this second scenario, the 0-mean
constraint is such that _**a**_ = [ _p_ 1 _,_ … _, pK_ ] _[T]_, and the scaling constant, defined as in



Equation (5), is found to be: _C_ = 1 −



∑ _K_
~~∑~~ _kK_ =1 _[p]_ [3] _k_ . If _X_ is uniformly distributed with _pk_ =
_k_ =1 _[p]_ [2] _k_



1∕ _K,_ ∀ _k_, then _C_ = ( _K_ - 1) _K_ [−][1] . As the number of levels _K_ increases, _C_ converges
to 1: this happens because the variance of the mean decreases and therefore the
0-mean constraint becomes increasingly less relevant. See the proof in the Supporting
Information.



**3.2** **Scaling constant**


The scaling step of Proposition 1 can be implemented in practice either by dividing the basis
~~√~~
matrix by _Cj_ or by multiplying the structure matrix by _Cj_ . The computation of the scaling

constants _Cj_ requires taking into account different aspects. First, all linear constraints imposed
on the effect (including the ones from the 0-mean constraint step) must be considered before
the application of the scaling procedure (see Section 4). Second, there is no guarantee that _Cj_
will be non-null and finite for all potential models, as it is a function of _𝜋j_ ( _xj_ ) _,_ _**D**_ _j_ ( _Xj_ ) _,_ _**Q**_ _j_ : hence,
the couples _fj_ ( _Xj_ ) and _𝜋j_ ( _xj_ ) must be designed to ensure that 0 _< Cj_ _<_ ∞ _,_ ∀ _j_ . Thirdly, it might


**374** FERRARI and VENTRUCCI


be easier to approximate the values of _Cj_ using a Monte Carlo simulation, sampling _N_ values
_xj,_ 1 _,_ … _, xj,N_ from _𝜋j_ ( _xj_ ) to compute an estimate _C_ _[̂]_ _j_ = _N_ [−][1][∑] _[N]_ _i_ =1 _**[D]**_ _[T]_ _j_ [(] _[x][i]_ [)] _**[Q]**_ _j_ [−] _**[D]**_ _[j]_ [(] _[x][i]_ [)][. The scaling method]
from Proposition 1 can be viewed as an extension of Sørbye and Rue (2014) to the case where
the covariates are treated as random variables. In this generalized setting, the _reference variance_

{ }
can be redefined as _𝜎_ ref [2] [=][ GM] _[X][j]_ Var _**u**_ _j_ [ _fj_ ( _Xj_ )| _𝜎j_ [2] [=][ 1] _[,][ X][j]_ []], where GM _X_ [⋅] = exp[E _X_ (log ⋅)]. Not
ing that Equation (5) can be expressed as _Cj_ = E _Xj_ {Var _**u**_ _j_ [ _fj_ ( _Xj_ )| _𝜎j_ [2] [=][ 1] _[,][ X][j]_ []}][, it is easy to see that]
the new scaling constant simply replaces the geometric mean (GM) with the expected value
(E). Importantly, expectation-based scaling has advantages over the geometric mean approach.
For example, geometric mean scaling would return a null _𝜎_ [2] [for] [linear] [effects] [and,]
ref [constant]
more generally, for all effects for which ∃ _x_ ∈  _j_ such that _𝜋j_ ( _xj_ ) _>_ 0 and Var _**u**_ _j_ [ _fj_ ( _x_ )| _𝜎j_ [2] _[,][ X][j]_ [=] _[ x]_ [] =][ 0.]
Additionally, the useful result in Remark 1 from the following section is only achievable thanks
to the linearity property of expectation, and therefore does not hold under geometric mean
scaling.


**3.3** **Interpretation of VP parameters**


We formally define VP priors for the class of models in Definition 1, and show how the corresponding reparametrization offers nicely interpretable parameters after the standardization.


**Definition 4** (VP priors). Consider the model from Definition 1. We define the VP
parameters as:



_V_ =


_**𝝎**_ =



∑ _J_



_𝜎_ [2]
_j_ _[,]_
_j_ =1



_𝜔_ 1 = _𝜎V_ 1 [2] _[,]_ […] _[, 𝜔][J]_ [−][1] [=] _𝜎VJ_ [2] −1 _[, 𝜔][J]_ [=][ 1][ −]




[



] _T_



(6)



∑ _J_ −1

_𝜔j_
_j_ =1



_._



Let _**J**_ represent the Jacobian associated with the transformation _**𝝈**_ [2] →

[ _V, 𝜔_ 1 _,_ … _, 𝜔J_ −1] _[T]_ . We call VP those priors implied on the original variance parameters
_**𝝈**_ [2] by the specification of independent priors on the VP parameters, i.e.,


_𝜋_ ( _𝜎_ 1 [2] _[,]_ […] _[, 𝜎]_ _J_ [2][) =] _[ 𝜋]_ [(] _[V]_ [)] _[𝜋]_ [(] _**[𝝎]**_ [)][|] _**[J]**_ [|] _[.]_


As defined, the class of VP priors is extremely broad, and can cover most of the works cited
in Section 1, in particular HD and R2D2 priors. In the case of a Gaussian likelihood, Definition 4
technically excludes the residual observation variance from the partitioning. It is possible to
include the residual variance simply by specifying the model so that _Y_ ∼ _N_ ( _𝜂,_ 0) and add an
IID effect in the linear predictor (see Section 5.1): we recommend this option only when it is
meaningful to include prior beliefs about the signal–to-noise ratio.
Proposition 1 enables us to derive interpretable expressions for the VP parameters, as detailed
in the following remark.


_Remark_ 1 (Interpretation of the VP parameters). Consider the class of models from
Definition 1 with parameters of interest _**𝜽**_ = [ _**u**_ 1 _,_ … _,_ _**u**_ _L, 𝜎L_ [2] +1 _[,]_ […] _[, 𝜎]_ _J_ [2][]] _[T]_ [. After the appli-]
cation of standardization (Proposition 1), the _V_ parameter from Equation (6) can be
rewritten as the sum of the expected variances of interest:


FERRARI and VENTRUCCI **375**



∑ _J_

Var _Xk,_ _**u**_ _k_ [ _fk_ ( _Xk_ )| _𝜎k_ [2][]]
_k_ = _L_ +1



_V_ =


=



∑ _L_

E _**u**_ _j_ {Var _Xj_ [ _fj_ ( _Xj_ )| _**u**_ _j_ ]| _𝜎j_ [2][} +]
_j_ =1



∑ _J_

E _**𝜽**_ {Var _**X**_ _,_ _**u**_ 1 _,_ … _,_ _**u**_ _J_ [ _fj_ ( _Xj_ )| _**𝜽**_ ]| _**𝝈**_ [2] } _._
_j_ =1



More importantly, the VP parameters from Equation (6) become equal to:


_V_ = E _**𝜽**_ {Var _**X**_ _,_ _**u**_ 1 _,_ … _,_ _**u**_ _J_ [ _𝜂_ | _𝜇,_ _**𝜽**_ ]| _**𝝈**_ [2] } _,_

_𝜔j_ = E _**𝜽**_ {Var _Xj,_ _**u**_ _j_ [ _fj_ ( _Xj_ )| _**𝜽**_ ]| _**𝝈**_ [2] } _j_ = 1 _,_ … _, J._

_V_


See the proof in Section A2 of the Appendix.


Remark 1 shows that, after standardization, the parameter _V_ is not only equal to the sum of the
expected variances of interest of the individual effects, but also exactly corresponds to the expected
variance of interest of the entire linear predictor _𝜂_ . It is important to note that this interpretation
of the VP parameters always holds a priori but not necessarily a posteriori, as Remark 1 relies
on the prior assumption of conditional independence among the model coefficients given the
variance parameters (not on the independence between the covariates which is not required), an
assumption violated a posteriori.


**4** **IGMRFS**


In this section, we discuss the steps required for a sensible application of the standardization
procedure to effects with IGMRF priors, which are often present in LGMs.
IGMRFs are improper multivariate Gaussian distributions, i.e., their structure matrix has a
rank-deficiency _d >_ 0 and therefore a non-empty null space, denoted by matrix _**S**_ with _d_ columns,
such that _**QS**_ = **0** . Rue and Held (2005) defined various classes of IGMRFs: in this paper, we focus
on a subclass of Definition 3.5 of Rue and Held (2005).


**Definition 5** (IGMRFs on regular locations on the line). Consider a vector of regular
locations denoted by _**x**_ = [1 _,_ … _, K_ ] _[T]_ . A random vector _**u**_ ∈ R _[K]_ is an IGMRF of order _d_
with parameters _**𝝁**_ and _**Q**_ on regular locations on the line if it has (improper) density:

[ ]
_𝜋_ ( _**u**_ | _𝜎_ [2] ) ∝ exp           - 2 [1] _𝜎_ [2] [(] _**[u]**_ [ −] _**[𝝁]**_ [)] _[T]_ _**[Q]**_ [(] _**[u]**_ [ −] _**[𝝁]**_ [)]


and _**QS**_ ( _d_ −1) = **0** where _**S**_ ( _d_ −1) = [ _**x**_ [0] _,_ _**x**_ [1] _,_ … _,_ _**x**_ _[d]_ [−][1] ] is called a polynomial matrix (note
that the exponentiation is to be interpreted element-wise).


An effect _f_ ( _X_ ) from Definition 1 with an IGMRF prior of order _d_ from Definition 5 can be
represented as:


_f_ ( _X_ ) = _**D**_ _[T]_ ( _X_ ) _**u**_ ; _**u**_ | _𝜎_ [2] ∼ _N_ ( **0** _, 𝜎_ [2] _**Q**_ [−] ) where _**QS**_ ( _d_ −1) = **0** _._ (7)


To understand the role of _𝜎_ [2] in such effects, we make use of the decomposition from Rue and
Held (2005, Section 3.4.1):


_**u**_ = _**H**_ ( _d_ −1) _**u**_ + ( _**I**_             - _**H**_ ( _d_ −1)) _**u**_


**376** FERRARI and VENTRUCCI


where _**u**_ is decomposed in a polynomial trend of degree _d_ - 1 and a residual part through the _hat_
_matrix_ _**H**_ ( _d_ −1) = _**S**_ ( _d_ −1)[ _**S**_ _[T]_ ( _d_ −1) _**[S]**_ [(] _[d]_ [−][1][)][]][−] _**[S]**_ _[T]_ ( _d_ −1) [. Since] _**[ QH]**_ [(] _[d]_ [−][1][)] [=] **[ 0]** [, the density of] _**[ u]**_ [ (Equation][ 5][) when]
_**𝝁**_ = **0** ) can be rewritten as:


[ ]
_𝜋_ ( _**u**_ ) ∝ exp          - 2 [1] _𝜎_ [2] [(] _**[u]**_ [ −] _**[H]**_ [(] _[d]_ [−][1][)] _**[u]**_ [)] _[T]_ _**[Q]**_ [(] _**[u]**_ [ −] _**[H]**_ [(] _[d]_ [−][1][)] _**[u]**_ [)] _._ (8)


Equation (8) clearly shows how _𝜎_ [2] does not control the dispersion of the IGMRF process
around its mean but rather only the deviation from its polynomial trend of degree _d_ - 1. Hence,
the _𝜎_ [2] parameter can only be correctly interpreted as the variance of the process after the removal
of the polynomial trend, which can be achieved by imposing the _null space constraints_ _**S**_ _[T]_ ( _d_ −1) _**[u]**_ [ =] **[ 0]** [.]
The new process _**u**_ | _**S**_ _[T]_ ( _d_ −1) _**[u]**_ [ =] **[ 0]** [ corresponds to a proper GMRF under constraints, whose variance]
parameter can be correctly interpreted as a measure of dispersion of the process from its mean,
i.e., **0** . However, the constrained process _**u**_ | _**S**_ _[T]_ ( _d_ −1) _**[u]**_ [ =] **[ 0]** [ is not equivalent to the original (uncon-]
strained) process _**u**_, since it “has lost” its polynomial trend of degree _d_ - 1. As a consequence,
_f_ ( _X_ ) from Equation (7) will also be different from _f_ ( _X_ )| _**S**_ _[T]_ ( _d_ −1) _**[u]**_ [ =] **[ 0]** [. Hence,] _[ 𝜎]_ [2] [alone is unable to]
capture the variance contribution of the original effect, only that one of the constrained version.
In order to specify processes from Equation (7) so that variance parameters can measure
their variance contribution, a single effect is therefore insufficient, and it becomes necessary to
use a representation of the process through multiple separate effects, each with its own associated variance parameter. In what follows, we describe how such an alternative representation,
denoted by _f_ [new] ( _X_ ), can be conveniently built for two classes of effects frequently employed in
LGMs: Section 4.1 considers IGMRF effects for discrete supports (e.g., areal/grid data, time series
observed at regular time points); Section 4.2 focuses on P-splines effects, which are used to model
nonlinear effects of continuous covariates.


**4.1** **IGMRFs for discrete effects**


Consider an IGMRF effect for a discrete covariate _X_ with support {1 _,_ … _, K_ } with finite _K_
defined as:



_f_ ( _X_ ) =



∑ _K_

I( _X_ = _k_ ) _uk_ ; _**u**_ | _𝜎_ [2] ∼ _N_ ( **0** _, 𝜎_ [2] _**Q**_ [−] ) where _**QS**_ ( _d_ −1) = **0** _._ (9)
_k_ =1



Equation (9) covers various discrete space and time effects, as the ICAR for areal data (Besag
& Kooperberg, 1995) and the RW1/RW2 processes for temporal data. Since _X_ usually represents
space (e.g., grid cell/county) or time (e.g., day/year) in most applications of Equation (9), we
define _X_ ∼ Unif{1 _,_ … _, K_ } to reflect that each location or time point is equally represented in the
population-level model: this assumption simplifies the derivation of theoretical results and avoids
giving preferences to some values of the support over others.
From the previous discussion, it is clear that the specification of the effect from Equation (9)
is inadequate whenever the interpretation of _𝜎_ [2] is desirable for prior specification, e.g., for VP
priors. In order to build an appropriate alternative representation, we start by considering the
effect under the null space constraints, which we formally denote by _f_ [(] _[r]_ [)] ( _X_ ) where _r_ stands for
_residual_ :



_f_ [(] _[r]_ [)] ( _X_ ) =



∑ _K_

I( _X_ = _k_ ) _uk_ ; _**u**_ | _**S**_ _[T]_ ( _d_ −1) _**[u]**_ [ =] **[ 0]** _[, 𝜎]_ _r_ [2] [∼] _[N]_ [(] **[0]** _[, 𝜎]_ [2][(] _[r]_ [)] _**[Q]**_ [−][)] _[.]_
_k_ =1


FERRARI and VENTRUCCI **377**


We know that the null space constraints _**S**_ _[T]_ ( _d_ −1) _**[u]**_ [ =] **[ 0]** [ guarantee that the coefficients] _**[ u]**_ [ have a null]
polynomial trend of degree _d_ - 1. In this particular case, this is also true for _f_ [(] _[r]_ [)] ( _X_ ) due to the
special choice of basis, i.e.,


E _X_ [ _X_ _[m]_ _f_ [(] _[r]_ [)] ( _X_ )] = 0 _m_ = 0 _,_ … _, d_           - 1 _._ (10)


See the proof in the Supporting Information. The result in Equation (10) proves that imposing the
null space constraints on the original process removes the polynomial trend. Therefore, _f_ ( _X_ ) can
be nicely reconstructed defining _f_ [new] ( _X_ ) as:


_f_ [new] ( _X_ ) = _f_ [(] _[t]_ [)] ( _X_ ) + _f_ [(] _[r]_ [)] ( _X_ ) (11)


where _f_ [(] _[t]_ [)] ( _X_ ) ( _t_ stands for _trend_ ) is designed to model the polynomial effect of degree _d_ - 1 for
covariate _X_, and _𝜎_ [2][(] _[t]_ [)] is the associated variance parameter.
In the case of _d_ = 1, _f_ [(] _[t]_ [)] ( _X_ ) is redundant as the polynomial trend would simply be a constant
effect with respect to _X_, whose inclusion would clash with the intercept parameter _𝜇_ in the linear
predictor and cause an identifiability issue. Hence, _f_ [new] ( _X_ ) = _f_ [(] _[r]_ [)] ( _X_ ), so that the single _𝜎_ [2][(] _[r]_ [)] is
sufficient to measure the variance contribution of the effect. For _d_ = 2 instead, the polynomial
term must be a linear effect _f_ [(] _[t]_ [)] ( _X_ ) = _X𝛽_ [(] _[t]_ [)] where _𝛽_ | _𝜎_ [2][(] _[t]_ [)] ∼ _N_ (0 _, 𝜎_ [2][(] _[t]_ [)] ), so that _f_ [new] ( _X_ ) = _f_ [(] _[t]_ [)] ( _X_ ) +
_f_ [(] _[r]_ [)] ( _X_ ). In general, _f_ [(] _[t]_ [)] ( _X_ ) will be designed using a vector of coefficients _**𝜷**_ of dimension _d_ - 1 (since
IGMRFs of order _d >_ 2 are rarely used in practice, the design of _f_ [(] _[t]_ [)] ( _X_ ) for _d >_ 2 is described in
the Supporting Information). We denote the coefficients of the trend effect _f_ [(] _[t]_ [)] ( _X_ ) as _**𝜷**_, instead of
_**u**_, simply to emphasize the linear/polynomial nature of these effects.
In order to obtain the interpretability of _𝜎_ [2][(] _[t]_ [)] and _𝜎_ [2][(] _[r]_ [)], the two effects must be separately
standardized according to Proposition 1. Note how the null space constraints already imply
E _X_ [ _f_ [(] _[r]_ [)] ( _X_ )] = 0 (i.e., a 0-mean constraint on _f_ [(] _[r]_ [)] ( _X_ )) for every _d >_ 0: hence, the standardization of
_f_ [(] _[r]_ [)] ( _X_ ) is invariant to the fixed/random effect classification.


_Remark_ 2. After standardizing both _f_ [(] _[t]_ [)] ( _X_ ) and _f_ [(] _[r]_ [)] ( _X_ ) terms from Equation (11), it
can be proved that:


Var _X,_ _**𝜷**_ _,_ _**u**_ [ _f_ [new] ( _X_ )| _𝜎_ [2][(] _[t]_ [)] _, 𝜎_ [2][(] _[r]_ [)] ] = _𝜎_ [2][(] _[t]_ [)] + _𝜎_ [2][(] _[r]_ [)] _,_ (12)


where _𝜎_ [2][(] _[r]_ [)] controls the polynomial contribution, i.e., the deviation of the polynomial
trend from the null mean of the process, while _𝜎_ [2][(] _[r]_ [)] controls the residual contribution, i.e., the deviation of the process from its polynomial trend. The process _f_ [new] ( _X_ )
converges to the original _f_ ( _X_ ) when _𝜎_ [2][(] _[t]_ [)] → ∞ (i.e., any possible realization of the
polynomial trend is equally likely). See the proof in the Supporting Information.


**Example 3** (Standardization of RW1/RW2/ICAR). The effect of a time covariate
with regularly spaced observations is often modelled in LGMs assuming a RW1 or
RW2 on the basis coefficients _**u**_ in Equation (9). An RW1 is an IGMRF of order _d_ = 1
with structure matrix:



_**Q**_ RW1 =



⎡ 1 −1 ⎤
⎢ ⎥

 - 1 2 −1

⎢ ⎥
⎢ … … … ⎥ _._ (13)
⎢ - 1 2 −1⎥
⎢⎣ - 1 1 ⎥⎦


**378** FERRARI and VENTRUCCI


Since _d_ = 1, it is sufficient to impose the null space constraint **1** _[T]_ _**u**_ = 0 to obtain a measure of the variance contribution of the effect. After standardization, an RW1 effect
can be written as:



_C_ _**[Q]**_ RW1 [−]



_f_ [new] ( _X_ ) =



∑ _K_ (

I( _X_ = _k_ ) _uk_ ; _**u**_ | **1** _[T]_ _**u**_ = 0 _, 𝜎_ [2] ∼ _N_ **0** _,_ _[𝜎]_ [2][(] _[r]_ [)]

_C_

_k_ =1



∑ _K_



)

(14)



where _C_ depends on the choice of the distribution on _X_ according to Equation (5) and
is such that Var _X,_ _**u**_ [ _f_ [new] ( _X_ )| _𝜎_ [2][(] _[r]_ [)] ] = _𝜎_ [2][(] _[r]_ [)] . Being an IGMRF of order 1, an ICAR effect
(Besag & Kooperberg, 1995) can also be introduced using Equation (14), replacing the
structure matrix with diag( _**W1**_ ) − _**W**_ where _**W**_ must be the adjacency matrix for the
given lattice. An RW2 is instead an IGMRF of order _d_ = 2 with structure matrix:



_**Q**_ RW2 =



⎡ 1 −2 1 ⎤
⎢ ⎥
⎢− 2 5 −4 1 ⎥
⎢ ⎥

1 −4 6 −4 1

⎢ ⎥
⎢ … … … … … ⎥ _._ (15)
⎢ ⎥
⎢ 1 −4 6 −4 1 ⎥
⎢ ⎥
⎢ 1 −4 5 −2⎥
⎢⎣ 1 −2 1 ⎥⎦



In this case, it is necessary to decompose the effect into a linear term and a residual one
subject to the null space constraints. After standardization of both terms, the effect
can be written as:




[−] [E] _[X]_ [[] _[X]_ []]
_f_ [new] ( _X_ ) = _[X]_ ~~√~~ _𝛽_ +

Var _X_ [ _X_ ]



∑ _K_

I( _X_ = _k_ ) _uk,_
_k_ =1



_𝛽_ | _𝜎_ [2][(] _[t]_ [)] ∼ _N_ (0 _, 𝜎_ [2][(] _[t]_ [)] ) _,_



(
_**u**_ | _**S**_ _[T]_ (1) _**[u]**_ [ =] **[ 0]** _[, 𝜎]_ [2][(] _[r]_ [)] [∼] _[N]_ **0** _,_ _[𝜎]_ [2][(] _[r]_ [)]



_C_ _**[Q]**_ RW2 [−]



)
_,_



so that Var _X,𝛽,_ _**u**_ [ _f_ [new] ( _X_ )| _𝜎_ [2][(] _[t]_ [)] _, 𝜎_ [2][(] _[r]_ [)] ] = _𝜎_ [2][(] _[t]_ [)] + _𝜎_ [2][(] _[r]_ [)] . The null space is _**S**_ (1) =

[ **1** _K,_ (1 _,_ … _, K_ ) _[T]_ ]. Under the assumption of _X_ ∼ Unif{1 _,_ … _, K_ }, the scaling constants
_C_ for different values of _K_ for both RW1 and RW2 are reported in the Supporting
Information.


**4.2** **IGMRFs for P-splines**


We now discuss standardization of P-splines (Eilers & Marx, 1996; Lang & Brezger, 2004), a popular approach to model the nonlinear effect of a continuous covariate _X_ . Formally, a P-spline is a
special case of the IGMRF effect in Equation (7),


_f_ ( _X_ ) = _**B**_ ( _X_ ) _[T]_ _**u**_ ; _**u**_ | _𝜎_ [2] ∼ _N_ ( **0** _, 𝜎_ [2] _**Q**_ [−] ) where _**QS**_ ( _d_ −1) = **0** _,_ (16)


FERRARI and VENTRUCCI **379**


where the basis _**D**_ ( _X_ ) is replaced by _**B**_ ( _X_ ) = [ _B_ 1( _X_ ) _,_ … _, BK_ ( _X_ )] _[T]_, with _Bk_ ( _X_ ) _, k_ = 1 _,_ … _, K,_ a B-spline
of any degree evaluated at _X_, and _**u**_ being the vector of spline coefficients with an IGMRF prior
on it, often a RW2. Note, the notation _**D**_ ( _X_ ) is replaced by _**B**_ ( _X_ ) to explicitly highlight that we now
focus on the special case of B-spline bases. Calculating B-splines requires the choice of a finite
interval [ _m, M_ ] over the support of _X_ . As a distribution for _X_, we find convenient to assume equal
weight over [ _m, M_ ], thus we take _X_ ∼ Unif( _m, M_ ).
Defining the concept of variance contribution for a P-spline is challenging as the variance
parameter _𝜎_ [2] of the IGMRF, see Equation (7), does not measure variance contribution directly.
A measure of variance contribution is given by the conditional variance, i.e., the variance of _f_ ( _X_ )
computed under null space constraints at each value of _X_ (see Figure 3). The challenge here is
that the latter not only depends on _𝜎_ [2] but also on the user-defined number of basis functions _K_
(Ventrucci & Rue, 2016).
The alternative representation required for the interpretability of the variance parameter
could be achieved using Equation (11) again, with _f_ [(] _[t]_ [)] ( _X_ ) being a polynomial trend effect (see
Section 4.1), and _f_ [(] _[r]_ [)] ( _X_ ) being equal to Equation (16) subject to _**S**_ _[T]_ ( _d_ −1) _**[u]**_ [ =] **[ 0]** [. However, this solu-]
tion is not viable as the use of a more complex basis is responsible for an important difference
with respect to Section 4.1: the null space constraints do not guarantee a null polynomial trend on
_fr_ ( _X_ ). As a consequence, the null space constraints are also no longer sufficient to guarantee the
0-mean constraint from Proposition 1 on _fr_ ( _X_ ): if _fr_ ( _X_ ) is to be treated as fixed, it would be necessary to further constrain the process during standardization to guarantee the 0-mean requirement
(step 1 of Proposition 1). Additionally, an identifiability issue arises between _ft_ ( _X_ ) and _fr_ ( _X_ ) for
_d_ ≥ 2, since both terms can potentially capture the polynomial trend in the effect of _X_, as _fr_ ( _X_ ) is
not appropriately constrained. Consequently, the variance parameters _𝜎t_ [2] [and] _[ 𝜎]_ _r_ [2] [can no longer be]
neatly interpreted as the polynomial and residual contributions to the variance.
In order to solve the issue raised by the more complex basis choice, we propose a procedure
whose implementation can guarantee identifiability between the trend and polynomial terms,
_f_ [(] _[t]_ [)] ( _X_ ) and _f_ [(] _[r]_ [)] ( _X_ ), and the consequent interpretability of their associated variance parameters.
The procedure presented below, named _Q modification_, solves the problem by replacing the
structure matrix with a new _**Q**_ _[̃]_ whose null space _**S**_ _[̃]_ is such that the null space constraints _**S**_ _[̃]_ _T_ _**u**_ = **0**
guarantee that _f_ [(] _[r]_ [)] ( _X_ ) has a null polynomial trend of degree _d_ - 1, i.e., Equation (10). The adequate
null space _**S**_ _[̃]_ is:

[ ]
_**S**_ _̃_ = E _X_ [ _X_ [0] _**B**_ ( _X_ )] _,_ … _,_ E _X_ [ _X_ [(] _[d]_ [−][1][)] _**B**_ ( _X_ )] _._ (17)


_**Q**_ _̃_ must then be found as the solution of _**Q**_ _̃_ _**S**_ _̃_ = **0** . We propose a solution that preserves sparsity
through the following decomposition:


_**Q**_ _̃_ = ( **𝚲** _**R**_ _̃_            - **𝚲** )− _._ (18)


**𝚲** must be a positive diagonal matrix of entries _**𝝀**_ = [ _𝜆_ 1 _,_ … _, 𝜆K_ ] and _**R**_ _[̃]_ a square matrix with the same
sparsity and sign structure of _**Q**_ . Finding a solution that guarantees _**Q**_ _[̃]_ _**S**_ _[̃]_ = **0** is now equivalent to
finding _**R**_ _[̃]_ such that _**R**_ _[̃]_ **𝚲** _**S**_ _[̃]_ = **0** . The correct entries of _**R**_ _[̃]_ are functions of the known elements of _**Q**_ _,_ _**S**_ _[̃]_,
and the unknown _**𝝀**_ . Hence, the new structure matrix _**Q**_ _[̃]_ is known up to _**𝝀**_ . In order to obtain the
closest possible solution to the original _**Q**_, we propose to find _**𝝀**_ that minimize the Kullback-Leibler
divergence (Rue & Tjelmeland, 2002):


_**𝝀**_ _̂_ = arg min _D_ KL (  _**Q**_ _̃_ ( _**𝝀**_ ) ||  _**Q**_ ) _._ (19)
_**𝝀**_ _>_ **0**


**380** FERRARI and VENTRUCCI


After the Q modification, the modified P-spline effect can be defined as _f_ [new] = _f_ [(] _[t]_ [)] ( _X_ ) + _f_ [(] _[r]_ [)] ( _X_ ),
where _f_ [(] _[t]_ [)] ( _X_ ) is defined as detailed in Section 4.1 and:


_f_ [(] _[r]_ [)] ( _X_ ) = _**B**_ _[T]_ ( _X_ ) _**u**_ ; _**u**_ | _**S**_ _[̃]_ _T_ _**u**_ = **0** _, 𝜎_ 2( _r_ ) ∼ _N_ ( **0** _, 𝜎_ 2( _r_ ) _**Q**_ _̃_ −)


As in Section 4.1, _𝜎_ [2][(] _[t]_ [)] can be interpreted, after standardization, as the polynomial contribution to the variance, while _𝜎_ [2][(] _[r]_ [)] measures the residual contribution. The decomposition in
Equation (18) preserves sparsity since _f_ [(] _[r]_ [)] ( _X_ ) can be equivalently defined as _f_ [(] _[r]_ [)] ( _X_ ) = _**B**_ _[T]_ ( _X_ ) **𝚲** _**u**_
_T_     and _**u**_ | _**S**_ _[̃]_ **𝚲** _**u**_ = **0** _, 𝜎_ 2( _r_ ) ∼ _N_ ( **0** _, 𝜎_ 2( _r_ ) _**R**_ _̃_ ), where _**R**_ _̃_ is, by design, as sparse as the original structure
matrix _**Q**_ .


**Example 4** (Standardization of a cubic P-spline effect). The most popular choice of
P-spline effects consists in the use of a cubic B-spline basis and an RW2 process on the
coefficients, so that the structure matrix is _**Q**_ RW2 from Equation (15) and _**Q**_ RW2 _**S**_ (1) = **0** .
Since it cannot be proven that the null space constraints imply a null polynomial trend
on the constrained process _f_ [(] _[r]_ [)] ( _X_ ), the Q modification procedure is implemented to
achieve interpretability of the variance parameters. First, the new null space _**S**_ _[̃]_ must
be found applying Equation (17):



⎤
⎥
⎥
⎥
⎥
⎥⎦



_**S**_ _̃_
_K_ ×2 [=]



⎡ _̃S_ 1 _,_ 0 _̃S_ 1 _,_ 1
⎢
⎢ _̃S_ 2 _,_ 0 _̃S_ 2 _,_ 1
⎢
⎢ … …
⎢⎣ _̃SK,_ 0 _̃SK,_ 1




[ ]
= E _X_ [ _**B**_ ( _X_ )] _,_ E _X_ [ _X_ _**B**_ ( _X_ )] _._



Then, a valid solution for _**Q**_ _[̃]_ from Equation (18) is found if _**R**_ _[̃]_ = _**G**_ _[̃]_ - _**W**_ _[̃]_ where:



( _l_ - _k_ ) _Wk,l_



_W̃k,l_ =



⎧
⎪
⎨
⎪⎩



_k_ ≠ _l_
_𝜆k𝜆l_ ( _[̃]_ _Sk,_ 0 _[̃]_ _Sl,_ 1 − _[̃]_ _Sk,_ 1 _[̃]_ _Sl,_ 0)



_𝜆k𝜆l_ ( _Sk,_ 0 _Sl,_ 1 − _Sk,_ 1 _Sl,_ 0) _,_ (20)


0 _k_ = _l_



]



_G̃k,l_ = I[ _k_ = _l_ ] 1

_𝜆k_ _[̃]_ _Sk,_ 0




[
∑ _K_

_W̃k,j𝜆j_ _[̃]_ _Sj,_ 0
_j_ =1



(21)



and _**W**_ = diag(diag( _**Q**_ RW2)) − _**Q**_ RW2. See the Supporting Information, which also contains the solution for an RW1 on the coefficients. Finally, the matrix _**Q**_ _[̃]_ that best
approximates the original _**Q**_ is found optimizing _**𝝀**_ according to Equation (19). After
standardization, the P-spline effect can be written as:


_f_ [new] ( _X_ ) = _f_ [(] _[t]_ [)] ( _X_ ) + _f_ [(] _[r]_ [)] ( _X_ ); (22)




[−] [E] _[X]_ [[] _[X]_ []]
_f_ [(] _[t]_ [)] ( _X_ ) = _[X]_ ~~√~~




_[X]_ _𝛽,_ _𝛽_ | _𝜎_ [2][(] _[t]_ [)] ∼ _N_ (0 _, 𝜎_ [2][(] _[t]_ [)] ); (23)

Var _X_ [ _X_ ]




[2][(] _[r]_ [)] - [)]

_**Q**_ _̃_ _._ (24)
_C_



(
_f_ [(] _[r]_ [)] ( _X_ ) = _**B**_ _[T]_ ( _X_ ) _**u**_ _,_ _**u**_ | _**S**_ _[̃]_ _T_ _**u**_ = **0** _, 𝜎_ 2( _r_ ) ∼ _N_ **0** _,_ _[𝜎]_ [2][(] _[r]_ [)]



Under the assumption of _X_ ∼ Unif{1 _,_ … _, K_ }, the scaling constants _C_ for different
values of _K_ are reported in the Supporting Information.


FERRARI and VENTRUCCI **381**


**F I G U R E** **2** Properties of _f_ [(] _[r]_ [)] ( _X_ ) using (a) _**Q**_ RW2 and (b) _**Q**_ _[̃]_ for _K_ = 20. Top panel: generalized inverse of the
structure matrix on the coefficients (each cell represents an element of the matrix). Bottom panel: realizations of
_f_ [(] _[r]_ [)] ( _x_ ) (grey) with corresponding linear trends (blue) when _𝜎_ [2][(] _[r]_ [)] = 1 after scaling.


In order to appreciate the impact of the Q modification, the behaviour of _f_ [(] _[r]_ [)] ( _X_ ) with _K_ = 20
from Example 4 is compared under the use of _**Q**_ RW2 and _**Q**_ _[̃]_ in Figure 2. First, the top panel shows
how the generalized inverse of _**Q**_ _[̃]_ displays a similar pattern to the one of the original _**Q**_ . The bottom
panel of Figure 2 shows realizations of _f_ [(] _[r]_ [)] ( _x_ ) and their linear trends when _𝜎_ [2][(] _[r]_ [)] = 1: the original
model has non-null trends despite the null space constraints, while the new model by design
removes the linear trends.
Figure 3 shows the impact of the Q modification on the shape of the conditional variance of
_f_ [(] _[r]_ [)] ( _X_ ), i.e., Var _**u**_ [ _f_ [(] _[r]_ [)] ( _X_ )| _𝜎_ [2] = 1 _, X_ = _x_ ]. It is important to note that the shape of the conditional
variance of the process after the Q modification tends to approximate that one of the original
model when _K_ → ∞. Because of this, the conditional variances after the modification (and the
standardization) for different values of _K_ are extremely similar (panel b), while this is not the
case for the original model (panel a), where we can see that _K_ impacts the conditional variance. Therefore, the Q modification has the important advantage of neutralizing the impact of
the choice of _K_ on the conditional variance function: in the proposed modified P-spline model, _K_
still controls the local flexibility of the process (i.e., its wiggliness) but not the global shape of the
smooth function, which is controlled by the shape of the conditional variance. Finally, note that
the impact of the Q modification is less relevant as _K_ grows, since the difference between _**S**_ (1) and
_**S**_ _̃_ decreases.


**382** FERRARI and VENTRUCCI


**F I G U R E** **3** Conditional variance of _f_ [(] _[r]_ [)] ( _X_ ) given _𝜎_ [2][(] _[r]_ [)] = 1 (after scaling) for different values of _K_ : (a) before
the Q modification, (b) after the Q modification.


**5** **EMPIRICAL** **RESULTS**


In this section, we investigate the practical benefits of the standardization procedure and the Q
modification through two simulation studies and a survival analysis application on real data.
Along with traditional independent priors, we consider VP priors, designed according to the HD
and/or R2D2 frameworks, depending on what the context allows. The models are fitted in R using
INLA (Rue et al., 2009).


**5.1** **Simulation study: Impact of standardization**


The aim of the first simulation study is to assess the practical impact of the application of the
standardization procedure from Proposition 1.
Consider the following simple model:


_Y_ ∼ _N_ ( _𝜇_ + _f_ ( _X_ ) _, 𝜎_ IID [2] [)] (25)

where _X_ ∼ Unif[1 _,_ 25] and _f_ ( _X_ ) = [∑][25] _k_ =1 [I][(] _[X]_ [=] _[ k]_ [)] _[u][k]_ [is to be considered random (i.e., standardiza-]
tion simplifies to scaling). The coefficients are specified as a first-order random walk (Example 3),
under the appropriate constraint and three different scaling strategies:



(



)




**•** Expectation scaling (Proposition 1): _**u**_ | **1** _[T]_ _**u**_ = 0 _, 𝜎_ RW1 [2] [∼] _[N]_

)

**•** No scaling: _**u**_ | **1** _[T]_ _**u**_ = 0 _, 𝜎_ RW1 [2] [∼] _[N]_ [(] **[0]** _[, 𝜎]_ RW1 [2] _**[Q]**_ RW1 [−]



_𝜎_ [2]
**0** _,_ RW1



_C_ _**Q**_ [−] RW1



)




**•** Geometric mean scaling (Sørbye & Rue, 2014): _**u**_ | **1** _[T]_ _**u**_ = 0 _, 𝜎_ RW1 [2] [∼] _[N]_



(

_𝜎_ [2]
**0** _,_ _𝜎_ RW1 [2] _**Q**_ [−] RW1
ref


FERRARI and VENTRUCCI **383**


where the scaling constants are equal to _C_ = 4 _._ 16 and _𝜎_ [2]
ref [=][ 3] _[.]_ [77. The parameters of the model in]
Equation (25) are therefore _𝜇,_ _**𝝈**_ [2] = [ _𝜎_ RW1 [2] _[, 𝜎]_ IID [2] []][. The only way to apply a VP reparameterization]
in this case is to include both _𝜎_ RW1 [2] _[, 𝜎]_ IID [2] [: hence, in order to comply with Definition][ 4][, the model]
must technically be redefined as _Y_ ∼ _N_ ( _𝜇_ + _f_ ( _X_ ) + _f_ IID( _X_ ) _,_ 0) where _f_ IID( _X_ ) = [∑][25] _x_ =1 [I][(] _[X]_ [=] _[ k]_ [)] _[u]_ [IID] _[,][k]_
and _**u**_ IID ∼ _N_ (0 _, 𝜎_ IID [2] _**[I]**_ [)][. Therefore:]


_𝜎_ [2]
_V_ = _𝜎_ RW1 [2] [+] _[ 𝜎]_ IID [2] [;] _𝜔_ = _._

_𝜎_ [2]
RW1 [+] _[ 𝜎]_ IID [2]


From the model in Equation (25), we also denote by _T_ and _𝜑_, respectively, the total variance in _Y_
given the model parameters and the proportion of variance due to _f_ ( _X_ ):


_T_ = Var _X,_ _**u**_ [ _Y_ | _𝜇,_ _**𝝈**_ [2] ]; _𝜑_ = [Var] Var _[X]_ _X_ _[,]_ _,_ _**[u]**_ _**u**_ [[] [ _[f]_ _Y_ [(] | _[X]_ _𝜇,_ [)][|] _**𝝈**_ _**[𝝈]**_ [2][2] ] []] _[.]_


Using VP priors, assumptions about _T_ and _𝜑_ can be introduced through priors on _V_ and _𝜔_ . This
is correct under expectation scaling since _V_ = _T_ and _𝜔_ = _𝜑_ . Instead, without scaling we have



_𝜎_ RW1 [2] _[C]_
_T_ = _𝜎_ [2]
RW1 _[C]_ [ +] _[ 𝜎]_ IID [2] [and] _[ 𝜑]_ [=] _𝜎_ [2] _[C]_ [+] _[𝜎]_



RW1 [and] _[ 𝜔]_ [are no longer equal to their]

_𝜎_ RW1 [2] _[C]_ [+] _[𝜎]_ IID [2] [, and the VP parameters] _[ V]_



intuitive interpretations, i.e., _T_ and _𝜑_ . The implied priors on _T_ and _𝜑_ would then be distorted in
the sense that they would not correctly reflect the user’s assumptions about these parameters. For
example:



)
_C_
(26)

[ _𝜑_ + _C_  - _𝜑C_ ] [2]



_𝜋_ ( _𝜑_ ) = _𝜋𝜔_



(
_𝜑_
_𝜑_ + _C_ - _𝜑C_



since _𝜔_ = _𝜑T_ [the] [Supporting] [Information][).] [Under] [geometric] [mean] [scaling,] [the]
_𝜑T_ +(1− _𝜑_ ) _TC_ [(see]



implied priors would also be distorted, since _T_ = _𝜎_ RW1 [2] _𝜎C_ ref [2] [+] _[ 𝜎]_ IID [2] [and] _[ 𝜑]_ [=] _𝜎_ RW1 [2] _𝜎_ [+] RW1 [2] _[𝜎]_ IID [2] _[𝜎]_ ref [2] [.]



To assess the impact of the lack of appropriate standardization, we first compare the distortion
of the implied prior on _T_ and _𝜑_ for three different prior choices, listed below.



_iid_
(a) _IG priors_ : _𝜎_ RW1 [2] _[, 𝜎]_ IID [2] ∼ IG(1 _,_ 5 _e_ - 5);

_iid_
(b) _PC priors_ : _𝜎_ RW1 [2] _[, 𝜎]_ IID [2] ∼ PC0(3 _,_ 0 _._ 05) where PC0 is a PC prior (Simpson et al., 2017) with base
model _𝜎_ [2] = 0;
(c) _HD prior_ : _V_ ∼ Jeffreys _, 𝜔_ ∼ Unif(0 _,_ 1).


Note that all chosen priors imply a symmetric density on _𝜔_ . In particular, prior (c) is an HD prior
that assumes a Uniform density for _𝜔_, which means that all values are equally likely a priori. In
this context, it is not meaningful to use an R2D2 prior since there is only one effect in the linear
predictor.
Figure 4 reports the density of the implied priors on _𝜑_ for the different scaling strategies.
The distortion caused by inappropriate scaling is assessed by comparing the black lines, i.e., the
desired priors on _𝜑_, to the other coloured lines, which report the actual implied priors on _𝜑_ .
First, it can be noted that the IG and HD prior choices (which return identical results for _𝜑_ ) seem
more sensitive to appropriate scaling than PC priors, since the distortion appears much larger.
Secondly, we note that large values of _𝜑_ are favoured under no scaling more than in the desired


**384** FERRARI and VENTRUCCI


**F I G U R E** **4** Implied prior on _𝜑_ : (a) IG priors; (b) PC priors; (c) HD prior. The results for (a) and (c) are
reported in the same panel as they are identical.


prior, due to the fact that _C >_ 1 (the contrary would be true for _C <_ 1). Thirdly, geometric mean
scaling always favours larger values more than the desired prior since by construction _C > 𝜎_ [2]
ref [.]
Nevertheless, the difference between expectation and geometric mean scaling methods appears
small, especially for PC priors: this result suggests that geometric mean scaling already removes
most of the distortion.
Since we have found non-negligible distortion in the prior, it is necessary to assess whether
this distortion affects posterior inference. To do so, a single observation of _Y_ is simulated for each
of the _K_ = 25 locations on _X_, i.e., _xi_ = _i_ for _i_ = 1 _,_ … _,_ 25. 200 datasets with _N_ = 25 are generated
in this manner using _𝜇_ = 0 _, T_ = 1 and three different values for _𝜑_ = 0.2, 0.5, 0.8. The datasets are
then fitted under the three prior choices and the three scaling strategies.
Figure 5 reports the bias of the posterior mean of _𝜑_ : there is evidence for a non-negligible
difference between the results of a scaled model (either geometric mean or expectation scaling)
and an unscaled one. As expected, this difference is larger in the case of the HD prior (c), while the
PC priors’ choice (b) appears more robust, and even more so the IG prior. In most scenarios, there
is no relevant difference between the geometric mean and the expectation scaling strategy. Finally,
note that the HD prior performs at least as well as the PC prior specification, as it exhibits the
smallest variation across simulations and fairly limited estimation bias for all values of _𝜑_ . Similar
conclusions can be drawn looking at the estimates for _T_, reported in the Supporting Information,
along with coverage rates.
In summary, the simulation further confirmed that VP priors can perform at least as well
as the popular PC priors. However, the former seems to suffer more severely from the distortion due to inappropriate standardization than the latter. In conclusion, the study just presented
highlighted the importance of the scaling step of the standardization procedure; the impact of
the 0-mean constraint step has been tested in contexts of Example 2 and found to be practically
negligible.


FERRARI and VENTRUCCI **385**


**F I G U R E** **5** Bias (estimate minus true value) of the posterior mean of _𝜑_ under the following prior choices:
(a) IG priors; (b) PC priors; (c) HD prior.


**5.2** **Simulation study: Impact of Q modification**



The goal of this second simulation study is to investigate the identifiability issue found in the
context of P-splines and assess the practical relevance of applying the proposed Q modification
(Section 4.2).
Consider a Gaussian likelihood model with a nonlinear effect√ for continuous covariate
_X_ ∼ Unif(0 _,_ 1) where _Y_ ∼ _N_ ( _f_ [(] _[t]_ [)] ( _X_ ) + _f_ [(] _[r]_ [)] ( _X_ ) _, 𝜎𝜖_ [2][)][,] _[f]_ [ (] _[t]_ [)][(] _[X]_ [) = (] _[X]_ [−] [0] _[.]_ [5][)] 12 _𝛽_ is the linear part and



_X_ ∼ Unif(0 _,_ 1) where _Y_ ∼ _N_ ( _f_ [(] _[t]_ [)] ( _X_ ) + _f_ [(] _[r]_ [)] ( _X_ ) _, 𝜎𝜖_ [2][)][,] _[f]_ [ (] _[t]_ [)][(] _[X]_ [) = (] _[X]_ [−] [0] _[.]_ [5][)] 12 _𝛽_ is the linear part and

_f_ [(] _[r]_ [)] ( _X_ ) = cos(2 _𝜋X_ ) is the nonlinear one. Let _𝜑_ denote the proportional contribution of the nonlinear effect to the variance in the linear predictor, _𝜑_ = Var _X_ [ _f_ [(] _[r]_ [)] ( _X_ )]
Var _X_ [ _f_ [(] _[t]_ [)] ( _X_ )+ _f_ [(] _[r]_ [)] ( _X_ )] [. 200 datasets are generated]



~~√~~
with _N_ = 300 observations and _𝜎𝜖_ [2] [=][ 1] _[, 𝛽]_ [=]



with _N_ = 300 observations and _𝜎𝜖_ [2] [=][ 1] _[, 𝛽]_ [=] 0 _._ 5 so that the variance contribution of _f_ [(] _[t]_ [)] ( _X_ ) and

_f_ [(] _[r]_ [)] ( _X_ ) is equal to 0.5, and so is _𝜑_ .
The response is then fitted using a Gaussian likelihood model with the modified version of
the P-spline effect from Equations (23) and (24) with _K_ = 10 (hence, _C_ = 0 _._ 835). The practical
impact of the Q modification is evaluated by comparing the results to a model using the original



(
prior on the coefficients, that is, _**u**_ | _**S**_ _[T]_ (1) _**[u]**_ [ =] **[ 0]** _[, 𝜎]_ [2][(] _[r]_ [)] [∼] _[N]_ **0** _,_ _[𝜎]_ _C_ [2][(] _[r]_ [)]



( )
prior on the coefficients, that is, _**u**_ | _**S**_ _[T]_ (1) _**[u]**_ [ =] **[ 0]** _[, 𝜎]_ [2][(] _[r]_ [)] [∼] _[N]_ **0** _,_ _[𝜎]_ _C_ [2][(] _[r]_ [)] _**[Q]**_ RW2 [−] for _C_ = 1 _._ 432.

Both models are fitted with four different priors, including two VP priors, one based on the
approach of the HD framework and the other on the R2D2 literature. Contrary to the previous
example, the VP reparametrization is only applied to the linear predictor, as our focus is on the
partition between the linear and nonlinear contributions.



_iid_
(a) _IG priors_ : _𝜎_ [2][(] _[t]_ [)] _, 𝜎_ [2][(] _[r]_ [)] _, 𝜎𝜖_ [2] ∼ IG(1 _,_ 5 _e_ - 5);

_iid_
(b) _PC priors_ : _𝜎_ [2][(] _[t]_ [)] _, 𝜎_ [2][(] _[r]_ [)] _, 𝜎𝜖_ [2] ∼ PC0(3 _,_ 0 _._ 05);
(c) _HD prior_ : _V_ = _𝜎_ [2][(] _[t]_ [)] + _𝜎_ [2][(] _[r]_ [)] ∼ Jeffreys _, 𝜔_ = _𝜎_ [2][(] _[t]_ [)] ∕ _V_ ∼ Unif(0 _,_ 1) _, 𝜎𝜖_ [2] [∼] [IG][(][1] _[,]_ [ 5] _[e]_ [ −] [5][)][;]
(d) _R2D2 prior_ : _R_ [2] = _V_ ∕( _V_ + _𝜎𝜖_ [2][) ∼] [Beta][(][1] _[,]_ [ 1][)] _[, 𝜔]_ [=] _[ 𝜎]_ [2][(] _[t]_ [)][∕] _[V]_ [∼] [Unif][(][0] _[,]_ [ 1][)] _[, 𝜎]_ _𝜖_ [2] [∼] [IG][(][1] _[,]_ [ 5] _[e]_ [ −] [5][)][.]


**386** FERRARI and VENTRUCCI


**F I G U R E** **6** Posterior means of _𝛽_, _𝜑_ (along with their true values in dashed lines), and summary of the CPO
values for the original _**Q**_ RW2 and its modified version _**Q**_ _[̃]_ : (a) IG priors; (b) PC priors; (c) HD prior; (d) R2D2 prior.


To evaluate the impact on the estimation of the linear/nonlinear contributions, Figure 6
reports the posterior means of _𝛽_ and _𝜑_, along with their true values reported via dashed black
lines. The results for prior choice (a) highlight the identifiability issue in the original P-spline
effect: without the Q modification, the estimates can be highly biased since the linear contribution
can be partially or totally absorbed by the nonlinear term. For all prior choices, the Q modification improves the estimates for _𝛽_ and _𝜑_, both in terms of bias and variance: the gap between the
two models is particularly relevant under prior (a), but the improvement is still evident for priors
(b), (c), and (d). Additionally, Figure 6 also reports a measure of predictive performance in terms
of the negative sum of the log conditional predictive ordinates (CPO) of all observations, defined
as CPO _i_ = _𝜋_ ( _yi_ | _**y**_ - _i_ ). The model after the Q modification performs as well as the original version
in terms of predictive ability.
The bias in the estimates of _𝛽_ and _𝜑_ is caused by the misattribution of the signal among the
effects in the linear predictor: specifically, the lack of orthogonality between the two effects before
the Q modification causes the P-spline nonlinear term to absorb part of the linear trend. The Q
modification removes this identifiability issue by enforcing orthogonality between the linear and
nonlinear effects, thus ensuring more accurate estimation of the model parameters and a correct
interpretation of the respective contributions of the effects.
Further simulations with different linear/nonlinear contribution ratios show similar results.
As _K_ grows, the Q modification becomes less important and the difference in estimation performance becomes negligible for _K_ _>_ 25.


FERRARI and VENTRUCCI **387**


**5.3** **Case study: Leukaemia in North West England**


We consider the dataset analysed by Henderson et al. (2002), already studied in Kneib and
Fahrmeir (2007) and Sørbye and Rue (2014). The dataset contains survival times of _N_ = 1043
patients, diagnosed with adult acute myeloid leukaemia between 1982 and 1998 in North West
England (UK). The following covariates are reported for each patient: _Age_, _Wbc_ (white blood
cell count at diagnosis), _Tdi_ (Townsend social deprivation index), _Sex_, _District_ (district of residence). Martino et al. (2011) illustrated how a survival analysis can be carried out in INLA
(i.e., via an LGM) under the assumption of a piecewise log-constant proportional hazard model
(Breslow, 1972). In this case, the linear predictor of the model (i.e., the log-hazard function) is
specified as:


_𝜂_ = _𝜇_ + _f_ 1( _Age_ ) + _f_ 2( _Wbc_ ) + _f_ 3( _Tdi_ ) + _f_ 4( _Sex_ ) + _fT_ ( _Time_ ) + _fS_ ( _District_ )


where _Time_ is a discretization of the survival time in _KT_ = 27 intervals.
The effects _f_ 1( _Age_ ), _f_ 2( _Wbc_ ), _f_ 3( _Tdi_ ) are modelled as P-spline effects with _K_ = 50 basis functions (Equations 23 and 24). We denote by _𝜎_ 1 [2][(] _[t]_ [)] _[, 𝜎]_ 2 [2][(] _[t]_ [)] _[, 𝜎]_ 3 [2][(] _[t]_ [)] the variance parameters of the trend
terms, and by _𝜎_ 1 [2][(] _[r]_ [)] _, 𝜎_ 2 [2][(] _[r]_ [)] _, 𝜎_ 3 [2][(] _[r]_ [)] the parameters of the residual terms. _f_ 4( _Sex_ ) is set to a group effect
(Example 2), while an ICAR model (Besag & Kooperberg, 1995) based on the adjacency matrix of
the districts is used for _fS_ ( _District_ ), and a first-order random walk is chosen for _fT_ ( _T_ ) (Equation 14):
these effects are, respectively, associated with variance parameters _𝜎_ 4 [2] _[, 𝜎]_ _S_ [2] _[, 𝜎]_ _T_ [2] [. All effects are treated]
as fixed; for convenience, a discrete Uniform distribution is assumed for _Sex, Time, District_, and
a continuous one for _Age, Wbc, Tdi_, on their respective empirical ranges (we checked that slight
modification to the ranges does not affect the results). On the basis of this distributional choice,
the necessary 0-mean and null space constraints are imposed on the effects.
In terms of VP priors, we implement here the HD framework by applying the VP reparameterization from Definition 4 to all the 9 variance parameters with priors _V_ ∼ Jeffreys and
_**𝝎**_ ∼ Dir(1 _,_ … _,_ 1). A more thoughtful prior design could entail, for instance, the use of PC0 priors
on the proportions _𝜎p_ [2][(] _[r]_ [)] ( _𝜎p_ [2][(] _[t]_ [)] + _𝜎p_ [2][(] _[r]_ [)] ) [−][1] for _p_ = 1 _,_ 2 _,_ 3 to penalize nonlinearity. An application of
R2D2 in this example would also be possible by following Yanchenko et al. (2024a).
In order to assess the impact of our proposal in this application, we compare the model results
before ( _No scaling_ ) and after ( _Scaling_ ) the application of scaling to all effects, since it is the most
relevant step of the standardization procedure. Note that, for the linear effects, the basis function
is always centered as in Equation (23), but not scaled by the standard deviation of the covariate in
the _No scaling_ case. Figure 7 reports the posterior mean of the effects for _Wbc, Tdi_ in both scenarios. Lack of appropriate scaling significantly affects the smoothness of the estimated functions,
making them more wiggly than necessary. The remaining effects are not significantly affected
by scaling. Overall, the scaled solution reports results that are coherent with past analyses of the
dataset, suggesting, for example a linear trend for both _Age_ and _Wbc_ (Kneib & Fahrmeir, 2007).
To assess the hyperparameter sensitivity of the results, we examined the impact of scaling under
several common choices of the symmetric Dirichlet hyperparameter, namely, 1, 1∕2, and 1∕9 (i.e.,
the reciprocal of the number of effects). Our results show that, in the absence of scaling, smaller
hyperparameter values improve model performance. In contrast, the results remain robust across
all hyperparameter choices when scaling is applied, further highlighting the benefits of scaling
(see Supporting Information). Finally, Table 1 compares the goodness-of-fit and the predictive
ability of the model, before and after scaling, under different prior choices: the model performance
always improves after scaling.


**388** FERRARI and VENTRUCCI


**F I G U R E** **7** Posterior means of (a) _f_ 2( _Wbc_ ) and (b) _f_ 3( _Tdi_ ) before (red) and after expectation scaling (black).


**TA B L E** **1** Comparison of model performance using the Deviance Information Criterion (DIC) and a
summary of the Conditional Predictive Ordinates (CPO _i_ ). The PC and IG prior choices are defined as i.i.d. priors
on the original variance parameters, with the same hyperparameter values of Section 5.2.

**DIC** **−** **[∑]** _**[N]**_ _**i**_ **=1** **[log CPO]** _**[i]**_


**Prior** **HD** **PC** **IG** **HD** **PC** **IG**


No scaling 5655.67 5654.43 5656.25 2841.73 2858.31 2847.96


Scaling 5654.35 5654.38 5654.38 2838.77 2838.91 2838.91


[Additional results and R code scripts to replicate the analysis are available at https://github](https://github.com/LFerrariIt/scaleGMRF/tree/master/Leuk_application)
[.com/LFerrariIt/scaleGMRF/tree/master/Leuk_application.](https://github.com/LFerrariIt/scaleGMRF/tree/master/Leuk_application)


**6** **DISCUSSION**


This paper introduced a standardization method that allows VP priors to be applied for general
LGMs, incorporating both fixed and random effects, including IGMRFs. Via a combination of
centering and scaling, the standardization procedure aligns the variance parameters with their
intuitive interpretation as the variance contribution of the corresponding effects. By bringing
together fixed and random effects under a common framework, this work removes a key obstacle to broader adoption of VP priors in applied Bayesian modelling. We proved the interpretation
of the VP parameters analytically after standardization. Simulation studies further confirmed the
importance of this standardization, especially of its scaling step, for the correct and interpretable
use of VP priors.
The proposed expectation-based scaling method showed negligible practical differences from
geometric mean-based scaling in our simulation settings. We expect the two methods to yield
similar results in most applied contexts. Nonetheless, we found that the scaling constants derived
under the two approaches can diverge for certain models, such as RW1 and RW2 processes, as
the model dimension increases (i.e., as the process is more densely observed). In these cases,


FERRARI and VENTRUCCI **389**


however, the effect on inference may be limited, since the prior influence on the corresponding
variance parameters decreases with increasing data. Despite small practical differences, we argue
that our expectation-based approach is preferable due to theoretical properties, specifically, the
interpretability result in Remark 1.
We also explored how the concept of variance contribution can be extended to IGMRFs. For
these models, variance alignment is achieved by separating the polynomial and residual components into distinct effects with their own variance parameters. While this decomposition is
straightforward in simpler cases, P-splines require an additional modification of the structure
matrix, which we called the _Q modification_, to resolve identifiability issues. This modification can
also be generalized to other structured IGMRFs, such as those formed via Kronecker products.
We believe that providing a general standardization procedure will stimulate further development within the VP framework. While the primary focus of this paper is not on prior specification
itself, our goal is to facilitate the incorporation of prior beliefs about relative variance contributions in a principled and interpretable way. After Fuglstad et al. (2020) and Yanchenko
et al. (2024a), our simulations further support the claim that VP priors offer a robust and intuitive
alternative to conventional priors, since they perform competitively both in terms of parameter estimation and predictive accuracy. Considerable work remains to fully explore the potential
of VP priors in this broader context, for example, to investigate the impact of the choice of
the probability distributions on the covariates and to assess the sensitivity to hyperparameter
choices.
In the future, it would be meaningful to investigate how the VP framework can be effectively
integrated into a broader prior specification strategy that accounts for potential correlation parameters in the structure matrices of the effects. Developing a framework for specifying an overall
joint prior on variance and correlation parameters would acknowledge that prior beliefs about
variance contributions may depend on the magnitude of correlation (see Fuglstad et al., 2019 for
an example of a joint prior on the marginal variance and range parameters of the Matern covariance). Another promising direction is opened up by the possibility to apply VP priors to models
including smooth nonlinear effects on covariates via P-splines, which may lead to the extension of
the use of the R2D2 approach for sparse regression in the context of generalized additive models
(Wei et al., 2020).
Finally, we highlight that the proposed standardization procedure is a valuable tool _regard-_
_less_ _of_ _the_ _chosen_ _prior_, by ensuring appropriate scaling and maintaining the conceptual distinction between fixed and random effects. In this sense, our contribution extends beyond
VP priors and offers a broadly applicable, principled approach to variance parameter scaling
in LGMs.


**ACKNOWLEDGEMENTS**
The authors wish to acknowledge the valuable support and feedback provided by Professor Garritt
Page, which greatly contributed to the development of this article.


**FUNDING** **INFORMATION**
The authors were supported by the European Union under the NextGeneration EU Programme
within the Plan “PNRR—Missione 4” “Istruzione e Ricerca”—Componente C2 Investimento
1.1 “Fondo per il Programma Nazionale di Ricerca e Progetti di Rilevante Interesse Nazionale
(PRIN)” by the Italian Ministry of University and Research (MUR), Project title: “METAbarcoding
for METAcommunities: towards a genetic approach to community ecology (META2)”, Project
code: 2022PA3BS2 (CUP E53D23007580006), MUR D.D. financing decree n. 1015 of 07/07/2023.


**390** FERRARI and VENTRUCCI


**DATA** **AVAILABILITY** **STATEMENT**
The data that support the findings of this study are available in the R package spBayesSurv
at [https://CRAN.R-project.org/package=spBayesSurv.](https://CRAN.R-project.org/package=spBayesSurv) These data were derived from the fol[lowing resources available in the public domain:—LeukSurv, https://search.r-project.org/CRAN](https://search.r-project.org/CRAN/refmans/spBayesSurv/html/LeukSurv.html)
[/refmans/spBayesSurv/html/LeukSurv.html.](https://search.r-project.org/CRAN/refmans/spBayesSurv/html/LeukSurv.html)


**ORCID**
_Luisa Ferrari_ [https://orcid.org/0000-0002-7922-6795](https://orcid.org/0000-0002-7922-6795)
_Massimo Ventrucci_ [https://orcid.org/0000-0001-6903-2076](https://orcid.org/0000-0001-6903-2076)


**REFERENCES**


Aguilar, J. E., & Bürkner, P.-C. (2023). Intuitive joint priors for Bayesian linear multilevel models: The r2d2m2
prior. _Electronic Journal of Statistics_, _17_, 1711–1767.
Besag, J., & Kooperberg, C. (1995). On conditional and intrinsic autoregressions. _Biometrika_, _82_, 733–746.
Bhattacharya, A., Pati, D., Pillai, N. S., & Dunson, D. B. (2015). Dirichlet–laplace priors for optimal shrinkage.
_Journal of the American Statistical Association_, _110_, 1479–1490.
Breslow, N. (1972). Disussion of regression models and life-tables by cox, dr. _Journal of the Royal Statistical Society._
_Series B, Statistical Methodology_, _34_, 216–217.
Eilers, P. H., & Marx, B. D. (1996). Flexible smoothing with b-splines and penalties. _Statistical Science_, _11_, 89–121.
Fahrmeir, L., Kneib, T., & Lang, S. (2004). Penalized structured additive regression for space-time data: A Bayesian
perspective. _Statistica Sinica_, _14_, 731–761.
Franco-Villoria, M., Ventrucci, M., & Rue, H. (2022). Variance partitioning in spatio-temporal disease mapping
models. _Statistical Methods in Medical Research_, _31_, 1566–1578.
Fuglstad, G.-A., Hem, I. G., Knight, A., Rue, H., & Riebler, A. (2020). Intuitive joint priors for variance parameters.
_Bayesian Analysis_, _15_, 1109–1137.
Fuglstad, G.-A., Simpson, D., Lindgren, F., & Rue, H. (2019). Constructing priors that penalize the complexity of
Gaussian random fields. _Journal of the American Statistical Association_, _114_ [, 445–452. https://doi.org/10.1080](https://doi.org/10.1080/01621459.2017.1415907)
[/01621459.2017.1415907](https://doi.org/10.1080/01621459.2017.1415907)
Gelman, A. (2006). Prior distributions for variance parameters in hierarchical models (comment on article by
Browne and Draper). _Bayesian Analysis_, _1_, 515–534.
Gelman, A., Carlin, J. B., Stern, H. S., & Rubin, D. B. (2013). _Bayesian data analysis_ . Chapman and Hall/CRC.
Hem, I., Fuglstad, G.-A., & Riebler, A. (2024). Makemyprior: Intuitive construction of joint priors for variance
parameters in R. _Journal of Statistical Software_, _110_, 1–39.
Hem, I. G., Selle, M. L., Gorjanc, G., Fuglstad, G.-A., & Riebler, A. (2021). Robust modeling of additive and
nonadditive variation with intuitive inclusion of expert knowledge. _Genetics_, _217_ (3), 1–14.
Henderson, R., Shimakura, S., & Gorst, D. (2002). Modeling spatial variation in leukemia survival data. _Journal of_
_the American Statistical Association_, _97_, 965–972.
Hodges, J. S. (2013). _Richly_ _parameterized_ _linear_ _models:_ _Additive,_ _time_ _series,_ _and_ _spatial_ _models_ _using_ _random_
_effects_ . CRC Press.
Kneib, T., & Fahrmeir, L. (2007). A mixed model approach for geoadditive hazard regression. _Scandinavian Journal_
_of Statistics_, _34_, 207–228.
Lang, S., & Brezger, A. (2004). Bayesian p-splines. _Journal of computational and graphical statistics_, _13_, 183–212.
Lunn, D., Spiegelhalter, D., Thomas, A., & Best, N. (2009). Rejoinder to commentaries on: The bugs project:
Evolution, critique and future directions. _Statistics in Medicine_, _28_, 3081–3082.
Martino, S., Akerkar, R., & Rue, H. (2011). Approximate Bayesian inference for survival models. _Scandinavian_
_Journal of Statistics_, _38_, 514–528.
Riebler, A., Sørbye, S. H., Simpson, D., & Rue, H. (2016). An intuitive Bayesian spatial model for disease mapping
that accounts for scaling. _Statistical methods in medical research_, _25_, 1145–1165.
Rue, H., & Held, L. (2005). _Gaussian Markov random fields: Theory and applications_ . CRC press.
Rue, H., Martino, S., & Chopin, N. (2009). Approximate Bayesian inference for latent Gaussian models by
using integrated nested laplace approximations. _Journal_ _of_ _the_ _Royal_ _Statistical_ _Society_ _Series_ _B:_ _Statistical_
_Methodology_, _71_, 319–392.


FERRARI and VENTRUCCI **391**


Rue, H., & Tjelmeland, H. (2002). Fitting Gaussian Markov random fields to Gaussian fields. _Scandinavian journal_
_of Statistics_, _29_, 31–49.
Simpson, D., Rue, H., Riebler, A., Martins, T. G., & Sørbye, S. H. (2017). Penalising model component complexity:
A principled, practical approach to constructing priors. _Statistical Science_, _32_, 1–28.
Sørbye, S. H., & Rue, H. (2014). Scaling intrinsic Gaussian Markov random field priors in spatial modelling. _Spatial_
_Statistics_, _8_, 39–51.
Ventrucci, M., & Rue, H. (2016). Penalized complexity priors for degrees of freedom in Bayesian p-splines.
_Statistical Modelling_, _16_, 429–453.
Wakefield, J. (2007). Disease mapping and spatial regression with count data. _Biostatistics_, _8_, 158–183.
Wei, R., Reich, B. J., Hoppin, J. A., & Ghosal, S. (2020). Sparse Bayesian additive nonparametric regression with
application to health effects of pesticides mixtures. _Statistica Sinica_, _30_, 55–79.
Yanchenko, E., Bondell, H. D., & Reich, B. J. (2024a). The r2d2 prior for generalized linear mixed models. _The_
_American Statistician_, _79_ (1), 10–40.
Yanchenko, E., Bondell, H. D., & Reich, B. J. (2024b). Spatial regression modeling via the r2d2 framework.
_Environmetrics_, _35_, 1–13.
Zhang, Y. D., Naughton, B. P., Bondell, H. D., & Reich, B. J. (2022). Bayesian regression using a prior on the model
fit: The r2-d2 shrinkage prior. _Journal of the American Statistical Association_, _117_, 862–874.


**SUPPORTING** **INFORMATION**
Additional supporting information can be found online in the Supporting Information section at
the end of this article.


**How to cite this article:** Ferrari, L., & Ventrucci, M. (2026). A standardization procedure
to incorporate variance partitioning-based priors in latent Gaussian models. _Scandinavian_
_Journal of Statistics_, _53_ [(1), 364–394. https://doi.org/10.1111/sjos.70042](https://doi.org/10.1111/sjos.70042)


**APPENDIX** **A**


**A1** **Proof of Proposition 1**
Proposition 1 can be proved in multiple steps. Recall that the _j_ _[th]_ effect from Definition 1 is defined
as _fj_ ( _Xj_ ) = **D** _j_ ( _Xj_ ) _[T]_ **u** _j_ . First, it can be noticed that the definition of intuitive interpretation requirement for the fixed effects (Equation 3) can be rewritten as a difference between two terms if the
variance is written in terms of difference of expectations:


E _**u**_ _j_ {Var _Xj_ [ _fj_ ( _Xj_ )| _**u**_ _j_ ]| _𝜎j_ [2][} =][ E] _**[u]**_ _j_ [{][E] _[X]_ _j_ [[] _[f]_ _j_ [ 2][(] _[X][j]_ [)][|] _**[u]**_ _[j]_ [] −] [E] _X_ [2] _j_ [[] _[f][j]_ [(] _[X][j]_ [)][|] _**[u]**_ _[j]_ []][|] _[𝜎]_ _j_ [2][}]

= E _**u**_ _j_ {E _Xj_ [ _fj_ [2][(] _[X][j]_ [)][|] _**[u]**_ _[j]_ []][|] _[𝜎]_ _j_ [2][} −] [E] _**[u]**_ _j_ [{][E][2] _Xj_ [[] _[f][j]_ [(] _[X][j]_ [)][|] _**[u]**_ _[j]_ []][|] _[𝜎]_ _j_ [2][}]


At this stage, the order of integration in the first term on the right-hand side can be changed as
long as E _**u**_ _j_ {E _Xj_ [ _fj_ [2][(] _[X][j]_ [)][|] _**[u]**_ _[j]_ []][|] _[𝜎]_ _j_ [2][}] [is] [finite] [(Fubini-Tonelli] [theorem).] [Inverting] [the] [expectations,] [the]
first term becomes equal to the marginal variance given _𝜎j_ [2][. The second term on the right-hand]
side can also be rewritten as a variance, noting that E _**u**_ _j_ {E _Xj_ [ _fj_ ( _Xj_ )| _**u**_ _j_ ]| _𝜎j_ [2][} =][ 0:]


E _**u**_ _j_ [Var _Xj_ [ _fj_ ( _Xj_ )| _**u**_ _j_ ]| _𝜎j_ [2][] =][ E] _[X]_ _j_ [{][E] _**[u]**_ _j_ [[] _[f]_ _j_ [ 2][(] _[X][j]_ [)][|] _[X][j]_ []][|] _[𝜎]_ _j_ [2][} −] [E] _**[u]**_ _j_ [{][E][2] _Xj_ [[] _[f][j]_ [(] _[X][j]_ [)][|] _**[u]**_ _[j]_ []][|] _[𝜎]_ _j_ [2][}]

= Var _Xj,_ _**u**_ _j_ [ _fj_ ( _Xj_ )| _𝜎j_ [2][] −] [Var] _**[u]**_ _j_ [{][E] _[X]_ _j_ [[] _[f][j]_ [(] _[X][j]_ [)][|] _**[u]**_ _[j]_ []][|] _[𝜎]_ _j_ [2][}]


**392** FERRARI and VENTRUCCI


Hence, a 0-mean constraint on _j_ = 1 _,_ … _, L_ (step 1 of Proposition 1) guarantees that
Var _**u**_ _j_ {E _Xj_ [ _fj_ ( _Xj_ )| _**u**_ _j_ ]| _𝜎j_ [2][} =][ 0] _[,]_ _[j]_ [ =][ 1] _[,]_ […] _[,][ L]_ [ so that:]


E _**u**_ _j_ [Var _Xj_ [ _fj_ ( _Xj_ )| _**u**_ _j_ ]| _𝜎j_ [2][] =][ Var] _[X]_ _j_ _[,]_ _**[u]**_ _j_ [[] _[f][j]_ [(] _[X][j]_ [)][|] _[𝜎]_ _j_ [2][]] _j_ = 1 _,_ … _, L_


√
Finally, if the original effect is replaced with _[̃]_ _fj_ ( _Xj_ ) = _fj_ ( _Xj_ )∕ _Cj_ (step 2 of Proposition 1), then we

obtain that:

[ ] [ ]
Var _Xj,_ _**u**_ _j_ _̃fj_ ( _Xj_ )| _𝜎j_ [2] = _C_ 1 _j_ Var _Xj,_ _**u**_ _j_ _fj_ ( _Xj_ )| _𝜎j_ [2]

{ [ ]}
1
= _Cj_ E _Xj_ Var _**u**_ _j_ _fj_ ( _Xj_ )| _𝜎j_ [2]


_𝜎_ [2] [ ]
_j_
= E _Xj_ _**D**_ _[T]_ _j_ [(] _[X][j]_ [)] _**[Q]**_ _j_ [−] _**[D]**_ _[j]_ [(] _[X][j]_ [)]
_Cj_


_𝜎_ [2]
_j_ _[C][j]_
= = _𝜎j_ [2] _[.]_

_Cj_


Hence, we have found that Proposition 1 ensures that:


_𝜎j_ [2] [=][ Var] _[X]_ _j_ _[,]_ _**[u]**_ _j_ [[] _[̃][f][j]_ [(] _[X][j]_ [)][|] _[𝜎]_ _j_ [2][]] _j_ = 1 _,_ … _, J_


but also that:


_𝜎j_ [2] [=][ E] _**[u]**_ _j_ [{][Var] _[X]_ _j_ [[] _[̃][f][j]_ [(] _[X][j]_ [)][|] _**[u]**_ _[j]_ []][|] _[𝜎]_ _j_ [2][}] _j_ = 1 _,_ … _, L_


which proves that the two steps of Proposition 1 are sufficient to guarantee Definition 3.


**A2** **Proof of Remark 1**
Proposition 1 guarantees that Equations (3–4) are respected. The first part of Remark 1 can be
proven by showing that both expressions can be rewritten as the expected variance of interest,
i.e., _𝜎_ [2]
_j_ [=][ E] _**[𝜽]**_ [{][Var] _**[X]**_ _[,]_ _**[u]**_ [1] _[,]_ […] _[,]_ _**[u]**_ _[J]_ [[] _[f][j]_ [(] _[X][j]_ [)][|] _**[𝜽]**_ []][|] _**[𝝈]**_ [2][}][. For fixed effects] _[ j]_ [ =][ 1] _[,]_ […] _[,][ L]_ [, the expression becomes:]


E _**𝜽**_ {Var _**X**_ _,_ _**u**_ 1 _,_ … _,_ _**u**_ _J_ [ _fj_ ( _Xj_ )| _**𝜽**_ ]| _**𝝈**_ [2] } = E _**u**_ _j_ {Var _Xj_ [ _fj_ ( _Xj_ )| _**u**_ _j_ ]| _𝜎j_ [2][}]


which corresponds to Equation (3). For random effects _j_ = _L_ + 1 _,_ … _, J_, the expression becomes:


E _**𝜽**_ {Var _**X**_ _,_ _**u**_ 1 _,_ … _,_ _**u**_ _J_ [ _fj_ ( _Xj_ )| _**𝜽**_ ]| _**𝝈**_ [2] } = E _𝜎j_ 2 [{][Var] _[X][j][,]_ _**[u]**_ _[j]_ [[] _[f][j]_ [(] _[X][j]_ [)][|] _[𝜎]_ _j_ [2][]][|] _[𝜎]_ _j_ [2][}]

= Var _Xj,_ _**u**_ _j_ [ _fj_ ( _Xj_ )| _𝜎j_ [2][]]

which corresponds to Equation (4). Hence, if _V_ = [∑] _[J]_ _j_ =1 _[𝜎]_ _j_ [2] [as in Equation (][6][) and the conditions of]
Definition 3 hold, then _V_ becomes the sum of the expected variances of interest of all the _J_ effects:



_V_ =



∑ _J_

E _**𝜽**_ {Var _**X**_ _,_ _**u**_ 1 _,_ … _,_ _**u**_ _J_ [ _fj_ ( _Xj_ )| _**𝜽**_ ]| _**𝝈**_ [2] }
_j_ =1


FERRARI and VENTRUCCI **393**


Secondly, it is necessary to prove that the expected variance of interest of the linear predictor, i.e.,
E _**𝜽**_ {Var _**X**_ _,_ _**u**_ 1 _,_ … _,_ _**u**_ _J_ [ _𝜂_ | _𝜇,_ _**𝜽**_ ]| _**𝝈**_ [2] }, is equal to the sum of the expected variances of interest of all the _J_
effects, or after standardization, the sum of _𝜎j_ [2][. First, we can note that:]



{



]



E _**𝜽**_ {Var _**X**_ _,_ _**u**_ 1 _,_ … _,_ _**u**_ _J_ [ _𝜂_ | _𝜇,_ _**𝜽**_ ]| _**𝝈**_ [2] } = E _**𝜽**_


Secondly, we can rewrite this expression as:



}

| _**𝝈**_ [2]


}



Var _**X**_ _,_ _**u**_ 1 _,_ … _,_ _**u**_ _J_




[
∑ _J_



_fj_ ( _Xj_ )| _**𝜽**_
_j_ =1




[
∑ _J_

_fj_ ( _Xj_ )| _**u**_ 1 _,_ … _,_ _**u**_ _L, 𝜎L_ [2] +1 _[,]_ […] _[, 𝜎]_ _J_ [2]
_j_ =1



]



| _𝜎_ 1 [2] _[,]_ […] _[, 𝜎]_ _L_ [2]



E _**u**_ 1 _,_ … _,_ _**u**_ _L_



{


Var _**X**_ _,_ _**u**_ _L_ +1 _,_ … _,_ _**u**_ _J_



This expression can be rewritten more concisely using the notation _**U**_ _F_ = [ _**u**_ 1 _,_ … _,_ _**u**_ _L_ ], _**U**_ _R_ =

[ _**u**_ _L_ +1 _,_ … _,_ _**u**_ _J_ ] _,_ _**𝝈**_ [2] _F_ [= [] _[𝜎]_ 1 [2] _[,]_ […] _[, 𝜎]_ _L_ [2][]][,] _**[ 𝝈]**_ [2] _R_ [= [] _[𝜎]_ _L_ [2] +1 _[,]_ […] _[, 𝜎]_ _J_ [2][]][ as:]



] }

| _**𝝈**_ [2] _F_



E _**U**_ _F_



{


Var _**X**_ _,_ _**U**_ _R_




[
∑ _J_

_fj_ ( _Xj_ )| _**U**_ _F,_ _**𝝈**_ [2] _R_
_j_ =1




[
∑ _J_



Then, we can write the argument of the expectation using the law of total variance as:



{


Var _**U**_ _R_

{




[
∑ _J_



}


}



]

| _**U**_ _F,_ _**𝝈**_ [2] _R_

]



Var _**X**_ _,_ _**U**_ _R_




[ ]
∑ _J_

_fj_ ( _Xj_ )| _**U**_ _F,_ _**𝝈**_ [2] _R_
_j_ =1



= E _**X**_




[
∑ _J_



_fj_ ( _Xj_ )| _**X**_ _,_ _**U**_ _F,_ _**𝝈**_ [2] _R_
_j_ =1



| _**U**_ _F,_ _**𝝈**_ [2] _R_

}



+ Var _**X**_

{




[
∑ _J_



E _**U**_ _R_



_fj_ ( _Xj_ )| _**X**_ _,_ _**U**_ _F,_ _**𝝈**_ [2] _R_
_j_ =1



]



| _**𝝈**_ [2] _R_



= E _**X**_



Var _**U**_ _R_




[
∑ _L_



_fj_ ( _Xj_ )| _**X**_ _,_ _**𝝈**_ [2] _R_
_j_ = _L_ +1



]



+ Var _**X**_



_fj_ ( _Xj_ )| _**U**_ _F_
_j_ =1



∑ _J_



=


=



∑ { [ ] }

E _**X**_ Var _**U**_ _R_ _fj_ ( _Xj_ )| _**X**_ _,_ _**𝝈**_ [2] _R_ | _**𝝈**_ [2] _R_
_j_ = _L_ +1



∑ _J_ { [ ]}

E _**X**_ E _**U**_ _R_ _fj_ [2][(] _[X][j]_ [)][|] _**[X]**_ _[,]_ _**[ 𝝈]**_ _R_ [2]
_j_ = _L_ +1




[
∑ _L_



]



+ Var _**X**_



_fj_ ( _Xj_ )| _**U**_ _F_
_j_ =1



∑ _J_




[
∑ _L_



]



+ Var _**X**_



_fj_ ( _Xj_ )| _**U**_ _F_
_j_ =1



If a 0-mean constraint is imposed on the _j_ = 1 _,_ … _, L_ effects, then:



⎫
⎪
⎬
⎪⎭



]



= E _**X**_



⎧
⎪
⎨
⎪⎩



]2



Var _**X**_




[
∑ _L_

_fj_ ( _Xj_ )| _**U**_ _F_
_j_ =1




[
∑ _L_




[
∑ _L_



_fj_ ( _Xj_ )
_j_ =1



| _**U**_ _F_


**394** FERRARI and VENTRUCCI


so that:




[
∑ _J_

Var _**X**_ _,_ _**U**_ _R_ _fj_ ( _Xj_ )| _**U**_ _F,_ _**𝝈**_ [2] _R_

_j_ =1



]



∑ _J_ { [ ]}

= E _**X**_ E _**U**_ _R_ _fj_ [2][(] _[X][j]_ [)][|] _**[X]**_ _[,]_ _**[ 𝝈]**_ _R_ [2]

_j_ = _L_ +1



]2



⎫
⎪
⎬
⎪⎭



+ E _**X**_



⎧
⎪
⎨
⎪⎩




[
∑ _L_



_fj_ ( _Xj_ )
_j_ =1



| _**U**_ _F_



If we consider again E _**U**_ _F_ {Var _**X**_ _,_ _**U**_ _R_ [∑ _Jj_ =1 _[f][j]_ [(] _[X][j]_ [)][|] _**[U]**_ _[F][,]_ _**[ 𝝈]**_ [2] _R_ ]| _**𝝈**_ [2] _F_ }, it can be written as:




[
∑ _J_

_fj_ ( _Xj_ )| _**U**_ _F,_ _**𝝈**_ [2] _R_
_j_ =1



]



}


=


}



∑ _J_ { [ ]}

E _**X**_ E _**U**_ _R_ _fj_ [2][(] _[X][j]_ [)][|] _**[X]**_ _[,]_ _**[ 𝝈]**_ _R_ [2]
_j_ = _L_ +1



E _**U**_ _F_



{


Var _**X**_ _,_ _**U**_ _R_



⎫
⎪
⎬| _**𝝈**_ [2] _F_
⎪⎭



⎫
⎪
⎬
⎪⎭



⎧
⎪
⎨E _**X**_
⎪⎩



⎧
⎪
⎨
⎪⎩



⎧[
⎪ ∑ _L_
⎨ _fj_ ( _Xj_ )
⎪⎩ _j_ =1



]2


| _**U**_ _F_



⎫
⎪
⎬ _._
⎪⎭



⎫
⎪
⎬ _._
⎪⎭



+ E _**U**_ _F_



Inverting the order of expectation, we get:



| _**𝝈**_ [2] _F_


]



{



| _**𝝈**_ [2] _F_



=


=


=


=


=



E _**U**_ _F_



Var _**X**_ _,_ _**U**_ _R_




[
∑ _J_



_fj_ ( _Xj_ )| _**U**_ _F,_ _**𝝈**_ [2] _R_
_j_ =1



∑ _J_ { [ ]}

E _**X**_ E _**U**_ _R_ _fj_ [2][(] _[X][j]_ [)][|] _**[X]**_ _[,]_ _**[ 𝝈]**_ _R_ [2]
_j_ = _L_ +1



∑ _J_ { [ ]}

E _**X**_ Var _**u**_ 1 _,_ … _,_ _**u**_ _J_ _fj_ ( _Xj_ )| _**𝝈**_
_j_ =1

∑ _J_ [ ]

_𝜎j_ [2][E] _[X]_ _j_ _**D**_ _[T]_ _j_ [(] _[X][j]_ [)] _**[Q]**_ _j_ [−] _**[D]**_ _[j]_ [(] _[X][j]_ [)]
_j_ =1



]2



⎧
⎪
⎨
⎪⎩




[
∑ _L_



⎫⎫
⎪⎪
⎬⎬
⎪⎭⎪⎭



+ E _**X**_



⎧
⎪
⎨E _**U**_ _F_
⎪⎩



⎧
⎪
⎨
⎪⎩



_fj_ ( _Xj_ )
_j_ =1



| _**𝝈**_ [2] _F_



∑ _J_ { [ ]}

E _**X**_ E _**U**_ _R_ _fj_ [2][(] _[X][j]_ [)][|] _**[X]**_ _[,]_ _**[ 𝝈]**_ _R_ [2]
_j_ = _L_ +1



+



∑ _L_ { [ ] }

E _**X**_ E _**U**_ _F_ _fj_ [2][(] _[X][j]_ [)] | _**𝝈**_ [2] _F_
_j_ =1



∑ _J_ { [ ]}

E _**X**_ E _**U**_ _F_ _,_ _**U**_ _R_ _fj_ [2][(] _[X][j]_ [)][|] _**[𝝈]**_ _F_ [2] _[,]_ _**[ 𝝈]**_ [2] _R_
_j_ =1



If scaling has been applied as in Proposition 1, then we know that:



∑ _J_ [ ]

_𝜎j_ [2][E] _[X]_ _j_ _**D**_ _[T]_ _j_ [(] _[X][j]_ [)] _**[Q]**_ _j_ [−] _**[D]**_ _[j]_ [(] _[X][j]_ [)] =
_j_ =1



∑ _J_

_𝜎j_ [2]
_j_ =1



which completes the proof.


