Bayesian Analysis (2020) **15**, Number 4, pp. 1109–1137

# **Intuitive Joint Priors for Variance Parameters**


Geir-Arne Fuglstad _[∗]_, Ingeborg Gullikstad Hem _[†]_, Alexander Knight _[‡]_,
H˚avard Rue _[§]_, and Andrea Riebler _[¶]_


**Abstract.** Variance parameters in additive models are typically assigned independent priors that do not account for model structure. We present a new framework for prior selection based on a hierarchical decomposition of the total variance
along a tree structure to the individual model components. For each split in the
tree, an analyst may be ignorant or have a sound intuition on how to attribute
variance to the branches. In the former case a Dirichlet prior is appropriate to
use, while in the latter case a penalised complexity (PC) prior provides robust
shrinkage. A bottom-up combination of the conditional priors results in a proper
joint prior. We suggest default values for the hyperparameters and offer intuitive
statements for eliciting the hyperparameters based on expert knowledge. The prior
framework is applicable for R packages for Bayesian inference such as INLA and
RStan.


Three simulation studies show that, in terms of the application-specific measures of interest, PC priors improve inference over Dirichlet priors when used to
penalise different levels of complexity in splits. However, when expressing ignorance in a split, Dirichlet priors perform equally well and are preferred for their
simplicity. We find that assigning current state-of-the-art default priors for each
variance parameter individually is less transparent and does not perform better
than using the proposed joint priors. We demonstrate practical use of the new
framework by analysing spatial heterogeneity in neonatal mortality in Kenya in
2010–2014 based on complex survey data.


**Keywords:** additive models, hierarchical variance decomposition, latent Gaussian
models, penalised complexity, joint prior distributions, variance parameters.

## **1 Introduction**


Bayesian hierachical models (BHMs) are ubiquitous in science due to their flexibility
and interpretability (Gelman and Hill, 2007; Gelman et al., 2013; Banerjee et al., 2014).
In this paper, we consider BHMs where the latent level consists of an additive combination of model components that are classified as fixed effects and random effects.
This subclass covers a range of common model classes such as generalised linear mixed


_∗_ Department of Mathematical Sciences, Norwegian University of Science and Technology, Alfred
Getz’ vei 1, 7034 Trondheim, Norway. Corresponding author: [geir-arne.fuglstad@ntnu.no](mailto:geir-arne.fuglstad@ntnu.no)

_†_ Department of Mathematical Sciences, Norwegian University of Science and Technology, Alfred
Getz’ vei 1, 7034 Trondheim, Norway, [ingeborg.hem@ntnu.no](mailto:ingeborg.hem@ntnu.no)

_‡_ Department of Mathematical Sciences, Norwegian University of Science and Technology, Alfred
Getz’ vei 1, 7034 Trondheim, Norway, [alexander.knight@ntnu.no](mailto:alexander.knight@ntnu.no)

_§_ CEMSE Division, King Abdullah University of Science and Technology, Thuwal 23955-6900, Saudi
Arabia, [haavard.rue@kaust.edu.sa](mailto:haavard.rue@kaust.edu.sa)

_¶_ Department of Mathematical Sciences, Norwegian University of Science and Technology, Alfred
Getz’ vei 1, 7034 Trondheim, Norway, [andrea.riebler@ntnu.no](mailto:andrea.riebler@ntnu.no)


_⃝_ c 2020 International Society for Bayesian Analysis [https://doi.org/10.1214/19-BA1185](https://doi.org/10.1214/19-BA1185)


1110 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


models (GLMMs) and generalised additive mixed models (GAMMs) (Fahrmeir and
Lang, 2001). In additive models, the total latent variance of the sum of the random
effects decomposes into the sum of the variance contributed by each random effect, and
each random effect has a variance parameter that controls its _a_ _priori_ contribution. We
present a general framework for constructing joint priors for these variance parameters
for BHMs, and suggest robust shrinkage priors for the reduced class of latent Gaussian
models (LGMs) where the model components are Gaussian conditional on the model
parameters (Rue et al., 2009, 2017; Bakka et al., 2018; Krainski et al., 2018).


There is no concensus on priors for variance parameters in BHMs (Lambert et al.,
2005; Gelman, 2006; Gelman et al., 2017). The default prior in the R package INLA
(Lindgren and Rue, 2015) is an inverse-gamma distribution InvGamma(1 _,_ 5 _·_ 10 _[−]_ [5] )
(Blangiardo and Cameletti, 2015), and the R package RStan (Carpenter et al., 2017;
Stan Development Team, 2018a) has implicit priors that are uniform on the range of legal values for the parameters (Stan Development Team, 2018b). WinBUGS, OpenBUGS
and JAGS used InvGamma(0 _._ 001 _,_ 0 _._ 001) distributions in their examples (Spiegelhalter
et al., 1996; Plummer, 2017), and the Stata manual employs InvGamma(0 _._ 01 _,_ 0 _._ 01) priors (StataCorp, 2017). Conjugacy provides InvGamma( _ϵ, ϵ_ ) distributions with computational advantages, but their use may result in severe problems (Gelman, 2006) and they
are generally inappropriate for variances of random effects (Lunn et al., 2009). Gelman
(2006) proposed heavier tails through Half-Cauchy(25) distributions on the standard
deviations, and others have investigated bounded uniform densities on the variances or
the logarithms of the variances (Lambert et al., 2005) and bounded uniform priors on
the standard deviations (Martinez-Beneito, 2013). Recently, Simpson et al. (2017) proposed a principle-based, robust prior termed penalised complexity (PC) prior that offers
shrinkage towards zero variance. In the case of LGMs, the PC prior is an exponential
distribution on the standard deviation.


However, general-purpose priors may not be suitable for a given application (Gelman
et al., 2017) and independent priors for each random effect cannot exploit the structure of the model (Simpson et al., 2017, Section 7). For example, in disease mapping,
prior elicitation is more meaningful for the total variance of the random effects than
their separate variances (Wakefield, 2006), and, for animal models in genetic settings,
the proportion of variability in a phenotypic trait being accounted for by genes is important (Holand et al., 2013). Further, the intraclass correlation (ICC) (McGraw and
Wong, 1996) in a random intercept model is linked to a generalised version of the coefficient of determination (Gelman and Hill, 2007), also known as _R_ [2], which expresses the
proportion of the total variance explained by the model components. However, putting
a prior on _R_ [2] requires a joint prior on the two variance parameters in the random intercept model. Additionally, in the context of regression, Som et al. (2014) discuss block
g-priors where regression coefficients are partitioned and shrinkage is applied to the _R_ [2]

of each partition.


Consider a simple multilevel model with responses _yi,j,k|ηi,j,k_ _∼_ Poisson(exp( _ηi,j,k_ )),
where _ηi,j,k_ = _ai_ + _bi,j_ + _ci,j,k_ for experiment _k_ on individual _j_ in group _i_ . We will term the
group effect, individual effect and measurement effect for A, B, and C, respectively, and
write the latent model as A+B+C for short hand. The total latent variance _t_ of A+B+C
decomposes as _t_ = _σ_ A [2] [+] _[σ]_ B [2] [+] _[σ]_ C [2] [, where] _[ σ]_ A [2] [,] _[ σ]_ B [2] [and] _[ σ]_ C [2] [are the variances of A, B and C,]


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1111


Figure 1: Hierarchical model decomposition. Gray boxes indicate preferred branches.


respectively. This standard parametrization facilitates independent priors on the variances and can be used to achieve the desired _a priori_ marginal properties for the random
effects. However, it is difficult to encode _a_ _priori_ knowledge on joint properties such as
the size of _t_ or preference for A over B or A+B over C in a transparent and intuitive way.


An obvious alternative is to parametrize the variance parameters as _t_ and the proportion of _t_ assigned to each random effect ( _ω_ A _, ω_ B _, ω_ C), where 0 _≤_ _ω_ A _, ω_ B _, ω_ C _≤_ 1 and
_ω_ A + _ω_ B + _ω_ C = 1. This is illustrated in Figure 1a by splitting A+B+C into the models
A, B and C. This parametrization is suitable for expressing ignorance about how the
variance should be attributed to the random effects. A simple way to assign the joint
prior is to set ( _ω_ A _, ω_ B _, ω_ C) _∼_ Dir( _a, a, a_ ), _a >_ 0, where Dir denotes the Dirichlet distribution (Balakrishnan and Nevzorov, 2003). This prior has no preference for one of the
random effects over the other and is invariant to the ordering of the random effects, and
we can select _a_ _>_ 0 to make the prior suitably vague. Together with the conditional
prior _π_ ( _t|ω_ A _, ω_ B _, ω_ C), this implicitly defines a proper joint prior for ( _σ_ A [2] _[, σ]_ B [2] _[, σ]_ C [2] [)] [that] [is]
invariant to permutations in the order of the random effects, but can incorporate prior
knowledge on _t_ . This has a similar flavor as the Dirichlet-Laplace prior by Bhattacharya
et al. (2015), which is a global-local shrinkage prior (Polson and Scott, 2010) that induces sparsity in regression. However, in this paper we will focus on random effects and
not fixed effects.


The simple split strategy is not always suitable and Riebler et al. (2016) demonstrated that for the BYM (Besag, York and Molli´e) model, which is a sum of a Besag
random effect and an unstructured random effect, a PC prior that penalises the added
complexity of the structured effect relative to the unstructured effect improves inference. For A+B+C, fewer levels of hierarchy may be preferred so that B is preferred to A
and C is preferred over A+B. This knowledge about relative complexity of the random
effects can be incorporated by splitting A+B+C hierarchically as shown in Figure 1b.
Here we first split A+B+C into A+B and C through _ω_ 1 = ( _σ_ A [2] [+] _[ σ]_ B [2] [)] _[/t]_ [,] [and] [then] [split]
A+B into A and B through _ω_ 2 = _σ_ A [2] _[/]_ [(] _[σ]_ A [2] [+] _[ σ]_ B [2] [),] [where] [0] _[ ≤]_ _[ω]_ [1] _[, ω]_ [2] _[≤]_ [1.] [The] [joint] [prior]
for ( _σ_ A [2] _[, σ]_ B [2] _[, σ]_ C [2] [)] [is] [then] [constructed] [by] [first] [selecting] _[π]_ [(] _[ω]_ [2][),] [then] _[π]_ [(] _[ω]_ [1] _[|][ω]_ [2][),] [and] [finally]
_π_ ( _t|ω_ 1 _, ω_ 2). Priors inducing shrinkage towards _ω_ 2 = 0 and _ω_ 1 = 0 can be chosen in
the lower and upper split, respectively. The shrinkage can be illustrated graphically as
shown in Figure 1c. For LGMs, PC priors offer a robust choice, but the framework is
general and other priors can be selected by the analyst. For example, if shrinkage is
only required at the top level, a Dirichlet prior for ( _ω_ 2 _,_ 1 _−_ _ω_ 2) could be combined with
a shrinkage prior for _ω_ 1 _|ω_ 2.


1112 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


The ideas generalize to more random effects through the selection of a hierarchical
decomposition of the model in the form of a tree, and the selection of a conditional
distribution for the attribution of the total variance to the branches for each split. The
joint prior is calculated in a bottom-up approach using these conditional distributions.
We suggest default values for the hyperparameters of the Dirichlet distribution based
on the marginal prior distributions for the proportions of variance assigned to each
branch of the split. This ensures that the default setting for the prior is well-behaved
as the number of branches in a split increases. Default values for the PC priors can
be selected based on moderate shrinkage of the proportion of variance. Additionally,
we discuss how to include expert knowledge through interpretable statements on the
total variance and the distribution of variance in the tree. The joint prior can contain
a mix of expert knowledge and default values that provide a weakly informative prior
(Gelman et al., 2008; Simpson et al., 2017). This means the prior framework with joint
priors is appropriate for default priors for software packages such as INLA and RStan.


The properties of the proposed priors are compared to the properties of default priors from software and vague priors from literature. This is a fair comparison since even
though the new priors account for model structure, they do not incorporate strong expert knowledge and are suggested to be used in a default way in Bayesian software. The
comparison is performed through three simulation studies: a simple random intercept
model with Gaussian responses, a latin square experiment with Gaussian responses, and
a spatial model with Binomial responses. To ease the presentation of the comparisons
and not overload the reader with results, we choose a set of targets for each simulation
study and compare the posteriors resulting from the different prior choices with respect
to the targets. Additional results are provided in the Supplementary Materials (Fuglstad
et al., 2019a). Furthermore, we provide example code in the Supplementary Materials
for producing results for different priors for the latin square model in Section 5.2. The
code is described in Section S4.3 in the Supplementary Materials.


We start by introducing the general framework in Section 2, then we introduce LGMs
and suitable priors for developing a new class of priors for LGMs in Section 3. The new
class of priors for LGMs is introduced in Section 4 and is applied to simulation studies
with Gaussian responses in Section 5. In Section 6 we present one simulation study with
Binomial response and explain how the approach can be used in practice. The paper
ends with a discussion in Section 7.

## **2 Tree-based hierarchical variance decomposition**


In this section we cover basic notation, and formally introduce additive models, hierarchical variance decomposition, and the new framework for joint priors for variances.


**2.1** **Additive** **models**


Let _**y**_ = ( _y_ 1 _, . . ., yn_ ) be a vector of _n_ _>_ 0 observations. We model the expected values
E( _yi_ ) = _g_ _[−]_ [1] ( _ηi_ ), _i_ = 1 _, . . ., n_, through a vector of linear predictors _**η**_ = ( _η_ 1 _, . . ., ηn_ ) and
a link function _g_ : R _→_ R. We consider models where the likelihood has parameters _**θ**_ L


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1113

and factors as _π_ ( _**y**_ _|_ _**η**_ _,_ _**θ**_ L) = [�] _[n]_ _i_ =1 _[π]_ [(] _[y][i][|][η][i][,]_ _**[ θ]**_ [L][).] [This] [covers] [models] [such] [as] [GLMMs] [and]
GAMMs. We term _**η**_ and its description as the latent part of the model.


We assume that the linear predictor is described as



_ηi_ = _β_ 0 + _**x**_ [T] _i_ _**[β]**_ [ +]




- _N_

_uj,kj_ [ _i_ ] _,_ _i_ = 1 _, . . ., n,_ (2.1)
_j_ =1



where _β_ 0 is the intercept, _**x**_ _i_ is the vector of covariates associated with observation _i_,
_**β**_ is a vector of coefficients, and _**u**_ _j_ = ( _u_ 1 _, . . ., umj_ ) is a random vector and _kj_ [ _i_ ] is the
associated element of _**u**_ _j_ for observation _i_ for _j_ = 1 _, . . ., N_ . The two first terms will be
called fixed effects and the last _N_ terms will be called random effects. To focus on the
joint prior for variance parameters, we will assume that each random effect _**u**_ _j_ has a
single model parameter, which is a variance _σj_ [2][. In general, the random effects may have]
other parameters such as correlation parameters and we discuss how to handle this in
Section 7.


We denote the vector of model parameters by _**θ**_ M = ( _σ_ 1 [2] _[, . . ., σ]_ _N_ [2] [). The BHM is com-]
pleted by specifying the latent model through _π_ ( _**u**_ _j|σj_ [2][)] [for] _[j]_ [= 1] _[, . . ., N]_ [,] [and] [the] [prior]
_π_ ( _β_ 0 _,_ _**β**_ _,_ _**θ**_ L _,_ _**θ**_ M). We follow common practice so that the prior satisfies _π_ ( _β_ 0 _,_ _**β**_ _,_ _**θ**_ L _,_ _**θ**_ M) =
_π_ ( _β_ 0) _π_ ( _**β**_ ) _π_ ( _**θ**_ L) _π_ ( _**θ**_ M). The major improvement over common practice is that we will
develop a framework for selecting intuitive joint priors for the variance parameters that
does not require that _π_ ( _**θ**_ M) = [�] _[N]_ _j_ =1 _[π]_ [(] _[σ]_ _j_ [2][).]


**2.2** **Hierarchical** **variance** **decomposition**


The additivity in (2.1) causes the total latent variance Var[ _ηi|β_ 0 _,_ _**β**_ _,_ _**θ**_ M] of linear predictor _i_ to decompose as the variance contributed by each random effect Var[ _ukj_ [ _i_ ] _|β_ 0 _,_ _**β**_ _, σj_ [2][],]
_j_ = 1 _, . . ., N_, for _i_ = 1 _, . . ., n_ . If random effect j is homogeneous, the variance parameter
of random effect _j_ will be a marginal variance in the sense that Var[ _ukj_ [ _i_ ] _|β_ 0 _,_ _**β**_ _, σj_ [2][] =] _[ σ]_ _j_ [2]
for _i_ = 1 _, . . ., n_ . If all random effects are homogeneous, the total latent variance of the
linear predictors is homogeneous, _t_ = Var[ _η_ 1 _|β_ 0 _,_ _**β**_ _,_ _**θ**_ M] = _· · ·_ = Var[ _ηn|β_ 0 _,_ _**β**_ _,_ _**θ**_ M] =
_σ_ 1 [2] [+] _[ . . .]_ [ +] _[ σ]_ _N_ [2] [.] [If] [random] [effect] _[j]_ [is] [heterogenous] [so] [that] [Var[] _[u][k]_ _j_ [[] _[i]_ []] _[|][β]_ [0] _[,]_ _**[ β]**_ _[, σ]_ _j_ [2][]] [varies] [for]
different values of _i_, the variance parameter _σj_ [2] [is selected to be comparable to a marginal]
variance; see the discussion in Section 3.1. We term the parameter _t_ = _σ_ 1 [2] [+] _[ . . .]_ [ +] _[ σ]_ _N_ [2]
the total latent variance.


We describe the attribution of _t_ to the individual random effects through a tree _T_ .
The construction of _T_ starts with a root node _T_ 0 = _{_ 1 _, . . ., N_ _}_ that contains all the
random effects, and in the first step we introduce _K_ 1 _>_ 1 child nodes _T_ 1 _, . . ., TK_ 1 that
partition _T_ 0 into _T_ 0 = _T_ 1 _∪· · · ∪_ _TK_ 1 . We continue this recursively for each child node
until all leaf nodes are singletons. This results in a tree _T_ with _S_ splits where there are
_Ks_ child nodes for split _s_ = 1 _, . . ., S_ . We have _S_ _≤_ _N_ _−_ 1, where _S_ = 1 is achieved by
directly splitting the root node to singletons as in Figure 1a and the maximum value is
achieved by only using dual splits such as in Figure 1b.


For each split _s_, the parent node _Ps_ is split into _Ks_ child nodes _C_ 1 _, . . ., CKs_ and we
will define a vector of parameters _**ω**_ _s_ = ( _ωs,_ 1 _, . . ., ωs,Ks_ ), _s_ = 1 _, . . ., S_ . The child nodes


1114 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


describethe proportiona partitioningof the totalof thevariancerandomineffectsthe parentin the node,parent [�] node, _j∈Ps_ _[σ]_ and _j_ [2][,] [that] we let [is] _**ω**_ [assigned] _s_ describe [to]
each child node through



⎛



_σj_ [2]
_j∈CKs_



1
_**ω**_ _s_ = ~~�~~

_j∈Ps_ _[σ]_ _j_ [2]



⎝ [�]




[�] _σj_ [2] _[, . . .,]_ 
_j∈C_ 1 _j∈C_



⎞


⎠ _,_ _s_ = 1 _, . . ., S._



We denote the _K_ _−_ 1 simplex by Δ _[K]_ = _{_ ( _x_ 1 _, . . ., xK_ ) _|_ [�] _[K]_ _k_ =1 _[x][k]_ [=] [1] _[, x][k]_ _[≥]_ [0] _[∀][k][}]_ [so]
that the restrictions are _**ω**_ _s_ _∈_ Δ _[K][s]_ for _s_ = 1 _, . . ., S_ . This means that the parameters
_ωs,Ks_ are superfluous for _s_ = 1 _, . . ., S_, but we keep them for ease of notation and
interpretability.


For any split _s_ = 1 _, . . ., S_, we term a child node and its decendants as a branch of
the split. The description of the model structure through a tree structure defines a reparametrization of ( _σ_ 1 [2] _[, . . ., σ]_ _N_ [2] [) to (] _[t,]_ _**[ ω]**_ [1] _[, . . .,]_ _**[ ω]**_ _[S]_ [), where] _[ S]_ [is the number of splits in the]
tree. The examples discussed in the introduction can be rephrased in this terminology,
and demonstrate that there is no unique selection of the tree.


**Example** **1** (Tree structure) **.** Consider three random effects A, B and C with marginal
variances ( _σ_ A [2] _[, σ]_ B [2] _[, σ]_ C [2] [).] [Let] [the] [root] [node] [be] _[T]_ [0] [=] _[ {]_ [A] _[,]_ [ B] _[,]_ [ C] _[}]_ [.]


Figure 1a, describes the case that the root node is partitioned into three children
_T_ 1 = _{_ A _}_, _T_ 2 = _{_ B _}_ and _T_ 3 = _{_ C _}_ . This leads to a reparametrization ( _t,_ _**ω**_ ), where
_t_ = _σ_ A [2] [+] _[ σ]_ B [2] [+] _[ σ]_ C [2] [and] _**[ω]**_ [= (] _[σ]_ A [2] _[, σ]_ B [2] _[, σ]_ C [2] [)] _[/t]_ [.]

Figure 1b shows the case that _T_ 0 is first partitioned into _T_ 1 = _{_ A _,_ B _}_ and _T_ 2 =
_{_ C _}_, and then _T_ 1 is partitioned into _T_ 3 = _{_ A _}_ and _T_ 4 = _{_ B _}_ . This results in a
reparametrization ( _t,_ _**ω**_ 1 _,_ _**ω**_ 2), where _t_ = _σ_ A [2] [+] _[σ]_ B [2] [+] _[σ]_ C [2] [,] _**[ω]**_ [1] [=] [(] _[σ]_ A [2] [+] _[σ]_ B [2] _[, σ]_ C [2] [)] _[/t]_ [and]
_**ω**_ 2 = ( _σ_ A [2] _[, σ]_ B [2] [)] _[/]_ [(] _[σ]_ A [2] [+] _[ σ]_ B [2] [).] _△_


**2.3** **Hierachical** **decomposition** **priors**


The tree-based hierarchical variance decomposition facilitates the construction of joint
priors that include prior belief about the relative sizes of groups of random effects. The
tree structure must be selected so that the desired comparisons can be made. Trees
such as shown in Figure 1a are useful for expressing ignorance about the attribution
of variance to the random effects, whereas trees such as shown in Figure 1b are useful
for imposing shrinkage to one of the branches in each dual split. Generally, a tree may
consist of a mixture of splits where the analyst wants to be informative and splits where
the analyst wants to express ignorance.


We propose to construct a joint prior for the marginal variance parameters in a
bottom-up approach where the prior for a given split only depends on descendant nodes
of the parent node.


**Assumption 1** (Bottom-up approach) **.** _For a tree structure with S_ _splits, π_ ( _{_ _**ω**_ _s}_ _[S]_ _s_ =1 [) =]

- _Ss_ =1 _[π]_ [(] _**[ω]**_ _[s][|{]_ _**[ω]**_ _[j][}][j][∈][D]_ [(] _[s]_ [)][)] _[,]_ _[where]_ _[D]_ [(] _[s]_ [)] _[is]_ _[the]_ _[set]_ _[of]_ _[descendant]_ _[splits]_ _[for]_ _[split]_ _[s]_ [ = 1] _[, . . ., S][.]_


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1115


This means that the joint prior for the decomposition uses a directed acyclic graph
so that parameters that belong to subsplits in different branches of a split are marginally
independent. We combine the prior for the decomposition of the variance with a conditional prior on the total variance of the random effects to form what we will call
_hierarchical_ _decomposition_ (HD) priors.


**Definition** **1** (Hierarchical decomposition (HD) priors) **.** Consider a BHM with an
additive latent structure with _N_ random effects with marginal variance parameters
_σ_ 1 [2] _[, . . ., σ]_ _N_ [2] [.] [Assume] [that] [the] [model] [structure] [is] [described] [by] [a] [tree] [that] [recursively]
partitions the set of random effects into singletons. Then a hierarchical decomposition
(HD) prior is given by



_π_ ( _σ_ 1 [2] _[, . . ., σ]_ _N_ [2] [) =] _[ π]_ [(] _[t][|{]_ _**[ω]**_ _[s][}][S]_ _s_ =1 [)]




- _S_

_π_ ( _**ω**_ _s|{_ _**ω**_ _j}j∈D_ ( _s_ )) _,_

_s_ =1



where _t_ = _σ_ 1 [2][+] _[. . .]_ [+] _[σ]_ _N_ [2] [,] _[ S]_ [is the number of splits, and] _[ D]_ [(] _[s]_ [) denotes the set of descendant]
splits for the parent node in split _s_ and _**ω**_ _s_ describes the proportions of the total variance
of a parent node assigned to its branches for _s_ = 1 _, . . ., S_ .

## **3 Latent Gaussian models and priors for the splits**


This section introduces LGMs and the priors we will use for the splits to build the
intuitive class of joint priors for the variance parameters for LGMs.


**3.1** **Latent** **Gaussian** **models**


LGMs constitute a subclass of BHMs with additive latent structure where the model
components are Gaussian conditional on the model parameters. We write the additive
model in (2.1) in vector form, _**η**_ = **1** _β_ 0 + **X** _**β**_ + [�] _[N]_ _j_ =1 **[A]** _[j]_ _**[u]**_ _[j]_ [,] [where] **[1]** [=] [(1] _[, . . .,]_ [ 1)] [is]
a column vector of length _n_, **X** is the _n × p_ design matrix that contains the covariates for each observation as rows, and **A** _j_ are sparse _n × mj_ matrices that select the
appropriate elements of the random effects for _j_ = 1 _, . . ., N_ . The latent Gaussian structure is achieved by _β_ 0 _∼N_ (0 _, σ_ I [2][),] _**[β]**_ _[∼N][p]_ [(] **[0]** _[, σ]_ F [2] **[I]** _[p]_ [),] [and] _**[u]**_ _[j][|][σ]_ _j_ [2] _[∼N][m]_ _j_ [(] **[0]** _[, σ]_ _j_ [2][Σ] _[j]_ [)] [for]
_j_ = 1 _, . . ., N_ . It is common to give _σ_ I [2] [and] _[ σ]_ F [2] [suitably vague values, and we will assume]
that _σ_ I [2] [and] _[σ]_ F [2] [are] [fixed] [and] [focus] [on] [the] [variance] [parameters] _[σ]_ 1 [2] _[, . . ., σ]_ _N_ [2] [.]


For non-intrinsic Gaussian random effects, such as independent and identically distributed (i.i.d.) random effects, stationary autoregressive processes and Mat´ern Gaussian random fields, the covariance matrix Σ of the random effect _**u**_ is chosen to be a
correlation matrix and the variance parameter _σ_ [2] is the marginal variance. However,
this does not work for intrinsic Gaussian Markov random fields (GMRFs) (Rue and
Held, 2005) such as the Besag model (Besag et al., 1991), the first-order random walk
and the second-order random walk (Rue and Held, 2005, Chapter 3). In this case there
is no well-defined concept of a marginal variance since they are defined through singular
precision matrices that cannot be inverted to find a covariance matrix. We follow Sørbye
and Rue (2014) and choose the variance parameter _σ_ [2] to be a representative value for
the marginal variance.


1116 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


**3.2** **Introducing** **shrinkage** **towards** **branches**


**Penalising** **complexity**


The fundamental basis for introducing robust shrinkage in our proposed class of priors
are the PC priors introduced in Simpson et al. (2017), which uses a set of principles to
derive model-component-specific prior distributions. The main idea is to regard a single
model component as a flexible extension of a so-called base model. In the simplest case
of an unstructured random effect, the base model would be to remove the effect entirely
from the linear predictor by letting the variance parameter go to zero. The idea is to
follow Occam’s razor and favour a simpler, more sparse or more intuitive model as long
as the data does not indicate otherwise. The PC priors have been used successfully in
a variety of contexts such as BYM models (Riebler et al., 2016), correlation parameters (Guo et al., 2017), autoregressive processes (Sørbye and Rue, 2018) and Mat´ern
Gaussian random fields (Fuglstad et al., 2019b).


Simpson et al. (2017) proposed to compute the complexity of the alternative model
relative to the base model using the Kullback-Leibler divergence (KLD) defined as




            -            _π_ ( _**u**_ _|ξ_ )
KLD( _π_ ( _**u**_ _|ξ_ ) _|| π_ ( _**u**_ _|ξ_ = 0)) = _π_ ( _**u**_ _|ξ_ ) log
_π_ ( _**u**_ _|ξ_ = 0)





d _**u**_ _,_ (3.1)



where _ξ_ is the flexibility parameter, and _ξ_ = 0 at the base model. The KLD is consequently transform ~~�~~ ed to an interpretable distance measure between two densities _f_ 1 and
_f_ 2: _d_ ( _f_ 1 _|| f_ 2) = 2KLD( _f_ 1 _|| f_ 2). In contrast to defining a prior for _ξ_ directly, a prior
is defined for _d_ . See Simpson et al. (2017) for detailed motivation.


We follow Simpson et al. (2017) and select an exponential distribution, where information provided by the user is used to determine the rate _λ_ . Usually this information
is provided by a probability statement about the tail probability of the prior,


_P_ ( _X_ ( _ξ_ ) _> U_ ) = _α._


Here, _X_ ( _ξ_ ) is an interpretable transformation of the parameter of the flexible extension,
_U_ can be thought of as a sensible upper bound, and _α_ is a small probability. A user can
express their knowledge by constraining tail probabilities of _X_ ( _ξ_ ) as above. Selecting
_U_ near a large plausible value for _X_ ( _ξ_ ) and _α_ small encodes weak information about _ξ_
(Simpson et al., 2017). This means that it is _a_ _priori_ unlikely that the value of _X_ ( _ξ_ )
exceeds _U_ . Finally, the prior can be transformed to the corresponding prior for the
flexibility parameter _ξ_ . An attractive feature of this principle-based construction is that
the resulting priors are proper and have a natural link to Jeffreys’ priors.


**Shrinking** **a** **marginal** **variance** **parameter**


In the case of a single Gaussian random effect with marginal variance _σ_ [2], the PC prior
with base model _σ_ [2] = 0 is an exponential prior on _σ_ . The rate parameter _λ_ can be set,
for example, by an _a_ _priori_ statement P( _σ_ _>_ _U_ ) = 0 _._ 05 so that the 95th percentile of
the prior for _σ_ is _U_ _>_ 0. Then the prior is an exponential prior with rate parameter
_λ_ = _−_ log( _α_ ) _/U_ which we denote as _σ_ _∼_ PCSD( _U, α_ ); see Simpson et al. (2017) for
details and derivation.


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1117


**Shrinking** **a** **weight** **parameter**


Consider the situation that the linear predictor only contains two random effects _A_ and
_B_ with variances _σ_ A [2] [and] _[σ]_ B [2] [,] [respectively.] [The] [proportion] [of] _[t]_ [=] _[σ]_ A [2] [+] _[ σ]_ B [2] [assigned] [to]
each random effect is described by _**ω**_ = (1 _−_ _ω, ω_ ) = ( _σ_ A [2] _[, σ]_ B [2] [)] _[/]_ [(] _[σ]_ A [2] [+] _[ σ]_ B [2] [).] [If] [one] _[a]_ _[priori]_
prefers the attribution _**ω**_ = _**ω**_ [0] = (1 _−ω_ 0 _, ω_ 0), shrinkage can be induced in the joint prior
for the variance parameters using a PC prior where _**ω**_ = _**ω**_ [0] is the base model. Here we
apply the KLD from (3.1) to express distance from the base model _**ω**_ [0] to the alternative
model _**ω**_, and penalise deviations from the base model according to the difference in
model complexity.


**Theorem** **1** (PC prior for dual split) **.** _Let_ _**u**_ 1 _and_ _**u**_ 2 _be_ _random_ _effects_ _of_ _an_ _LGM_
_that_ _enter_ _the_ _linear_ _predictor_ _through_ **A** 1 _**u**_ 1 _∼Nn_ ( **0** _, σ_ 1 [2][˜Σ][1][)] _[and]_ **[A]** [2] _**[u]**_ [2] _[∼N][n]_ [(] **[0]** _[, σ]_ 2 [2][˜Σ][2][)] _[.]_
_Assume that_ Σ [˜] 1+Σ [˜] 2 _is non-singular._ [1] _Let ω_ = _σ_ 2 [2] _[/]_ [(] _[σ]_ 1 [2][+] _[σ]_ 2 [2][)] _[ and]_ [ Σ(] _[w]_ [) = (1] _[−][ω]_ [)˜Σ][1][+] _[ω]_ [˜Σ][2] _[.]_
_Then_ _the_ ~~�~~ _distance_ _from_ _the_ _base_ _model_ Σ( _ω_ 0) _to_ _the_ _alternative_ _model_ Σ( _ω_ ) _is_ _given_ _by_
_d_ ( _ω_ ) = tr(Σ( _ω_ 0) _[−]_ [1] Σ( _ω_ )) _−_ _n −_ log _|_ Σ( _ω_ 0) _[−]_ [1] Σ( _ω_ ) _|_ _for_ 0 _≤_ _ω_ 0 _≤_ 1 _._



_The_ _PC_ _prior_ _for_ _ω_ _with_ _base_ _model_ _ω_ 0 = 0 _is_



_λ|d_ _[′]_ ( _ω_ ) _|_
1 _−_ exp( _−λd_ (1)) [exp (] _[−][λd]_ [ (] _[ω]_ [))] _[,]_ 0 _< w_ _<_ 1 _,_ Σ [˜] 1 _non-singular,_

2 ~~_[√]_~~ _ω_ (1 _−λ_ exp( _−λ_ )) [exp(] _[−][λ]_ ~~_[√]_~~ _[ω]_ [)] _[,]_ 0 _< ω_ _<_ 1 _,_ Σ [˜] 1 _singular,_



_π_ ( _ω_ ) =



⎧
⎨


⎩



_where λ >_ 0 _is the hyperparameter. We suggest to set λ so that the median is ω_ m = 0 _._ 25 _._



_For_ _base_ _model_ 0 _< ω_ 0 _<_ 1 _,_ _the_ _PC_ _prior_ _whose_ _median_ _is_ _equal_ _to_ _ω_ 0 _is_



_λ|d_ _[′]_ ( _ω_ ) _|_
2[1 _−_ exp( _−λd_ (0))] [exp (] _[−][λd]_ [ (] _[ω]_ [))] _[,]_ 0 _< ω_ _< ω_ 0 _,_

_λ|d_ _[′]_ ( _ω_ ) _|_
2[1 _−_ exp( _−λd_ (1))] [exp (] _[−][λd]_ [ (] _[ω]_ [))] _[,]_ _ω_ 0 _< ω_ _<_ 1 _,_



_π_ ( _ω_ ) =



⎧
⎨


⎩



_where_ _λ >_ 0 _is_ _a_ _hyperparameter._ _We_ _suggest_ _to_ _set_ _λ_ _so_ _that_


P(logit(1 _/_ 4) + logit( _ω_ 0) _<_ logit( _ω_ ) _<_ logit( _ω_ 0) + logit(3 _/_ 4)) = 1 _/_ 2 _._


_Base_ _model_ _equal_ _to_ _ω_ 0 = 1 _follows_ _directly_ _by_ _reversing_ _the_ _roles_ _of_ _**u**_ 1 _and_ _**u**_ 2 _._


_Proof._ See Section S1.1 in the Supplementary Materials.


The default values in each case are specified as to place most of the prior mass in
a small interval on the _ω_ scale around _ω_ 0, but to also ensure large deviations from _ω_ 0
are _a_ _priori_ plausible; in this sense they are weakly informative (Gelman, 2006; Gelman
et al., 2008). Sections 5.1 and 5.2 show that the results from the inference are stable
to changes in these hyperparameters; which in turn shows that these _λ_ ’s provide weak
information. If the analyst has expert knowledge this should be used instead of the
default values. Large _ω_ might be 0.75 for test-retest reliability in a psychology study
(Cicchetti, 1994) but 0.4 for the genetic heritability of a trait (Shen et al., 2016).


1If this were not the case, some elements of the sum of **A** 1 _**u**_ 1 and **A** 2 _**u**_ 2 would be exactly equal and
we would choose a subset of maximal size so that Σ [˜] 1 + Σ [˜] 2 was non-singular for comparing the effects
of **A** 1 _**u**_ 1 and **A** 2 _**u**_ 2.


1118 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


**3.3** **Expressing** **a** **priori** **ignorance** **about** **a** **split**


**Exchangeability**


In some cases the analyst does not want to express an _a_ _priori_ preference for any of the
branches in a split in the tree. This can be achieved indirectly through a series of dual
splits. For example, by replacing the split in Figure 1a by the series of dual splits as
shown in Figure 1b where the left-hand side has a base model of 2 _/_ 3 in the first split and
the left-hand side has a base model of 1 _/_ 2 for the second split. In total this is specifying
a base model of 1 _/_ 3 of the total variance to each random effect, but the resulting prior
is not invariant to permutations of A, B and C in Figure 1b. See Section S2 of the
Supplementary Materials for details. When the goal is to express ignorance about the
decomposition of the variance, one can use a base model of equal attribution of the total
variance to each random effect and choose an exchangeable prior for ( _σ_ A [2] _[, σ]_ B [2] _[, σ]_ B [2] [).] [This]
can be done, for example, through a Dirichlet prior.


**Dirichlet** **prior**


The Dirichlet prior of order _K_ _≥_ 2 with parameters _a_ 1 _, . . ., aK_ _>_ 0 is given by



1
_π_ ( _**ω**_ ) =
_B_ ( _a_ 1 _, . . ., aK_ )




- _K_

_ωk_ _[a][k][−]_ [1] _,_ _**ω**_ = ( _ω_ 1 _, . . ., ωK_ ) _∈_ Δ _[K]_ _,_
_k_ =1



where _B_ is the multivariate beta function, and Δ _[K]_ is the _K −_ 1 simplex. Since there is
no preference for any random effect, we consider the symmetric Dirichlet distribution
where _a_ 1 = _. . ._ = _aK_ = _a_ _>_ 0, where _a_ is the hyperparameter that must be selected
by the analyst. For _a_ = 1 the prior is uniform, for _a_ _<_ 1 the prior has peaks at the
vertrices of Δ _[K]_, and for _a_ _>_ 1 the mode is _**ω**_ = (1 _, . . .,_ 1) _/K_ . The prior is invariant
to permutations of the elements of _**ω**_ for any value of _a_ _>_ 0 and it is computationally
cheap for arbitrary dimensions _K_ .


The hyperparameter _a_ can be selected by considering the marginal properties of
_π_ ( _**ω**_ ). The marginal prior _π_ ( _ω_ 1) _∝_ _ω_ 1 _[a][−]_ [1] (1 _−_ _ω_ 1) [(] _[K][−]_ [1)] _[a][−]_ [1], 0 _<_ _ω_ 1 _<_ 1, is a Beta
distribution whose quantiles are dependent both on the values of _a_ and _K_ . We select _a_
by requiring P(logit(1 _/_ 4) _<_ logit( _ω_ 1) _−_ logit( _ω_ 0) _<_ logit(3 _/_ 4)) = 1 _/_ 2. By symmetry the
same marginal properties are satisfied for _ωi_, _i_ = 2 _, . . ., K_ .

## **4 Hierarchical decomposition priors for LGMs**


In this section we introduce the new class of intuitive joint priors for the variance
parameters in LGMs.


**4.1** **Accounting** **for** **model** **structure**


In the general formulation of HD priors in Definition 1, the prior is composed of conditional priors that for each split depends on all descendant splits. This is impractical


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1119


because computing PC priors would require new KLDs to be computed every time the
prior is evaluated. We take a pragmatic approach where we decide on a set of base
models, which expresses our best prior guess, and condition on these.


**Assumption 2** (Simplified conditioning) **.** _For a given tree with S_ _splits and base models_
_{_ _**ω**_ 1 [0] _[, . . .,]_ _**[ ω]**_ _S_ [0] _[}][,]_ _[we]_ _[replace]_ _[π]_ [(] _**[ω]**_ _[s][|{]_ _**[ω]**_ _[j][}][j][∈]_ [D(] _[s]_ [)][)] _[with]_ _[π]_ [(] _**[ω]**_ _[s][|{]_ _**[ω]**_ _[j]_ [=] _**[ ω]**_ _j_ [0] _[}][j][∈]_ [D(] _[s]_ [)][)] _[,]_ _[s]_ [ = 1] _[, . . ., S][.]_


Under this assumption a new class of HD priors for LGMs are constructed by combining intuition about shrinkage and ignorance through independent priors for the splits.


**Prior class 1** (HD priors for LGMs) **.** _Assume the LGM contains N_ _random effects with_
_variances σ_ 1 [2] _[, . . ., σ]_ _N_ [2] _[and that the hierarchical decomposition of the variance is described]_
_through_ _a_ _tree_ _with_ _S_ _splits._ _Under_ _base_ _models_ _{_ _**ω**_ 1 [0] _[, . . .,]_ _**[ ω]**_ _S_ [0] _[}][,]_ _[the]_ _[prior]_ _[is]_



_π_ ( _σ_ 1 [2] _[, . . ., σ]_ _N_ [2] [) =] _[ π]_ [(] _[t][|{]_ _**[ω]**_ _[s][}][S]_ _s_ =1 [)]




- _S_

_π_ ( _**ω**_ _s|{_ _**ω**_ _j_ = _**ω**_ _j_ [0] _[}]_ _j∈_ D( _s_ ) [)] _[,]_
_s_ =1



_where the total latent variance is t_ = _σ_ 1 [2] [+] _[. . .]_ [+] _[σ]_ _N_ [2] _[, and]_ _**[ ω]**_ _[i]_ _[∈]_ [Δ] _[l][s]_ _[, where][ l][s]_ _[is the number]_
_of_ _branches_ _in_ _split_ _s,_ _s_ = 1 _, . . ., S._


_For_ _each_ _of_ _the_ _S_ _splits,_ _the_ _analyst_ _can_ _express_ _ignorance_ _through_ _a_ _Dirichlet_ _prior_
_or sequence of PC priors as described in Section 3.3, or express preference to the selected_
_base_ _models_ _as_ _described_ _in_ _Section_ _3.2._ _The_ _selection_ _of_ _π_ ( _t|{_ _**ω**_ _s}_ _[S]_ _s_ =1 [)] _[must]_ _[be]_ _[done]_ _[in]_
_the_ _context_ _of_ _the_ _likelihood_ _as_ _described_ _in_ _Section_ _4.2._


This prior is computationally inexpensive since the overall prior probability density
factorises into independent conditional distributions that consist of PC priors, which
can be precomputed, and Dirichlet priors, which are cheap to compute.


We demonstrate the use of HD priors through one example where the analyst wants
to express ignorance and one example where the analyst wants to penalise complexity.


**Example** **2** (Non-nested random effects) **.** Consider responses _y_ 1 _, . . ., yn_, described by
the Gaussian linear model _yi|ηi_ _∼N_ ( _ηi, σ_ R [2] [)] [with]


_ηi_ = _μ_ + _h_ 1(Age _i_ ) + _h_ 2(Weight _i_ ) + _h_ 3(Income _i_ ) _,_ _i_ = 1 _,_ 2 _, . . ., n,_


where _μ_ is the intercept, _h_ 1, _h_ 2 and _h_ 3 are smooth effects of the covariates expressed
as second-order random walks (Rue and Held, 2005), and _σ_ R [2] [is] [the] [residual] [variance.]
Assume that one has no _a_ _priori_ preference for the three smooth effects, and decide to
encode the decomposition of the total latent variance as shown Figure 1a, where A, B
and C represents the three smooth of covariates effects. Let _**ω**_ 1 denote the proportions
of variance assigned to model components and let _t_ denote the total latent variance. We
construct an HD prior by assigning a Dirichlet prior to _**ω**_ 1, and handle _t|_ _**ω**_ 1 as discussed
in Section 4.2. _△_


**Example** **3** (Shrinkage in multilevel models) **.** The latent part of the multilevel model
in Section 1 can be written in vector form as _**η**_ = **A** A _**u**_ A + **A** B _**u**_ B + **A** C _**u**_ C, where
**A** A, **A** B and **A** C are sparse matrices selecting the appropriate group, individual and


1120 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


measurement effects, respectively. Assume we use an LGM, then _**u**_ 1 _∼NG_ ( **0** _, σ_ A [2] **[I]** _[G]_ [),]
_**u**_ 2 _∼NGP_ ( **0** _, σ_ B [2] **[I]** _[GP]_ [ )] [and] _**[u]**_ [3] _[∼N][GP K]_ [(] **[0]** _[, σ]_ C [2] **[I]** _[GP K]_ [),] [where] _[G]_ [is] [the] [number] [of] [groups,]
_P_ is the number of individuals per group, and _K_ is the number of measurements per
individual.


If we prefer shrinkage towards fewer levels in the multilevel model as shown in
Figure 1c, we decompose the total latent variance _t_ = _σ_ A [2] [+] _[σ]_ B [2] [+] _[σ]_ C [2] [through] [two]
splits. For the split at the root node, we decompose _t_ according to the proportions
_**ω**_ 1 = ( _σ_ A [2] [+] _[ σ]_ B [2] _[, σ]_ C [2] [)] _[/t]_ [.] [Then] [in] [the] [second] [split] [we] [decompose] _[σ]_ A [2] [+] _[ σ]_ B [2] [according] [to] [the]
proportions _**ω**_ 2 = ( _σ_ A [2] _[, σ]_ B [2] [)] _[/]_ [(] _[σ]_ A [2] [+] _[ σ]_ B [2] [).]

We use an HD prior where we apply base models _**ω**_ 1 [0] [=] [(0] _[,]_ [ 1),] [which] [prefers] [C]
over A+B, and _**ω**_ 2 [0] [=] [(0] _[,]_ [ 1),] [which] [prefers] [B] [over] [A.] [Due] [to] [the] [desire] [for] [shrinkage]
we apply PC priors and use Theorem 1 with base model _**ω**_ 2 [0] [to] [compute] _[π]_ [(] _**[ω]**_ [2][).] [We]
define _**u**_ ˜1 = **A** A _**u**_ A + **A** B _**u**_ B and _**u**_ ˜2 = **A** C _**u**_ C. Then if we condition on _**ω**_ 2, the top
split in Figure 1c compares _**u**_ ˜1 _|_ _**ω**_ 2 _∼Nn_ ( **0** _,_ ( _σ_ A [2] [+] _[ σ]_ B [2] [)(] _[ω]_ [2] _[,]_ [1] **[A]** [A] **[A]** [T] A [+] _[ ω]_ [2] _[,]_ [2] **[A]** [B] **[A]** B [T][))] [and]
_**u**_ ˜2 _∼Nn_ ( **0** _, σ_ 3 [2] **[A]** [3] **[A]** [T] 3 [),] [and] [the] [conditional] [prior] _[π]_ [(] _**[ω]**_ [1] _[|]_ _**[ω]**_ [2] [=] _**[ω]**_ 2 [0][)] [can] [be] [computed]
using Theorem 1 with base model _**ω**_ 1 [0] [conditional] [on] _**[ω]**_ [2] [=] _**[ω]**_ 2 [0][.] [The] [joint] [prior] [is] [then]
_π_ ( _**ω**_ 1 _,_ _**ω**_ 2) = _π_ ( _**ω**_ 1 _|_ _**ω**_ 2 = _**ω**_ 2 [0][)] _[π]_ [(] _**[ω]**_ [2][),] [and] [an] [appropriate] [prior] [is] [chosen] [for] _[π]_ [(] _[t][|]_ _**[ω]**_ [1] _[,]_ _**[ ω]**_ [2][)] [as]
described in Section 4.2. _△_


**4.2** **Accounting** **for** **the** **likelihood**


Meaningful priors for the total latent variance _t_ depend on the likelihood and prior
beliefs about the responses in the specific application (Gelman et al., 2017). We provide
tools for expressing scale-invariance for the variances of the random effects and the
measurement error when the responses are Gaussian, or shrinkage for the total latent
variance of the random effects.


Under a Gaussian likelihood, the selection of the unit of measurement by the analyst
affects the sizes of the variances. However, when the residual variance _σ_ R [2] [is expected to]
be well-identified, we can define the prior on _t_ relative to _σ_ R [2] [and] [shrink] _[t]_ [by] [preferring]
to describe the total variance _V_ = _t_ + _σ_ R [2] [in the model by] _[ σ]_ R [2] [. This can be complemented]
by a scale-independent Jeffreys’ prior on _V_ to achieve a scale-invariant joint prior for
the variance parameters.


**Prior** **class** **2** (HD priors with Gaussian likelihoods) **.** _Assume_ _an_ _HD_ _prior_ _from_ _Prior_
_class_ _1_ _is_ _desired_ _for_ _an_ _LGM_ _with_ _Gaussian_ _responses_ _with_ _residual_ _variance_ _σ_ R [2] _[.]_ _[First]_
_select_ _the_ _prior_ _on_ _the_ _decomposition_ _of_ _the_ _total_ _latent_ _variance_ _t._ _Then_ _augment_ _the_
_tree_ _by_ _an_ _extra_ _node_ _on_ _the_ _top_ _with_ _variance_ _V_ = _t_ + _σ_ R [2] _[.]_ _[The]_ _[new]_ _[top]_ _[node]_ _[has]_
_one_ _branch_ _with_ _residual_ _variance_ _and_ _the_ _other_ _branch_ _is_ _the_ _subtree_ _describing_ _the_
_latent_ _model._ _Let_ _**ω**_ R = (1 _−_ _σ_ R [2] _[/V, σ]_ R [2] _[/V]_ [ )] _[and]_ _[assume]_ _[shrinkage]_ _[through]_ _[a]_ _[PC]_ _[prior]_
_π_ ( _**ω**_ R _|{_ _**ω**_ _s_ = _**ω**_ _s_ [0] _[}][S]_ _s_ =1 [)] _[with]_ _[base]_ _[model]_ _**[ω]**_ R [0] [= (0] _[,]_ [ 1)] _[.]_


_If_ _V_ _is_ _assigned_ _a_ _scale-invariant_ _prior,_ _the_ _full_ _joint_ _prior_ _is_


_π_ ( _V,_ _**ω**_ R _, {_ _**ω**_ _s}_ _[S]_ _s_ =1 [)] _[ ∝]_ _[π]_ [(] _**[ω]**_ [R] _[|{]_ _**[ω]**_ _[s]_ [=] _**[ ω]**_ _s_ [0] _[}][S]_ _s_ =1 [)] _[π]_ [(] _[{]_ _**[ω]**_ _[s][}]_ _s_ _[S]_ =1 [)] _[/V,]_ _V_ _>_ 0 _,_ _**ω**_ R _∈_ Δ [2] _,_


_and_ _**ω**_ _s_ _∈_ Δ _[l][s]_ _,_ _where_ _ls_ _is_ _the_ _number_ _of_ _branches_ _in_ _split_ _s,_ _for_ _s_ = 1 _, . . ., S._


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1121


_Proof._ The scale-invariant prior is _π_ ( _V |_ _**ω**_ R _, {_ _**ω**_ _s}_ _[S]_ _s_ =1 [)] _[∝]_ [1] _[/V]_ [,] [and] _[π]_ [(] _**[ω]**_ [R] _[,][ {]_ _**[ω]**_ _[s][}][S]_ _s_ =1 [)] [=]
_π_ ( _**ω**_ R _|{_ _**ω**_ _s}_ _[S]_ _s_ =1 [)] _[π]_ [(] _[{]_ _**[ω]**_ _[s][}]_ _s_ _[S]_ =1 [)]


If the likelihood is binomial with a logit link function, a scale for the random effects
is induced through their effects on the odds-ratio. Similarly, for a Poisson likelihood
with a log link function, there is a scale for the random effects through their effects on
the relative risk. In these cases, scale-invariance is not meaningful and we can induce
shrinkage on the total variance of the random effects by using the PC prior for variance
from Simpson et al. (2017).


**Prior** **class** **3** (HD priors with shrinkage on latent variance) **.** _Assume_ _an_ _HD_ _prior_
_from_ _Prior_ _class_ _1_ _is_ _desired_ _for_ _an_ _LGM_ _where_ _shrinkage_ _on_ _the_ _total_ _latent_ _variance_
_is_ _appropriate._ _First_ _select_ _the_ _prior_ _on_ _the_ _decomposition_ _of_ _the_ _total_ _latent_ _variance_
_t._ _Then_ _t_ _can_ _be_ _shrunk_ _towards_ 0 _by_ _a_ _PC_ _prior_ _π_ ( _t|{_ _**ω**_ _s}_ _[S]_ _s_ =1 [)] _[with]_ _[base]_ _[model]_ _[t]_ [0] [= 0] _[.]_
_This_ _results_ _in_
_λ_ _√_
_π_ ( _t, {_ _**ω**_ _s}_ _[S]_ _s_ =1 [) =] ~~_√_~~ _t_ ) _π_ ( _{_ _**ω**_ _s}_ _[S]_ _s_ =1 [)] _[,]_
2 _t_ [exp(] _[−][λ]_

_t >_ 0 _,_ _and_ _**ω**_ _i_ _∈_ Δ _[l][s]_ _,_ _where_ _ls_ _is_ _the_ _number_ _of_ _branches_ _in_ _split_ _s,_ _for_ _s_ = 1 _, . . ., S,_ _and_
_λ >_ 0 _is_ _a_ _hyperparameter._



_√_
_t_ [exp(] _[−][λ]_



_t_ ) _π_ ( _{_ _**ω**_ _s}_ _[S]_ _s_ =1 [)] _[,]_



_λ_ exp( _−λ_ _t_ ) _/_ (2 _t_ ), _t >_ 0 (Simpson et al., 2017).



~~_√_~~
_t_ ) _/_ (2



_t_ ), _t >_ 0 (Simpson et al., 2017).



We illustrate how the hyperparameter can be selected by considering the prior on
the total latent variance in the case of a Binomial likelihood.


**Example** **4** (Shrinking latent variance) **.** Let logit( _p_ ) = _μ_ + _x_, where _x_ _∼N_ (0 _, t_ ), for
a _t >_ 0, and _μ_ is considered fixed. The latent variance _t_ is difficult to interpret directly
due to the non-linear link function, but we can interpret it through the effect on the
odds-ratio, _p/_ (1 _−_ _p_ ) = exp( _μ_ ) exp( _x_ ). The hyperparameter _λ_ in Prior class 3 can, for
example, be set so that the relative change in the odds-ratio, exp( _x_ ), is between 1 _/_ 2
and 2 with probability 90%, P(1 _/_ 2 _<_ exp( _x_ ) _<_ 2) = 0.90. _△_

## **5 Case studies: Gaussian responses**


In this section we investigate the performance of HD priors compared to a set of commonly used standard priors for two simulation studies with Gaussian responses.


**5.1** **Random** **intercept** **model**


The _random_ _intercept_ _model_ is given by _yi,j_ = _αi_ + _εi,j_ for _j_ = 1 _, . . ., ni_, _i_ = 1 _, . . ., n_ g,
where _ni_ is the size of group _i_, and _n_ g is the number of groups. The random intercepts are
i.i.d. Gaussian with variance _σα_ [2] [and the residual effects are i.i.d. Gaussian with variance]
_σ_ R [2] [.] [The] [total] [latent] [variance] [is] _[t]_ [=] _[σ]_ _α_ [2] [and] [the] [total] [variance] [is] _[V]_ [=] _[σ]_ R [2] [+] _[ σ]_ _α_ [2] [.] [We]
introduce the proportion of the total variance explained by the latent model _ω_ = _σα_ [2] _[/V]_ [,]
and decompose _V_ as _σα_ [2] [=] _[ωV]_ [and] _[σ]_ R [2] [=] [(1] _[ −]_ _[ω]_ [)] _[V]_ [ .] [We] [desire] [shrinkage] [towards] [the]


1122 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


Figure 2: Model structure and prior for _ω_ in the random intercept model with 10 individuals in each group and prior median _ω_ m = 0 _._ 25. The prior is independent of the
number of groups. a) Tree structure, b) prior for _ω_, and c) prior for distance _d_ ( _ω_ ).


base model _ω_ [0] = 0 and use an HD prior based on the tree structure in Figure 2a, where
the prior on _ω_ is calculated using Theorem 1 and we use the scale-invariant prior from
Prior class 2. The specification of the hyperparameter of the HD prior is done through
the median _ω_ m of _π_ ( _ω_ ). The resulting prior for _ω_ is shown in Figure 2b for _ω_ m = 0 _._ 25
and the corresponding prior for the distance _d_ ( _ω_ ) discussed in Section 3.2 is shown in
2c. Further details can be found in Section S3.1 of the Supplementary Materials.


The intraclass correlation (ICC) for the random intercept model is given by _σα_ [2] _[/]_ [(] _[σ]_ R [2] [+]
_σα_ [2] [),] [which equals] [the] [weight] [parameter] _[ ω]_ [.] [Thus] [the] [shrinkage] [of] [the] [ICC] [is completely]
controlled in the construction of the prior and expert knowledge about the ICC can be
incorporated directly. Further, _ω_ can be linked to a generalised version of the coefficient
of determination, _R_ [2], suggested by Gelman and Hill (2007); see Section S3.2 in the
Supplementary Materials for details.


We use the R-package RStan (Stan Development Team, 2018a) to perform the inference for the simulation study. We use HD priors from Prior class 2 with shrinkage
from PC priors on _ω_ with hyperparameters _ω_ m = 0 _._ 25 (P-HD-25), _ω_ m = 0 _._ 5 (P-HD-50)
and _ω_ m = 0 _._ 75 (P-HD-75), and an HD prior from Prior class 2 where the PC prior
is replaced by a Dirichlet prior on ( _ω,_ 1 _−_ _ω_ ) (P-HD-D) with default hyperparameter.
Additional priors are Jeffreys’ prior on the residual variance combined with different
priors on the random intercepts variance or standard deviation: the default INLA prior
InvGamma(1 _,_ 5 _×_ 10 _[−]_ [5] ) (P-INLA), Half-Cauchy(25) (P-HC), and PCSD(3 _,_ 0 _._ 05) (P-PC).
This gives seven joint priors. Each scenario in the simulation study consists of 500
datasets which are simulated from the random intercept model for _n_ g _∈{_ 5 _,_ 10 _,_ 50 _}_,
and 10, 50, or varying number of individuals in each group. We select true values
_ω_ _∈{_ 0 _._ 1 _,_ 0 _._ 25 _,_ 0 _._ 5 _,_ 0 _._ 75 _,_ 0 _._ 9 _}_ and select true total variance _V_ = 1 in every scenario.


We evaluate the performance of the different priors with respect to posterior inference for total variance _V_ and ICC _ω_ . We use the bias of log( _V_ ) and logit( _ω_ ), calculated
using the estimated median minus the true value, and the 80% empirical coverage, found
by counting the number of times the true value is contained in the 80% equal-tailed
credible interval. We use the same settings for the call to the stan function for all priors and scenarios in the simulation study. RStan reports a _divergent_ _transition_ for each
iteration of the Markov chain Monte Carlo (MCMC) sampler that runs into numerical
instabilities (Carpenter et al., 2017). In Figure S3.1 in the Supplementary Materials we


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1123


Figure 3: Results for logit( _ω_ ) for the random intercept simulation study. True value of
_ω_ shown on the _x_ -axis, the number of groups is shown on left-hand side, and the group
size is 10. Results for P-INLA are only shown when it leads to stable inference.


report the proportion of datasets that resulted in at most 0.1% divergent transitions for
each prior and scenario. This is used as a measure of stability of the inference scheme
for each prior, and the dataset and prior combinations causing unstable inference are
removed from the study.


The results in Figure 3 are for _ng_ _∈{_ 10 _,_ 50 _}_ and group size 10, and show that PHD-25 performs at least as good in terms of bias and coverage of logit( _ω_ ) as P-INLA,
P-HC and P-PC. The magnitude of the bias decreases and the coverage approaches
80% for all four priors when the number of groups increases, which is expected as the
amount of information about the parameters in the datasets increases. Figures S3.3–
S3.7 in the Supplementary Materials show that the HD priors perform at least as good
in terms of bias and coverage for logit( _ω_ ) as P-INLA, P-HC and P-PC also for the other
combinations of the number of groups and group sizes, and that the same conclusions
as for logit( _ω_ ) also holds for log( _V_ ).


Furthermore, Figures S3.3–S3.7 show that the behaviour of the four HD priors is
stable with respect to the choice of _ω_ m when group size is 10, and that P-HD-D performs
worse than P-HD-25, P-HD-50 and P-HD-75 for all values of the true weight except 0.5.
For 10 groups with two observations per group, the risk of overfitting is high because
low information about the parameters may lead to overestimating the weight parameter
and estimating spurious signals in the group effect. In this setting, P-HD-25 leads to
overfitting for true weight equal to 0 _._ 1, but underfitting for true weight equal to 0 _._ 25,
0 _._ 5, 0 _._ 75 and 0 _._ 9. P-HD-50, P-HD-75 and P-HD-D result in overfitting for true weight
equal to 0 _._ 1 and 0 _._ 25, but underfitting for true weight equal to 0 _._ 5, 0 _._ 75 and 0 _._ 9. See
Section S3.4 in the Supplementary Materials for additional details.


Figure S3.1 shows that P-INLA is the only prior that is heavily affected by divergent
transitions during the inference for scenarios with 10 or 50 groups. Part of the problem
with P-INLA is that it results in a bi-modal posterior for _σα_ [2] [;] [see] [Figure] [S3.2.] [The]


1124 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


new HD priors are preferred for the random intercept model due to their intuitive
definition, where the structure of the shrinkage is directly available in Figure 2a, and
interpretability of the parametrization which aids prior elicitation.


**5.2** **Latin** **square** **experiment**


Consider an experiment where a latin square design (Hinkelmann and Kempthorne,
1994) is used to control for two nuisance sources of noise. For example, a field split into
rows and columns where different levels of strength of a new fertilizer is applied to each
plot. We assume there are nine possible levels of the treatment so that a 9 _×_ 9 grid of
plots is necessary for a full latin square design. We focus on random effects and exclude
fixed effects from the model, and assume that the responses can be modelled by


_yi,j_ = _αi_ + _βj_ + _γk_ [ _i,j_ ] + _εi,j,_ _i, j_ = 1 _, . . .,_ 9 _,_ (5.1)

where _**α**_ = ( _α_ 1 _, . . ., α_ 9) _∼N_ 9( **0** _, σ_ r [2] **[I]** [9][)] [is] [an] [i.i.d.] [effect] [of] [row,] _**[β]**_ [=] [(] _[β]_ [1] _[, . . ., β]_ [9][)] _[∼]_
_N_ 9( **0** _, σ_ c [2] **[I]** [9][)] [is] [an] [i.i.d.] [effect] [of] [column,] _**[γ]**_ [=] [(] _[γ]_ [1] _[, . . ., γ]_ [9][)] [is] [the] [effect] [of] [the] [treatment,]
_k_ [ _i, j_ ] denotes the treatment assigned to row _i_ and column _j_, and _**ε**_ = ( _ε_ 1 _,_ 1 _, . . ., ε_ 9 _,_ 9) _∼_
_N_ 81( **0** _, σ_ R [2] **[I]** [81][)] [is] [the] [residual] [noise.]


We believe that the effect of the treatment is ordered, and that the treatment effect
consists of a smooth signal of interest _**γ**_ [(1)] = ( _γ_ 1 [(1)] _[, . . ., γ]_ 9 [(1)][)] [and] [random] [noise] _**[γ]**_ [(2)] [=]
( _γ_ 1 [(2)] _[, . . ., γ]_ 9 [(2)][)] [we] [have] [to] [control] [for.] [The] [signal] [is] [given] [a] [second-order] [random] [walk]
model described by _N_ 9( **0** _, σ_ RW2 [2] **[Q]** _[−]_ RW2 [1] [),] [where] _[σ]_ RW2 [2] [is] [the] [variance] [and] **[Q]** _[−]_ RW2 [1] [is] [a]
slight abuse of notation to describe the intrinsic second-order random walk defined by
the precision matrix **Q** RW2, and the noise is _**γ**_ [(2)] _∼N_ 9( **0** _, σ_ t [2] **[I]** [9][).] [We] [use] [the] [constraints]
�9
_i_ =1 _[γ]_ _i_ [(1)] = 0 and [�][9] _i_ =1 _[iγ]_ _i_ [(1)] = 0 to remove the implicit intercept and linear effect,
respectively.


We set the true standard deviations equal,� _σ_ r = _σ_ c =  - _σ_ t = _σ_ R = 0 _._ 1, and let the
true effect of treatment be given by _xi_ = _C_ ( _i −_ 5) [2] _−_ 20 _/_ 3, _i_ = 1 _, . . .,_ 9. We entertain
three scenarios: _C_ = 0 for no effect of treatment (S1), _C_ = 0 _._ 05 for medium effect of
treatment (S2) and _C_ = 0 _._ 2 for strong effect of treatment (S3). More details on the
true treatment effect is included in Section S4.1 in the Supplementary materials, see
especially Figure S4.2. We simulate 500 datasets for each scenario and analyse them
with four choices of priors.


The three default priors used are Jeffreys’ prior for _σ_ R [2] combined with
InvGamma(1 _,_ 5 _×_ 10 _[−]_ [5] ) for _σ_ r [2][,] _[ σ]_ c [2][,] _[ σ]_ t [2] [and] _[ σ]_ RW2 [2] [(P-INLA), or Half-Cauchy(25) (P-HC)]
or PCSD(3 _,_ 0 _._ 05) (P-PC) for _σ_ r, _σ_ c, _σ_ t and _σ_ RW2. We select an HD prior from Prior
class 2 using the model structure in Figure 4a, where the triple split has a Dirichlet
prior and the two other splits have PC priors (P-HD-D3). We also decompose the triple
split into the two dual splits as shown in Figure 4b, and use a PC prior on all four
splits according to the shrinkage structure in the figure (P-HD-25). In all cases we use
default values for the hyperparameters. See Section S2 in the Supplementary Materials
for more details on changing a triple split to two dual splits. Figures S4.3, S4.4, S4.10
and S4.11 in the Supplementary Materials show that the implementation of the triple
split has little influence on the targets of the analysis.


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1125


Figure 4: Model structure for the latin square simulation study. Gray nodes indicate
base models. (1 _/_ 3 _,_ 1 _/_ 3 _,_ 1 _/_ 3), (1 _/_ 3 _,_ 2 _/_ 3), and (1 _/_ 2 _,_ 1 _/_ 2) indicates that the base model for
the split is a combination of the branches. a) Original, and b) alternative structure.


Figure 5: Results from the latin square experiment simulation study.



The targets of the analysis are the posterior distribution of the structured treatment
effect _**γ**_ [(1)] and the model fit. The former will be assessed by the continuous rank probability score (CRPS) (Gneiting and Raftery, 2007) and the latter by the leave-one-out
log predictive score (LOO-LPS) _−_ [1] �81 [log] _[ π]_ [(] _[y][i][|]_ _**[y]**_ _[−][i]_ [).] [The] [CRPS] [is] [a] [proper] [scoring]



logrulepredictiveand givenscoreby [1] 9 (LOO-LPS)�9 _i_ =1 - _−∞∞_ [(] _[F]_ _−_ _[i]_ [(] 81 [1] _[x]_ [)] - _[ −]_ 81 _i_ [I] =1 [(] _[x]_ [log] _[ ≥]_ _[ π][x][i]_ [(][))] _[y][i]_ [2] _[|]_ [ d] _**[y]**_ _[−][x][i]_ [,][).][where][The] [CRPS] _[F][i]_ [is] [the][is] [a][cumulative][proper] [scoring][dis-]

tribution function for the posterior of _γi_ [(1)], _xi_ is the true effect of treatment _i_, and I is
the Heaviside function, and is estimated using the procedure of Jordan et al. (2017). We
report the proportion of datasets leading to no more than 0.1% divergent transitions for
each prior and scenario, and use this as a measure on stability of the inference. These
numbers can be seen in Figure S4.5 in the Supplementary Materials, and show that
all priors lead to similar stability. The datasets leading to more than 0.1% divergent
transitions for one or more priors are removed from the study.


The main results from the simulation study are displayed in Figure 5. Low LOO-LPS
indicates good model fit and low CRPS indicates good predictive power for the treatment effect. P-INLA gives a poorer model fit than the other priors, and with respect




[1] 9 �9 _i_ =1 - _−∞∞_ [(] _[F][i]_ [(] 81 _[x]_ [)] _[ −]_ _i_ [I] =1 [(] _[x][ ≥]_ _[x][i]_ [))][2][ d] _[x]_ [,] [where] _[F][i]_ [is] [the] [cumulative] [dis-]


1126 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


to predictive power, the HD priors P-HD-D3 and P-HD-25 perform best for S2 and
S3. The high predictive power of P-INLA for S1 is due to the fact that P-INLA has a
peak at low variance and produces a posterior for the treatment effect with mean closer
to zero and lower variance. Overall, the HD prior performs well across all scenarios.
The results are stable to changes in the construction of the HD prior and the choice of
hyperparameters; see Section S4.2 in the Supplementary Materials for details. The HD
priors are preferable to the other priors because of their intuitive parametrization and
the interpretability of the _a_ _priori_ assumptions placed on the joint prior of the variance
parameters. Further, P-HD-D3 is preferred to P-HD-25 since they perform similar and
P-HD-D3 is more intuitive.

## **6 Case studies: Binomial responses**


In this section we study neonatal mortality counts arising from complex surveys through
a simulation study, and show how to practically apply the HD priors.


**6.1** **Background**


Neonatal mortality is an important indicator of health and well-being in a country and is
included in Goal 3.2 of the Sustainable Development Goals (SDGs) (General Assembly of
the United Nations, 2015), and mapping child mortality is an important area of current
research (Golding et al., 2017; Wakefield et al., 2019; Li et al., 2019). We define neonatal
mortality as the rate of deaths within the first month of life per live birth. An important
source of data for neonatal mortality is the nationally-representative household surveys
performed by Demographic and Health Surveys (DHS). The survey performed by DHS
in 2014 in Kenya targets its 47 counties, which is the relevant administrative level
for health policies (Kenya National Bureau of Statistics et al., 2015). The target of the
simulation study in Section 6.2 and the analysis in Section 6.3 is the spatial heterogeneity
in neonatal mortality in Kenya in the time period 2010 to the time of the survey.


From the survey we can extract the number of live births, _bi,j,k_, and the number of
neonatal deaths, _yi,j,k_, in household _k_ in cluster _j_ in county _i_ . We also have an indicator
_xi,j_ specifying whether the cluster is rural (0) or urban (1) and each household has an
inclusion probability _πi,j,k_ of being included in the survey sample. See the Section S5.1
in the Supplementary Materials for more background.


**6.2** **Simulation** **study**


In this section we use the _n_ = 290 constituencies shown in Figure 6a. [2] We assume
that _mi_ = 6 clusters are visited in constituency _i_, _i_ = 1 _, . . ., n_, and consider births
_bi,j_ and neonatal deaths _yi,j_ in cluster _j_ in constituency _i_ . We assume that there are
_bi,j_ = 25 live births in each cluster and the outcomes are simulated according to the


2Preliminary investigations revealed that 47 counties provided too little information to learn about
model structure in the data. We instead use the 290 constituencies of Kenya for the simulations study.


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1127


Figure 6: Map and model structure for the Kenya neonatal mortality simulation study.


model _yi,j|pi,j_ _∼_ Binomial( _bi,j, pi,j_ ) for


logit( _pi,j_ ) = _ηi,j_ = _μ_ + _ui_ + _vi_ + _νi,j,_ _j_ = 1 _, . . ., mi,_ _i_ = 1 _, . . ., n,_


where _μ_ is a joint intercept, _**u**_ = ( _u_ 1 _, . . ., un_ ) has a Besag distribution with variance _σ_ B [2] [and] [a] [sum-to-zero] [constraint,] _**[v]**_ [=] [(] _[v]_ [1] _[, . . ., v][n]_ [)] _[∼N][n]_ [(] **[0]** _[, σ]_ IID [2] **[I]** _[n]_ [),] [and] _**[ν]**_ [=]
( _ν_ 1 _,_ 1 _, . . ., νn,mn_ ) _∼NM_ ( **0** _, σ_ C [2] **[I]** _[M]_ [)] [with] _[M]_ [=] _[ m]_ [1][ +] _[ . . .]_ [ +] _[ m][n]_ [= 6] _[ ·]_ [ 290 = 1740.]


We use the structure for the prior shown in Figure 6b to make an HD prior from
Prior class 3 with PC priors on all splits according to the base models indicated in the
figure (P-HD-25) and an HD prior from Prior class 3 where a Dirichlet prior distributes
variance to the three model components (P-HD-D). In all cases, the splits have default
hyperparameter values and we select the hyperparameter in the PC prior on total variance, _t_ = _σ_ B [2] [+] _[σ]_ IID [2] [+] _[σ]_ C [2] [, so that P(] _[t >]_ [ 3) = 0] _[.]_ [05. Further, we use InvGamma(1] _[,]_ [ 5] _[×]_ [10] _[−]_ [5][)]
for _σ_ B [2] [,] _[σ]_ IID [2] [and] _[σ]_ C [2] [(P-INLA),] [Half-Cauchy(25)] [for] _[σ]_ [B][,] _[σ]_ [IID] [and] _[σ]_ [C] [(P-HC),] [and] [the]
joint prior proposed in Riebler et al. (2016) (P-PC), where _σ_ B [2] [and] _[σ]_ IID [2] [has] [a] [PC] [prior]
of the type introduced in this paper with P( _σ_ B [2] _[/]_ [(] _[σ]_ B [2] [+] _[ σ]_ IID [2] [)] _[<]_ [0] _[.]_ [5)] [=] [2] _[/]_ [3] [and] _[σ]_ C [2] [is]
given an independent PC prior _σ_ C _∼_ PCSD(3 _,_ 0 _._ 05).


Based on the final report from the survey (Kenya National Bureau of Statistics
et al., 2015) the estimated national level of neonatal mortality is 0 _._ 022 for 2010–2014,
and we set _μ_ = logit(0 _._ 022). Further, we choose _σ_ C [2] [=] [0] _[.]_ [1] [and] [create] [five] [scenarios]
by combining this with _σ_ IID [2] [=] _[σ]_ B [2] [=] [0] [(S1),] _[σ]_ IID [2] [=] [0] _[.]_ [4] [and] _[σ]_ B [2] [=] [0] [(S2),] _[σ]_ IID [2] [=]
_σ_ B [2] [=] [0] _[.]_ [2] [(S3),] _[σ]_ IID [2] [=] [0] _[.]_ [04] [and] _[σ]_ B [2] [=] [0] _[.]_ [36] [(S4),] [and] _[σ]_ IID [2] [=] [0] [and] _[σ]_ B [2] [=] [0] _[.]_ [4] [(S5).] [We]
simulate 500 datasets for each scenario. The main targets of the simulation study are
the structured part of the spatial heterogeneity through the posterior of _**u**_, the degree
of structure in the spatial heterogeneity through _ω_ [(2)] = _σ_ B [2] [(] _[σ]_ B [2] [+] _[ σ]_ IID [2] [)] _[−]_ [1][,] [and] [how] [well]
the underlying neonatal mortality is estimated through the posterior of the intercept
_μ_ . The performance is assessed through the CRPS (see Section 5.2) of _**u**_, the bias of
the posterior median of _ω_ [(2)], and the bias of the posterior median and the coverage of
the 80% equal-tailed credible interval for _μ_ . We use the proportion of datasets leading
to at most 0.1% divergent transitions as a measure of stability in the inference, these


1128 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


Figure 7: Main results from the Kenya neonatal mortality simulation study. Left to right:
bias of the intercept _μ_, CRPS of _**u**_ and bias of _ω_ [(2)] . Scenario shown on the x-axes.


numbers can be seen in Figure S5.1 in the Supplementary Materials, and show that
P-INLA leads to more unstable inference than the others.


Figure 7 shows the main results from the simulation study. We drop datasets that
cause more than 0.1% divergent transitions for at least one of the priors from each
scenario. All priors have a tendency to overestimate the intercept, with P-INLA doing
worse than the others, P-INLA gives close to exact estimates when the true value of _ω_ [(2)]

is 0 (in S2) and 1 (in S5), but performs worse than the other priors for S3 and S4. Figure
S5.2 in the Supplementary Materials shows that P-HD-25 performs better than P-HD-D
except in S3 where the Dirichlet prior is closest to the truth, and that _ω_ [(1)] tends to be
underestimated under all the priors. P-HD-25 is preferred because overall it performs
at least as good as the other priors P-HC and P-PC, and P-HD-25 is an intuitive and
well-behaved prior that takes the hierarchical structure of the model into account.


**6.3** **Neonatal** **mortality** **in** **Kenya**


This section follows the notation introduced in Section 6.1. The survey consists of 13183
households with one or more live births, distributed over 1593 clusters that are distributed over _n_ = 47 counties. In total there are 376 deaths among 17664 children.
Figure 8c shows the counties and the weighted neonatal mortality by the inverse inclusion probabilities, and it is unclear if there is a structured spatial pattern. The
neonatal mortality is assumed to follow a survival model with constant hazard through
the first month of life, and we use a latent Gaussian model with a binomial likelihood,
_yi,j,k|bi,j,k, pi,j,k_ _∼_ Binomial( _bi,j,k, pi,j,k_ ), a logit link function, and a linear latent Gaussian model
_ηi,j,k_ = logit( _pi,j,k_ ) = _μ_ + _xi,jβ_ + _ui_ + _vi_ + _νi,j_ + _εi,j,k,_ (6.1)


where _μ_ is an overall intercept, _β_ is the effect of urban, _**u**_ is a Besag model with variance
_σ_ 11 [2] [,] _**[v]**_ [is] [a] [Gaussian] [i.i.d.] [effect] [of] [county] [with] [variance] _[σ]_ 12 [2] [,] _**[ν]**_ [is] [a] [Gaussian] [i.i.d.] [effect]
of cluster with variance _σ_ 2 [2][,] [and] _**[ε]**_ [is] [a] [Gaussian] [i.i.d.] [effect] [of] [household] [with] [variance]
_σ_ 3 [2][.] [In] [this] [model,] _**[u]**_ [and] _**[v]**_ [provide] [structured] [and] [unstructured,] [respectively,] [between-]
county variation, _**ν**_ provides between-cluster variation, and _**ε**_ provides within-cluster


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1129


variation. The Besag effect has a sum-to-zero constraint to make the overall intercept
identifiable. The random effects of cluster and household are necessary to account for the
dependence induced between sampled households due to the clustering in the sampling
design. We assume that there is no difference between the effect of urbanicity between
different counties.


The model has four variance parameters that must be assigned a joint prior. The
first step is to choose the tree structure. For simplicity’s sake, the alternatives to the full
model (6.1) we would entertain are first _ηi,j,k_ = _μ_ + _xi,jβ_ + _vi_, then we would add _ui_,
so _νi,j_, and at last _εi,j,k_ . We prefer coarser unstructured effects over finer unstructured
effects since we would like to explain the data at a coarser level if possible, and we
prefer the unstructured spatial effect over the structured spatial effect since we want
to reduce the risk of estimating spurious spatial signals. This gives the nested tree
structure in Figure 8a where the household effect, cluster effect and Besag effect are
sequentially split off from the total latent variance. We construct an HD prior based
on the tree structure with PC priors with default hyperparameter values for the splits,
and induce shrinkage on the total latent variance as in Prior class 3 with a PC prior
where P(Total variance _>_ 11 _._ 296) = 0 _._ 05. This corresponds to _a_ _priori_ equal-tailed
90% credible interval of (0 _._ 1 _,_ 10) for the effect of the random effects on the odds-ratio,
exp( _ui_ + _vi_ + _νi,j_ + _εi,j,k_ ). This allows for high variation in the data and is used because
the data is observed at the household level. The splits in Figure 8a are given PC priors
with default hyperparameters and bases models as indicated in the figure.


The model is parameterized by total standard deviation _σ_ T, and proportion of household variance to total variance of the random effects _ω_ [(1)], proportion of cluster variance
to the sum of cluster and county variance _ω_ [(2)], and the proportion of structured spatial
variance to county variance _ω_ [(3)] . The priors and posteriors of the proportions _ω_ [(1)], _ω_ [(2)]

and _ω_ [(3)] are shown in Figure 8e. The total standard deviation has a posterior median
of 1 _._ 47, and the prior and posterior can be seen in Figure S5.3 in the Supplementary
Materials. The results show that the data only weakly informs about the proportion
of structured to unstructured spatial effects, which indicates that the data provide no
strong evidence in favor of or against a structured spatial effect. Also the posterior of
_ω_ [(2)] is similar to the prior, but there is a strong signal in the posterior of _ω_ [(1)] that there
is non-negligible household-level dependence. A plausible explanation for the weak signals in _ω_ [(2)] and _ω_ [(3)] is that there is substantial noise coming from high variance in the
household-level random effect and weak information from the Binomial likelihood due
to few successes and few numbers of trials.


As shown in Figure 8b the proportion of the total latent variance attributed to the
structured spatial effect is low and the posterior median is 0.56%. The estimated spatial
effect in Figure 8d only explains a small part of the variation seen in the observed data
in Figure 8c. One should be careful to draw conclusions about spatial variation based
on Figure 8d because the data is only weakly informative about the split between the
structured and the unstructured spatial random effects _ω_ [(3)], and there is only weak
evidence for the spatial effect being different from 0 as shown in Figure S5.5 in the
Supplementary Materials. The fact that the comparisons of priors and posteriors for
_ω_ [(2)] and _ω_ [(3)] directly informs about the weak signal in the data is an advantage of


1130 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


Figure 8: Description of model structure, map of observed mortality, and results for
neonatal mortality in Kenya.


the parametrization through proportions of variance, and a strong argument for setting
priors on _ω_ [(2)] and _ω_ [(3)] rather than independent priors on the variance of each effect
since the resulting posteriors for _ω_ [(2)] and _ω_ [(3)] are strongly dependent on the resulting
implicit priors for _ω_ [(2)] and _ω_ [(3)] .


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1131


One could argue for other splits in the tree in Figure 8a such as preferring finer level
effects to coarser level effects because one does not want to estimate spurious clusterlevel or county-level effects, but the key point of this application is that it is easy
to set up the prior based on _a_ _priori_ assumptions and the assumptions are available
to other scientists at a glance. With the traditional approach of independent priors,
the resulting prior on the total variance of the random effects and the distribution of
this total variance to the different random effects is obfuscated. Furthermore, if expert
knowledge indicates that stronger relative shrinkage of the variances than the default
setting is needed, the medians of the conditional priors for _ω_ [(1)], _ω_ [(2)] and _ω_ [(3)] can be
reduced.

## **7 Discussion**


Independent priors for the variance parameters in a BHM result in an implicit prior
on the total variance of the random effects, _t_, and the attribution of _t_ to the random
effects. Additive models are typically built in a modular fashion, but these implicit
priors are not consistent with respect to adding or removing random effects. In the case
of Gaussian responses, both the prior for _t_ and the prior for _t_ relative to the size of the
residual variance change. The proposed HD priors overcomes these shortcomings, and
respect the defined model structure and are consistent for _t_ and the attribution of _t_ to
the different random effects for different selections of random effects.


The HD priors admit a visual representation through trees that allow transparent
communication of the assumptions made in constructing the priors and facilitate discussion around the assumptions. The tree clearly specifices where shrinkage has been
applied, and in some cases lead to more intuitive parametrization that is more suitable
for elicitation of priors. For the random intercept model, the tree-based hierarchical
variance decomposition leads to a parameterisation in terms of _t_ and the ICC. A prior
on these parameters is more interpretable than separate priors on the group variance
and individual variance, which obfuscates the joint effect of the priors. The increased
interpretability of joint priors compared to independent priors addresses concerns raised
about transparency for point processes where prior sensitivity is a major concern (Sørbye
et al., 2018).


The mix of robust PC priors for shrinkage and simple Dirichlet priors for expressing
ignorance, allows principled priors that respect the relative complexity of the random
effects when shrinkage is necessary, and intuitive exchangeability when no random effects
are preferred or no model structure is apparent. The simulation studies show that this
approach performs better than a completely unstructured approach with a Dirichlet
prior attributing _t_ to the different random effects, but that Dirichlet priors perform well
for subgroups of the random effects where there is no nested structure or difference in
complexity.


HD priors with default settings for the hyperparameters performs well, but there are
corner cases like no treatment effect in the latin square experiment and no structured
spatial effect for the binomial data, which are best handled by the default INLA prior.
However, this prior has a peak in the prior distribution for low variances and generally


1132 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


performs surprisingly bad. The HD priors perform comparable to component-wise PC
priors and separate half-Cauchy priors for the marginal variances. The main benefit of
the HD priors over other default priors is their combination of intuitive graphical representation with robust inference that behaves well across a range of different scenarios.


The calculation of PC priors is more complex in the context of correlation parameters, but multivariate PC priors have been developed for more complex random effects
such as autoregressive processes (Sørbye and Rue, 2017) and spatial Mat´ern models
(Fuglstad et al., 2019b). These can be integrated into the HD prior framework by first
defining priors on the correlation parameters, and then constructing the joint prior for
the variance parameters with the correlation parameters fixed to reasonable values. This
follows the pragmatic mindset of Assumption 2 of producing priors that are computationally feasible, intuitive and practically useful.


A key focus for future work is to exploit sparsity in the precision matrices of the
random effects. This is important when shrinkage is desired through PC priors because many models such as random walks, Besag models, and Gaussian random fields
(Lindgren et al., 2011) have dense covariance matrices, but can be expressed through
sparse precision matrices. Assume that the total variance is split between random effects with sparse precision matrices **Q** 1 and **Q** 2, where **Q** 1 corresponds to the base
model. Let 0 _<_ _ω_ _<_ 1, then the KLD used in Theorem 1 consists of the trace of
**Q** 1[(1 _−_ _ω_ ) **Q** _[−]_ 1 [1] [+] _[ω]_ **[Q]** 2 _[−]_ [1][], which can be computed quickly through the techniques in Rue]
and Held (2010, Section 12.1.7.10), and the determinant det[ **Q** 1[(1 _−_ _ω_ ) **Q** _[−]_ 1 [1] [+] _[ω]_ **[Q]** 2 _[−]_ [1][]] =]
det[(1 _−_ _ω_ ) **Q** 2 + _ω_ **Q** 1](det[ **Q** 2]) _[−]_ [1], which can be computed quickly through Cholesky
factorizations.


We aim to further broaden the advantages of the HD priors in the future by constructing a joint prior for the variance parameters and the fixed effects. However, this
will require re-thinking of the concept of total latent variance as it is the values of the
coefficients of the fixed effects and not their variance that determines the amount of
variance they explain. Instead of starting with the concept of marginal variances, it is
natural to begin with the classical concept of explained variance and use ideas from
block-wise g-priors (Som et al., 2014) to distribute variance inside a group of covariates. In a multilevel model this would connect the attribution of explained variance
to different levels to generalised coefficients of determinations. Additionally, towards
non-parametric regression by including a combination of a linear effect of a covariate
and a smooth effect of a covariate, and explicitly putting a prior on the degree of nonlinearity (Simpson et al., 2017, Section 7). However, there are still open questions and
this addition is outside the scope of this paper.


The choice of tree structure for HD priors should be guided by the application at
hand, for example, by considering the relative complexity of the random effects. When
expert knowledge is available, the default values for the hyperparameters should be
replaced by values elicited based on expert knowledge. We believe that the advantages
of the HD priors over independent priors mean that they should be used as the default
option in software for Bayesian analysis. However, it is necessary to make the selection
and computation of HD prior for a specific problem easier for analysts. We plan to
address this by providing a separate R package, which is compatible with INLA, that


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1133


provides a graphical user interface for selecting the tree structure and selecting priors
for the splits, and has the option to pre-compute priors for use in RStan. This will
allow analysts to experiment with different _a_ _priori_ assumptions and produce graphical
figures that summarize their assumptions and can be communicated to fellow scientists.
This will encourage transparancy and clarity in _a_ _priori_ assumptions in the scientific
community.

## **Supplementary Material**


Supplement to “Intuitive joint priors for variance parameters”
(DOI: [10.1214/19-BA1185SUPP;](https://doi.org/10.1214/19-BA1185SUPP) .zip). The Supplementary Materials consist of a supplementary document providing additional results and discussion, and example code for
the latin square model. The code is described in the Section S4.3 of the supplementary
document.

## **References**


Bakka, H., Rue, H., Fuglstad, G.-A., Riebler, A., Bolin, D., Illian, J., Krainski, E.,
Simpson, D., and Lindgren, F. (2018). “Spatial modeling with R-INLA: A review.”
_Wiley_ _Interdisciplinary_ _Reviews:_ _Computational_ _Statistics_ [, 10(6): e1443. MR3873676.](https://www.ams.org/mathscinet-getitem?mr=3873676)
doi: [https://doi.org/10.1002/wics.1443.](https://doi.org/10.1002/wics.1443) 1110


Balakrishnan, N. and Nevzorov, V. B. (2003). _A_ _primer_ _on_ _statistical_ _distributions_ .
Hoboken, NJ: John Wiley & Sons. [MR1988562.](https://www.ams.org/mathscinet-getitem?mr=1988562) doi: [https://doi.org/10.1002/](https://doi.org/10.1002/0471722227)
[0471722227.](https://doi.org/10.1002/0471722227) 1111


Banerjee, S., Carlin, B. P., and Gelfand, A. E. (2014). _Hierarchical_ _Modeling_ _and_ _Anal-_
_ysis_ _for_ _Spatial_ _Data_ [. Boca Raton, FL: Chapman and Hall/CRC. MR3362184.](https://www.ams.org/mathscinet-getitem?mr=3362184) 1109


Besag, J., York, J., and Molli´e, A. (1991). “Bayesian image restoration, with two applications in spatial statistics.” _Annals_ _of the Institute of Statistical Mathematics_, 43(1):
1–20. [MR1105822.](https://www.ams.org/mathscinet-getitem?mr=1105822) doi: [https://doi.org/10.1007/BF00116466.](https://doi.org/10.1007/BF00116466) 1115


Bhattacharya, A., Pati, D., Pillai, N. S., and Dunson, D. B. (2015). “Dirichlet–
Laplace priors for optimal shrinkage.” _Journal_ _of_ _the_ _American_ _Statistical_ _Associa-_
_tion_, 110(512): 1479–1490. [MR3449048.](https://www.ams.org/mathscinet-getitem?mr=3449048) doi: [https://doi.org/10.1080/01621459.](https://doi.org/10.1080/01621459.2014.960967)
[2014.960967.](https://doi.org/10.1080/01621459.2014.960967) 1111


Blangiardo, M. and Cameletti, M. (2015). _Spatial_ _and_ _spatio-temporal_ _Bayesian_ _models_
_with R-INLA_ [. West Sussex, United Kingdom: John Wiley & Sons. MR3364017.](https://www.ams.org/mathscinet-getitem?mr=3364017) 1110


Carpenter, B., Gelman, A., Hoffman, M. D., Lee, D., Goodrich, B., Betancourt, M.,
Brubaker, M., Guo, J., Li, P., and Riddell, A. (2017). “Stan: A probabilistic programming language.” _Journal_ _of_ _Statistical_ _Software_, 76(1). 1110, 1122


Cicchetti, D. V. (1994). “Guidelines, criteria, and rules of thumb for evaluating normed
and standardized assessment instruments in psychology.” _Psychological_ _assessment_,
6(4): 284. 1117


1134 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


Fahrmeir, L. and Lang, S. (2001). “Bayesian inference for generalized additive mixed
models based on Markov random field priors.” _Journal of the Royal Statistical Society:_
_Series_ _C_, 50(2): 201–220. [MR1833273.](https://www.ams.org/mathscinet-getitem?mr=1833273) doi: [https://doi.org/10.1111/1467-9876.](https://doi.org/10.1111/1467-9876.00229)
[00229.](https://doi.org/10.1111/1467-9876.00229) 1110


Fuglstad, G.-A., Hem, I. G., Knight, A., Rue, H., and Riebler, A. (2019a). “Supplement
to “Intuitive joint priors for variance parameters”.” _Bayesian Analysis_ [. doi: https://](https://doi.org/10.1214/19-BA1185SUPP)
[doi.org/10.1214/19-BA1185SUPP.](https://doi.org/10.1214/19-BA1185SUPP) 1112


Fuglstad, G.-A., Simpson, D., Lindgren, F., and Rue, H. (2019b). “Constructing priors
that penalize the complexity of Gaussian random fields.” _Journal_ _of_ _the_ _American_
_Statistical_ _Association_, 114(525): 445–452. [MR3941267.](https://www.ams.org/mathscinet-getitem?mr=3941267) doi: [https://doi.org/10.](https://doi.org/10.1080/01621459.2017.1415907)
[1080/01621459.2017.1415907.](https://doi.org/10.1080/01621459.2017.1415907) 1116, 1132


Gelman, A. (2006). “Prior distributions for variance parameters in hierarchical models.”
_Bayesian_ _Analysis_, 1(3): 515–534. [MR2221284.](https://www.ams.org/mathscinet-getitem?mr=2221284) doi: [https://doi.org/10.1214/](https://doi.org/10.1214/06-BA117A)
[06-BA117A.](https://doi.org/10.1214/06-BA117A) 1110, 1117


Gelman, A., Carlin, J. B., Stern, H. S., Dunson, D. B., Vehtari, A., and Rubin,
D. B. (2013). _Bayesian_ _Data_ _Analysis_ . Boca Raton, FL: Chapman and Hall/CRC.
[MR3235677.](https://www.ams.org/mathscinet-getitem?mr=3235677) 1109


Gelman, A. and Hill, J. (2007). _Data_ _Analysis_ _Using_ _Regression_ _and_ _Multi-_
_level/Hierarchical_ _Models_, volume 1. New York, New York: Cambridge University
Press. 1109, 1110, 1122


Gelman, A., Jakulin, A., Pittau, M. G., Su, Y.-S., et al. (2008). “A weakly informative
default prior distribution for logistic and other regression models.” _The_ _Annals_ _of_
_Applied_ _Statistics_, 2(4): 1360–1383. [MR2655663.](https://www.ams.org/mathscinet-getitem?mr=2655663) doi: [https://doi.org/10.1214/](https://doi.org/10.1214/08-AOAS191)
[08-AOAS191.](https://doi.org/10.1214/08-AOAS191) 1112, 1117


Gelman, A., Simpson, D., and Betancourt, M. (2017). “The prior can often only be
understood in the context of the likelihood.” _Entropy_, 19(10): 555. 1110, 1120


General Assembly of the United Nations (2015). “Resolution adopted by the General
Assembly on 25 September 2015.” A/RES/70/1. 1126


Gneiting, T. and Raftery, A. E. (2007). “Strictly proper scoring rules, prediction,
and estimation.” _Journal_ _of_ _the_ _American_ _Statistical_ _Association_, 102(477): 359–378.
[MR2345548.](https://www.ams.org/mathscinet-getitem?mr=2345548) doi: [https://doi.org/10.1198/016214506000001437.](https://doi.org/10.1198/016214506000001437) 1125


Golding, N., Burstein, R., Longbottom, J., Browne, A. J., Fullman, N., OsgoodZimmerman, A., Earl, L., Bhatt, S., Cameron, E., Casey, D. C., et al. (2017). “Mapping under-5 and neonatal mortality in Africa, 2000–15: a baseline analysis for the
Sustainable Development Goals.” _The_ _Lancet_, 390(10108): 2171–2182. 1126


Guo, J., Riebler, A., and Rue, H. (2017). “Bayesian bivariate meta-analysis of diagnostic
test studies with interpretable priors.” _Statistics_ _in_ _Medicine_, 36(19): 3039–3058.
[MR3670407.](https://www.ams.org/mathscinet-getitem?mr=3670407) doi: [https://doi.org/10.1002/sim.7313.](https://doi.org/10.1002/sim.7313) 1116


Hinkelmann, K. and Kempthorne, O. (1994). _Design_ _and_ _Analysis_ _of_ _Experiments,_


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1135


_Volume_ _1:_ _Introduction_ _to_ _Experimental_ _Design_ . John Wiley & Sons. [MR2129060.](https://www.ams.org/mathscinet-getitem?mr=2129060)
doi: [https://doi.org/10.1002/0471709948.](https://doi.org/10.1002/0471709948) 1124


Holand, A. M., Steinsland, I., Martino, S., and Jensen, H. (2013). “Animal models and
integrated nested Laplace approximations.” _G3:_ _Genes,_ _Genomes,_ _Genetics_, g3-113.
1110


Jordan, A., Kr¨uger, F., and Lerch, S. (2017). “Evaluating probabilistic forecasts with
the R package scoringRules.” _arXiv_ _preprint_ [arXiv:1709.04743.](https://arxiv.org/abs/1709.04743) 1125


Kenya National Bureau of Statistics, Ministry of Health/Kenya, National AIDS Control
Council/Kenya, Kenya Medical Research Institute, and National Council for Population and Development/Kenya (2015). _Kenya_ _Demographic_ _and_ _Health_ _Survey_ _2014_ .
Rockville, MD, USA. URL [http://dhsprogram.com/pubs/pdf/FR308/FR308.pdf.](http://dhsprogram.com/pubs/pdf/FR308/FR308.pdf)
1126, 1127


Krainski, E. T., G´omez-Rubio, V., Bakka, H., Lenzi, A., Castro-Camilio, D., Simpson,
D., Lindgren, F., and Rue, H. (2018). _Advanced_ _Spatial_ _Modeling_ _with_ _Stochastic_
_Partial Differential Equations using R and INLA_ . Boca Raton, FL: CRC press. Github
version www.r-inla.org/spde-book. 1110


Lambert, P. C., Sutton, A. J., Burton, P. R., Abrams, K. R., and Jones, D. R. (2005).
“How vague is vague? A simulation study of the impact of the use of vague prior
distributions in MCMC using WinBUGS.” _Statistics_ _in_ _Medicine_, 24(15): 2401–2428.
[MR2151713.](https://www.ams.org/mathscinet-getitem?mr=2151713) doi: [https://doi.org/10.1002/sim.2112.](https://doi.org/10.1002/sim.2112) 1110


Li, Z., Hsiao, Y., Godwin, J., Martin, B. D., Wakefield, J., Clark, S. J., et al. (2019).
“Changes in the spatial distribution of the under-five mortality rate: Small-area analysis of 122 DHS surveys in 262 subregions of 35 countries in Africa.” _PloS_ _one_, 14(1):
e0210645. 1126


Lindgren, F. and Rue, H. (2015). “Bayesian spatial modelling with R-INLA.” _Journal_
_of_ _Statistical_ _Software_, 63(19): 1–25. [MR2490553.](https://www.ams.org/mathscinet-getitem?mr=2490553) doi: [https://doi.org/10.1016/](https://doi.org/10.1016/S0169-7161(05)25033-2)
[S0169-7161(05)25033-2.](https://doi.org/10.1016/S0169-7161(05)25033-2) 1110


Lindgren, F., Rue, H., and Lindstr¨om, J. (2011). “An explicit link between Gaussian
fields and Gaussian Markov random fields: the stochastic partial differential equation
approach.” _Journal of the Royal Statistical Society: Series B (Statistical Methodology)_,
73(4): 423–498. [MR2853727.](https://www.ams.org/mathscinet-getitem?mr=2853727) doi: [https://doi.org/10.1111/j.1467-9868.2011.](https://doi.org/10.1111/j.1467-9868.2011.00777.x)
[00777.x.](https://doi.org/10.1111/j.1467-9868.2011.00777.x) 1132


Lunn, D., Spiegelhalter, D., Thomas, A., and Best, N. (2009). “The BUGS project:
Evolution, critique and future directions.” _Statistics_ _in_ _Medicine_, 28(25): 3049–3067.
[MR2750401.](https://www.ams.org/mathscinet-getitem?mr=2750401) doi: [https://doi.org/10.1002/sim.3680.](https://doi.org/10.1002/sim.3680) 1110


Martinez-Beneito, M. A. (2013). “A general modelling framework for multivariate disease mapping.” _Biometrika_, 100(3): 539–553. [MR3094436.](https://www.ams.org/mathscinet-getitem?mr=3094436) doi: [https://doi.org/](https://doi.org/10.1093/biomet/ast023)
[10.1093/biomet/ast023.](https://doi.org/10.1093/biomet/ast023) 1110


McGraw, K. O. and Wong, S. P. (1996). “Forming inferences about some intraclass
correlation coefficients.” _Psychological_ _methods_, 1(1): 30. 1110


1136 _**Intuitive**_ _**Joint**_ _**Priors**_ _**for**_ _**Variance**_ _**Parameters**_


Plummer, M. (2017). “JAGS version 4.3. 0 user manual [Computer software manual].” _Retrieved from_ sourceforge.net/projects/mcmc-jags/files/Manuals/4.x.
1110


Polson, N. G. and Scott, J. G. (2010). “Shrink globally, act locally: Sparse Bayesian regularization and prediction.” _Bayesian statistics_ [, 9: 501–538. MR3204017. doi: https://](https://www.ams.org/mathscinet-getitem?mr=3204017)
[doi.org/10.1093/acprof:oso/9780199694587.003.0017.](https://doi.org/10.1093/acprof:oso/9780199694587.003.0017) 1111


Riebler, A., Sørbye, S. H., Simpson, D., and Rue, H. (2016). “An intuitive Bayesian
spatial model for disease mapping that accounts for scaling.” _Statistical_ _Methods_ _in_
_Medical_ _Research_, 25(4): 1145–1165. [MR3541089.](https://www.ams.org/mathscinet-getitem?mr=3541089) doi: [https://doi.org/10.1177/](https://doi.org/10.1177/0962280216660421)
[0962280216660421.](https://doi.org/10.1177/0962280216660421) 1111, 1116, 1127


Rue, H. and Held, L. (2005). _Gaussian_ _Markov_ _random_ _fields:_ _theory_ _and_ _applica-_
_tions_ . Boca Raton, Florida: CRC press. [MR2130347.](https://www.ams.org/mathscinet-getitem?mr=2130347) doi: [https://doi.org/10.](https://doi.org/10.1201/9780203492024)
[1201/9780203492024.](https://doi.org/10.1201/9780203492024) 1115, 1119


Rue, H. and Held, L. (2010). “Discrete Spatial Variation.” In Gelfand, A. E., Diggle,
P., Guttorp, P., and Fuentes, M. (eds.), _Handbook_ _of_ _Spatial_ _Statistics_, Handbooks
of Modern Statistical Methods, chapter 12, 171–200. Boca Raton, FL: CRC Press.
[MR2730942.](https://www.ams.org/mathscinet-getitem?mr=2730942) doi: [https://doi.org/10.1201/9781420072884-c12.](https://doi.org/10.1201/9781420072884-c12) 1132


Rue, H., Martino, S., and Chopin, N. (2009). “Approximate Bayesian inference for latent
Gaussian models by using integrated nested Laplace approximations.” _Journal_ _of_ _the_
_Royal_ _Statistical_ _Society:_ _Series_ _B_, 71(2): 319–392. [MR2649602.](https://www.ams.org/mathscinet-getitem?mr=2649602) doi: [https://doi.](https://doi.org/10.1111/j.1467-9868.2008.00700.x)
[org/10.1111/j.1467-9868.2008.00700.x.](https://doi.org/10.1111/j.1467-9868.2008.00700.x) 1110


Rue, H., Riebler, A., Sørbye, S. H., Illian, J. B., Simpson, D. P., and Lindgren, F. K.
(2017). “Bayesian computing with INLA: A review.” _Annual_ _Review_ _of_ _Statistics_
_and_ _Its_ _Application_, 4(1): 395–421. [MR3634300.](https://www.ams.org/mathscinet-getitem?mr=3634300) doi: [https://doi.org/10.1214/](https://doi.org/10.1214/16-STS576)
[16-STS576.](https://doi.org/10.1214/16-STS576) 1110


Shen, K.-K., Dor´e, V., Rose, S., Fripp, J., McMahon, K. L., de Zubicaray, G. I., Martin,
N. G., Thompson, P. M., Wright, M. J., and Salvado, O. (2016). “Heritability and
genetic correlation between the cerebral cortex and associated white matter connections.” _Human_ _brain_ _mapping_, 37(6): 2331–2347. 1117


Simpson, D., Rue, H., Riebler, A., Martins, T. G., and Sørbye, S. H. (2017). “Penalising
model component complexity: a principled, practical approach to constructing priors.” _Statistical_ _Science_, 32(1): 1–28. [MR3634300.](https://www.ams.org/mathscinet-getitem?mr=3634300) doi: [https://doi.org/10.1214/](https://doi.org/10.1214/16-STS576)
[16-STS576.](https://doi.org/10.1214/16-STS576) 1110, 1112, 1116, 1121, 1132


Som, A., Hans, C. M., and MacEachern, S. N. (2014). “Block hyper-g priors in Bayesian
regression.” _arXiv_ _preprint_ [arXiv:1406.6419.](https://arxiv.org/abs/1406.6419) [MR3321977.](https://www.ams.org/mathscinet-getitem?mr=3321977) 1110, 1132


Sørbye, S. H., Illian, J. B., Simpson, D. P., Burslem, D., and Rue, H. (2018). “Careful
prior specification avoids incautious inference for log-Gaussian Cox point processes.”
_Journal_ _of_ _the_ _Royal_ _Statistical_ _Society:_ _Series_ _C_ _(Applied_ _Statistics)_, 68: 543–564.
[MR3937462.](https://www.ams.org/mathscinet-getitem?mr=3937462) doi: [https://doi.org/10.1111/rssc.12321.](https://doi.org/10.1111/rssc.12321) 1131


Sørbye, S. H. and Rue, H. (2014). “Scaling intrinsic Gaussian Markov random field


_G.-A._ _Fuglstad,_ _I._ _G._ _Hem,_ _A._ _Knight,_ _H._ _Rue,_ _and_ _A._ _Riebler_ 1137


priors in spatial modelling.” _Spatial_ _Statistics_, 8: 39–51. [MR3326820.](https://www.ams.org/mathscinet-getitem?mr=3326820) doi: [https://](https://doi.org/10.1016/j.spasta.2013.06.004)
[doi.org/10.1016/j.spasta.2013.06.004.](https://doi.org/10.1016/j.spasta.2013.06.004) 1115


Sørbye, S. H. and Rue, H. (2017). “Penalised complexity priors for stationary autoregressive processes.” _Journal_ _of_ _Time_ _Series_ _Analysis_, 38(6): 923–935. [MR3714116.](https://www.ams.org/mathscinet-getitem?mr=3714116)
doi: [https://doi.org/10.1111/jtsa.12242.](https://doi.org/10.1111/jtsa.12242) 1132


Sørbye, S. H. and Rue, H. (2018). “Fractional Gaussian noise: Prior specification and
model comparison.” _Environmetrics_, 29(5–6): e2457. 1116


Spiegelhalter, D., Thomas, A., Best, N., and Gilks, W. (1996). “BUGS 0.5* Examples
Volume 2 (version ii).” _MRC_ _Biostatistics_ _Unit_ . 1110


Stan Development Team (2018a). “RStan: the R interface to Stan.” R package version
2.18.1. URL [http://mc-stan.org/.](http://mc-stan.org/) 1110, 1122


Stan Development Team (2018b). “Stan Modeling Language Users Guide and Reference
Manual, version 2.18.0.” _Technical_ _report_ . URL [http://mc-stan.org.](http://mc-stan.org) 1110


StataCorp (2017). _Stata_ _Bayesian_ _analysis,_ _Reference_ _manual_ . StataCorp LLC, College
Station, TX, 15 edition. 1110


Wakefield, J. (2006). “Disease mapping and spatial regression with count data.” _Bio-_
_statistics_, 8(2): 158–183. 1110


Wakefield, J., Fuglstad, G.-A., Riebler, A., Godwin, J., Wilson, K., and Clark, S. J.
(2019). “Estimating under-five mortality in space and time in a developing world
context.” _Statistical_ _Methods_ _in_ _Medical_ _Research_, 28(9): 2614–2634. [MR4000184.](https://www.ams.org/mathscinet-getitem?mr=4000184)
doi: [https://doi.org/10.1177/0962280218767988.](https://doi.org/10.1177/0962280218767988) 1126


