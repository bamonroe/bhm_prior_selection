Bayesian Analysis (2015) **10**, Number 2, pp. 321–349

# **Sensitivity Analysis for Bayesian Hierarchical** **Models**


Ma�lgorzata Roos _[∗]_, Thiago G. Martins _[†]_, Leonhard Held _[‡]_, and H˚avard Rue _[§]_


**Abstract.** Prior sensitivity examination plays an important role in applied Bayesian analyses. This is especially true for Bayesian hierarchical models, where interpretability of the parameters within deeper layers in the hierarchy becomes challenging. In addition, lack of information together with identifiability issues may
imply that the prior distributions for such models have an undesired influence on
the posterior inference. Despite its importance, informal approaches to prior sensitivity analysis are currently used. They require repetitive re-fits of the model with
ad-hoc modified base prior parameter values. Other formal approaches to prior
sensitivity analysis suffer from a lack of popularity in practice, mainly due to their
high computational cost and absence of software implementation. We propose a
novel formal approach to prior sensitivity analysis, which is fast and accurate.
It quantifies sensitivity without the need for a model re-fit. Through a series of
examples we show how our approach can be used to detect high prior sensitivities
of some parameters as well as identifiability issues in possibly over-parametrized
Bayesian hierarchical models.


**Keywords:** Base prior, formal local sensitivity measure, Bayesian robustness,
calibration, Hellinger distance, Bayesian hierarchical models, identifiability,
overparametrisation.

## **1 Introduction**


Today applied statisticians have a wealth of both frequentist and Bayesian procedures
at their disposal. The prominent feature of the latter approach is its ability to incorporate prior knowledge in the analysis. This feature, however, is both a benefit and
a challenge. A Bayesian model is said to be sensitive and non-robust with respect to
the prior distribution if its posterior distribution dramatically changes when the base
prior parameter values are altered slightly. Recently, implementation of the hierarchical
framework has led to the development of increasingly intricate models. Unfortunately,
their complexity makes an extensive elicitation of the base prior for each hierarchy
layer practically impossible. Instead, base priors tend to be determined in a rather casual fashion and without appropriate reflection, so arbitrary and inaccurately specified
parameter values may arise. At the same time, due to model complexity, the impact


_∗_ Division of Biostatistics, Institute for Social and Preventive Medicine, University of Zurich, Zurich,
Switzerland, [mroos@ifspm.uzh.ch](mailto:mroos@ifspm.uzh.ch)

_†_ Department of Mathematical Sciences, Norwegian University for Science and Technology,
Trondheim, Norway, [guerrera@math.ntnu.no](mailto:guerrera@math.ntnu.no)

_‡_ Division of Biostatistics, Institute for Social and Preventive Medicine, University of Zurich, Zurich,
Switzerland, [leonhard.held@ifspm.uzh.ch](mailto:leonhard.held@ifspm.uzh.ch)

_§_ Department of Mathematical Sciences, Norwegian University for Science and Technology,
Trondheim, Norway, [hrue@math.ntnu.no](mailto:hrue@math.ntnu.no)


_⃝_ c 2015 International Society for Bayesian Analysis DOI: [10.1214/14-BA909](http://dx.doi.org/10.1214/14-BA909)


322 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


of possibly misspecified prior parameter values on outputs is unclear. In addition, the
adequacy of the sample size needed for a reliable estimation of each layer is, in fact,
unknown. It can happen that models are overparametrized (Carlin and Louis, 1998) and
it may not be obvious to decide which parameters are well identified by the data and
which are not (Dawid, 1979; Gelfand and Sahu, 1999). Hence, development of complex
Bayesian models without any prior robustness diagnostics may be problematic. In order
to ensure reliable and robust results, it is crucial to verify how sensitive the resulting
posteriors are for each prior input.


The relevance of sensitivity and uncertainty analyses for exploring complex models
has been highly emphasized in the literature. Oakley and O’Hagan (2004) and Saltelli
et al. (2008) applied it successfully in the context of frequentist analysis. In Bayesian
statistics the general sensitivity concept treated by Geisser (1992), Clarke and Gustafson
(1998), Millar and Stewart (2007) and Zhu et al. (2011) involves broad issues like influential observations, uncertainty of the sampling model and prior inadequacy.


**1.1** **Bayesian** **formal** **sensitivity** **analysis**


Sensitivity to the prior parameter specification is a crucial part of the general sensitivity
setting (Berger et al., 2000). R´ıos Insua et al. (2000) and Ruggeri (2008) argue that
inappropriate prior parameter specifications can lead to distorted findings for both
influential observations and uncertainty of the sampling model. To date, two approaches
to sensitivity analysis can be distinguished: the global and the local one. The global
approach considers the class of all priors compatible with the elicited prior information
and computes the range of the posteriors as the prior varies over the class. This range is
typically found by determining the “extreme” priors in the class that yield maximally
distant posteriors, without explicitly carrying out the analysis for every prior in the
class. In contrast, the local sensitivity approach is interested in the rate of change in
posterior with respect to changes in the prior, and usually uses differential calculus to
approximate it. Despite its desirability the global approach is impractical in the Bayesian
hierarchical framework whereas the local one, recommended by Gustafson (2000) and
Sivaganesan (2000), is the method of choice.


The local sensitivity approach routinely applied in complex Bayesian hierarchical
models can determine which model components are hard to learn from the data. Elevated worst-case sensitivity estimates quickly identify priors that may need more careful
attention. Since at the stage of model construction, the analyst might be interested in
specific directions on the hyperparameter space (Kadane, 1992), there is a strong need
for investigation of the circular sensitivity around particular base prior parameter values.


Gustafson (2000) distinguishes a variety of frameworks for local Bayesian robustness investigation. They differ with respect to the posterior results (distribution or
summaries), by prior perturbations (geometric or parametric), whether the worst-case
sensitivity is measured in the absolute or relative sense, and by the discrepancy measure used. In particular, McCulloch (1989) following Cook (1986) approximated prior
worst-case robustness by the principal eigenvalue of an appropriate infinitesimal ratio.
This approach has been further refined by Zhu et al. (2011).


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 323


**1.2** **Informal** **approaches** **and** **dedicated** **software**


Surprisingly, despite considerable theoretical advances in formal sensitivity analysis, it
is barely used in every-day practice. In the few cases when the extent of prior robustness is assessed, brute force and informal approaches are used instead. An informal
technique consists of repetitive fits of the model with ad hoc modified prior inputs.
If the posteriors subjectively do not differ much, non-sensitivity and robustness are
claimed. The main drawback of such an approach is that it requires several re-fits of
the model, which may be extremely time consuming. The informal approach gives no
guidance about how the input modifications should be performed and how differences in
the results should be judged in a standardized way. Consequently, in order to guarantee
reproducibility of Bayesian robustness considerations, the use of a formal sensitivity
approach is required.


Although the need and importance of a formal prior robustness investigation have
been ubiquitously approved, its lack of popularity in practice seems mainly due to the
non-existence of such a facility in current Bayesian programs (Ruggeri, 2008). Hence,
Berger et al. (2000) and Lesaffre and Lawson (2012) stress that a development of a
formal robustness methodology, which is feasible, fairly quick, operating with low extra
computing effort and provided by default in a dedicated software, is strongly required.
Furthermore, in order to become widely used, its compatibility with the Markov chain
Monte Carlo (Gilks et al., 1996) framework is beneficial.


**1.3** **Scope** **of** **paper**


In this paper we suggest the use of a Bayesian formal _ϵ_ -local sensitivity, which can
be conveniently applied to Bayesian hierarchical models. The novelty of our approach
hinges on the choice of an epsilon grid, a set of base prior parameter specifications
modified in a standardized way. The typical worst-case sensitivity estimates provided
by the infinitesimal local approach are complemented by circular sensitivity summaries.
Our approach guarantees a nearly instantaneous sensitivity assessment without any
need for a model re-fit. Prior and posterior Hessian approximations are not required
any more. Instead, the base prior specification and the corresponding marginal posterior
density alone are used. Because our _ϵ_ -local sensitivity approach operates without much
additional computational effort, it is a convenient measure for an every-day use.


The remainder of this article is organized as follows: Section 2 defines the _ϵ_ -local
sensitivity measure and its calibration with respect to the unit-variance normal distribution. In Section 3, an illustrative and comparative example is discussed. General
approaches for fast numerical computation and epsilon grid search are presented in Section 4. Although our local robustness approach is generally applicable, its performance
for a range of applications with increasing complexity and several latent models is presented in Section 5 and in the Supplementary Material. In these examples we show
how to use the proposed methodology in practice to identify sensitive parameters. Some
concluding remarks are given in Section 6. Appendices A-C provide proofs and discuss
computational issues.


324 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_

## **2 Local sensitivity**


**2.1** **Definition**


Given a scalar _θ_, we denote by _π_ _**γ**_ 0 ( _θ_ ) and _π_ _**γ**_ 0 ( _θ|_ _**y**_ ) the base prior density with parameter
values fixed at _**γ**_ 0 and the resulting marginal posterior density for _θ_, respectively. The
corresponding entities indexed by _**γ**_ originate from a prior distribution with arbitrary
parameter values denoted by _**γ**_ .


We define the _ϵ_ -local circular sensitivity _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [)] [as] [the] [set] [of] [ratios]


         -          _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [) =] d( _π_ _**γ**_ ( _θ|_ _**y**_ ) _, π_ _**γ**_ 0 ( _θ|_ _**y**_ )) _,_ for _**γ**_ _∈_ G _**γ**_ 0 ( _ϵ_ ) _,_ (1)
_ϵ_


with the grid G _**γ**_ 0 ( _ϵ_ ) of parameter values defined by


G _**γ**_ 0 ( _ϵ_ ) = _{_ _**γ**_ : d( _π_ _**γ**_ ( _θ_ ) _, π_ _**γ**_ 0 ( _θ_ )) = _ϵ},_ (2)


where d( _·, ·_ ) denotes a convenient discrepancy measure between two densities. We can
also consider G _**γ**_ 0( _ϵ_ ) as a contour line around _**γ**_ 0. In our definition, the distributional assumption of the prior _π_ _**γ**_ ( _θ_ ) for one particular component _θ_ of the Bayesian hierarchical
model is held fixed and only its parameter values _**γ**_ are allowed to vary.


In practice, we use a fixed small _ϵ_ for sensitivity evaluation instead of its infinitesimal
approximation. We suggest detailed exploration of the local geometry implied by d( _·, ·_ )
in the space of prior distributions. This is done by a numerical search for the prior
parameter value grid, G _**γ**_ 0 ( _ϵ_ ), with center set at _**γ**_ 0 and the distance value fixed to _ϵ_ as
in Equation (2). Our circular approach naturally adjusts for possible non-orthogonalities
of the prior parametrisation as it examines all directions in the space of prior parameter
values on equal footing.


Circular sensitivity can be conveniently summarized by a single number. For example, the worst-case sensitivity _S_ _**γ**_ 0 ( _ϵ_ ) is defined to be the maximum of the circular
sensitivity _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [)]

_S_ _**γ**_ 0 ( _ϵ_ ) = max� _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [)] �= max d( _π_ _**γ**_ ( _θ|_ _**y**_ ) _, π_ _**γ**_ 0 ( _θ|_ _**y**_ )) _._ (3)
_**γ**_ _∈_ G _**γ**_ 0 ( _ϵ_ ) _ϵ_


In this paper, we mainly concentrate on the worst-case sensitivity, though alternative
estimates such as mean, median or minimum could be also reported.


For complex Bayesian hierarchical models the sensitivity of each model component
_θ_ is addressed separately. Computations are conducted according to Equations (1)–
(3). The only input required for sensitivity estimation is the base prior distribution
specification _π_ _**γ**_ 0 ( _θ_ ) and the corresponding marginal posterior density _π_ _**γ**_ 0 ( _θ|_ _**y**_ ). As a
first step, the worst-case robustness _S_ _**γ**_ 0 ( _ϵ_ ) is checked. A high value indicates that a
particular prior has to be investigated with more care. It could be caused by a variety
of reasons. One, a misplaced prior distribution leading to a prior-data conflict (Box,
1980; Evans and Moshonov, 2006). Two, an inappropriate prior caused by misspecified


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 325


prior parameter values. Three, an insufficient sample size at the hierarchy level under
consideration. At this step any other circular sensitivity summary such as mean, median
or minimum can be applied.


In the second step, the circular sensitivity _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [)] [in] [all] [directions] [around] _**[γ]**_ [0] [can]
be referred to. Circular sensitivity can be easily depicted, as will be shown later. The
circular sensitivity plots indicate directions in which the most pronounced sensitivity
value modification was found. The plot’s shape depends on the prior distribution, the
base prior parameter specification and the model assumed. The choice of the base prior
parameter values at the model construction stage can be conveniently guided by circular
sensitivity and its summary values.


As our approach uses posterior and prior densities directly, a convenient discrepancy measure d( _·, ·_ ) to quantify the discrepancy between two distributions is required
(Gustafson, 2000). One possible choice is the _φ_ -divergence, also called _f_ -divergence,
between two densities _π_ 0 and _π_ 1. It is defined as




      -      _π_ 0( _θ_ )
D _φ_ ( _π_ 1 _, π_ 0) = _π_ 1( _θ_ ) _φ_
_π_ 1( _θ_ )




_dθ,_ (4)



where _φ_ is a smooth convex function (Amari, 1990; Amari and Nagaoka, 2000). The
Kullback-Leibler divergence and Hellinger distance are special cases of the _φ_ -divergence:
with _φKL_ ( _x_ ) = _x_ log( _x_ ) for the Kullback-Leibler divergence and _φH_ ( _x_ ) = ( ~~_[√]_~~ _x_ _−_ 1) [2] _/_ 2
for the Hellinger distance (Dey and Birmiwal, 1994). Robert (1996) found that the
Kullback-Leibler divergence and the Hellinger distance can frequently be used interchangeably but that the Hellinger distance is more natural as a true distribution distance.


Our preference for the Hellinger distance (Le Cam, 1986) is motivated by convenience. The Hellinger distance is advantageous given marginal posterior distributions
and prior distributions provided numerically and attaining nonzero values only on a
finite discrete set of points (Roos and Held, 2011). It is a symmetric measure of discrepancy between two densities _π_ 0 and _π_ 1:



_−∞_



H( _π_ 1 _, π_ 0) =


=


=




1
2

1
2

1
2




~~�~~ _∞_


_−∞_

~~�~~ _∞_


_−∞_



�� - �2
_π_ 1( _θ_ ) _−_ _π_ 0( _θ_ ) _dθ_


- ~~�~~ _π_ 1( _θ_ ) _−_ 2 _π_ 1( _θ_ ) _π_ 0( _θ_ ) + _π_ 0( _θ_ ) _dθ_




- ~~�~~ _∞_
2 _−_ 2




~~�~~ _π_ 1( _θ_ ) _π_ 0( _θ_ ) _dθ_




 = 1 _−_ BC( _π_ 1 _, π_ 0) _._




                        - _∞_                         Here, the Bhattacharyya coefficient BC( _π_ 1 _, π_ 0) = _−∞_ _π_ 1( _θ_ ) _π_ 0( _θ_ ) _dθ_ measures the
affinity of both densities (Bhattacharyya, 1943). Note that the Hellinger distance is
invariant to any one-to-one transformation (for example logarithmic, inverse or squareroot) of both densities (Jeffreys, 1961; Roos and Held, 2011).


326 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


We assume throughout that the Gaussian distribution is parametrized by mean _μ_ and
precision _λ_ and the gamma distribution with shape _α_ and rate _β_ parameters leading
to expectation _α/β_ and variance _α/β_ [2] . For both distributions the Hellinger distance
between densities with differing parameter values can be computed analytically.


Rao (1945) and Dawid (1977) discussed the direct correspondence of the Bhattacharyya coefficient and the Fisher information matrix. In the context of differential
geometry, Amari (1990) stated that both the Hellinger distance and the Bhattacharyya
distance are directly related to the Riemannian distance. Indeed, the Hellinger distance
introduces a non-Euclidean geometry on the space of probability distributions. As an example, consider the gamma prior assumed in Section 5 for the precision of the structured
intrinsic conditional autoregressive component. Figure 1 shows contour plots of G _**γ**_ 0 ( _ϵ_ ) in
Equation (2) based on the Hellinger distance with center set at _**γ**_ 0 = ( _α_ 0 _, β_ 0) = (1 _,_ 0 _._ 34).
Equal scaling of x and y-axes highlights that the contours tend to be ellipses rather than
circles in Euclidean geometry.


Figure 1: Contour plots of G _**γ**_ 0 ( _ϵ_ ) for gamma distribution with center _**γ**_ 0 = ( _α_ 0 _, β_ 0) =
(1 _,_ 0 _._ 34) based on the Hellinger distance.


**2.2** **Calibration** **and** **interpretation**


Calibration of differences between two distributions has the following advantage: for a
particular reference distribution, the experimenter can assess the relevance of the discrepancy in terms of the natural parameter of the benchmark. Various calibrations have
been suggested by McCulloch (1989), Dey and Birmiwal (1994), Goutis and Robert
(1998) and Roos and Held (2011). Although _φ_ -divergence and Kullback-Leibler divergence have been discussed in the literature, the calibration of the Hellinger distance with


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 327


respect to the unit-variance normal distribution, derived in Lemma 2 in Appendix B4,
seems to be new. It states that for a particular Hellinger distance value _h_ and

         _μ_ ( _h_ ) = _−_ 8 log(1 _−_ _h_ [2] )


the Hellinger distance between two unit-variance normal distributions with shifted locations satisfies
H(N( _μ_ ( _h_ ) _,_ 1) _,_ N(0 _,_ 1)) = _h._


Note that _μ_ ( _h_ ) is the calibration of the Hellinger distance _h_, as _h_ between any two
densities is the same as that between N(0,1) and N( _μ_ ( _h_ ),1). Given the Hellinger distance
_h_ between any two densities, we can quantify discrepancies between them, in terms of the
differences in mean from 0 to _μ_ ( _h_ ) for normal distribution with unit standard deviation.


Usually, it is expected that the sensitivity estimates in Equations (1) and (3) attain
values smaller than 1. This indicates that changes in the marginal posteriors are smaller
than changes in priors. Sensitivity close to 1 denotes that differences in marginal posteriors and priors are comparable. However, in practice both the circular _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [)] [and]
the worst-case _S_ _**γ**_ 0 ( _ϵ_ ) sensitivity estimates can attain values larger than 1 leading to
so-called super-sensitivity (Plummer, 2001). This possibility has been already attested
by McCulloch (1989), Clarke and Gustafson (1998), P´erez et al. (2006) and Zhu et al.
(2011). M¨uller (2012) truncates at 1 to remove excessively large sensitivity values, but
we prefer to report the unmodified sensitivity estimates.


In order to get an impression about the relevance of sensitivity values, we calibrate both the numerator and the denominator of the sensitivity measures defined in
Equations (1) and (3). Interestingly, this ratio of calibrated Hellinger distances can be
conveniently approximated by the ratio of Hellinger distances involved in the sensitivity
estimates themselves as




 - _μ_ H( _π_ _**γ**_ ( _θ|_ _**y**_ ) _, π_ _**γ**_ 0 ( _θ|_ _**y**_ ))




[0] _._ (5)

_ϵ_



_**y**_ ) _, π_ _**γ**_ 0 ( _θ|_ _**y**_ )) _≈_ [H(] _[π]_ _**[γ]**_ [(] _[θ][|]_ _**[y]**_ [)] _[, π]_ _**[γ]**_ [0] [(] _[θ][|]_ _**[y]**_ [))]

_μ_ ( _ϵ_ ) _ϵ_



To see this, apply the following property of the logarithm to the numerator and denominator of the left hand side: log(1 _−x_ ) _≈−x_, for _−_ 1 _≤_ _x <_ 1, with _x_ = _h_ [2] . Therefore, the
sensitivity estimates obtained in applications can be directly interpreted as an approximation of the ratio of calibrated Hellinger distances with respect to the unit-variance
normal distribution. Although a particular choice of _ϵ_ anchors our calibration, the above
observation offers an option to interpret the sensitivity magnitude independently of any
particular _ϵ_ value used for the grid computation. Apart of that, the use of the ratio of
calibrations leads to its applicability for the whole range of sensitivity values including
small and super-sensitivities.


We will show in the illustrative example in Section 3 that calibration with respect
to the unit-variance normal distribution converts sensitivity values obtained under the
Kullback-Leibler divergence and the Hellinger distance and helps to compare them on
an equal footing. Therefore, it is very useful for a unified understanding of sensitivity
estimates obtained by different approaches.


328 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_

## **3 An illustrative and comparative conjugate example**


For illustration, we consider an analytical conjugate example which was also discussed
by McCulloch (1989), Zhu et al. (2011) and Box (1980). On the basis of this example we
will compare the infinitesimal local sensitivity method based on the Kullback-Leibler
divergence with the _ϵ_ -local sensitivity approach suggested in Section 2.1. The latter
approach will be illustrated for both the Kullback-Leibler divergence and the Hellinger
distance. The infinitesimal local sensitivity approach based on the Kullback-Leibler
divergence was originally developed by McCulloch (1989). It is a special case of the more
general methodology suggested by Zhu et al. (2011) when constrained to the KullbackLeibler divergence and parametric perturbations of the base prior. Due to the explicit
analytical formulae in the conjugate example, we can provide a software-independent
comparison of the methods.


Assume that observations _y_ 1 _, y_ 2 _, . . ., yn_ are realisations of _iid_ N( _m, κ_ _[−]_ [1] ) random
variables, where _κ_ is fixed and _m_ _∼_ N( _μ, λ_ _[−]_ [1] ). Due to conjugacy the posterior dis
                   tribution is also normal with _m |_ _**y**_ _∼_ N ( _nκy_ ¯ + _λμ_ ) _/_ ( _nκ_ + _λ_ ) _,_ ( _nκ_ + _λ_ ) _[−]_ [1][�] . Following
Box (1980), we assume that _κ_ = 1, _n_ = 4, _y_ ¯ = 76 and _σ_ ˆ [2] = 3 _._ 33. We are interested
in the local sensitivity estimate of _θ_ = _m_ at the base prior parameter specification
_**γ**_ 0 = ( _μ_ 0 _, λ_ 0) = (70 _,_ 0 _._ 5).


**3.1** **Infinitesimal** **local** **sensitivity** **based** **on** **the** **Kullback-Leibler**
**divergence**


In the following we describe the approach proposed by McCulloch (1989) and Zhu et al.
(2011). Recall the definition of the Kullback-Leibler divergence _D_ KL( _π_ 0 _||π_ ) given in
Equation (4) for an appropriate choice of _φKL_ ( _x_ ) = _x_ log( _x_ ). It is an unbounded, nonsymmetric and directed measure, where the distribution having density _π_ 0 is assumed to
be the true probability distribution and the distribution leading to density _π_ is treated as
an arbitrary one. Now apply the Taylor expansion to the numerator and denominator
of the worst-case infinitesimal local sensitivity _S_ at _**γ**_ 0 defined in Equation (6) and
approximate it by the principal eigenvalue of the ratio of two matrices


_S_ = lim DKL( _π_ _**γ**_ 0 ( _θ|_ _**y**_ ) _||π_ _**γ**_ ( _θ|_ _**y**_ )) _≈_ max (6)
_**γ**_ _→_ _**γ**_ 0 DKL( _π_ _**γ**_ 0 ( _θ_ ) _||π_ _**γ**_ ( _θ_ )) eigenvalue _[{]_ [(] **[D]** [2] _[i]_ [(] _**[γ]**_ [0][))] _[−]_ [1] **[D]** [2] _[d]_ [(] _**[γ]**_ [0][)] _[}][.]_


Here _d_ ( _**γ**_ ) = DKL( _π_ _**γ**_ 0 ( _θ|_ _**y**_ ) _||π_ _**γ**_ ( _θ|_ _**y**_ )), _i_ ( _**γ**_ ) = DKL( _π_ _**γ**_ 0 ( _θ_ ) _||π_ _**γ**_ ( _θ_ )) and **D** [2] _f_ ( _**x**_ ) denotes
the second derivative matrix of the function _f_ evaluated at _**x**_ .


In our conjugate example with unknown _θ_ = _m_, the analytical formulae in Appendix B2 for the eigenvalue analysis of the matrix ( **D** [2] _i_ ( _**γ**_ 0)) _[−]_ [1] **D** [2] _d_ ( _**γ**_ 0) can be applied.
The resulting worst-case infinitesimal local sensitivity, which is the principal eigenvalue,
is equal to 3.284 and the corresponding principal eigenvectors lie on the line _λ_ = -2.677 _μ_ .
The second eigenvalue is equal to 4e-04 with the corresponding eigenvector lying on the
line _λ_ = 0.093 _μ_ . Both eigenvectors are indicated by red and green lines in Figures 2
and 3.


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 329


Figure 2: Comparison of local sensitivity methods in the plane spanned by parameters _μ_
and _λ_ centered at _**γ**_ 0 = ( _μ_ 0 _, λ_ 0) = (70 _,_ 0 _._ 5). Red and green lines: eigenvectors provided
by the infinitesimal local approach based on the Kullback-Leibler divergence with the
principal eigenvalue (worst-case sensitivity) 3.284. Black curve: circular sensitivity in
each direction provided by the _ϵ_ -local approach for _ϵ_ 0 = 5e-05 based on the KullbackLeibler divergence. Blue curve: circular _ϵ_ -local sensitivity based on the Hellinger distance
for _ϵ_ 0 = 0.00354. In both cases the choice of _ϵ_ 0 corresponds to a shift in means by
0.01 with respect to the unit-variance normal distribution. The worst-case sensitivities
provided by the _ϵ_ -local approaches are indicated by a red dot. The scale for sensitivity
values (0.1, 0.2, _. . ._, 0.9, 1) is indicated by grey circles. Sensitivity equal to 0.5 is marked
by a black circle, whereas the value 1 is marked by a red one.


For a vector _**θ**_ the method is usually applied to the whole parameter vector of the
base priors. Both matrices **D** [2] _d_ ( _**γ**_ 0) and **D** [2] _i_ ( _**γ**_ 0) for numerator and denominator are
usually estimated by the corresponding Fisher information matrices. In practice, for a
large cardinality of _**θ**_ it may be difficult to determine the direction indicated by the
loadings of the principal eigenvector in the multidimensional space. Only the worstcase infinitesimal local sensitivity is provided by the principal eigenvalue. Usually, the
matrix ( **D** [2] _i_ ( _**γ**_ 0)) _[−]_ [1] **D** [2] _d_ ( _**γ**_ 0) is non-symmetric. Consequently, complex eigenvalues are
possible. One may use analytical formulae to compute the Fisher information matrices
under base prior and posterior or draw MCMC samples for their approximation.


Equation (6) is an infinitesimal local sensitivity measure expressed by a ratio. Incorporation of the denominator in such a measure is crucial. McCulloch (1989) concentrates
on sensitivity of posterior distributions to parametric changes in the base prior parameter values quantified by the Kullback-Leibler divergence. In contrast, the methodology


330 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


suggested by Zhu et al. (2011) accounts for three sources of sensitivity: data, model and
prior. Moreover, it is capable of handling additive contamination class perturbations of
the form
_ω_ ( _t_ ) = _p_ ( _θ_ ) + _t_ ( _g_ ( _θ_ ) _−_ _p_ ( _θ_ )) _,_


with _p_ ( _θ_ ) and _g_ ( _θ_ ) denoting the base prior and the contaminating distributions, respectively. Such perturbations are conveniently examined in the context of differential
geometry. Comparison of sensitivity estimates produced by different discrepancy measures is necessary. Ibrahim et al. (2011) use a parametric bootstrap to indicate sensitivity
‘significance’ in the context of survival models.


**3.2** **Epsilon** **local** **sensitivity** **based** **on** **the** **Kullback-Leibler**
**divergence**


The first step for the _ϵ_ -local sensitivity computation consists of finding the epsilon grid
G _**γ**_ 0 ( _ϵ_ ) with respect to the chosen discrepancy measure and the base prior distribution
by Equation (2). For a normal prior and the Kullback-Leibler divergence, the epsilon
grid G _**γ**_ 0 ( _ϵ_ ) is specified by analytical formulae (16) and (17) derived in Appendix B3.
Additionally, given a specific _ϵ_ value, the circular _ϵ_ -local sensitivity _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [)] [defined] [in]
Equation (1) can also be computed analytically for each _**γ**_ = ( _μ, λ_ ) _∈_ G _**γ**_ 0 ( _ϵ_ ).

In particular, for _ϵ_ 0 = 5e-05, the resulting circular sensitivity estimates _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [0][)] [at]
_**γ**_ 0 = ( _μ_ 0 _, λ_ 0) = (70 _,_ 0 _._ 5) are shown in Figure 2 and the worst-case sensitivity _S_ _**γ**_ 0 ( _ϵ_ 0) =
3.304 is indicated by the red dot on the black curve. Note that this plot operates in
three dimensions as it indicates the values of the circular sensitivity in each direction
around _**γ**_ 0. The scale for sensitivity values (0.1, 0.2, _. . ._, 0.9, 1) is indicated by grey
circles. Sensitivity equal to 0.5 is marked by a black circle, whereas the value 1 is
marked by a red one. The worst-case sensitivity found by the _ϵ_ -local approach lies on
the red line, which corresponds to the principal eigenvector given by the infinitesimal
local sensitivity approximation. Estimates of both worst-case sensitivity values based
on the Kullback-Leibler divergence are comparable.


In case of the _ϵ_ -local sensitivity it is interesting to look at its calibration (see Lemma 1
in Appendix B3). In particular, the choice of _ϵ_ 0 = 5e-05 for the Kullback-Leibler divergence corresponds to a shift by 0.01 in means of the unit-variance normal distribution.
Following Equation (18), the calibrated sensitivity values are obtained by taking the
square root of the circular _ϵ_ -local sensitivity estimates based on the Kullback-Leibler
divergence (Figure 3). The calibrated worst-case sensitivity equals 1.82 as compared to
the calibrated infinitesimal local sensitivity 1.81.


There are two objections towards the use of the Kullback-Leibler divergence as a
discrepancy measure. First, for complex Bayesian hierarchical models prior elicitation
at different hierarchy levels is frequently impossible. Therefore, it is quite premature
to assume a non-elicited base distribution to be a true one, as the directed KullbackLeibler divergence actually does. Second, the unboundedness of the Kullback-Leibler
divergence makes it less suitable for dealing with numerical representations of marginal
posterior distributions (Roos and Held, 2011). To overcome both objections the use


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 331


Figure 3: Comparison of calibrated local sensitivity estimates from Figure 2 centered at
_**γ**_ 0 = ( _μ_ 0 _, λ_ 0) = (70 _,_ 0 _._ 5). Red and green lines: eigenvectors provided by the infinitesimal
local approach based on the Kullback-Leibler diverge ~~_√_~~ nce with the calibrated principal
eigenvalue (worst-case infinitesimal local sensitivity) 3 _._ 284 = 1.81. Black curve: circu
lar sensitivity in each direction provided by the _ϵ_ -local approach for _ϵ_ 0 = 5e-05 based
on the Kullback-Leibler divergence. Blue curve: circular _ϵ_ -local sensitivity based on the
Hellinger distance for _ϵ_ 0 = 0.00354. In both cases the choice of _ϵ_ 0 corresponds to a shift
in means by 0.01 with respect to the unit-variance normal distribution. The worst-case
sensitivities provided by the _ϵ_ -local approaches are indicated by a red dot. Both black
and blue curves match well. Scale for sensitivity values (0.1, 0.2, _. . ._, 0.9, 1) is indicated
by grey circles. Sensitivity equal to 0.5 is marked by a black circle, whereas the value 1
is marked by a red one.


of a symmetric and bounded discrepancy measure might be more appropriate. In the
next subsection we discuss the _ϵ_ -local sensitivity based on the Hellinger distance, which
exhibits both requested properties.


**3.3** **Epsilon** **local** **sensitivity** **based** **on** **the** **Hellinger** **distance**


In the case of the normal prior and the Hellinger distance, the epsilon grid is provided by
the analytical Formulae (20) and (21) in Appendix B4. In particular, for _ϵ_ 0 = 0.00354 at
_**γ**_ 0 = ( _μ_ 0 _, λ_ 0) = (70 _,_ 0 _._ 5), circular sensitivity estimates _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [0][)] [are] [computed] [as] [shown]
by the blue curve in Figure 2. The worst-case sensitivity _S_ _**γ**_ 0 ( _ϵ_ 0) indicated by the red
dot on the blue curve attains a value of 1.823. The position of the worst-case sensitivity
in the polar plot indicates that the marginal posterior is affected more by changes in the


332 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


precision _λ_ than in the expectation _μ_ . Further computational aspects of this example
are discussed in Appendix C5. Notably, the _ϵ_ -local sensitivity estimates are stable for a
wide range of _ϵ_ value choices.


We apply the calibration of the Hellinger distance with respect to the unit-variance
normal distribution discussed in Section 2.2 and Appendix B4. The use of _ϵ_ 0 = 0.00354
for the epsilon grid search corresponds to a unit-variance normal distribution with mean
equal to 0.01. Furthermore, by Formula (5), the circular _ϵ_ -local sensitivity estimates
_S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [0][)] [approximate] [directly] [the] [corresponding] [sensitivities] [with] [respect] [to] [the] [unit-]
variance normal distribution. Calibration revealed super-sensitivity for _m_, as the mean
change in the unit-variance normal distributions for the posteriors is 182.3% of the mean
change in the unit-variance normal distributions for priors.


The calibration facilitates the comparison of sensitivity estimates when different
discrepancy measures are used. Although the _ϵ_ -local sensitivity estimates based on the
Kullback-Leibler divergence and the Hellinger distance in Figure 2 differ, the corresponding _ϵ_ -local sensitivity estimates calibrated with respect to the unit-variance normal distribution in Figure 3 agree well. Both worst-case _ϵ_ -local sensitivity estimates lie
on the red principal eigenvector line provided by the infinitesimal local approach.


The _ϵ_ -local sensitivity approach concentrates only on the parametric sensitivity to
the base prior parameter values for one model component _θ_ . It is easier and less general
than the methodology developed by Zhu et al. (2011). However, it is capable of providing not only the worst-case sensitivity estimate but also other descriptive statistics
of circular sensitivity estimates and circular plots. For example, a circular plot with
the worst-case _ϵ_ -local sensitivity 0.88, median 0.86 and minimum 0.71 conveys different information than the values 0.83, 0.4 and 0.03. Careful investigation of each model
component may provide valuable suggestions at the stage of model building.


Interestingly, the _ϵ_ -local sensitivity approach suggests that the frequently used adhoc methodology (changing base prior parameter values and re-fitting the model) can
easily be turned into a formal one. In fact, it suggests not to use arbitrary parameter
values but standardized ones contained in the epsilon grid. When doing so, for a small
epsilon value the _ϵ_ -local sensitivity estimates will be close to the infinitesimal local ones.

## **4 Numerical computation**


We still need to address two vital topics dealing with the instantaneous computation
of posterior density for differing prior parameter values and computation of the sensitivity measure itself. The previous illustrative example is quite optimistic, as both the
marginal posterior distribution and the epsilon grid are accessible analytically. In practice, however, the estimates of the marginal posteriors ( _θ_ [(] _[j]_ [)] _,_ ˜ _π_ _**γ**_ 0 ( _θ_ [(] _[j]_ [)] _|_ _**y**_ )) are obtained
numerically on a finite set _j_ = 1 _, . . ., J_ corresponding to a non-negligible probability
mass. In addition, the epsilon grid for any arbitrary prior distribution based on the
Hellinger distance cannot be computed analytically. Below we concentrate on the first
issue, while the general methodology needed for the _ϵ_ -grid search will be explained in
the next subsection.


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 333


**4.1** **Fast** **computation**


In general, given the marginal posterior density _π_ _**γ**_ 0 ( _θ|_ _**y**_ ) computed for the base prior
_π_ _**γ**_ 0 ( _θ_ ), the marginal posterior density _π_ _**γ**_ ( _θ|_ _**y**_ ) for the prior _π_ _**γ**_ ( _θ_ ) with a new parameter
specification _**γ**_ instead of _**γ**_ 0 can be computed instantaneously by


_π_ _**γ**_ ( _θ|_ _**y**_ ) _∝_ _[π]_ _**[γ]**_ [0] [(] _[θ][|]_ _**[y]**_ [)] (7)

_π_ _**γ**_ 0 ( _θ_ ) _[π]_ _**[γ]**_ [(] _[θ]_ [)]


as proved in Appendix A. It is an extremely general and useful formula, which is the
basis of, among others, importance sampling (Gelman et al., 2004) and the Laplace
approximation (Tierney and Kadane, 1986). Formula (7) applied to an estimate of the
marginal posterior distribution at the base prior _π_ ˜ _**γ**_ 0 ( _θ|_ _**y**_ ), provided by any tool capable
of supplying the marginal posterior density, gives


_π_ ˜ _**γ**_ ( _θ|_ _**y**_ ) _∝_ _[π]_ [˜] _**[γ]**_ [0] [(] _[θ][|]_ _**[y]**_ [)]

_π_ _**γ**_ 0 ( _θ_ ) _[π]_ _**[γ]**_ [(] _[θ]_ [)] _[.]_


This general observation allows for an instantaneous computation of the Hellinger distance between two marginal posteriors _π_ ˜ _**γ**_ 0 ( _θ|_ _**y**_ ) and _π_ ˜ _**γ**_ ( _θ|_ _**y**_ ), arising from two slightly
shifted prior parameter values _**γ**_ 0 and _**γ**_ _∈_ G _**γ**_ 0( _ϵ_ ). Note that

~~�~~
H(˜ _π_ _**γ**_ ( _θ|_ _**y**_ ) _,_ ˜ _π_ _**γ**_ 0 ( _θ|_ _**y**_ )) = 1 _−_ BC(˜ _π_ _**γ**_ ( _θ|_ _**y**_ ) _,_ ˜ _π_ _**γ**_ 0 ( _θ|_ _**y**_ )) _,_


with - ~~�~~
BC(˜ _π_ _**γ**_ ( _θ|_ _**y**_ ) _,_ ˜ _π_ _**γ**_ 0 ( _θ|_ _**y**_ )) _≈_ _π_ ˜ _**γ**_ ( _θ|_ _**y**_ )˜ _π_ _**γ**_ 0 ( _θ|_ _**y**_ ) _dθ_


leads directly to the circular sensitivity estimates _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [)] [and] [the] [worst-case] [sensitivity]
_S_ _**γ**_ 0 ( _ϵ_ ) in Equations (1) and (3). In general, for the computation of BC(˜ _π_ _**γ**_ ( _θ|_ _**y**_ ) _,_ ˜ _π_ _**γ**_ 0 ( _θ|_ _**y**_ ))
numerical integration is used. Although the true BC( _π_ _**γ**_ ( _θ|_ _**y**_ ) _, π_ _**γ**_ 0 ( _θ|_ _**y**_ )) is restricted to
the unit interval, the numerical approximation occasionally gives values slightly larger
than one making truncation at 1 necessary. In such a case, minimal circular _ϵ_ -local
sensitivity estimates equal 0 and non-smooth circular sensitivity plots would indicate
this numerical inaccuracy.


The above approach to instantaneous computation of the posterior for differing prior
parameter specifications makes the model re-fit unnecessary. Similar computations can
be carried out within any framework supporting marginal posterior density _π_ _**γ**_ 0 ( _θ|_ _**y**_ )
estimation and capable of providing ( _θ_ [(] _[j]_ [)] _,_ ˜ _π_ _**γ**_ 0 ( _θ_ [(] _[j]_ [)] _|_ _**y**_ )), for _j_ = 1 _, . . ., J_, numerically.
We recommend, however, that for marginal posterior densities of precisions their logtransformed representations are used.


**4.2** **Grid** **search**


For a fixed, small _ϵ_ the search for the grid G _**γ**_ 0 ( _ϵ_ ) defined in Equation (2) requires exploration of the geometry around the prior parameter values _**γ**_ 0 in the space of distributions
_π_ _**γ**_ ( _θ_ ) (Figure 1). The goal is to find the set of prior parameter specifications _**γ**_ such


334 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


that Hellinger distance between _π_ _**γ**_ ( _θ_ ) and the base prior _π_ _**γ**_ 0 ( _θ_ ) is equal to _ϵ_ fulfilling
G _**γ**_ 0 ( _ϵ_ ) = _{_ _**γ**_ : H( _π_ _**γ**_ ( _θ_ ) _, π_ _**γ**_ 0 ( _θ_ )) _−_ _ϵ_ = 0 _}_ . In order to find an _ϵ_ grid for a base prior
distribution in, say, two dimensions, a suitable transformation of the Cartesian coordinates ( _γ_ [(1)] _, γ_ [(2)] ) to the polar coordinates ( _φ, r_ ) is used, where _φ_ and _r_ denote the angle
in radians and modulus, respectively. For sake of stability of the algorithm log( _r_ ) = _z_ is
considered. We use a scaling factor (exp( _z_ ) cos( _φ_ ) _,_ exp( _z_ ) sin( _φ_ )) which transforms the
base prior parameter values ( _γ_ 0 [(1)] _[, γ]_ 0 [(2)][)] [into] [an] _[ϵ]_ [-distant] [pair] [(] _[γ]_ [(1)] _[, γ]_ [(2)][)] [by] [finding] [the]
roots of the analytical equation numerically. Further details on the back-transformation
used can be found in Appendix C3.


The polar coordinates approach guarantees that each direction is treated equally as
the angles _φ_ run through an equidistant grid on the [ _−π, π_ ] interval. It also implies a
natural ordering of the grid points. This polar approach is applied to both normal and
gamma priors used in applications in Section 5 and in the Supplementary Material but
could easily be applied to any other two-parameter prior distribution of interest or even
extended to higher dimensions.

## **5 Application: Disease mapping**


In this section we demonstrate the use of the circular and worst-case _ϵ_ -local sensitivity
for a Bayesian hierarchical model. Additional examples for data sets with increasing hierarchical model complexity are discussed in the Supplementary Material. Marginal posterior densities necessary for the _ϵ_ -local sensitivity computation have been contributed
by inla (Appendix C).


The analysis of spatial variation of lip cancer in Scotland is based on a non-conjugate
hierarchical model (Breslow and Clayton, 1993). Here, we consider observed ( _y_ ) and
expected ( _e_ ) cases of lip cancer registered between 1975 and 1980 in each of the _n_ = 56
counties in Scotland. We include an intercept _α_ ; a covariable _x_ denoting the proportion
of individuals who are employed in agriculture, fishing or forestry scaled by 1 _/_ 10; a
known offset log _e_ ; and spatial components _**v**_ and _**u**_ .


The unstructured region specific random effects in the vector _**v**_ consist of independent Gaussian distributed random variables _vi_ _iid∼_ N(0 _, τ_ _**v**_ _[−]_ [1][)] [with] [precision] _[τ]_ _**[v]**_ [,]
for which the gamma prior is assumed. A more involved structured intrinsic conditional autoregressive model (Besag et al., 1991) in component _**u**_ assumes that conditions for neighbouring random effects tend to be similar. The Gaussian random field
_**u**_ = ( _u_ 1 _, u_ 2 _, . . ., un_ ) is defined as



_ui|uj, i ̸_ = _j, τ_ _**u**_ _∼_ N( [1]

_ni_




- 1

_uj,_ ) _,_
_i∼j_ _niτ_ _**u**_



where _i_ _∼_ _j_ indicates that two random effects _i_ and _j_ are neighbours and _ni_ is the
number of neighbouring entities of the _i_ th object. In order to guarantee the identifiability
of the intercept, a sum-to-zero constraint on each connected component is used. For the
precision _τ_ _**u**_ a gamma prior is assumed.


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 335


Let _yi_ be realisations of _Yi | μi_ _ind∼_ Po( _μi_ ) _,_ _i_ = 1 _, . . ., n_ and consider the following six
models:


log _μi_ = log _ei_ + _α_ + _vi_ (8)

log _μi_ = log _ei_ + _α_ + _ui_ (9)

log _μi_ = log _ei_ + _α_ + _vi_ + _ui_ (10)

log _μi_ = log _ei_ + _α_ + _βx_ + _vi_ (11)

log _μi_ = log _ei_ + _α_ + _βx_ + _ui_ (12)

log _μi_ = log _ei_ + _α_ + _βx_ + _vi_ + _ui._ (13)


Table 1: Worst-case sensitivity estimates for model components _α_, _β_, _τ_ _**v**_ and _τ_ _**u**_ in
Section 5 for _ϵ_ 0 = 0.00354.


Model _α_ _β_ _τ_ _**v**_ _τ_ _**u**_


(8) 0.004 0.244
(9) 0.002 0.245
(10) 0.002 1.587 0.274
(11) 0.005 0.004 0.237
(12) 0.004 0.004 0.268
(13) 0.005 0.005 1.568 0.355


As Fong et al. (2010, Supplementary Material) provided a very careful probabilistic
elicitation of prior values, we have adopted their choices here. For the intercept _α_ and
the regression coefficient of the covariate _β_, we assumed normal priors with base prior
parameter specification set at _**γ**_ 0 [(] _[α]_ [)] _[,]_ [(] _[β]_ [)] = (0 _,_ 0 _._ 001). For the unstructured and structured
components we assumed gamma priors for _τ_ _**v**_ and _τ_ _**u**_ with base prior parameter values
set to _**γ**_ 0 [(] _**[v]**_ [)] = (1 _,_ 0 _._ 14) and _**γ**_ 0 [(] _**[u]**_ [)] = (1 _,_ 0 _._ 34), respectively. Table 1 reports the worstcase sensitivities for each component in all six models for the base prior parameter
values defined above. As an example, consider the sensitivity values in the last row of
Table 1. For the structured component _**u**_, a worst-case sensitivity of 0.355 was found.
This value means that the mean change in the unit-variance normal distributions for
posteriors is 35.5% of the mean change of unit-variance normal distributions in the
prior. In contrast, super-sensitivity of 1.568 for the unstructured component _**v**_ shows
that the mean change in the unit-variance normal distributions for posteriors is 156.8%
of the mean change in the unit-variance normal distributions for priors. Figure 4 shows
for precisions _**v**_ and _**u**_ in model (13) the polar circular sensitivity plot centered at _**γ**_ 0
and rolled out on the line. Note that the left panel plots operate in three dimensions as
they indicate the values of the circular sensitivity in each direction around _**γ**_ 0 [(] _**[v]**_ [)] (top)
and _**γ**_ 0 [(] _**[u]**_ [)] (bottom).


Fong et al. (2010, Supplementary Material) prefer to always include both the unstructured _**v**_ and the structured _**u**_ components together. They argue that since the _**u**_


336 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


Figure 4: Polar sensitivity plots (left) centered at _**γ**_ 0 and sensitivity plots rolled out
on the line (right) in each polar direction of the circular _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [0][)] [together] [with] [the]

worst-case sensitivity _S_ _**γ**_ 0 ( _ϵ_ 0) (red dot) for _τ_ _**v**_ at _**γ**_ 0 [(] _**[v]**_ [)] (top) and _τ_ _**u**_ at _**γ**_ 0 [(] _**[u]**_ [)] (bottom)
components in model (13) in Section 5 for _ϵ_ 0 = 0.00354. The scale for sensitivity values
(0.1, 0.2, _. . ._, 0.9, 1) is indicated by grey circles and lines, respectively. Sensitivity equal
to 0.5 is marked by a black colour, whereas the sensitivity value 1 is marked by a red
one.


model contains only a single parameter to govern both the spatial extent of dependence and the strength of this dependence, there is no place for pure randomness to
be accommodated. Their models (0.3), (0.5), (0.4) and (0.6) are complemented with
worst-case sensitivity estimates in rows corresponding to Equations (8), (10), (11) and
(13) in Table 1. We conclude that for the lip cancer in Scotland data, having both _**v**_
and _**u**_ components at the same time in the model leads to super-sensitive marginal
posteriors for the precision of the unstructured component with respect to the base
prior parameter values choice. Apparently, a simultaneous inclusion of both latent components ends in an overparametrized model and the unstructured component becomes
nearly non-identifiable (Eberly and Carlin, 2000).


This example illustrates that, ideally, the local sensitivity estimates should be investigated prior to the release of the final model. We suggest that the sensitivity values
should be openly disclosed to the readers. In cases where a model with an included
super-sensitivity is released, its justification or all measures aiming for its impact mitigation should be discussed.


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 337

## **6 Discussion**


We introduced and utilized a new formal local robustness measure, which was able
to automatically handle both circular and worst-case sensitivities in complex Bayesian
hierarchical models. It hinged on two essential ingredients. Firstly, on an appropriately
generated grid, which provided a well standardized way for modification of the base prior
parameter values. Secondly, on an instantaneous computation of the marginal posterior
density for priors with parameter values contained in the grid without any model re-fit.


It is a formal local robustness approach which dispenses with Taylor-expansion approximation and infinitesimal asymptotics. Instead, the circular sensitivity is computed
for each polar direction chosen equidistantly on [ _−π, π_ ] around _**γ**_ 0. It can be both plotted and summarized by a single number. Whereas infinitesimal methods restrict their
output to the worst-case value, the deeper insight offered by the circular sensitivity and
its versatile summaries seems to be new.


We provided a careful and extensive investigation of the properties of the introduced
_ϵ_ -local sensitivity and comparison with the infinitesimal local sensitivity on conjugate
examples, where estimation of exact analytical sensitivity estimates was possible. Due
to limited space we reported mostly on the worst-case values. However, for all of the
examples, polar plots of the circular sensitivity estimates and other summaries were
available for more careful investigation.


As expected, a strong influence of the sample size on the prior sensitivity estimates emerged (Appendix C and Supplementary Material). Indeed, we observed that
our measure automatically adjusts for increasing sample size by returning smaller prior
sensitivity estimates. The choice of the _ϵ_ for the grid search did not have much influence on sensitivity estimates but anchored calibration in terms of unit-variance normal
distribution with shifted means. Our novel calibration use gave rise to a convenient interpretation of sensitivity estimates independent of the actual _ϵ_ choice. It also provided
a device to compare the sensitivity estimates obtained under both the Kullback-Leibler
divergence and the Hellinger distance assumptions.


We identified several model components and base parameter values requiring more
careful attention. Sensitivity estimates in Section 5 indicated clear identifiability problems when both unstructured and structured models were included in the model simultaneously. We believe that inclusion of both latent models at the same time leads to an
overparametrized model.


One possible drawback of our approach is that we investigated _ϵ_ -local sensitivity
for each model component separately while keeping all other model component prior
parameters fixed at their base values. It can happen, however, that a model is insensitive
to changes in only one input at time, while being sensitive to simultaneous changes
in more than one input. We believe, however, that _ϵ_ -local sensitivity for each model
component separately is actually what we are able to interpret in practice.


Another possible drawback of our approach is that it hinges on the choice of _ϵ_ for the
grid search surrounding the base prior parameter values. Fortunately, we were able to
show in Appendix C and in the Supplementary Material that sensitivity estimates stay


338 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


numerically stable over a wide range of _ϵ_ values leaving much freedom for its choice.
We stress, however, that it is essential to apply the same _ϵ_ for all model components
for which local sensitivity is examined in order to provide well standardized robustness
comparisons.


MCMC methodology is often not directly compatible with many of the robust
Bayesian techniques that had been developed (Berger et al., 2000). Lesaffre and Lawson
(2012) admit that a routine use of sensitivity procedures in MCMC cannot be afforded
due to a substantial computational burden. In contrast, our fast _ϵ_ -local sensitivity estimation technique has a potential to be implemented without much extra cost by any
framework capable of estimating of marginal posterior densities. In particular, when
the marginal posterior densities are obtained from MCMC samples followed by RaoBlackwellization (Gelfand and Smith, 1990) or a dynamic exploration of posteriors by
means of importance sampling (Narasimhan, 2005), our approach can be directly used.


The formal _ϵ_ -local sensitivity measure gave a novel, reasonable, easy to interpret
and useful piece of information about the marginal posterior distribution sensitivity
to base prior parameter values. Its use was not restricted to conjugate examples but
was easily extended to complex Bayesian hierarchical models giving new insight into
the identifiability of model components given the data at hand. Additionally, we were
able to determine in applications which model components were hard to learn from the
data and identified several base prior parameter specifications requiring more careful
attention. Therefore, we believe that thanks to our formal sensitivity measure checking
for local robustness in complex Bayesian hierarchical models will become a part of
routine statistical practice.

## **Appendix A (Proof)**


**Proof** **of** **the** **general** **Formula** **(7)** **used** **for** **instantaneous** **computation** **of** **the**
**slightly** **shifted** **posterior** **in** **Section** **4.1**



Denote by _**γ**_ 0 the base prior parameter values and by _**γ**_ the shifted prior parameter
values. Reformulate



_π_ _**γ**_ 0 ( _θ|_ _**y**_ ) = _[π]_ [(] _**[y]**_ _[|][θ]_ [)] _[π]_ _**[γ]**_ [0] [(] _[θ]_ [)]



_π_ _**γ**_ 0 ( _**y**_ )



to obtain


Consequently,


can be rewritten as



_π_ ( _**y**_ _|θ_ ) = _[π]_ _**[γ]**_ [0] [(] _[θ][|]_ _**[y]**_ [)] _[π]_ _**[γ]**_ [0] [(] _**[y]**_ [)] _._

_π_ _**γ**_ 0 ( _θ_ )




_[|][θ]_ [)] _[π]_ _**[γ]**_ [(] _[θ]_ [)] = _[π]_ _**[γ]**_ [0] [(] _[θ][|]_ _**[y]**_ [)] _[π]_ _**[γ]**_ [0] [(] _**[y]**_ [)] _[π]_ _**[γ]**_ [(] _[θ]_ [)]

_π_ _**γ**_ ( _**y**_ ) _π_ _**γ**_ ( _**y**_ ) _π_ _**γ**_ 0 ( _θ_ )



_π_ _**γ**_ ( _θ|_ _**y**_ ) = _[π]_ [(] _**[y]**_ _[|][θ]_ [)] _[π]_ _**[γ]**_ [(] _[θ]_ [)]



_π_ _**γ**_ ( _**y**_ ) _π_ _**γ**_ 0 ( _θ_ )



_π_ _**γ**_ ( _θ|_ _**y**_ ) = _[c]_ [(] _**[y]**_ [)] _[π]_ _**[γ]**_ [0] [(] _[θ][|]_ _**[y]**_ [)] _[π]_ _**[γ]**_ [(] _[θ]_ [)]




[0] [(] _[θ][|]_ _**[y]**_ [)] _[π]_ _**[γ]**_ [(] _[θ]_ [)]

_∝_ _[π]_ _**[γ]**_ [0] [(] _[θ][|]_ _**[y]**_ [)] _[π]_ _**[γ]**_ [(] _[θ]_ [)]
_π_ _**γ**_ 0 ( _θ_ ) _π_ _**γ**_ 0 ( _θ_ )



_,_
_π_ _**γ**_ 0 ( _θ_ )



with _c_ ( _**y**_ ) = _π_ _**γ**_ 0 ( _**y**_ ) _/π_ _**γ**_ ( _**y**_ ).


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 339

## **Appendix B (Sensitivity)**


**B1** **Additional** **literature** **about** **sensitivity**


Saltelli et al. (2004) and Cacuci et al. (2005) are prominent advocates for uncertainty
analyses with respect to influential observations and the sampling distribution in complex frequentist models. In Bayesian statistics an additional source of uncertainty due
to prior inadequacy has to be accounted for. The general sensitivity concept studied by
Lavine (1992), Geisser (1993) and Gustafson and Wasserman (1995) involves consideration of influential observations, uncertainty of the sampling distribution and prior
inadequacy at the same time. Zhu et al. (2007) develop the differential geometric approach to deal with this topic via geometric contamination of the prior distribution. The
earlier advances by Van der Linde (2007) become special cases of this general methodology. The robustness analysis deals with the sensitivity of the results with respect to
the prior distribution only. Weiss and Cook (1992) suggested a graphical approach for
assessing posterior influence. In particular, the local approach is discussed by Gustafson
(1996). Other advances to local Bayesian robustness can be found in Kass et al. (1989)
and Weiss (1996).


**B2** **Formulae** **for** **the** **infinitesimal** **local** **sensitivity** **based** **on** **the**
**Kullback-Leibler** **divergence** **in** **Section** **3.1**


Denote by _π_ _**γ**_ 0 and _π_ _**γ**_ the densities of the N( _μ_ 0 _, λ_ _[−]_ 0 [1][)] [(true)] [and] [N(] _[μ, λ][−]_ [1][)] [(arbitrary)]
distributions, respectively. The value of the Kullback-Leibler divergence




       -       DKL( _π_ _**γ**_ 0 _||π_ _**γ**_ ) = _π_ _**γ**_ 0 ( _θ_ ) log _ππ_ _**γγ**_ 0(( _θθ_ ))




_dθ_



between both distributions reads as follows:

                -                 DKL( _π_ _**γ**_ 0 _||π_ _**γ**_ ) = _λ_ ( _μ_ 0 _−_ _μ_ ) [2] _/_ 2 + _λ/λ_ 0 _−_ 1 _−_ log( _λ/λ_ 0) _/_ 2 _._ (14)



For a fixed value of _**γ**_ 0 = ( _μ_ 0 _, λ_ 0) it is a function with respect to two arbitrary
parameters ( _μ, λ_ ) = _**γ**_ . In the illustrative example in Section 3 the model component of interest is a scalar _θ_ = _m_ . Equation (14) leads to an explicit expression for
_i_ ( _**γ**_ ) = DKL( _π_ _**γ**_ 0 ( _m_ ) _||π_ _**γ**_ ( _m_ )). Due to conjugacy, Equation (14) can be re-used for analytical computation of the Kullback-Leibler divergence between two normal posteriors with parameters ( _nκy_ ¯ + _λ_ 0 _μ_ 0) _/_ ( _nκ_ + _λ_ 0), _nκ_ + _λ_ 0, ( _nκy_ ¯ + _λμ_ ) _/_ ( _nκ_ + _λ_ ), _nκ_ + _λ_
instead of _μ_ 0, _λ_ 0, _μ_ and _λ_, respectively. This provides an analytical expression for
_d_ ( _**γ**_ ) = DKL( _π_ _**γ**_ 0 ( _m|_ _**y**_ ) _||π_ _**γ**_ ( _m|_ _**y**_ )). Additionally, the entries of the matrices **D** [2] _i_ ( _**γ**_ 0) and
**D** [2] _d_ ( _**γ**_ 0) can be derived analytically by differentiating functions _i_ ( _**γ**_ ) and _d_ ( _**γ**_ ) twice
with respect to the parameters _μ_ and _λ_ and evaluating the result at _**γ**_ 0 = ( _μ_ 0 _, λ_ 0). This
leads to - 



    **D** [2] _i_ ( _**γ**_ 0) = _λ_ 00 (2 _λ_ 0 [2] 0 [)] _[−]_ [1]







and




    _λ_ [2] 0 _[/]_ [(] _[nκ]_ [ +] _[ λ]_ 0 [)] _nκλ_ 0( _μ_ 0 _−_ _y_ ¯) _/_ ( _nκ_ + _λ_ 0) [2]
**D** [2] _d_ ( _**γ**_ 0) = [2] [2] _[−]_ [1] [2]



_nκλ_ 0( _μ_ 0 _−_ _y_ ¯) _/_ ( _nκ_ + _λ_ 0) [2] (2( _nκ_ + _λ_ 0) [2] ) _[−]_ [1] + ( _nκ_ ( _μ_ 0 _−_ _y_ ¯)) [2] _/_ ( _nκ_ + _λ_ 0) [3]





_._


340 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


The observation that **D** [2] _i_ ( _**γ**_ 0) and **D** [2] _d_ ( _**γ**_ 0) are Fisher information matrices for the prior
and posterior families of distributions evaluated at _**γ**_ 0 leads to an identical analytical
finding. The resulting matrix ( **D** [2] _i_ ( _**γ**_ 0)) _[−]_ [1] **D** [2] _d_ ( _**γ**_ 0) = **A** has the following entries:


_a_ 11 = _λ_ 0 _/_ ( _nκ_ + _λ_ 0) _,_
_a_ 12 = _nκ_ ( _μ_ 0 _−_ _y_ ¯) _/_ ( _nκ_ + _λ_ 0) [2] _,_
_a_ 21 = 2 _nκλ_ [3] 0 [(] _[μ]_ [0] _[−]_ _[y]_ [¯][)] _[/]_ [(] _[nκ]_ [ +] _[ λ]_ [0][)][2] _[,]_
_a_ 22 = _λ_ [2] 0 _[/]_ [(] _[nκ]_ [ +] _[ λ]_ [0][)][2][ + 2] _[λ]_ [2] 0 [(] _[nκ]_ [(] _[μ]_ [0] _[−]_ _[y]_ [¯][))][2] _[/]_ [(] _[nκ]_ [ +] _[ λ]_ [0][)][3] _[.]_


The eigenvalues can be also computed analytically by using Equation (15) with entries
from matrix **A** :

             - ~~�~~             eigenvalue [max] _,_ eigenvalue [min] = _a_ 11 + _a_ 22 + _/ −_ ( _a_ 11 _−_ _a_ 22) [2] + 4 _a_ 21 _a_ 12 _/_ 2 _._ (15)


**B3** **Formulae** **for** _**ϵ**_ **-local** **sensitivity** **based** **on** **the** **Kullback-Leibler**
**divergence** **in** **Section** **3.2**


**Epsilon** **grid** **based** **on** **the** **Kullback-Leibler** **divergence**


For normal prior distribution, following equation (14), the search for the epsilon grid


G _**γ**_ 0( _ϵ_ ) = _{_ _**γ**_ : DKL( _π_ _**γ**_ 0 _||π_ _**γ**_ ) = _ϵ}_


corresponds to finding roots of


DKL( _π_ _**γ**_ 0 _||π_ _**γ**_ ) _−_ _ϵ_ = 0 _,_


which is equivalent to finding roots _μ_ of

             -             _λ_ ( _μ_ 0 _−_ _μ_ ) [2] _/_ 2 + _λ/λ_ 0 _−_ 1 _−_ log( _λ/λ_ 0) _/_ 2 _−_ _ϵ_ = 0


for _λ_, _μ_ 0 and _λ_ 0 kept fixed. For normal prior the roots themselves

~~�~~
_μ_ [min] _, μ_ [max] = _μ_ 0 _−_ _/_ + _−_ ( _λ/λ_ 0 _−_ 1 _−_ log( _λ/λ_ 0) _−_ 2 _ϵ_ ) _/λ_ (16)


and an approximate range of _λ_, in which roots exist,

                   - ~~�~~ ~~�~~
_λ_ [min] _, λ_ [max] = _λ_ 0 1 + _ϵ −_ _/_ + _ϵ_ (2 + _ϵ_ ) (17)


can be found analytically.


**Calibration** **of** **the** **Kullback-Leibler** **divergence** **with** **respect** **to** **the** **normal**
**distribution**


Calibration of the Kullback-Leibler divergence with respect to the normal distribution
has been already discussed by Goutis and Robert (1998). For sake of completeness, we
provide the formulae explicitly below. They follow directly from Equation (14).


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 341


**Lemma** **1.** _Calibration_ _of_ _the_ _Kullback-Leibler_ _divergence_ _k_ _between_ _two_ _normal_ _densi-_
_ties_ _N(0,1)_ _and_ _N(μ,1),_ _respectively,_ _can_ _be_ _computed_ _as_ _follows:_


_k_ ( _μ_ ) = _μ_ [2] _/_ 2


~~_√_~~
_μ_ ( _k_ ) = 2 _k_


**Proof.** Assume _μ_ 0 = 0, _μ_ 1 = _μ_ with precisions _λ_ 0 = _λ_ 1 = 1 in Equation (14 ~~_√_~~ ) to
obtain _k_ ( _μ_ ) = _k_ = _μ_ [2] _/_ 2. Solve this equality with respect to _μ_ to obtain _μ_ ( _k_ ) = 2 _k_ .


Note that _μ_ ( _k_ ) is the calibration of the Kullback-Leibler divergence _k_ with respect
to the unit-variance normal distribution, as _k_ between any two densities is the same
as that between N(0,1) and N( _μ_ ( _k_ ),1). If the Kullback-Leibler divergence _k_ between
any two densities is known, then we can quantify discrepancies between them in terms
of the differences in mean from 0 to _μ_ ( _k_ ) for normal distribution with unit standard
deviation. In particular, if we are interested in a mean shift _μ_ = 0 _._ 01, then the value
of the _ϵ_ for the epsilon grid based on the Kullback-Leibler divergence can be fixed by
shift _k_ ( _μ_ ) =in _k_ means(0 _._ 01) =by _ϵ_ 00.01= 5e-05. In such a case, thefor unit-variance normalchoice ofdistribution _ϵ_ 0 = 5e-05 corresas _μ_ ( _ϵ_ 0) = ~~_[√]_~~ pond2 _ϵ_ 0 s to the= 0 _._ 01.
Moreover,




 - _μ_ DKL( _π_ _**γ**_ 0 ( _θ|_ _**y**_ ) _||π_ _**γ**_ ( _θ|_ _**y**_ )) =

_μ_ ( _ϵ_ )




DKL( _π_ _**γ**_ 0 ( _θ|_ _**y**_ ) _||π_ _**γ**_ ( _θ|_ _**y**_ )) _._ (18)

_ϵ_



Consequently, the square root of the _ϵ_ -local sensitivity based on the Kullback-Leibler
divergence corresponds to the _ϵ_ -local sensitivity in terms of unit-variance normal distributions.


**B4** **Formulae** **for** _**ϵ**_ **-local** **sensitivity** **based** **on** **the** **Hellinger** **distance** **in**
**Section** **3.3**


**Hellinger** **distance** **between** **two** **normal** **distributions**


Denote by _π_ _**γ**_ 0 and _π_ _**γ**_ the densities of the N( _μ_ 0 _, λ_ _[−]_ 0 [1][)] [and] [N(] _[μ, λ][−]_ [1][)] [distributions,]
respectively. The value of the Hellinger distance




~~�~~ _∞_ - ~~�~~ ~~�~~ ~~�~~ 2

_π_ _**γ**_ ( _θ_ ) _−_ _π_ _**γ**_ 0 ( _θ_ ) _dθ_

_−∞_



H( _π_ _**γ**_ _, π_ _**γ**_ 0 ) =




~~�~~
1
2



between both distributions reads as follows:




    
2 _λ_ 0 _λ_ _−_ _[λ]_ [0] _[λ]_ [(] _[μ]_ [0] _[ −]_ _[μ]_ [)][2]

_λ_ 0 + _λ_ [exp] 4( _λ_ 0 + _λ_ )



4( _λ_ 0 + _λ_ )



H( _π_ _**γ**_ _, π_ _**γ**_ 0 ) =







1 _−_




~~�~~
2 ~~_[√]_~~ _λ_ 0 _λ_




_._ (19)


342 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


**Epsilon** **grid** **based** **on** **the** **Hellinger** **distance**


For normal prior distribution, following equation (19), the search for the epsilon grid


G _**γ**_ 0 ( _ϵ_ ) = _{_ _**γ**_ : H( _π_ _**γ**_ _, π_ _**γ**_ 0 ) = _ϵ}_


corresponds to finding roots of


H( _π_ _**γ**_ _, π_ _**γ**_ 0 ) _−_ _ϵ_ = 0 _,_



which is equivalent to finding roots _μ_ of

~~�~~

             
2 ~~_[√]_~~ _λ_ 0 _λ_

_[λ]_ [0]




    
2 _λ_ 0 _λ_ _−_ _[λ]_ [0] _[λ]_ [(] _[μ]_ [0] _[ −]_ _[μ]_ [)][2]

_λ_ 0 + _λ_ [exp] 4( _λ_ 0 + _λ_ )





_−_ (1 _−_ _ϵ_ [2] ) = 0



4( _λ_ 0 + _λ_ )



for _λ_, _μ_ 0 and _λ_ 0 kept fixed. For normal prior not only the range of _λ_, in which roots
exist,

                -                 _λ_ [min] _, λ_ [max] = _λ_ 0 1 _−_ _/_ + 1 _−_ (1 _−_ _ϵ_ [2] ) [4] ~~[�]~~ [2] (1 _−_ _ϵ_ [2] ) _[−]_ [4] (20)


but also roots themselves




~~�~~
_λ_ 0 + _λ_
2 ~~_[√]_~~ _λ_ 0 _λ_



_μ_ [min] _, μ_ [max] = _μ_ 0 _−_ _/_ +


can be found analytically.






- 
- [+] _[ λ]_ [)]

_−_ [4(] _[λ]_ [0] log (1 _−_ _ϵ_ [2] )
_λ_ 0 _λ_




(21)



Given the grid G _**γ**_ 0 ( _ϵ_ ) specified by equations (20) and (21) the circular _ϵ_ -local sensitivity _S_ _**γ**_ _[c]_ 0 [(] _[ϵ]_ [)] [defined] [in] [Equation] [(][1][)] [based] [on] [the] [Hellinger] [distance] [can] [be] [obtained]
analytically. The Hellinger distance between two priors can be directly computed analytically using Equation (19). Due to conjugacy, Equation (19) can be used again for
analytical computation of the Hellinger distance between two posteriors with parameters ( _nκy_ ¯ + _λ_ 0 _μ_ 0) _/_ ( _nκ_ + _λ_ 0), _nκ_ + _λ_ 0, ( _nκy_ ¯ + _λμ_ ) _/_ ( _nκ_ + _λ_ ), _nκ_ + _λ_ instead of _μ_ 0, _λ_ 0,
_μ_ and _λ_, respectively.


**Calibration** **of** **the** **Hellinger** **distance** **with** **respect** **to** **the** **normal** **distribution**


**Lemma** **2.** _Calibration_ _of_ _the_ _Hellinger_ _distance_ _h_ _between_ _two_ _normal_ _densities_ _N(0,1)_
_and_ _N(μ,1),_ _respectively,_ _can_ _be_ _computed_ _as_ _follows:_


         _h_ ( _μ_ ) = 1 _−_ exp( _−μ_ [2] _/_ 8)

         _μ_ ( _h_ ) = _−_ 8 log(1 _−_ _h_ [2] )


**Proof.** Assume ~~�~~ _μ_ 0 = 0, _μ_ 1 = _μ_ and precisions _λ_ 0 = _λ_ 1 = 1 in Equation (19) to
obtain _h_ ( _μ_ ) = _h_ = 1 _−_ exp( _−μ_ [2] _/_ 8). Solve this equality with respect to _μ_ to obtain

~~�~~
_μ_ ( _h_ ) = _μ_ = _−_ 8 log(1 _−_ _h_ [2] ).


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 343


Table 2: Normal calibration of the Hellinger distance. Left: equal _μ_ = 1 increment,
Right: equal _h_ = 0 _._ 1 increment.

|h = h(μ) μ(h) = μ|h = h(μ) μ(h) = μ|
|---|---|
|0.000<br>0.000<br>0.343<br>1.000<br>0.627<br>2.000<br>0.822<br>3.000<br>0.930<br>4.000<br>0.978<br>5.000<br>0.994<br>6.000<br>0.999<br>7.000<br>1.000<br>8.000<br>1.000<br>9.000|0.000<br>0.000<br>0.100<br>0.284<br>0.200<br>0.571<br>0.300<br>0.869<br>0.400<br>1.181<br>0.500<br>1.517<br>0.600<br>1.890<br>0.700<br>2.321<br>0.800<br>2.859<br>0.900<br>3.645|



Table 2 demonstrates the values obtained by calibration. In particular, if we are

interested in a mean shift _μ_ = 0 _._ 01, then the value of the _ϵ_ for the epsilon grid based on
the Hellinger distance can be fixed by _h_ ( _μ_ ) = _h_ (0 _._ 01) = _ϵ_ 0 = 0.00354. By Equation (5)
the _ϵ_ -local sensitivity based on the Hellinger distance approximates the calibrated ratio

with respect to the unit-variance normal distribution directly.

## **Appendix C (Computations)**


**C1** R-INLA **review**


A wide range of Bayesian problems are covered by the latent Gaussian models framework and therefore effectively handled by INLA (Rue et al., 2009). An R package
[(http://www.r-inla.org)](http://www.r-inla.org) called INLA serves as an interface to the inla program.
Its usage is similar to the familiar user-friendly glm function in R. The inla program
allows the user to conveniently perform approximate Bayesian inference in latent Gaus
sian models. It is a fast and very versatile program, providing full Bayesian analysis

of generalized linear mixed models (Fong et al., 2010; Martins et al., 2013). Computa
tionally expensive models on high-dimensional data within stochastic partial differential
equations (SPDEs) framework (Lindgren et al., 2011) can be tackled by inla as well.
As output marginal posterior densities of all parameters in the model together with
summary characteristics are offered by default. Although inla provides diagnostics
for outlying observations via the conditional predictive ordinate (CPO) (Pettit, 1990;

Geisser, 1993) default prior sensitivity diagnostics are still missing. Here, we closed this
gap and provided a ready to use priorSens package in R facilitating the use of sensitivity
measure described here.


344 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


**C2** **The** priorSens **package** **in** R


We developed a handy package priorSens for routine, every-day sensitivity computation. It can be used with practically no extra programming effort needed for default
sensitivity investigations within inla or to obtain standardized grids for any alternative robustness considerations. Currently, normal and gamma priors are supported by
the package covering already a great amount of possible models. Further extensions to
support other priors are possible. The priorSens package can be obtained upon request
from the authors. We plan to include it as a default option in the R-INLA framework.
All sensitivity estimates presented in Section 5 and in the Supplementary Material were
evaluated by the priorSens package in R. The required input for the _ϵ_ -local sensitivity to
be computed are the base prior distribution _π_ _**γ**_ 0 ( _θ_ ) with parameter specification _**γ**_ 0 and
the corresponding marginal posterior density ( _θ_ [(] _[j]_ [)] _,_ ˜ _π_ _**γ**_ 0 ( _θ_ [(] _[j]_ [)] _|_ _**y**_ )) obtained numerically on
a finite set _j_ = 1 _, . . ., J_ .


**C3** **Back-transformation** **for** **grid** **search**


As discussed in Section 4.2 the scaling factor (exp( _z_ ) cos( _φ_ ) _,_ exp( _z_ ) sin( _φ_ )) is transformed back to Cartesian coordinates using

                 -                  _γ_ 0 [(1)] + _r_ cos( _φ_ ) _c_ _[x]_ ( _φ_ )


and - _γ_ 0 [(2)] + _r_ sin( _φ_ ) _c_ _[y]_ ( _φ_ ) _,_



where


and




   _r_ _[∗]_ (0) if _φ ∈_ [ _−π/_ 2 _, π/_ 2],
_c_ _[x]_ ( _φ_ ) =
_r_ _[∗]_ ( _π_ ) if _φ ∈_ [ _π/_ 2 _, −π/_ 2]


   _r_ _[∗]_ ( _π/_ 2) if _φ ∈_ [0 _, π_ ],
_c_ _[y]_ ( _φ_ ) =
_r_ _[∗]_ ( _−π/_ 2) if _φ ∈_ [ _π,_ 0],



with _r_ _[∗]_ ( _δ_ ), for radian values _δ_ = _−π/_ 2, 0, _π/_ 2, _π_, denoting the modulus values obtained
at _δ_ angles during a pre-exploration of the polar coordinate space. The factors _c_ _[x]_ ( _φ_ )
and _c_ _[y]_ ( _φ_ ) are necessary to scale the problem so that _r_ is close to 1 across different prior
distributions. This practice standardizes the task of computing G _**γ**_ 0 ( _ϵ_ ), which makes
the numerical algorithm more stable and generally applicable. The priorSens package
relies on the above general approach for epsilon grid search.


**C4** **Epsilon** **local** **sensitivity** **in** R-INLA


Our fast general _ϵ_ -local sensitivity methodology based on the Hellinger distance can be
implemented without much extra cost by any technique capable of computing marginal
posterior distributions, in particular, by the R-INLA framework. In practice, however,
two settings for the cardinality of the grid G _**γ**_ 0 ( _ϵ_ ) and the value of _ϵ_ have to be fixed
(Section 4.2). In applications shown below we consider 400 polar directions and use one
particular _ϵ_ 0 = 0.00354 for the grid search, which corresponds to a unit-variance normal


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 345


distribution with mean equal to 0.01 with respect to the Hellinger distance (Section 2.2
and Appendix B4). The continued illustrative example in Appendix C5 and several examples in the Supplementary Material indicate that the exact sensitivity estimates stay
stable over a wide range of _ϵ_ values. Therefore, there is some room for a tolerable _ϵ_ choice.


Our inla computations were run for default settings: simplified Laplace strategy
for the marginal posterior approximation and central composite design (CCD) for integrating out the hyperparameters. In general, for precisions log-transformed marginal
posterior densities were used. In addition, for precisions of latent components in R-INLA
such as unstructured (“iid”), structured (“ICAR”), the latent Gaussian random walk of
the first (“rw1”) and the second (“rw2”) order an appropriate tuning for their marginal
posteriors provided by the function inla.hyperpar() was utilized.


**C5** **Epsilon** **local** **sensitivity** **based** **on** **the** **Hellinger** **distance**
**(continued)**


Reconsider the model and data introduced in the illustrative example in Section 3.
We studied the _ϵ_ -local sensitivity values, when the exact analytical marginal posterior
density estimate was replaced by the approximate one provided by inla. The worst-case
sensitivity S [inla] _**γ**_ 0 [(] _[ϵ]_ [0][) = 1.82 for the][ inla][-driven approach and the analytically computed]
estimate S [exact] _**γ**_ 0 ( _ϵ_ 0) = 1.82 agreed perfectly well. Absolute and relative error ranges of
the inla-driven circular sensitivity estimates with respect to the analytical ones were
equal to (-2e-04, 0.0015) and (-0.006, 0.006), respectively.


In addition, we varied the grid epsilon values and computed both the corresponding
analytical and inla-driven sensitivities. Table 3 shows good agreement of exact and
inla-driven sensitivity estimates for the small sample at hand. Besides, we assumed
in the example a larger sample size _n_ = 100 instead of _n_ = 4 keeping all other model
parameters fixed. We expected less sensitivity to the prior parameter values when much
evidence is provided by the data. Again exact and inla-driven sensitivity estimates
agreed very well. Lower estimates of sensitivity obtained for _n_ = 100 in Table 3 indicated
that our measure reacts reasonably to increased sample size. Interestingly, even if the
prior is clearly displaced but there is enough evidence in the data ( _n_ = 100), sensitivity
can be small.


Table 3: Worst-case sensitivity estimates for mean as a function of _ϵ_ for _n_ = 4 and
_n_ = 100 in the example in Section 3 and Appendix C5 at _**γ**_ 0 = ( _μ_ 0 _, λ_ 0) = (70 _,_ 0 _._ 5).


_ϵ_ S [exact] _**γ**_ 0 ( _ϵ_ ) _, n_ = 4 S [inla] _**γ**_ 0 [(] _[ϵ]_ [)] _[, n]_ [ = 4] S [exact] _**γ**_ 0 ( _ϵ_ ) _, n_ = 100 S [inla] _**γ**_ 0 [(] _[ϵ]_ [)] _[, n]_ [ = 100]


0.0001 1.81 1.81 0.43 0.43
0.0005 1.81 1.82 0.43 0.43
0.0010 1.82 1.82 0.43 0.43
0.0050 1.83 1.83 0.43 0.43
0.0100 1.84 1.84 0.44 0.44


346 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


For several additional examples discussed in the Supplementary Material we once
again detected an excellent agreement of the inla-driven and exact sensitivity estimates.
As expected, a strong influence of the sample size on the prior sensitivity estimates
emerged. Indeed, we observed that our measure automatically adjusts for increasing
sample size by returning smaller prior sensitivity estimates.

## **Supplementary Material**


Supplementary Material: Sensitivity Analysis for Bayesian Hierarchical Models (DOI:
[10.1214/14-BA909SUPP;](http://dx.doi.org/10.1214/14-BA909SUPP) .pdf).

## **References**


Amari, S. (1990). _Differential-Geometrical_ _Methods_ _in_ _Statistics._ _2nd_ _Edition._ _Lecture_
_Notes_ _in_ _Statistics_, volume 28. Springer-Verlag. 325, 326


Amari, S. and Nagaoka, H. (2000). _Methods_ _of_ _Information_ _Geometry_ . Oxford University Press. 325


Berger, J. O., R´ıos Insua, D., and Ruggeri, F. (2000). “Bayesian Robustness.” In R´ıos
Insua, D. and Ruggeri, F. (eds.), _Robust_ _Bayesian_ _Analysis_, 1–32. Springer-Verlag.
322, 323, 338


Besag, J., York, J., and Molli´e, A. (1991). “Bayesian image restoration, with two applications in spatial statistics.” _Annals_ _of_ _the_ _Institute_ _of_ _Statistical_ _Mathematics_,
43(1): 1–59. 334


Bhattacharyya, A. (1943). “On a measure of divergence between two statistical populations defined by their probability distributions.” _Bulletin_ _of_ _the_ _Calcutta_ _Mathe-_
_matical_ _Society_, 35: 99–109. 325


Box, G. (1980). “Sampling and Bayes’ inference in scientific modelling and robustness.”
_Journal_ _of_ _the_ _Royal_ _Statistical_ _Society,_ _Series_ _A_, 143(4): 383–430. 324, 328


Breslow, N. E. and Clayton, D. G. (1993). “Approximate inference in generalized linear
mixed models.” _Journal_ _of_ _the_ _American_ _Statistical_ _Association_, 88(421): 9–25. 334


Cacuci, D. G., Ionescu-Bujor, M., and Navon, I. M. (2005). _Sensitivity_ _and_ _Uncertainty_
_Analysis._ _Volume_ _II,_ _Applications_ _to_ _Large-Scale_ _Systems_ . Chapman & Hall. 339


Carlin, B. and Louis, T. (1998). _Bayes and Empirical Bayes Methods for Data Analysis_ .
Chapman & Hall/CRC. 322


Clarke, B. and Gustafson, P. (1998). “On the overall sensitivity of the posterior distribution to its inputs.” _Journal_ _of_ _Statistical_ _Planning_ _and_ _Inference_, 71(1-2): 137–150.
322, 327


Cook, R. (1986). “Assessment of local influence.” _Journal_ _of_ _the_ _Royal_ _Statistical_
_Society,_ _Series_ _B_, 48(2): 133–169. 322


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 347


Dawid, A. (1977). “Further comments on some comments on a paper by Bradley Efron.”
_The_ _Annals_ _of_ _Statistics_, 5(6): 1249. 326


- (1979). “Conditional independence in statistical theory.” _Journal_ _of_ _the_ _Royal_
_Statistical_ _Society._ _Series_ _B_ _(Methodological)_, 41(1): 1–31. 322


Dey, D. and Birmiwal, L. (1994). “Robust Bayesian analysis using divergence measures.”
_Statistics_ _&_ _Probability_ _Letters_, 20(4): 287–294. 325, 326


Eberly, L. and Carlin, B. (2000). “Identifiability and convergence issues for Markov
chain Monte Carlo fitting of spatial models.” _Statistics_ _in_ _Medicine_, 19(17-18): 2279–
2294. 336


Evans, M. and Moshonov, H. (2006). “Checking for prior-data conflict.” _Bayesian_
_Analysis_, 4(1): 893–914. 324


Fong, Y., Rue, H., and Wakefield, J. (2010). “Bayesian inference for generalized linear
mixed models.” _Biostatistics_, 11(3): 397–412. 335, 343


Geisser, S. (1992). “Bayesian perturbation diagnostics and robustness.” In Goel, P.
and Iyengar, N. (eds.), _Bayesian_ _Analysis_ _in_ _Statistics_ _and_ _Econometrics_, 289–301.
Springer-Verlag. 322


- (1993). _Predictive_ _Inference:_ _An_ _Introduction_ . Chapman & Hall, Inc. 339, 343


Gelfand, A. and Sahu, S. (1999). “Identifiability, improper priors, and Gibbs sampling
for Generalized Linear Models.” _Journal_ _of_ _the_ _American_ _Statistical_ _Association_,
94(445): 247–253. 322


Gelfand, A. and Smith, A. (1990). “Sampling-based approaches to calculating marginal
densities.” _Journal_ _of_ _the_ _American_ _Statistical_ _Association_, 85(410): 398–409. 338


Gelman, A., Carlin, J., Stern, H., and Rubin, D. (2004). _Bayesian_ _Data_ _Analysis._ _2nd_
_Edition_ . Chapman & Hall/CRC. 333


Gilks, W., Richardson, S., and Spiegelhalter, D. (1996). _Markov_ _Chain_ _Monte_ _Carlo_ .
Chapman & Hall. 323


Goutis, C. and Robert, C. (1998). “Model choice in generalised linear models:
A Bayesian approach via Kullback-Leibler projections.” _Biometrika_, 85(1): 29–37.
326, 340


Gustafson, P. (1996). “Local sensitivity of inferences to prior marginals.” _Journal_ _of_
_the_ _American_ _Statistical_ _Association_, 91(434): 774–781. 339


- (2000). “Local robustness in Bayesian analysis.” In R´ıos Insua, D. and Ruggeri, F.
(eds.), _Robust_ _Bayesian_ _Analysis_, 71–88. Springer-Verlag. 322, 325


Gustafson, P. and Wasserman, L. (1995). “Local sensitivity diagnostics for Bayesian
inference.” _The_ _Annals_ _of_ _Statistics_, 23(6): 2153–2167. 339


Ibrahim, J., Zhu, H., and Tang, N. (2011). “Bayesian local influence for survival models.”
_Lifetime_ _Data_ _Analysis_, 17(1): 43–70. 330


Jeffreys, H. (1961). _Theory_ _of_ _Probability_ . Oxford University Press. 325


348 _**Sensitivity**_ _**Analysis**_ _**for**_ _**Bayesian**_ _**Hierarchical**_ _**Models**_


Kadane, J. (1992). “Comments to: “Bayesian perturbation diagnostics and robustness”
by S. Geisser.” In Goel, P. and Iyengar, N. (eds.), _Bayesian_ _Analysis_ _in_ _Statistics_
_and_ _Econometrics_, 298–300. Springer-Verlag. 322


Kass, R., Tierney, L., and Kadane, J. (1989). “Approximate methods for assessing
influence and sensitivity in Bayesian analysis.” _Biometrika_, 76(4): 663–674. 339


Lavine, M. (1992). “Local predictive influence in Bayesian linear models with conjugate
priors.” _Communications in Statistics - Simulation and Computation_, 21(1): 269–283.
339


Le Cam, L. (1986). _Asymptotic Methods in Statistical Decision Theory_ . Springer-Verlag.

325


Lesaffre, E. and Lawson, A. (2012). _Bayesian_ _Biostatistics_ . John Wiley & Sons. 323,
338


Lindgren, F., Rue, H., and Lindstr¨om, J. (2011). “An explicit link between Gaussian
fields and Gaussian Markov random fields: the stochastic differential equation approach.” _Journal_ _of_ _the_ _Royal_ _Statistical_ _Society,_ _Series_ _B._, 73(4): 423–498. 343


Martins, T., Simpson, D., Lindgren, F., and Rue, H. (2013). “Bayesian computing with
INLA: new features.” _Computational_ _Statistics_ _&_ _Data_ _Analysis_, 67: 68–83. 343


McCulloch, R. (1989). “Local model influence.” _Journal_ _of_ _the_ _American_ _Statistical_
_Association_, 84(406): 473–478. 322, 326, 327, 328, 329


Millar, R. and Stewart, W. (2007). “Assessment of locally influential observations in
Bayesian models.” _Bayesian_ _Analysis_, 2(2): 365–384. 322


M¨uller, U. (2012). “Measuring prior sensitivity and prior informativeness in large
Bayesian models.” _Journal_ _of_ _Monetary_ _Economics_, 59(6): 581–597. 327


Narasimhan, B. (2005). “Lisp-Stat to Java to R.” _Journal_ _of_ _Statistical_ _Software_, 13(4):
1–10. 338


Oakley, J. and O’Hagan, A. (2004). “Probabilistic sensitivity analysis of complex models: a Bayesian approach.” _Journal_ _of_ _the_ _Royal_ _Statistical_ _Society,_ _Series_ _B_, 66(3):
751–769. 322


P´erez, C., Mart´ın, J., and Rufo, M. (2006). “MCMC-based local parametric sensitivity
estimation.” _Computational_ _Statistics_ _&_ _Data_ _Analysis_, 51(2): 823–835. 327


Pettit, L. (1990). “The conditional predictive ordinate for the normal distribution.”
_Journal_ _of_ _the_ _Royal_ _Statistical_ _Society,_ _Series_ _B_, 52(1): 175–184. 343


Plummer, M. (2001). “Local sensitivity in Bayesian graphical models.”
URL [http://www-ice.iarc.fr/~martyn/papers/sensitivity.ps](http://www-ice.iarc.fr/~martyn/papers/sensitivity.ps) 327


Rao, C. (1945). “Information and the accuracy attainable in the estimation of statistical
parameters.” _Bulletin_ _of_ _the_ _Calcutta_ _Mathematical_ _Society_, 37: 81–91. 326


R´ıos Insua, D., Ruggeri, F., and Mart´ın, J. (2000). “Bayesian Sensitivity Analysis.” In
Saltelli, A., Chan, K., and Scott, E. M. (eds.), _Sensitivity_ _Analysis_, 225–244. John
Wiley & Sons. 322


_Ma�lgorzata_ _Roos,_ _Thiago_ _G._ _Martins,_ _Leonhard_ _Held,_ _and_ _H˚avard_ _Rue_ 349


Robert, C. (1996). “Intrinsic losses.” _Theory_ _and_ _Decision_, 40(2): 191–214. 325


Roos, M. and Held, L. (2011). “Sensitivity analysis in Bayesian generalized linear mixed
models for binary data.” _Bayesian_ _Analysis_, 6(2): 259–278. 325, 326, 330


Rue, H., Martino, S., and Chopin, N. (2009). “Approximate Bayesian inference for
latent Gaussian models by using integrated nested Laplace approximations.” _Journal_
_of_ _the_ _Royal_ _Statistical_ _Society,_ _Series_ _B._, 71(2): 319–392. 343


Ruggeri, F. (2008). “Bayesian Robustness.” _Forum:_ _Robustness_ _Analysis,_ _In:_ _European_
_Working_ _Group_ _“Multiple_ _Criteria_ _Decision_ _Aiding”_, 3(17). 322, 323


Saltelli, A., Ratto, M., Andres, T., Campolongo, F., Cariboni, J., Gatelli, D., Saisana,
M., and Tarantola, S. (2008). _Global_ _Sensitivity_ _Analysis._ _The_ _Primer_ . John Wiley
& Sons. 322


Saltelli, A., Tarantola, S., Campolongo, F., and Ratto, M. (2004). _Sensitivity_ _Analysis_
_in_ _Practice._ _A_ _Guide_ _to_ _Assessing_ _Scientific_ _Models_ . John Wiley & Sons. 339


Sivaganesan, S. (2000). “Global and local robustness approaches: uses and limitations.”
In R´ıos Insua, D. and Ruggeri, F. (eds.), _Robust Bayesian Analysis_, 89–108. SpringerVerlag. 322


Tierney, L. and Kadane, J. (1986). “Accurate approximations for posterior moments
and marginal densities.” _Journal_ _of_ _the_ _American_ _Statistical_ _Association_, 81(393):
82–86. 333


Van der Linde, A. (2007). “Local influence on posterior distributions under multiplicative modes of perturbation.” _Bayesian_ _Analysis_, 2(2): 319–332. 339


Weiss, R. (1996). “An approach to Bayesian sensitivity analysis.” _Journal_ _of_ _the_ _Royal_
_Statistical_ _Society,_ _Series_ _B._, 58(4): 739–750. 339


Weiss, R. and Cook, R. (1992). “A graphical case statistic for assessing posterior influence.” _Biometrika_, 79(1): 51–55. 339


Zhu, H., Ibrahim, J., Lee, S., and Zhang, H. (2007). “Perturbation selection and influence measures in local influence analysis.” _The Annals of Statistics_, 35(6): 2565–2588.
339


Zhu, H., Ibrahim, J., and Tang, N. (2011). “Bayesian influence analysis: a geometric
approach.” _Biometrika_, 98(2): 307–323. 322, 327, 328, 330, 332


**Acknowledgments**


The authors are grateful to Finn Lindgren, Daniel Simpson, Daniel Saban´es Bov´e and Andrea
Riebler for enlightening discussions, to Diego Morosoli for acquiring references and to Isaac
Gravestock for looking over the final version of the manuscript. They thank two Editors and
two anonymous reviewers for their valuable comments and suggestions on an earlier version of
the manuscript.


