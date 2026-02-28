Psychological Methods


© 2020 American Psychological Association 2020, Vol. 25, No. 6, 787–801
ISSN: 1082-989X [http://dx.doi.org/10.1037/met0000265](http://dx.doi.org/10.1037/met0000265)


Partitioning Variation in Multilevel Models for Count Data



George Leckie, William J. Browne,
and Harvey Goldstein
University of Bristol



Juan Merlo
Lund University



Peter C. Austin
University of Toronto


_**Abstract**_
A first step when fitting multilevel models to continuous responses is to explore the degree of clustering
in the data. Researchers fit variance-component models and then report the proportion of variation in the
response that is due to systematic differences between clusters. Equally they report the response
correlation between units within a cluster. These statistics are popularly referred to as variance partition
coefficients (VPCs) and intraclass correlation coefficients (ICCs). When fitting multilevel models
to categorical (binary, ordinal, or nominal) and count responses, these statistics prove more challenging
to calculate. For categorical response models, researchers appeal to their latent response formulations and
report VPCs/ICCs in terms of latent continuous responses envisaged to underly the observed categorical
responses. For standard count response models, however, there are no corresponding latent response
formulations. More generally, there is a paucity of guidance on how to partition the variation. As a result,
applied researchers are likely to avoid or inadequately report and discuss the substantive importance of
clustering and cluster effects in their studies. A recent article drew attention to a little-known exact
algebraic expression for the VPC/ICC for the special case of the two-level random-intercept Poisson
model. In this article, we make a substantial new contribution. First, we derive exact VPC/ICC
expressions for more flexible negative binomial models that allows for overdispersion, a phenomenon
which often occurs in practice. Then we derive exact VPC/ICC expressions for three-level and randomcoefficient extensions to these models. We illustrate our work with an application to student absenteeism.


_**Translational**_ _**Abstract**_
Multilevel models (random effects, mixed-effects or hierarchical linear models) are now a standard
generalization of conventional regression models for analyzing clustered and longitudinal data in the
social, psychological, behavioral, and medical sciences. A natural first step in any multilevel analysis is
to report the degree of clustering in the response because it is the assumed presence of clustering which
is the fundamental motivation for fitting multilevel models. Confirming that there is a statistically
significant degree of clustering is not enough. One must additionally communicate the practical
importance of clustering and this is done by reporting variance partition coefficients (VPCs) and
intraclass correlation coefficients (ICCs). Different VPC and ICC expressions have been proposed and
are now widely used for multilevel continuous, binary, ordinal, and nominal response models. In contrast,
there has been almost no work for multilevel count response models. In this article, we propose new VPC
and ICC expressions for both Poisson and negative binomial count response models. We then derive
VPC/ICC expressions for three-level and random-coefficient extensions to these models. We illustrate all
our work with an application to student absenteeism.


_Keywords:_ multilevel Poisson model, multilevel negative binomial model, count data, variance partition
coefficient, intraclass correlation coefficient


_Supplemental_ _materials:_ http://dx.doi.org/10.1037/met0000265.supp



This article was published Online First April 20, 2020.
X [George](https://orcid.org/0000-0003-1486-745X) Leckie, X William J. Browne, and X Harvey Goldstein,
Centre for Multilevel Modelling, School of Education, University of
Bristol; X [Juan](https://orcid.org/0000-0001-8379-9708) Merlo, Unit for Social Epidemiology, Faculty of
Medicine, Lund University; Peter C. Austin, Institute of Health Management, Policy and Evaluation, University of Toronto.
This work was presented in April 2019 at the 12th International
Multilevel Conference in Utrecht, the Netherlands [(https://multilevel](https://multilevel.fss.uu.nl/)
[.fss.uu.nl/).](https://multilevel.fss.uu.nl/) An earlier version of this article is available on arXiv
[(https://arxiv.org/abs/1911.06888).](https://arxiv.org/abs/1911.06888) This research was funded by



United Kingdom Economic and Social Research Council Grant ES/
R010285/1. This work was produced using statistical data from
Office for National Statistics (ONS), United Kingdom. The use of the
ONS statistical data in this work does not imply the endorsement of
the ONS in relation to the interpretation or analysis of the statistical
data.
Correspondence concerning this article should be addressed to George
Leckie, Centre for Multilevel Modelling, School of Education, University
of Bristol, 35 Berkeley Square, Bristol BS8 1JA, United Kingdom. E-mail:
[g.leckie@bristol.ac.uk](mailto:g.leckie@bristol.ac.uk)



787


788 LECKIE, BROWNE, GOLDSTEIN, MERLO, AND AUSTIN



Multilevel models (mixed-effects, random effects, or hierarchical linear models) are now a standard generalization of conventional regression models for analyzing clustered and longitudinal
data in the social, psychological, behavioral, and medical sciences.
Examples include students within schools, respondents within
neighborhoods, patients with hospitals, repeated measures within
subjects, and panel survey waves on households. Multilevel models have been further generalized to handle a wide range of
response types, including, continuous, categorical (binary or dichotomous, ordinal, and nominal or discrete choice), count, and
survival responses. Standard introductions to these models can be
found in textbooks by Goldstein (2011b); Hox, Moerbeek, and van
de Schoot (2017); Raudenbush and Bryk (2002), and Snijders and
Bosker (2012).
A natural first step in any multilevel analysis is to report the
degree of clustering in the response because it is the assumed
presence of clustering which is the fundamental motivation for
fitting multilevel models. Confirming that there is a statistically
significant degree of clustering is not enough. One must additionally communicate the practical importance of clustering and this is
done by reporting variance partition coefficients (VPCs; Goldstein,
Browne, & Rasbash, 2002) and intraclass correlation coefficients
(ICCs). Consider a study of the relationships between a continuous
student math score _yij_ for student _i_ ( _i_ - 1, . . ., _nj_ ) in school _j_ ( _j_ 1, . . ., _J_ ) and a range of student and school characteristics **x** _ij_ .
Suppose we simply fit a linear regression model _yij_ - **x** _ij′_ - - _eij_ .
Here the concern is student math scores will correlate within schools
(within-cluster dependence), even after adjusting for the covariates
Corr� _yij_, _yi_ - _j_ - **x** _ij_, **x** _i_ - _j_ - - 0, thereby violating the linear regression
assumption of independent residuals Corr� _eij_, _ei_ - _j_ - **x** _ij_, **x** _i_ - _j_ - - 0. Such
clustering would typically be due to unmodelled student and school
influences on math scores that vary between schools (between-cluster
heterogeneity). A two-level linear regression _yij_ - **x** _ij′_ - - _uj_ - _eij_
attempts to account for these influences, and therefore clustering,
by including a school random-intercept effect _uj_ . The total residual
variance is then decomposed into separate within- and betweenschool components � _e_ [2] and � _u_ [2] . The proportion of response variance
which lies between schools (conditional on any covariates) can
then be estimated and reported using the VPC. This statistic is
calculated as the ratio of the estimated between-school variance to
the total residual variance, - _u_ [2] /(� _u_ [2] �� _e_ [2] ). The VPC therefore
summarizes the “importance” of the clusters in influencing the
response, in our case the importance of schools in influencing
student outcomes above and beyond the modeled student and
school characteristics. As such, the VPC is widely quoted in
multilevel studies. In the case of the current random-intercept
model, the expression for the VPC is also the expression for the
expected correlation between two students from the same school
(conditional on any covariates), and when this interpretation is
favored the expression is referred to as the ICC. As a result of their
common expression, the two terms and interpretations are used
interchangeably. This is poor practice, especially as the VPC and
ICC expressions diverge as soon as random coefficients are added
to the model (Goldstein et al., 2002). It is therefore prudent to
always apply the correct interpretation to each statistic. In this
article, we shall focus primarily on expressions for the VPC and
therefore the VPC interpretation. For readers seeking further details on the VPC and ICC for two-level variance-components,



random-intercept, and random-coefficient continuous response
models, this can be found in [Supplemental](http://dx.doi.org/10.1037/met0000265.supp) Materials S1.
VPCs (and ICCs) can also be calculated in three- and higherlevel models for continuous responses as well as in models with
more complex cross-classified, multiple membership, spatial, and
dyadic random effect structures (Leckie, 2013a, 2013b, 2013c). In
these settings, VPCs are often used to ascertain the relative importance of different sources of clustering in influencing the response, for example, in establishing the relative importance of
family, school, and area effects on student attainment (Rasbash,
Leckie, Pillinger, & Jenkins, 2010). VPCs can also be calculated
after fitting models with random coefficients. In all these cases, the
VPC expressions become more complex and do not necessarily
continue to have parallel ICC interpretations, but these issues are
well described in the literature (see, for example, Snijders &
Bosker, 2012).
For multilevel models for categorical responses (e.g., binary,
ordinal, and nominal logistic regression), calculating VPCs (and
ICCs) is less straightforward as these statistics typically no longer
have closed-form expressions (they involve integrating out the
random effects which can only be achieved via numerical integration). The standard approach to this problem is to appeal to the
latent-response formulations of these models and to report VPCs in
terms of unobserved continuous variables envisaged to underlie
the observed categorical responses (see, for example, Snijders &
Bosker, 2012). For example, in a study of whether students pass an
exam, we would appeal to the notion of a continuous exam score
scale underlying the observed binary pass or fail status and we
would report the VPC in terms of this latent variable, that is, in
terms of the propensity for the student to pass the exam. An
appealing feature of this approach is that it allows one to calculate
VPCs for categorical responses using essentially the same expressions as those derived for continuous responses.
For multilevel models for count responses (e.g., Poisson and
negative binomial models), however, there are no corresponding
latent-response formulations (Skrondal & Rabe-Hesketh, 2004)
and so it is less obvious how one should calculate the VPC (or
ICC). More generally there is a paucity of guidance on how to
partition response variation when modeling counts. As a result,
applied researchers are likely to avoid or inadequately report and
discuss the substantive importance of clustering and cluster effects
in their studies.
A notable exception is the work by Stryhn, Sanchez, Morley,
Booker, and Dohoo (2006) and later Austin, Stryhn, Leckie, and
Merlo (2018) who show that the VPC (and ICC) for the special
case of a two-level random-intercept Poisson model does have a
closed-form and so can be calculated exactly with a simple algebraic expression. However, for many applications, researchers now
routinely apply multilevel negative binomial models to account for
overdispersion (the phenomenon whereby the variance of the
observed counts is larger than that implied by the expectation) and
so exact algebraic expressions for the VPC are also needed for
these more flexible models. Likewise, many researchers now routinely fit count models allowing for three or more levels or random
coefficients and so VPC expressions must also be extended to
account for these modeling extensions.
Given the lack of VPC (and ICC) expressions for count response
models, the only approach available to researchers is the simulation method proposed by Goldstein, Browne, and Rasbash (2002)


PARTITIONING VARIATION IN MULTILEVEL COUNT MODELS 789



and further illustrated by Browne, Subramanian, Jones, and Goldstein (2005); Leckie and Goldstein (2015), and Leckie, Pillinger,
Jones, and Goldstein (2012). While these articles discuss this
method in the context of two-, three-, and four-level binary response models, the method readily extends to the count models
discussed here. Indeed, Austin et al. (2018) used this simulation
method to confirm that the exact algebraic expression introduced
there for the VPC for the special case of the two-level randomintercept Poisson model gives the correct value. The simulation
method involves using the fitted model to simulate new count
responses and then calculating the within- and between-cluster
variances for these simulated data. The VPC can then be calculated
in the usual way. The principal disadvantage of using the simulation method to calculate the VPC is that it has not been implemented in software, forcing researchers to write their own code
which is error prone, especially for models with complex random
effect structures, and it is perhaps for these reasons why one sees
so few applications of the method in practice.
In this article, we propose a simpler approach. We derive exact
algebraic expressions for the VPC and ICC for the two-level
random-intercept Poisson and negative binomial model (mean
dispersion or NB2 version) and the three-level and randomcoefficient extensions of these models. We focus on these two
models as they are most widely discussed (see, for example,
Snijders & Bosker, 2012). However, we note that other multilevel
models for count responses are increasingly being implemented in
standard software (R, SAS, SPSS, Stata). We therefore additionally review two of these models and derive their VPC and ICC
expressions [(Supplemental](http://dx.doi.org/10.1037/met0000265.supp) Materials S3): the Poisson model with
an overdispersion random effect and the constant dispersion or
NB1 version of the negative binomial model. We motivate and
illustrate each methodological development with an application to
student absenteeism. We confirm our derivations are correct by
showing they give the same VPC and ICC estimates for our
[application as those based on the simulation method (Supplemen-](http://dx.doi.org/10.1037/met0000265.supp)
[tal Materials S8). We share the Stata and R code and output for our](http://dx.doi.org/10.1037/met0000265.supp)
application in order to assist readers in applying our VPC and ICC
expressions in their own work (Supplemental Materials S9 and
[S10).](http://dx.doi.org/10.1037/met0000265.supp)



**Application:** **Student** **Absenteeism** **From** **School**


Student absenteeism and its detrimental effects on student learning
are ongoing concerns in the U.S. (Economic Policy Institute, 2018),
United Kingdom (Department for Education, 2019a), and many other
countries. In response, school accountability systems are increasingly
monitoring student absenteeism rates alongside more traditional attainment and progress measures (Leckie & Goldstein, 2017). Student
absenteeism is known to vary by student demographic and socioeconomic characteristics, factors which also vary across school intakes. It
would therefore seem important to adjust student absenteeism rates
for school differences in student composition before making any
attempt to hold schools accountable for their performance (Prior,
Goldstein, & Leckie, 2019). In this application, we explore these ideas
using multilevel models for count data. These models and notion of
adjusting student outcomes for student characteristics when comparing schools are analogous to those used to estimate school valueadded effects on student attainment (Castellano & Ho, 2013; Goldstein, 1997; Leckie & Goldstein, 2009, 2019; OECD, 2008;
Raudenbush & Willms, 1995).
Our study relates to students in London schools who completed
their compulsory secondary schooling at the end of the 2016/17
academic year (students aged 15/16 in United Kingdom school year
11; equivalent to U.S. 10th grade). The study was granted ethical
approval by the Ethics Committee at the Faculty of Social Sciences
and Law, University of Bristol. The data are drawn from the national
pupil database, a census of all students in state-maintained schools in
England (Department for Education, 2019b). The data are a threelevel hierarchy consisting of 66,955 students (Level 1) nested in 434
schools (Level 2) nested in 32 school districts (Level 3). The response
is a count of the number of days students were absent from school
during the academic year. Figure 1 presents the frequency distribution. The distribution is positively skewed, with students, on average,
absent for 8.41 days over the academic year, but with an interquartile
range spanning from 2 to 10 days (variance 124.39; min 0; max 156).
Figure 2 illustrates variation in the mean number of days by district
(left: _M_ - 8.36; IQR � 7.56, 9.10) and by school (right: _M_ - 8.46;
IQR � 6.56, 9.76). We see meaningful differences in student absence
rates, especially between schools.



_Figure_ _1._ Distribution of number of days absent over the school year.


790 LECKIE, BROWNE, GOLDSTEIN, MERLO, AND AUSTIN


_Figure_ _2._ Mean number of days absent by district (left panel) and school (right panel). The horizontal line
depicts the student sample mean.



We will start by fitting “empty” or “null” two- and three-level
random-intercept Poisson and negative binomial models with no
covariates. In the continuous response case, these models are referred
to as variance-component models and we use that terminology here.
The aim of these initial analyses is to quantify the degree of clustering
and overdispersion in the data and we do this using our VPC (and
ICC) expressions. Thus, we will use the VPC expressions to summarize the importance of district and school differences in student
absenteeism rates in explaining why some students miss more school
days than others. We will then pick a preferred model and extend it by
entering student characteristics as covariates. Here our aim is to not
just study the predictors of student absenteeism, but to adjust for these
factors so that the resulting predicted school random effects provide
more meaningful estimates of school influences on students that
are plausibly related to factors within schools’ control. At this point
we will recalculate the VPCs and explore how they now vary as a
function of the covariates and we investigate the predicted school
effects. Last, we shall fit a random-coefficients model, in which we
allow the effect of one of the student covariates to vary across schools,
and we will explore the implications this has for calculating and
interpreting the VPC.
We fit all models using maximum likelihood estimation (via adaptive quadrature) as implemented in the “mepoisson” and “menbreg”
commands in Stata (StataCorp, 2019). See Rabe-Hesketh and Skrondal (2012, Chapter 13) for an excellent introduction to multilevel
models for count responses using Stata. These models can equally be
fitted by maximum likelihood estimation in other standard software
(R, SAS, SPSS) or by Markov chain Monte Carlo (MCMC) methods.


**Two-Level** **Variance-Components** **Poisson** **Model** **for**
**Count** **Responses**


In this section, we review the two-level variance-components Poisson model and then present expressions for the VPC and ICC. We
delay the introduction of covariates to later. This model and VPC and
ICC expressions were also the focus of Austin et al. (2018), but here
we explore in more detail the assumptions the model makes regarding
the observed counts in order to help readers better understand the
expressions and their interpretations. These steps will then help the
reader when we explore more complex count models which are



the primary focus of this article and which were not considered by
Austin et al. (2018). For readers seeking a refresher on the conventional (single-level) Poisson model (and other count models), we
recommend the textbook by Long (1997).
The two-level variance-components Poisson model, like all multilevel models, is a conditional (cluster-specific) model as opposed to a
marginal (population-averaged) model: Estimation and inference are
based on conditioning on the values of the random effects as well as
the covariates as opposed to conditioning only on the covariates and
averaging over the population of clusters. The model is therefore
described in terms of the conditional expectation and variance of the
response. Importantly, conditional and marginal models for count
responses make different inferences. This contrasts conditional and
marginal models for continuous responses where inferences coincide.
Our view is that in applied research it is usually informative to explore
and contrast both approaches. Indeed, as we shall go on to discuss, the
VPC and ICC are marginal statistics derived from the marginal
expectation, variance, covariance, and correlation of the responses. It
turns out, one can recover these marginal statistics from the conditional model, but these statistics are rarely presented in textbook and
other discussions of the model (Supplemental Materials S2). We
therefore review each marginal statistic below prior to presenting our
expressions for the VPC and ICC. We hope that this treatment
provides a further useful resource for readers. We provide full deri[vations in Supplemental Materials S4.](http://dx.doi.org/10.1037/met0000265.supp)


**Model**


Let _yij_ denote the count for unit _i_ ( _i_  - 1, . . ., _nj_ ) in cluster _j_ ( _j_  1, . . ., _J_ ). In terms of our application, the units will be students,
the clusters schools, and the count will be the number of days each
student is absent from school over the course of the school year.
We can then write the two-level variance-component Poisson
model for _yij_ as follows


_yij_ | � _ij_                - Poisson(� _ij_ )


ln(� _ij_ ) ��0 � _uj_ (1)


_uj_               - _N_ �0, �2 _u_               

PARTITIONING VARIATION IN MULTILEVEL COUNT MODELS 791



where � _ij_ denotes the expected count, �0 denotes the intercept, and
_uj_ is the cluster random intercept effect, assumed normally distributed with zero mean and variance - _u_ [2] .


**Conditional** **Statistics**


The conditional expectation of _yij_ (given _uj_ ) is given by

      - _ijC_      - E( _yij_ | _uj_ ) � exp(�0 � _uj_ ) (2)


which in this model is simply equal to - _ij_ in Equation 1, but this
will not be the case in the next model, hence the introduction of the
“C” superscript.
The conditional variance is given by

       - _ijC_       - Var( _yij_ | _uj_ ) �� _ijC_ (3)


Thus, the conditional variance of the counts is assumed to equal
the conditional expectation. In practice, this equidispersion assumption often fails, with the variance of the observed counts in
many clusters being larger or smaller than that implied by the
cluster mean, phenomena known as overdispersion (extra-Poisson
variability) or underdispersion, respectively. Overdispersion is far
more common than underdispersion and is typically attributed to
unobserved unit-level covariates. In terms of our application, such
covariates might include student prior attainment, demographics,
or socioeconomic status and we will introduce these into the
analysis later.
The conditional covariance and correlation between the response measurements on two units _i_ and _i_ ’ from the same cluster
_j_, _yij_ and _yi_ - _j_ are zero and so the observed counts are assumed
conditionally independent (sometimes referred to as the local
independence assumption). The conditional covariance and correlation between response measurements on two units from two
different clusters are also assumed equal to zero.


**Marginal** **Statistics**


The marginal expectation of _yij_ (now averaged over _uj_ ) is given
by

      - _ijM_      - E( _yij_ ) � exp��0 ��2 _u_ ⁄ 2� (4)


The marginal variance of _yij_ is given by

    - _ijM_    - Var( _yij_ ) �� _ijM_    - �� _ijM_ �2�exp��2 _u_    -    - 1� (5)


The marginal variance is therefore a quadratic function of the
marginal expectation and is larger than the marginal expectation if
there is clustering, - _u_ [2] - 0.
The marginal covariance of _yij_ and _yi_  - _j_ is given by


Cov� _yij_, _yi_       - _j_       -       - �� _ijM_ �2�exp��2 _u_       -       - 1� (6)


and the associated correlation can then be calculated in the usual
way to give

ICC _ij_, _i_   - _j_   - Corr( _yij_, _yi_   - _j_ ) �   - _ijM_ ��� _ijM_ ���2 _ijM_ �exp�2�exp��2 _u_ ����2 _u_ �1�� 1� (7)


This marginal correlation can be interpreted as the ICC as it is
the response correlation between two units in the same cluster. In
contrast, the marginal covariance and correlation between response



measurements on two units from two different clusters is equal to
zero as the units do not share a cluster random effect.
The VPC is defined as the proportion of the marginal response
variance which lies between clusters. Thus, we can only calculate
the VPC after we have partitioned the marginal variance into
level-specific components. The expression for the marginal variance (Equation 5) does just this (Supplemental [Materials](http://dx.doi.org/10.1037/met0000265.supp) S4.2).
Specifically, the first term � _ij_ _[M]_ captures the average variance within
clusters in the observed unit-level counts _yij_ around the expected
counts � _ij_ _[C]_ (Equation 3; the unit-level or Level-1 component), while
the second term (� _ij_ _[M]_ ) [2] {exp(� _u_ [2] ) - 1} captures the variance between clusters in their expected counts - _ij_ _[C]_ attributable to the
cluster random intercept effect _uj_ (the cluster-level or Level-2
component). The expression for the VPC can then be derived in the
usual way: as the ratio of the Level-2 component of the marginal
variance divided by the summation of the Level-2 and -1 components to give


level-2 variance
È
VPC _ij_    - �Ç� _ijM_    - [2] �exp�� _ijM_ ��� [2] 2 _u_ ��exp� �1�� 2 _u_ ��    - 1�Ç� _ijM_ (8)

level-2 variance level-1 variance


and this expression is identical to that for the marginal correlation
or ICC given in Equation 7. This expression, but where exp(�0 �

- _u_ [2] /2) is substituted in for � _ij_ _[M]_, was published in Austin et al. (2018).
Studying Equation 8, we see that the VPC is an increasing function
of both the marginal expectation - _ij_ _[M]_ and the cluster variance - _u_ [2] .
This VPC is strictly the Level-2 VPC. The Level-1 VPC—the
proportion of the marginal response variance which lies within
clusters—is then simply equal to one minus the Level-2 VPC.


**Application** **Continued**


Model 1 is a two-level variance-components Poisson model
(Equation 1). The model includes a school random intercept to
investigate and account for potential school clustering. The
estimated intercept is 2.085. The model estimates the school
variance to be 0.100 and a likelihood ratio test confirms that
this between school variation in student absence rates is statistically significant (Model 1 vs. a conventional single-level
Poisson model, but with no covariates; results not shown; �1 [2] 53194, _p_ - .001). What is less clear is the practical or substantive importance of these estimates. In particular, is a value of
0.100 for the school variance big? Does school clustering
matter? Do we care? We can answer these questions using the
VPC, ICC, and other estimated marginal statistics.
The parameter estimates imply a marginal expectation of
8.46 (application of Equation 4) and a marginal variance of
15.98 (application of Equation 5). The marginal expectation
approximately equals the sample mean number of days (8.41),
but the marginal variance is far below the sample variance of
124.39 and so this model proves inadequate for these data. To
understand this last point, it proves revealing to decompose the
estimated marginal variance into level-specific components.
The school component equals 7.52 while the student component
equals 8.46. The resulting school VPC equals 0.47 and so a very
high 47% of the marginal variance is due to systematic differ

792 LECKIE, BROWNE, GOLDSTEIN, MERLO, AND AUSTIN



ences between schools (application of Equation 8). This suggests that school level factors, or at least school-level variation
in student characteristics, account for almost half the modeled
variation in student absenteeism between students. The ICC in
this model (application of Equation 7) equals the VPC: The
response correlation between two students in the same school is
very high at 0.47. Finally, note that the student component of
the marginal variance appears identical to the marginal expectation, both are estimated as 8.46. This is indeed the case; in this
model these two terms are equal by definition (see the last term
in the denominator of Equation 8). In the next section, we will
explore the negative binomial model which relaxes this constraint and therefore allows the data to be more dispersed than
that implied by the mean and so will likely provide a better fit
to the data and a more reasonable estimate of the marginal
variance.
We note that when desired, interval estimates can also be
calculated for the VPCs (and ICCs; Goldstein et al., 2002) and
for that matter each of the other marginal statistics. When
models are fitted by maximum likelihood estimation, a 95%
confidence interval for each statistic can be constructed via the
delta method (the VPC is a nonlinear combination of the model
parameters) or via multilevel bootstrapping (e.g., fitting the
model to 1,000 bootstrapped samples to obtain a sampling
distribution for the VPC; Goldstein, 2011a). When the model is
fitted by MCMC methods, a 95% credible interval can be
calculated using the MCMC chain for the posterior distribution
of the VPC.


**Two-Level** **Variance-Component** **Negative** **Binomial**
**Model** **for** **Count** **Responses**


In this section, we shift our focus to the more flexible
two-level variance-component negative binomial model (mean
dispersion or NB2 version) which allows for overdispersion.
This work now moves beyond that presented in Austin et al.
(2018). In particular, the expression we present for the VPC and
ICC is an important new result. See [Supplemental](http://dx.doi.org/10.1037/met0000265.supp) Materials
[S3.2](http://dx.doi.org/10.1037/met0000265.supp) for a parallel presentation of the constant dispersion or
NB1 version of this model and associated VPC and ICC expressions. We provide full derivations of all marginal statistics
in Supplemental Materials S4.


**Model**


The negative binomial model (mean dispersion or NB2 version)
is an extension of the Poisson model that adds a normally distributed unit-level overdispersion random effect to represent omitted
unit-level variables that are envisaged to be driving any overdispersion. In contrast to the conventional cluster random intercept
effect, this does not induce any dependence among the units. The
model can be written as


_yij_ | � _ij_                - Poisson(� _ij_ )

ln(� _ij_ ) ��0 � _uj_           - _eij_ (9)

_uj_               - _N_ �0, �2 _u_               
exp( _eij_ ) � Gamma�� [1] [,][ �][�]



where _eij_ denotes the overdispersion random effect. Thus, in this
model, two units with the same random intercept effect value may
nonetheless differ in their expected counts - _ij_, with such differences attributed to the two units differing in terms of their values
on the omitted unit-level variables. The exponentiated overdispersion random effect exp( _eij_ ) is assumed gamma distributed with
shape and scale parameters 1/� and - and is therefore distributed
with mean 1 and variance or overdispersion parameter �. The
larger � is, the greater the overdispersion. When �� 0, the model
simplifies to the Poisson model (Equation 1) and so we can
conduct a likelihood ratio test to compare the two models to see
whether the estimated overdispersion is statistically significant.


**Conditional** **Statistics**


In this model, we can again calculate the conditional expectation
and variance of the response. However, here we must first integrate out the overdispersion random effect because this is not
typically of substantive interest. The conditional expectation of _yij_
(given _uj_ but averaged over _eij_ ) has the same form as in the Poisson
model (Equation 2), with

      - _ijC_      - E( _yij_ | _uj_ ) � exp(�0 � _uj_ ) (10)


and we see that, in contrast to the Poisson model, - _ij_ _[C]_ �� _ij_ . The
conditional variance is then given by

     - _ijC_      - Var( _yij_ | _uj_ ) �� _ijC_      - �� _ijC_ �2� (11)


Thus, the conditional variance is now a quadratic function of the
conditional expectation and is larger than the conditional expectation if �� 0. Therefore, the usual variance-mean relationship for
the Poisson model (Equation 3) is relaxed, allowing overdispersion
with respect to the conditional expectation (� _ij_ _[C]_ �� _ij_ _[C]_ ).


**Marginal** **Statistics**


The marginal expectation of _yij_ (now averaged over _uj_ as well as
_eij_ ) is given by

      - _ijM_      - E( _yij_ ) � exp��0 ��2 _u_ ⁄ 2� (12)


which is the same as that for the Poisson model (Equation 4).
The marginal variance of _yij_ is given by

   - _ijM_   - Var( _yij_ ) �� _ijM_   - �� _ijM_ �2�exp��2 _u_ �(1 ��) � 1� (13)


which differs from that for the Poisson model (Equation 5) via the
inclusion of the additional multiplicative term (1 ��). Thus, in
this model, the marginal variance is larger than the marginal
expectation if there is clustering � _u_ [2] - 0 _or_ overdispersion �� 0.
The marginal covariance of _yij_ and _yi_  - _j_ (averaged over _uj_, _eij_ and
_ei_ - _j_ ) is given by


Cov( _yij_, _yi_       - _j_ ) � �� _ijM_ �2�exp��2 _u_       -       - 1� (14)


and is the same as that for the Poisson model (Equation 6). The
marginal correlation of _yij_ and _yi_ - _j_ is then given by

ICC _ij_, _i_  - _j_  - Corr( _yij_, _yi_  - _j_ ) �  - _ijM_  - ���� _ijMijM_ ��22��expexp����2 _u_ 2 _u_ ��(1� ��1� ) � 1�


(15)


PARTITIONING VARIATION IN MULTILEVEL COUNT MODELS 793



and as with the Poisson model can be interpreted as the ICC. The
expression differs from that for the Poisson model (Equation 7)
only in the inclusion of the additional multiplicative term (1 ��)
in the denominator.
As with the Poisson model, we can partition the marginal
variance (Equation 13) into level-specific components which capture the within- and between-cluster variance in _yij_ [(Supplemental](http://dx.doi.org/10.1037/met0000265.supp)
[Materials](http://dx.doi.org/10.1037/met0000265.supp) S4.2). The resulting Level-2 VPC is given by


level-2 variance
È
VPC _ij_ - �� _ijM_ - [2] �exp��2 _u_ ���� _ijM_ �1 [2] ��exp���� _ijM_ 2 _u_ ����1�� _ijM_ - [2] exp��2 _u_ �� (16)

Ç Ç
level-2 variance level-1 variance


and this expression is identical (after rearranging terms) to that for
the marginal correlation or ICC given in Equation 15.
Studying Equation 16, we see that, as in the Poisson case
(Equation 8), the VPC is an increasing function of both the
marginal expectation � _ij_ _[M]_ and the cluster variance � _u_ [2] . However, the
VPC is now also a decreasing function of the overdispersion
parameter �. This makes sense. As the overdispersion increases,
all else equal, the more unmodelled variation there is at Level 1
and so the VPC decreases.
Comparing the two VPC expressions (Equations 8 and 16), we
see that the expression for the Level-2 component of the marginal
variance is the same and so it is only the expression for the Level-1
component which varies across models. This makes sense as the
models differ only in their treatment of overdispersion, which is
viewed as a Level-1 phenomenon. The overdispersion parameter in
the negative binomial model leads the expression for the Level-1
component of the marginal variance to exceed that of the Poisson
model. The marginal variance is simply the summation of the
Level-2 and -1 variances and so is also expected to be higher in the
negative binomial model compared to that of the Poisson model.


**Application** **Continued**


Model 2 is a two-level variance-components negative binomial model (Equation 9). The model includes a student overdispersion random effect to account for any within-school variation due to omitted student influences. The overdispersion
parameter estimated to be 0.877 and a likelihood ratio test
confirms that there is significant overdispersion (Model 2 vs.
Model 1: �1 [2] - 363096, _p_ - .001). What is less clear is the
practical or substantive importance of this additional variation.
Put simply, is a value of 0.877 for the overdispersion parameter
big? Does overdispersion matter? Do we care? Here too, we can
answer these questions using the estimated VPC, ICC and other
marginal statistics.
Most importantly, the estimated marginal variance now increases from 15.98 to 84.10 (application of Equation 13). As
expected, the school component remains approximately stable
and so the increase in the marginal variance is brought about by
the student component which increases nine fold from 8.46 to
77.15 (it is no longer constrained to equal the marginal variance). This increase indicates that, even within schools, student
absenteeism is far from a random Poisson process, rather the
model suggests that there is substantial within school variability



driven by omitted student characteristics. This increase in the
student component in turn has a dramatic impact on the estimated VPC. The model now estimates the school VPC to be
0.08 (application of Equation 16), suggesting that it is in fact
omitted student-specific factors rather than omitted schoolspecific factors that are likely the dominant cause of the variation in absenteeism. This estimated VPC is far lower than the
estimate of 0.47 reported for Model 1. Thus, an important
finding is that by ignoring overdispersion, the Poisson model
grossly overestimates the relative importance of schools
in these data. More generally, the Poisson VPC is biased
upward in the presence of overdispersion.


**Three-Level** **Variance-Components** **Models** **for** **Count**
**Data** **and** **Calculation** **of** **VPCs**


In this section, we focus on three-level variance components
negative binomial model and restrict our presentation of the marginal statistics to only the VPC and ICC. Recall, however, that the
Poisson model is simply the special case of the negative binomial
model with no overdispersion (see previous sections). The parallel
VPC and ICC expressions for the Poisson model can therefore
be obtained by setting �� 0 in the expressions below. We
[provide full derivations of all marginal statistics in Supplemen-](http://dx.doi.org/10.1037/met0000265.supp)
tal [Materials](http://dx.doi.org/10.1037/met0000265.supp) S5.


**Model**


Let _yijk_ denote the count for unit _i_ ( _i_  - 1, . . ., _nj_ ) in cluster _j_ ( _j_  1, . . ., _Jk_ ) in supercluster _k_ ( _k_ - 1, . . ., _K_ ). In terms of our
application, the units are students, the clusters schools, and
the superclusters school districts. The three-level variancecomponents negative binomial model can then be written as


_yijk_ | � _ijk_               - Poisson(� _ijk_ )


ln(� _ijk_ ) ��0 � _vk_         - _ujk_         - _eijk_ (17)


_vk_               - _N_ �0, �2 _v_               

_ujk_                - _N_ �0, �2 _u_                

exp( _eijk_ ) � Gamma�� [1] [,][ �][�]


where _vk_ is the new supercluster random-intercept effect assumed
normally distributed with zero mean and variance � _v_ [2] and all other
terms are defined as before.


**Marginal** **Statistics**


With two higher levels there is now interest in reporting the
relative importance of both superclusters and clusters as separate
sources of response variation. As in the simpler two-level setting
we can decompose the marginal variance into level-specific components and we use these to construct different VPC statistics
(Supplemental [Materials](http://dx.doi.org/10.1037/met0000265.supp) S5.2).


794 LECKIE, BROWNE, GOLDSTEIN, MERLO, AND AUSTIN



The Level-3 VPC can then be written as


VPC(3) _ijk_


level-3 variance
È

 - �� _ijkM_  - [2] �exp��2 _v_  -  - 1�

�� _ijkM_   - [2] �exp��2 _v_   -   - 1�   - �� _ijkM_   - [2] exp��2 _v_ ��exp��2 _u_   -   - 1�
###### Ç Ç

level-3 variance level-2 variance

�� _ijkM_  - �� _ijkM_  - [2] exp��2 _v_ ��2 _u_ ��
###### Çlevel-1 variance (18)


and is interpreted as the proportion of response variance which lies
between superclusters. This expression can also be interpreted as
an ICC as it also gives the response correlation between two units
in the same supercluster, but different clusters.
The Level-2 VPC can be written as


VPC(2) _ijk_


level-2 variance
###### È

 - �� _ijkM_ �2�exp���2 _v_ �� _ijkM_ ��12exp� ���2 _v_ ����exp _ijkM_ �2�exp�2 _u_ ����2 _v_ �1��exp��2 _u_  -  - 1�
### Ç Ç

level-3 variance level-2 variance

  -  - _ijkM_  - �� _ijkM_ �2exp��2 _v_ ��2 _u_ ��
### Çlevel-1 variance (19)


and is interpreted as the proportion of response variance which lies
within superclusters, between clusters. This VPC does not have a
corresponding ICC interpretation. This can be seen by realizing
that the implied correlation would be between two units in different superclusters, but the same cluster and this is not a possibility
in hierarchical data.
We can also calculate the proportion of response variance collectively attributable to superclusters and clusters. This VPC is
calculated by replacing the numerator in the previous equations
with the sum of the Level-3 and Level-2 variance components of
the marginal variance.


VPC(2, 3) _ijk_



level-3 variance
È

- [�][�] _[ijk]_ _M_ �2�exp��2 _v_ - - 1�



level-2 variance
###### È

- �� _ijkM_ �2exp��2 _v_ ��exp��2 _u_ - - 1�



�� _ijkM_ �2�exp��2 _v_ - - 1� ��� _ijkM_ �2exp��2 _v_ ��exp��2 _u_ - - 1�
#### Ç Ç

level-3 variance level-2 variance



�� _ijkM_  - �� _ijkM_ �2exp��2 _v_ ��2 _u_ ��
##### Çlevel-1 variance (20)


This expression once again has an ICC interpretation, namely,
the response correlation between two units in the same supercluster and the same cluster. This correlation will be stronger than that
between two units in the same supercluster but different clusters
(Equation 18) as here the units share not only unobserved supercluster influences (captured by the Level-3 component in the
numerator), but now additionally share unobserved cluster influences (captured by the Level-2 component in the numerator). The
Level-1 VPC—the proportion of the marginal response variance
which lies within clusters—is simply equal to one minus this joint
Level-3 and Level-2 VPC.



**Application** **Continued**


Model 3 is a three-level variance-components negative binomial
model (Equation 17). The model includes a district random intercept effect to investigate and account for potential superclustering
by district. The estimated intercept is effectively unchanged from
that of Model 2 the two-level negative binomial model. The model
estimates the district variance to be just 0.006, but a likelihood
ratio test confirms that this district variation in student absence
rates is nonetheless statistically significant (Model 3 vs. Model 2:
�1 [2] - 7.29, _p_ - .001). The school variance in turn decreases by
0.006 from 0.093 to 0.087. The overdispersion parameter is also
unchanged. This is expected as overdispersion is treated as a
Level-1 phenomenon and so is unaffected by whether the school
variation is decomposed into separate within and between district
components, as we have done here, or not. Here, we immediately
see that districts are of little practical or substantive importance
when studying student absenteeism as the estimated variance is
only around a tenth of the magnitude of the school variance. We
can confirm this using the estimated VPC statistics for three-level
models (Equations 18, 19, and 20). Districts, schools, and students
account for 0.5%, 8%, and 92% of the variance in days absent,
respectively. We will therefore not consider three-level models
further in this application. Thus, our preferred model for these data
is the two-level negative binomial model which allows for both
school clustering and overdispersion due to omitted student characteristics.


**Random-Intercept** **Models** **With** **Covariates** **for**
**Count** **Responses**


So far we have explored two- and three-level variance components Poisson and negative binomial models and we have shown
how to calculate their VPC expressions. We now explore extending these models to include covariates.


**Model**


Thus, for the two-level Poisson and negative binormal models
(Equations 1 and 9) we now replace �0 with **x** _ij′_ - (in the three-level
versions of these models replace �0 with **x** _ijk′_ �) where **x** _ij_ denotes
the vector of unit- and cluster-level covariates (including the
intercept and any cross-level interactions), and � is the associated
vector of regression coefficients. We note that where units have
different exposures (e.g., students observed for different lengths of
time) an offset (a covariate with regression coefficient set to 1)
measuring the log exposure may be added to **x** _ij_ to account for the
expected variation that this would produce in the observed counts.
The exponentiated regression coefficients can be interpreted as
incidence-rate ratios (IRRs) or ratios of expected counts (e.g.,
exp(�1) gives the ratio of incidence rates or expected counts when
_x_ 1 _ij_ increases by one unit, holding all other covariates constant).


**Conditional** **and** **Marginal** **Statistics**


We must also update the expressions for the conditional and
marginal statistics so that they now condition on **x** _ij_ . Thus, we also
replace �0 with **x** _ij′_ - in all these expressions. It should be clear that
all these expressions, including the VPC and ICC, now depend on
the covariates via the marginal expectation. When an offset is also


PARTITIONING VARIATION IN MULTILEVEL COUNT MODELS 795



included, these expressions will additionally be a function of this
variable. It is therefore important to inspect how the estimated
statistics vary as a function of the marginal expectation and potentially specific covariates. A simple approach is to compute each
statistic for each unit in the data based on the covariate values for
that unit and we shall follow this approach in the application (some
readers may find it helpful to look ahead to Figure 3). Typically,
one will also want to summarize these distributions. A natural
choice is to report the means of these distributions (or perhaps the
medians accompanied by the interquartile ranges to communicate
their variability). We shall also follow this approach in the application. Alternatively, one might calculate each statistic at specific
meaningful values of the covariates. For example, at the covariate
values associated with prototypical units and clusters.


**Application** **Continued**


Recall that the motivation for our application is that school
accountability systems are increasingly using student absenteeism
rates to hold schools to account and that a particular concern is that
differences between schools may in part reflect school intake
differences in student demographic and socioeconomic characteristics rather than school influences on students’ absenteeism. Thus,
the next step in our analysis is to adjust for student demographic
and socioeconomic characteristics so that the resulting predicted
school random effects provide more meaningful estimates of
school influences on students. To improve our interpretation of the
model results, we will also recalculate the VPCs and show how
they now vary as a function of the covariates.

Table 2 presents the results. Model 4 is a two-level negative
binomial model where we include seven student covariates: prior
attainment (test score quintile, based on tests taken 5 years earlier
just before the start of secondary schooling), age (season of birth:



autumn, winter, spring, summer; note that grade retention and
acceleration are not features of the United Kingdom education
system so children vary only in their month of birth, not their
academic year of birth), gender, ethnicity (White, mixed, Asian,
Black, other), language (English or not), special educational needs
(SEN), and free school meals (FSM). [Table](http://dx.doi.org/10.1037/met0000265.supp) S7.1 in the online
supplemental materials presents summary statistics.
A likelihood ratio test confirms that the current model is preferred to its empty counterpart (Model 4 vs. Model 2; �1 [2] - 6609,
_p_ - .001) and so adding the covariates improves the fit of the
model. The current model also continues to be preferred to its
single-level counterpart (Model 4 vs. a single-level negative binomial model with covariates; results not shown; �1 [2] - 5653, _p_ .001) indicating that significant school clustering remains even
after adjusting for the covariates. Similarly, the current model
continues to be preferred to its Poisson counterpart (a two-level
random-intercept Poisson model with covariates; results not
shown; �1 [2] - 314336, _p_ - .001) and so overdispersion also
remains in the residual variation, even after adjusting for the
covariates.
Examining the parameter estimates, we see that all predictors
are statistically significant. Student absenteeism in London is, on
average, higher among lower prior attainers, older students, girls,
White students, those who speak English as a first language, those
with SEN, and those on FSM. These findings agree with recent
applied work in this area (Prior et al., 2019). The exponentiated
regression coefficients can be interpreted as incidence-rate ratios
(IRRs) or ratios of expected counts. Consider, for example, the
FSM estimate of 0.377. The estimated IRR is 1.46 (� exp(0.377)).
Thus, FSM students are, on average, predicted to miss almost 1
and a half days for every day missed by otherwise equivalent
non-FSM students. This differential is substantial.



_Figure_ _3._ Relationship between the predicted school variance partition coefficients (VPC) and marginal
expectation (top panel). Distribution of predicted student-level marginal expectation values (bottom panel). Plots
are based on Model 2.


796 LECKIE, BROWNE, GOLDSTEIN, MERLO, AND AUSTIN



In contrary to our expectation, introducing the student characteristics does not lead the school variance to decrease as we
move from Model 2 to Model 4. This suggests that school
differences in student absenteeism are not predicted by school
differences in student characteristics. Indeed, the school variance increases from 0.093 to 0.103. Such increases can also
occur when fitting multilevel models to continuous responses
and various explanations have been given which are well summarized by Snijders and Bosker (2012, Chapter 7; e.g., clusterlevel confounding and covariates with no clustering). We do not
explore this issue further here. The overdispersion parameter
decreases from 0.877 to 0.782.
Next consider the marginal statistics. The reported values are
the sample means of the statistics where we have calculated
each statistic for every student in the sample, respecting their
covariate values. The VPC depends on the covariates via the
marginal expectation (Equation 16). In Figure 3 we therefore
plot the predicted VPC against the predicted marginal expectation for all students in the sample (top panel). The figure also
plots the sample distribution of predicted marginal expectation
values (bottom panel). The VPC increases from approximately
0.085 to 0.105 as we increase the marginal expectation from its
minimum to maximum predicted values. In models where the
covariates have greater explanatory power, the marginal expectation and therefore VPC would be expected to vary more.
Rather than plot the VPC distribution, researchers will often
prefer to report a single summary statistic. A natural choice is
to report the mean, in our case 0.10. This VPC is slightly higher
than that reported in the variance-components model, 0.08,
suggesting that the covariates have explained a higher proportion of student variation in the data compared with the school
variation which perhaps might be expected given the covariates
are defined at the student level.



Table 2 presents sample mean values for the various other
marginal statistics. These estimates are broadly similar to those
reported for the two-level variance-components negative binomial
model (Table 1, Model 2). This again suggests the covariates have
low explanatory power. The most important predictors for student
absenteeism would appear to lie beyond those available to us here.
Again, this finding agrees with recent applied work in this area
(Prior et al., 2019). Nonetheless, as detecting outlying schools was
one of the motivations for this illustrative application, Figure 4
presents a scatterplot of the predicted school random effects from
the current model (Model 4) against those based on the variancecomponents model (i.e., the null model; Model 2; left panel). The
figure also presents the scatterplot in terms of the ranks of these
two sets of predicted school effects (right panel). The corresponding correlations are 0.91–0.92, which are relatively high and show
that adjusting for school differences in student composition, at
least with respect to the factors examined here, leads to only a
modest reordering of schools.
An important question is: What would be the consequences of
ignoring the significant and substantial residual overdispersion
seen in these data? We can answer this question by fitting a
Poisson version of the current two-level random-intercept model
(not shown). We have already noted that this model provides a
statistically worse fit, but what are the implications for the model
results? The regression coefficients are almost identical. The standard errors, however, are approximately one third those in the
negative binomial model. The Poisson estimates are therefore
spuriously precise because they ignore the overdispersion. Accordingly, they should not be trusted. Note, however, that were the
models to additionally include school-level covariates, then the
standard errors on these regression coefficient would be expected
to be far more similar across the Poisson and negative binomial
versions of the model, as the precision with which the coefficients



Table 1
_Estimates_ _for_ _Variance_ _Components_ _Models_ _Fitted_ _to_ _the_ _Student_ _Absenteeism_ _Data_



Model 2: Two-level
variance-components
negative binomial model



Model 3: Three-level
variance-components
negative binomial model



Parameter/statistic



Model 1: Two-level
variance-components
Poisson model



Parameter estimates
�0 - Intercept 2.085 (0.015) 2.088 (0.015) 2.086 (0.020)

- _v_ [2] - District variance 0.006 (0.003)

- _u_ [2] - School variance 0.100 (0.007) 0.093 (0.007) 0.087 (0.007)

- - Overdispersion 0.877 (0.005) 0.877 (0.005)


Marginal statistics

Marginal expectation 8.46 8.45 8.44
Marginal variance 15.98 84.10 83.79
District (Level 3) component 0.42
School (Level 2) component 7.52 6.95 6.50
Student (Level 1) component 8.46 77.15 76.87
District (Level 3) VPC 0.005
School (Level 2) VPC 0.47 0.08 0.08
Student (Level 1) VPC 0.53 0.92 0.92


Fit statistics

Deviance 785142 422046 422039


_Note_ . Number of districts: _K_ - 32; number of schools: _J_ - 434; number of students: _N_ - 66,955; VPC variance partition coefficients. Standard errors in parentheses.


PARTITIONING VARIATION IN MULTILEVEL COUNT MODELS 797


Table 2
_Estimates_ _for_ _Two-Level_ _Random-Intercept_ _and_ _-Coefficient_ _Models_ _Fitted_ _to_ _the_ _Student_
_Absenteeism_ _Data_



Model 5: Two-level
random-coefficient
negative binomial model



Parameter/statistic



Model 4: Two-level
random-intercept
negative binomial model



Parameter estimates
�0 - Intercept 2.126 (0.021) 2.126 (0.021)
�1 - Prior attainment: Quintile 2 �0.051 (0.012) �0.048 (0.012)
�2 - Prior attainment: Quintile 3 �0.118 (0.012) �0.116 (0.012)
�3 - Prior attainment: Quintile 4 �0.222 (0.012) �0.219 (0.012)
�4 - Prior attainment: Quintile 5 �0.330 (0.014) �0.326 (0.014)
�5 - Age: Spring born 0.026 (0.011) 0.026 (0.011)
�6 - Age: Winter born 0.077 (0.011) 0.078 (0.011)
�7 - Age: Autumn born 0.112 (0.011) 0.112 (0.011)
�8 - Female 0.122 (0.009) 0.122 (0.009)
�9 - Ethnicity: Mixed �0.073 (0.014) �0.074 (0.014)
�10 - Ethnicity: Asian �0.194 (0.013) �0.198 (0.013)
�11 - Ethnicity: Black �0.422 (0.011) �0.421 (0.011)
�12 - Ethnicity: Other �0.194 (0.017) �0.195 (0.017)
�13 - Language not English �0.244 (0.009) �0.242 (0.009)
�14 - Special educational needs (SEN) 0.267 (0.011) 0.267 (0.011)
�15 - Free school meal (FSM) 0.377 (0.008) 0.372 (0.013)
�2 _u_ 0 - School intercept variance 0.103 (0.007) 0.116 (0.009)
�2 _u_ 15 - School FSM variance 0.035 (0.005)

- _u_ 20,15 - School intercept-FSM covariance �0.027 (0.005)

- - Overdispersion 0.782 (0.005) 0.775 (0.005)


Marginal statistics

Marginal expectation 8.50 8.52
Marginal variance 87.05 87.20
School (Level 2) variance 8.71 9.04
Student (Level 1) variance 78.34 78.17
School (Level 2) VPC 0.10 0.10
Student (Level 1) VPC 0.90 0.90


Fit statistics

Deviance 415438 415268


_Note_ . Number of schools: _J_ - 434; number of students: _N_ - 66,955; VPC - variance partition coefficients.
Reference categories. Prior attainment: Quintile 1 (lowest prior attainment); Age: Summer born (youngest in
year); Ethnicity: White. Standard errors in parentheses. Sample average values are reported for the marginal
statistics as each statistic is a function of the covariates.



of cluster-level covariates are estimated is determined primarily by
the cluster-level variance and the estimate of this parameter is
similar in both models.


**Random** **Coefficient** **Models** **for** **Count** **Responses**


In this section, we focus on the two-level random-coefficient
negative binomial model (i.e., regression coefficients, not just the
intercept, are now allowed to vary across clusters) and the associated VPC and ICC statistics. The parallel VPC and ICC expressions for the Poisson model can again be obtained by setting ��
0 in the expressions below. The three-level versions of both
models can be easily extended to include random coefficients in a
parallel fashion [(Supplemental](http://dx.doi.org/10.1037/met0000265.supp) Materials S6).


**Model**


The two-level random-coefficient negative binomial model for
_yij_ can be written as follows



_yij_ | � _ij_                - Poisson(� _ij_ )

ln(� _ij_ ) � **x** _ij′_           -           - **z** _ij′_ **u** _j_ (21)


**u** _j_          - _N_ (0, � **u** )

exp( _eij_ ) � Gamma�� [1] [,][ �][�]


where **z** _ij_ denotes a vector of unit- and cluster-level covariates
(typically an intercept and a subset of the unit-level covariates in
**x** _ij_ ) and **u** _j_ is the associated vector of cluster random coefficient
effects, assumed multivariate normally distributed with zero mean
vector and covariance matrix - **u** .


**Marginal** **Statistics**


The expression for the VPC is as in Equation 16, but where  - _u_ [2]

_′_
is replaced by the cluster variance function **z** _ij_ - **uz** _ij_ . For simplicity,
consider the special case of a model with a random intercept and
one random coefficient associated with the first predictor _x_ 1 _ij_ . In


798 LECKIE, BROWNE, GOLDSTEIN, MERLO, AND AUSTIN


_Figure 4._ Scatterplots of predicted school effect values (left panel) and ranks (right panel) from unadjusted and
covariate adjusted two-level random-intercept negative binomial models (Models 2 and 4).



this case we have **z** _ij′_ **u** _j_ - _u_ 0 _j_ - _u_ 1 _jx_ 1 _ij_ and **z** _ij′_ - **uz** _ij_ ��2 _u_ 0 2� _u_ 01 _x_ 1 _ij_ ��2 _u_ 1 _x_ 12 _ij_ . The expression for the VPC is then given by


level-2 variance
# È
�� _ijM_ �2�exp��2 _u_ 0     - 2� _u_ 01 _x_ 1 _ij_ ��2 _u_ 1 _x_ 12 _ij_     -     - 1�
VPC _ij_ - �� _ijM_ �2�exp��2 _u_ 0 - 2� _u_ 01 _x_ 1 _ij_ ��2 _u_ 1 _x_ 12 _ij_ - - 1�
## ~~Ç~~

level-2 variance

�� _ijM_   - �� _ijM_ �2exp��2 _u_ 0   - 2� _u_ 01 _x_ 1 _ij_ ��2 _u_ 1 _x_ 12 _ij_ ��
## Çlevel-1 variance (22)


where the marginal expectation is defined as follows

   - _ijM_    - exp� **x** _ij′_    -    - ��2 _u_    - 2� _u_ 01 _x_ 1 _ij_ ��2 _u_ 1 _x_ 12 _ij_    - ⁄ 2� (23)


The VPC is now a function of both the marginal expectation and
the cluster variance function. This expression can also be interpreted as an ICC as it is also the response correlation between two
units in the same cluster with the same covariate values. The
expression becomes more complex if we wish to consider two
units with different covariate values.


**Application** **Continued**


Model 4, the previous two-level random-intercept negative binomial model, predicted FSM students in London miss, on average, almost 1 and a half days for every day missed by otherwise
equivalent non-FSM students. In Model 5 we now allow this
average effect to vary across schools by introducing a random
coefficient for FSM. The estimates are presented in Table 2. The
model has two extra parameters, a random slope variance � _u_ 215 and
a random intercept-slope covariance - _u_ 015. A likelihood ratio test
confirms that this model is statistically preferred to the simpler
random-intercept model (Model 5 vs. Model 4: �2 [2] - 170, _p_ .001). The random FSM effects are assumed normally distributed
with an estimated mean of 0.372 (IRR - 1.45) and an estimated
variance of 0.035. The 95% limits of this distribution are 0.005 and



0.739 (IRR - 1.00, 2.09). Thus, the FSM gap in student absenteeism varies substantially across London schools with FSM students in some schools missing no more days, on average, than
otherwise equivalent non-FSM students in these schools, but with
FSM students in other schools missing, on average, over 2 days for
every day missed by otherwise equivalent non-FSM students in
those schools. This variation is substantial and would seem worthy
of further investigation.

Figure 5 explores the relationship between the predicted VPC
and the marginal expectation in the same way that we did for
simpler random-intercept model (see Figure 3). The predicted VPC
now varies not only as a function of the marginal expectation, but
also as a function of the estimated school variance function � _u_ 20 2� _u_ 015 _x_ 15 _ij_ �� _u_ 215 _x_ 152 _ij_, where _x_ 15 _ij_ denotes the dummy variable for
FSM (Equation 22). The school variance function simplifies to � _u_ 20

for non-FSM students and - _u_ 20 - 2� _u_ 015 �� _u_ 215 for FSM students
and thus results in two predicted values: 0.116 for non-FSM
students and 0.097 for FSM students. This in turn leads to two
distinct relationships between the predicted VPC and the marginal
expectation and these are plotted in the figure. The relationship for
non-FSM students lies above that for FSM students. Thus, for any
given predicted number of days absent, the estimated VPC is
higher for non-FSM students than for FSM students. This suggests
that the influence of school attended on student absenteeism is
more pronounced for non-FSM students than FSM students. That
is, non-FSM students appear more sensitive and susceptible to
their environments with respect to being absent from school than
FSM students. This is another interesting finding worthy of further
exploration and again highlights the additional insights provided
when one calculates VPCs in count models.


**Discussion**


In this article, we have derived exact algebraic expressions for
variance partition coefficients (VPCs) and intraclass correlation
coefficients (ICCs) for multilevel Poisson and negative binomial


PARTITIONING VARIATION IN MULTILEVEL COUNT MODELS 799


_Figure 5._ Relationship between the predicted school VPC and marginal expectation by student FSM status (top
panel). Distribution of predicted student-level marginal expectation values by student FSM status (bottom panel).
Plots are based on Model 5. FSM - free school meal; VPC - variance partition coefficients.



models (mean dispersion or NB2 version) for count data. We have
done this for two- and three-level variance-components and
random-intercept versions of these models and we have shown
how these can be extended to accommodate models with random
coefficients. Parallel derivations are provided in [Supplemental](http://dx.doi.org/10.1037/met0000265.supp)
[Materials S3 for two further count models: the Poisson model with](http://dx.doi.org/10.1037/met0000265.supp)
an overdispersion random effect and the constant dispersion or
NB1 negative binomial model. This work significantly extends
that of Stryhn et al. (2006) and Austin et al. (2018) who focus only
on the special case of a two-level random-intercept Poisson model.
We are not aware of any other publications presenting the VPC (or
ICC) expressions for the three more general count models that we
consider in this article and that allow for overdispersion and so all
of these results are new. We view these extensions as important as
overdispersion is a phenomenon which typically occurs in practice.
While the presented VPC expressions have the same general
form as those for continuous and categorical responses (when
expressed in terms of the latent responses underlying the observed
binary, ordinal, or nominal responses), they are nonetheless more
complex, as the VPCs are functions of the covariates. One must
therefore choose the covariate values at which to evaluate the
VPCs. This is also the case in random-coefficient models for
continuous and categorical responses, so this is not a new idea for
readers familiar with those models (Goldstein et al., 2002). A
natural choice is to simply calculate the VPC for each observation
in the data and then report the mean (or perhaps the median
accompanied by the interquartile range to communicate the variability). There is no reason why such calculations cannot be
automated in software as standard postestimation commands and
we encourage software developers to do this.
While we have focused on deriving exact algebraic expressions
for VPCs in different count models, our research also suggests
some more general recommendations for applied researchers analyzing multilevel count data.



First, when count data exhibit overdispersion, as is frequently
the case, the standard multilevel Poisson model will prove
inadequate. The regression coefficients and cluster variance are
not expected to be systematically affected by the overdispersion. The VPC and ICC, however, will be biased upward in the
standard Poisson model and so the degree of clustering and the
cluster effects will appear more important than they truly are.
Furthermore, the standard errors of the regression coefficient
for unit-level (Level-1) covariates, but not cluster-level
(Level-2) covariates, will be biased downward, potentially leading to Type I errors of inference and incorrect research conclusions.
Second, we have found the Poisson model to sometimes run into
computational difficulties when fitted to data with substantial
overdispersion. We therefore recommend negative binomial models because these account for overdispersion and so avoid the
problems associated with the standard Poisson model. Of these, the
mean dispersion is the more widely discussed and is the version
we focused on in this article. However, in software where negative
binomial models are not implemented, or where the researcher is
more familiar with the Poisson model, one alternative would be to
simply add a unit-level overdispersion random effect to the Poisson model (Supplemental Materials S3). The resulting model is
very similar in form to the mean dispersion negative binomial
model and would be expected to lead to similar results. However,
in contrast to the negative binomial model, this overdispersed
Poisson model proves computationally burdensome and may prove
prohibitive in large data settings as it requires integrating out the
unit-level overdispersion random effect.
Our most important recommendation, however, is that researchers should always explore competing models on their data, just as
we have done so here. One learns more from the data doing this
than by restricting attention to any one model.


800 LECKIE, BROWNE, GOLDSTEIN, MERLO, AND AUSTIN



An additional benefit of our work is that we derive and present
expressions for not just the VPC, but the marginal expectation,
variance, covariance, correlation, and ICC for all four models
[considered in the article (Supplemental Materials S3, S4, S5, S6).](http://dx.doi.org/10.1037/met0000265.supp)
These expressions are rarely found in one place and so we hope
that our treatment provides a useful resource for readers. For
example, researchers can use these expressions when simulating
count data or when designing simulation studies to choose true
parameter values such that they imply a certain marginal expectation and variance and degree of clustering or overdispersion in
the population.
Our research also suggests some areas for further work. First,
we have explained how our exact algebraic expressions for the
VPC and ICC for two-level models extend to three-level models
and the same steps can be followed to further extend these expressions to four- and higher-level settings. However, it is less obvious
how they extend to cross-classified and multiple membership
models and more work is required here.
Second, we have restricted our attention to the standard versions
of the Poisson and negative binomial models which use the canonical log link. In some applications, researchers may wish
instead to use the identity or power link functions and these will
lead to different VPC and ICC expressions. More generally, there
are a range of more complex multilevel count models which we
have not explored, but for which it should be possible to extend
the exact algebraic VPC expressions presented here to incorporate
those modeling extensions. These include generalized negative
binomial models, with-zeros or zero-inflated models, truncated
and censored models, hurdle or two-part models, and mixture
models (Cameron & Trivedi, 2013). We leave these extensions for
future research.
Third, count data can be modeled as ordinal data, possibly after
some grouping of counts to limit the number of observed categories. An advantage of this approach is that we can then appeal to
the latent response formulation of ordinal models and their associated VPC and ICC expressions. We can then model the counts as
being due to a latent continuous process that, on crossing progressively higher thresholds, leads to progressively higher values of the
observed count. This approach would seem most useful when there
are just a few low observed counts, say 0, 1, and 2 or more, as
when there are many categories, as there would be in our application, this would lead to many additional threshold parameters to
be estimated. One solution would be to merge adjacent categories,
but the resulting coarsening of the data will often prove unappealing. Goldstein and Kounali (2009) present an alternative solution
which is to apply a smoothing function to the threshold parameters.
Relevant to the current work, they show how with sufficient
structure imposed on these threshold parameters the ordered probit
model reduces to the standard Poisson model. They refer to this
formulation of the Poisson model as the “Poison latent normal
transformation.” Thus, in future work we shall explore whether
this alternative formulation of the Poisson model in terms of a
latent continuous response variable also leads to a simple and easy
to interpret VPC and ICC expression which can complement those
shown here for the observed count data.



**References**


Austin, P. C., Stryhn, H., Leckie, G., & Merlo, J. (2018). Measures of
clustering and heterogeneity in multilevel Poisson regression analyses of
rates/count data. _Statistics_ _in_ _Medicine,_ _37,_ 572–589. [http://dx.doi.org/](http://dx.doi.org/10.1002/sim.7532)
[10.1002/sim.7532](http://dx.doi.org/10.1002/sim.7532)
Browne, W. J., Subramanian, S. V., Jones, K., & Goldstein, H. (2005).
Variance partitioning in multilevel logistic models that exhibit overdispersion. _Journal of the Royal Statistical Society Series A, 168,_ 599–613.
[http://dx.doi.org/10.1111/j.1467-985X.2004.00365.x](http://dx.doi.org/10.1111/j.1467-985X.2004.00365.x)
Cameron, A. C., & Trivedi, P. K. (2013). _Regression analysis of count data_
(2nd ed.). Cambridge, UK: Cambridge University Press. [http://dx.doi](http://dx.doi.org/10.1017/CBO9781139013567)
[.org/10.1017/CBO9781139013567](http://dx.doi.org/10.1017/CBO9781139013567)
Castellano, K. E., & Ho, A. D. (2013). _A_ _practitioner’s_ _guide_ _to_ _growth_
_models_ . Washington, DC: Council of Chief State School Officers.
Department for Education. (2019a). _A_ _guide_ _to_ _absence_ _statistics:_ _March_
_2019_ . London, UK: Author.
Department for Education. (2019b). _National pupil database_ . London, UK:
Author. Retrieved from [https://www.gov.uk/government/collections/](https://www.gov.uk/government/collections/national-pupil-database)
[national-pupil-database](https://www.gov.uk/government/collections/national-pupil-database)
Economic Policy Institute. (2018). _Student absenteeism. Who misses school_
_and_ _how_ _missing_ _school_ _matters_ _for_ _performance_ . Washington, DC:
Author.
Goldstein, H. (1997). Methods in school effectiveness research. _School_
_Effectiveness and School Improvement, 8,_ [369–395. http://dx.doi.org/10](http://dx.doi.org/10.1080/0924345970080401)
[.1080/0924345970080401](http://dx.doi.org/10.1080/0924345970080401)
Goldstein, H. (2011a). Bootstrapping in multilevel models. In J. J. Hox &
J. K. Roberts (Eds.), _Handbook_ _of_ _advanced_ _multilevel_ _analysis_ (pp.
163–171). New York, NY: Routledge.
Goldstein, H. (2011b). _Multilevel_ _statistical_ _models_ (4th ed.). Chichester,
UK: Wiley.
Goldstein, H., Browne, W., & Rasbash, J. (2002). Partitioning variation in
multilevel models. _Understanding_ _Statistics:_ _Statistical_ _Issues_ _in_ _Psy-_
_chology,_ _Education,_ _and_ _the_ _Social_ _Sciences,_ _1,_ 223–231.
Goldstein, H., & Kounali, D. (2009). Multilevel multivariate modelling of
childhood growth, numbers of growth measurements and adult characteristics. _Journal of the Royal Statistical Society Series A, 172,_ 599–613.
[http://dx.doi.org/10.1111/j.1467-985X.2008.00576.x](http://dx.doi.org/10.1111/j.1467-985X.2008.00576.x)
Hox, J. J., Moerbeek, M., & van de Schoot, R. (2017). _Multilevel analysis:_
_Techniques_ _and_ _applications_ (3rd ed.). London, UK: Routledge. [http://](http://dx.doi.org/10.4324/9781315650982)
[dx.doi.org/10.4324/9781315650982](http://dx.doi.org/10.4324/9781315650982)
Leckie, G. (2013a). _Cross-classified_ _multilevel_ _models—Concepts._
_LEMMA_ _VLE_ _Module_ _12,_ _1–60_ . Retrieved from [http://www.bristol.ac](http://www.bristol.ac.uk/cmm/learning/online-course/)
[.uk/cmm/learning/online-course/](http://www.bristol.ac.uk/cmm/learning/online-course/)
Leckie, G. (2013b). _Multiple_ _membership_ _multilevel_ _models—Concepts._
_LEMMA_ _VLE_ _Module_ _13,_ _1–61_ . Retrieved from [http://www.bristol.ac](http://www.bristol.ac.uk/cmm/learning/online-course/)
[.uk/cmm/learning/online-course/](http://www.bristol.ac.uk/cmm/learning/online-course/)
Leckie, G. (2013c). _Three-level_ _multilevel_ _models—Concepts._ _LEMMA_
_VLE_ _Module_ _11,_ _1–47_ . Retrieved from [http://www.bristol.ac.uk/cmm/](http://www.bristol.ac.uk/cmm/learning/online-course/)
[learning/online-course/](http://www.bristol.ac.uk/cmm/learning/online-course/)
Leckie, G., & Goldstein, H. (2009). The limitations of using school league
tables to inform school choice. _Journal_ _of_ _the_ _Royal_ _Statistical_ _Society_
_Series_ _A,_ _172,_ 835–851. [http://dx.doi.org/10.1111/j.1467-985X.2009](http://dx.doi.org/10.1111/j.1467-985X.2009.00597.x)
[.00597.x](http://dx.doi.org/10.1111/j.1467-985X.2009.00597.x)
Leckie, G., & Goldstein, H. (2015). A multilevel modelling approach to
measuring changing patterns of ethnic composition and segregation
among London secondary schools, 2001–2010. _Journal_ _of_ _the_ _Royal_
_Statistical Society Series A, 178,_ [405–422. http://dx.doi.org/10.1111/rssa](http://dx.doi.org/10.1111/rssa.12066)
[.12066](http://dx.doi.org/10.1111/rssa.12066)
Leckie, G., & Goldstein, H. (2017). The evolution of school league tables
in England 1992–2016: ‘contextual value-added’, ‘expected progress’
and ‘progress 8’. _British_ _Educational_ _Research_ _Journal,_ _43,_ 193–212.
[http://dx.doi.org/10.1002/berj.3264](http://dx.doi.org/10.1002/berj.3264)
Leckie, G., & Goldstein, H. (2019). The importance of adjusting for pupil
background in school value-added models: A study of Progress 8 and


PARTITIONING VARIATION IN MULTILEVEL COUNT MODELS 801



school accountability in England. _British Educational Research Journal,_
_45,_ 518–537. [http://dx.doi.org/10.1002/berj.3511](http://dx.doi.org/10.1002/berj.3511)
Leckie, G., Pillinger, R., Jones, K., & Goldstein, H. (2012). Multilevel
modelling of social segregation. _Journal of Educational and Behavioral_
_Statistics,_ _37,_ 3–30. [http://dx.doi.org/10.3102/1076998610394367](http://dx.doi.org/10.3102/1076998610394367)
Long, J. S. (1997). _Regression_ _models_ _for_ _categorical_ _and_ _limited_ _depen-_
_dent variables. Advanced quantitative techniques in the social sciences_ .
Thousand Oaks, CA: Sage.
OECD. (2008). _Measuring improvements in learning outcomes: Best prac-_
_tices_ _to_ _assess_ _the_ _value-added_ _of_ _schools_ . Paris, France: Organisation
for Economic Co-operation and Development Publishing & Centre for
Educational Research and Innovation.
Prior, L., Goldstein, H., & Leckie, G. (2019). School value-added models
for multivariate academic and non-academic student outcomes: A more
rounded approach to using student data to inform school accountability.
_arXiv:2001.01996_ . Retrieved from [https://arxiv.org/abs/2001.01996](https://arxiv.org/abs/2001.01996)
Rabe-Hesketh, S., & Skrondal, A. (2012). _Multilevel_ _and_ _longitudinal_
_modeling using stata: Categorical, count and survival responses_ (Vol. 2,
3rd ed.). College Station, TX: Stata Press.
Rasbash, J., Leckie, G., Pillinger, R., & Jenkins, J. (2010). Children’s
educational progress: Partitioning family, school and area effects. _Jour-_
_nal of the Royal Statistical Society Series A, 173,_ [657–682. http://dx.doi](http://dx.doi.org/10.1111/j.1467-985X.2010.00642.x)
[.org/10.1111/j.1467-985X.2010.00642.x](http://dx.doi.org/10.1111/j.1467-985X.2010.00642.x)
Raudenbush, S. W., & Bryk, A. S. (2002). _Hierarchical_ _linear_ _models:_
_Applications and data analysis methods_ (2nd ed.). Thousand Oaks, CA:
Sage.



Raudenbush, S. W., & Willms, J. (1995). The estimation of school effects.
_Journal_ _of_ _Educational_ _and_ _Behavioral_ _Statistics,_ _20,_ 307–335. [http://](http://dx.doi.org/10.3102/10769986020004307)
[dx.doi.org/10.3102/10769986020004307](http://dx.doi.org/10.3102/10769986020004307)
Skrondal, A., & Rabe-Hesketh, S. (2004). _Generalized_ _latent_ _variable_
_modeling:_ _Multilevel,_ _longitudinal,_ _and_ _structural_ _equation_ _models_ .
Boca Raton, FL: Chapman & Hall/CRC. [http://dx.doi.org/10.1201/](http://dx.doi.org/10.1201/9780203489437)
[9780203489437](http://dx.doi.org/10.1201/9780203489437)
Snijders, T. A. B., & Bosker, R. J. (2012). _Multilevel_ _analysis:_ _An_ _intro-_
_duction_ _to_ _basic_ _and_ _advanced_ _multilevel_ _modelling_ (2nd ed.). London,
UK: Sage.
StataCorp. (2019). Stata statistical software: Release 16 (Version 16)

[Computer software]. College Station, TX: Author. Retrieved from
[http://www.stata.com](http://www.stata.com)
Stryhn, H., Sanchez, J., Morley, P., Booker, C., & Dohoo, I. R. (2006).
Interpretation of variance parameters in multilevel Poisson models.
_Proceedings_ _of_ _the_ _11th_ _Symposium_ _of_ _the_ _International_ _Society_ _for_
_Veterinary_ _Epidemiology_ _an_ _Economics_ (pp. 702–704). Cairns, Australia: International Society for Veterinary Epidemiology an Economics.
Retrieved from [http://www.sciquest.org.nz/node/64294](http://www.sciquest.org.nz/node/64294)


Received September 26, 2019
Revision received December 20, 2019
Accepted February 7, 2020                

