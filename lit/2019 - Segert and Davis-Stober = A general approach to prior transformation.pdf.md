Journal of Mathematical [Psychology](https://doi.org/10.1016/j.jmp.2019.04.002) 91 (2019) 103–118


Contents lists available at [ScienceDirect](http://www.elsevier.com/locate/jmp)

# Journal of Mathematical Psychology


journal homepage: [www.elsevier.com/locate/jmp](http://www.elsevier.com/locate/jmp)

# A general approach to prior transformation [✩]

Simon Segert [a][,][∗], Clintin P. Davis-Stober [b]

a _Princeton_ _University,_ _United_ _States_
b _University_ _of_ _Missouri,_ _United_ _States_



a r t i c l e i n f 

_Article_ _history:_
Received 9 December 2017
Received in revised form 1 April 2019
Accepted 7 April 2019
Available online 3 May 2019


_Keywords:_
Bayesian priors
Inference
Bayes factors
Cognitive models


**1.** **Introduction**



a b s t r a c t


We present a general method for setting prior distributions in Bayesian models where parameters
of interest are re-parametrized via a functional relationship. We generalize the results of Heck and
Wagenmakers (2016) by considering the case where the dimension of the auxiliary parameter space
does not equal that of the primary parameter space. We present numerical methods for carrying out
prior specification for statistical models that do not admit closed-form solutions. Taken together, these
results provide researchers a more complete set of tools for setting prior distributions that could be
applied to many cognitive and decision making models. We illustrate our approach by re-analyzing
data under the Selective Integration model of Tsetsos et al. (2016). We find, via a Bayes factor analysis,
that the selective integration model with all four parameters generally outperforms both the threeparameter variant (omitting early cognitive noise) and the _w_ = 1 variant (omitting selective gating),
as well as an unconstrained competitor model. By contrast, Tsetsos et al. found the three parameter
variant to be the best performing in a BIC analysis (in the absence of a competitor). Finally, we also
include a pedagogical treatment of the mathematical tools necessary to formulate our results, including
a simple ‘‘toy’’ example that illustrates our more general points.
© 2019 Elsevier Inc. All rights reserved.



We present a general approach for setting prior distributions
in Bayesian models when parameters of interest are
re-parametrized via some (typically non-linear) functional relationship. Such cases naturally arise in cognitive modeling. For
example, a model of decision making may have parameters that
correspond to the probability of selecting particular choice alternatives; these parameters may, in turn, be functions of other
parameters, e.g., ones relating to neural constructs such as memory and perception. As the sophistication of such cognitive models
increases, so too does the need for general quantitative tools to
specify such models.

Our approach generalizes the work of Heck and Wagenmakers
(2016) and it is instructive to revisit their motivating example. Let _**θ**_ denote a vector of real-valued, bounded parameters
corresponding to a statistical model. Heck and Wagenmakers
(Henceforth HW) considered psychological models that can be


✩ This work was supported by National Science Foundation, USA grants SES
14-59866 (PI: C. Davis-Stober) and NIH, USA grant (K25AA024182, PI: C. DavisStober). Any opinions, findings, conclusions, or recommendations expressed in
this publication are those of the authors and do not necessarily reflect the views
of their funding agencies or universities.
∗
Corresponding author.
_E-mail_ _addresses:_ [ssegert@princeton.edu](mailto:ssegert@princeton.edu) (S. Segert), [stoberc@missouri.edu](mailto:stoberc@missouri.edu)
(C.P. Davis-Stober).


[https://doi.org/10.1016/j.jmp.2019.04.002](https://doi.org/10.1016/j.jmp.2019.04.002)
0022-2496/© 2019 Elsevier Inc. All rights reserved.



operationalized as order constraints of the form _θi_ ≤ _θj_ . A
prominent example is the class of multinomial processing tree
models (Batchelder & Riefer, 1999). For these models, the order
constraints on the _**θ**_ values encode the psychological theory and
are of primary interest to the researcher. These models can be
re-parametrized by replacing the _**θ**_ values via a set of auxiliary
parameters, denoted _**η**_, which are functions of the original _**θ**_
parameters (Klauer, Singmann, & Kellen, 2015; Moshagen, 2010;
Singmann & Kellen, 2013). This re-parametrization yields additional substantive interpretation. HW demonstrated that placing
non-informative priors on the _**η**_ parameters can lead to highly
informative (and potentially nonsensical) priors on the orderconstrained space defined by the original _**θ**_ parameters. The resulting Bayes factor for the order-constrained model can be highly
sensitive to the priors placed on the auxiliary parameters _**η**_ . HW
showed how to solve for a prior distribution on the _**η**_ parameters
that yields a uniform prior on the order-constrained parameter
space defined by the _**θ**_ parameters. They termed this process _prior_
_adjustment_ . To avoid confusion with prior adjustment in the sense
of Mulder (2014), we will utilize the term ‘‘prior transformation’’
instead.

We generalize prior transformation to the case where the
constrained parameter space of allowable _**θ**_ values need not be
full-dimensional within the space of all possible unconstrained
_**θ**_ values and where the re-parametrization need not be one-toone. In other words, there may be multiple or even infinitely


104 _S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_



many auxiliary parameter values which correspond to a single
value of _**θ**_ . In later sections we provide an illustration of exactly
this case. We also present computational methods to carry out
prior transformation for cases when the desired prior distribution
cannot be obtained analytically and/or the model in question does
not have a closed-form representation and must be simulated.
This allows prior transformation to be carried out on a wide range
of statistical models. We present general guidelines for carrying
out these computational approximations.

We also show that the basic point made by HW applies to
models that are not explicitly described as order-constrained
models. We generalize prior transformation to the case where
the constraints upon the _**θ**_ values are _implicitly_ defined by a given
theory and show that similar issues occur. These constraints may
not be easily solvable in terms of _**θ**_ and may be highly complex
and non-linear in form. As a running example, we consider the _Se-_
_lective Integration (SI)_ model of Tsetsos, Moran, Moreland, Chater,
Usher, and Summerfield (2016a, 2016b). SI is a computational
process model of multi-attribute choice, where the probability
of choosing one alternative over another is a non-linear function
of the (fixed) choice attributes and four psychological parameters: (1) early cognitive noise, (2) selective attentional gating, (3)
memory leak, and (4) late cognitive noise. Repeated choices are
modeled within a binomial random variable framework with the
SI model providing the probability of selecting one alternative
over another.

To evaluate SI using Bayes factors, we would need to specify
prior distributions over these four parameters. [1] If we do not have
strong prior beliefs for the parameter values, we may want to
proceed by letting the ‘‘data speak’’ and specifying a uniform or
otherwise non-informative prior. While perhaps not immediately
apparent, the four parameters in SI serve an auxiliary role to
the binary choice probabilities, which are of primary interest,
as the constraints on the binary choice probabilities define the
primary parameter space of the model when considering choice
frequencies. These binary choice probabilities can be directly
estimated via the actual choices made by the decision maker. The
four SI parameters are latent in nature and not directly measured;
they are estimated through the likelihood function using binary
choice data. As we show in Section 5, SI can be described as
an order-constrained model on the binary choice probabilities
themselves. These constraints are not easily solved for and are
highly non-linear in nature.

In HW’s terminology, the four SI parameters are playing the
role of the auxiliary parameters _**η**_ and the binary choice probabilities are playing the role of _**θ**_, which, depending upon the goals
of the researcher, may require prior transformation. We demonstrate that placing uniform priors over these four SI parameters
yields highly informative, and potentially nonsensical, priors over
the binary choice probabilities, _**θ**_ . We then demonstrate how prior
transformation, in the more general contexts we consider, allow
us to calculate Bayes factors for SI that are based on uniform
priors at the binary choice probability level. We stress that there
is nothing unique about the SI model for making our more general
point. The same basic argument could be applied to well-known
models such as Cumulative Prospect Theory (Tversky & Kahneman, 1992). The core idea is that for many cognitive models
there are both primary and auxiliary parameter spaces and we
must deal with the non-linear relationships between them (which
often encodes the psychological theory) when specifying prior
distributions.

For illustrative and mathematical tractability purposes, we
consider the case of prior transformation for when the desired


1 Tsetsos et al. (2016a, 2016b) did not specify a prior distribution for the four
parameters of SI; they applied classical methods only.



prior distribution on the primary parameters _**θ**_ is uniform. However, our core results (Main Formulas 1 and 2) are general in
that they could be applied to arbitrary continuous distributions
over _**θ**_ . Similar to the perspective offered by Lee and Vanpaemel
(2017), one could specify any _informative_ prior over _**θ**_ and use our
results to obtain the requisite priors over _**η**_ . Thus, our approach
provides researchers a general set of tools when carrying out
prior transformation for Bayesian models.

Our main results leverage theorems from differential geometry and related fields. In Section 2, we provide a brief tutorial
on this topic as it relates to our main results. Also, where appropriate, we reference sections in the Appendix, which contain
additional mathematical descriptions and derivations. Readers
who wish to skip the tutorial section can go directly to Section 3,
which contains the primary technical results of the paper. In
Section 4, we formally connect prior transformation to Bayesian
statistical evidence and Bayes factors. In Section 5, we evaluate
the Selective Integration model of Tsetsos et al. (2016a, 2016b)
under prior transformation for multiple versions of the model
and compare differences in the Bayes factor. Finally, we end with
a discussion and directions for future work. All of our code is
available via an online supplement.


**2.** **Tutorial on the interface between differential geometry and**
**probability** **measures**


In this section, we present a brief tutorial on the necessary
concepts and results from differential geometry needed to apply our main results (Section 3). Throughout our tutorial presentation, we will use a simple toy example to illustrate the
mathematics.


_2.1._ _Toy_ _example_


Suppose that we are astrophysicists interested in devising a
statistical model of the intensity of solar flares, depending on
the location on the surface of the sun. To be more concrete, let
us suppose that we have somehow devised a Bayesian statistical
model for the intensity of a flare on a given day, which we write
as _P_ (intensity of flare = _x_ |location = _p_ ), where _p_ ∈ R [3] is a point
on the surface of the sun (relative to a coordinate system whose
origin is at the center of the sun), and _x_ is the intensity of the
flare (measured in units of energy, say, joules). Clearly, for the
sake of interpretability of the resulting model, it does not make
sense to allow _p_ to be an arbitrary point of R [3] . Indeed, it would
be nonsensical to obtain a prediction about a ‘‘solar flare’’ on the
surface of Mercury by blind mathematical extrapolation. Thus the
surface of the sun acts as a _constrained parameter space_ inside the
ambient space R [3] .

For the purposes of this exposition, we assume the sun to be
a perfect sphere; moreover, we will take our unit of length to be
equal to the radius of the sun, so that the sun has radius equal
to 1 in these units. With these conventions, we may describe our
constrained parameter space mathematically as _S_ [2] := { _p_ ∈ R [3] :
∥ _p_ ∥= 1}.

Now that we have identified the parameter space, the question
arises of how to work with it. Ultimately, we are going to want
to compute Bayesian evidence, which comes down to integrating
a likelihood over the constrained parameter space. When the
geometry of the parameter space is particularly simple, such as
in this example, this step is not especially difficult, and would
probably not be given much thought. As we see in our re-analysis
of the SI model, however, there are situations where the geometry
is not so simple and it is not so obvious how to proceed. Our
objective here is therefore to examine carefully the logic of this
step, so that it may be extended to these more complex situations.


_S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_ 105



In the context of this specific example, it is quite natural and
sensible to switch to a representation of the sphere in terms of
longitude and latitude. [2] With the exception of the two poles,
every point on the sphere has a unique longitude and latitude,
which, in notational contrast to the standard cartographical conventions concerning longitude and latitude, we shall take to lie
in the intervals [0 _,_ 2 _π_ ) and (0 _, π_ ), respectively. In fact, it will
later turn out to be technically easier to deal with the case
where the domain is an open set, so we will henceforth take
the domain of the longitude to instead be (0 _,_ 2 _π_ ), in order to
maintain consistency with future sections. This has the effect of
excluding the prime meridian, and will not materially affect the
following discussion.

We may then re-express our original likelihood _P_ ( _x_ | _p_ ) in terms
of the longitude and latitude of the sun as follows: _P_ ( _x_ | _θ, φ_ ) := _[def]_

_P_ ( _x_ | _pθ,φ_ ), where _pθ,φ_ denotes the unique point with the given
longitude and latitude. The advantage to doing this is that the
domain of the parameter values ( _θ, φ_ ) is now simply a rectangle
(0 _,_ 2 _π_ ) × (0 _, π_ ), with the consequence that any mathematical
obstacles stemming from the geometry of the constrained parameter space have been circumvented. At this point, we might
simply place a uniform prior over the parameter space (0 _,_ 2 _π_ ) ×
(0 _, π_ ) and carry out our analysis directly.

Yet, the decision to specify a point by its longitude and latitude
is ultimately somewhat arbitrary. Indeed, as one learns in school,
there are a multitude of different map projections, each of which
may be preferred over others in certain situations. For example,
let us consider the Mercator projection. To follow the exposition,
it is only necessary to know that this projection specifies each
point _p_ on the sphere (again, with the exception of the poles and
prime meridian) by a unique pair of numbers ( _θ, y_ ) ∈ (0 _,_ 2 _π_ ) ×
(−1 _,_ 1) which, crucially, ‘‘live’’ in a different space than the longitude and latitude [3] pair ( _θ, φ_ ) ∈ (0 _,_ 2 _π_ ) × (0 _, π_ ). Just as before,
an astrophysicist might be tempted to place a uniform prior over
the Mercator parameter space (0 _,_ 2 _π_ ) × (−1 _,_ 1) and proceed
to compute Bayesian evidence and other various quantities of
interest.

Now the potential issue becomes clear. There is no reason
why we should expect that a uniform prior on (0 _,_ 2 _π_ ) × (0 _, π_ )
should give us the same Bayesian evidence as a uniform prior
on (0 _,_ 2 _π_ ) × (−1 _,_ 1). This is precisely the situation described by
HW. In the context of this example, HW showed that this can be
addressed by solving for the Mercator projection parameters in
terms of the longitude and latitude parameters. In other words,
by computing the function which carries the pair ( _θ, φ_ ) to the
unique pair ( _θ_ [′] _, y_ [′] ) that corresponds to the same point on the
sphere. HW showed that the equivalent prior can be solved for
in terms of the Jacobian determinant of this function, essentially
as an application of the multivariate change-of-variables formula.
They termed the prior so obtained the ‘‘implied distribution’’.

However, the formalism of HW is not sufficient to cover all
possible cases of interest. Firstly, it is not applicable to constrained spaces of positive codimension such as our example of
the sphere (indeed, this is a 2-dimensional space inside of R [3],


2 While longitude and latitude are typically introduced only for the Earth, the
logic behind them applies just as well to the sun. Indeed, the sun rotates around
an axis, allowing us to define the positions of the North and South poles just as
for the earth. This in turn gives a well-defined notion of latitude. The definition
of longitude depends on an arbitrary choice of a semi-circular arc connecting
the poles, analogous to the Prime Meridian, which we shall suppose has been
made.
3 For the benefit of the interested reader, we mention that this projection is
constructed to ensure that angles on the surface of the sphere are not distorted.
This can be accomplished by enclosing the earth in a cylinder around the
equator, projecting outwards, and then unrolling the cylinder. However, there is
a twist, in that, instead of projecting the line of latitude at angle _φ_ to the height
sin _φ_, we project this line of latitude to the height _y_ = ln(tan( _π/_ 4 + _φ/_ 2)).



and thus has codimension 3 − 2 = 1). Note that in our example,
we could not directly compute the implied distribution on the
surface of the sphere; we could only express it indirectly via its
implied distributions on the parameter spaces. Nevertheless, it is
easy to conceive of a situation in which it might be more natural
to work directly with a distribution on the surface of the sphere.
For example, it would be quite sensible to want a prior which
is proportional to the surface area on the sphere, and then work
backwards to express it in terms of a set of parameters. If we
want to carry out this procedure in complete generality, then,
we will need, at the very least, a way to measure surface area of
a _k_ -dimensional constrained space in an _n_ -dimensional ambient
parameter space. By contrast, HW formulated their results only
for _n_ -dimensional subsets of _n_ -dimensional parameter spaces.

Secondly, their approach does not handle parameter transformations that are non-invertible. To illustrate, let us consider
the following situation: due to certain (hypothetical) advances in
astronomical theory, we are led to postulate that the solar flare
intensity in fact only depends on the latitude of the point _p_ and
that our likelihood function is a function of the latitude only.
Given a prior on the space (0 _,_ 2 _π_ ) × (0 _, π_ ), we would like to
express an equivalent prior (i.e. one that yields the same evidence
value when integrated against the likelihood) in terms of the
latitude only. The approach of HW is not applicable here, because
the map taking a point to its latitude is not one-to-one (that is,
many different points are at the same latitude). Let us denote the
map taking a pair of angles ( _θ, φ_ ) ∈ (0 _,_ 2 _π_ ) × (0 _, π_ ) to the corresponding point on the sphere by _F_, and let us denote the map
taking a point on the sphere to its latitude by _H_ : _S_ [2] →[0 _, π_ ]. We
can thus recast our problem as describing the implied distribution
of _H_ - _F_ : (0 _,_ 2 _π_ ) × (0 _, π_ ) →[0 _, π_ ], where ‘‘◦’’ is function
composition. Alternatively, given a prior directly on _S_ [2], we wish
to compute the implied distribution under _H_ . How can this be
done? It is here that the geometric picture provides invaluable
intuition and ultimately points towards a general solution.

We stress that these two issues are not merely contrived features of our toy example. To wit, we shall encounter both of them
when we consider the SI model of Tsetsos et al. (2016a, 2016b)
in Section 5. More specifically, we will see that (in a certain
restricted case), the SI model may be abstractly described as a
2-dimensional surface sitting in a three-dimensional space (under
the experimental design they considered), each axis of which
corresponds to a probability of a participant selecting a given
alternative. Hence, the relevance of the first issue raised above.
Furthermore, in their specification of the model, this surface is in
fact parametrized by a function of _three_ latent parameters. Hence
the relevance of the second issue.

Having ‘‘pulled back the curtain’’ to explain the rationale guiding the construction of our toy example, let us now return to it
in order to try to understand how the issues that it illustrates
may be resolved in this simplified setting. We can schematically
imagine the effect of the map _H_ as squashing together all points
at the same latitude. We may thus ‘‘slice up’’ the sun into a
collection of circles (including degenerate one-point circles at
the poles) of the same latitude. It is intuitively plausible that
the density of the implied distribution on the ‘‘latitude space’’

[0 _, π_ ] should be a combination of two contributions: (1), an
integral/average taken over the corresponding line of latitude,
and (2) a ‘‘latitude-wise stretch factor’’ analogous to the Jacobian
determinant that appears in the ordinary change of variables
formula from multivariate calculus.

The language of differential geometry is perfectly suited to
make these geometric intuitions precise, and, furthermore, fluency in this language will allow for application of the same
reasoning to far more general situations than considered in this
toy example. We state the main formulas in Section 3 that make
these ideas precise and general.


106 _S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_



_2.2._ _The_ _definition_ _of_ _a_ _manifold_


All of the constrained spaces which we will consider are examples of _smooth_ _manifolds_ . Intuitively speaking, a manifold _M_
is a space which ‘‘looks like’’ Euclidean space near each point.
More precisely, for each point _p_ ∈ _M_, we require that there exists
a smooth invertible function from a subset _V_ ⊂ _M, p_ ∈ _V_, to
Euclidean space of appropriate dimension, exactly like the map
in the previous section _S_ [2] −{poles, prime meridian} → (0 _,_ 2 _π_ ) ×
(0 _, π_ ) which takes a point to its longitude and latitude, and is
only defined for points other than the two poles and the prime
meridian. Evocatively, such a map is sometimes called a ‘‘chart’’,
although we shall not use this terminology. We may thus say that
a manifold is a space such that every point is in the domain of a
chart (a collection of charts whose domains cover the manifold is
sometimes called an ‘‘atlas’’).

Let us now introduce some standard notation: for _p_ ∈ R _[n]_ and
_ϵ_ _>_ 0, we set _B_ _[n]_ _ϵ_ [(] _[p]_ [)] [:=] [{] _[q]_ [∈] [R] _[n]_ [:] [∥] _[p]_ [ −] _[q]_ [∥] _[<]_ _[ϵ]_ [}][.] [A] [subset] _[U]_ [⊂] [R] _[n]_

is said to be _open_ if for any _p_ ∈ _U_, we can find _ϵp_ _>_ 0 with the
property that _B_ _[n]_ _ϵp_ [(] _[p]_ [)][ ⊂] _[U]_ [(the] [subscript] [is] [simply] [a] [reminder] [that]
the value of _ϵ_ may depend on the point _p_ ). We can now give a
formal definition.



**Definition 1.** Let _M_ be a subset of R _[n]_ . We say that _M_ is a manifold
of dimension _k_, if, for any _p_ ∈ _M_, we can find

1. an open set _U_ ⊂ R _[k]_,
2. an open set _V_ ⊂ R _[n]_ containing _p_, and
3. a smooth one-to-one map _C_ : _U_ → R _[n]_, such that _C_ ( _U_ ) =
_V_ ∩ _M_ .

Such a _C_ is called a _local_ _parametrization_ [4] near the point _p_ .


How does one check that a given space _M_ is a manifold?
One easy but important special case is when _M_ is of the form
_F_ ( _U_ ), where _U_ ⊂ R _[k]_ is an open set and _F_ : R _[k]_ → R _[n]_ is a
smooth, one-to-one function. In this case, we can just take the
local parametrization to be _F_ .

Returning to our sun example, let us consider _M_ = _S_ [2] with
_n_ = 3. We would like to show that _M_ is a manifold of dimension 2. We will consider the south pole _p_ = (0 _,_ 0 _,_ −1), with
the verification for other points being similar. Letting _x, y_ and
_z_ denote the standard Cartesian coordinates, we note that on a
small neighborhood of _p_, we can express _z_ as a function of _x_ and _y_
via _z_ = − ~~√~~ 1 − _x_ [2] - _y_ [2] . To match up with the definition, we may



via _z_ = − ~~√~~ 1 − _x_ [2] - _y_ [2] . To match up with the definition, we may

take _C_ : _B_ [2] _ϵ_ [(0)] [→] [R][3] [defined] [by] _[C]_ [(] _[x][,][ y]_ [)] [=] [(] _[x][,][ y][,]_ [ −] ~~√~~ 1 − _x_ [2] - _y_ [2] ),



take _C_ : _B_ [2] _ϵ_ [(0)] [→] [R][3] [defined] [by] _[C]_ [(] _[x][,][ y]_ [)] [=] [(] _[x][,][ y][,]_ [ −] ~~√~~ 1 − _x_ [2] - _y_ [2] ),

where _ϵ_ is any sufficiently small number (any _ϵ_ _<_ 1 will do).
In this case _C_ ( _B_ [2] _ϵ_ [(0))] [=] [{][(] _[x][,][ y][,][ z]_ [)] [∈] _[S]_ [2] [:] _[x]_ [2] [+] _[y]_ [2] _[<]_ _[ϵ,][ z]_ _[<]_ [0][}][,]
which is geometrically equal to the intersection of _S_ [2] with the
open cylinder _B_ [2] _ϵ_ [(0)] [×][ (][−∞] _[,]_ [ 0).] [This cylinder plays] [the role of] [the]
set _V_ in the definition of a manifold.

One can imagine that this kind of reasoning would be difficult
to carry out in more complicated examples. Indeed, to find the
local parametrization, we effectively had to solve for _z_ in terms
of _x_ and _y_ given the constraint _x_ [2] + _y_ [2] + _z_ [2] = 1. We were lucky that
the constraint had an algebraically simple form, but this need not
always be the case. Fortunately, there is a general theorem that
lets us bypass such arguments in cases such as this one where
the manifold is defined by a constraint.


4 Actually, our usage of the term ‘‘manifold’’ here is somewhat nonstandard.
What we call manifolds would be more properly called ‘‘embedded submanifolds
of Euclidean space’’. The more common definition of a manifold does not make
reference to any ambient space. However, we will not need the additional
abstraction afforded by this definition. In addition, the essential concepts are
somewhat easier to grasp in this special case, and may be all extended to the
more general case in a relatively straightforward way, as detailed in e.g., Boothby
(2003).



**Theorem** **1.** _Let_ _f_ : R _[n]_ → R _be_ _a_ _smooth_ _function_ _and_ _let_ _c_ ∈ R _._
_Then_ _f_ [−][1] ( _c_ ) _is_ _a_ _smooth_ _manifold_ _of_ _dimension_ _n_ - 1 _provided_ _that_
_the_ _vector_ _of_ _partial_ _derivatives_ ( _∂_ 1 _f_ ( _x_ ) _, . . ., ∂nf_ ( _x_ )) _is_ _nonzero_ _for_
_every_ _x_ ∈ _f_ [−][1] ( _c_ ) _._


This is a special case of the well-known Implicit Function
Theorem. See Boothby (2003) for a more general statement and
proof.

Let us demonstrate this theorem applied to the sphere (our
‘‘sun’’ example). We may write _S_ [2] = _f_ [−][1] (0) where _f_ ( _x, y, z_ ) =
_x_ [2] + _y_ [2] + _z_ [2] - 1. The matrix of partial derivatives at the point
_p_ = ( _x, y, z_ ) is given by (2 _x,_ 2 _y,_ 2 _z_ ). If ( _x, y, z_ ) ∈ _f_ [−][1] (0), then this
vector can never be equal to (0 _,_ 0 _,_ 0) (this would contradict the
assumption that _x_ [2] + _y_ [2] + _z_ [2] = 1). This establishes, via the above
theorem, that _S_ [2] is indeed a manifold of dimension 2.

It is worth noting that many of the constrained spaces that we
are likely to encounter are not quite manifolds in the sense of the
formal definition we gave. For example, the unit interval [0 _,_ 1]
does not satisfy the manifold condition at the endpoints. It is an
example of a ‘‘manifold with boundary’’, which may be thought
of as a manifold of dimension _k_ with a manifold of dimension
_k_ - 1 ‘‘glued onto it’’. In this example, we have the bona-fide 1manifold (0 _,_ 1), with the 0-dimensional manifold {0 _,_ 1} (i.e., the
manifold consisting of two discrete points) glued on. Because we
are ultimately interested in performing integrals over manifolds,
and because the boundary pieces will always have measure zero,
we will not concern ourselves too much with this issue.


_2.3._ _Jacobians_ _on_ _manifolds_


Our generalization of the results of HW, which we describe in
detail in Section 3, will require a generalization of the notion of a
Jacobian. Similar to the usage of the Jacobian matrix applied in the
change of variables formula used by HW, we now consider how to
define the Jacobian derivative of a map between manifolds. The
basic idea is quite simple. First, recall that the Jacobian matrix
at _p_ ∈ R _[n]_ of a map _F_ : R _[n]_ → R _[m]_ is simply the matrix of partial
derivatives of the components of _F_, evaluated at the point _p_ . Now
consider a smooth map _F_ : _M_ → _N_, where _M_ and _N_ are manifolds
and let _p_ ∈ _M_ . [5] As we saw in the previous subsection, we can
choose local parametrizations near _p_ ∈ _M_ and near _F_ ( _p_ ) ∈ _N_,
which amounts to identifying small neighborhoods around _p_ and
_F_ ( _p_ ) with open subsets of Euclidean space.

Under these identifications, we may regard _F_ as a map between Euclidean spaces, and take its Jacobian derivative as before,
thereby obtaining a matrix.

More precisely, let _CM_ : _U_ 1 → _M_ and _CN_ : _U_ 2 → _N_ be local
parametrizations with _p_ ∈ _CM_ ( _U_ 1) and _F_ ( _CM_ ( _U_ 1)) ⊂ _CN_ ( _U_ 2). Here
_U_ 1 and _U_ 2 are subsets of Euclidean space. Then the composition [6]

_CN_ [−][1] [◦] _[F]_ [ ◦] _[C][M]_ [is] [a] [function] [from] _[U]_ [1] [to] _[U]_ [2][,] [that] [is] [to] [say,] [its] [domain]
and range are both subsets of Euclidean spaces, and as such this
function has a well-defined Jacobian matrix. This is illustrated by
the following diagram:


Here, the dotted line denotes the composition _CN_ [−][1] [◦] _[F]_ [ ◦] _[C][M]_ [.] [Note]
that the function _CN_ [−][1] is only defined on the image _CN_ ( _U_ 2).
Finally, to compute the Jacobian of _F_ at _p_, we should evaluate
the Jacobian of _CN_ [−][1] [◦] _[F]_ [ ◦] _[C][M]_ [at] _[ C]_ _M_ [−][1][(] _[p]_ [)][ ∈] _[U]_ [1][. It is essential to realize]


5 When we formulate our results in Section 3.1, _M_ and _N_ will be parameter
spaces for a given model, and _F_ will be a reparametrization function.
6 Here _CN_  - [1] denotes the inverse function of _CN_, not the preimage.


_S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_ 107



that this procedure depends upon which local parametrizations
on _M_ and _N_ that we chose. We refer the reader to Appendix A.4
for an alternative formulation of this idea which is closer to
the approach followed in many modern textbooks on differential
geometry.


_2.4._ _Measures_ _on_ _manifolds_ _and_ _Riemannian_ _structures_


In this final subsection regarding mathematical preliminaries,
we explain how to work with measures defined on manifolds.
This is crucial for the body of the paper, since the priors that
we will consider in this paper are most naturally regarded as
measures on appropriate manifolds.

The construction of these measures is intimately tied to the
way in which we decide to measure lengths and angles on the
manifold. In the second half of this section, we will flesh out
this connection, and see how it leads naturally to the notion of
a _Riemannian structure_ on a manifold which, essentially, is simply
a choice of how to measure lengths and angles on the surface of
the manifold.

The first step is to understand how one may ‘‘inherit’’ the
geometry of the ambient Euclidean space to compute the volume
of a (measurable) subset of a manifold, thus defining a measure
supported on _M_ which we denote by _VM_ . We will then explain
how to compute this measure relative to a local parametrization.

Note that this notion of volume need not coincide with the
Lebesgue measure on the ambient space; indeed, if _M_ ⊂ R _[n]_ has
dimension _k_ _<_ _n_, it can be shown that _M_ has Lebesgue measure
zero. On the other hand, if, _L_ ⊂ R [2] is a line segment, then there
is a well-defined notion of length, or ‘‘1-dimensional volume’’ of
_L_, even though it has measure zero, and we will show how to
extend this idea to general manifolds.

Not surprisingly, we will proceed by approximating the manifold with shapes of a known volume, and then take an appropriate limit. Given a _k_ -dimensional manifold, we choose a local
parametrization _C_ with domain _U_ ⊂ R _[k]_ . We now explain how to
compute the volume of _C_ ( _U_ ).

We divide _U_ into very small cubes with edges parallel to the
coordinate axes, of length _∆x_ . Let _x_ be the bottom left-most
point of such a square. Then the image of this square will be
approximately a _k_ -dimensional parallelotope spanned [7] by the
vectors { _∆x_ ( _∂iC_ )( _x_ )} _i_ .

By letting the grid get finer, it seems quite plausible that the
sum of the volumes of all such paralellotopes should approach a
well-defined limit. This can be made precise, although we omit
the details here. Let us denote this limiting measure by _VM_ . To
be explicit, _VM_ ( _C_ ( _U_ )) is defined by approximating _C_ ( _U_ ) by a fine
mesh of parallelotopes, and taking the limiting volume as the
mesh sizes approaches zero.

Now, in order to complete the construction, we only need
to figure out how to compute the volume of an arbitrary _k_ dimensional parallelotope.

Let us first introduce a standard notation:


**Definition** **2.** Let _v_ 1 _, . . ., vk_ ⊂ R _[n]_ be an arbitrary collection of
vectors. The Gram matrix of the vectors is the _k_ × _k_ matrix whose
( _i, j_ ) entry is given by the standard dot product ( _vi, vj_ ). We denote
this matrix by _G_ ({ _vi_ }) or just _G_ ( _v_ ) when the context is clear. The
determinant of the Gram matrix is called the Gram determinant.
It is always non-negative. [8]


7 Recall that a parallelotope is simply the higher-dimensional generalization
of a parallelogram.
8 To see this, note that _G_ ({ _vi_ }) = _AT A_ where _A_ is the matrix whose _i_ th
column is equal to _vi_ . If _v_ is an eigenvector of _G_ with eigenvalue _λ_, then
_λ_ | _v_ | [2] = ( _Gv, v_ ) = ( _A_ _[T]_ _Av, v_ ) = ( _Av, Av_ ) =| _Av_ | [2] ≥ 0. Since _G_ is a symmetric
matrix, the determinant is equal to the product of eigenvalues, which is a
product of non-negative numbers.



_VM_ ( _C_ ( _U_ )) is given by ∫ _U_ √ _det_ ( _G_ ({ _∂j_ ( _C_ ( _x_ ))} _j_ )) _dx_, i.e., the integral of

the square root of the Gram determinant of the partial derivatives
of _C_ . [9]

We now turn to the notion of a Riemannian structure on a
manifold, which was already implicit in our construction of the
measure _VM_ . To begin, let us note that the construction of _VM_
could be generalized substantially at the cost of introducing a
small amount of conceptual overhead.

Indeed, returning to the proposition, we see that, in defining
the measure _VM_, we only needed to measure the volume of parallelotopes, and via the proposition, we saw that these volumes
could be expressed only in terms of inner products of vectors.
Let us imagine repeating the construction, except replacing ( _vi, vj_ )
everywhere with a different inner product on R _[n]_ .

Any such inner product is necessarily of the form ( _vi, Avj_ )
where _A_ is a symmetric, positive definite matrix. In fact, we can
even allow _A_ to depend on the origin of the vectors _vi, vj_ . Thus,
given two vectors _v_ and _w_ with the same origin _p_, we may
define a ‘‘generalized inner product’’ ( _v, A_ ( _p_ ) _w_ ), and repeat the
construction of the measure _V_, replacing the old inner product
everywhere with this new inner product. Such a matrix-valued
function _A_ is called a Riemannian metric on R _[n]_, and denoted


9 In contrast to the _∆x_ notation in the preceding paragraph, ‘‘dx’’ here
denotes an integral against the (possibly multi-dimensional) Lebesgue measure
on _U_ .



We have the following result.



**Proposition** **1.** _Let_ _P_ ⊂ R _[n]_ _be_ _a_ _k-dimensional_ _parallelotope_
_spanned by the k vectorsof_ _P_ _is_ _given_ _by_ _Volk_ ( _P_ ) = _v_ 1 [√] _, . . ., vdet_ ( _Gk_ ({ _. Thenvi_ })) _, the k-dimensional volumei.e.,_ _the_ _square_ _root_ _of_ _the_

_Gram_ _determinant._


This is a direct generalization of the familiar fact that for a
linear map _A_ : R _[n]_ → R _[n]_, the image _A_ ([0 _,_ 1] _[n]_ ) of the unit
hypercube has volume | _det_ ( _A_ ) |. Indeed, the image is exactly the
parallelotope spanned by the columns of _A_ . If we denote these
columns by _v_ 1 _, . . ., vn_, then the Gram matrix is simply _G_ ({ _vi_ }) =
_A_ _[T]_ _A_, and because _A_ is a square matrix, we have _det_ ( _A_ _[T]_ _A_ ) =
_det_ ( _A_ _[T]_ ) _det_ ( _A_ ) = _det_ ( _A_ ) [2], by the multiplicativity property of the
determinant. The proof of the proposition depends only on basic
properties of volume and of the determinant, and is omitted.

Having completed the construction, let us now see how we
might work with the measure _VM_ . Returning to our ‘‘sun’’ example _S_ [2] ⊂ R [3], we consider the parametrization _C_ : (0 _,_ 2 _π_ ) × (0 _, π_ )
in terms of longitude and latitude. Let _U_ ⊂ (0 _,_ 2 _π_ ) × (0 _, π_ ) be an
open set; we will compute _VM_ ( _C_ ( _U_ )).

To construct the ‘‘mesh’’, we choose a collection { _xi_ } = {( _θi,_
_φi_ )} of evenly-spaced grid points in U. Let us denote the spacing
by _∆x_ . In particular, the area of a square formed by 4 adjacent
grid points is ( _∆x_ ) [2] .

Next, we cover the portion _C_ ( _U_ ) ⊂ _S_ [2] with 2-dimensional
parallelograms with origin _C_ ( _xi_ ) and spanned by _∂jC_ ( _xi_ ) _∆x_, _j_ =
1 _,_ 2. Note that, e.g., _C_ ( _xi_ + _∆x_ (1 _,_ 0)) ≈ _C_ ( _xi_ ) + _∂_ 1 _C_ ( _xi_ ) _∆x_, so that
the parallelograms do indeed approximately touch at the edges.

Using simple trigonometry, one verifies that _C_ ( _θ, φ_ ) =
(sin _φ_ cos _θ,_ sin _φ_ sin _θ,_ cos _φ_ ), and by a short computation, we
see _∂θ_ _C_ = (− sin _φ_ sin _θ,_ sin _φ_ cos _θ,_ 0) and _∂φC_ = (cos _φ_ cos _θ,_
cos _φ_ sin _θ,_ - sin _φ_ ). By the proposition, each parallelogram has
area given by the square root _det_ ( _G_ ({ _∂θ_ _C_ ( _xi_ ) _, ∂φ_ ( _C_ ( _xi_ ))})) of the

[√]



area given by the square root _det_ ( _G_ ({ _∂θ_ _C_ ( _xi_ ) _, ∂φ_ ( _C_ ( _xi_ ))})) of the

[√]

gram determinant. A bit of tedious computation reveals this to
be equal to sin( _φi_ ). Our approximate volume of _C_ ( _U_ ) is thus
∑

_i_ [sin(] _[φ][i]_ [)(] _[∆][x]_ [)][2][, and passing to the limit in the usual fashion, this]
becomes an integral of the function sin( _φ_ ) over _U_ .

Using similar heuristic arguments, one can convince oneself
that for any manifold _M_ with local parametrization _C_, the value of
_VM_ ( _C_ ( _U_ )) is given by ∫ √ _det_ ( _G_ ({ _∂j_ ( _C_ ( _x_ ))} _j_ )) _dx_, i.e., the integral of



√
_U_


108 _S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_



_g_ for historical reasons. [10] Let us denote the measure obtained
by measuring the volumes of parallelotopes in this way by _VMg_ [.]
This measure is called the _Induced_ _Riemannian_ _volume_ _form_ _on_
_M_ . By retracing the above arguments, it may be easily seen that
_VMg_ [(] _[C]_ [(] _[U]_ [))] [=] ∫ _U_ √ _det_ ( _G_ _[g]_ ({ _∂j_ ( _C_ ( _x_ ))} _j_ )) _dx_, where the ( _i, j_ ) entry of



holds for any bounded measurable _f_ : _Im_ ( _F_ ) → R. [14] See
Appendix A.3 for a brief discussion and additional examples of
pushforward measures.

To perform Bayesian analysis, it is necessary to specify a prior
distribution _π_ . We consider two possible ways to do this:

(1) Specify a prior on the auxiliary parameter space, and push
it forward by _F_ to obtain the corresponding distribution on the
primary parameter space.

(2) Specify a prior on the primary parameter space, and try to
‘‘reverse engineer’’ an appropriate prior on the auxiliary parameter space which pushes forward to this distribution.

The discrepancy between these two approaches was pointed
out in the special case of multinomial processing tree models by
HW. Using our notation, they considered the case of _**η**_ ∈[0 _,_ 1] _[n]_,
_F_ ( _**η**_ ) _i_ = [∏] _j_ ≥ _i_ _[η][j][,]_ [ 1][ ≤] _[i]_ [ ≤] _[n]_ [.] [They] [showed] [how] [to] [construct] [a] [prior]

on _**η**_ whose pushforward is equal to the uniform distribution on
_Im_ ( _F_ ) = {( _θ_ 1 _, . . ., θn_ ) ∈[0 _,_ 1] _[n]_ : _θ_ 1 ≤· · · ≤ _θn_ }, thus carrying
out Option (2), for the uniform distribution with support equal
to _Im_ ( _F_ ). Observe that in this (HW’s) setup, _n_ = _m_ .

As we now show, their argument can be adapted to a wide
class of smooth functions _F_ . Our approach differs in several respects. First, whereas they started with a primary parameter
space of a known geometry (implied by a multinomial processing
tree model), our starting point is the parametrization function _F_,
and the resulting primary parameter space _Im_ ( _F_ ) may be very
difficult or impossible to express as a system of equalities and
inequalities. Second, we do not need to assume that the image of
_F_ is full-dimensional in the codomain, nor do we need to assume
_F_ is one-to-one.

We must define what is meant by a uniform prior on the
primary parameter space. Let us fix a Riemannian metric on R _[n]_ .
We use the fact that the primary parameter space is an immersed
submanifold in R _[n]_, and therefore inherits a Riemannian metric
through restriction of this metric as described in Section 2.4 (see
also Chavel, 2006). In general, for any such submanifold _M_, we
will denote the corresponding measure [15] on _M_ by _VM_ . For the
primary parameter space, we will use the notation _Vprim_ = _VIm_ ( _F_ ).

Our aim is to construct a distribution _µ_ on _D_ for which _F_ ∗ _µ_ =
_Vprim_ . In other words, _µ_ will be a prior distribution on the auxiliary parameter space which induces the uniform distribution on
the primary parameter space. Fix a Riemann metric on _D_, and let
_Vaux_ = _VD_ denote the corresponding measure.

We now recall the well-known _co-area formula_ of Riemannian
Geometry (see Chavel, 2006, for a precise statement and proof),
which generalizes both the basic change of variables formula
(used by HW for prior transformation when _n_ = _m_ ) and Fubini’s
Theorem.



definition which we find more convenient is that the pushforward measure _F_ ∗ _π_ is the unique measure for which the equation

∫ ∫



∫
_f_ ( _**θ**_ )( _F_ ∗ _π_ )( _d_ _**θ**_ ) =



_f_ ( _F_ ( _**η**_ )) _π_ ( _d_ _**η**_ ) _,_



√
_U_



_VM_ [(] _[C]_ [(] _[U]_ [))] [=] ∫ _U_ √ _det_ ( _G_ _[g]_ ({ _∂j_ ( _C_ ( _x_ ))} _j_ )) _dx_, where the ( _i, j_ ) entry of

the matrix _G_ _[g]_ ({ _∂j_ ( _C_ ( _x_ ))} _j_ ) is equal to _gC_ ( _x_ )( _∂i_ ( _C_ ( _x_ )) _, ∂j_ ( _C_ ( _x_ ))).

The practical-minded reader may wonder what is to be gained
by this abstraction. The justification is that non-Euclidean metrics
are in fact encountered ‘‘in the wild’’, even if it is not immediately
apparent. Perhaps the prototypical example of this is the Jeffreys
prior, which is the measure induced by the so-called Fisher–Rao
metric on the ambient space (Amari, 1985). Alternatively, this
formalism may be used when the axes of the space are naturally
regarded as having different units, in which case the matrix _A_ can
be taken to be diagonal, with the entries being conversion factors
to a common scale.

Finally, we mention that it is possible to define a Riemannian
metric directly on a manifold, without reference to such a metric
on the ambient space. To do so, we simply need, for each point
_p_ ∈ _M_, an inner product _gp_ ( _v, w_ ), where _v_ and _w_ are any two
vectors whose origins are at _p_ and which are tangent to _M_ . (More
precisely, _v_ and _w_ should lie in the _tangent_ _space_ _TpM_, which is
formally defined in Appendix A.4). In addition, _gp_ is required to
depend smoothly on _p_, in a certain precise sense. A manifold with
such structure is called a _Riemannian_ _manifold_ .


**3.** **General** **prior** **transformation**


_3.1._ _Setup_ _and_ _general_ _formulas_


Let us consider a Bayesian statistical model that is expressed
in terms of a parameter vector _**θ**_ ∈ R _[n]_ which is itself defined in
terms of another parameter vector _**η**_ ∈ R _[m]_ by a relation of the
form _**θ**_ = _F_ ( _**η**_ ) for some smooth function _F_ . [11]

As an example, and a concrete way to connect to HW, we
could consider _**θ**_ as a vector of choice probabilities applied to a
set of _n_ binary decision tasks, while considering _**η**_ as a vector of
theoretical latent parameters that govern the choice probabilities
(e.g., ‘‘loss aversion’’). Abstractly speaking, _F_ may also be regarded
as a (possibly many-to-one) reparametrization function, and conversely, any reparametrization function also may be regarded as
a model in this sense.

We suppose that _**η**_ is constrained to lie in a compact subset
_D_ ⊂ R _[m]_ . The most convenient case, and the one which is relevant
to our examples, is when _D_ is a rectangle [∏] _i_ _[m]_ =1 [[] _[a][i][,][ b][i]_ []][. We refer to]
the set _D_ as the _auxiliary_ _parameter_ _space_, and the image _Im_ ( _F_ ) ⊂
R _[n]_ as the _primary_ _parameter_ _space_ of the model. [12]

Observe that any prior distribution _π_ on the auxiliary parameter space induces a distribution _F_ ∗ _π_ on the primary parameter
space defined by the formula ( _F_ ∗ _π_ )( _U_ ) = _π_ ( _F_ [−][1] _U_ ), for each
measurable _U_ ⊂ _Im_ ( _F_ ). [13]

This distribution is called the _pushforward_ _of_ _π_ _by_ _F_, or the
_implied_ _distribution_ _on_ _**θ**_, to use HW’s terminology. An equivalent


10 More precisely, the correspondence with our notation is given by
_gp_ ( _v, w_ ) := ( _v, A_ ( _p_ ) _w_ ).
11 Note that the letter _F_ was used previously in Section 2.3 to denote a smooth
map between manifolds. This notational reprisal is deliberate, as it will shortly
become apparent how the present usage of _F_ is equivalent to the previous usage.
12 Technical remark: Because _D_ is not an open set, some care is required in
saying what it means for the function _F_ to be ‘‘smooth’’. To be precise, we will
require that _F_ is continuous on _D_, and is smooth when restricted to the open
cube [∏] _i_ _[m]_ =1 [(] _[a][i][,][ b][i]_ [).]
13 Here, and in the rest of the paper, we use the notation _F_ −1 _U_ to denote
the preimage, defined by { _x_ ∈ _Domain_ ( _F_ ) : _F_ ( _x_ ) ∈ _U_ }. In case _U_ consists of a
single point _p_, we abuse notation slightly and denote the preimage by _F_ [−][1] ( _p_ ).
This should not be confused with the inverse function, which, in general, will
not exist.



14 A word on notation: We will make use of both equivalent notations ∫ _hµ_
and ∫ _h_ ( _x_ ) _µ_ ( _dx_ ) for the integral of a function _h_ against a measure _µ_, the latter
in cases in which one could possibly lose track of which variable corresponds to
which measure, and the former when there is little possibility for such confusion.
15 This need not be a probability measure, although it could be made so
by dividing by _Vprim_ ( _Im_ ( _F_ )) with no change to the following analysis. In what
follows, we do not need to assume the measures are normalized.
16 This means that at every point _p_ ∈ _M_, the differential _Jac_ ( _F_ ) _p_ : _TpM_ →
_TF_ ( _p_ ) _N_ is a surjective linear map.



**Coarea** **Formula** **1.** _Let_ _M_ _and_ _N_ _be_ _Riemannian_ _manifolds_ _and_
_F_ : _M_ → _N_ _a_ _surjective_ _submersion._ [16] _Then_ _for_ _any_ _measurable_
_function_ _f_ : _M_ → R _,_ _one_ _has_

∫ (∫ ) ∫



(∫



)



_JF_ ( _q_ ) _f_ ( _q_ ) _VM_ ( _dq_ )
_M_



∫
_VN_ ( _dp_ ) =



_N_



_fVF_ −1( _p_ )
_F_ [−][1] ( _p_ )


_S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_ 109



_where_ _JF_ _is_ _the_ _generalized_ _Jacobian_ _determinant._ [17] _defined_ _as_ _JF_ =
~~√~~ _det_ ( _Jac_ ( _F_ ) _Jac_ ( _F_ ) _[T]_ ) _,_ _det_ _is_ _the_ _usual_ _determinant_ _of_ _a_ _linear_ _oper-_

_ator,_ _and_ _VF_ −1( _p_ ) _is_ _the_ _measure_ _induced_ _on_ _the_ _preimage_ _F_ [−][1] ( _p_ )
_by_ _the_ _Riemannian_ _metric_ _on_ _M._ _The_ _formula_ _also_ _holds_ _when_
_dim_ ( _F_ [−][1] ( _p_ )) = 0 _,_ _provided_ _one_ _interprets_ _the_ _integrals_ _over_ _these_
_sets_ _as_ _discrete_ _sums._


For readers familiar with the usual calculus change of variables
formula, the Coarea formula is quite similar with the generalized
Jacobian determinant, _JF_ ( _q_ ), replacing the usual Jacobian determinant. See Appendix A.1 for more details on the generalized
Jacobian determinant.

This formula allows us to compute the pushforward (i.e., implied distribution) of any continuous distribution on the auxiliary parameter space, even if the mapping is neither one-to-one
or onto. To do this, fix a measurable function _h_ : _D_ → R
and consider the measure _hVaux_ . We shall compute the density



_d_ ( _F_ ∗( _hVaux_ ))



_dV_ ∗ _primaux_ .
Consider an arbitrary measurable function _f_ : _Im_ ( _F_ ) → R and
apply the Coarea formula to the function _f_ ( _F_ ( _**η**_ )) _h_ ( _**η**_ ) _/JF_ ( _**η**_ ) to see

∫ (∫ )



_Vprim_ ( _d_ _**θ**_ )



(∫



_f_ ( _F_ ( _**η**_ )) _h_ ( _**η**_ )



( _**η**_ )) _h_ ( _**η**_ ) )

_VF_ −1 _**θ**_ ( _d_ _**η**_ )
_JF_ ( _**η**_ )



_Im_ ( _F_ )

∫
=



_F_ [−][1] _**θ**_



_f_ ( _F_ ( _**η**_ )) _h_ ( _**η**_ ) _Vaux_ ( _d_ _**η**_ ) _._
_D_



Note that for each _**η**_ ∈ _F_ [−][1] _**θ**_, we have _f_ ( _F_ ( _**η**_ )) = _f_ ( _**θ**_ ) (by
definition of the inverse image). So we can pull this term out of
the integral to see that

∫ (∫ )



_Vprim_ ( _d_ _**θ**_ )



(∫
_f_ ( _**θ**_ )
_Im_ ( _F_ )



_F_ [−][1] _**θ**_



_h_ ( _**η**_ ) )

_VF_ −1 _**θ**_ ( _d_ _**η**_ )
_JF_ ( _**η**_ )



_h_ ( _**η**_ )



∫
= _f_ ( _F_ ( _**η**_ )) _h_ ( _**η**_ ) _Vaux_ ( _d_ _**η**_ ) _._

_D_



Since _f_ was arbitrary, we have established the following result.


**Main** **Formula** **1.** _The_ _pushforward_ _of_ _the_ _measure_ _hVaux_ _is_ _given_
_by_



way that each of these integrals is unchanged will have no effect
on the pushforward. In this way, Main Formulas 1 and 2 generalize the main results of HW. Indeed, even if the desired prior
on the primary parameter space were an arbitrary distribution,
i.e., not the uniform distribution, Main Formulas 1 and 2 describe
the relationship necessary for specifying the correct prior on the
auxiliary parameter space to obtain it.


_3.2._ _Special_ _cases_


The purpose of this section is to show how Main Formula 2
directly generalizes several familiar formulas, in particular, the
multivariate change of variables formula as well as the formula
for the area of a parametric surface.

In particular we will make the following simplifying assumptions for the remainder of this section: (1) _F_ is injective, and
(2), the ambient metric on the codomain R _[n]_ of _F_ is the standard
Euclidean metric. Observe that Assumption (1) requires that the
dimension of the auxiliary parameter space does not exceed the
dimension of the codomain. This assumption also implies that _F_
is a parametrization of the manifold _Im_ ( _F_ ).

We will make use of the formula for _JF_ derived in Appendix A.1.
Before we begin the calculations, let us observe that each preimage _F_ [−][1] ( _**θ**_ ) consists of a single point. Thus, strictly speaking,
these sets are not Riemannian manifolds, and the measure _VF_ −1 _θ_
is accordingly not defined. However, as we explained in the statement of the Coarea Formula, if _M_ is a single point, the formula
still holds provided that we define _VM_ so that _VM_ ( _M_ ) = 1. In
particular, under this convention, _Vol_ ( _F_ [−][1] _θ_ ) = _VF_ −1 _θ_ ( _F_ [−][1] _θ_ ) = 1
for each _θ_ . After we have finished our calculations, it will be
apparent that this is the only sensible choice.

To calculate _JF_, we endow _D_ with the standard Euclidean
metric. Consider now the bases _ei_ = _∂_ _**η**_ _i_ and _fi_ = _Jac_ ( _F_ ) _ei_ =
_∂_ _**ηi**_ _F_ of the tangent spaces _T_ _**η**_ _D_ and _TF_ ( _**η**_ ) _Im_ ( _F_ ). Note that both
the matrix representation _jac_ ( _e, f_ ) of _Jac_ ( _F_ ) and the Gram matrix
_G_ ( _e_ ) are equal to the identity matrix. [18] Therefore, the formula in
Appendix A.1 for _JF_ simplifies to



(∫
_F_ ∗( _hVaux_ )( _d_ _**θ**_ ) =

_F_ [−][1] ( _**θ**_ )



_h_ ( _**η**_ ) )

_VF_ −1( _**θ**_ )( _d_ _**η**_ )
_JF_ ( _**η**_ )



_Vprim_ ( _d_ _**θ**_ ) _._



_JF_ = √



_det_ ( _G_ ( _f_ )) _._



Applying the above formula to the function _h_ ( _**η**_ ) = _JF_ ( _**η**_ ) _g_ ( _F_ ( _**η**_ ))
_Vol_ ( _F_ [−][1] ( _F_ ( _**η**_ ))) [,]
and noting that the Riemannian volume of the inverse image is
given by _Vol_ ( _F_ [−][1] _**θ**_ ) = ∫ _F_ [−][1] _**θ**_ _[V][F]_ [−][1] _**[θ]**_ [(] _[d]_ _**[η]**_ [),] [we] [conclude] [the] [following]

result.



**Main** **Formula** **2.** _For_ _any_ _measurable_ _function_ _g_ : _Im_ ( _F_ ) → R _,_ _the_
_measure_

_JF_ ( _**η**_ ) _g_ ( _F_ ( _**η**_ ))
_µ_ ( _d_ _**η**_ ) = _Vol_ ( _F_ [−][1] ( _F_ ( _**η**_ ))) _Vaux_ ( _d_ _**η**_ ) _,_

_satisfies_ _F_ ∗ _µ_ = _gVprim._


If the denominator looks strange, we remind the reader that
in general, _F_ [−][1] ( _F_ ( _η_ )) ̸= _η_, as _F_ need not be one-to-one.

Finally, we observe that if _F_ is not one-to-one, then there are
infinitely many measures _µ_ such that _F_ ∗ _µ_ = _fVprim_ . Indeed, we
see from Main Formula 1 that the Radon–Nikodym derivative



Recall that _G_ ( _f_ ) _ij_ = ( _fi, fj_ ) _Im_ ( _F_ ). However, we assumed that the
metric on _Im_ ( _F_ ) was the restriction of the Euclidean metric on the
ambient space, so we have _G_ ( _f_ ) _ij_ = ( _fi, fj_ ) where _fi_ and _fj_ are now
considered as vectors in R _[n]_, and (· _,_ ·) denotes the standard dot
product. Therefore, we obtain the following result.


**Main** **Formula** **3.** _Under_ _the_ _assumptions_ (1) _and_ (2) _at_ _the_ _begin-_
_ning_ _of_ _Section_ 3.2 _,_ _we_ _have_



_JF_ ( _**η**_ ) = √



_det_ ( _G_ ({ _∂_ _**ηi**_ _F_ } _i_ ))



_where_ _G_ ({ _∂_ _**ηi**_ _F_ } _i_ ) _denotes_ _the_ _Gram_ _matrix_ _of_ _the_ _vectors_ { _∂_ _**ηi**_ _F_ } _i,_ _as_
_defined_ _in_ _Section_ 2.4 _._


Now, let us further specialize to the case when _m_ = _n_ = _Im_ ( _F_ )
(i.e, _F_ is locally invertible). Consider the classical Jacobian matrix
_jacij_ = _∂_ _**η**_ _j_ _Fi_ . We observe that


∑
( _∂_ _**ηi**_ _F_ _, ∂_ _**ηj**_ _F_ ) = _∂_ _**ηi**_ _Fk∂_ _**ηj**_ _Fk_

_k_


∑
= ( _jac_ ) _[t]_ _ik_ [(] _[jac]_ [)] _[kj]_


_k_

= (( _jac_ ) _[t]_ _jac_ ) _ij._


18 The general definitions of _Jac_ ( _F_ ) and _G_ are given in Appendix A.4 and
Section 2.4, respectively.



_d_ ( _F_ ∗( _hVaux_ ))



_h_ ( _**η**_ )
_F_ [−][1] _θ_ _JF_ ( _**η**_ ) _[V][F]_ [−][1] _[θ]_ [(] _[d]_ _**[η]**_ [)] [depends] [only] [on] [the] [integral]



_FdV_ ∗( _hVprimaux_ )) ( _θ_ ) = ∫



of _h/JF_ over the sets _F_ [−][1] ( _**θ**_ ), so changing the function _g_ in such a



17 The transpose here is taken with respect to the inner products induced
on the tangent spaces by the metric, that is ( _Jac_ ( _F_ ) _v, w_ ) _N_ = ( _v, Jac_ ( _F_ ) _[T]_ _w_ ) _M_ for
all _v_ and _w_ . Note that for a fixed choice of bases, the matrix of _Jac_ ( _F_ ) _[T]_ is
not necessarily equal to the transpose of the matrix of _Jac_ ( _F_ ). This is discussed
further in Appendix A.1.


110 _S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_



giving us the matrix equality _G_ ( _f_ ) = ( _jac_ ) _[t]_ _jac_ . Since _m_ = _n_, _jac_
is a square matrix. Recalling basic properties of determinants, we
obtain

~~√~~ √



√
_det_ ( _G_ ( _f_ )) =



_det_ ( _jac_ ) [2] = | _det_ ( _∂_ _**η**_ _j_ _Fi_ )| _._



By Main Formulas 2 and 3, we have shown


_F_ ∗(| _det_ ( _∂iFj_ )( _**η**_ )| _d_ _**η**_ ) = _d_ _**θ**_ _._



Recalling the definition of the pushforward measure, this is equivalent to saying that for any measurable _g_ : _Im_ ( _F_ ) → R, we
have

∫ ∫



∫
_g_ ( _**θ**_ ) _d_ _**θ**_ =
_Im_ ( _F_ )



_g_ ( _F_ ( _**η**_ ))| _det_ ( _∂iFj_ )( _**η**_ )| _d_ _**η**_ _,_
_D_



which is nothing other than the standard change of variables
formula. This special case coincides with the situation considered
by HW, albeit using our notation.

The next case of interest is when _m_ = 2 _, n_ = 3.
In this case, _G_ ( _f_ ) is a 2 × 2 matrix, so we can simply compute
the determinant:


_det_ ( _G_ ( _f_ )) = _G_ ( _f_ )11 _G_ ( _f_ )22 − _G_ ( _f_ )12 _G_ ( _f_ )21

= | _∂_ _**η**_ **1** _F_ | [2] | _∂_ _**η**_ **2** _F_ | [2]   - ( _∂_ _**η**_ **1** _F_ _, ∂_ _**η**_ **2** _F_ ) [2]

2
= ⏐⏐ _∂η_ 1 _F_ × _∂η_ 2 _F_ ⏐⏐ _,_


where × is the usual cross product in R [3] .

Thus we obtain


_F_ ∗(⏐⏐( _∂η_ 1 _F_ × _∂η_ 2 _F_ )( _**η**_ )⏐⏐ _d_ _**η**_ ) = _Vprim,_


where _Vprim_ is the measure on _Im_ ( _F_ ) induced by the Euclidean
metric on R [3] . This is nothing other than the formula for the area
of a parametric surface.

Finally, although not immediately relevant to our purposes, it
is also interesting to note that Fubini’s Theorem can be deduced
directly from the Coarea formula simply by letting _D_ = R [2] and
_F_ ( _x, y_ ) = _x_ .


_3.3._ _Discrete_ _distributions_


It is worth noting that the same procedure in Section 3.1 can
be carried out for discrete distributions as well. That is, assume _D_
is a finite set and let _F_ : _D_ → R be a function. Then the preimages
_F_ [−][1] _y_ are disjoint sets and ∪ _y_ ∈ _Im_ ( _F_ ) _F_ [−][1] _y_ = _D_ . One immediately
deduces the following.


**Coarea** **Formula** **2.** _For_ _any_ _function_ _f_ : _D_ → R _,_ _and_ _measure_ _π_
_on_ _D,_ _one_ _has_



_3.4._ _Discrete_ _approximations_


The advantage in simplicity of the discrete formula over the
continuous analogue is obvious. In practice, one may employ a
scheme such as the following to approximate a continuous computation by a discrete one. Further, such approximations could
be applied when the model in question does not have an analytic
expression and must be simulated.

Suppose we are given a continuous measure _hVaux_ on the
auxiliary space and we wish to compute the pushforward.

Let us divide the range _Im_ ( _F_ ) into a finite set F of disjoint
(full-dimensional) subsets which cover _Im_ ( _F_ ). We consider the
measure _π_ : F → R, _π_ ( _X_ ) = ( _F_ ∗( _hVaux_ ))( _X_ ). By definition of the
pushforward, we see that


∫
_π_ ( _X_ ) = _h_ ( _**η**_ ) _Vaux_ ( _d_ _**η**_ ) _._

_F_ [−][1] _X_


We have made some progress because now instead of having
to integrate over the zero-measure sets _F_ [−][1] _**θ**_, each with its own
measure _VF_ −1 _**θ**_, we need only integrate over finitely many positive
measure sets _F_ [−][1] _X_, all with the same known measure _Vaux_ . These
integrals can in turn be approximated by a straightforward Monte
Carlo procedure. Specifically, we maintain an array [ _nX_ ] _X_ ∈F and
initialize _nX_ = 0 for each _X_ . We then take successive samples _**η**_ _i_
from the distribution _hVaux_ . For each sample, we determine which
set _X_ contains _F_ ( _**η**_ _i_ ) and then increase _nX_ by 1. After any number
of samples, the estimate _π_ ˜ ( _X_ ) for _π_ ( _X_ ) is simply

_nX_
_π_ ˜ ( _X_ ) = _._
∑

_X_ ∈F _[n][X]_


As a basic consistency check, one may ask about the quality
of the approximation as the discretization gets finer, in the sense
that sup _X_ ∈F _Diam_ ( _X_ ) → 0, where _Diam_ ( _X_ ) is the maximal distance
between any two points of _X_ . Choose a sample point _xX_ ∈ _X_
for each _X_ ∈ F and consider the measure _νX_ = [∑] _X_ ∈F _[δ][x]_ _X_ _[π]_ [(] _[X]_ [).]

One would hope that this measure converges to the true measure _F_ ∗( _hVaux_ ) in some sense. In particular, one might hope for
convergence in distribution; that is that ∫ _fdνX_ → ∫ _fd_ ( _F_ ∗( _hVaux_ ))
for all bounded measurable _f_ . Unfortunately this does not hold
in general. Indeed, if _Im_ ( _F_ ) = R and _F_ ∗( _hVaux_ ) is the Lebesgue
measure, then this would imply that the Riemann integral of
every measurable function is equal to its Lebesgue integral, which
is false. However, we do get convergence if we instead assume
that _f_ is continuous (or even that the set of discontinuities is
countable), which is probably sufficient for most applications. [19]

In anticipation of Section 4, we note that the result of this
procedure can also be used to construct an approximation to the
uniform measure on _Im_ ( _F_ ). To do this, we set F _>_ 0 = { _X_ ∈ F :

_π_ ˜ ( _X_ ) _>_ 0}. Then consider the measure _ν_ : F → R where _ν_ ( _X_ ) is
given by



_ν_ ( _X_ ) =



∑



∑ ∑

_f_ ( _x_ ) _π_ ( _x_ ) =

_x_ ∈ _D_ _y_ ∈ _Im_ (



_f_ ( _x_ ) _π_ ( _x_ )



⎞


⎠ _._



{ _VIm_ ( _F_ )( _X_ )
∑ _Y_ ∈F _>_ 0 _[V][Im]_ [(] _[F]_ [)][(] _[Y]_ [)] _X_ ∈ F _>_ 0 _,_

0 otherwise _._



_y_ ∈ _Im_ ( _F_ )



⎛

⎝ [∑]

_x_ ∈ _F_ [−][1] _y_



Arguing as before, we apply the formula to the function _g_ ( _F_ )
for some function _g_ : _Im_ ( _F_ ) → R to see



⎞


⎠ _._



∑



∑ ∑

_g_ ( _F_ ( _x_ )) _π_ ( _x_ ) =

_x_ ∈ _D_ _y_ ∈ _Im_ (



_g_ ( _y_ )

_y_ ∈ _Im_ ( _F_ )



⎛

⎝ [∑] _π_ ( _x_ )

_x_ ∈ _F_ [−][1] _y_



Taken together, we obtain the following result.



**Main** **Formula** **4.** _In_ _the_ _above_ _setting,_ _the_ _pushforward_ _F_ ∗ _π_ _is_
_given_ _by_ ( _F_ ∗ _π_ )( _y_ ) = [∑] _x_ ∈ _F_ [−][1] _y_ _[π]_ [(] _[x]_ [)] _[for]_ _[y]_ ∈ _Im_ ( _F_ ) _._ _Moreover,_



_given_ _by_ ( _F_ ∗ _π_ )( _y_ ) = [∑] _x_ ∈ _F_ [−][1] _y_ _[π]_ [(] _[x]_ [)] _[for]_ _[y]_ ∈ _Im_ ( _F_ ) _._ _Moreover,_

_given_ _any_ _measure_ _ν_ _on_ _Im_ ( _F_ ) _,_ _we_ _have_ _F_ ∗ _π_ = _ν_ _where_ _π_ ( _x_ ) =



This converges to _VIm_ ( _F_ ) in the same sense as before.

Of course, we still have to calculate _VIm_ ( _F_ )( _X_ ) for each such _X_ .
If _Im_ ( _F_ ) is not full-dimensional in the codomain, this might be
challenging. But if _Im_ ( _F_ ) is full-dimensional (i.e., _F_ is full-rank),
then we can simply choose F to be a collection of small boxes, in
which case the computation of _VIm_ ( _F_ )( _X_ ) is trivial (assuming the
ambient metric is Euclidean).


19 Indeed, it is not hard to show that, for any probability measure _µ_, and
continuous, compactly-supported function _f_ the integral ∫ _fdµ_ is the limit of
any sequence of Riemann sums for which the maximum diameter of any set in
the partition tends to zero.



_ν_ ( _F_ ( _x_ ))
_Cardinality_ ( _F_ [−][1] _F_ ( _x_ )) _[.]_


_S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_ 111



We have included an interactive demonstration of this technique in a Jupyter Notebook in the online supplement. [20]

Finally, we remark that this technique becomes intractable
for ambient spaces of sufficiently large dimension. Indeed, a
_d_ -dimensional grid containing only 10 bins per dimension would
have a total of 10 _[d]_ boxes. Nonetheless, this procedure can be put
to good use when the dimensions of the spaces in question are
not too large, as we will see in our re-analysis of the SI model.


**4.** **Statistical** **inference**


_4.1._ _Computation_ _of_ _evidence_


Let us return to the setup of Section 3.1, in which we have
a functional relation of the form _**θ**_ = _F_ ( _**η**_ ). Equivalently, the
conditional distribution _p_ ( _**θ**_ | _**η**_ ) is a delta function _δF_ ( _**η**_ )( _**θ**_ ) centered at _F_ ( _**η**_ ). We will now assume in addition that we have
a likelihood function _L_ _**y**_ ( _**θ**_ ) = _P_ ( _**y**_ | _**θ**_ ), where _**y**_ is the observed
data. The motivating example is when _**θ**_ is a vector of success
probabilities for a collection of binary decision problems, and _L_
is the standard binomial likelihood. For a fixed prior _π_ on _**η**_, the
Bayesian evidence of _**y**_ is by definition _Pπ_ ( _**y**_ ). To express this in
terms of _L_, we first marginalize over _**η**_, and then apply a further
marginalization to _P_ ( _**y**_ | _**η**_ ), in order to express it in terms of _**θ**_ .
Thus:



Returning to the setting of Section 3.1, consider a prior distribution _π_ on the parameters _**η**_ ∈ R _[m]_, a smooth function _F_ ( _**η**_ ) ∈
R _[n]_, and a likelihood function _L_ _**y**_ ( _**θ**_ ) _,_ _**θ**_ ∈ R _[n]_ . The Prior Predictive
Distribution _PP_ ( _**y**_ ) is by definition the likelihood of _**y**_ averaged
over the prior. Thus, we have [21]



∫
_PP_ ( _**y**_ ) = _L_ _**y**_ ( _F_ ( _**η**_ )) _π_ ( _d_ _**η**_ ) _._

R _[m]_

The pushforward _F_ ∗ _π_ can also be expressed in a similar way.
Consider the function _M_ _**θ**_ ( _**η**_ ) = _δF_ ( _**η**_ )( _**θ**_ ). It is straightforward to
verify that


∫
( _F_ ∗ _π_ )( _d_ _**θ**_ ) = _M_ _**θ**_ ( _**η**_ ) _π_ ( _d_ _**η**_ ) _._

R _[m]_

We see that _PP_ and _F_ ∗ _π_ are both averages of appropriate
functions with respect to the prior _π_ . They differ in the stage at
which the average is taken. While the measure PP averages over
both of the steps _**η**_ → _**θ**_ and _**θ**_ → _**y**_, the measure _F_ ∗ _π_ averages
only over the step _**η**_ → _**θ**_ and cannot ‘‘see’’ the data-generating
step _**θ**_ → _**y**_ . Thus, the pushforward is appropriate for studying
relations between the quantities _**η**_ and _**θ**_ which are purely internal
to the theory, while the PP is appropriate if we are also interested
in the process by which the data are generated, which is codified
by the function _Ly_ .

Finally, there is a precise mathematical relation between _PP_
and _F_ ∗ _π_ . In words, _PP_ is obtained by averaging the likelihood _L_
with respect to _F_ ∗ _π_ . That is,



∫
_Pπ_ ( _**y**_ ) =


∫
=


∫
=


∫
=


∫
=



_P_ ( _**y**_ | _F_ ( _**η**_ )) _π_ ( _d_ _**η**_ )
_D_


_L_ _**y**_ ( _F_ ( _**η**_ )) _π_ ( _d_ _**η**_ ) _._
_D_



_P_ ( _**y**_ | _**η**_ ) _π_ ( _d_ _**η**_ )
_D_



)
_P_ ( _**y**_ | _**θ**_ ) _P_ ( _**θ**_ | _**η**_ ) _d_ _**θ**_ _π_ ( _d_ _**η**_ )


)
_P_ ( _**y**_ | _**θ**_ ) _δF_ ( _**η**_ )( _**θ**_ ) _d_ _**θ**_ _π_ ( _d_ _**η**_ )



_D_


_D_



(∫


(∫



By the change of variables formula, we have


∫
_Pπ_ ( _**y**_ ) = _L_ _**y**_ ( _**θ**_ )( _F_ ∗ _π_ )( _d_ _**θ**_ ) _._

_Im_ ( _F_ )


Thus we see the importance of the pushforward measure _F_ ∗ _π_
for the purposes of evidence computation. The above formula
shows that the evidence can only ‘‘see’’ the measure _F_ ∗ _π_, and not
the measure _π_ itself. Accordingly, it is imperative to understand
the pushforward of a given measure _π_, which is given by our
Main Formula 1. Conversely, it is sometimes the case that we
want to set a prior _ν_ directly on _Im_ ( _F_ ), but that the geometry of
the space _Im_ ( _F_ ) is sufficiently complex to make a direct integral
over the space difficult to compute, while the geometry of _D_ is
trivial (as in our examples to follow). In this case, one would
want to instead carry out the integral in terms of _**η**_ . To do so,
one must use a prior _π_ with the correct pushforward _F_ ∗ _π_ = _ν_ .
Doing so ensures that one gets the same answer as if the integral
were computed directly over the space _Im_ ( _F_ ); hence, our Main
Formula 2.


_4.2._ _Relation_ _to_ _predictive_ _distributions_


A reader with a background in Bayesian statistics might wonder about the relation between the pushforward measure _F_ ∗ _π_ and
the Prior Predictive Distribution.


20 [https://figshare.com/articles/Segert_Davis-Stober_2018_Supplementary_](https://figshare.com/articles/Segert_Davis-Stober_2018_Supplementary_Material_2/7289642)
[Material_2/7289642.](https://figshare.com/articles/Segert_Davis-Stober_2018_Supplementary_Material_2/7289642)



∫
_PP_ ( _**y**_ ) = _L_ _**y**_ ( _**θ**_ )( _F_ ∗ _π_ )( _d_ _**θ**_ ) _._

R _[n]_


This can be seen by applying the change of variables formula to
the defining equation for _PP_ .


**5.** **Example** **using** **the** **selective** **integration** **model**


In this section, we present an illustration of prior transformation using the Selective Integration (SI) model of Tsetsos et al.
(2016a, 2016b). Our motivation for this example is twofold. As
a qualitative generalization of HW, the SI model does not place
_explicit_ order-constraints upon choice probabilities. We use the
term ‘‘explicit’’ in the sense that it is not straightforward to
describe the set of viable binary choice probabilities defined by
the SI model as a set of non-linear inequality constraints. In other
words, the allowable set of binary choice probabilities forms a
smooth manifold in the appropriate probability space and it is not
straightforward to geometrically describe this manifold. While
this, and other models of choice, are not explicitly described as
(potentially non-linear) constraints on parameters, the general
prior transformation insights by HW hold to these cases as well.
Secondly, we consider three versions of SI (the same versions
considered by Tsetsos et al., 2016a, 2016b). All three versions of
SI have the property that the corresponding mapping between
auxiliary and primary parameter spaces is not one-to-one. One
version of SI we consider, with no selective gating, admits a
closed-form solution for the adjusted prior in terms of the auxiliary parameters. The other two versions do not, and therefore
provide an illustration of our computational methods via discrete
approximation.

We re-analyze all choice data from Experiment 4 in Tsetsos
et al. (2016a, 2016b) for three versions of SI under both ‘adjusted’
and ‘unadjusted’ priors. We show that prior transformation has a
major impact on the resulting Bayes factors for these data.


21 Note that although this bears a striking resemblance to the formula for the
evidence of the result _**y**_, it is not the same thing. While the Evidence is calculated
for a specific empirical result _**y**_ (and thus is a real number), the Prior Predictive
Distribution is instead a probability distribution over the space of all possible
results. Evaluating this distribution (or its density, in the case of a continuous
distribution) at the point _**y**_ gives the evidence of _**y**_ .


112 _S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_



_5.1._ _Selective_ _integration_ _definition_


Tsetsos et al. (2016a, 2016b) presented a computational model
of choice termed the _selective_ _integration_ model. This model is
defined recursively as a sequential sampling process. Said simply,
decision makers repeatedly sample information about sets of
presented choice alternatives. This information is then integrated
via accumulator functions. Cognitive as well as perceptual noise
are explicitly modeled within this framework. A key aspect to
the model is that choice alternative information from an inferior
alternative is systematically downweighted relative to the superior alternative, i.e., selectively integrated. For example, if two
plane tickets have the same price for the same destination but
one ticket is first class then the economy class information for
the second ticket will be downweighted further as a ‘‘loss’’.

The following definition specifies the selective integration
model for a binary forced choice between two options, _A_ and
_B_ . Let _S_ denote a finite set of choice alternatives. Let _t_ denote
the current discrete time step. Let _sA_ ( _t_ ) _, sB_ ( _t_ ) denote stimulus
intensities (e.g., bar heights in pixels) at _t_, and let _**ρ**_ _A_ ( _t_ ) _,_ _**ρ**_ _B_ ( _t_ ) be
standard normal random variables independent of each other and
independent across different values of _t_ . Let _**X**_ _A_ ( _t_ ) _,_ _**X**_ _B_ ( _t_ ) denote
the corrupted subjective representations of the stimuli, due to
_(early)_ _noise_ with noise parameter _σ_, defined by


_**X**_ _A_ ( _t_ ) = _σ_ - _**ρ**_ _A_ ( _t_ ) + _sA_ ( _t_ ) _,_ _**X**_ _B_ ( _t_ ) = _σ_ - _**ρ**_ _B_ ( _t_ ) + _sB_ ( _t_ ) _._


Let _w_ ∈[0 _,_ 1] be a _selective_ _gating_ _parameter_ and define a _gain_
_function ψ_ as the step-function


{ 1 _,_ if _x >_ 0 _,_
_ψ_ ( _x_ ) =
_w_ if _x <_ 0 _._


Define


_**I**_ _A_ ( _t_ ) = _ψ_ ( _**X**_ _A_ ( _t_ )− _**X**_ _B_ ( _t_ ))· _**X**_ _A_ ( _t_ ) _,_ _**I**_ _B_ ( _t_ ) = _ψ_ ( _**X**_ _B_ ( _t_ )− _**X**_ _A_ ( _t_ ))· _**X**_ _B_ ( _t_ ) _,_


to serve as momentary inputs to _accumulators_ _**Y**_ _A_ ( _t_ ) and _**Y**_ _B_ ( _t_ )
that integrate the attribute values of the two sequences _A_ and
_B_ according to the difference equations


_**Y**_ _A_ ( _t_ ) = (1 − _λ_ ) · _**Y**_ _A_ ( _t_ - 1) + _**I**_ _A_ ( _t_ ) + _ξ_ - _**ζ**_ _A_ ( _t_ ) _,_

_**Y**_ _B_ ( _t_ ) = (1 − _λ_ ) · _**Y**_ _B_ ( _t_ - 1) + _**I**_ _B_ ( _t_ ) + _ξ_ - _**ζ**_ _B_ ( _t_ ) _,_

with _λ_ an _integration leak_, _ξ_ _late noise_, [22] and let _**ζ**_ _A_ ( _t_ ) _,_ _**ζ**_ _B_ ( _t_ ) be two
standard normal random variables independent of each other and
across different values of _t_ . The accumulators are initialized as


_**Y**_ _A_ (0) = _**Y**_ _B_ (0) = 0 _._


At the end of the presentation of stimuli, the model chooses
sequence A if _**Y**_ _A_ ( _t_ ) _>_ _**Y**_ _B_ ( _t_ ), chooses sequence B if _**Y**_ _B_ ( _t_ ) _>_ _**Y**_ _A_ ( _t_ ),
and chooses either with equal probability if _**Y**_ _A_ ( _t_ ) = _**Y**_ _B_ ( _t_ ).


_5.2._ _Base_ _model_ _specifications_


We will analyze three different versions of the SI model described above, each with two different prior distributions. By a
‘‘version’’, we simply mean a specification of a subset of the fourdimensional auxiliary parameter space to which the model is
constrained, leaving the prior distribution unspecified.

The first such version is simply the full model from the previous section, with an additional constraint on the maximum value
of the parameters. This is done to make it possible to define a
prior which is uniform over the parameters of the model. The
bounds 0 ≤ _λ, w_ ≤ 1 are required by the model, but _σ_ and
_ξ_ can a priori take any positive value. We impose the bounds
0 ≤ _σ, ξ_ ≤ 20. The second version is obtained from the first


22 We assume 0 ≤ _ξ_ ∈ R.



simply by requiring _σ_ = 0 (no early noise), leaving the other
bounds unchanged. The final version is obtained from the first by
requiring _w_ = 1 (no selective gating), again leaving the other
bounds unchanged. For the first two of the above models, we
estimated the value of the Bayesian evidence for the prior which
is uniform over the corresponding primary parameter space using
the procedure outlined in Section 3.4. We used a 50 × 50 × 50
grid on the image space [0 _,_ 1] [3] of choice probabilities, and 10
million samples from the auxiliary parameter space. To estimate
the vector of probabilities corresponding to each such sample, we
used a Monte Carlo simulation with 10000 iterations.

In the case of _w_ = 1, as we will see in the next section,
an exact formula for the probabilities is available, as well as
for the prior distribution which implies a uniform distribution
on the primary parameter space. Therefore, the evidence values
can be expressed as integrals over the parameters of known
functions (i.e., the product of the binomial likelihood with the
prior density), which we estimated using a simple Monte Carlo
algorithm, again with 10 million iterations.

The primary parameter spaces of the first two models can be
approximated by simulation; that is, we randomly select a point
in the auxiliary parameter space, and then plot the corresponding
3-tuple of probabilities. A consequence is that the spatial density
of sample points in the primary parameter space approximates
the pushforward of the uniform measure. The results are shown
in Fig. 1.


_5.3._ _Prior_ _specification_ _for_ _restricted_ _selective_ _integration_


In case _w_ = 1, Davis-Stober, Brown, Park, and Regenwetter
(2017) showed that the choice probabilities _PAB, PBC_ _, PAC_ can be
expressed explicitly in terms of the remaining three parameters.
We denote by _F_ the corresponding function. Thus:


( _PAB, PBC_ _, PAC_ ) = _F_ ( _λ, σ, ξ_ ) _,_


see Davis-Stober et al. (2017) for the explicit formula.

We are therefore in the situation of Section 3.1, in which the
value of a parameter vector is given as a function of another
parameter vector. In fact, the variables _ξ_ and _σ_ appear only in the
combination √ _ξ_ [2] + _σ_ [2], so we define the variable _α_ = ~~√~~ _σ_ [2] + _ξ_ [2] .



combination √ _ξ_ [2] + _σ_ [2], so we define the variable _α_ = ~~√~~ _σ_ [2] + _ξ_ [2] .

As in the previous section, we used the bounds 0 ≤ _λ_ ≤ 1 _,_ 0 ≤
_σ, ξ_ ≤ 20. [23]

To conclude this section, we illustrate visually the difference
between the two prior distributions (i.e., uniform over auxiliary
parameter space vs. uniform over primary parameter space).

To sample from the primary parameter space when the auxiliary parameters are uniformly distributed, we can simply sample
from the auxiliary space, then apply the function _F_ to obtain a
point on the primary parameter space. The distribution of the
points so obtained approximates the pushforward of the uniform
prior by _F_ . The results are shown in Fig. 2.

By contrast, we now construct a prior _ν_ 1 on the parameters
_λ_ and _α_ which implies a uniform distribution on the primary
parameter space; thus _F_ ∗ _ν_ 1 = _Vprim_ . This can be done. [24] with


23 The number 20 for the bounds on the _ξ_ and _σ_ parameters is somewhat
arbitrary, and, unlike the bound on _λ_, does not reflect a theoretical constraint.
In fact, the form of the adjusted prior can provide guidance on such parameter
bounds. For example, the sharp peak in the density of Fig. 3 suggests that the
choice of cutoff will not dramatically affect the shape of the transformed prior,
provided that the cutoff is sufficiently large that it encompasses most of the mass
of the peak. Bear in mind that this figure was constructed to have a uniform
implied distribution on the primary parameter space. Other transformed priors
may have very different forms and the bounds on _ξ_ and _σ_ may matter for such
cases.
24 In fact, the hypotheses under which these formulas were derived are
not strictly met, because the Jacobian determinant is not positive everywhere.
However, the prior distribution constructed in this way will still have the
required property, because it will, by construction, assign zero probability to
the region on which the determinant vanishes.



_ξ_ [2] + _σ_ [2], so we define the variable _α_ = ~~√~~


_S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_ 113


**Fig.** **1.** Samples from the SI primary parameter space (identity stimulus order)



Main Formulas 2 and 3. The density is shown in Fig. 3. Intuitively,
the value of this density describes the amount that _F_ ‘‘stretches’’
a small neighborhood around that point in the auxiliary space.
A large value of the density means that a small neighborhood
around the point covers a disproportionately large area in the
primary parameter space.

For comparison, we turn to _[d]_ _dV_ [(] _[F]_ _prim_ [∗] _[ν]_ [2][)] [, i.e., the density of the push-]

forward measure, where _ν_ 2 is uniform over the three remaining
model parameters. This is done in two steps. Consider the function _H_ ( _λ, α_ ), which is obtained from _F_ by replacing √ _σ_ [2] + _ξ_ [2]

with _α_, and _G_ ( _λ, σ, ξ_ ) = ( _λ,_ √ _σ_ [2] + _ξ_ [2] ). By construction _F_ =

_H_ - _G_ . We make use of the formula



∗( _G_ ∗ _ν_ 2)| _C_ ) _πα_

( _H_ ( _λ, α_ )) =
_dVprim_ 2 _JH_ (



For comparison, we turn to _[d]_ [(] _[F]_ [∗] _[ν]_ [2][)]



_σ_ [2] + _ξ_ [2]



To summarize, a vanishing Jacobian determinant over a nonnull set implies the pushforward measure is singular. Dividing
up the auxiliary parameter space according to whether or not
the Jacobian determinant vanishes has the effect of decomposing
the pushforward into the sum of a continuous measure and an
absolutely singular measure. We will now use Main Formulas 1
and 2 to compute the density of the continuous part.

Because _H_ | _C_ is one-to-one, each preimage _H_ [−][1] ( _p_ ) _, p_ ∈[0 _,_ 1] [3]

consists of a single point (or is empty). Accordingly, it follows
from Main Formula 1 and the expression for _G_ ∗ _ν_ 2 that



with _α_, and _G_ ( _λ, σ, ξ_ ) = ( _λ,_ √ _σ_ [2] + _ξ_ [2] ). By construction _F_ =

_H_ - _G_ . We make use of the formula


( _H_ - _G_ )∗ _ν_ 2 = _H_ ∗( _G_ ∗ _ν_ 2) _,_


which is a general property of the pushforward, and not specific
to this situation.

Let us ignore the bounds _σ, ξ_ ≤ 20 for now (i.e., we take _ν_ 2 to
be an improper prior). We will explain how to incorporate them
shortly.

A straightforward calculation shows the Jacobian matrix is
given by



~~√~~
values of _α_, and zero if _α_ ≥



_d_ ( _H_ ∗( _G_ ∗ _ν_ 2)| _C_ )



2 _JH_ ( _λ, α_ ) _[,]_



provided ( _λ, α_ ) ∈ _C_ . Therefore, the density of the continuous part
of _F_ ∗ _π_, evaluated at _H_ ( _λ, α_ ), is given by 2 _JHπα_ ( _λ,α_ ) [.]
To incorporate the bound _σ, ξ_ ≤ 20, the only difference is that
_πα/_ 2 must be replaced with _Vol_ ({ _σ_ [2] + _ξ_ [2] = _α,_ 0 ≤ _σ, ξ_ ≤ 20}),
which is geometrically the intersection of a quarter-circle with
a square. Accordingly, the value of this volume remains _πα/_ 2
provided that _α_ ≤ 20, but is smaller than _πα/_ 2 for larger [27]



)



_JacG_ ( _λ, σ, ξ_ ) =



( 1 0 0
0 ~~√~~ _σ_ ~~√~~ _ξ_



_σ_ ~~√~~ _ξ_

_σ_ [2] + _ξ_ [2] _σ_ [2]



_,_



_σ_ [2] + _ξ_ [2]



and _JG_ [2] [is] [simply] [the] [determinant] [of] [the] [product] [of] [this] [matrix]
with its transpose; thus _JG_ = 1.

By Main Formula 1, we see that ( _G_ ∗ _ν_ 2)( _dλ, dα_ ) = _Vol_ ( _G_ [−][1] ( _λ,_
_α_ )) _dλdα_ . Recalling that _σ_ and _ξ_ were constrained to be positive,
we see that _G_ [−][1] ( _λ, α_ ) is geometrically a quarter-circle with radius
_α_, so has volume _πα/_ 2. Thus ( _G_ ∗ _ν_ 2)( _dλ, dα_ ) = ( _πα/_ 2) _dλdα_ .

We next turn to the function _H_ . By the calculations in Section 3.2, _JH_ is given by the magnitude of the cross product of
the columns of the corresponding Jacobian matrix. Note that the
value of _JH_ is precisely what is plotted in Fig. 3, modulo a constant
multiplicative factor.

As can be seen in Fig. 3, there are regions of the auxiliary
parameter space in which the Jacobian determinant of _H_ vanishes, making direct application of Main Formula 1 problematic.
To proceed, we partition the space into two disjoint regions _C_ =
{ _JH_ _>_ 0} and _D_ = { _JH_ = 0}. We may write [25] _G_ ∗ _ν_ 2 = ( _G_ ∗ _ν_ 2)| _C_ +
( _G_ ∗ _ν_ 2)| _D_, and _H_ ∗( _G_ ∗ _ν_ 2) = _H_ ∗(( _G_ ∗ _ν_ 2)| _C_ ) + _H_ ∗(( _G_ ∗ _ν_ 2)| _D_ ) Then since
_JH_ _>_ 0 on _C_, we can apply the Main Formulas to compute the
pushforward of ( _G_ ∗ _ν_ 2)| _C_ . The measure _H_ ∗(( _G_ ∗ _ν_ 2)| _D_ ) is absolutely
singular, which means it is supported on a null set of _Vprim_ . [26]


25 Recall that the restriction of a measure _µ_ to a measurable subset _A_ is
defined by _µ_ | _A_ ( _X_ ) = _µ_ ( _A_ ∩ _X_ ).
26 More abstractly, partitioning the auxiliary parameter space thusly gives the
Lebesgue decomposition of the pushforward.



values of _α_, and zero if _α_ ≥ 2 ∗ 20. In addition, one must

multiply by a normalizing factor to ensure that _ν_ 2 integrates to 1.
Since exact knowledge of the density of _F_ ∗ _ν_ 2 is not required for
our analysis, we content ourselves with the computation for the
improper version of _ν_ 2, and leave the details of the extension to
the case where _ν_ 2 has bounded support to the interested reader.

As for the singular part, numerical simulations suggest that
it is supported along the ‘‘ridge" at the bottom of the primary
parameter space, as visible in Fig. 2.


_5.4._ _Data_ _illustration_


In the previous sections, we described the 3 base models, and
the two priors used for each model. For each combination of the
21 subjects and 6 stimuli orderings, we computed the Bayesian
evidence for each of the 6 models. Thus, there were 3 × 2 ×
21 × 6 = 756 values in total. In addition, for each subject and
permutation combination, we computed the evidence for the ‘‘encompassing model" in which ( _PAB, PBC_ _, PAC_ ) is chosen uniformly
at random from the unit cube. Since the likelihood function is a
product of binomials, it is easy to check that the evidence for this
model is a product of beta functions and binomial coefficients.
Using this as our comparison model, we obtained Bayes factors
for each of the 6 models, 21 subjects, and 6 stimuli orderings. See
also Heck, Hilbig, and Moshagen (2017) for a similar approach.



~~√~~
27 The exact value for the case 20 ≤ _α_ ≤ 2 ∗ 20 may be worked out with

simple calculus.


114 _S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_


As is demonstrated by the figure, adjusting the priors can
make a difference in the overall performance of the model. As
can be seen from the figure, the strength of evidence in the
_σ_ = 0 variant was enhanced by switching from a prior that
is uniform over the auxiliary parameter space to one that is
uniform over the primary parameter space. By contrast, this same
transformation had the opposite effect in the _w_ = 1 model.
Finally, the distribution of Bayes factors showed almost no change
in the full model. This is consistent with Fig. 1, which shows
that samples drawn from the pushforward appear to be nearly
uniformly distributed over the primary parameter space.



**Fig.** **2.** Samples taken from the pushforward of a uniform measure (for the
identity stimuli ordering). The pushforward is singular along the ‘‘ridge" where
the surface meets the PBC axis.


**Fig.** **3.** The density of the measure _ν_, which implies a uniform distribution on
the primary parameter space, for the SI model with _w_ = 1.


We used the data from ‘‘cyclic trials" in Tsetsos et al.’s Experiment 4. [28] In this experiment, participants were presented with
sequences of choice stimuli. These stimuli were comprised of bar
graphs indicating attributes of job candidates. After a fixed number of stimuli presentations, participants were asked to select the
candidate who had, on average, superior attributes. The choice
stimuli were designed to induce violations of the transitivity
axiom. We refer readers to Tsetsos et al. (2016a, 2016b) for the
complete experimental descriptions.

Fig. 4 shows the distribution of Bayes factors thus obtained,
broken down by model and prior. As per Jeffreys’s (1961) classification, we considered a Bayes Factor above 3.16 to be substantial
evidence in favor of the model in question over the encompassing
model, a value between 1 and 3.16 to be very weak evidence
in favor, a value between _._ 316 and 1 to be very weak evidence
against, and a value less than _._ 316 to be substantial evidence
against.


28 The full results and Matlab code are available at figshare.com/
articles/Segert_Davis-Stober_2017_Supplementary_Material_/5686648.



Next, we assess model performance at the individual-level.
We calculated overall Bayes factors for each subject, for each
model, by taking the product of Bayes factors across experimental
conditions — similar to the ‘‘group Bayes factor’’ of Prince, Brown,
and Heathcote (2012). This allows us to select a best-fitting model
for each subject via the usual Bayesian model comparison by
Bayes factor. For models defined by a uniform distribution over
the auxiliary parameters, all 21 participants were best described
by the full SI model. For models defined by a uniform distribution
over the primary parameters, 15 subjects were best described
by the full SI model, while the remaining 6 were best described
by the three parameter SI model. No subject was best described
by the encompassing (unconstrained) model nor any model with
_w_ = 1. These results support Tsetsos et al.’s (2016a, 2016b) general conclusions, as gating ( _w_ _<_ 1) was a necessary component
for high-performing models. Our results provide some nuance to
these conclusions as not all subjects appear to be using early
noise in the same way under our analysis. We note that this is
the first analysis of these data to directly compare the variants of
the SI model to an unconstrained competitor model. This is also
the first analysis of the SI model using the dis-aggregated data
from Experiment 4. As argued by Davis-Stober, Park, Brown, and
Regenwetter (2016), aggregating the data across experimental
conditions, for each individual, can lead to data artifacts - see
also the reply by Tsetsos et al. (2016b).


_5.5._ _Parameter_ _estimation_



_dV_ ∗ _prim_  - _F_ of the pushforward of this measure, expressed in
terms of _λ_ and _α_ .

The corresponding densities are plotted in Fig. 5 for a single subject in a single permutation setting. [29] The corresponding
parameter estimates are then given by maximizing these two
functions.


**6.** **Discussion**


The question of prior selection is one that has obvious implications for practitioners of Bayesian statistics. Many prescriptions
for such a question, e.g., maximizing entropy subject to certain constraints or choosing a uniform distribution, depend upon


29 More definitely, the data are from subject 4, in the identity permutation
setting.



Finally, we briefly consider the question of obtaining point
estimates _λ_ [∗] and _α_ [∗] of the parameters of the restricted model.
There are (at least) two nonequivalent ways to do this. We may
maximize either the posterior density _π_ | _D_, giving the standard
_maximum_ _a_ _posteriori_ estimate, or we may maximize the density



_d_ ( _F_ ∗( _π_ | _D_ ))


_S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_ 115


**Fig.** **4.** Distribution of Strength of Evidence against Encompassing Model, by Model and Prior.


paper, we have demonstrated a general procedure for describing
how a given choice of prior transforms under a wide class of
re-parameterizations, and demonstrated explicitly how the results obtained thereby differ from simply applying a uniform
prior to both parametrizations. Our results provide a comprehensive set of tools for researchers to construct priors, over either
the primary or auxiliary parameter space, which are equivalent
to a specified prior over the other parameter space, thereby
generalizing the core results of HW.

In our illustration using the Selective Integration model of
Tsetsos et al. we solved for the prior distribution on the auxiliary
parameter space that yielded a uniform prior over the primary
parameter space. In this case, it happens that both the primary
parameter space and the space of empirical choice frequencies
(data outcomes) are both subsets of the same space. This connects, at least conceptually, to the work of Chandramouli and
Shiffrin (2016), who developed a framework for Bayesian inference where priors are placed at the level of data/outcomes. Our
results are not limited to the usage of non-informative priors and
could be used to construct informative priors for inference, along
the lines described by Lee and Vanpaemel (2017).



**Fig.** **5.** Density of Posterior vs. Density of Pushforward of Posterior.


the specific parametrization of a model, [30] which can be problematic when there is not a canonical parametrization. In this


30 More explicitly, suppose our model is parametrized by a single parameter
_x_ ∈[0 _,_ 1]. If we were to set a prior on _x_ by maximizing the entropy, we would
have _x_ ∼ _Unif_ (0 _,_ 1).
However, if another researcher has the same theory, but instead chooses to
parametrize it by the parameter _y_ = _x_ [2] ∈[0 _,_ 1], and defines a prior on _y_ in
the same way, then his prior will differ from ours even after accounting for
the re-parametrization, in the sense that the pushforward of our prior by the



Considerations of prior transformation can arise in many contexts. For example, Danaher, Roy, Chen, Mumford, and Schisterman (2012) developed a Bayesian model in which the parameters
of interest were constrained to lie within a polyhedral space. To
enable efficient posterior sampling, they re-parametrized these
values using a Minkowski–Weyl decomposition. Said simply, using this decomposition, any value within this polyhedral space
can be represented as a simple linear sum of alternative parameters. Viewed within our results, the parameters corresponding
to the Minkowski–Weyl decomposition serve as the auxiliary
parameters. Our results make explicit the relationship between
these parameter spaces and allow solving for priors over the
Minkowski–Weyl decomposition that yield desired priors over
the original (primary) parameters.


function _f_ ( _x_ ) = _x_ [2] will not be equal to his prior. In fact, while his prior will be
uniform over _y_, it is easy to check that _d_ ( _f_ ∗ _Unifdy_ (0 _,_ 1)) = 2 ~~[√]~~ 1 _y_ [.]


116 _S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_



The primary contribution of our paper regards the setting of
priors over re-parametrized spaces so that the researcher obtains
their desired model. Within the contexts we considered, the
parameter space of interest will be constrained, either linearly
or, more generally, non-linearly. We did not go into details on
how to efficiently calculate Bayes factors over such constrained
spaces. For that, we refer the reader to Hoijtink (2011). Equally
important is determining the inequality constraints themselves,
i.e., developing the order-constrained model in which to test,

- see Mulder (2016) and Dittrich, Leenders, and Mulder (2017)
for further discussion on ‘‘informative hypotheses’’ and related
concepts.

Finally, we comment on what some readers may consider to
be the relatively high level of mathematical sophistication with
which we formulated our results. In particular, some readers
may wonder whether the same goal could be achieved with less
machinery. We reiterate from Section 2.1 that our use of this
machinery was necessitated by the following two features of the
Tsetsos model: (1) The parameter space has positive codimension
relative to the ambient space, and (2) The parameter transformation is not one-to-one, and in fact is infinitely-many-to-one.
Furthermore, as we demonstrated in Section 2.1, these features,
far from being idiosyncrasies specific to the Tsetsos model, can
in fact arise quite naturally in much simpler settings. As such, we
believe that this example provides a convincing demonstration
of both the necessity and the power of the differential–geometric
tools which we have introduced.

Further directions might include the development of efficient
sampling techniques for measures derived from a Riemannian
volume form, especially on manifolds which are not parametrized
simply by a single coordinate patch.


**Appendix**


_A.1._ _Expression_ _for_ _the_ _Jacobian_


The goal of this appendix is to derive a ‘‘local’’ expression for
_JF_ in terms of arbitrary bases of the tangent spaces.

Consider bases _ei_ and _fi_ of _T_ _**η**_ _D_ and _TF_ ( _**η**_ ) _Im_ ( _F_ ) respectively. Let
_jac_ ( _e, f_ ) denote the matrix of _Jac_ ( _F_ ) with respect to these bases.
In other words, _Jac_ ( _F_ ) _ei_ = [∑] _j_ _[jac]_ [(] _[e][,][ f]_ [ )] _[ji][f][j]_ [.]

We denote by _N_ the matrix of _Jac_ ( _F_ ) _[T]_ with respect to the same
bases. Then the matrix of _Jac_ ( _F_ ) _Jac_ ( _F_ ) _[T]_ with respect to the basis _fi_
is given by _j_ ( _e, f_ ) _N_ . Since the determinant does not depend on the
choice of basis, we have _JF_ = _det_ ( _MN_ ). It remains only to express
_N_ in terms of _jac_ ( _e, f_ ). A word of caution: _N_ is not necessarily
equal to the matrix transpose of _jac_ ( _e, f_ ).

In preparation for what follows, we define the _Gram_ _matrices_
_G_ ( _e_ ) _ij_ = ( _ei, ej_ ) _aux_ and _G_ ( _f_ ) _ij_ = ( _fi, fj_ ) _prim_ where the subscripts
refer to the Riemannian metrics on the auxiliary and primary
parameter spaces, respectively.

Recall that the transpose is defined by the equation


( _v, Jac_ ( _F_ )( _**η**_ ) _[T]_ _w_ ) _aux_ = ( _Jac_ ( _F_ )( _**η**_ ) _v, w_ ) _prim,_


for any _v_ ∈ _T_ _**η**_ _D_ and _w_ ∈ _TF_ ( _**η**_ ) _Im_ ( _F_ ). In particular,


( _ej, Jac_ ( _F_ )( _**η**_ ) _[T]_ _fi_ ) _aux_ = ( _Jac_ ( _F_ )( _**η**_ ) _ej, fi_ ) _prim._


Recall that _N_ denotes the matrix of _Jac_ ( _F_ ) _[T]_ with respect to the
bases _ei_ and _fj_ ; thus _Jac_ ( _F_ ) _[T]_ _fi_ = [∑] _k_ _[N][ki][e][k]_ [.] [Taking] [the] [dot] [product]

of both sides with _ej_, we see


( _ej, Jac_ ( _F_ ) _[T]_ _fi_ ) _aux_ = ( _Jac_ ( _F_ )( _**η**_ ) _ej, fi_ ) _prim_



∑
= _NkiG_ ( _e_ ) _kj_ = ( _N_ _[t]_ _G_ ( _e_ )) _ij._


_k_


On the other hand, we have


∑
( _Jac_ ( _F_ )( _**η**_ ) _ej, fi_ ) _prim_ = _jac_ ( _e, f_ ) _kj_ ( _fk, fi_ )


_k_


∑
= _jac_ ( _e, f_ ) _kjG_ ( _f_ ) _ki_


_i_

= ( _jac_ ( _e, f_ ) _[t]_ _G_ ( _f_ )) _ji_
= ( _jac_ ( _e, f_ ) _[t]_ _G_ ( _f_ )) _[t]_ _ij_
= ( _G_ ( _f_ ) _jac_ ( _e, f_ )) _ij._


Here the lower-case _t_ denotes the matrix transpose, and should
not be confused with the operator transpose, denoted by upper
case _T_ .

We therefore obtain the matrix equation _N_ _[t]_ _G_ ( _e_ ) = _G_ ( _f_ ) _jac_ ( _e,_
_f_ ) for _N_, giving _N_ = ( _G_ ( _f_ ) _jac_ ( _e, f_ ) _G_ ( _e_ ) [−][1] ) _[t]_ = _G_ ( _e_ ) [−][1] _jac_ ( _e, f_ ) _[t]_ _G_ ( _f_ )
(we have used the fact that the inverse of a symmetric matrix is
again symmetric). Therefore, we obtain the formula:



and _Vprim_ ( _Ia,b_ ) = ∫ _ab_ [|] _[ γ]_ [ ′][(] _[t]_ [)][ |] _[FR]_ _[dt]_ [.]
We take _Vaux_ to be simply the uniform measure on _D_ .
Consider now the Jacobian _Jac_ ( _F_ ) : _T_ ( _x,y_ ) _D_ → _TF_ ( _x,y_ ) _Im_ ( _F_ ).
We compute the matrix with respect to the bases _ei_ = _∂x, ∂y_
and _f_ = (1 _,_ 2( _x_ [2] + _y_ [2] )) of the two spaces. Our computations
above show that _Fi_ = _Jac_ ( _ei_ ) = 2 _xif_, so the matrix is simply
_jacx,y_ = [(] 2 _x,_ 2 _y_ [)] .



√
_JF_ = _det_ ( _Jac_ ( _F_ ) _Jac_ ( _F_ ) _[T]_ ) =


_A.2._ _Worked_ _example_



_det_ ( _jac_ ( _e, f_ ) _G_ ( _e_ ) [−][1] _jac_ ( _e, f_ ) _[t]_ _G_ ( _f_ )) _._



In order to illustrate the techniques to those unfamiliar, we
provide a worked example which is slightly more involved than
the special cases discussed in the text. In particular, this example
will deal with the case where the pre-images are manifolds,
rather than single points.

Consider _D_ = {( _x, y_ ) ∈ R [2] : _x_ [2] + _y_ [2] ≤ 1} and let _F_ : _D_ →

[0 _,_ 1] [2] _, F_ ( _x, y_ ) = ( _x_ [2] + _y_ [2] _,_ ( _x_ [2] + _y_ [2] ) [2] ). Note that _F_ is geometrically
equal to the segment {( _x, y_ ) : _y_ = _x_ [2] _, x_ ∈[0 _,_ 1]}.

We will suppose that the ambient space [0 _,_ 1] [2] is endowed
with the Fisher–Rao metric _gFR_, which may be written as follows



1
( _gFR_ )( _p,q_ )(( _∂x_ )( _p,q_ ) _,_ ( _∂x_ )( _p,q_ )) =

_p_ (1 − _p_ )



1
( _gFR_ )( _p,q_ )(( _∂y_ )( _p,q_ ) _,_ ( _∂y_ )( _p,q_ )) =

_q_ (1 − _q_ )
( _gFR_ )( _p,q_ )(( _∂x_ )( _p,q_ ) _,_ ( _∂y_ )( _p,q_ )) = 0 _._



(More precisely, it is the Fisher–Rao metric, considering ( _p, q_ )
to be the vector of success probabilities of two independent
Bernoulli distributions with one trial). In what follows, we will
drop the subscript for notational clarity.

We compute _Fx_ = 2 _x_ (1 _,_ 2( _x_ [2] + _y_ [2] )) _, Fy_ = 2 _y_ (1 _,_ 2( _x_ [2] + _y_ [2] )). The
tangent space _TF_ ( _x,y_ ) _Im_ ( _F_ ) is given by the span of these; thus we
may take (1 _,_ 2( _x_ [2] + _y_ [2] )) as a basis for this space.

To describe the corresponding measure _Vprim_ on _Im_ ( _F_ ), consider the ‘‘interval’’ _Ia,b_ = {( _x, y_ ) : _x_ = _y_ [2] _, x_ ∈[ _a, b_ ]} ⊂ _Im_ ( _F_ ). We
will compute _Vprim_ ( _Ia,b_ ).

To do so, we consider the parametrization _γ_ : [ _a, b_ ] → _Ia,b_,
_γ_ ( _t_ ) = ( _t, t_ [2] ). The metric at the point _γ_ ( _t_ ) is given by _g_ 11 =

_t_ (11− _t_ ) _[,][ g]_ [22] [=] _t_ [2] (11− _t_ [2] ) _[,][ g]_ [12] [=] [0.] [Since] _[ γ]_ [ ′][(] _[t]_ [)][ =] _[ ∂][x]_ [ +][ 2] _[t][∂][y]_ [,] [we] [see]



~~√~~



1 4 _t_ [2]

_t_ (1 − _t_ ) [+] _t_ [2] (1 −



1



_t_ [2] (1 − _t_ [2] ) _[,]_



| _γ_ [′] ( _t_ ) | _FR_ =


_S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_ 117



Since we put the Euclidean metric on _D_, the Gram matrix _G_ ( _e_ )
is the identity. Note that the Fisher–Rao metric at the point _F_ ( _x, y_ )
is given by _g_ 11 = ( _x_ [2] + _y_ [2] )(11−( _x_ [2] + _y_ [2] )) _[,][ g]_ [22] [=] ( _x_ [2] + _y_ [2] ) [2] (11−( _x_ [2] + _y_ [2] ) [2] ) _[,][ g]_ [12] [=]
0, so


_G_ ( _f_ ) = _g_ 11 + 4( _x_ [2] + _y_ [2] ) [2] _g_ 22



1
=



1 4

( _x_ [2] + _y_ [2] )(1 − ( _x_ [2] + _y_ [2] )) [+] 1 − ( _x_ [2]



1 − ( _x_ [2] + _y_ [2] ) [2]



5( _x_ [2] + _y_ [2] ) + 1
= ( _x_ [2] + _y_ [2] ) − ( _x_ [2] + _y_ [2] ) [3] _[.]_



By our result, we have


~~√~~
_JF_ ( _x, y_ ) = _det_ ( _jacG_ ( _f_ ) _jac_ _[t]_ _G_ ( _e_ ) [−][1] )

~~√~~
= 4( _x_ [2] + _y_ [2] ) _G_ ( _f_ )



√


= 2



5( _x_ [2] + _y_ [2] ) + 1

1 − ( _x_ [2] + _y_ [2] ) [2] _[.]_



Finally, we observe that _F_ [−][1] ( _a,_ _b_ ) is a circle with radius [√] _a_,

and accordingly has length 2 _π_ ~~[√]~~ _a_ . Appealing to our Main For
mula 2, we conclude that the measure



5( _x_ [2] + _y_ [2] ) + 1

1 − ( _x_ [2] + _y_ [2] ) [2] _[dxdy][,]_



1
_µ_ =



√



2 _π_ √



_x_ [2] + _y_ [2] [2]



satisfies _F_ ∗ _µ_ = _Vprim_, that is to say, that _µ_ ( _F_ [−][1] _U_ ) = _Vprim_ ( _U_ )
for every measurable _U_ . The direct verification of this result is
a straightforward and instructive exercise, which we leave to the
reader.


_A.3._ _Definition_ _of_ _the_ _pushforward_ _measure_


While this is a standard notion in measure theory, we realize
this its formal definition might look mysterious on a first reading.
Considering how crucial this concept is to our results, we shall
illustrate through examples how this concept in fact corresponds
to something very natural and simple, and we hope that after
reading this section the reader will be able to see the connection
between the formal definition and this simple operation.

Firstly, consider a mound of dirt sitting on the unit square

[0 _,_ 1] [2] . The volume of dirt above a region _U_ ⊂[0 _,_ 1] [2] is ∫ _U_ _[h]_ [(] _[x][,][ y]_ [)]

_dxdy_, where _h_ ( _x, y_ ) denotes the height in meters of the mound at
the given point. Assume for convenience that the total volume of
dirt is equal to 1 cubic meter. Taking an equivalent perspective,
this mound defines a probability distribution _µmound_ on [0 _,_ 1] [2],
where _µmound_ ( _U_ ) = ∫ _U_ _[h]_ [.]

Imagine that there are very sturdy walls located along the
_x_ and _y_ axes. Now further imagine driving a bulldozer into the
mound in such a way that the metal plate in the front of the
vehicle ends up parallel and very close to the wall along the
_x_ -axis, thereby squashing the mound into a very thin sheet lying
along this wall. Allowing ourselves some pedagogic license, we
will suppose that we have driven the bulldozer ‘‘infinitely close’’
to the wall, so that the 3-dimensional mound becomes a 2dimensional sheet. Let _g_ ( _x_ ) denote the height of the sheet so
obtained at the point _x_ ∈[0 _,_ 1]. As before, we consider this sheet
as defining a probability measure _µsheet_ defined on [0 _,_ 1], where
_µsheet_ ( _V_ ) = ∫ _V_ _[g]_ [(] _[x]_ [)] _[dx]_ [is] [the] [total] _[area]_ [of] [dirt] [lying] [above] [the]

region _V_ ⊂[0 _,_ 1]. [31] Let _F_ ( _x, y_ ) ∈[0 _,_ 1] denote the position along
the wall that a particle at the location ( _x, y_ ) ∈[0 _,_ 1] [2] will end up
at after we have pushed it with the bulldozer (evidently, _F_ ( _x, y_ ) =
_x_ ). Then the measure _µsheet_ corresponding to the squashed dirt


31 The reader may verify that _g_ does indeed integrate to 1, by applying Fubini’s
theorem.



mound is equal to the pushforward _F_ ∗ _µmound_ of the measure
corresponding to the original mound.

For an alternative example in the discrete setting, let _Cities_
denote the set of all American cities (which we take to also
include all small towns, villages, etc.). Define a measure _µcity_ on
this set by letting _µcity_ ( _C_ ) denote the population of the city _C_ .
Similarly, let _States_ denote the set of American states, and define
the measure _µstate_ by setting _µstate_ ( _S_ ) to be the population of
the state _S_ . Then _µstate_ = _F_ ∗ _µcity_ where _F_ : _Cities_ → _States_
is the function whose value on a city is equal to the state in
which it is located. We leave the reader to contemplate how
this interpretation may be extended to other similar binning or
coarsening procedures.


_A.4._ _Alternative_ _formulation_ _of_ _Jacobians_ _on_ _manifolds_


In this subsection, we present an alternative presentation of
defining Jacobians on manifolds. The main issue which this formulation addresses is the dependence of the previous definition
on the choice of local coordinate systems. From a conceptual
standpoint, removing this dependence is more satisfying, because
it makes clear that the Jacobian is an _intrinsic_ quantity depending
only the geometry of the situation (i.e., the manifolds _M_ and _N_,
in addition to the map _F_ ), and not on how we choose to describe
it (i.e., the local parametrizations _CM_ and _CN_ ).

The conceptual leap needed to formulate this idea is to consider the Jacobian to be a _linear_ _operator_ rather than a matrix.
To a manifold _M_ ⊂ R _[k]_ and a point _p_ ∈ _M_, we will define
the _tangent_ _space_ _TpM_ to be the linear subspace of R _[k]_ which
best approximates _M_ near _p_ . The Jacobian operator _Jac_ ( _F_ )( _p_ ) :
_TpM_ → _TF_ ( _p_ ) _N_ of _F_ at _p_ is then the best linear approximation
of the function _F_ near the point _p_ .


**Definition** **3.** If _M_ ⊂ R _[n]_ is a manifold, then the tangent space
_TpM_ is given by _Span_ { _∂iC_ (0)} where _C_ is any local parametrization
near _p_, and _∂iC_ (0) is the ordinary partial derivative, considered as
a vector in R _[n]_ . It can be shown that the subspace so obtained does
not depend on the choice of _C_ .


Given a manifold _M_, one may wonder how to compute the
tangent space. If _M_ = _F_ ( _U_ ) where _U_ ⊂ R _[k]_ is an open set
and _F_ : R _[n]_ is a smooth one-to-one map, then this is easy: we
have _TpM_ = _Span_ { _∂iF_ ( _F_ [−][1] ( _p_ ))} _[k]_ _i_ =1 [,] [that] [is,] [the] [span] [of] [the] [ordi-]
nary partial derivatives of the function _F_, evaluated at the point
_F_ [−][1] ( _p_ ) ∈ _U_ . For a space like the sphere, which is instead defined
by a constraint, it is less obvious how to proceed. We state an
extension to the Implicit function theorem which addresses this:


**Proposition** **2.** _Let_ _M_ = _f_ [−][1] (0) _where_ _f_ : R _[n]_ _is_ _a_ _smooth_ _function_
_with_ _the_ _property_ _that_ _the_ _vector_ ( _∂_ 1 _f_ ( _x_ ) _, . . ., ∂nf_ ( _x_ )) _is_ _nonzero_ _for_
_every_ _x_ ∈ _f_ [−][1] (0) _._

_Then_ _the_ _tangent_ _space_ _at_ _any_ _point_ _p_ ∈ _M_ _is_ _given_ _by_ _TpM_ =
_ker_ ({ _∂if_ ( _p_ )} _i_ ) _,_ _i.e._ _the_ _kernel_ _of_ _the_ _Jacobian_ _matrix_ _of_ _f ._


As in the previous section, this result generalizes to the case
when _M_ is defined by multiple simultaneous constraints, we refer
to Boothby (2003) for the details.

Applying this result to the sphere, we see that if _p_ = ( _x, y, z_ ) ∈
_S_ [2], then _TpS_ [2] = _Ker_ (2 _x,_ 2 _y,_ 2 _z_ ). In other words, _TpS_ [2] is equal


118 _S._ _Segert_ _and_ _C.P._ _Davis-Stober_ _/_ _Journal_ _of_ _Mathematical_ _Psychology_ _91_ _(2019)_ _103–118_



tothe orthogonal complement of the vector connecting _p_ to the
origin, as in accord with geometric intuition.

Having seen how to define the spaces on which the Jacobian
operates, let us now give its formal definition:


**Definition** **4.** Let _F_ : _M_ → _N_ be a map between manifolds, and
let _p_ ∈ _M_ . The Jacobian of _F_ at _p_, _Jac_ ( _F_ )( _p_ ) : _TpM_ → _TF_ ( _p_ ) _N_ is the
linear operator defined as follows:

1. Choose local parametrizations _CM_ : _U_ 1 → _M_ and _CN_ : _U_ 2 →
_N_ such that _p_ ∈ _CM_ ( _U_ 1) and _F_ ( _p_ ) ∈ _CN_ ( _U_ 2)

2. Compute the ordinary Jacobian matrix of the function _CN_ [−][1] [◦]
_F_ - _CM_ : _U_ 1 → _U_ 2 evaluated at the point _CM_ [−][1][(] _[p]_ [)][ ∈] _[U]_ [1][.]
3. Recall from the definition of the tangent space that the
sets _bM_ = { _∂iCM_ } and _bN_ = { _∂iCN_ } are bases of _TpM_ and _TF_ ( _p_ ) _N_
(here the partial derivatives are evaluated at the points _CM_ [−][1][(] _[p]_ [)]
and _CN_ [−][1][(] _[F]_ [(] _[p]_ [)),] [respectively).]
4. Define _Jac_ ( _F_ )( _p_ ) to be the linear operator whose matrix with
respect to the bases _bM_ and _bN_ from step 3 is equal to the matrix
from step 2.

It can be checked that the linear operator so obtained does not
depend on the parametrizations chosen in part 1.


Finally, let us explicitly compute a Jacobian matrix. Let us
consider the map _F_ : _S_ [2] → R defined by _F_ ( _x, y, z_ ) = _x_ + _y_ + _z_ [2]

for ( _x, y, z_ ) ∈ _S_ [2] . We will compute the Jacobian of this map at the
south pole _SP_ .

As in the definition, let us denote _M_ = _S_ [2] and _N_ = R.
To perform the computation, we simply follow the recipe
given in the definition.

1. As described in the previous section, we may take the
local parametrization of _S_ [2] near the pole to be the function
_CM_ : _B_ [2] 1 _/_ 2 [(0)] [→] _[S]_ [2][,] _[C][M]_ [(] _[x][,][ y]_ [)] [=] [(] _[x][,][ y][,]_ [ −] √1 − _x_ [2] - _y_ [2] ). For the

codomain, we may take the trivial parametrization _CN_ : R → R,
_CN_ ( _x_ ) = _x_ .

2. The function _F_ [˜] = _CN_ [−][1]   - _F_   - _CM_ is given by _F_ [˜] ( _x, y_ ) =
_F_ ( _CM_ ( _x, y_ )) = _x_ + _y_ + (1 - _x_ [2] - _y_ [2] ). By standard calculus, the
Jacobian matrix of this function at the point ( _x, y_ ) is given by
_(_ 1 − 2 _x,_ 1 − 2 _y)_ . We need to evaluate this matrix at the point
( _x, y_ ) = _CM_ [−][1][(] _[SP]_ [)] [=] [(0] _[,]_ [ 0).] [We] [denote] [the] [matrix] [so] [obtained] [by]
_jac_ ; thus _jac_ = (1 _,_ 1).

3. We have _∂_ 1 _F_ (0 _,_ 0) = (1 _,_ 0 _,_ 0) and _∂_ 2 _F_ (0 _,_ 0) = (0 _,_ 1 _,_ 0).
Thus _bM_ = {(1 _,_ 0 _,_ 0) _,_ (0 _,_ 1 _,_ 0)} is a basis for the tangent space
_TSP_ _S_ [2] . Similarly, _bN_ = {1} is a basis for _TF_ ( _SP_ )R Let us denote the
elements of these bases by _b_ [1] _M_ _[,][ b]_ [2] _M_ [and] _[b]_ [1] _N_ [.]
4. The Jacobian derivative _Jac_ ( _F_ )( _p_ ) : _TSP_ _S_ [2] → _TF_ ( _SP_ )R is
uniquely defined by the requirements that _Jac_ ( _F_ )( _SP_ )( _b_ [1] _M_ [)] =
_jac_ 1 _,_ 1 _b_ [1] _N_ [and] _[Jac]_ [(] _[F]_ [)(] _[SP]_ [)(] _[b]_ [2] _M_ [)] [=] _[jac]_ [1] _[,]_ [2] _[b]_ [1] _N_ [,] [where] _[jac][i][,][j]_ [denote] [the]
entries of the matrix _jac_ = (1 _,_ 1) from above. In this case,
_Jac_ ( _F_ )( _SP_ ) maps each element of the basis {(1 _,_ 0 _,_ 0) _,_ (0 _,_ 1 _,_ 0)} of
_TSP_ _S_ [2] to 1 ∈ _TF_ ( _SP_ )R

We invite the reader to consider different parametrizations in
Step 1, and verify that this same linear operator is obtained.



**References**


Amari, S. (1985). _[Differential-geometrical](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb1)_ _methods_ _in_ _statistics_ . Berlin: Springer
[Verlag.](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb1)
Batchelder, W. H., & Riefer, D. M. [(1999).](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb2) Theoretical and empirical review of

multinomial process tree modeling. _[Psychonomic](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb2)_ _Bulletin_ _&_ _Review_, _6_, 57–86.
Boothby, W. M. (2003). _An_ _introduction_ _[to](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb3)_ _differentiable_ _manifolds_ _and_ _Riemannian_

_geometry_ . [Academic](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb3) press.
Chandramouli, S. H., & Shiffrin, R. M. (2016). Extending Bayesian induction.

_Journal_ _of_ _[Mathematical](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb4)_ _Psychology_, _72_, 38–42.
Chavel, I. (2006). _Riemannian_ _[geometry:](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb5)_ _a_ _modern_ _introduction_ . Cambridge

[university](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb5) press.
Danaher, M. R., Roy, A., Chen, Z., [Mumford,](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb6) S. L., & Schisterman, E. F. (2012).

Minkowski–Weyl Priors for models [with](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb6) parameter constraints: an analysis
of the BioCycle study. _Journal_ _of_ _the_ _[American](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb6)_ _Statistical_ _Association_, _107_ (500),
[1395–1409.](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb6)
Davis-Stober, C. P., Brown, N., Park, S., & Regenwetter, M. (2017). Recasting a

biologically motivated computational [model](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb7) within a Fechnerian and random
utility framework. _Journal_ _of_ _[Mathematical](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb7)_ _Psychology_, _77_, 156–164.
Davis-Stober, C. P., Park, S., Brown, [N.,](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb8) & Regenwetter, M. (2016). Reported

violations of rationality may be [aggregation](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb8) artifacts. _Proceedings_ _of_ _the_
_National_ _Academy_ _of_ _[Sciences](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb8)_, _113_ (33), E4761–E4763.
Dittrich, D., Leenders, R. T. A., & Mulder, J. (2017). Network autocorrelation

modeling: a Bayes factor approach for testing (multiple) precise and interval hypotheses. _Sociological_ _Methods_ _&_ _Research_, [http://dx.doi.org/10.1177/](http://dx.doi.org/10.1177/0049124117729712)
[0049124117729712.](http://dx.doi.org/10.1177/0049124117729712)
Heck, D. W., Hilbig, B. E., & Moshagen, [M.](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb10) (2017). From information processing

to decisions: Formalizing and [comparing](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb10) psychologically plausible choice
models. _Cognitive_ _[psychology](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb10)_, _96_, 26–40.
Heck, D. W., & Wagenmakers, E. -J. [(2016).](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb11) Adjusted priors for Bayes fac
tors involving reparameterized [order](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb11) constraints. _Journal_ _of_ _Mathematical_
_Psychology_, _[73](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb11)_, 110–116.
Hoijtink, H. (2011). _Informative_ _[hypotheses:](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb12)_ _Theory_ _and_ _practice_ _for_ _behavioral_ _and_

_social_ _scientists_ . [Chapman](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb12) and Hall/CRC.
Jeffreys, H. (1961). _The_ _theory_ _of_ _[probability](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb13)_ . Oxford University Press.
Klauer, K. C., Singmann, H., & Kellen, [D.](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb14) (2015). Parametric order constraints in

multinomial processing tree models: [An](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb14) extension of Knapp and Batchelder
(2004). _Journal_ _of_ _[Mathematical](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb14)_ _Psychology_, _64_, 1–7.
Lee, M. D., & Vanpaemel, W. (2017). [Determining](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb15) informative priors for cognitive

models. _Psychonomic_ _[Bulletin](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb15)_ _&_ _Review_, 1–14.
Moshagen, M. (2010). multiTree: A [computer](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb16) program for the analysis of

multinomial processing tree models. _[Behavior Research Methods](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb16)_, _42_ (1), 42–54.
Mulder, J. (2014). Prior adjusted [default](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb17) Bayes factors for testing (in)equality

constrained hypotheses. _[Computational](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb17)_ _Statistics_ _&_ _Data_ _Analysis_, _71_,
[448–463.](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb17)
Mulder, J. (2016). Bayes factors for [testing](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb18) order-constrained hypotheses on

correlations. _Journal_ _of_ _[Mathematical](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb18)_ _Psychology_, _72_, 104–115.
Prince, M., Brown, S., & Heathcote, [A.](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb19) (2012). The design and analysis of

state-trace experiments. _[Psychological](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb19)_ _Methods_, _17_ (1), 78–99.
Singmann, H., & Kellen, D. (2013). [MPTinR:](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb20) Analysis of multinomial processing

tree models in R. _Behavior_ _[Research](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb20)_ _Methods_, _45_ (2), 560–575.
Tsetsos, K., Moran, R., Moreland, J., [Chater,](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb21) N., Usher, M., & Summerfield, C.

(2016a). Economic irrationality is [optimal](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb21) during noisy decision making.
_Proceedings_ _of_ _the_ _National_ _Academy_ _of_ _[Sciences](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb21)_ _of_ _the_ _United_ _States_ _of_ _America_,
_113_ (1), [3102–3107.](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb21)
Tsetsos, K., Moran, R., Moreland, J. C., [Chater,](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb22) N., Usher, M., & Summerfield, C.

(2016b). Reply to Davis-Stober et [al.:](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb22) Violations of rationality in a psychophysical task are not aggregation artifacts. _Proceedings_ _of_ _the_ _National_
_Academy_ _of_ _[Sciences](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb22)_, E4764–E4766.
Tversky, A., & Kahneman, D. (1992). [Advances](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb23) in prospect theory: Cumulative

representation of uncertainty. _[Journal](http://refhub.elsevier.com/S0022-2496(17)30259-6/sb23)_ _of_ _Risk_ _and_ _Uncertainty_, _5_, 297–323.


