### **Journal of the American Statistical Association**

**[ISSN: 0162-1459 (Print) 1537-274X (Online) Journal homepage: www.tandfonline.com/journals/uasa20](https://www.tandfonline.com/journals/uasa20?src=pdf)**

## **Statistical Methods for Eliciting Probability** **Distributions**


**Paul H Garthwaite, Joseph B Kadane & Anthony O'Hagan**


**To cite this article:** Paul H Garthwaite, Joseph B Kadane & Anthony O'Hagan (2005) Statistical
Methods for Eliciting Probability Distributions, Journal of the American Statistical Association,
[100:470, 680-701, DOI: 10.1198/016214505000000105](https://www.tandfonline.com/action/showCitFormats?doi=10.1198/016214505000000105)


**To link to this article:** [https://doi.org/10.1198/016214505000000105](https://doi.org/10.1198/016214505000000105)


Published online: 01 Jan 2012.


[Submit your article to this journal](https://www.tandfonline.com/action/authorSubmission?journalCode=uasa20&show=instructions&src=pdf)


Article views: 2794


[View related articles](https://www.tandfonline.com/doi/mlt/10.1198/016214505000000105?src=pdf)


[Citing articles: 55 View citing articles](https://www.tandfonline.com/doi/citedby/10.1198/016214505000000105?src=pdf)


Full Terms & Conditions of access and use can be found at
[https://www.tandfonline.com/action/journalInformation?journalCode=uasa20](https://www.tandfonline.com/action/journalInformation?journalCode=uasa20)


# **Statistical Methods for Eliciting** **Probability Distributions**

Paul H. GARTHWAITE, Joseph B. KADANE, and Anthony O’HAGAN


Elicitation is a key task for subjectivist Bayesians. Although skeptics hold that elicitation cannot (or perhaps should not) be done, in practice
it brings statisticians closer to their clients and subject-matter expert colleagues. This article reviews the state of the art, reflecting the
experience of statisticians informed by the fruits of a long line of psychological research into how people represent uncertain information
cognitively and how they respond to questions about that information. In a discussion of the elicitation process, the first issue to address is
what it means for an elicitation to be successful; that is, what criteria should be used. Our answer is that a successful elicitation faithfully
represents the opinion of the person being elicited. It is not necessarily “true” in some objectivistic sense, and cannot be judged in that way.
We see that elicitation as simply part of the process of statistical modeling. Indeed, in a hierarchical model at which point the likelihood
ends and the prior begins is ambiguous. Thus the same kinds of judgment that inform statistical modeling in general also inform elicitation
of prior distributions. The psychological literature suggests that people are prone to certain heuristics and biases in how they respond to
situations involving uncertainty. As a result, some of the ways of asking questions about uncertain quantities are preferable to others, and
appear to be more reliable. However, data are lacking on exactly how well the various methods work, because it is unclear, other than by
asking using an elicitation method, just what the person believes. Consequently, one is reduced to indirect means of assessing elicitation
methods. The tool chest of methods is growing. Historically, the first methods involved choosing hyperparameters using conjugate prior
families, at a time when these were the only families for which posterior distributions could be computed. Modern computational methods,
such as Markov chain Monte Carlo, have freed elicitation from this constraint. As a result, now both parametric and nonparametric methods
are available for low-dimensional problems. High-dimensional problems are probably best thought of as lacking another hierarchical level,
which has the effect of reducing the as-yet-unelicited parameter space. Special considerations apply to the elicitation of group opinions.
Informal methods, such as Delphi, encourage the participants to discuss the issue in the hope of reaching consensus. Formal methods,
such as weighted averages or logarithmic opinion pools, each have mathematical characteristics that are uncomfortable. Finally, there is the
question of what a group opinion even means, because it is not necessarily the opinion of any participant.


KEY WORDS: Bayesian; Group decision; Heuristics and biases; Prior distribution; Subjective probability.



1. THE ELICITATION CONTEXT


1.1 Introduction


Elicitation is the process of formulating a person’s knowledge and beliefs about one or more uncertain quantities into a
( joint) probability distribution for those quantities. In the context of Bayesian statistical analysis, elicitation most often arises
as a method for specifying the prior distribution for one or more
unknown parameters of a statistical model. In this context, the
prior distribution will be combined with the likelihood through
Bayes’ theorem to derive the posterior distribution. However,
this is not the only context in which elicitation is important.
Much of the literature on elicitation has been concerned with
formulating a probability distribution for uncertain quantities
when there are no data with which to augment the knowledge
expressed in that distribution. This situation arises in decision
making where uncertainty about the “state of nature” needs to
be expressed as a probability distribution to derive (and then
maximize) expected utility. Similarly, it arises in the use of
mechanistic models. Such models are built in almost all areas of
science and technology to describe, understand, and predict the
behavior of complex physical processes. The user is required to


Paul H. Garthwaite is Professor of Statistics, Faculty of Mathematics
and Computing, The Open University, Milton Keynes, U.K. (E-mail:
_P.H.Garthwaite@open.ac.uk_ ). Joseph B. Kadane is Leonard J. Savage University Professor of Statistics and Social Science, Carnegie Mellon University, Pittsburgh, PA 15213 (E-mail: _kadane@stat.cmu.edu_ ). Anthony O’Hagan
is Professor of Statistics, University of Sheffield, Sheffield, U.K. (E-mail:
_a.ohagan@sheffield.ac.uk_ ). Professor Kadane’s work was supported in part by
National Institutes of Health grant IR01 GM868950-01 and by the National
Science Foundation grant 0139911. Any opinions, findings, and conclusions
or recommendations expressed in this material are those of the author(s) and
do not necessarily reflect the views of the National Science Foundation. Garthwaite and O’Hagan’s work was supported by research funding from the U.K.
Department of Health and the Centre for the Bayesian Statistics in Health Economics, University of Sheffield.



specify the values of appropriate model inputs to run the model
and obtain outputs, but there is generally uncertainty about the
“true” values of the inputs. It is then important to formulate that
uncertainty and to propagate it through the model so as to quantify the uncertainty in model outputs.
It is convenient to think of the elicitation task as involving a _facilitator_, who helps the _expert_ formulate the expert’s
knowledge in probabilistic form. In the context of eliciting a
prior distribution for a Bayesian analysis, it is the expert’s prior
knowledge that is being elicited, but in general the objective is
to express the expert’s _current_ knowledge in probabilistic form.
If the expert is a statistician, or is very familiar with statistical
concepts, then there may be no formal need for a facilitator, but
this is rare in practice. Elicitation is a complex process that demands a range of skills if it is to be done well, and the role of
facilitator is an important one.
What does it mean for an elicitation to be done well? It is important to distinguish between the quality of an expert’s knowledge and the accuracy with which that knowledge is translated
into probabilistic form. An elicitation is done well if the distribution that is derived accurately represents the expert’s knowledge, regardless of how good that knowledge is. The expert
might, for instance, believe very strongly in a certain scientific
hypothesis. Then the elicitation is accurate if it derives a suitably high probability for that hypothesis being true, even if it is
subsequently found to be false. Even if the rest of the scientific
community is much more skeptical and inclined to give the hypothesis a low probability, this expert believes in the hypothesis, and therefore accurate elicitation of this expert’s knowledge
and beliefs should derive a high probability for it.


**© 2005 American Statistical Association**
**Journal of the American Statistical Association**
**June 2005, Vol. 100, No. 470, Review Article**
**DOI 10.1198/016214505000000105**



**680**


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 681



To achieve accurate elicitation is by no means straightforward, even if we wish to elicit the expert’s beliefs about just a
single event or hypothesis (or, equivalently, for a binary random
variable). In this case we require only a single probability, but
the expert may be unfamiliar with the meaning of probabilities.
Even when the expert is familiar with probabilities and their
meaning, it is not easy to accurately assess a probability value
for an event.
If we now consider the task of eliciting a distribution for
a continuous random variable _X_, then implicitly this involves
eliciting an infinite collection of probabilities _F(x)_ = _P(X_ ≤ _x)_
for all of the possible values of _x_ . This is clearly impossible,
and in practice an expert can make only a finite number (and
usually a rather small number) of statements of belief about _X_ .
These statements might take the form of individual probabilities or quantiles of the distribution [i.e., _P(X_ ≤ _x)_ for a few
distinct values of _x_ ] or might be some other summaries of the
distribution, such as modes. When it comes to a joint distribution for a collection of random quantities, the magnitude of the
elicitation task is much larger still.
Given the difficulty involved, why is it worth the effort to
attempt elicitation? One reason has to do with the use of elicitations to make decisions. Often a reasonable goal for elicitation is to capture the “big message” in the expert’s opinion. The
details (e.g., the exact shape of the expert’s opinion) may not
matter for the decision to be reached. Even when the decision
is sensitive to the exact shape of the elicited distribution, it is
not the decision, but rather the expected utility of the decision,
that matters. And expected utility of the optimal decision is very
often robust to details of the expert’s opinion.
A second reason why elicitation is worthwhile has to do with
the use of elicitations to make inferences, particularly for making possible the calculation of posterior distributions. In such a
situation, elicitation encourages the expert and the facilitator to
consider the meaning of the parameters being elicited. This has
two helpful consequences. First, it brings the analysis closer to
the application by demanding attention to what is being modeled and what is reasonable to believe about it. Second, it helps
make the posterior distributions, once calculated, into meaningful quantities.
Elicitation is properly conceived of as part of the familiar
process of statistical modeling. Statisticians are used to stating a likelihood for an applied problem. This is an opinion
about how the data are generated, conditional on certain parameters. Hierarchical models, such as random-effects models
and models with latent variables, involve distributions on some
parameters, conditional on yet other parameters. What we are
calling “elicitation” in this article is merely the final step in
this process, the statement of probability distributions of the
highest-level parameters in such a hierarchy. One should keep
in mind that the usual principles of statistical modeling apply to
elicitation as well.


1.2 The Elicitation Process


Figure 1 is a schematic representation of the elicitation
process in terms of four separate stages:


1. The _setup_ stage consists of preparing for the elicitation—
selecting the expert(s), training the expert(s), identifying
what aspects of the problem to elicit, and so on.



Figure 1. The Elicitation Process.


2. The next stage is to _elicit_ specific summaries of the experts’ distributions for those aspects. This is obviously
the core of the process, and one where psychologists have
contributed at least as much to the methodology as statisticians.
3. The next stage is to _fit_ a ( joint) probability distribution to
those summaries. In practice, this stage often blurs with
the previous stage, in the sense that the choice of what
summaries to elicit is often influenced by choice of what
distributional form the facilitator intends to fit.
4. Elicitation is almost invariably an iterative process, and
the fourth stage involves assessing the _adequacy_ of the
elicitation, with the option then of returning to the second
stage and eliciting more summaries from the expert(s).


This article is structured in accordance with this schematic.
The remainder of Section 1 concerns topics relevant to the setup
of the elicitation, Section 2 deals with the interaction with the
expert to elicit specific summaries, Section 3 addresses how
to fit a probability distribution to the elicited summaries, and
Section 4 deals with assessing the accuracy of elicitation. Section 5 considers questions that arise when beliefs are elicited
from several experts, and Section 6 offers some discussion and
challenges for future research.


1.3 Whose Beliefs?


We have presented elicitation as the process of formulating
in probabilistic terms the beliefs of an expert, but who is the
expert? The use of the term “expert” suggests an emphasis on
persons to whom society and/or his or her peers attribute special knowledge about the matters being elicited. In practice, we


682 Journal of the American Statistical Association, June 2005



often seek to identify the best available knowledge about the
quantities of interest, and for this purpose we can consider the
“expert” a real expert. There are other uses of elicitation, however, in which the expert has little or no expertise in the usual
sense of that word. For example, to study adolescent decision
making around risky behaviors, one might want to ask adolescents how they perceive those risks. Here the very point of the
study is the lack of experts of the experts.
The simple answer to the question of “who is the expert” is
that the expert is the person whose knowledge we wish to elicit;
the term “expert” does not necessarily signify any more than
that. An important point to bear in mind when eliciting from
an acknowledged expert is that expertise can bring biases if the
expert has some kind of personal interest in the result. Suppose,
for example, that a radiation expert is asked for an opinion about
how serious a health problem is engendered by the radiation release at Chernobyl. Such an expert may have spent much of
his or her adult life becoming an expert on radiation. How well
that expertise pays off in terms of social attention (and grants)
depends on how urgent society perceives the issues the expert
studies to be. Hence such an expert has an incentive to emphasize the dangers. (For more on this kind of bias, see Kadane and
Winkler 1988.)
In Section 5 we consider the case of multiple experts, where
often the desire is to combine the expertise of several people.
In such a case it is sensible to try to ensure that the experts’
knowledge is complementary. Where their knowledge overlaps,
it is more difficult to account for this (as we discuss in Sec. 5),
and there is less gain from using the extra experts.


1.4 Conducting the Elicitation


We outline here various aspects of good practice in the
conduct of elicitations. Many of these aspects can be ignored
in an informal elicitation, but they are important considerations
wherever substantive decisions or inferences may depend on the
expert’s distribution:


 - The objective is to elicit a distribution to represent the expert’s current knowledge. It is very useful to have a summary of what that knowledge is based on.

 - Any financial or personal interest that the expert might
have in the inferences or decisions that will depend (even
marginally) on the expert’s distribution must be declared.

 - Training should be given to familiarize the expert with the
interpretation of probability and with whatever concepts
and properties of probability will be required in the elicitation. It is useful to run through a dummy elicitation exercise to provide practice in the protocol that the facilitator
proposes to use.

 - A record should be kept of the elicitation. This should ideally set out all of the questions that were asked by the facilitator together with the expert’s responses, as well as the
process by which a probability distribution was fitted to
those responses.


Note that these recommendations apply whatever the specific
protocol that the facilitator will use to elicit the expert’s beliefs.
We now proceed with a detailed discussion of how to construct
a suitable protocol.



2. PSYCHOLOGICAL CONSIDERATIONS AND
ELICITING SUMMARIES


An elicitation method forms a bridge between an expert’s
opinions and an expression of these opinions in a statistically
useful form. Thus the development of an elicitation method requires some understanding of both the psychological part and
the statistical part of the bridge. As Hogarth (1975, p. 284)
pointed out, “assessment techniques should be designed both
to be compatible with man’s abilities and to counteract his deficiencies.” In this section we present some results from psychological research that should be taken into account when
forming methods of quantifying an expert’s opinion. Much of
the fundamental work in this area stems from the 1960s and
early 1970s, and good reviews of this research have been given
by Hampton, Moore, and Thomas (1973), Hogarth (1975),
Huber (1974), Lichtenstein, Fischhoff, and Phillips (1982),
Peterson and Beach (1967), Slovic and Lichtenstein (1971), and
Tversky (1974). Later reviews have been given by Chaloner
(1996), Cooke (1991), Hogarth (1987), Kadane and Wolfson
(1998), Meyer and Booker (2001), Morgan and Henrion (1990),
Wallsten and Budescu (1983), and Wolfson (1995).


2.1 Heuristics and Biases


A body of psychological research has been concerned with
the question of how a person assesses the probability of an
event, or how he or she judges which of two or more events is
the more likely to occur. It appears that intuitive judgments in
these tasks are based on a limited number of mental operations,
or _heuristics_ . In general these heuristics are quite effective, but
they can lead to severe errors and systematic bias. Elicitation
techniques commonly require novel assessment tasks. An appreciation of the strategies that people use to quantify their
opinions can give an indication of how (and how well) these
tasks might be performed (Meyer and Booker 2001).
One commonly used heuristic is _judgment by representative-_
_ness._ This is applicable for questions of the form: “what is the
probability that an object _A_ belongs to a class _B_ ?” or “what
is the probability that event _A_ will generate an event _B_ ?” In
answering these questions, which in effect require assessment
of the probability _P(B_ | _A)_, people typically compare the main
features of _A_ and _B_ and assign a probability depending on the
degree of similarity between them. A common error made with
this kind of judgment is that little or no attention is paid to
the unconditional probability of _B_ . As an illustration, consider
the problem of evaluating the probabilities that an individual,
Mr. X, who has been described as “meticulous, introverted,
meek, and solemn,” is engaged in one of the following occupations: farmer, salesman, pilot, librarian, or physician. Most
people perform this task by assessing the similarity of Mr. X to
the stereotype of that occupation, and order the occupations by
the extent to which Mr. X is representative of these stereotypes.
They completely ignore base rates, such as the relative number of salesmen to librarians, and assign a high probability to
Mr. X being a librarian (Kahneman and Tversky 1973). Similar
results have been obtained by Hammerton (1975) and Nisbett,
Borgida, Crandall, and Reed (1976).
Another commonly used heuristic is _judgment_ _by_ _availabil-_
_ity._ This is used when a person estimates the frequency of a


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 683



class or the probability of an event by the ease with which examples are recalled or occurrences come to mind. Examples
from large classes are usually recalled better and faster than examples from less frequent classes, and likely occurrences are
easier to imagine than unlikely ones. Hence mental availability is often a helpful indicator for the assessment of frequency
and probability, but availability is also affected by factors other
than frequency or probability. For example, suppose that one is
asked whether a randomly chosen word from an English text is
more likely to start with an “r” or to have “r” as its third letter. It
is easier to recall words by their starting letter (e.g., _r_ ed, _r_ ank,
_r_ ogue, _r_ oad, _r_ ope) than by their third letter (e.g., pa _r_ k, bi _r_ d,
wi _r_ e). Hence most people judge that “r” is more likely to be
the first letter of a word rather than the third letter, although the
reverse is actually true (Tversky and Kahneman 1973). Recall
is also affected by such factors as familiarity, salience, and recency, and newsworthy events also impact disproportionately
on memory, so one might overestimate the probability of a
plane crash with fatalities, for example, particularly if such a
crash has happened recently. Thus the judgment-by-availability
heuristic, though useful, can lead to marked error.
Perhaps the most widely used heuristic for probability assessment is _judgment by anchoring and adjustment._ With this strategy, a person estimates an unknown quantity by starting from
some initial value and then adjusting it to obtain a final estimate. The starting value, which is usually termed the _anchor_,
could be suggested by the nature of the problem or the way it
is formulated. Regardless of the source of the starting value,
the adjustment is usually too small (Slovic 1972), a phenomenon called _anchoring_ . An experiment conducted by Tversky
and Kahneman (1974) elegantly demonstrated this effect. Subjects were asked to estimate various quantities, stated in percentages (e.g., the percentage of African countries in the United
Nations). They were given randomly chosen starting values and
first had to decide whether the value they had been given was
too high or too low, and then adjust it until they reached their
best estimate. Through insufficient adjustment, subjects whose
starting values were high ended up with substantially higher estimates than those who started with low values. For example,
the median estimates of the percentage of African countries in
the United Nations was 25% for subjects who received 10% as
their starting point and 45% for those who received 65% as their
starting point.
As mentioned earlier, when subjects use the judgment-byrepresentativeness heuristic to assess probabilities, they tend
to ignore prior probabilities. However, many experiments have
shown that if subjects are made aware of the prior probabilities and are asked to modify these in the light of fresh sample
data, then the assessed posterior probabilities are too close to
the prior probabilities compared with the revision indicated by
Bayes’ theorem; that is, the subjects’ modifications to the prior
probabilities are insufficient. This type of insufficiency when
modifying probabilities to reflect new data is called _conser-_
_vatism_ (Edwards and Phillips 1964). One possible explanation
of this phenomenon is that subjects use the anchoring and adjustment strategy; the prior probability acts as the anchor and
the adjustment is insufficient.
A typical experiment to demonstrate conservatism is one in
which subjects assess the probability that colored poker chips



have been drawn from one of two bookbags, where the two
bags contain different compositions of chips (e.g., 70% red and
30% blue in one bag and 30% red and 70% blue in the other).
A coin is tossed to select one of the bags, so the prior probability of each bag is .50, and the experimenter then draws a
succession of chips with replacement from the selected bag, indicating their color to the subject. Having observed the sample
evidence, the subject states his or her posterior probability that
the bookbag sampled contained the predominantly red or the
predominantly blue proportion of chips. Subjects’ revisions are
normally conservative compared with the objective probability
calculated by Bayes’ theorem. For example, when the sample
contains eight red chips and four blue chips, subjects commonly
give a probability for the “red” bookbag of about .75, whereas
the posterior probability calculated by Bayes’ theorem is .97.
Several studies have attempted to counteract conservatism
through varying the experimental procedure. In case subjects
avoided approaching the bounds of the probability scale, unbounded odds estimates have been used instead of probability
estimates. Rewards have been added to provide an incentive
to perform well, and sample sizes, sequence lengths, and prior
probabilities have been varied. Some of these changes have influenced the degree of conservatism, but they have not eliminated it (see, e.g., Peterson and Beach 1967, pp. 32–33, for a
review of such experiments). In some experiments, however, the
basic experimental situation has been modified so as to make
it more complex, and in these more complex situations, conservatism is not always a dominating influence (Youssef and
Peterson 1973).
Research has also demonstrated that in many circumstances,
people expect a sample from a population to represent all of the
essential characteristics of that population, even if the sample
is small. Tversky and Kahneman (1971) refered to this false
logic as the “law of small numbers,” which asserts that the
law of large numbers applies to small numbers as well. One
experiment that demonstrated this fallacious belief used, as subjects, audiences at two psychology meetings. Through a questionnaire, the psychologists were asked to decide sample sizes
and also to relate sample sizes to inferences that they would
make from hypothesis tests. The following is a typical example of the questions asked: “Suppose you have run an experiment on 20 subjects and have obtained a significant result which
confirms your theory ( _z_ = 2 _._ 23, _p < ._ 05, two-tailed test). You
now have cause to run an additional group of 10 subjects. What
do you think the probability is that the results will be significant, by a one-tailed test, separately for this group?” (Tversky
and Kahneman 1971, p. 105). The median answer from the two
groups was .85. However, if one assumes a noninformative prior
distribution for the mean before the first sample was taken, then
the true probability is only .48. The error is readily attributable to belief in the “law of small numbers”; people expect all
samples to have virtually identical characteristics. For similar
reasons, answers to other parts of the questionnaire indicated
that most of the respondents (a) were too easily convinced by
early results from a small experiment, (b) tested their research
hypotheses on small samples without realizing the high odds
against detecting the effects being studied, and (c) rarely attributed unexpected results to sampling variability, because they
found a causal explanation for every observed effect.


684 Journal of the American Statistical Association, June 2005



The results demonstrating the “law of small numbers” and
those demonstrating conservatism appear somewhat contradictory. The former suggest that people overestimate the value of
sample evidence, whereas the latter suggest that people underestimate it. One conjecture is that conservatism has an effect if
people first formulate their opinion before being given the sample evidence, whereas the “law of small numbers” has an effect
if people obtain the sample evidence before first formulating
their opinion (Garthwaite 1983, p. 17).
Another form of systematic error that affects people’s judgments, _hindsight bias_ can arise when people are asked to assess
their a priori probability of an event that has actually occurred.
For example, they might be asked whether the dismantling of
the Berlin Wall was predictable before it happened: “In 1988,
what was the probability that the Berlin Wall would come down
within the next 5 years?” In 1988 it may have seemed unlikely
that communism would soon collapse and East and West Germany would reunite, but with hindsight the economic problems
of communist countries make it seem almost inevitable. Knowledge of what has occurred tends to distort memory, and people
tend to exaggerate their a priori probability for an event that
has occurred. An experiment that clearly demonstrates this phenomenon was conducted by Fischhoff and Beyth (1975). Just
before President Nixon’s visit to China and the USSR in 1972,
subjects were asked to assess the probabilities of various possible outcomes of his visit, such as “President Nixon will meet
Mao at least once” and “the United States and the USSR will
agree to a joint space program.” Shortly after the visit and without forewarning, the subjects were asked to recall the probabilities they had given to these events before Nixon’s visit, and
they were also asked which events they thought had actually
occurred. Results showed that subjects generally overestimated
their a priori probabilities for the events they thought had occurred and underestimated their a priori probabilities for the
events they thought had not occurred.
We have described only some of the heuristics that people
use to make numeric judgments, and the biases that affect such
judgments. These heuristics were chosen for their relevance to
elicitation methods. For example, conservatism, hindsight bias,
and the “law of small numbers” relate to people’s response to
data, and various elicitation methods use the impact of sample
data on opinion to quantify beliefs about variances and covariances, as we discuss in Section 1.3. An extensive list of heuristics and biases in human judgment was given by Hogarth (1987,
pp. 216–222).


2.2 What Summaries to Elicit


In designing an elicitation method, there is usually choice as
to which quantities the expert is asked to assess. If possible,
quantities should be chosen that are usually assessed reasonably competently. People’s ability to estimate simple statistical quantities, such as means and variances, has been examined
in psychological research over several decades. Such quantities
could constitute part of the elicitation method for almost any
form of prior distribution.
Several experiments have investigated subjects’ capability
to judge sample proportions (Erlick 1964; Nash 1964; Pitz
1965, 1966; Shuford 1961; Simpson and Voss 1961; Stevens



and Galanter 1957). In these experiments, binary data were displayed to subjects for a limited period of time, and the subjects were then asked to estimate one of the sample proportions.
For example, Shuford (1961) projected 20 × 20 matrices onto
a screen, one at a time. The elements of each matrix were red
squares and blue squares, and subjects observed a matrix for
1 second in some trials and for 10 seconds in others. After each
trial, subjects had to estimate the proportion of squares that had
been, say, red. In this and similar experiments, subjects generally assessed the sample proportion very accurately, with the
mean of subjects’ estimates differing from the true sample proportion by less than .05 in most cases.
Similar experiments have been used to investigate people’s
ability to estimate measures of central tendency (Beach and
Swenson 1966; Peterson and Miller 1964; Spencer 1961, 1963).
Typically, a sample of numbers is displayed to subjects, who are
then asked to estimate the mode, median, or mean of the sample. When the sample distribution is approximately symmetric,
so that these three measures are numerically similar, subjects’
estimates have shown a high degree of accuracy (Beach and
Swenson 1966; Spencer 1961). However, an experiment conducted by Peterson and Miller (1964) used a sample drawn from
a population whose distribution was highly skewed; subjects’
assessments of the median and mode were again reasonably accurate, but assessments of the mean were biased toward the median.
In most applications, determining a prior distribution requires estimation of the variances of unknown scalar quantities
and/or sample errors. Regrettably, it seems that people are poor
both at interpreting the meaning of “variance” and at assigning numerical values to it. When estimating relative variability, empirical evidence indicates that people are influenced by
the mean of the stimuli and estimate the coefficient of variation, rather than the variance. For example, Hofstatter (1939)
obtained assessments of the variability in the lengths of sticks
tied in bundles. He found that the assessed value increased with
the sample variance, as it should, but as the means increased,
the assessed value decreased. Lathrop (1967) replicated this result. Even allowing for the effect of means, systematic differences still arise between intuitive judgments of sample variance
and objective values. If large deviations from the mean predominate, as when, for example, the sample is drawn from a
population whose distribution is bimodal, then the variance is
overestimated. On the other hand, if small deviations from the
mean predominate, as when, for example, the population distribution is normal, then the variance is underestimated (Beach
and Scopp 1968).
One way of eliciting variances that avoids their direct assessment is to elicit credible intervals; such intervals are useful
in their own right and can yield estimates of variances if suitable distributional assumptions are made (cf. Sec. 3.2). There
are two main approaches to assessing credible intervals for a
scalar quantity, the _fixed interval_ method and the _variable inter-_
_val_ method. Let _X_ denote the scalar quantity of interest. With
the fixed interval method, the range of values that _X_ can take
is partitioned into intervals by the statistician/psychologist who
organizes the elicitation method. For each interval, the expert
assesses the probability that _X_ will fall in that interval. (In principle, assessed probabilities must sum to 1.) The probabilities


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 685



are also sometimes elicited through odds assessment. The expert first indicates the interval that he or she believes is most
likely to contain _X_ . Then, for each of the other “less likely intervals,” he or she states odds that _X_ will take a value in the
“less likely interval” as opposed to the “most likely interval.”
The probabilities associated with each interval are then calculated by imposing the constraint that their sum must be 1.
With the variable interval method, the expert identifies points
that correspond to specified percentiles of his or her subjective
distribution. A method of bisection is often used that entails a
sequence of questions of the following form:


Q1. Can you determine a value (the expert’s median) such
that _X_ is equally likely to be less than or greater than
this point?
Q2. Suppose you were told that _X_ is below your assessed
median; can you now determine a new value (the lower
quartile) such that it is equally likely that _X_ is less than
or greater than this value?
Q3. Suppose you were told that _X_ is above your assessed
median; can you now determine a new value (the upper
quartile) such that it is equally likely that _X_ is less than
or greater than this value?


An advantage of this line of questioning is that only judgments of equal odds are required, an intuitively easier task than
specifying percentiles that divide a probability in the ratio of,
say, 4:1.
Many experiments have examined people’s performance at
assessing credible intervals. If the credible intervals calibrate
well with reality, then the proportion of _p_ % central credible intervals that contain the true value of _X_ (the quantity to which
they relate) should be about _p_ %. (The concept of calibration is
discussed further in Sec. 4.4.) These experiments have demonstrated that assessing credible intervals is a task that people
perform reasonably well, but there is a clear tendency for central credible intervals to be too short, so that this proportion is
less than _p_ %. This bias is called _overconfidence_ ; people believe
they are more accurate in estimating _X_ than is justified (Keren
1991). Lichtenstein et al. (1982, pp. 325–326) tabulated 28 sets
of results in which 50% central credible intervals were elicited
for continuous scalar quantities. Less than 50% of the intervals
contained the true value of the scalar in 23 of the 28 sets, and the
proportion never exceeded 57% in any of the sets. The evidence
is conflicting as to whether the fixed-interval method or the
variable-interval method gives better calibration; Seaver, von
Winterfeldt, and Edwards (1978) found that the fixed-interval
method performed better, whereas Murphy and Winkler (1974)
found the converse. With the variable-interval method, it is also
unclear which percentiles should be elicited. The median and
quartiles are most commonly assessed (using the method of
bisection), and although this has sometimes yielded good results (e.g., Murphy and Winkler 1974; Peterson, Snapper, and
Murphy 1972), other empirical work has found that overconfidence is less if the 33rd and 67th percentiles are assessed
(Barclay and Peterson 1973; Garthwaite and O’Hagan 2000).
Most empirical work has involved scalar quantities, and the results may not generalize to more complex models. For example,
Garthwaite (1989) elicited 50% central predictive intervals for
the dependent variable in a simple linear regression model and



found that far more than 50% of the intervals contained correct
values. The subjects had drawn graphs to help make their assessments, and this may have improved the accuracy of their
median assessments and led to predictive intervals that were
more likely to contain a true value.
Much empirical research has investigated subjects’ ability to
assess the extreme “tails” of a distribution. For example, Alpert
and Raiffa (1969) used the variable interval method to elicit
98% central credible intervals. They asked “almanac” questions
of the following kind:

How many foreign cars were imported into the U.S. in 1968?
(a) Make a high estimate such that you feel there is only a 1% probability the
true answer would exceed your estimate.
(b) Make a low estimate such that you feel there is only a 1% probability the
true answer would be below this estimate (Alpert and Raiffa 1969, pp. 16–17).

It should have been somewhat of a “surprise” to a subject
to find the true value of a quantity falling outside an interval;
43% of all assessments produced such surprises. This information was given as feedback before a second session. In this second session, 23% of the assessments produced surprises, still a
very high percentage. One reason for the large number of surprises is that assessing tails of distributions is a difficult task,
mainly because it requires the consideration of events that are
unlikely, so that comparisons do not come readily to mind. (It
is unfortunate that assessing probabilities for rare events is difficult, because expert opinion is of paramount importance when
sample data are scarce.)
Task characteristics have an effect on how an expert views
a problem and the assessments that are elicited. Visual aids to
help people quantify their opinions have been tried, such as urns
full of colored balls (Raiffa 1968), light pens on colored screens
(Barclay and Randall 1975), or simply asking assessors to mark
a point on a line whose endpoints are 0 and 1 (for probabilities)
or 0–100% for proportion. Probability wheels (Spetzler and
Stael von Holstein 1975) are another visual aid that have been
used with some success (Morgan and Henrion 1990, p. 126). In
its simplest form, a probability wheel comprises a round pieshaped disc of one color that is partly covered by a “slice” of a
different color and a pointer. The size of the slice can be varied,
and the expert adjusts its size so that if the pointer is spun, then
the probability it lands within the slice is equal to the expert’s
probability for some specified event.
Efforts to influence how people consider probabilities have
also been explored, such as asking them to suggest scenarios that would lead to an unlikely event (Slovic and Fischhoff
1977), influence diagrams (Howard and Matheson 1984), getting subjects to think carefully about the substantive details of
each judgment (Koriat, Lichtenstein, and Fischoff 1980), and
disaggregating an implicit hypothesis into its constituent hypotheses (Johnson, Hershey, Meszaros, and Kunreuther 1993).
As an example of the effect of disaggregation, Fischhoff,
Slovic, and Lichtenstein (1978) questioned experts (i.e., car
mechanics) about the probable reasons for a car not starting.
The experts assessed the probability that it would not start “for
some reason other than the battery, engine or fuel system,” and
their average probability was .22. They also assessed the probability that it would not start for more specific reasons: failure
of the ignition system, failure of the starting system, and so on.
Combining the probabilities of the latter disaggregated reasons
gave an average of .44 as the probability that the car would not


686 Journal of the American Statistical Association, June 2005



start for a reason other than the battery, engine, or fuel system.
This result is consistent with other empirical work; there is ample evidence that the sum of separate probability assessments
for constituent hypotheses generally gives a much larger probability than a single probability assessment of the combined
hypothesis that they form (see Tversky and Koehler 1994 for a
review). Morgan and Henrion (1990, p. 116) commented that
“it has become something of an article of faith in the decision
analysis community that disaggregation of an elicitation problem holds the potential for significantly improved performance
on many assessment tasks.”
There has been research aimed at converting probabilistic phrases (such as “quite likely” and “extremely probable”)
into numeric values (Wallsten, Budescu, Rapoport, Zwick, and
Forsyth 1986; Mosteller and Yountz 1990), and at differentiating situations where verbal expressions of probability are
preferable to numeric ones, or vice versa (Windschitl and Wells
1996). People are generally more comfortable expressing their
uncertainty in verbal terms rather than numerically. Unfortunately, there is considerable variation in the probabilities that
different people attach to the same phrase, and the context
also affects the probability that a person associates with a
phrase (Lichtenstein and Newman 1967; Beyth-Marom 1982;
Wallsten et al. 1986). The response mode in which subjects
are asked to give assessments also affects judgments. For example, Gigerenzer (1996) found that numeric expressions that
are formally equivalent, such as frequencies and probabilities,
are not always treated as equivalent in subjective uncertainty
judgments. Also, it seems better to elicit probabilities in terms
of populations of events, such as “what proportion of students
starting a Ph.D. will complete it within 5 years?” rather than as
single (one-shot) events, like “if a new Ph.D. student is picked
at random, what is the probability that he or she will complete
the Ph.D. within 5 years?” (Gigerenzer 1996; Koehler 1996).
As noted earlier, a good elicitation method should yield
a probability distribution that accurately reflects the expert’s
opinion, but this is hard to check. A pragmatic alternative is to
compare assessed distributions with true values when these are
known; see the discussion of calibration in Section 4.4. Several experiments have attempted to train subjects to improve
the calibration/objective accuracy of their assessments. These
have typically found that objective accuracy is substantially
improved by training, but that such biases as overconfidence
are tempered rather than eliminated (Schaefer and Borcherding
1973; Lichtenstein and Fischhoff 1980). In these experiments,
training usually took the form of feedback, in which subjects
were told the correct values after making the assessments and
the trainer stressed the direction of biases and how the expert
might reduce them. The benefits of effective feedback can be
seen in the performance of weather forecasters. Weather forecasters make regular predictions for the same quantities each
day (e.g., temperature, probability of precipitation), and thus
soon learn the accuracy of their forecasts. Experiments have
generally found them to be quite well calibrated. For example, Peterson et al. (1972) conducted an experiment with two
meteorologists who gave forecasts of the maximum and minimum temperatures on the following day. Using the method of
bisection, the meteorologists expressed their forecasts in terms



of 50% central credible intervals. They showed good calibration; out of 55 forecasts, 28 fell inside the 50% credible intervals, 18 fell outside the 50% credible intervals, and 9 fell on the
boundaries.
In the main, research concerned with descriptive statistics
for scalars has produced relatively clear-cut conclusions. People are capable of estimating proportions, modes, and medians
of samples, but are slightly less proficient at assessing sample
means if the sample distribution is highly skewed and often
have serious misconceptions about variances. People are reasonable at quantifying opinions as credible intervals using the
fixed-interval and variable-interval methods. However, there is
a general tendency for the assessed distributions to imply a
greater degree of confidence than is justifiable. Practice, coupled with feedback, can reduce this bias, but assessing the
extreme tails of distributions (e.g., 98% credible intervals) is
difficult, and although training should reduce bias, it will not
eradicate it. Visual aids can prove useful, and task characteristics often have a marked impact on the assessments that are
elicited.


2.3 Multivariate Elicitation


When the expert’s opinion is sought on two or more unknown
variables, the output of the elicitation should be the expert’s
_joint_ probability distribution for those variables. The task is now
more complex than when eliciting a distribution for a single
variable, and the facilitator must inevitably ask more complex
questions.
An important special case is where the variables are independent, meaning that if the expert were to obtain new information about some of the variables, it would not change his
or her beliefs about the others. The concept of independence is
straightforward to explain, and independence between variables
is a relatively simple judgment for the expert to make. It is also
a very convenient judgment, because when all of the variables
are independent, their joint distribution is just the product of
their marginals. The elicitation exercise then reduces to eliciting the expert’s beliefs about each variable separately, so only
univariate elicitation techniques are required. Using independence to decompose a multivariate elicitation task into simpler
univariate tasks is consistent with the idea of disaggregation.
Psychological research also indicates that when events are
independent, joint probabilities should be assessed via univariate probabilities, because people exhibit systematic bias when
making joint probability assessments. In particular, people tend
to overestimate the probability of conjunctive events and underestimate the probability of disjunctive events. For example,
Bar-Hillel (1973) found that people tended to overestimate the
probability of drawing a red marble seven times in succession
from a bag containing 90% red marbles and 10% white marbles,
and to underestimate the probability of drawing a red marble
at least once in seven successive draws from a bag containing
10% red marbles and 90% white marbles. These errors can be
explained as the result of anchoring; the probability of an elementary event provides an obvious starting point for estimating the probability of both conjunctive and disjunctive events.
For conjunctive events, the probability of the elementary event
must be reduced, which is done insufficiently, and for disjunctive events it must be increased, which again is done insufficiently.


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 687



Discussion of the physical or historical relationships among
variables can make judgments of independence or conditional independence clear. With many elicitation methods, it
is transparent as to what assessments would correspond to independence, and in application of these methods, subjective
independence between some pairs of parameters is often observed (see examples in Garthwaite and Dickey 1991, 1992).
It should be noted, however, that assumptions of independence
often make assessment tasks easier for an expert. For example,
if an expert has assessed the marginal distribution of _X_, and
_X_ and _Y_ are independent, then the conditional distribution of
_X_ | _Y_ is easily specified as “no change.” It may be that experts
are too willing to accept independence where it does not strictly
apply.
Even where variables are dependent, it may be possible to
restructure the problem by expressing it in terms of independent variables. An example might be where we seek a medical
expert’s opinion on the effectiveness of two treatments in a clinical trial. Letting _X_ and _Y_ denote the relevant measures of effectiveness of the two treatments, we would not typically have
independence between _X_ and _Y_ . If the expert learned that _X_,
the effectiveness of the first treatment, was higher than he or she
originally expected, then this would generally lead him or her
to have an increased expectation of the effectiveness of the second treatment, _Y_ . This may be because the expert believes that
the treatments act in similar ways, but it may also be because
of uncertainty in patient recruitment. That is, if _X_ is smaller
than expected, say, this may be because the trial has recruited
patients who are more ill and will thereby suggest a smaller
value for _Y_ . However, the expert might be willing to accept independence between two _functions_ of _X_ and _Y_ . For instance,
it may be reasonable to suppose independence between _X_ and
_Z_ = _Y/X_ . Here _Z_ is the relative effectiveness of treatment 2
over treatment 1. Such a structure is often appropriate where
treatment 1 is standard care or placebo and treatment 2 is a new
or active treatment. Where both treatments are new, the asymmetry of the aforementioned structure may not be appealing,
but the expert might be happy to express independence between
_(X_ + _Y)/_ 2, the mean effectiveness, and the difference _Y_ - _X_ .
Bayesian hierarchical models are natural examples of structuring dependent variables in terms of conditional independence.
O’Hagan (1998) emphasized the role of structuring as an aid
to elicitation, and Kadane and Schum (1996) provided an extended example of complex structuring of beliefs.
In cases where variables are dependent and obviously cannot
be reduced to independence in this way, we cannot escape from
the complexity of multivariate elicitation. We can (and generally should) elicit summaries of the expert’s marginal distributions, but these no longer characterize the joint distribution
completely. The question then arises as to which summaries of
the expert’s joint distributions are most effective and reliable to
elicit.
Although statisticians usually model dependence in terms of
correlations, directly eliciting correlation coefficients or covariances might be expected to encounter at least as many problems
as directly eliciting means and variances in univariate elicitation. Psychological research has primarily considered eliciting
correlation between variables that might be considered as being
drawn from some population. For example, Clemen, Fischer,



and Winkler (2000) examined the following six methods of eliciting a correlation between weight and height in a population of
male MBA students:


1. Verbal (nonnumeric) description of the strength of a
correlation on a seven-point scale ranging from “very
strong negative relationship” to “very strong relationship.” [Clemen et al. (2000) made strong assumptions to
convert the verbal assessments to correlations.]
2. Direct assessment of the correlation by specifying a value
between −1 and 1.
3. Ask the subject to imagine that a person has been picked
at random from the population. Give the person’s percentile for one variable, and ask the subject to assess the
person’s percentile for the second variable.
4. Ask the subject to imagine that two people, A and B, have
been picked at random from the population. Conditional
on A being greater than B for one variable, ask the subject
to assess the probability that A is also greater than B for
the other variable.
5. Ask the subject to imagine that a person has been picked
at random from the population. Then ask the subject to
assess the probability that for both variables, the person is
below a specified percentile.
6. Ask the subject to imagine that a person has been picked
at random from the population. Conditional on the person
being below a specified percentile for one variable, ask
the subject to assess the probability that the person is also
below that percentile for the second variable.


Clemen et al. (2000) found that method 2 performed best.
This is surprising, because others have suggested that the direct
assessment of moments is a poor method of quantifying opinion (Morgan and Henrion 1990; Kadane and Wolfson 1998;
Gokhale and Press 1982). Method 4 asks for a concordance
probability to be assessed, which can be equated to a value of
Kendall’s _τ_ . Assumptions of normality are then made so as to
relate Kendall’s _τ_ to the Pearson correlation coefficient. Assessment of concordance probabilities to examine correlation
has been examined by Gokhale and Press (1982), who found
it preferable to alternative methods that they considered, and
by Kunda and Nisbett (1986), who concluded that reasonably
accurate correlation estimates were obtained provided that subjects were very familiar with observations from the population
in question and the data related naturally to a numeric scale.
In several experiments, subjects were shown samples from a
bivariate population and then asked to judge the “degree of relatedness” between the two variables; the subjects made use of
only a limited portion of the available data, sometimes basing
their judgments on just the proportion of time that the positive
outcome for one of the binary variables occurred with a positive
outcome for the other (Smedslund 1963; Inhelder and Piaget
1958; Jenkins and Ward 1965; Ward and Jenkins 1965).
Statistically, it is important to distinguish between eliciting
an expert’s _beliefs_ _about_ a population correlation coefficient
and eliciting the _value of_ the correlation in the expert’s beliefs
about two variables. In the first case, the correlation coefficient
is the variable whose probability distribution we wish to elicit.
This task was addressed by, for instance, Gokhale and Press
(1982). The second case is the situation that arises in multivariate elicitation, where the correlation (or some other measure


688 Journal of the American Statistical Association, June 2005



of association) is to be elicited as one summary of the expert’s
joint distribution for two variables. Where the two variables can
be considered single draws from a population, then some of the
aforementioned methods may be appropriate. However, many
cases of multivariate elicitation do not fit this situation. Consider, for example, eliciting someone’s beliefs about the fuel
economy and the acceleration of a new car. This car is not some
random draw from any population. Methods 4–6 in Clemen
et al.’s study are no longer appropriate.
None of the methods examined by Clemen et al. (2000) use
graphs in any way. It seems likely that graphical methods could
perform better, especially because it is very natural to plot a
graph to describe the relationship between two variables, such
as height and weight. This approach would represent association between variables in terms of regression, which is related to
correlation. For two variables _X_ and _Y_, for instance, we might
try to elicit the regression function _m(x)_ = _E(Y_ | _X_ = _x)_ . If the
expert accepts the proposition that this function is linear, then
we might simply elicit _m(x_ 1 _)_ and _m(x_ 2 _)_ for any _x_ 1 ̸= _x_ 2. Eliciting more than two points on the function (“overfitting”; see
Sec. 4.1) would allow for checking of the assumption of linearity, or for more accurate fitting of a straight line. Here, as
elsewhere, it may be preferable to elicit medians rather than
means.
A body of psychological research has examined multiple regression. In this research, the _x_ -variables are generally referred
to as “cues,” _Y_ is termed the “criterion,” and the regression coefficients are referred to as “cue weights.” Subjects predict the
value of the criterion, basing their predictions on the known values of the cues. In a wide variety of situations, it has been found
that a subject’s responses can be represented quite well by a
linear model that relates the criterion to the cues. The correlations between subjects’ responses and the responses predicted
by linear models (fitted to the same responses that determined
the model) have generally taken values in the .70’s when the
judgmental task is from a “real world” situation, and in the
.80’s and .90’s for less complex artificial tasks. In some studies,
the model derived from one sample of predictions was used to
forecast a second sample of predictions. The forecasts produced
in this way were only slightly less accurate than those produced by a model actually based on the second sample of predictions (Einhorn 1971; Slovic and Lichtenstein 1968; Wiggins
and Hoffman 1968). Experiments also show that, provided that
cues are monotonically related to the predicted variable, a simple linear combination of main effects will do a remarkably
good job at forecasting a subject’s assessments, even if subjects know that interactions exist. One implication is that when
eliciting the dependence of one variable on one or more other
variables, it is reasonable to constrain an expert’s assessments
to fit a linear model and to ignore interactions, unless it becomes clear that some interactions are important. An extensive
review of cue-weighting experiments was given by Slovic and
Lichtenstein (1971).
A joint distribution involves more than modeling conditional
means or medians. Hence eliciting a joint distribution is more
complex than determining an expert’s cue weights. Generally,
conditional probabilities are a natural way to augment marginal
probabilities when trying to specify a joint probability distribution, and in particular they allow conditional dispersion to be



elicited and modeled. Conditional medians and other quantiles
were extensively exploited by, for instance, Kadane, Dickey,
Winkler, Smith, and Peters (1980), and Dickey, Dawid, and
Kadane (1986), and Kadane (1996).
An alternative to conditional probabilities is joint probabilities. Having elicited _P(X_ ≤ _x)_, for instance, we might elicit
the joint probability _P(X_ ≤ _x,Y_ ≤ _y)_ or, equivalently, the conditional probability _P(Y_ ≤ _y_ | _X_ ≤ _x)_ . Note, however, that conditional probabilities are usually elicited in the form of _P(Y_ ≤
_y_ | _X_ = _x)_, and that conditioning on _X_ ≤ _x_ rather than on _X_ = _x_
may be cognitively more complex. But we have already noted
that experts do not assess joint probabilities accurately, even
when the variables are independent. We might also expect joint
probabilities like _P(X_ ≤ _x,Y_ ≤ _y)_ to be subject to a kind of
representativeness bias and so to be positively/negatively biased if the association between _X_ and _Y_ is positive/negative,
although this does not appear to have been studied.


3. FITTING A DISTRIBUTION


Once the facilitator has obtained from the expert a number
of specific statements, the elicitation task is completed by converting these into a probability distribution. Different levels of
complexity are found in the fitting of a probability distribution
to the expert’s statements. If the elicitation is to obtain a prior
distribution that will then be updated in a Bayesian analysis of
some additional data, then it is usual to fit a probability distribution using standard parametric families of distributions. But
if the elicitation is to formulate uncertainty about inputs to a
decision problem or a mathematical model, such as in the risk
assessment of a complex engineering project, then much more
simplistic elicitation and fitting are common.


3.1 Uniform and Triangular Distributions


The simplest form of elicitation is to ask the expert to specify
a range [ _a,b_ ] in which the parameter is believed to lie. If this
is all that is elicited from the expert, then it is common to assume a uniform probability distribution over [ _a,b_ ]. This can be
criticized as being too simplistic in at least two respects. First,
the expert almost certainly would not believe that the unknown
quantity in question is as likely to be very close to the limits
_a_ and _b_ as to be at a more central point in the interval. Second, unless the range [ _a,b_ ] represents absolute physical limits
to the possible values of the quantity (in which case the first
criticism applies even more strongly), it is unreasonable to give
zero probability to the event that the quantity lies outside the
range.
As a simple response to the first criticism, another common
practice is to use a triangular distribution. For this purpose, the
expert is asked to also specify a mode, say _c_ . Then the assumed
distribution has the density



_x_   - _a_
2 if _a_ ≤ _x_ ≤ _c_
_(b_ - _a)(c_ - _a)_

_b_   - _x_
2 if _c_ ≤ _x_ ≤ _b._
_(b_ - _a)(b_ - _c)_



_f (x)_ =










The acceptability of uniform and triangular distributions as representations of uncertainty about model inputs in engineering
applications is indicated by their featuring strongly in the work
of Oberkampf, Helton, Joslyn, Wojtkiewicz, and Ferson (2004),


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 689



but O’Hagan and Oakley (2004) criticized this practice as a failure to elicit adequately.
Even where substantially more information is elicited from
the expert, uniform distributions may be assumed over intervals. Suppose that in addition to the range [ _a,b_ ] the expert also
specifies probabilities _p_ 1, _p_ 2 _,...,pk_ that the uncertain quantity
lies in the intervals [ _a,c_ 1], _(c_ 1 _,c_ 2] _,...,(ck_ −1 _,b_ ]. (This may be
done by fixing the _ci_ ’s and asking for the probabilities, or by
fixing the _pi_ ’s and asking for quantiles.) Then the facilitator
may simply assign the histogram distribution

_f (x)_ = _pi_ if _ci_ −1 ≤ _x_ ≤ _ci,_ _i_ = 1 _,_ 2 _,...,k,_
_ci_       - _ci_ −1

where _c_ 0 = _a_ and _ck_ = _b_ . Although the expert’s beliefs would
generally be better represented by a distribution with smooth
density function, this histogram form may be adequate, particularly if _k_ is not small. Feedback to the expert may be useful at
this point.


3.2 Fitting Parametric Distributions


More complex elicitation methods usually impose structure
on an expert’s opinion by assuming that his or her knowledge
can be well represented by some member of a specified family
of distributions. If the expert has specified information that fits
a convenient parametric distribution, then it makes sense to use
it. This strategy has costs and benefits similar to those for statistical modeling in general. Members of the hypothesized family
are distinguished by parameters (called _hyperparameters_ ), and
the elicitation task then reduces to choosing appropriate hyperparameter values to capture the main features of the expert’s
opinion. If the expert’s opinion does not correspond approximately to any member of the family, then discrepancies are
likely to show up in the expert’s answers to elicitation questions, much as a sampling model can succumb to traditional
diagnostic checks. The family of distributions is typically chosen to be the natural conjugate family (or a tractable extension
of that family), which facilitates subsequent analysis if sample
data become available, although advances in Bayesian computation through Markov Chain Monte Carlo (MCMC) methods make it viable to use other families. For many sampling
models, the conjugate family is reasonably flexible and can represent a variety of opinion through suitable choice of hyperparameters. Further flexibility is available through mixtures of
conjugates. Dalal and Hall (1983) and Diaconis and Ylvisaker
(1985) demonstrated that such mixtures can arbitrarily accurately represent any actual belief, although we are not aware of
any work in which an expert’s opinion is elicited in terms of a
mixture, other than for variable selection problems (Garthwaite
and Dickey 1992, 1996).
Two elicitation tasks that have attracted substantial attention
are quantifying opinion about a Bernoulli process and quantifying opinion about a linear regression model. We first focus on each of these problems in turn before briefly discussing
elicitation for other sampling models. The judgmental tasks
that are central in much of this work are the assessment of
means, medians, and quantiles; revision of opinion when sample data become available; and specification of relevant aspects of a “prior sample” whose information content would
approximately equate to one’s knowledge. As a guiding principle, experts should be asked questions about quantities that are



meaningful to them. This suggests that questions should generally concern _observable_ quantities rather than unobservable
parameters, although questions about proportions and means
also might be considered suitable, because psychological research suggests that people can relate to these quantities. However, in some application areas, particular statistical models are
so familiar to experts that their parameters have acquired wellunderstood scientific meaning. It then may be appropriate to ask
experts directly about such parameters, as discussed by Kadane
(1980) and Winkler (1980). The facilitator should always try to
understand what terms make the expert most comfortable for
elicitation.
Four basic methods have been used to quantify subjective
opinion about _p_, the unknown parameter of a Bernoulli process.
In illustrating assessment questions, we suppose that _p_ is the
proportion of students at the University of Chicago who are
male, which is the example used by Winkler (1967), the first
author to address this elicitation problem:


1. One method is to ask the expert to specify his or her median estimate of _p_ and to give one or more quantiles (usually at least two) of his or her subjective distribution for _p_ .
These may be plotted and a smooth cumulative distribution function drawn through them, giving a nonparametric
representation of the expert’s opinion. More commonly,
it is assumed that the expert’s opinion can be well represented by a beta distribution (the conjugate distribution
for Bernoulli sampling), and a beta distribution is selected
whose quantiles are similar to those that the expert gave.
The beta distribution might be selected using a table presented by Winkler (1972, table 5) that lists several quantiles for a variety of parameter values. We call this method
of elicitation the _quantile_ _method_ ; it is also often called
the credible interval method.
2. The second method is the _hypothetical_ _future_ _sample_
(HFS) method. The expert first estimates the proportion
under consideration (e.g., the proportion of students who
are male) and then revises his or her opinion in the light of
information from additional (hypothetical) samples. For
example, he or she might be asked questions of the following form: “Suppose that a random sample of 50 students was taken and 20 of them were male. Now what
is the probability that one additional student, chosen at
random, is male?” Again it is assumed that the expert’s
opinion corresponds to a beta distribution; its parameters
are uniquely determined by the expert’s prior and posterior (given the hypothetical sample data) estimates of
the proportion. In general, the expert is confronted with
several hypothetical samples in which the number of students and the proportion of these who are male is varied.
Each sample yields a separate estimate of the hyperparameters, and some form of averaging is used to reconcile
them. A general issue arises here in the case where the resulting hyperparameter estimates are very disparate. This
may reflect assessment inaccuracy; that is, the beta distribution may be correct and the elicitation method may be
the best available, but the expert’s answers to questions
are subject to substantial variability. It also warns that the
elicited distribution could be a poor representation of the
expert’s opinion, either because his or her opinion does


690 Journal of the American Statistical Association, June 2005



not correspond to a beta distribution or because the elicitation method asks questions that are hard to answer accurately. (See also the discussion of internal consistency
in Sec. 4.1.)
3. The third method is the _equivalent_ _prior_ _sample_ (EPS)
method, in which an expert expresses his or her knowledge as an equivalent prior sample. Quoting Winkler
(1967, p. 779), “you have some knowledge concerning
University of Chicago students. Can you determine two
numbers _r_ and _n_ such that your knowledge would be
roughly equivalent to having observed exactly _r_ males in
a random sample of _n_ University of Chicago students, assuming that you had very little knowledge about the sex
of University of Chicago students before seeing this sample?” The prior distribution is taken to be a beta distribution with parameters _r_ and _n_   - _r_ .
4. The fourth method is the _probability_ _density_ _function_
(PDF) method, in which the expert specifies the most
likely value of _p_, say _p_ �, and then assesses other points of
the pdf for _p_ . For example, Winkler (1967) asked the expert to give two values of _p_ (one on each side of _p_    - ) that
he or she considers half as likely as _p_   - and to also specify quantiles, which in this context are defined as points
that divide the area under the graph of the pdf in specified proportions. (There is obviously strong similarity between the PDF method and the quantile method, because
both make use of quantile assessments.) A nonparametric
estimate of the expert’s opinion may be obtained by asking the expert to draw a graph of his or her pdf, taking into
account the assessments that he or she has given.


All four of these methods tend to produce prior distributions
that are unrealistically “tight” (i.e., they have variances that are
too small). For example, Schaefer and Borcherding (1973) conducted an experiment in which 22 subjects used the EPS and
quantile methods to quantify their opinions about various proportions. Before any training, each subject assessed 50% central credible intervals for 18 different proportions using each
method. The proportion of these intervals that contained true
values was 15.7% for the EPS method and 22.5% for the quantile method. The task in the HFS method, revising opinion in
the light of additional data, is similar to the task in the “bookbag
and poker chips” experiment, where conservatism has a marked
effect. Insufficient revision of opinion when given the hypothetical data would result in a distribution that is too tight. Winkler
(1967) felt that conservatism also influenced the EPS method,
hypothesizing that subjects equated their knowledge to too large
a sample size, through not realizing the value of sample information. The quantile method tends to yield distributions that are
again too tight, but slightly less tight than the PDF method and
much less tight than the HFS and EPS methods (Winkler 1967).
On this basis, the quantile method seems preferable, and it also
seems to be the method of choice when judged by scoring rules.
(Scoring rules are discussed in Section 4.3.) Some experiments
have examined the effect of training and found that the bias of
tightness was reduced for all three methods with particularly
marked improvement for the EPS method (Stael von Holstein
1971; Schaefer and Borcherding 1973).
Elicitation methods have also been devised that avoid direct questions about _p_ (which is not an observable quantity)



and instead ask questions about sampling distributions, such
as “the number of students who would be male in a random
sample of 20 University of Chicago students.” Chaloner and
Duncan (1983) used this form of question in a variant of the
PDF method. The expert states the most likely number of males
in the sample, say _x_, and then assesses the relative likelihood
of _x_ males rather than _x_ - 1 males, and of _x_ males rather than
_x_ + 1 males. Assessments are used to estimate the parameters
of a beta distribution, and the implied endpoints of the shortest
50% predictive interval are then calculated. These endpoints are
given as feedback to the expert, who comments on the length
of the interval. If it is found to be too short or too long, then
the parameter estimates are revised repeatedly until he or she
is satisfied with its length. The process is repeated for a variety of sample sizes, and the resulting parameter estimates are
amalgamated in some way. Gavaskar (1988) also used questions
about the sampling distribution in a variant of the HFS method.
The expert first specifies the most likely number of males in
a random sample of some specified size (as in the method of
Chaloner and Duncan), then revises his or her assessment after
being given hypothetical sample data, as in the HFS method.
Assessments are again used to determine the parameters of a
beta distribution. This is done for a variety of sample sizes.
Interestingly, Gavaskar (1988) conducted a computer simulation to assess the sensitivity of parameter estimates to errors
in an expert’s assessments. He compared his method with that
of Chaloner and Duncan (1983) and found that his method is
much less sensitive. However, in a simulation study of this nature it is difficult to choose appropriate error distributions for
the different methods, and Gavaskar may have underestimated
the magnitude of errors induced by hypothetical samples.
Turning to multiple linear regression, we suppose that the
model is _Y_ = **x** [′] _**β**_ + _ϵ_, where _ϵ_ ∼ N _(_ 0 _, σ_ [2] _)_ . The usual conjugate
prior distribution specifies that _σ_ [2] equals _ωδ_ times the reciprocal of a chi-squared random variable with _δ_ degrees of freedom,
and that given _σ_, _**β**_ has a normal distribution with some mean **b**
and some variance–covariance matrix _σ_ [2] **R** . For this prior distribution, _ω_, _δ_, **b**, and **R** are the hyperparameters that need to
be assessed via an expert’s assessments.
To quantify opinion about these quantities, Zellner (1972)
suggested questioning an expert about the regression coefficients. Some experts may be able to think of regression coefficients directly, but, as noted earlier, it is usually better to
question people about observable quantities, such as _Y_, rather
than ask direct questions about unobservable quantities, such
as regression coefficients. Toward this end, elicitation methods
generally ask the expert about observations _Y_ 1 _,...,Ym_ at values **x** 1 _,...,_ **x** _m_ of **x** . But this has the possible disadvantage that
uncertainty about _Y_ results both from the expert’s uncertainty
about the values of the regression coefficients and from random
variation. To separate these sources of uncertainty, Garthwaite
and Dickey (1988) asked questions about _Y i_, the average value
of _Y_ if a large number of observations were taken at a single
value, **x** _i_, arguing that averages are quantities to which people can relate and that an expert can give assessments about _Y i_
without the need to consider random error. We briefly outline
the elicitation methods that have been proposed by Kadane
et al. (1980), Oman (1985), and Ibrahim and Laud (1994),
all of which question an expert about _Y_, and the method of


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 691



Garthwaite and Dickey (1988). In this section we refer to these
articles as KDWS&P, Oman, I&L, and G&D.
Let **X** = _(_ **x** 1 _,...,_ **x** _m)_ [′] . We refer to **x** _i_ as a design point
and **X** as a design matrix. To estimate the mean vector, **b**, at
each design point KDWS&P elicited the median of _Y_, G&D
elicited the median of _Y_, Oman elicited the mean of _Y_, and I&L
elicited the expert’s “best guess” (or prior prediction) of _Y_ . Denoting the vector of assessments by **y**, the methods equate **b**
to _(_ **X** [′] **X** _)_ [−][1] **X** [′] **y**, which would be the least squares estimate if
**y** were a vector of observations at **X** . (Oman also offered two
less attractive alternative methods of eliciting **b** : asking the expert to assess **b** directly and asking the expert to specify prior
expected covariances.) With each elicitation method, the distribution of _Y_ at any design point is unimodal and symmetric, so
that in principle, it should not matter which point estimate of _Y_
is assessed. In practice, however, the distribution of _Y_ may be
skew. Then the (unanswered) question arises as to which feature of a skew distribution should correspond to the point of
symmetry of a symmetric distribution that is used to represent
it.
To elicit _ω_ and _δ_, G&D and I&L elicited assessments that
depend only on experimental error. In the approach of G&D,
the expert is asked to suppose that two observations are taken
at the same design point. It is pointed out to the expert that
the observations will not be identical because of random variation, and the median of their absolute difference is elicited.
The expert is then given a hypothetical datum, and he or she
then states the updated median of their absolute difference.
The approach of I&L uses assessments of the mean and variance of the precision ( _σ_ [−][2] ) to determine _ω_ and _δ_, and in related work (Laud and Ibrahim 1995), I&L used assessments
of the median and the 95th percentile of the distribution of
the precision. Oman did not elicit _ω_ and _δ_, and restricted his
posterior analysis to inferences that depend only on a point
estimate of _σ_ [2], which he obtained using empirical Bayes methods. KDWS&P elicited _δ_ by asking the expert to assess the median ( _y._ 50), upper quartile ( _y._ 75), and 93.75th percentile ( _y._ 9375)
at a design point. ( _y._ 9375 is elicited through repeated bisections, _y._ 50 −→ _y._ 75 −→ _y._ 875 −→ _y._ 9375.) The ratio _(y._ 9375 _y._ 50 _)/(y._ 75 - _y._ 50 _)_ depends only on _δ_ and hence provides an
estimate of it. KDWS&P repeated the assessments at several
design points and averaged the estimates of _δ_ that each set of
assessments yielded. The method used by KDWS&P to elicit
_ω_ is complex; see that article for details. A central task is to
ask the expert to suppose that two independent observations are
taken at a specified design point. The expert assesses the median
of one of the observations, then, given a hypothetical value for
that observation, assesses the conditional median of the other
observation.
All of the aforementioned methods may be criticized for using assessment tasks that people are not very good at performing. Direct questions about the distribution of a precision (I&L)
are surely hard to answer; G&D used conditional assessments
that will be biased by conservatism and also elicited only the
minimum number of assessments to determine the hyperparameters; KDWS&P found that extreme values of _δ_ were not
uncommon with their method. Because _δ_ is a difficult hyperparameter to assess, it is a good idea to elicit more than one
estimate of it and then reconcile the different estimates in some



is set equal to _yi,._ 50| _y_ 1 [0] _[,...,y]_ _i_ [0] −1 [+] [1] 2 [{] _[S(Y][i]_ [|] _[y]_ 1 [0] _[,...,y]_ _i_ [0] −1 [}][1] _[/]_ [2][.]

An iterative method is used to calculate **P** . Let **P** _i_ denote
the covariance matrix of _Y_ 1 _,...,Yi_ and put **P** 1 = _S(Y_ 1 _)_ . The
following equations give **P** _i_ +1 after **P** 1 _,...,_ **P** _i_ have been estimated. Put

    **L** _i_ +1 = _(yi_ +1 _,._ 50| _y_ 1 [0] _[)]_ [ −] _[y]_ _i_ [∗] +1 _[,...,]_

�′
_(yi_ +1 _,._ 50| _y_ 1 [0] _[,...,y]_ _i_ [0] _[)]_ [ −] _[y]_ _i_ [∗] +1 _,_ (3)



way. If estimates of _δ_ are to be combined arithmetically, then
empirical evidence favors taking their geometric mean rather
than their average (Al-Awadhi and Garthwaite 2001).
The most complicated hyperparameter to elicit is the variance–covariance matrix, _σ_ [2] **R** ; this may contain a large number
of elements that each need to be elicited, and the matrix itself
must be positive definite. In KDWS&P, a crucial step in assessing this matrix is the elicitation of a variance–covariance
matrix for a multivariate _t_ -distribution. We next describe that
part of their method in some detail, because it can be useful in
a variety of elicitation problems. Also, it requires sophisticated
mathematics and statistics, so it illustrates that statisticians need
to be involved in the development of elicitation methods; in the
past some statisticians have suggested that their development
should be left to psychologists.
Suppose that **Y** = _(Y_ 1 _,...,Ym)_ [′] has a multivariate _t_ -distribution on _δ_ degrees of freedom, **Y** ∼ _tδ(_ **a** _,_ **P** _)_ . Then **P** is referred
to as the _spread_ of **Y**, _S(_ **Y** _)_, and if _δ >_ 2, then the variance of
**Y** equals [ _δ/(δ_ - 2 _)_ ] **P** . (The variance does not exist if _δ <_ 2.)
To elicit **P**, KDWS&P proceeded as follows.
The expert assesses the following:

(a) His or her medians of _Y_ 1 _,...,Ym_, denoted by _y_ 1 [∗] _[,...,y]_ _m_ [∗] [.]
(b) The upper quartile of _Y_ 1, denoted by _y_ 1 _,._ 75.
(c) Conditional medians. A conditional set of values is built
up in stages, and at the _i_ th stage ( _i_ = 1 _,...,m_ - 1), it consists
of values _y_ 1 [0] _[,...,y]_ _i_ [0][.] [Given] _[Y]_ [1] [=] _[ y]_ 1 [0] _[,...,Y][i]_ [=] _[ y]_ _i_ [0][,] [the] [expert]
gives the conditional median of _Yj_ for _j_ = _i_ + 1 _,...,m_, the
assessment being denoted by _yj,._ 50| _y_ 1 [0] _[,...,y]_ _i_ [0][.]
(d) Conditional quartiles. Given _y_ 1 [0] _[,...,y]_ _i_ [0][,] [the] [expert] [as-]
sesses _yi_ +1 _,._ 75| _y_ 1 [0] _[,...,y]_ _i_ [0][,] [the] [conditional] [upper] [quartile]
of _Yi_ +1.


In elicitation methods, the most widely used method of estimating variances or spreads from quartile assessments is to
divide the assessed interquartile range or semi-interquartile
range by the corresponding range of a standard distribution.
Let _t_ [ _δ,._ 75] denote the semi-interquartile range of a univariate _t_ -distribution on _δ_ degrees of freedom. The variance of _Yi_
does not exist if _δ <_ 2, so KDWS&P determined spreads,

_S(Y_ 1 _)_ = _(y_ 1 _,._ 75 − _y_ 1 [∗] _[)]_ [2] _[/(t]_ [[] _[δ,.]_ [75][]] _[)]_ [2] (1)


and


_S(Yi_ +1| _y_ 1 [0] _[,...,y]_ _i_ [0] _[)]_

1 _[,...,y]_ _i_ [0][}] [−{] _[y][i]_ [+][1] _[,.]_ [50][|] _[y]_ 1 [0] _[,...,y]_ _i_ [0][}] _[)]_ [2]
= _[(]_ [{] _[y][i]_ [+][1] _[,.]_ [75][|] _[y]_ [0] _,_ (2)

_(t_ [ _δ_ + _i,._ 75] _)_ [2]



for _i_ = 1 _,...,m_ - 1. The order of the conditional assessments
enables conditions to be based on an expert’s earlier answers;
_y_ 1 [0] [is set equal to] _[ y]_ 1 [∗] [+] [1] 2 [{] _[S(Y]_ [1] _[)]_ [}][1] _[/]_ [2][ and for] _[ i]_ [ =][ 2] _[,...,m]_ [−] [1,] _[ y]_ _i_ [0]



_y_ 1 [0] [is set equal to] _[ y]_ 1 [∗] [+] [1] 2 [{] _[S(Y]_ [1] _[)]_ [}][1] _[/]_ [2][ and for] _[ i]_ [ =][ 2] _[,...,m]_ [−] [1,] _[ y]_ _i_ [0]

is set equal to _yi,._ 50| _y_ 1 [0] _[,...,y]_ _i_ [0] −1 [+] [1] 2 [{] _[S(Y][i]_ [|] _[y]_ 1 [0] _[,...,y]_ _i_ [0] −1 [}][1] _[/]_ [2][.]




    **L** _i_ +1 = _(yi_ +1 _,._ 50| _y_ 1 [0] _[)]_ [ −] _[y]_ _i_ [∗] +1 _[,...,]_


692 Journal of the American Statistical Association, June 2005



**T** _i_ +1


=


and









_y_ 1 [0] [−] _[y]_ 1 [∗] _(y_ 2 _,._ 50| _y_ 1 [0] _[)]_ [ −] _[y]_ 2 [∗] _..._ _(yi,._ 50| _y_ 1 [0] _[)]_ [ −] _[y]_ _i_ [∗]
_y_ 1 [0] [−] _[y]_ 1 [∗] _y_ 2 [0] [−] _[y]_ 2 [∗] _..._ _(yi,._ 50| _y_ 1 [0] _[,y]_ 2 [0] _[)]_ [ −] _[y]_ _i_ [∗]
_..._ _..._ _..._ _..._
_y_ 1 [0] [−] _[y]_ 1 [∗] _y_ 2 [0] [−] _[y]_ 2 [∗] _..._ _yi_ [0] [−] _[y]_ _i_ [∗]



−1





× **L** _i_ +1 _,_ (4)



_S(Yi_ +1| _y_ 1 [0] _[,...,y]_ _i_ [0] _[)]_ [ · [][1][ +] _[ i/δ]_ []]
_S(Yi_ +1 _)_ =

1 + _δ_ [−][1] _(y_ 1 [0] _[,...,y]_ _i_ [0] _[)]_ [′] **[P]** _i_ [−][1] _[(y]_ 1 [0] _[,...,y]_ _i_ [0] _[)]_

+ **T** [′] _i_ +1 **[P]** _[i]_ **[T]** _[i]_ [+][1] _[.]_ (5)


Then put



A feature common to the methods of KDWS&P and G&D is
that a structured set of sequential questions is used to ensure
that **P** is a positive-definite matrix.
A drawback of the KDWS&P method is that for many of
its tasks, the expert must update his or her beliefs on the basis of hypothetical data, so assessments are likely to be biased
by conservatism. Al-Awadhi and Garthwaite (1998) suggested
a modification whereby the diagonal elements of **P** are estimated from unconditional assessments. This modification involves scaling **P** in such a way that estimates of correlations are
unchanged, so **P** is still certain to be a positive-definite matrix
while the impact of conservatism should be reduced. Al-Awadhi
and Garthwaite also suggested estimating the spreads of univariate distributions from assessments of both the lower and
upper quartiles (KDWS&P used medians and upper quartiles),
so that an estimated spread reflects both halves of the subjective distribution. Eliciting both quartiles also enables marked
asymmetry to come to light.
Turning to elicitation methods for other sampling models,
Al-Awadhi and Garthwaite (1998) gave a method of eliciting
a conjugate prior distribution for sampling from a multivariate normal distribution. This work was extended by Garthwaite
and Al-Awadhi (2001) so as to elicit a more flexible nonconjugate prior distribution. The methods follow the approach of
KDWS&P to assess spread matrices (with the modifications
mentioned earlier) and follow the approach of G&D to elicit
degrees of freedom parameters. The method of KDWS&P was
also exploited by Dickey et al. (1986) to develop assessment
methods for matrix- _t_ models, and a closely related method
was used by Garthwaite and Al-Awadhi (2003) for logistic
regression. An elicitation method for logistic regression was
also given by Chen, Ibrahim, and Yiannoutsos (1999), using
ideas similar to those of I&L. Chaloner and Duncan (1987) extended their method of quantifying opinion about a Bernoulli
process (Chaloner and Duncan 1983) so as to elicit a Dirichlet
distribution that represents prior opinion about a multinomial
sampling model. Other models that have attracted attention include the proportional hazards model (Chaloner, Church, Matts,
and Louis 1993), Weibull lifetime distributions (Singpurwalla
and Song 1987), AR(1) time series models (Kadane, Chan, and
Wolfson 1996), and ANOVA models (Black and Laskey 1989).
Graphical feedback is an important component in the methods
of Chaloner and Duncan (1983, 1987), Chaloner et al. (1993),
and Garthwaite and Al-Awadhi (2003), and it seems to provide
a potentially powerful means of improving the quality of assessed distributions.
Almost all of the foregoing methods represent expert opinion by some form of conjugate distribution. This approach
has limitations, however, notably when the sampling model
is a multivariate normal distribution. In a frequentist analysis for a multivariate normal distribution, the sample variance–
covariance matrix is both an estimate of the population variance
and, after division by the sample size, it is also the estimated
variance of the sample mean. In the standard conjugate distribution, a single variance matrix again fulfills both of these roles.
An experiment by Al-Awadhi and Garthwaite (2001) demonstrated that this is inappropriate. The experiment examined different forms of assessment task and compared alternative ways
of estimating hyperparameters. To quantify opinion about the




   **P** _i_ +1 = **P** _i_ **P** _i_ **T** _i_ +1
**T** [′] _i_ +1 **[P]** _[i]_ _S(Yi_ +1 _)_





_._ (6)



The procedure stops when **P** _m_ = **P** has been obtained. Results
of KDWS&P show that **P** is certain to be a positive-definite
matrix.
A general feature of the method of KDWS&P is that for
every hyperparameter, it elicits more assessments than necessary, then uses some form of averaging to obtain hyperparameter estimates. This is obviously sound practice. Also, the largest
deviations from the averages are reported to the expert, so that
he or she can judge the extent to which some of his or her answers may be in error and require changing. The **x** -values used
in the elicitation method will affect the quality with which expert opinion is captured; Kadane and Wolfson (1998) have suggested a procedure for their selection.
Oman did not attempt to relate **R** to the expert’s opinion. Instead, he chose design points that cover the region of interest
and that gave a design matrix **X** that is as near orthogonal as
possible. He then set **R** equal to _τ(_ **X** [′] **X** _)_ [−][1] and estimated _τ_ using empirical Bayes methods. I&L were concerned with the
analysis of designed experiments. They let **X** be the design
matrix for which data will be gathered and, like Oman, assumed that **R** = _τ(_ **X** [′] **X** _)_ [−][1] . The hyperparameter _τ_ was then
chosen to reflect the weight that should be attached to the expert’s opinion, relative to the weight that should be attached
to the data from the experiment. Hence, I&L chose **R** partly
to reflect the expert’s knowledge, but their approach is pragmatic and is not a serious attempt to determine the prior variance of _**β**_ . (Otherwise, their approach implies that prior opinion
about the relationship between _Y_ and the _x_ -variables depends
on the experiment to be conducted.)
G&D developed an alternative method of eliciting **P**, the
spread of **Y** . It is based on a novel assessment task that requires
the expert to first select the design point at which he or she
can predict _Y_ most accurately, and then repeat this task several
times with an increasing set of restrictions on the _x_ -values that
he or she can choose. The method was developed for the use
of industrial chemists and exploits their experience in choosing design points to conduct experiments. It is not as flexible as
the method of KDWS&P, in that it cannot be used with polynomial regression or with _x_ -variables that are factors. However,
it can be extended to elicit prior distributions that are suitable
for variable selection problems (Garthwaite and Dickey 1992).


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 693



vector of means, it proved preferable to ask directly about the
means rather than individual observations, whereas, to quantify
opinion about the variance matrix, it was better to ask about
deviations from the mean. One alternative to the conjugate distribution is to assume that the population mean and variance are
independent in the prior distribution, an approach followed by
Garthwaite and Al-Awadhi (2001).
Elicitation methods for broader classes of problem have also
been proposed. Bedrick, Christensen, and Johnson (1996) suggested a method for generalized linear models in which the predictive distributions at different design points are elicited and
then combined to form a prior distribution. For convenience,
Bedrick et al. (1996) mainly considered the case where the predictive distributions are independent of each other, so that combining them is straightforward. They discussed the properties
of their method and showed its similarities to data augmentation. Clemen and Reilly (1999) considered the general problem
of constructing a joint prior distribution for several hyperparameters. They used a copula to form the joint distribution from
an expert’s subjective judgments of marginal distributions and
correlations. No restrictions are placed on the marginal distributions, and dependence between the marginals is modeled by the
copula that underlies a multivariate normal distribution. People
are not good at assessing correlations; Clemen and Reilly addressed this problem by discussing various techniques for their
assessment. They reported a small empirical study that forms
part of the basis for their well-informed views; Clemen et al.
(2000) reported a larger study of methods for assessing correlations, as discussed in Section 1.2.3.


3.3 Nonparametric Fitting


Any fitting of a parametric distribution to the expert’s stated
summaries implies assumptions about the form of the expert’s
underlying probability distribution. Although the distributional
form may be acceptable to the expert, he or she is rarely in
a position to question the assumptions critically. This is even
more true in the case of multivariate distributions.
Just as there are many statisticians who are uncomfortable
with parametric assumptions in modeling data, it is arguably
preferable in elicitation not to make parametric assumptions.
(Indeed, in the subjective Bayesian framework, the likelihood
is just as much a judgment as the prior distribution, and in principle should also be formally elicited.) A number of nonparametric approaches are possible in elicitation.
One such approach is simply to decline to fit a distribution
at all and use just the expert’s stated summaries and nothing
more. It is then necessary to find ways to make use of this limited specification of beliefs. In the context of eliciting a prior
distribution for a Bayesian analysis, _Bayes linear_ methods have
been advocated by Goldstein (1999). The Bayes linear approach
is based on eliciting only first- and second-order moments (i.e.,
means, variances, and covariances). The underlying theory is a
Bayesian analog of the Gauss–Markov theorem, but advocates
of the Bayes linear approach have developed a substantial body
of methodology to facilitate applications. Bearing in mind the
difficulties in asking experts to assess moments that we noted
earlier, the Bayes linear approach places higher demands on the
statistical understanding of the expert, or else requires a substantial training input.



Berger and O’Hagan (1988) effectively allowed the expert’s
prior distribution for a single unknown parameter to be any unimodal distribution having specified quantiles. They computed
the range of posterior inferences over this range of prior distributions for given data. This is a fully nonparametric approach,
although it may be difficult to implement in more complex situations. It also allows distributions that the expert would easily be able to reject as inaccurate representations of his or her
knowledge.
A more recent approach is that of Oakley and O’Hagan
(2002), which adopts the framework of modern Bayesian nonparametrics. The expert’s beliefs about the random variable _X_
are supposed to be represented by a probability density function
_f (x)_ . To the facilitator, _f_ is an unknown function. The facilitator has a prior distribution for _f_, and this is updated to a posterior distribution after “observing” the “data” comprising the
expert’s summary values. It is worth looking at this formulation
in a little more detail.
Because _f_ is a function, the facilitator’s prior is a distribution over the space of possible values of the whole function.
Formally, Oakley and O’Hagan (hereafter O&O) supposed that
_f_ has a Gaussian process prior distribution. Its prior expectation is a member _g(x_ ; _θ)_ of some parametric family indexed
by _θ_ . The prior variance of _f (x)_ is _g(x_ ; _θ)_ [2] _σ_ [2], and the correlation between _f (x)_ and _f (x_ [′] _)_ is a decreasing function of
the distance between _x_ and _x_ [′] . The hyperparameters _θ_ and _σ_ [2]

are given weak prior distributions. The method is nonparametric because _f_ is not assumed to be exactly a member of any
parametric family, and indeed it is allowed to take any form.
Nevertheless, the facilitator expects the expert’s true density
function _f_ to be close, in some sense, to some member of the
parametric family, defined by _g_ . The closeness is governed by
the hyperparameter _σ_ [2], which O&O learn about from the expert’s summary “data.” The correlation in the Gaussian process
ensures that departures of _f (x)_ from _g(x_ ; _θ)_ are smooth.
Although in this approach the facilitator does not constrain
the expert’s distribution to fit the parametric family, it is clear
that the facilitator is supplying some information through the
Gaussian process prior distribution. There is the belief that
_f_ should be “close” to _g_, and that it should be a “smooth” density function. Technically, from the assumption of a Gaussian
process, the facilitator has a normal distribution for each _f (x)_ .
Strictly speaking, this cannot be completely realistic, because
_f (x)_ must be nonnegative, but the normality is important for
tractability of the O&O approach, and O&O argued that it is
relatively innocuous.
A significant benefit of the O&O approach is that it yields
a complete posterior distribution for _f_ . The posterior mean
can be considered an estimate of _f_ based on the expert’s summaries, and hence as the elicited probability distribution. But
O&O then have a posterior distribution that quantifies the possible inaccuracy of the elicitation, as called for by Dickey (1980).
They gave the following synthetic example in which the expert’s true probability distribution is a bimodal mixture of two
normal distributions. The parametric family _g_ is the normal
family. Figure 2 shows the posterior median and 95% pointwise credible intervals for _f_ based on seven elicited quantiles
from the expert. We see that, despite believing initially that the
expert’s distribution would be similar to a normal distribution,
the facilitator’s posterior distribution accurately reproduces the
expert’s true bimodal _f_ .


694 Journal of the American Statistical Association, June 2005



Figure 2. Example From Oakley and O’Hagan (2002): Median and
Pointwise 95% Intervals for the Expert’s Density Function (—) and the
True Density Function (···).


4. TESTING ADEQUACY OF ELICITATION


In view of the many practical difficulties of elicitation, how
can one know whether the elicited distribution is in any sense an
_adequate_ representation of the expert’s knowledge? Before addressing this question, we should consider whether there is, in
some sense, a “true” representation. Does the expert have a true
personal probability distribution for the uncertain quantities?
Winkler (1967, p. 778) noted that “the assessor has no builtin prior distribution which is there for the taking. That is, there
is no ‘true’ prior distribution. Rather, the assessor has certain
prior knowledge which is not easy to express quantitatively
without careful thought. An elicitation technique used by the
statistician does not elicit a ‘true’ prior distribution, but in a
sense helps to draw out an assessment of a prior distribution
from the prior knowledge. Different techniques may produce
different distributions because the method of questioning may
have some effect on the way the problem is viewed.”
In contrast O’Hagan (1988) explicitly defined “true” probabilities as those that would result if the expert were capable
of perfectly accurate assessments of his or her own beliefs.
O’Hagan regarded different “stated” probabilities that might
result from different elicitation methods as more or less inaccurate attempts to specify the expert’s underlying “true” probabilities. This differs from Winkler’s position, which seems to
be that the results of different elicitations are all assessments of
slightly different probabilities. A possible reconciliation is that
a “true” distribution would be the result of a method that leads
the expert to view the problem from as complete and unbiased
a perspective as possible through appropriate use of cognitive
tools.
In this section we first consider how to test the internal consistency of the expert’s statements, together with any assumptions made by the facilitator. We then discuss assessing the
adequacy of the elicitation, in terms of whether the acknowledged inaccuracies in the elicitation process matter.


4.1 Internal Consistency


A system of probability statements is _coherent_ if the probabilities are all consistent with the laws of probability. If, for



instance, an expert states that _P(E)_ = _._ 4, _P(F)_ = _._ 3, and
_P(E_ or _F)_ = _._ 6, when _E_ and _F_ are mutually exclusive events,
then these probabilities are noncoherent. One way to check the
quality of an expert’s statements is for the facilitator to ask for
sets of probability assessments that allow tests of coherence. We
must expect that the expert’s elicited statements will fail coherence tests. This is almost inevitable in view of the imprecision
with which the expert can make these judgments. The question
then arises of how we should reconcile the internal inconsistency of the elicited statements.
In the case of an incoherent set of individual probabilities,
as in the example of mutually exclusive events, the simplest
answer is to confront the expert with the inconsistency and to
invite him or her to revise one or more of the stated values. In
general, we should expect this revision to improve the expert’s
assessments.
A careful examination of reconciling incoherent probability
assessments was given by Lindley, Tversky, and Brown (1979,
hereafter LT&B). In their approach, reconciliation is done by
the person that we have called the facilitator. The facilitator
takes a view of the accuracy with which the expert will have
been able to assess the stated probabilities. Thus, in the example of mutually exclusive events, the facilitator needs to formulate a joint probability distribution, _p(_ **e** _,_ **s** _)_, for the expert’s
underlying true probabilities, **e** = _(P(E),P(F))_, and for the
expert’s stated probabilities, **s**, for the three events _E_, _F_, and
“ _E_ or _F_ .” The facilitator’s beliefs about the expert’s true probabilities would then be expressed by _p(_ **e** | **s** = _(._ 4 _,._ 3 _,._ 6 _))_ . In
practice, LT&B envisage the facilitator formulating the joint
distribution via a prior distribution _p(_ **e** _)_ for **e** and a “likelihood,”
_p(_ **s** | **e** _)_, for the expert’s assessment errors, so that the facilitator’s posterior distribution, _p(_ **e** | **s** = _(._ 4 _,._ 3 _,._ 6 _))_, is derived by
Bayes’ theorem. LT&B refered to this solution as the “internal
approach” and contrasted it with an “external approach” that we
consider in Section 5.1.
As in the simpler method of asking the expert to revise his or
her own probabilities, this reconciliation can lead to more accurate assessments. However, the improvement is now formally
expressed by the familiar Bayesian property that the facilitator’s
posterior distribution of **e** will generally be more concentrated
than the prior distribution.
In the context of eliciting a parametric probability distribution, sometimes only as many summaries are elicited as are
required to identify unique values of the required hyperparameters. It is then usually the case that any set of elicited summaries will be consistent with the fitted distribution, and hence
with each other. It is therefore not possible to find any noncoherence. However, if the facilitator asks for at least one more
summary from the expert, then it becomes possible to test for
coherence. This has been called _overfitting_ .
Note, however, that inconsistent statements from the expert
may simply indicate that the expert’s distribution cannot be adequately represented by a member of the chosen family. For
instance, if the expert’s beliefs are sought concerning a proportion _π_, then the facilitator might choose to work with the assumption of a beta distribution. Then if the expert specifies that
the median is .4, the lower quartile is .3, and the upper quartile is .5, no beta distribution will fit these specifications. The
_Be(_ 4 _._ 733 _,_ 7 _._ 1 _)_ distribution fits the median and lower quartile,


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 695



but has an upper quartile of .494, whereas the _Be(_ 4 _._ 16 _,_ 6 _._ 24 _)_
distribution fits the median and upper quartile but has a lower
quartile of .293. But it is unreasonable to expect the expert to
specify these quantiles to such accuracy, and the assessments in
practice would not be seen as challenging the assumption of a
beta distribution. Instead, a compromise such as _Be(_ 4 _._ 4 _,_ 6 _._ 6 _)_,
with lower quartile .296 and upper quartile .497, clearly fits the
elicited values very well.
If, on the other hand, the expert had specified a lower quartile of .2, then the beta family assumption is called into question. The expert’s distribution appears to be negatively skewed
but with a median below .5, two properties that are together
inconsistent with a beta distribution. In general, given a set of
expert statements that is larger than needed to identify unique
hyperparameters, we can choose an elicited distribution that in
some sense fits the elicited statements as closely as possible;
an example of this was given by O’Hagan (1998). The quality
of fit can be seen as indicating the accuracy of the elicitation,
whereas a sufficiently poor fit casts doubt on the assumed family of distributions.
In the case of parametric elicitation, then, overfitting has the
potential to either refine the specification of hyperparameters or
refine the assumed distributional family. Notice, however, that
to decide between these two options, the facilitator needs to
have some idea of the accuracy of the expert’s judgments. Given
that this kind of judgment by the facilitator is required, it may be
that an extension of the approach of LT&B might be developed
for this case, but we are not aware of any published work in this
direction.
O&O’s nonparametric elicitation method of Section 3.3 also
adopts essentially the approach of LT&B, in the sense that the
facilitator derives a posterior distribution for the expert’s underlying density function, using the elicited statements as data.
Note, however, that O&O treated the expert’s summaries as
error-free, and so they did not consider an analog of the likelihood _p(_ **s** | **e** _)_ .
An idea similar to overfitting is feedback. For instance, in a
parametric elicitation when we have elicited enough summaries
to fit a unique member of the chosen family, instead of eliciting one or more further summaries, the facilitator informs the
expert of the values of those summaries that are implied by the
expert’s statements so far (and the assumed distributional form).
In general, feedback entails displaying the implications of other
statements and inviting the expert to confirm or deny that these
are reasonable expressions of his or her beliefs. Whereas overfitting will almost invariably expose inconsistencies in the expert’s statements, feedback often simply results in the expert
confirming the implied values. As such, overfitting is generally
preferred, but feedback can be very useful to show complex implications, such as displaying the fitted density function graphically.
It is also worth noting that the expert will often make qualitative statements during elicitation that can be checked informally
against quantitative summaries. For instance, the expert may
appear uncertain and have difficulty specifying a credible interval, and yet may actually give a narrower interval than in another task where he or she informally indicated more certainty.
The facilitator should be alert for any opportunity to assist
the expert by checking the consistency of his or her opinions,
whether expressed or implied.



4.2 Fitness for Purpose


Although overfitting and coherence checking have the potential to improve the elicitation process, appreciable imprecision
will inevitably remain in the elicited summaries and in the fitted distribution. Whether this imprecision matters depends on
the purpose for which the elicitation is performed.
Recognition that an elicited prior distribution does not necessarily accurately reflect the expert’s knowledge has led to
quite widespread use of sensitivity analysis in Bayesian statistics (O’Hagan and Forster 2004, chap. 8). This may involve
varying the hyperparameters of a parametric fit or performing
a more sophisticated variation of all aspects of the distribution.
The general thrust of the _robust Bayesian_ movement was to allow the true prior distribution to lie in a nonparametric class
of distributions containing the elicited distribution. Then this
approach proceeded to derive bounds on posterior inferences
as the prior varied across the class of possible priors. Berger
(1994) has reviewed this body of research. A more common use
of sensitivity analysis in practical Bayesian analyses has been
to just explore in an ad hoc way a small number of alternative
prior distributions.
Whether the elicitation is done to obtain a prior distribution
for some Bayesian analysis, obtain expert judgments for inputs
of some decision model, or for some other purpose, sensitivity analysis has the same objective. It is to determine whether,
when the elicited distribution is varied to other distributions that
might also be consistent with the expert’s knowledge, the results
derived from that distribution change appreciably. If not, then
the elicitation has adequately represented the expert’s knowledge.
How can we determine whether the result changes “appreciably” as the elicited distribution changes? There is a clear answer
to this question when the result is a decision that is to be made
optimally with respect to some utility function. Then it is not
the change in the decision that matters, but rather the change
in expected utility. We consider the difference between, on the
one hand, the expected utility obtained by the optimal decision
with respect to the elicited prior, and on the other hand, the
maximum expected utility that can be obtained by the optimal
decision with respect to any other distribution in the class. This
difference represents the potential gain in expected utility that
might be obtained by more careful elicitation (see Kadane and
Chuang 1978; and Chuang 1984).
The problem with any sensitivity analysis is to specify the
class of distributions. If we allow the distribution to vary more
from the elicited distribution, then we can expect greater discrepancies in the results. The classes of priors used in robust
Bayesian analysis are arbitrary and not based on analysis of
the elicitation process. The “internal approach” of LT&B and
the method of O&O both yield the facilitator’s posterior distribution for the expert’s underlying probabilities or density
function. They therefore provide formal expression of the uncertainty around the particular elicited distribution, which can
in principle form the basis for subsequent sensitivity analysis.
Their formal structures are more complex to apply, but otherwise there seems no alternative to the kind of informal ad hoc
sensitivity analysis most commonly used.


696 Journal of the American Statistical Association, June 2005



4.3 Scoring Rules


In empirical work, probability distributions may be elicited
for uncertain quantities whose actual values are known to the
experimenters. In other circumstances, such as weather forecasting, predictive distributions may be assessed for quantities
whose values become known subsequently. In both cases it can
be useful to compare assessed probability distributions with the
observed data to provide an objective measure of its accuracy.
This is the purpose of a scoring rule.
Formally, a scoring rule is a formula for awarding a score to
the expert, which can be thought of as a reward. It is a function both of the expert’s elicited probability distribution for the
uncertain quantity and of that quantity’s true value.
One common application of scoring rules is in the comparison of alternative elicitation methods or different variants
of an elicitation method. In empirical research, one elicitation
method is often judged to be better than another if it gets better
scores. Note, however, that better scores result both from the
expert assessing his or her beliefs more accurately and from the
expert having more (or more accurate) knowledge.
Other purposes of a scoring rule are to provide an incentive for experts to record their opinions well and to help train
experts to quantify their opinions accurately. Toward this end,
it is important that a scoring rule should encourage experts to
record their true beliefs. More precisely, “the scoring rule is
constructed according to the basic idea that the resulting device should oblige each participant to express his true feelings,
because any departure from his own personal probability results in a diminution of his own average score as he sees it”
(de Finetti 1962, p. 359). A scoring rule with this property
is termed _proper_ . Various proper scoring rules have been proposed; several, including those most commonly used, have been
described by Matheson and Winkler (1976) (see also O’Hagan
and Forster 2004, secs. 2.54–2.58).


4.4 Calibration


There is a large and somewhat murky literature on the subject of calibration. At its simplest, the idea of calibration is that
perhaps a person’s elicited probabilities show a particular flaw,
in that of the events the person says has probability _p_ of happening, some function _g(p)_ of them actually occur. Then the
thought is that when the person announces _p_ as his or her probability of some event, knowing better, the user of this information has _g(p)_ as his or her probability (Lichtenstein et al.
1982). Such a program has the following flaw. Suppose that the
person being elicited is faced with a coin that he or she believes
to be fair, and hence announces _p_ = [1]

2 [as the elicited probabil-]
ity of “tails.” What values can _g(_ [1] _[)]_ [ take? Because the] _[ g(]_ [·] _[)]_ [ are]



ity of “tails.” What values can _g(_ [1]

2 _[)]_ [ take? Because the] _[ g(]_ [·] _[)]_ [ are]
supposed to be probabilities, and “tails” and “heads” are mutually exclusive and exhaustive, we must have _g(_ [1] _[)]_ [ +] _[ g(]_ [1] _[)]_ [ =][ 1,]



ally exclusive and exhaustive, we must have _g(_ [1] [1]

2 _[)]_ [ +] _[ g(]_ 2 _[)]_ [ =][ 1,]

that is, _g(_ [1] [1] [Now] [suppose] [there] [are] [three] [events] [equally]

2 _[)]_ [ =] 2 [.]

likely in the view of the person being elicited. The same argument shows _g(_ [1] [1] _[g(]_ [2] [2] [Indeed,] [this] [argument]

3 _[)]_ [ =] 3 [and] 3 _[)]_ [ =] 3 [.]

demonstrates _g(r)_ = _r_ for every rational number _r_ . An assumption of continuity or measurability of _g_ then suffices to show
that _g(x)_ = _x_ for all real numbers, 0 _< x <_ 1. Hence recalibration on this basis contradicts the coherence of either the pretransformation or post-transformation probabilities. Note that



this argument does not apply to functions _g_ that involve more
than the elicited probabilities. For example, it is not a contradiction to coherence to think that a person may be overconfident
in the sense that the probability ( [1]

2 [) assigned to the interquartile]
range is too high, and hence the probability assigned to the tails
is too low. Similarly, it might be noticed that a weather forecaster systematically overpredicts rain and hence underpredicts
the event of no rain.
In practice, calibration is relevant where, as in the case of
weather forecasters, experts regularly make similar probability
statements, so that it is possible to check calibration and feedback is immediate, relevant, and frequent. Even without a formal calibration check, receiving regular feedback would tend to
ensure that their forecasts are reasonably well calibrated.


5. MULTIPLE EXPERTS


5.1 Synthesizing Separate Elicitations


Where important decisions or inferences are to be made, it
is common to wish to draw on the expertise of several experts.
A number of approaches have been proposed as to how to elicit
and how to synthezise the different experts’ knowledge. Formal
methods of combining probability distributions have been reviewed by Genest and Zidek (1986), who provided a very useful annotated bibliography, and French (1985), among others.
We first consider the situation where the experts do not interact.
Separate probability distributions are elicited from the experts
in separate elicitation sessions. It is then natural to ask how we
can synthesize these different distributions into a single distribution.
Two of the most popular methods fall into the category
known as _opinion_ _pools_ . The linear opinion pool is a convex
combination (a weighted average) of the individual probability
distributions composing it, and the logarithmic opinion pool is a
normalized weighted geometric mean (equivalent to applying a
linear pool to the logarithms of the individual probability densities and then normalizing the result). An important property
that an opinion pool might be expected to have is that it be externally Bayesian (Madansky 1978), meaning that when there is
an agreed-on likelihood function, the opinion pool of the posterior distributions should coincide with the posterior distribution
obtained from the opinion pool of the prior distributions. Except in trivial cases, the linear opinion pool fails to have this
property, whereas the logarithmic pool does have it, when the
weights sum to 1. However, a second property that we might require is invariance to event combination. Suppose, for instance,
that we elicit the experts’ probabilities for two mutually exclusive events, _A_ and _B_ . Letting _C_ be the event ‘ _A_ or _B_,’ each
expert’s probabilities (assuming that they are coherent) satisfy
_P(C)_ = _P(A)_ + _P(B)_ . Combination invariance would then require that the same property should hold for the pooled probabilities of _A_, _B_, and _C_ . McConway (1981) showed that only
the linear opinion pool satisfies a general marginalization criterion of this type. It is therefore not possible to find a mechanistic opinion pooling method that is externally Bayesian and also
satisfies the marginalization criterion.
Note that the logarithmic opinion pool also suffers from the
fact that a single expert’s opinion that a certain set has zero




[1] [1]

2 _[)]_ [ =] 2




[1] [1]

2 _[)]_ [ +] _[ g(]_ 2




[1] [1]

3 _[)]_ [ =] 3




[1] _[g(]_ [2]

3 [and] 3




[2] [2]

3 _[)]_ [ =] 3


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 697



probability implies that the pool must also assign zero probability to that set. (See Genest and Zidek 1986 for a wide-ranging
discussion of these issues.)
Both linear and logarithmic pools allow the assignment of
different weights to the experts, which can be used to give
more weight to experts whose probability distributions are believed to be more accurate. Cooke (1991) described a method
of choosing weights based on the experts’ performance in assessing distributions for _seed_ _variables_, quantities whose true
value is known to the facilitator but not to the experts. Evidence
that this produces better elicitation than simple equal weighting
of the experts was presented by Cooke and Goossens (2000).
Mechanistic pooling methods can lead to a form of double
counting of expertise if the knowledge of some of the experts
overlaps substantially. Then it is inappropriate to weight them
all equally with other experts, but the method of seed variables
will also tend to overweight such a group.
Another criticism of all of these pooling methods is that it
is not clear whose opinion (if anyone’s) the resulting probability distribution represents. A quite different approach to putting
multiple experts’ opinions together is to imagine each opinion as data input to a single “supra Bayesian,” who uses these
opinions to update his or her views. This is the “external approach” of Lindley et al. (1979). It was proposed earlier by
Morris (1974) and further developed by Lindley (1985), French
(1985), and Genest and Schervish (1985); see also the discussion of Genest and Zidek (1986). This approach is completely
Bayesian, but requires a very substantial elicitation of the supra
Bayesian’s opinions about the expert opinions to be pooled.


5.2 Group Elicitation Methods


We now consider approaches where the experts interact as a
group. One simple and practical group elicitation approach is
to bring the experts together to discuss the uncertain quantity
or quantities about which their beliefs are to be elicited, and
through this sharing of their expertise to seek a consensus view.
In effect this treats the group as a single individual. Phillips
(1999) presented a formal justification of this _behavioral aggre-_
_gation_ approach. There are, however, new psychological issues
that arise in the interaction between the members of the group.
This kind of group elicitation requires a knowledgeable and experienced facilitator who needs to be aware of the possibilities
of strong personalities in the group having too much weight in
the discussion, or of judgments based on overlapping experience being overweighted through being repeated in the discussions. It may also be that the pressure to reach consensus leads
to the experts suppressing dissenting views, or, alternatively, it
may not be possible to reach consensus.
The Delphi method is a formal technique for managing group
interaction. The method proceeds by first eliciting the various
experts’ opinions separately, then feeding each expert’s views to
all of the other experts along with some explanation of that expert’s reasoning. The experts are then invited to revise their own
elicitations. The method then operates iteratively, feeding back
the revised elicitations to all of the experts, with explanation
of the reasons for revisions, and so on. Although some of the
complications of group interaction are removed, the method is
likely to produce a less efficient sharing of knowledge than the
behavioral aggregation approach. It is also still necessary for



the facilitator to manage the interaction, because one expert’s
reasons may have undue influence if expressed very forcefully.
Pill (1971) reviewed the Delphi technique. In addition, there is
a truly vast literature on its use in political science and government.
A variant of Delphi was discussed by DeGroot (1974), who
proposed that each expert revise his or her opinion by applying
a linear opinion pool to all of the experts’ distributions, with
weights that reflect the importance that each pool member puts
on the opinions of each of the other participants. The system
of revisions then forms a Markov chain whose transition matrix
is the matrix of weights, and DeGroot obtained conditions for
them to converge to a consensus (see also Lehrer 1976).
In contrast with these practical group elicitation methods, we
can also take a more axiomatic approach and try to identify
what would be rational ways for experts to seek a combined
expression of beliefs. Bayesian theory is profoundly a theory
of rational individual decision making. Basically, it imposes a
minimal condition on an individual’s statements of what bets
would be acceptable (namely the avoidance of Dutch book),
and accepts all responses that meet that condition. How can this
theory be extended to groups?
To make progress on this question, one needs to know how
the group decision making structure works, and how it relates
to the views of the individuals in the group. Before starting, it
is necessary to eliminate one obvious special case, that of a dictatorship. In a dictatorship, the choices of only one individual
(i.e., of the dictator) matter for the group’s choices. Hence if the
dictator behaves individually in accordance with the Bayesian
axioms, then the group will as well. By eliminating this case,
we insist that more than one person’s views matter in how the
group makes decisions.
Suppose instead that the group makes decisions by majority
vote, and that each member of the group has a transitive ranking of the alternatives. Surely in this case, something reasonable
should be true of the decisions of such a group. Well, perhaps
not. Suppose that the group consists of three people (the minimum for interesting majority votes) and are expressing preferences among three alternatives, _A_, _B_, and _C_ . Suppose that
voter 1 ranks them in that order, that is, prefers _A_ to _B_ to _C_ .
These preferences will be denoted by _A > B_ _>_ _C_ . Suppose
that voter 2’s preferences are _B > C > A_ and that voter 3’s are
_C > A > B_ . In a choice between _A_ and _B_, voters 1 and 3 prefer
_A_ to _B_ . Between _B_ and _C_, voters 1 and 2 prefer _B_ to _C_ . Finally,
between _A_ and _C_, voters 2 and 3 prefer _C_ to _A_ . Hence the
majority preferences can be summarized as _A > B_ _> C_ _> A_ .
No group utility function can summarize such choices, because
they are not transitive. This simple example has been generalized to every nondictatorial group decision making process in
a celebrated theorem of Arrow (1951, 1963). A huge literature
has grown up around this result, mainly under the heading of
political economy.
A second approach to the problem of Bayesian group decision making asks for preferences between risky outcomes
for two or more Bayesians. However they make their decisions, these Bayesians seek to compromise. The only condition imposed on their compromises is that they obey the Pareto
Principle: If each member of the group prefers _A_ to _B_, then
the compromise cannot prefer _B_ . Suppose that there are two


698 Journal of the American Statistical Association, June 2005



Bayesians and three alternatives. Two trivial cases must be dealt
with first. If the Bayesians agree in their probabilities, then
every nontrivial convex combination of their utilities, together
with their agreed-on probability, will provide a Bayesian compromise satisfying the Pareto Principle. Similarly, if their utilities coincide, then every nontrivial convex combination of their
probabilities will similarly suffice. The result of Seidenfeld,
Schervish, and Kadane (1989) is that these are the only cases
in which a Bayesian compromise can be found. When there
are more than two Bayesians involved, the Pareto Principle is
less binding, so in certain cases a Bayesian compromise can be
found (see Goodman 1988).
These results cast serious doubt on what might be meant by
the probability and utility function of a group seeking to be
Bayesian. In what sense can the probability and utility function
of the group be representative of the decisions that the group
might make?


6. DISCUSSION


From the 1960s to at least the early 1980s, research in elicitation was substantial and was characterized by some close
collaborations between statisticians and psychologists. More
recently, there seems to have been much less research in both
the statistics and psychology communities, and collaboration
between them has lapsed. There are signs that this is changing, however. The growing sophistication of Bayesian computational methods has led to a dramatic increase in the breadth
and complexity of Bayesian applications. Bayesians are beginning to show more interest in elicitation and, being freed from
the computational constraint to use tractable, conjugate priors,
there is a need to develop processes capable of eliciting complex, nonstandard distributions. A recent review of case studies
and software has been provided by Kadane and Wolfson (1998).
This in turn is likely to lead to renewed collaboration with psychologists.
Despite the existence of a broad and diverse literature in
elicitation, which has provided many valuable procedures and
insights, there remains very considerable scope for further research. Some important topics are the following:


 - Multivariate elicitation. Where it is necessary to elicit a
joint probability distribution for two or more quantities,
there has been relatively little investigation by psychologists (particularly when the quantities cannot be regarded
as instances of a larger population). In this context, also,
the available multivariate parametric families typically impose unrealistic constraints on experts’ beliefs.

 - Nonparametric elicitation. We find the use of uniform, triangular, or histogram distributions unrealistic, but fitting
parametric distributions also imposes constraints that may
be unrealistic. There has been little work on nonparametric
fitting, and although a nonparametric fit might represent
an expert’s beliefs more accurately, it is not clear whether
this will actually matter in practice (see the discussion in
Sec. 4.2).

 - Graphical tools. We believe that there is substantial, as-yet
relatively unexplored, potential in graphical methods.



An aim of much statistical research is to wring as much from
data as we possibly can, but using expert opinion better (or using it at all) could add more information than slight improvement in efficiency through better techniques of data analysis.
Too often, ad hoc methods must be used when an expert’s opinion is to be quantified; ideally, there should be a range of tried
and tested elicitation methods in a statistician’s toolbox.


_[Received June 2004. Revised November 2004.]_


REFERENCES


Al-Awadhi, S. A., and Garthwaite, P. H. (1998), “An Elicitation Method for
Multivariate Normal Distributions,” _Communications_ _in_ _Statistics,_ _Part_ _A—_
_Theory and Methods_, 27, 1123–1142.

(2001), “Prior Distribution Assessment for a Multivariate Normal Distribution: An Experimental Study,” _Journal of Applied Statistics_, 28, 5–23.
Alpert, M., and Raiffa, H. (1969), “A Progress Report on the Training of Probability Assessors,” unpublished manuscript, Harvard University; reprinted in
_Judgement and Uncertainty: Heuristics and Biases_, eds. D. Kahneman et al.,
Cambridge, U.K.: Cambridge University Press, 1982, pp. 306–334.
Arrow, K. (1951, 1963), _Social Choice and Individual Values_, New York: Wiley.
Barclay, S., and Peterson, C. R. (1973), “Two Methods for Assessing Probability Distributions,” Technical Report 73-1, Decisions and Designs, Inc.
Barclay, S., and Randall, L. S. (1975), “Interactive Decision Analysis Aids for
Intelligence Analysis,” Technical Report Dt/TR 75-4, Decisions and Designs,
Inc.
Bar-Hillel, M. (1973), “On the Subjective Probability of Compound Events,”
_Organizational Behavior and Human Performance_, 9, 396–406.
Beach, L. R., and Scopp, T. S. (1968), “Intuitive Statistical Inferences About
Variances,” _Organizational Behavior and Human Performance_, 3, 109–123.
Beach, L. R., and Swenson, R. G. (1966), “Intuitive Estimation of Means,”
_Psychonomic Science_, 5, 161–162.
Bedrick, E. J., Christensen, R., and Johnson, W. (1996), “A New Perspective on
Priors for Generalized Linear Models,” _Journal_ _of_ _the_ _American_ _Statistical_
_Association_, 91, 1450–1460.
Berger, J. O. (1994), “An Overview of Robust Bayesian Analysis” (with discussion), _Test_, 3, 5–124.
Berger, J. O., and O’Hagan, A. (1988), “Ranges of Posterior Probabilities
for Unimodal Priors With Specified Quantiles,” in _Bayesian Statistics 3_, eds.
J. M. Bernardo et al., Oxford, U.K.: Oxford University Press, pp. 45–66.
Beyth-Marom, R. (1982), “How Probable Is Probable? A Numerical Translation of Verbal Probability Expressions,” _Journal of Forecasting_, 1, 257–269.
Black, P., and Laskey, K. (1989), “Models for Elicitation in Bayesian ANOVA:
Implementation and Application,” in _ASA Proceedings of the Statistical Com-_
_puting Section_, American Statistical Association, pp. 247–252.
Chaloner, K. (1996), “The Elicitation of Prior Distributions,” in _Case_ _Studies_
_in_ _Bayesian_ _Biostatistics_, eds. D. Berry and D. Stangl, New York: Marcel
Dekker, pp. 141–156.
Chaloner, K. M., Church, T., Matts, J. P., and Louis, T. A. (1993), “Graphical
Elicitation of a Prior Distribution for an AIDS Clinical Trial,” _The Statistician_,
42, 341–353.
Chaloner, K. M., and Duncan, G. T. (1983), “Assessment of a Beta Distribution:
PM Elicitation,” _The Statistician_, 32, 174–180.

(1987), “Some Properties of the Dirichlet-Multinomial Distribution
and Its Use in Prior Elicitation,” _Communications_ _in_ _Statistics,_ _Part_ _A—_
_Theory and Methods_, 16, 511–523.
Chen, M.-H., Ibrahim, J. G., and Yiannoutsos, C. (1999), “Prior Elicitation,
Variable Selection and Bayesian Computation for Logistic Regression Models,” _Journal of the Royal Statistical Society_, Ser. B, 61, 223–242.
Chuang, D. T. (1984), “Further Theory of Stable Decisions,” in _Robust-_
_ness_ _of_ _Bayesian_ _Analyses_, ed. J. B. Kadane, Amsterdam: North-Holland,
pp. 165–228.
Clemen, R. T., Fischer, G. W., and Winkler, R. L. (2000), “Assessing Dependence: Some Experimental Results,” _Management Science_, 46, 1100–1115.
Clemen, R. T., and Reilly, T. (1999), “Correlations and Copulas for Decision
and Risk Analysis,” _Management Science_, 45, 208–223.
Cooke, R. M. (1991), _Experts_ _in_ _Uncertainty:_ _Opinion_ _and_ _Subjective_ _Proba-_
_bility in Science_, Oxford, U.K.: Oxford University Press.
Cooke, R. M., and Goossens, L. H. J. (2000), “Procedures Guide for Structured
Expert Judgment in Accident Consequence Modelling,” _Radiation Protection_
_Dosimetry_, 90, 303–309.
Dalal, S., and Hall, W. J. (1983), “Approximating Priors by Mixtures of Natural Conjugate Priors,” _Journal_ _of_ _the_ _Royal_ _Statistical_ _Society_, Ser. B, 45,
278–286.
de Finetti, B. (1962), “Does It Make Sense to Speak of ‘Good Probability Appraisers’?” in _The Scientist Speculates—An Anthology of Partly-Baked Ideas_,
ed. I. J. Good, London: Heineman, pp. 356–364.


Garthwaite, Kadane, and O’Hagan: Eliciting Probability Distributions 699



DeGroot, M. H. (1974), “Reaching a Consensus,” _Journal of the American Sta-_
_tistical Association_, 69, 118–121.
Diaconis, P., and Ylvisaker, D. (1985), “Quantifying Prior Opinion,” in
_Bayesian Statistics 2_, eds. J. M. Bernardo et al., Amsterdam: North-Holland,
pp. 133–156.
Dickey, J. (1980), “Beliefs About Beliefs, a Theory of Stochastic Assessments of Subjective Probabilities,” in _Bayesian_ _Statistics_, eds. J. Bernardo,
M. H. DeGroot, D. V. Lindley, and A. F. M. Smith, Valencia, Spain: Valencia
University Press, pp. 471–487.
Dickey, J. M., Dawid, A. P., and Kadane, J. B. (1986), “Subjective Probability
Assessment Methods for Multivariate- _t_ and Matrix- _t_ Models,” in _Bayesian_
_Inference and Decision Techniques: Essays in Honor of Bruno de Finetti_, eds.
P. K. Goel and A. Zellner, New York: Elsevier, pp. 177–195.
Edwards, W., and Phillips, L. D. (1964), “Man as Transducer for Probabilities
in Bayesian Command and Control Systems,” in _Human Judgements and Op-_
_timality_, eds. G. L. Bryan and M. W. Shelley, New York: Wiley, pp. 305–313.
Einhorn, H. J. (1971), “Use of Nonlinear, Noncompensatory Models as a Function of Task and Amount of Information,” _Organizational Behaviour and Hu-_
_man Performance_, 6, 1–27.
Erlick, D. E. (1964), “Absolute Judgment of Discrete Quantities Randomly Distributed Over Time,” _Journal of Experimental Psychology_, 67, 475–482.
Fischhoff, B., and Beyth, R. (1975), “I knew It Would Happen: Remembered
Probabilities of Once-Future Things,” _Organizational_ _Behavior_ _and_ _Human_
_Performance_, 13, 1–16.
Fischhoff, B., Slovic, P., and Lichtenstein, S. (1978), “Fault Trees: Sensitivity of Estimated Failure Probabilities to Problem Representation,” _Journal of_
_Experimental Psychology: Human Perception and Performance_, 4, 330–334.
French, S. (1985), “Group Consensus Probability Distributions: A Critical Survey,” in _Bayesian Statistics 2_, eds. J. M. Bernardo et al., Amsterdam: NorthHolland, pp. 83–210.
Garthwaite, P. H. (1983), “Assessment of Prior Distributions for Normal Linear
Models,” unpublished doctoral thesis, University College of Wales at Aberystwyth.

(1989), “Fractile Assessments for a Linear Regression Model: An Experimental Study,” _Organizational_ _Behavior_ _and_ _Human_ _Performance_, 43,
188–206.
Garthwaite, P. H., and Al-Awadhi, S. A. (2001), “Non-Conjugate Prior Distribution Assessment for Multivariate Normal Sampling,” _Journal of the Royal_
_Statistical Society_, Ser. B, 63, 95–110.

(2003), “Quantifying Opinion About a Logistic Regression Using Interactive Graphics,” unpublished manuscript, The Open University, Dept. of
Statistics.
Garthwaite, P. H., and Dickey, J. M. (1988), “Quantifying Expert Opinion in
Linear Regression Problems,” _Journal of the Royal Statistical Society_, Ser. B,
50, 462–474.

(1991), “An Elicitation Method for Multiple Linear Regression Models,” _Journal of Behavioral Decision Making_, 4, 17–31.

(1992), “Elicitation of Prior Distributions for Variable-Selection Problems in Regression,” _The Annals of Statistics_, 20, 1697–1719.

(1996), “Quantifying and Using Expert Opinion for Variable-Selection
Problems in Regression” (with discussion), _Chemometrics_ _and_ _Intelligent_
_Laboratory Systems_, 35, 1–43.
Garthwaite, P. H., and O’Hagan, A. (2000), “Quantifying Expert Opinion in the
U.K. Water Industry: An Experimental Study,” _The Statistician_, 49, 455–477.
Gavaskar, U. (1988), “A Comparison of Two Elicitation Methods for a Prior
Distribution for a Binomial Parameter,” _Management Science_, 34, 784–790.
Genest, C., and Schervish, M. J. (1985), “Modelling Expert Judgments for
Bayesian Updating,” _The Annals of Statistics_, 13, 1198–1212.
Genest, C., and Zidek, J. V. (1986), “Combining Probability Distributions:
A Critique and an Annotated Bibliography” (with discussion), _Statistical Sci-_
_ence_, 1, 114–148.
Gigerenzer, G. (1996), “On Narrow Norms and Vague Heuristics: A Reply to
Kahneman and Tversky,” _Psychological Review_, 103, 592–596.
Gokhale, D. V., and Press, S. J. (1982), “Assessment of a Prior Distribution for
the Correlation Coefficient in a Bivariate Normal Distribution,” _Journal of the_
_Royal Statistical Society_, Ser. A, 145, 237–249.
Goldstein, M. (1999), “Bayes Linear Analysis,” in _Encyclopedia of_ _Statistical_
_Sciences_ (update volume 3), eds. S. Kotz et al., New York: Wiley, pp. 29–34.
Goodman, J. (1988), “Existence of Compromises in Simple Group Decisions,”
unpublished doctoral dissertation, Carnegie Mellon University, Dept. of Statistics.
Hammerton, M. (1975), “A Case of Radical Probability Estimation,” _Journal of_
_Experimental Psychology_, 101, 242–254.
Hampton, J. M., Moore, P. G., and Thomas, H. (1973), “Subjective Probability
and Its Measurement,” _Journal_ _of_ _the_ _Royal_ _Statistical_ _Society_, Ser. A, 136,
21–42.
Hofstatter, P. R. (1939), “Uber die Schatzung von Gruppeneigenschaften,”
_Zeitschrift fur Psychologie_, 145, 1–44.



Hogarth, R. M. (1975), “Cognitive Processes and the Assessment of Subjective
Probability Distributions,” _Journal_ _of_ _the_ _American_ _Statistical_ _Association_,
70, 271–294.

(1987), _Judgement and Choice_ (2nd ed.), Chichester, U.K.: Wiley.
Howard, R. A., and Matheson, J. (1984), “Influence Diagrams,” in _Readings on_
_the Principles and Applications of Decision Analysis_, eds. R. A. Howard and
J. Matheson, Palo Alto, CA: Strategic Decision Group, pp. 719–762.
Huber, G. P. (1974), “Methods for Quantifying Subjective Probabilities and
Multi-Attribute Utilities,” _Decision Sciences_, 5, 430–458.
Ibrahim, J. G., and Laud, P. W. (1994), “A Predictive Approach to the Analysis
of Designed Experiments,” _Journal_ _of_ _the_ _American_ _Statistical_ _Association_,
89, 309–319.
Inhelder, B., and Piaget, J. (1958), _The Growth of Logical Thinking From Child-_
_hood to Adolescence_, New York: Basic Books.
Jenkins, H. M., and Ward, W. C. (1965), “The Judgment of Contingency Between Responses and Outcomes,” _Psychological Monographs_, 79, 1–17.
Johnson, E. J., Hershey, J., Meszaros, J., and Kunreuther, H. (1993), “Framing, Probability Distortions, and Insurance Decisions,” _Journal_ _of_ _Risk_ _and_
_Uncertainty_, 7, 35–51.
Kadane, J. B. (1980), “Predictive and Structural Methods for Eliciting Prior
Distributions,” in _Bayesian_ _Analysis_ _in_ _Econometrics_ _and_ _Statistics:_ _Essays_
_in_ _Honor_ _of_ _Harold_ _Jeffreys_, ed. A. Zellner, Amsterdam: North-Holland,
pp. 89–93.

(ed.) (1996), _Bayesian Methods and Ethics in a Clinical Trial Design_,
New York: Wiley.
Kadane, J. B., Chan, N. H., and Wolfson, L. J. (1996), “Priors for Unit Root
Models,” _Journal of Econometrics_, 75, 99–111.
Kadane, J. B., and Chuang, D. T. (1978), “Stable Decision Problems,” _The An-_
_nals of Statistics_, 6, 1095–1110.
Kadane, J. B., Dickey, J., Winkler, R., Smith, W., and Peters, S. (1980), “Interactive Elicitation of Opinion for a Normal Linear Model,” _Journal_ _of_ _the_
_American Statistical Association_, 75, 845–854.
Kadane, J. B., and Schum, D. A. (1996), _A Probabilistic Analysis of the Sacco_
_and Vanzetti Evidence_, New York: Wiley.
Kadane, J. B., and Winkler, R. L. (1988), “Separating Probability Elicitation
From Utilities,” _Journal of the American Statistical Association_, 83, 357–363.
Kadane, J. B., and Wolfson, L. J. (1998), “Experiences in Elicitation” (with
discussion), _The Statistician_, 47, 1–20.
Kahneman, D., and Tversky, A. (1973), “On the Psychology of Prediction,”
_Psychological Review_, 80, 237–251.
Keren, G. (1991), “Calibration and Probability Judgments: Conceptual and
Methodological Issues,” _Acta Psychologica_, 77, 217–273.
Koehler, J. J. (1996), “The Base Rate Fallacy Reconsidered: Normative, Descriptive and Methodological Challenges,” _Behavioral_ _and_ _Brain_ _Sciences_,
19, 1–17.
Koriat, A., Lichtenstein, S., and Fischhoff, B. (1980), “Reasons for Confidence,” _Journal of Experimental Psychology: Human Learning and Memory_,
6, 107–118.
Kunda, Z., and Nisbett, R. E. (1986), “The Psychometrics of Everyday Life,”
_Cognitive Psychology_, 18, 195–224.
Lathrop, R. G. (1967), “Perceived Variability,” _Journal_ _of_ _Experimental_ _Psy-_
_chology_, 73, 498–502.
Laud, P. W., and Ibrahim, J. G. (1995), “Predictive Model Selection,” _Journal_
_of the Royal Statistical Society_, Ser. B, 57, 247–262.
Lehrer, K. (1976), “When Rational Disagreement Is Impossible,” _Nous_, 10,
327–332.
Lichtenstein, S., and Fischhoff, B. (1980), “Training for Calibration,” _Organi-_
_zational Behavior and Human Performance_, 26, 149–171.
Lichtenstein, S., Fischhoff, B., and Phillips, L. D. (1982), “Calibration of
Probabilities: The State of the Art to 1980,” in _Judgement_ _and_ _Uncertainty:_
_Heuristics and Biases_, eds. D. Kahneman et al., Cambridge, U.K.: Cambridge
University Press, pp. 306–334.
Lichtenstein, S., and Newman, J. R. (1967), “Empirical Scaling of Common
Verbal Phrases Associated With Numerical Probabilities,” _Psychonomic Sci-_
_ence_, 9, 563–564.
Lindley, D. V. (1985), “Reconciliation of Discrete Probability Distributions,” in
_Bayesian Statistics 2_, eds. J. M. Bernardo et al., Amsterdam: North-Holland,
pp. 375–390.
Lindley, D. V., Tversky, A., and Brown, R. V. (1979), “On the Reconciliation of
Probability Assessments” (with discussion), _Journal_ _of_ _the_ _Royal_ _Statistical_
_Society_, Ser. A, 142, 146–180.
Madansky, A. (1978), “Externally Bayesian Groups,” unpublished manuscript,
University of Chicago.
Matheson, J. E., and Winkler, R. L. (1976), “Scoring Rules for Continuous
Probability Distributions,” _Management Science_, 22, 1087–1096.
McConway, K. J. (1981), “Marginalization and Linear Opinion Pools,” _Journal_
_of the American Statistical Association_, 76, 410–414.


700 Journal of the American Statistical Association, June 2005



Meyer, M. A., and Booker, J. M. (2001), _Eliciting and Analyzing Expert Judg-_
_ment:_ _A_ _Practical_ _Guide_, Philadelphia: ASA/Society of Industrial and Applied Mathematics.
Morgan, M. G., and Henrion, M. (1990), _Uncertainty: A Guide to Dealing With_
_Uncertainty in Quantitative Risk and Policy Analysis_, Cambridge, U.K.: Cambridge University Press.
Morris, P. A. (1974), “Decision Analysis Expert Use,” _Management_ _Science_,
20, 1233–1241.
Mosteller, F., and Yountz, C. (1990), “Quantifying Probabilistic Expressions”
(with discussion), _Statistical Science_, 5, 2–34.
Murphy, A. H., and Winkler, R. L. (1974), “Credible Interval Temperature
Forecasting: Some Experimental Results,” _Monthly_ _Weather_ _Review_, 102,
784–794.
Nash, H. (1964), “The Judgment of Linear Proportions,” _American Journal of_
_Psychology_, 77, 480–484.
Nisbett, R. E., Borgida, E., Crandall, R., and Reed, H. (1976), “Popular Induction: Information Is Not Necessarily Informative,” in _Cognition_ _and_ _Social_
_Behaviour_, Hillsdale, MD: Lawrence Erlbaum, pp. 113–133.
Oakley, J. E., and O’Hagan, A. (2002), “Uncertainty in Prior Elicitations:
A Nonparametric Approach,” Research Report No. 521/02, University of
Sheffield, Dept. of Probability and Statistics.
Oberkampf, W. L., Helton, J. C., Joslyn, C. A., Wojtkiewicz, S. F., and
Ferson, S. (2004), “Challenge Problems: Uncertainty in System Response
Given Uncertain Parameters,” _Reliability Engineering and System Safety_, 85,
11–19.
O’Hagan, A. (1988), _Probability:_ _Methods_ _and_ _Measurement_, London: Chapman & Hall.

(1998), “Eliciting Expert Beliefs in Substantial Practical Applications”
(with discussion), _The Statistician_, 47, 21–35.
O’Hagan, A., and Forster, J. J. (2004), _Kendall’s Advanced Theory of Statistics,_
_Vol. 2B, Bayesian Inference_ (2nd ed.), London: Edward Arnold.
O’Hagan, A., and Oakley, J. E. (2004), “Probability Is Perfect, but We Can’t
Elicit It Perfectly,” _Reliability Engineering and System Safety_, 85, 239–248.
Oman, S. D. (1985), “Specifying a Prior Distribution in Structured Regression
Problems,” _Journal of the American Statistical Association_, 80, 190–195.
Peterson, C. R., and Beach, L. R. (1967), “Man as an Intuitive Statistician,”
_Psychological Bulletin_, 68, 29–46.
Peterson, C. R., and Miller, A. (1964), “Mode, Median and Mean as Optimal
Strategies,” _Journal of Experimental Psychology_, 68, 363–367.
Peterson, C. R., Snapper, K. J., and Murphy, A. H. (1972), “Credible Interval
Temperature Forecasts,” _Bulletin of the American Meteorological Society_, 53,
966–970.
Phillips, L. D. (1999), “Group Elicitation of Probability Distributions: Are
Many Heads Better Than One?” in _Decision Science and Technology: Reflec-_
_tions on the Contributions of Ward Edwards_, eds. J. Shanteau et al., Norwell,
MA: Kluwer Academic, pp. 313–330.
Pill, J. (1971), “The Delphi Method: Substance, Context, a Critique, and an
Annotated Bibliography,” _Socio-Economic Planning and Sciences_, 5, 57–71.
Pitz, G. F. (1965), “Response Variables in the Estimation of Relative Frequency,” _Perceptual and Motor Skills_, 21, 867–873.

(1966), “The Sequential Judgment of Proportion,” _Psychonomic_ _Sci-_
_ence_, 4, 397–398.
Raiffa, H. (1968), _Decision_ _Analysis:_ _Introductory_ _Lectures_ _on_ _Choice_ _Under_
_Uncertainty_, Reading, MA: Addison-Wesley.
Schaefer, R. E., and Borcherding, K. (1973), “The Assessment of Subjective
Probability Distributions: A Training Experiment,” _Acta_ _Psychologics_, 37,
117–129.
Seaver, D. A., von Winterfeldt, D., and Edwards, W. (1978), “Eliciting Subjective Probability Distributions on Continuous Variables,” _Organizational_ _Be-_
_havior and Human Performance_, 21, 379–391.
Seidenfeld, T., Schervish, M. J., and Kadane, J. B. (1989), “On the Shared Preferences of Two Decision Makers,” _Journal of Philosophy_, 86, 225–244.
Shuford, E. H. (1961), “Percentage Estimation of Proportion as a Function of
Element Type, Exposure Type, and Task,” _Journal of Experimental Psychol-_
_ogy_, 61, 430–436.
Simpson, W., and Voss, J. F. (1961), “Psychophysical Judgments of Probabilistic Stimulus Sequences,” _Journal of Experimental Psychology_, 62, 416–422.



Singpurwalla, N. D., and Song, M. S. (1987), “The Analysis of Weibull Lifetime Data Incorporating Expert Opinion,” in _Probability and Bayesian Statis-_
_tics_, ed. R. Viertl, New York: Plenum, pp. 431–442.
Slovic, P. (1972), “From Shakespeare to Simon: Speculations—and Some
Evidence—About Man’s Ability to Process Information,” _Oregon_ _Research_
_Institute Monograph_, Vol. 12, No. 2.
Slovic, P., and Fischhoff, B. (1977), “On the Psychology of Experimental Surprises,” _Organizational Behavior and Human Performance_, 3, 544–551.
Slovic, P., and Lichtenstein, S. C. (1968), “The Relative Importance of Probabilities and Payoffs in Risk Taking,” _Journal_ _of_ _Experimental_ _Psychology_
_Monograph Supplement_, 78, No. 3, Part 2.


(1971), “Comparison of Bayesian and Regression Approaches to the
Study of Information Processing in Judgement,” _Organizational_ _Behavior_
_and Human Performance_, 6, 649–744.
Smedslund, J. (1963), “The Concept of Correlation in Adults,” _Scandinavian_
_Journal of Psychology_, 4, 165–173.
Spencer, J. (1961), “Estimating Averages,” _Ergonomics_, 4, 317–328.

(1963), “A Further Study of Estimating Averages,” _Ergonomics_, 6,
255–265.
Spetzler, C. S., and Stael von Holstein, C.-A. S. (1975), “Probability Encoding
in Decision Analysis,” _Management Science_, 22, 340–358.
Stael von Holstein, C.-A. S. (1971), “An Experiment in Probabilistic Weather
Forecasting,” _Journal of Applied Meteorology_, 10, 635–645.
Stevens, S. S., and Galanter, E. H. (1957), “Ratio Scales and Category Scales
for a Dozen Perceptual Continua,” _Journal_ _of_ _Experimental Psychology_, 54,
377–411.
Tversky, A. (1974), “Assessing Uncertainty,” _Journal_ _of_ _the_ _Royal_ _Statistical_
_Society_, Ser. B, 36, 148–159.
Tversky, A., and Kahneman, D. (1971), “The Belief in the Law of Small Numbers,” _Psychological Bulletin_, 76, 105–110.

(1973), “Availability: A Heuristic for Judging Frequency and Probability,” _Cognitive Psychology_, 5, 207–232.

(1974), “Judgment Under Uncertainty: Heuristics and Biases,” _Sci-_
_ence_, 185, 1124–1131.
Tversky, A., and Koehler, D. J. (1994), “Support Theory: A Nonextensional Representation of Subjective Probability,” _Psychological_ _Review_, 10,
547–567.
Wallsten, T. S., and Budescu, D. V. (1983), “Encoding Subjective Probabilities: A Psychological and Psychometric Review,” _Management_ _Science_, 29,
151–173.
Wallsten, T. S., Budescu, D. V., Rapoport, A., Zwick, R., and Forsyth, B.
(1986), “Measuring Vague Meanings of Probability Terms,” _Journal_ _of_ _Ex-_
_perimental Psychology: General_, 115, 348–365.
Ward, W. C., and Jenkins, H. M. (1965), “The Display of Information and the
Judgment of Contingency,” _Canadian Journal of Psychology_, 19, 231–241.
Wiggins, N., and Hoffman, P. J. (1968), “Three Models of Clinical Judgement,”
_Journal of Abnormal Psychology_, 73, 70–77.
Windschitl, P. D., and Wells, G. L. (1996), “Measuring Psychological Uncertainty: Verbal versus Numeric Methods,” _Journal_ _of_ _Experimental_ _Psychol-_
_ogy: Applied_, 2, 343–364.
Winkler, R. L. (1967), “The Assessment of Prior Distributions in Bayesian
Analysis,” _Journal of the American Statistical Association_, 62, 776–800.

(1972), _Introduction_ _to_ _Bayesian_ _Inference_ _and_ _Decision_, New York:
Holt, Rinehart & Winston.

(1980), “Prior Information, Predictive Distributions, and Bayesian
Model Building,” in _Bayesian_ _Analysis_ _in_ _Econometrics_ _and_ _Statistics:_ _Es-_
_says in Honor of Harold Jeffreys_, ed. A. Zellner, Amsterdam: North-Holland,
pp. 95–109.
Wolfson, L. J. (1995), “Elicitation of Priors and Utilities for Bayesian Analysis,” unpublished doctoral thesis, Carnegie Mellon University, Dept. of Statistics.
Youssef, Z. I., and Peterson, C. R. (1973), “Intuitive Cascaded Inferences,” _Or-_
_ganizational Behavior and Human Performance_, 10, 349–358.
Zellner, A. (1972), “On Assessing Informative Prior Distributions for Regression Coefficients,” unpublished manuscript.


**Applied Spatial Statistics for Public Health Data.**


Lance A. Waller and Carol A. Gotway _Peter Diggle_ 702


**Foundations** **of** **Risk** **Analysis:** **A** **Knowledge** **and** **Decision-**
**Oriented Perspective.**


Terje Aven _David L. Banks_ 703


**Comparison Methods for Stochastic Models and Risks.**


Alfred Müller and Dietrich Stoyan _Peter C. Kiessler_ 704


**Quantitative Methods in Population Health: Extensions of Or-**
**dinary Regression.**


Mari Palta _Erinn M. Hade_ 704


**The** **Statistical** **Evaluation** **of** **Medical** **Tests** **for** **Classification**
**and Prediction.**


Margaret Sullivan Pepe _Peihua Qiu_ 705


**Computational Methods in Statistics and Econometrics.**


Hisashi Tanizaki _Giovanni Baiocchi_ 705


**Numerical** **Issues** **in** **Statistical** **Computing** **for** **the** **Social** **Sci-**
**entist.**
Micah Altman, Jeff Gill, and
Michael P. McDonald _T. Van Zandt_ 707


**Experimental** **and** **Quasi-Experimental** **Designs** **for** **General-**
**ized Causal Inference.**
William R. Shadish, Thomas D. Cook,
and Donald T. Campbell _Christine M. Anderson-Cook_ 708


**Planning, Construction, and Statistical** **Analysis of Compara-**
**tive Experiments.**

Francis G. Giesbrecht and
Marcia L. Gumpertz _Christine M. Anderson-Cook_ 708


# **Book Reviews**

**Applied Longitudinal Analysis.**

Garrett M. Fitzmaurice, Nan M. Laird,
and James H. Ware _Geert Molenberghs_ 709


**Generalized** **Latent** **Variable** **Modeling:** **Multilevel,** **Longitudi-**
**nal, and Structural Equation Models.**


Anders Skrondal and Sophia Rabe-Hesketh _Luisa Corrado_ 710


**Microarray Quality Control.**

Wei Zhang, Ilya Shmulevich, and
Jaakko Astola _Robert Gentleman_ 711


**Design and Analysis of DNA Microarray Investigations.**

Richard M. Simon, Edward L. Korn,
Lisa M. McShane, Michael D. Radmacher,
George W. Wright, and Yingdong Zhao _Robert Gentleman_ 711


**Mathematical** **and** **Statistical** **Methods** **for** **Genetic** **Analysis**
**(2nd ed.).**


Kenneth Lange _Vincent J. Carey_ 712


**Resampling Methods for Dependent Data.**


S. N. Lahiri _Vincent J. Carey_ 712


**Recurrent Events Data Analysis for Product Repairs, Disease**
**Recurrences, and Other Applications.**


Wayne B. Nelson _Russell V. Lenth_ 713


**Telegraphic Reviews** 713


**© 2005 American Statistical Association**
**Journal of the American Statistical Association**
**June 2005, Vol. 100, No. 470, Book Reviews**
**DOI 10.1198/016214505000000178**



**701**


