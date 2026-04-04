# Response to CE Injectivity Proof Review

This note responds to the two issues raised in `injectivity_codex.md`.

## Issue 1: Incorrect sign claim in the kernel proof

Codex is correct. The sentence at line 806:

> "since `rho < w + (1-w)rho` for every `w in (0,1)` and `rho > 0`, the
> logarithm is negative, so `f_w'(rho) < 0` everywhere"

is false for `rho > 1`. The difference `[w + (1-w)rho] - rho = w(1 - rho)`
changes sign at `rho = 1`, so the logarithm in `f_w'` is negative only for
`rho < 1`, zero at `rho = 1`, and positive for `rho > 1`.

### Assessment

The sentence is not load-bearing. The proof does not use `f_w'(rho) < 0`
anywhere after stating it. The actual proof of `H_w'(rho) < 0` works
entirely through the second derivative:

1. `f_w''(rho) = w(1-w) / [rho(w + (1-w)rho)] > 0` (correct, verified)
2. The numerator of `H_w'(rho)` has derivative `f_w''(rho)(1 - rho)`,
   which is positive for `rho < 1` and negative for `rho > 1`
3. Therefore the numerator increases up to `rho = 1` and decreases
   afterward
4. Since it equals zero at `rho = 1`, that is the unique maximum, so the
   numerator is strictly negative for all `rho != 1`
5. The denominator `(1 - rho)^2 > 0`, so `H_w'(rho) < 0`

Steps 1-5 are all correct and do not depend on the sign of `f_w'`. The
false sentence was stated as a preliminary observation ("This already tells
us something useful") but its content is never invoked in the remainder of
the proof.

### Fix

Delete or restrict the sentence. Two options:

**Option A (delete):** Remove the three sentences from "This already tells
us something useful" through "so `f_w'(rho) < 0` everywhere." The proof
flows directly from the expression for `f_w'(rho)` to "We now return to
`H_w` itself and apply the quotient rule."

**Option B (restrict):** Replace with a correctly scoped remark: "Note that
for `rho < 1`, the argument of the logarithm is less than 1, so
`f_w'(rho) < 0` on `(0, 1)`. For `rho > 1` the sign reverses, so `f_w'`
alone does not establish the result. The quotient-rule argument below
handles both cases."

I recommend Option A for brevity.


## Issue 2: Cross-branch injectivity

Codex is correct that the proof as written establishes monotonicity on each
connected component of `s != 0` but does not explicitly rule out the
possibility that `c_3(s_1) = c_3(s_2)` with `s_1 < 0 < s_2`.

### Assessment

The gap is real but the fix is one sentence. The key fact is that `c_3(s)`
is continuous at `s = 0`. The log-utility limit

```
c_3(0) = y_hi^{w_1} * y_lo^{1 - w_1}
```

is well-defined and is approached from both sides (this is the standard
L'Hopital limit of the power mean `(w y_hi^s + (1-w) y_lo^s)^{1/s}` as
`s -> 0`). The proof establishes `dc_3/ds < 0` for all `s != 0` (given
`b != 0`). Continuity at `s = 0` combined with strict monotonicity on each
branch implies `c_3` is strictly decreasing on all of `R`:

- On `s < 0`: `c_3` is strictly decreasing, with `c_3(s) -> c_3(0)` as
  `s -> 0^-`
- On `s > 0`: `c_3` is strictly decreasing, with `c_3(s) -> c_3(0)` as
  `s -> 0^+`

Since `c_3(0)` is the same from both sides, the images `c_3((-inf, 0))`
and `c_3((0, inf))` are `(c_3(0), inf)` and `(0, c_3(0))` respectively
(assuming `c_3 -> inf` as `s -> -inf` and `c_3 -> 0` as `s -> +inf`, which
holds for `y_hi > y_lo > 0`). These intervals are disjoint, so no value is
attained on both branches.

### Fix

After the conclusion that `dc_3/ds != 0` for all `s != 0`, add one
sentence:

> "Because `c_3(s)` is continuous at `s = 0` (the power-mean expression
> converges to the geometric mean `y_hi^{w_1} y_lo^{1-w_1}` from both
> sides), strict monotonicity on each branch extends to global strict
> monotonicity on all of `R`, and therefore global injectivity on
> `s != 0`."

This is the branch-separation step that Codex correctly identifies as
missing.

### Note on Codex's numerical check

Codex's numerical verification using the paper's design (L1 = (10, 50) at
p = 0.25, L3 = (12, 44) at p = 0.25) is consistent with the above: `c_3`
is strictly decreasing on both branches with no image overlap. The
continuity argument explains why this must be true for any design with
`y_hi > y_lo > 0` and `b != 0`.


## Summary

| Issue | Severity | Status | Fix |
|-------|----------|--------|-----|
| 1. False sign claim in `H_w` proof | Correctness error in text | Result unaffected | Delete 3 sentences |
| 2. Cross-branch injectivity | Proof gap | Result true, argument missing | Add 1 sentence on continuity |

Both fixes are local and do not require any restructuring of the proof.
The algebraic core (derivative identity, kernel representation, second-
derivative argument) is sound.
