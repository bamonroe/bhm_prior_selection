/*
 * rdu_bhm_ce.stan
 *
 * Rank-Dependent Utility BHM with Quantity-Space Prior (Beta-Beta).
 * Stan equivalent of code/posteriors/rdu_bhm_ce.py + code/posteriors/common_fns.py.
 *
 * The prior on subject-level parameters is specified in quantity space
 * (3 certainty equivalents + 1 choice probability) via a Beta-Beta
 * hierarchical model, then transformed to parameter space with a
 * Jacobian correction:
 *
 *     p(theta) = p_Y(g(theta)) |det J_g(theta)|
 *
 * where g: (r, phi_raw, eta_raw, lambda_raw) -> (CE1, CE2, CE3, P_choice).
 *
 * Subject parameters (4 per subject, centered parameterization):
 *   r          : CRRA risk aversion coefficient (unconstrained)
 *   phi_raw    : log(Prelec curvature), phi = exp(phi_raw) > 0
 *   eta_raw    : log(Prelec elevation), eta = exp(eta_raw) > 0
 *   lambda_raw : log(Fechner noise),  lambda = exp(lambda_raw) > 0
 *
 * Hyperparameters (8 total, on [0,1]):
 *   mu_pop[k]  ~ Beta(a_mu[k], b_mu[k])   population mean for quantity k
 *   psi_pop[k] ~ Beta(a_psi[k], b_psi[k]) dispersion for quantity k
 *   k = 1..4 for CE1, CE2, CE3, P_choice
 *
 * Population prior (Beta-Beta):
 *   z_ik = (Y_ik - lo_k) / (hi_k - lo_k)   normalized to [0,1]
 *   z_ik | mu_k, psi_k ~ Beta(alpha_pop, beta_pop)
 *     alpha_pop = mu_k * (1 - psi_k) / psi_k
 *     beta_pop  = (1 - mu_k) * (1 - psi_k) / psi_k
 *
 * Likelihood: Contextual utility (Wilcox 2011) with Fechner noise.
 *   Delta = (V_L - V_R) / (lambda * (u_max - u_min))
 *   P(choose L) = inv_logit(Delta)
 *
 * Data format:
 *   Lottery prizes must be sorted ascending within each triple.
 *   Probabilities sum to 1.  Binary lotteries should be zero-padded
 *   to 3-outcome format: prizes = [x_lo, x_lo, x_hi],
 *                         probs  = [1-p, 0, p].
 *   Choices: 1 = Left, 0 = Right.
 *
 * The Jacobian |det J_g(theta)| is computed analytically via a three-component
 * decomposition: Prelec W map, CE power-mean map, and choice probability.
 * See code/jacobian_symbolic.py for the derivation and verification.
 */

functions {
  /**
   * Normalized CRRA utility function.
   * v(x) = x^(1-r) / (1-r)
   * u(x) = (v(x) - v(x_min)) / (v(x_max) - v(x_min))
   * Continuous at r = 1, where u(x) = ln(x/x_min) / ln(x_max/x_min).
   */
  real crra_utility(real x, real r) {
    real x_s = fmax(x, 1e-10);
    real r_s = fmin(fmax(r, -10.0), 10.0);
    real s = 1 - r_s;
    real x_min = 10.0;
    real x_max = 50.0;
    if (abs(s) < 1e-10)
      return log(x_s / x_min) / log(x_max / x_min);
    else {
      real xs = pow(x_s, s);
      real xmin_s = pow(x_min, s);
      real xmax_s = pow(x_max, s);
      return (xs - xmin_s) / (xmax_s - xmin_s);
    }
  }

  /**
   * Inverse of normalized CRRA utility.
   * CE = (u * (x_max^s - x_min^s) + x_min^s)^(1/s)
   * At r = 1: CE = x_min * (x_max / x_min)^u
   */
  real crra_inverse(real u_val, real r) {
    real r_s = fmin(fmax(r, -10.0), 10.0);
    real s = 1 - r_s;
    real x_min = 10.0;
    real x_max = 50.0;
    if (abs(s) < 1e-10)
      return x_min * pow(x_max / x_min, u_val);
    else {
      real xmin_s = pow(x_min, s);
      real xmax_s = pow(x_max, s);
      real ce_s = u_val * (xmax_s - xmin_s) + xmin_s;
      return pow(fmax(ce_s, 1e-30), 1.0 / s);
    }
  }

  /**
   * Prelec 2-parameter probability weighting function.
   * w(p) = exp(-eta * (-ln p)^phi)
   */
  real prelec_pwf(real p, real phi, real eta) {
    real phi_s = fmax(phi, 0.01);
    real eta_s = fmax(eta, 0.01);
    real p_s = fmin(fmax(p, 1e-10), 1 - 1e-10);
    return exp(-eta_s * pow(-log(p_s), phi_s));
  }

  /**
   * RDU value for a 3-outcome lottery (prizes sorted ascending).
   *
   * Cumulative probabilities: C3 = p3, C2 = p2 + p3.
   * Decision weights: pi3 = w(C3), pi2 = w(C2) - w(C3), pi1 = 1 - w(C2).
   */
  real rdu_value(real u1, real u2, real u3,
                 real p1, real p2, real p3,
                 real phi, real eta) {
    real C3 = p3;
    real C2 = p2 + p3;
    real w_C3 = prelec_pwf(C3, phi, eta);
    real w_C2 = prelec_pwf(C2, phi, eta);
    return (1.0 - w_C2) * u1 + (w_C2 - w_C3) * u2 + w_C3 * u3;
  }

  /**
   * Certainty equivalent for a 3-outcome lottery under RDU.
   */
  real rdu_ce(real r, real phi, real eta,
              vector prizes, vector probs) {
    real u1 = crra_utility(prizes[1], r);
    real u2 = crra_utility(prizes[2], r);
    real u3 = crra_utility(prizes[3], r);
    real v = rdu_value(u1, u2, u3, probs[1], probs[2], probs[3], phi, eta);
    return crra_inverse(v, r);
  }

  /**
   * P(choose A) for a reference lottery pair (contextual utility).
   */
  real choice_prob_ref(real r, real phi, real eta, real lam,
                       vector prizes_A, vector probs_A,
                       vector prizes_B, vector probs_B) {
    real u_A1 = crra_utility(prizes_A[1], r);
    real u_A2 = crra_utility(prizes_A[2], r);
    real u_A3 = crra_utility(prizes_A[3], r);
    real u_B1 = crra_utility(prizes_B[1], r);
    real u_B2 = crra_utility(prizes_B[2], r);
    real u_B3 = crra_utility(prizes_B[3], r);

    real V_A = rdu_value(u_A1, u_A2, u_A3,
                         probs_A[1], probs_A[2], probs_A[3], phi, eta);
    real V_B = rdu_value(u_B1, u_B2, u_B3,
                         probs_B[1], probs_B[2], probs_B[3], phi, eta);

    real u_max = fmax(fmax(u_A1, fmax(u_A2, u_A3)),
                      fmax(u_B1, fmax(u_B2, u_B3)));
    real u_min = fmin(fmin(u_A1, fmin(u_A2, u_A3)),
                      fmin(u_B1, fmin(u_B2, u_B3)));
    real u_range = fmax(u_max - u_min, 1e-10);

    return inv_logit((V_A - V_B) / (lam * u_range));
  }

  /**
   * Full 4D forward map: theta -> (CE1, CE2, CE3, P_choice).
   */
  vector forward_map(real r, real phi_raw, real eta_raw, real lambda_raw,
                     array[] vector ref_prizes, array[] vector ref_probs,
                     vector pair_A_prizes, vector pair_A_probs,
                     vector pair_B_prizes, vector pair_B_probs) {
    real phi = exp(phi_raw);
    real eta = exp(eta_raw);
    real lam = exp(lambda_raw);
    vector[4] Y;
    Y[1] = rdu_ce(r, phi, eta, ref_prizes[1], ref_probs[1]);
    Y[2] = rdu_ce(r, phi, eta, ref_prizes[2], ref_probs[2]);
    Y[3] = rdu_ce(r, phi, eta, ref_prizes[3], ref_probs[3]);
    Y[4] = choice_prob_ref(r, phi, eta, lam,
                           pair_A_prizes, pair_A_probs,
                           pair_B_prizes, pair_B_probs);
    return Y;
  }

  /**
   * CE of a binary lottery in (r, w) space via the power mean formula.
   * CE = (w * xhi^s + (1-w) * xlo^s)^(1/s),  s = 1-r.
   */
  real ce_binary(real w, real xlo, real xhi, real s) {
    real inner = w * pow(xhi, s) + (1 - w) * pow(xlo, s);
    return pow(inner, 1.0 / s);
  }

  /**
   * dCE/dw for a binary lottery: (xhi^s - xlo^s) / (s * CE^(s-1)).
   */
  real dce_dw(real w, real xlo, real xhi, real s) {
    real ce = ce_binary(w, xlo, xhi, s);
    return (pow(xhi, s) - pow(xlo, s)) / (s * pow(ce, s - 1));
  }

  /**
   * dCE/dr for a binary lottery (analytic, via log-differentiation).
   *
   * dCE/dr = (CE/s) * [ln(CE) - (w*xhi^s*ln(xhi) + (1-w)*xlo^s*ln(xlo)) / inner]
   *
   * The bracketed term is the Jensen gap in log-space: ln(CE) minus the
   * weighted average of ln(outcome).
   */
  real dce_dr(real w, real xlo, real xhi, real s) {
    real xhi_s = pow(xhi, s);
    real xlo_s = pow(xlo, s);
    real inner = w * xhi_s + (1 - w) * xlo_s;
    real ce = pow(inner, 1.0 / s);
    real weighted_lnx = w * xhi_s * log(xhi) + (1 - w) * xlo_s * log(xlo);
    return (ce / s) * (log(ce) - weighted_lnx / inner);
  }

  /**
   * log |det J_g(theta)| — fully analytic decomposition.
   *
   * The 4x4 Jacobian factors into three independent components:
   *
   *   log|det J| = log|det J_W| + log|det J_CE| + log|dP/dlambda_raw|
   *
   * Component 1 (Prelec W map):
   *   W: (phi_raw, eta_raw) -> (w1, w2)
   *   det(J_W) = w1*w2*A1*A2*phi*(ln(t1) - ln(t2))
   *
   * Component 2 (CE map in (r,w1,w2) space):
   *   With shared probability p3=p1 (so w3=w1), the 3x3 determinant is:
   *   det(J_CE) = dCE2/dw2 * (dCE1/dr*dCE3/dw1 - dCE3/dr*dCE1/dw1)
   *
   * Component 3 (choice probability):
   *   dP/dlambda_raw = -delta * P * (1-P)
   */
  real log_abs_det_jac(real r, real phi_raw, real eta_raw, real lambda_raw,
                       array[] vector ref_prizes, array[] vector ref_probs,
                       vector pair_A_prizes, vector pair_A_probs,
                       vector pair_B_prizes, vector pair_B_probs) {
    real phi = exp(phi_raw);
    real eta = exp(eta_raw);
    real lam = exp(lambda_raw);
    real s = 1 - r;

    // --- Component 1: Prelec W map ---
    // Probabilities on better outcome for L1 and L2
    // (probs are [p_worst, p_mid, p_best] sorted ascending by prize)
    real p1 = ref_probs[1][3];  // L1 prob on best outcome
    real p2 = ref_probs[2][3];  // L2 prob on best outcome
    real t1 = -log(p1);
    real t2 = -log(p2);
    real A1 = eta * pow(t1, phi);
    real A2 = eta * pow(t2, phi);
    real w1 = exp(-A1);
    real w2 = exp(-A2);
    real log_det_W = -A1 - A2 + 2 * eta_raw
                     + phi * (log(t1) + log(t2))
                     + phi_raw + log(abs(log(t1 / t2)));

    // --- Component 2: CE map in (r, w1, w2) space ---
    // Shared probability: w3 = w1 (p3 = p1)
    real xlo1 = ref_prizes[1][1]; real xhi1 = ref_prizes[1][3];
    real xlo2 = ref_prizes[2][1]; real xhi2 = ref_prizes[2][3];
    real xlo3 = ref_prizes[3][1]; real xhi3 = ref_prizes[3][3];

    real dce1_dw1_val = dce_dw(w1, xlo1, xhi1, s);
    real dce2_dw2_val = dce_dw(w2, xlo2, xhi2, s);
    real dce3_dw1_val = dce_dw(w1, xlo3, xhi3, s);  // w3 = w1

    real dce1_dr_val = dce_dr(w1, xlo1, xhi1, s);
    real dce3_dr_val = dce_dr(w1, xlo3, xhi3, s);

    real det_CE = dce2_dw2_val
                  * (dce1_dr_val * dce3_dw1_val - dce3_dr_val * dce1_dw1_val);
    real log_det_CE = log(abs(det_CE));

    // --- Component 3: dP/dlambda_raw ---
    // Recompute Delta for the reference pair
    real u_A1 = crra_utility(pair_A_prizes[1], r);
    real u_A2 = crra_utility(pair_A_prizes[2], r);
    real u_A3 = crra_utility(pair_A_prizes[3], r);
    real u_B1 = crra_utility(pair_B_prizes[1], r);
    real u_B2 = crra_utility(pair_B_prizes[2], r);
    real u_B3 = crra_utility(pair_B_prizes[3], r);

    real V_A = rdu_value(u_A1, u_A2, u_A3,
                         pair_A_probs[1], pair_A_probs[2], pair_A_probs[3],
                         phi, eta);
    real V_B = rdu_value(u_B1, u_B2, u_B3,
                         pair_B_probs[1], pair_B_probs[2], pair_B_probs[3],
                         phi, eta);

    real u_max = fmax(fmax(u_A1, fmax(u_A2, u_A3)),
                      fmax(u_B1, fmax(u_B2, u_B3)));
    real u_min = fmin(fmin(u_A1, fmin(u_A2, u_A3)),
                      fmin(u_B1, fmin(u_B2, u_B3)));
    real u_range = fmax(u_max - u_min, 1e-10);

    real delta = (V_A - V_B) / (lam * u_range);
    real P_val = inv_logit(delta);
    real log_dP = log(abs(delta)) + log(P_val) + log1m(P_val);

    return log_det_W + log_det_CE + log_dP;
  }
}

data {
  int<lower=1> N_subjects;
  int<lower=1> N_choices;

  // Observed choices: 1 = Left (A), 0 = Right (B)
  array[N_subjects, N_choices] int<lower=0, upper=1> choices;

  // 3-outcome lottery data (prizes sorted ascending, probs sum to 1)
  array[N_subjects, N_choices] vector[3] prizes_L;
  array[N_subjects, N_choices] vector[3] prizes_R;
  array[N_subjects, N_choices] vector[3] probs_L;
  array[N_subjects, N_choices] vector[3] probs_R;

  // 3 reference lotteries for CE-space prior (3-outcome padded)
  array[3] vector[3] ref_lot_prizes;
  array[3] vector[3] ref_lot_probs;

  // Reference pair for choice probability identification
  vector[3] ref_pair_A_prizes;
  vector[3] ref_pair_A_probs;
  vector[3] ref_pair_B_prizes;
  vector[3] ref_pair_B_probs;

  // Beta-Beta hyperprior parameters for 4 quantities
  // (k=1: CE1, k=2: CE2, k=3: CE3, k=4: P_choice)
  vector<lower=0>[4] hp_a_mu;
  vector<lower=0>[4] hp_b_mu;
  vector<lower=0>[4] hp_a_psi;
  vector<lower=0>[4] hp_b_psi;
  vector[4] hp_lo;   // lower bound per quantity
  vector[4] hp_hi;   // upper bound per quantity
}

parameters {
  // Population hyperparameters on [0,1]
  vector<lower=0, upper=1>[4] mu_pop;   // population mean per quantity
  vector<lower=0, upper=1>[4] psi_pop;  // dispersion per quantity

  // Subject-level parameters (unconstrained; prior comes from quantity space)
  vector[N_subjects] r;
  vector[N_subjects] phi_raw;
  vector[N_subjects] eta_raw;
  vector[N_subjects] lambda_raw;
}

model {
  // ---- Hyperpriors ----
  for (k in 1:4) {
    mu_pop[k] ~ beta(hp_a_mu[k], hp_b_mu[k]);
    psi_pop[k] ~ beta(hp_a_psi[k], hp_b_psi[k]);
  }

  // ---- Subject-level: quantity-space prior + Jacobian + likelihood ----
  for (i in 1:N_subjects) {
    real phi_i = exp(phi_raw[i]);
    real eta_i = exp(eta_raw[i]);
    real lam_i = exp(lambda_raw[i]);

    // Forward map: theta_i -> (CE1, CE2, CE3, P_choice)
    vector[4] Y = forward_map(r[i], phi_raw[i], eta_raw[i], lambda_raw[i],
                               ref_lot_prizes, ref_lot_probs,
                               ref_pair_A_prizes, ref_pair_A_probs,
                               ref_pair_B_prizes, ref_pair_B_probs);

    // Beta-Beta population prior on normalized quantities
    for (k in 1:4) {
      real z_k = (Y[k] - hp_lo[k]) / (hp_hi[k] - hp_lo[k]);
      z_k = fmin(fmax(z_k, 1e-10), 1 - 1e-10);

      real psi_k = fmin(fmax(psi_pop[k], 1e-6), 1 - 1e-6);
      real mu_k = fmin(fmax(mu_pop[k], 1e-6), 1 - 1e-6);
      real ratio = (1 - psi_k) / psi_k;
      real alpha_pop = fmax(mu_k * ratio, 1e-4);
      real beta_pop  = fmax((1 - mu_k) * ratio, 1e-4);

      target += beta_lpdf(z_k | alpha_pop, beta_pop);
      target += -log(hp_hi[k] - hp_lo[k]);   // Jacobian dz/dY = 1/(hi-lo)
    }

    // Jacobian correction for change of variables theta -> Y
    target += log_abs_det_jac(r[i], phi_raw[i], eta_raw[i], lambda_raw[i],
                               ref_lot_prizes, ref_lot_probs,
                               ref_pair_A_prizes, ref_pair_A_probs,
                               ref_pair_B_prizes, ref_pair_B_probs);

    // Data likelihood (contextual utility / Fechner)
    for (j in 1:N_choices) {
      real u_L1 = crra_utility(prizes_L[i, j][1], r[i]);
      real u_L2 = crra_utility(prizes_L[i, j][2], r[i]);
      real u_L3 = crra_utility(prizes_L[i, j][3], r[i]);
      real u_R1 = crra_utility(prizes_R[i, j][1], r[i]);
      real u_R2 = crra_utility(prizes_R[i, j][2], r[i]);
      real u_R3 = crra_utility(prizes_R[i, j][3], r[i]);

      real V_L = rdu_value(u_L1, u_L2, u_L3,
                           probs_L[i, j][1], probs_L[i, j][2], probs_L[i, j][3],
                           phi_i, eta_i);
      real V_R = rdu_value(u_R1, u_R2, u_R3,
                           probs_R[i, j][1], probs_R[i, j][2], probs_R[i, j][3],
                           phi_i, eta_i);

      real u_max = fmax(fmax(u_L1, fmax(u_L2, u_L3)),
                        fmax(u_R1, fmax(u_R2, u_R3)));
      real u_min = fmin(fmin(u_L1, fmin(u_L2, u_L3)),
                        fmin(u_R1, fmin(u_R2, u_R3)));
      real u_range = fmax(u_max - u_min, 1e-10);

      real delta = (V_L - V_R) / (lam_i * u_range);
      target += bernoulli_logit_lpmf(choices[i, j] | delta);
    }
  }
}

generated quantities {
  // Subject parameters on natural scale
  array[N_subjects] real phi;
  array[N_subjects] real eta;
  array[N_subjects] real lambda;

  for (i in 1:N_subjects) {
    phi[i] = exp(phi_raw[i]);
    eta[i] = exp(eta_raw[i]);
    lambda[i] = exp(lambda_raw[i]);
  }
}
