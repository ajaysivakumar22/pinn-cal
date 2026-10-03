/**
 * PINN-Cal — Scientific & Engineering Concepts Glossary
 *
 * Provides progressive disclosure explanations for all key RF, MIMO,
 * mutual coupling, and machine learning calibration terms.
 */

export interface ConceptDefinition {
  id: string;
  name: string;
  category: 'RF & Array' | 'Impairments' | 'PINN & Machine Learning' | 'Beamforming & Metrics';
  summary: string;
  simpleExplanation: string;
  technicalExplanation: string;
  formula?: string;
  whyItMatters: string;
}

export const CONCEPTS: Record<string, ConceptDefinition> = {
  phase_error: {
    id: 'phase_error',
    name: 'Phase Impairment (σ_φ)',
    category: 'Impairments',
    summary: 'Random hardware delay variation between individual antenna transceivers.',
    simpleExplanation:
      'In a 5G massive MIMO array, all 64 antennas must emit waves in synchronized lockstep. A phase error means one antenna is slightly behind or ahead in time, distorting the collective signal wave.',
    technicalExplanation:
      'Modeled as an independent zero-mean Gaussian perturbation delta_phi_n ~ N(0, sigma_phi^2), typical sigma_phi = 20 degrees in mmWave transceivers due to LO distribution skew and mixer variations.',
    formula: 'h_n = (1 + \\delta g_n) \\cdot \\exp(j \\cdot \\delta\\phi_n)',
    whyItMatters:
      'Uncalibrated phase errors cause beam pointing deflection, gain reduction, and elevated sidelobes that leak interference to adjacent users.',
  },
  gain_error: {
    id: 'gain_error',
    name: 'Gain Impairment (σ_g)',
    category: 'Impairments',
    summary: 'Random power output variations among RF power amplifier channels.',
    simpleExplanation:
      'Even when commanded to output the exact same power, manufacturing differences cause some antenna amplifiers to emit slightly stronger or weaker signals than others.',
    technicalExplanation:
      'Modeled as a log-normal gain mismatch delta_g_n ~ N(0, sigma_g^2) in dB, standard deviation sigma_g = 2.0 dB across 64 RF frontend branches.',
    formula: '|h_n| = 10^{\\delta g_n / 20}',
    whyItMatters:
      'Gain mismatches broaden the main beamwidth and destroy sidelobe suppression (PSLL), degrading 5G spatial multiplexing capacity.',
  },
  mutual_coupling: {
    id: 'mutual_coupling',
    name: 'Mutual Coupling (Z-Matrix)',
    category: 'RF & Array',
    summary: 'Electromagnetic cross-talk between closely spaced antenna elements.',
    simpleExplanation:
      'Because 64 antennas are packed tightly on a small circuit board (half a wavelength = 5.35 mm apart at 28 GHz), energy radiating from one antenna leaks into its neighbors, altering their currents.',
    technicalExplanation:
      'Modeled via a compact reciprocal impedance network Z in C^{64 x 64}. Nearest-neighbor coupling c_1 = 0.12, spatial decay alpha = 0.55, phase step beta = 0.35 rad. Z_true differs from nominal Z_nom by up to +/-20% hidden uncertainty.',
    formula: 'Z_{m,n} = Z_{self} \\cdot c_1 \\cdot e^{-\\alpha |m-n|} \\cdot e^{-j \\beta |m-n|} \\quad (m \\neq n)',
    whyItMatters:
      'Mutual coupling makes antenna currents dependent on all neighboring elements. Calibration must account for both coupling and individual hardware mismatches.',
  },
  pinn: {
    id: 'pinn',
    name: 'Physics-Informed Neural Network (PINN)',
    category: 'PINN & Machine Learning',
    summary: 'A deep neural network constrained by electromagnetic circuit laws during training.',
    simpleExplanation:
      'Instead of learning blindly from data, a PINN is taught electromagnetic equations. If its predictions violate physical circuit laws (like Ohm’s law V = Z * I), it penalizes itself.',
    technicalExplanation:
      'PINN-Cal V4 two-stage architecture: Tikhonov physics preconditioner (δ=0.15) combined with bounded MLP residual denoiser (128 → 128 → 64 → 128, α=0.50, 33,088 parameters). The residual network takes 128 real features representing Re/Im(c_prior) and outputs 128 real values representing Δc, yielding final compensation ĉ = c_prior + Δc.',
    formula: '\\hat{c} = c_{prior} + \\Delta c, \\quad \\Delta c = \\alpha \\tanh(r_{\\theta}(\\text{Re/Im}(c_{prior}))), \\quad \\alpha = 0.50, \\; \\delta = 0.15',
    whyItMatters:
      'Physics constraints guide the network to physically plausible calibration solutions even in high-dimensional (64-element) mmWave regimes.',
  },
  compensation_weight: {
    id: 'compensation_weight',
    name: 'Calibration Compensation Weight (C)',
    category: 'Beamforming & Metrics',
    summary: 'Complex correction factors multiplied with antenna excitation to cancel impairments.',
    simpleExplanation:
      'If antenna #17 is lagging by +7 degrees with +1.1 dB excess power, its compensation weight advances it by -7 degrees and attenuates it by -1.1 dB, restoring ideal emission.',
    technicalExplanation:
      'Defined as C_true = V_ideal / V_hardware = H^-1. When applied to impaired hardware voltage V_hw, V_corr = C * V_hw = V_ideal.',
    formula: 'V_{corrected} = C \\odot V_{hardware} \\approx V_{ideal}',
    whyItMatters:
      'Applying accurate digital or analog compensation weights restores the intended far-field radiation pattern without physically rebuilding the RF hardware.',
  },
  beamforming: {
    id: 'beamforming',
    name: 'Beamforming & Phased Arrays',
    category: 'Beamforming & Metrics',
    summary: 'Constructive wave interference directing radio energy toward a specific receiver.',
    simpleExplanation:
      'Rather than broadcasting radio waves in all directions like a lightbulb, 64 antennas coordinate their waves like a laser pointer, focusing energy directly at the mobile user.',
    technicalExplanation:
      'Uniform Linear Array (ULA) radiation pattern P(theta) = |a(theta)^H * w|^2 / N^2, where a(theta) is the steering vector and w is the array weight vector.',
    formula: 'a_n(\\theta) = \\exp\\left(j \\frac{2\\pi}{\\lambda} x_n \\sin\\theta\\right)',
    whyItMatters:
      '5G mmWave signals suffer severe atmospheric attenuation. Beamforming provides the high directional antenna gain necessary for multi-gigabit wireless links.',
  },
  nmse: {
    id: 'nmse',
    name: 'Normalized Mean Squared Error (NMSE)',
    category: 'Beamforming & Metrics',
    summary: 'Decibel metric quantifying total calibration vector accuracy.',
    simpleExplanation:
      'Measures how close the estimated calibration vector is to the ground truth. More negative values in decibels mean higher accuracy and lower residual error.',
    technicalExplanation:
      'NMSE(dB) = 10 * log10( ||C_pred - C_true||^2 / ||C_true||^2 ). An NMSE of -14.45 dB represents ~15.1% relative vector error.',
    formula: '\\text{NMSE} = 10 \\log_{10} \\left( \\frac{\\|C_{pred} - C_{true}\\|^2}{\\|C_{true}\\|^2} \\right)',
    whyItMatters:
      'Standard benchmark metric across research literature for comparing calibration methods on equal footing.',
  },
  snr: {
    id: 'snr',
    name: 'Signal-to-Noise Ratio (SNR)',
    category: 'Impairments',
    summary: 'Ratio of true antenna current power to measurement receiver noise power.',
    simpleExplanation:
      'When measuring antenna currents, thermal electronics noise contaminates the observation. Higher SNR (e.g. 30 dB) means a cleaner measurement; lower SNR (10 dB) means heavy static.',
    technicalExplanation:
      'Complex additive white Gaussian noise w ~ CN(0, sigma_w^2) added to physical coupled current I_true: I_obs = I_true + w. Nominal operating SNR = 20 dB.',
    formula: '\\text{SNR} = 10 \\log_{10}\\left(\\frac{P_{signal}}{P_{noise}}\\right)',
    whyItMatters:
      'Calibration algorithms must remain robust in low-SNR environments where measurement noise obscures subtle hardware errors.',
  },
  beamwidth_psll: {
    id: 'beamwidth_psll',
    name: 'Beamwidth (HPBW) & Sidelobe Level (PSLL)',
    category: 'Beamforming & Metrics',
    summary: 'Key beam quality metrics measuring angular sharpness and leakage.',
    simpleExplanation:
      'Beamwidth is how narrow the main signal beam is (-3 dB width). PSLL is the ratio of the peak main beam to the tallest unintended side leakage beam.',
    technicalExplanation:
      'For a 64-element half-wavelength ULA, theoretical HPBW ~ 1.78 degrees at broadside. PSLL for uniform weighting is approximately -13.2 dB.',
    formula: '\\text{PSLL} = \\max_{\\text{sidelobes}} P(\\theta) - P(\\theta_{peak}) \\quad (\\text{in dB})',
    whyItMatters:
      'Hardware impairments degrade PSLL and broaden beamwidth, causing unintended interference with neighboring mobile cells.',
  },
};
