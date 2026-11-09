/**
 * Prediction-Powered Inference (PPI) for Debiasing
 * Based on: Angelopoulos et al. (2023) "Prediction-Powered Inference"
 * https://arxiv.org/abs/2301.09633
 */

export interface PPIGroupStats {
  group: string;
  n_labeled: number;
  n_total: number;
  /** Classical (naive) positive rate from labels only */
  classicalRate: number;
  /** PPI-corrected positive rate (bias-corrected using ML predictions) */
  ppiRate: number;
  /** Lower bound of 95% CI for ppiRate */
  ppiCILow: number;
  /** Upper bound of 95% CI for ppiRate */
  ppiCIHigh: number;
  /** Rectifier value (how much model systematically over/under-predicts) */
  rectifier: number;
  /** Whether the CI is valid (n_labeled >= 5) */
  valid: boolean;
}

export interface PPIResult {
  groups: PPIGroupStats[];
  /** PPI-corrected SPD: ppiRate[unprivileged] - ppiRate[privileged] */
  ppiSPD: number;
  /** Classical SPD (from naive rates) */
  classicalSPD: number;
  /** PPI-corrected Disparate Impact */
  ppiDI: number;
  /** Whether the PPI improvement is statistically significant */
  significantImprovement: boolean;
  privilegedGroup: string;
  unprivilegedGroup: string;
  /** The lambda parameter (how much to trust predictions vs labels). Range [0,1] */
  lambda: number;
}

export interface PPIDebiasedRow {
  row: Record<string, any>;
  /** Original label value */
  originalLabel: string;
  /** PPI-corrected label (may be flipped) */
  correctedLabel: string;
  /** Whether this row's label was corrected */
  wasCorrected: boolean;
  /** ML model prediction for this row (0 or 1) */
  modelPrediction: number;
  /** Model prediction confidence / probability (approximated) */
  confidence: number;
}

/**
 * Compute PPI group statistics.
 *
 * @param allData        Full dataset (labeled + unlabeled pool)
 * @param labeledIndices Indices into allData that have ground-truth labels (train split)
 * @param predictions    ML model predictions for ALL rows (length = allData.length)
 * @param predOnLabeled  ML model predictions for the labeled subset only
 * @param targetColumn   Name of the target/outcome column
 * @param sensitiveColumns Array of sensitive attribute column names
 * @param positiveClass  String value that represents the positive class
 * @param lambda         Weight for prediction term (0 = pure classical, 1 = full PPI)
 * @param alpha          Significance level (default 0.05 → 95% CI)
 */
export function computePPIStats(
  allData: Record<string, any>[],
  labeledIndices: number[],
  predictions: number[],          // for all rows
  predOnLabeled: number[],        // for labeled rows only (same order as labeledIndices)
  targetColumn: string,
  sensitiveColumns: string[],
  positiveClass: string,
  lambda: number = 1.0,
  alpha: number = 0.05,
): PPIResult {
  const getGroup = (row: Record<string, any>) =>
    sensitiveColumns.map(col => `${col}: ${row[col]}`).join(' | ');

  // Collect unique groups
  const groupSet = new Set(allData.map(r => getGroup(r)));
  const groups = Array.from(groupSet).filter(
    g => !g.includes('null') && !g.includes('undefined') && g !== ''
  );

  const zAlpha2 = alpha === 0.05 ? 1.96 : alpha === 0.01 ? 2.576 : 1.645;

  const groupStats: PPIGroupStats[] = groups.map(group => {
    // All-data indices for this group
    const allGroupIdx = allData
      .map((r, i) => ({ r, i }))
      .filter(({ r }) => getGroup(r) === group)
      .map(({ i }) => i);

    // Labeled indices for this group
    const labeledGroupIdx = labeledIndices.filter(i => getGroup(allData[i]) === group);

    const n_total = allGroupIdx.length;
    const n_labeled = labeledGroupIdx.length;

    // Classical rate (from labels only)
    let classicalPos = 0;
    labeledGroupIdx.forEach(i => {
      if (String(allData[i][targetColumn]) === positiveClass) classicalPos++;
    });
    const classicalRate = n_labeled > 0 ? classicalPos / n_labeled : 0;

    if (n_labeled < 5 || n_total < 2) {
      return {
        group, n_labeled, n_total, classicalRate,
        ppiRate: classicalRate, ppiCILow: 0, ppiCIHigh: 1,
        rectifier: 0, valid: false,
      };
    }

    // Model predictions for all rows in this group
    const predsGroup = allGroupIdx.map(i => predictions[i]);
    const meanPredAll = predsGroup.reduce((a, b) => a + b, 0) / predsGroup.length;

    // Variance of predictions over all group rows
    const varPredAll = predsGroup.reduce((acc, p) => acc + (p - meanPredAll) ** 2, 0) / predsGroup.length;

    // Model predictions for labeled rows in this group
    // Map labeledGroupIdx → position in predOnLabeled array
    const predLabeledGroup: number[] = [];
    const labelsGroup: number[] = [];
    labeledGroupIdx.forEach((globalIdx) => {
      // posInLabeled within labeledIndices
      const pos = labeledIndices.indexOf(globalIdx);
      if (pos >= 0 && pos < predOnLabeled.length) {
        predLabeledGroup.push(predOnLabeled[pos]);
        labelsGroup.push(String(allData[globalIdx][targetColumn]) === positiveClass ? 1 : 0);
      }
    });

    // Rectifier: mean(Y_labeled - Ŷ_labeled)
    const rectifierVals = labelsGroup.map((y, i) => y - predLabeledGroup[i]);
    const rectifier = rectifierVals.reduce((a, b) => a + b, 0) / Math.max(1, rectifierVals.length);

    // PPI estimate: lambda * meanPredAll + rectifier (when lambda = 1, full PPI)
    const ppiRate = Math.max(0, Math.min(1, lambda * meanPredAll + rectifier));

    // Variance of rectifier values
    const varRectifier = rectifierVals.reduce(
      (acc, r) => acc + (r - rectifier) ** 2, 0
    ) / Math.max(1, rectifierVals.length);

    // PPI CI: z * sqrt(lambda^2 * varPredAll / n_total + varRectifier / n_labeled)
    const ciHalfWidth = zAlpha2 * Math.sqrt(
      (lambda * lambda * varPredAll) / n_total +
      varRectifier / n_labeled
    );
    const ppiCILow  = Math.max(0, ppiRate - ciHalfWidth);
    const ppiCIHigh = Math.min(1, ppiRate + ciHalfWidth);

    return { group, n_labeled, n_total, classicalRate, ppiRate, ppiCILow, ppiCIHigh, rectifier, valid: true };
  });

  // Sort by ppiRate descending to identify privileged / unprivileged
  const validGroups = groupStats.filter(g => g.valid && g.n_total >= 5);
  if (validGroups.length < 2) {
    const all = groupStats.sort((a, b) => b.ppiRate - a.ppiRate);
    return {
      groups: groupStats,
      ppiSPD: 0, classicalSPD: 0, ppiDI: 1,
      significantImprovement: false,
      privilegedGroup: all[0]?.group || 'N/A',
      unprivilegedGroup: all[all.length - 1]?.group || 'N/A',
      lambda,
    };
  }

  validGroups.sort((a, b) => b.ppiRate - a.ppiRate);
  const privileged   = validGroups[0];
  const unprivileged = validGroups[validGroups.length - 1];

  const ppiSPD       = unprivileged.ppiRate - privileged.ppiRate;
  const classicalSPD = unprivileged.classicalRate - privileged.classicalRate;

  const ppiDI = privileged.ppiRate > 0
    ? unprivileged.ppiRate / privileged.ppiRate
    : 1;

  // Significant improvement if PPI CIs do not overlap between groups
  const ciOverlap = unprivileged.ppiCIHigh > privileged.ppiCILow || privileged.ppiCIHigh > unprivileged.ppiCILow;
  const significantImprovement = !ciOverlap;

  return {
    groups: groupStats,
    ppiSPD,
    classicalSPD,
    ppiDI,
    significantImprovement,
    privilegedGroup:   privileged.group,
    unprivilegedGroup: unprivileged.group,
    lambda,
  };
}

/**
 * Generate a PPI-debiased dataset.
 * 
 * Uses the PPI-corrected group rates as targets, then:
 * 1. For rows where the model is *confident* and the label differs from model prediction,
 *    preferentially flip labels toward PPI target rates.
 * 2. This produces a dataset whose group positive rates match the PPI-corrected estimates
 *    rather than the potentially biased historical labels.
 */
export function generatePPIDebiasedDataset(
  allData: Record<string, any>[],
  predictions: number[],
  ppiResult: PPIResult,
  targetColumn: string,
  sensitiveColumns: string[],
  positiveClass: string,
  negativeClass: string,
  strength: number = 1.0,   // 0 = no change, 1 = full PPI correction
): { debiasedData: Record<string, any>[]; correctionLog: string[]; totalFlips: number } {
  const getGroup = (row: Record<string, any>) =>
    sensitiveColumns.map(col => `${col}: ${row[col]}`).join(' | ');

  const correctionLog: string[] = [];
  let totalFlips = 0;

  // Build a lookup of target rates from PPI
  const groupTargetMap = new Map<string, number>();
  ppiResult.groups.forEach(g => {
    // Interpolate between classicalRate and ppiRate based on strength
    const target = g.classicalRate + (g.ppiRate - g.classicalRate) * strength;
    groupTargetMap.set(g.group, Math.max(0, Math.min(1, target)));
  });

  // Deep copy all data
  const debiasedData = allData.map(r => ({ ...r }));

  // Process each group
  const groupNames = Array.from(groupTargetMap.keys());
  for (const group of groupNames) {
    const groupRows = debiasedData
      .map((r, i) => ({ r, i }))
      .filter(({ r }) => getGroup(r) === group);

    if (groupRows.length === 0) continue;

    const targetRate = groupTargetMap.get(group)!;
    const currentPosCount = groupRows.filter(({ r }) => String(r[targetColumn]) === positiveClass).length;
    const targetPosCount  = Math.round(targetRate * groupRows.length);
    const flipDelta        = targetPosCount - currentPosCount;

    if (flipDelta === 0) {
      correctionLog.push(`Group "${group}": no correction needed (rate already at PPI target ${(targetRate * 100).toFixed(1)}%)`);
      continue;
    }

    if (flipDelta > 0) {
      // Need more positives: flip negatives → positive, prioritizing rows where model predicts positive
      const candidates = groupRows
        .filter(({ r }) => String(r[targetColumn]) !== positiveClass)
        .sort((a, b) => predictions[b.i] - predictions[a.i]); // model-confident positives first

      let flipped = 0;
      for (const { i } of candidates) {
        if (flipped >= flipDelta) break;
        debiasedData[i][targetColumn] = positiveClass;
        flipped++;
        totalFlips++;
      }
      correctionLog.push(
        `Group "${group}": flipped ${flipped} neg→pos (PPI target rate: ${(targetRate * 100).toFixed(1)}%, classical: ${(currentPosCount / groupRows.length * 100).toFixed(1)}%)`
      );
    } else {
      // Need fewer positives: flip positives → negative, prioritizing rows where model predicts negative
      const candidates = groupRows
        .filter(({ r }) => String(r[targetColumn]) === positiveClass)
        .sort((a, b) => predictions[a.i] - predictions[b.i]); // model-confident negatives first

      let flipped = 0;
      for (const { i } of candidates) {
        if (flipped >= -flipDelta) break;
        debiasedData[i][targetColumn] = negativeClass;
        flipped++;
        totalFlips++;
      }
      correctionLog.push(
        `Group "${group}": flipped ${flipped} pos→neg (PPI target rate: ${(targetRate * 100).toFixed(1)}%, classical: ${(currentPosCount / groupRows.length * 100).toFixed(1)}%)`
      );
    }
  }

  return { debiasedData, correctionLog, totalFlips };
}

/**
 * Find the optimal lambda via cross-validation on labeled data.
 * Lambda controls how much we weight the prediction term vs labels.
 * Returns the lambda in [0, 1] that minimizes MSE on a validation fold.
 */
export function findOptimalLambda(
  labeledData: Record<string, any>[],
  labelsY: number[],
  predsOnLabeled: number[],
  sensitiveColumns: string[],
): number {
  if (labelsY.length < 10) return 0.5; // default for tiny datasets

  const lambdas = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  let bestLambda = 0.5;
  let bestMSE = Infinity;

  const foldSize = Math.floor(labelsY.length * 0.2);
  if (foldSize < 2) return 0.5;

  for (const lam of lambdas) {
    let totalMSE = 0;
    // Simple hold-out validation
    const valY    = labelsY.slice(-foldSize);
    const valPred = predsOnLabeled.slice(-foldSize);
    const trainY  = labelsY.slice(0, -foldSize);
    const trainPred = predsOnLabeled.slice(0, -foldSize);

    // Compute rectifier on training fold
    const rect = trainY.length > 0
      ? trainY.reduce((acc, y, i) => acc + (y - trainPred[i]), 0) / trainY.length
      : 0;

    // PPI estimate for each validation row
    for (let i = 0; i < valY.length; i++) {
      const ppiEst = lam * valPred[i] + rect;
      totalMSE += (valY[i] - ppiEst) ** 2;
    }
    const mse = totalMSE / valY.length;
    if (mse < bestMSE) { bestMSE = mse; bestLambda = lam; }
  }

  return bestLambda;
}
