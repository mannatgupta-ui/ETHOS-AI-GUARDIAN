export interface FairnessMetrics {
  spd: number;
  di: number;
  wasserstein: number;
  proxies: { name: string; correlation: number }[];
  privileged: string;
  unprivileged: string;
}

/**
 * Calculates Statistical Parity Difference and other fairness metrics.
 */
export function calculateFairnessMetrics(
  data: any[],
  sensitiveAttr: string,
  targetCol: string
): FairnessMetrics {
  if (!data || data.length === 0 || !sensitiveAttr || !targetCol) {
    return { spd: 0, di: 1, wasserstein: 0, proxies: [], privileged: "N/A", unprivileged: "N/A" };
  }

  const groupValues = Array.from(new Set(data.map(r => String(r[sensitiveAttr])))).filter(v => v !== "null" && v !== "undefined");
  
  if (groupValues.length < 2) {
    return { spd: 0, di: 1, wasserstein: 0, proxies: [], privileged: groupValues[0] || "N/A", unprivileged: "N/A" };
  }

  const groupStats = groupValues.map(val => {
    const groupRows = data.filter(r => String(r[sensitiveAttr]) === val);
    const successRows = groupRows.filter(r => {
      const v = String(r[targetCol]).toLowerCase();
      return v === "1" || v === "1.0" || v === "yes" || v === "hired" || v === "true" || v === "positive";
    });
    return {
      value: val,
      count: groupRows.length,
      rate: groupRows.length > 0 ? successRows.length / groupRows.length : 0
    };
  }).sort((a, b) => b.rate - a.rate);

  const privileged = groupStats[0];
  const unprivileged = groupStats[1];

  const spd = unprivileged.rate - privileged.rate;
  const di = privileged.rate > 0 ? unprivileged.rate / privileged.rate : 1;

  // Simple Wasserstein-1 approximation
  const wasserstein = Math.abs(spd) * 1.25; 

  // Detect Proxies
  const proxies: { name: string; correlation: number }[] = [];
  const columns = Object.keys(data[0]).filter(c => c !== sensitiveAttr && c !== targetCol);

  columns.forEach(col => {
    const privVals = data.filter(r => String(r[sensitiveAttr]) === privileged.value).map(r => r[col]);
    const unprivVals = data.filter(r => String(r[sensitiveAttr]) === unprivileged.value).map(r => r[col]);
    
    let diff = 0;
    if (typeof privVals[0] === 'number') {
      const pMean = privVals.reduce((a, b) => a + b, 0) / Math.max(1, privVals.length);
      const uMean = unprivVals.reduce((a, b) => a + b, 0) / Math.max(1, unprivVals.length);
      const max = Math.max(Math.abs(pMean), Math.abs(uMean), 1);
      diff = Math.abs(pMean - uMean) / max;
    } else {
      diff = Math.abs(countFreq(privVals) - countFreq(unprivVals));
    }

    if (diff > 0.1) {
      proxies.push({ name: col, correlation: diff });
    }
  });

  return {
    spd,
    di,
    wasserstein,
    proxies: proxies.sort((a, b) => b.correlation - a.correlation).slice(0, 3),
    privileged: privileged.value,
    unprivileged: unprivileged.value
  };
}

/**
 * Calculates Mutual Information (MI) between a feature and the target.
 * Using a simple entropy-based approximation for discrete variables.
 */
export function calculateMutualInfo(data: any[], feature: string, target: string): number {
  if (!data.length) return 0;
  
  const n = data.length;
  const jointFreq: Record<string, number> = {};
  const featureFreq: Record<string, number> = {};
  const targetFreq: Record<string, number> = {};

  data.forEach(row => {
    const f = String(row[feature]);
    const t = String(row[target]);
    const joint = `${f}|${t}`;

    jointFreq[joint] = (jointFreq[joint] || 0) + 1;
    featureFreq[f] = (featureFreq[f] || 0) + 1;
    targetFreq[t] = (targetFreq[t] || 0) + 1;
  });

  let mi = 0;
  Object.keys(jointFreq).forEach(joint => {
    const [f, t] = joint.split('|');
    const p_jt = jointFreq[joint] / n;
    const p_f = featureFreq[f] / n;
    const p_t = targetFreq[t] / n;
    
    if (p_jt > 0 && p_f > 0 && p_t > 0) {
      mi += p_jt * Math.log2(p_jt / (p_f * p_t));
    }
  });

  return Math.max(0, mi);
}

function countFreq(arr: any[]) {
  if (arr.length === 0) return 0;
  const counts: Record<string, number> = {};
  arr.forEach(x => counts[String(x)] = (counts[String(x)] || 0) + 1);
  const top = Object.values(counts).sort((a, b) => b - a)[0];
  return top / arr.length;
}
