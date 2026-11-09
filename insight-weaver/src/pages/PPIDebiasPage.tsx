import { useData } from "@/context/DataContext";
import { PageHeader } from "@/components/PageHeader";
import {
  FlaskConical, Zap, Info, RefreshCw, TrendingUp,
  ShieldCheck, AlertTriangle, Binary, Download, Layers, Target, Database,
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useMemo, useCallback, useRef } from "react";
import { useNavigate } from "react-router-dom";
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
  CartesianGrid, ReferenceLine,
} from "recharts";
import { PageFooter } from "@/components/PageFooter";
import { toast } from "sonner";

type ModelId = "rf" | "xgb" | "knn" | "logistic" | "svm" | "nn";

interface PPIGroupStats {
  group: string;
  n_labeled: number;
  n_total: number;
  classicalRate: number;
  ppiRate: number;
  ppiCILow: number;
  ppiCIHigh: number;
  rectifier: number;
  valid: boolean;
}

interface PPIResult {
  groupStats: PPIGroupStats[];
  ppiSPD: number;
  classicalSPD: number;
  ppiDI: number;
  privilegedGroup: string;
  unprivilegedGroup: string;
  lambda: number;
  debiasedData: Record<string, any>[];
  correctionLog: string[];
  totalFlips: number;
  positiveClass: string;
  negativeClass: string;
}

const MODEL_OPTIONS: { id: ModelId; name: string; color: string }[] = [
  { id: "rf",       name: "Random Forest",       color: "hsl(260,70%,65%)" },
  { id: "xgb",      name: "XGBoost",             color: "hsl(45,90%,55%)"  },
  { id: "knn",      name: "K-Nearest Neighbors", color: "hsl(190,90%,50%)" },
  { id: "logistic", name: "Logistic Regression", color: "hsl(210,80%,55%)" },
  { id: "svm",      name: "SVM",                 color: "hsl(340,75%,55%)" },
  { id: "nn",       name: "Neural Network",      color: "hsl(280,80%,65%)" },
];

export default function PPIDebiasPage() {
  const {
    dataset, setDebiasedDataset, sensitiveColumns, setSensitiveColumns,
    targetColumn, setTargetColumn, ppiResult, setPPIResult,
    ppiDebiasedDataset, setPPIDebiasedDataset,
  } = useData();
  const navigate = useNavigate();

  const [selectedModel, setSelectedModel] = useState<ModelId>("rf");
  const [lambda, setLambda] = useState(0.8);
  const [strength, setStrength] = useState(1.0);
  const [runState, setRunState] = useState<"idle" | "running" | "done">("idle");
  const [progress, setProgress] = useState(0);
  const [statusMsg, setStatusMsg] = useState("");
  const [localResult, setLocalResult] = useState<PPIResult | null>(ppiResult as any);
  const workerRef = useRef<Worker | null>(null);

  const handleRunPPI = useCallback(() => {
    if (!dataset || !targetColumn || sensitiveColumns.length === 0) {
      toast.error("Select Target Column and at least one Sensitive Attribute first.");
      return;
    }

    if (workerRef.current) workerRef.current.terminate();
    workerRef.current = new Worker(
      new URL("../workers/mlWorker.ts", import.meta.url),
      { type: "module" }
    );

    setRunState("running");
    setProgress(0);
    setLocalResult(null);
    setStatusMsg("Initializing PPI pipeline...");

    workerRef.current.onmessage = (e: MessageEvent) => {
      const { type, payload, value, status, message } = e.data;

      if (type === "PPI_PROGRESS") {
        setProgress(value);
        setStatusMsg(status);
      }

      if (type === "PPI_DEBIAS_DONE") {
        setLocalResult(payload);
        setPPIResult(payload);

        const debiasedInfo = {
          ...dataset,
          data: payload.debiasedData,
          fileName: `${dataset.fileName.split(".")[0]}_ppi_debiased.csv`,
        };
        setPPIDebiasedDataset(debiasedInfo);
        setDebiasedDataset(debiasedInfo);

        setRunState("done");
        setProgress(100);
        toast.success(`PPI Debiasing complete! ${payload.totalFlips} label corrections applied.`);
      }

      if (type === "PPI_ERROR") {
        toast.error("PPI failed: " + message);
        setRunState("idle");
      }
    };

    workerRef.current.onerror = (e) => {
      toast.error("Worker error: " + e.message);
      setRunState("idle");
    };

    workerRef.current.postMessage({
      type: "PPI_DEBIAS",
      payload: {
        rawData: dataset.data,
        targetColumn,
        sensitiveColumns,
        selectedModel,
        hyperParams: { n_estimators: 50, max_depth: 10, n_neighbors: 5, C: 1 },
        lambda,
        strength,
      },
    });
  }, [dataset, targetColumn, sensitiveColumns, selectedModel, lambda, strength,
      setPPIResult, setPPIDebiasedDataset, setDebiasedDataset]);

  const handleExport = () => {
    if (!ppiDebiasedDataset) return;
    import("papaparse").then(({ default: Papa }) => {
      const csv = Papa.unparse(ppiDebiasedDataset.data);
      const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = ppiDebiasedDataset.fileName;
      link.click();
      URL.revokeObjectURL(url);
      toast.success("PPI-debiased dataset exported!");
    });
  };

  const rateChartData = useMemo(() => {
    if (!localResult?.groupStats) return [];
    return localResult.groupStats
      .filter(g => g.valid || g.n_total >= 5)
      .map(g => ({
        group: g.group.length > 20 ? g.group.slice(0, 18) + "…" : g.group,
        fullGroup: g.group,
        classical: parseFloat((g.classicalRate * 100).toFixed(2)),
        ppi: parseFloat((g.ppiRate * 100).toFixed(2)),
        ciLow: parseFloat((g.ppiCILow * 100).toFixed(2)),
        ciHigh: parseFloat((g.ppiCIHigh * 100).toFixed(2)),
        rectifier: parseFloat((g.rectifier * 100).toFixed(2)),
        n_total: g.n_total,
        n_labeled: g.n_labeled,
      }));
  }, [localResult]);

  const spdImprovement = localResult
    ? Math.abs(localResult.classicalSPD) - Math.abs(localResult.ppiSPD)
    : 0;

  if (!dataset) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-6 text-center">
        <Database className="h-16 w-16 text-primary/10 animate-pulse" />
        <p className="text-muted-foreground font-display uppercase tracking-widest opacity-50">No Dataset Loaded</p>
        <button
          onClick={() => navigate("/")}
          className="px-10 py-4 rounded-full bg-primary text-white text-xs font-black"
        >
          Upload Dataset
        </button>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-10 pb-24">
      <PageHeader
        title="PPI Debiasing Lab"
        description="Prediction-Powered Inference (Angelopoulos et al., 2023) · Statistically-Valid Label Correction"
        icon={<FlaskConical className="h-5 w-5 text-primary" />}
      />

      {/* Theory Banner */}
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        className="glass-card p-6 border-primary/30 bg-primary/5 flex gap-5"
      >
        <div className="p-3 rounded-xl bg-primary/15 text-primary shrink-0 self-start">
          <Info className="h-5 w-5" />
        </div>
        <div>
          <h3 className="text-sm font-black uppercase tracking-widest text-primary mb-1">
            What is Prediction-Powered Inference?
          </h3>
          <p className="text-xs text-foreground/80 leading-relaxed max-w-4xl">
            PPI uses an ML model's predictions on the <strong>full dataset</strong> (unlabeled pool) to augment statistical inference from your smaller <strong>labeled split</strong>.
            It computes a <strong>bias rectifier</strong> term — the systematic error between model predictions and ground-truth labels — and applies it to produce{" "}
            <strong>corrected group positive rates</strong> with valid 95% confidence intervals. The debiased dataset is generated by flipping labels (prioritizing model-confident predictions)
            until each group's positive rate matches its PPI estimate. This approach is provably valid: unlike heuristic reweighting, PPI yields statistically calibrated corrections.
          </p>
          <div className="mt-3 flex flex-wrap gap-3">
            {[
              { label: "Estimator", formula: "θ̂_PPI = λ·mean(Ŷ_all) + rectifier" },
              { label: "Rectifier", formula: "Δ = mean(Y_labeled − Ŷ_labeled)" },
              { label: "95% CI",    formula: "θ̂_PPI ± 1.96·√(λ²·Var(Ŷ)/N + Var(Δ)/n)" },
            ].map(({ label, formula }) => (
              <div key={label} className="px-4 py-2 rounded-xl bg-card border border-border font-mono text-[10px]">
                <span className="text-muted-foreground uppercase font-black mr-2">{label}:</span>
                <span className="text-primary">{formula}</span>
              </div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Configuration */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Left: Controls */}
        <div className="lg:col-span-2 space-y-5">
          <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-7">
            <h3 className="font-display font-bold text-sm uppercase tracking-widest mb-5 flex items-center gap-2 text-foreground/80">
              <Target className="h-4 w-4 text-primary" /> Column Configuration
            </h3>
            <div className="grid md:grid-cols-2 gap-5">
              <div>
                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">
                  Target / Outcome Column
                </label>
                <select
                  value={targetColumn ?? ""}
                  onChange={e => setTargetColumn(e.target.value)}
                  className="w-full px-4 py-3 rounded-xl bg-card border border-border text-sm outline-none focus:border-primary transition-colors"
                >
                  <option value="">Select outcome column...</option>
                  {dataset.columnStats.map(c => (
                    <option key={c.name} value={c.name}>{c.name}</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">
                  Protected Attributes (Sensitive)
                </label>
                <select
                  value=""
                  onChange={e => {
                    const val = e.target.value;
                    if (val && !sensitiveColumns.includes(val))
                      setSensitiveColumns([...sensitiveColumns, val]);
                  }}
                  className="w-full px-4 py-3 rounded-xl bg-card border border-border text-sm outline-none focus:border-primary transition-colors"
                >
                  <option value="">Add protected attribute...</option>
                  {dataset.columnStats
                    .filter(c => c.name !== targetColumn && !sensitiveColumns.includes(c.name))
                    .map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
                {sensitiveColumns.length > 0 && (
                  <div className="flex flex-wrap gap-2 mt-2">
                    {sensitiveColumns.map(col => (
                      <span
                        key={col}
                        className="inline-flex items-center gap-1.5 px-3 py-1 rounded-xl bg-primary/10 border border-primary/20 text-xs font-black text-primary"
                      >
                        {col}
                        <button
                          onClick={() => setSensitiveColumns(sensitiveColumns.filter(c => c !== col))}
                          className="hover:text-rose-500 font-bold outline-none"
                        >✕</button>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.05 }}
            className="glass-card p-7"
          >
            <h3 className="font-display font-bold text-sm uppercase tracking-widest mb-5 flex items-center gap-2 text-foreground/80">
              <Layers className="h-4 w-4 text-accent" /> PPI Hyperparameters
            </h3>
            <div className="grid md:grid-cols-3 gap-6">
              <div>
                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">
                  Annotator Model
                </label>
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value as ModelId)}
                  className="w-full px-4 py-3 rounded-xl bg-card border border-border text-sm outline-none focus:border-primary transition-colors"
                >
                  {MODEL_OPTIONS.map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
                <p className="text-[10px] text-muted-foreground mt-1.5 ml-1">
                  Used as the proxy annotator for unlabeled examples.
                </p>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Lambda (λ)</label>
                  <span className="text-[10px] font-mono text-primary bg-primary/10 px-2 py-0.5 rounded-md border border-primary/20">
                    {lambda.toFixed(1)}
                  </span>
                </div>
                <input
                  type="range" min={0} max={1} step={0.1}
                  value={lambda}
                  onChange={e => setLambda(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <div className="flex justify-between text-[9px] text-muted-foreground mt-1">
                  <span>0 = Pure Labels</span>
                  <span>1 = Full PPI</span>
                </div>
              </div>

              <div>
                <div className="flex justify-between items-center mb-2">
                  <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground">Correction Strength</label>
                  <span className="text-[10px] font-mono text-primary bg-primary/10 px-2 py-0.5 rounded-md border border-primary/20">
                    {(strength * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range" min={0} max={1} step={0.05}
                  value={strength}
                  onChange={e => setStrength(parseFloat(e.target.value))}
                  className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                />
                <div className="flex justify-between text-[9px] text-muted-foreground mt-1">
                  <span>0 = No Change</span>
                  <span>100% = Full PPI Correction</span>
                </div>
              </div>
            </div>
          </motion.div>
        </div>

        {/* Right: Run Panel */}
        <motion.div
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          className="glass-card p-7 flex flex-col justify-between gap-6 border-primary/20"
        >
          <div className="space-y-5">
            <h3 className="font-display font-bold text-sm uppercase tracking-widest text-foreground/80">
              PPI Pipeline
            </h3>
            <div className="space-y-3">
              {[
                { step: "1", label: "Train annotator model on 80% split",          done: runState !== "idle" },
                { step: "2", label: "Predict on full dataset (unlabeled pool)",     done: runState !== "idle" && progress > 30 },
                { step: "3", label: "Compute rectifier (label vs pred error)",      done: runState !== "idle" && progress > 55 },
                { step: "4", label: "PPI estimates + 95% confidence intervals",     done: runState !== "idle" && progress > 72 },
                { step: "5", label: "Generate corrected dataset",                   done: runState === "done" },
              ].map(({ step, label, done }) => (
                <div key={step} className={`flex items-center gap-3 p-3 rounded-xl transition-all ${done ? "bg-primary/10 border border-primary/20" : "bg-card border border-border opacity-50"}`}>
                  <div className={`h-6 w-6 rounded-full flex items-center justify-center text-[10px] font-black shrink-0 ${done ? "bg-primary text-white" : "bg-muted text-muted-foreground"}`}>
                    {done ? "✓" : step}
                  </div>
                  <span className={`text-[11px] font-bold ${done ? "text-primary" : "text-muted-foreground"}`}>{label}</span>
                </div>
              ))}
            </div>

            {runState === "running" && (
              <div className="space-y-2">
                <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                  <motion.div
                    className="h-full rounded-full bg-gradient-to-r from-primary to-accent"
                    animate={{ width: `${progress}%` }}
                    transition={{ duration: 0.4, ease: "easeOut" }}
                  />
                </div>
                <p className="text-[10px] text-muted-foreground font-mono text-center animate-pulse">{statusMsg}</p>
              </div>
            )}
          </div>

          <div className="space-y-3">
            <button
              onClick={handleRunPPI}
              disabled={runState === "running" || !targetColumn || sensitiveColumns.length === 0}
              className="w-full py-5 rounded-2xl font-display font-black text-sm flex items-center justify-center gap-3 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed bg-gradient-to-r from-primary to-accent text-white hover:scale-[1.02] shadow-glow"
            >
              {runState === "running"
                ? <><RefreshCw className="h-5 w-5 animate-spin" /> Running PPI...</>
                : <><Zap className="h-5 w-5" /> Run PPI Debiasing</>
              }
            </button>
            {runState === "done" && (
              <button
                onClick={handleExport}
                className="w-full py-3 rounded-xl text-xs font-black uppercase tracking-widest border border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/10 transition-all flex items-center justify-center gap-2"
              >
                <Download className="h-3.5 w-3.5" /> Export PPI Dataset
              </button>
            )}
          </div>
        </motion.div>
      </div>

      {/* Results */}
      <AnimatePresence>
        {localResult && runState === "done" && (
          <motion.div
            initial={{ opacity: 0, y: 32 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-8"
          >
            {/* KPI Row */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-5">
              {[
                {
                  label: "PPI SPD",
                  value: `${(Math.abs(localResult.ppiSPD) * 100).toFixed(2)}%`,
                  sub: `Classical: ${(Math.abs(localResult.classicalSPD) * 100).toFixed(2)}%`,
                  color: Math.abs(localResult.ppiSPD) < 0.15 ? "text-emerald-400" : "text-rose-400",
                  icon: <Binary className="h-4 w-4" />,
                },
                {
                  label: "PPI DI Ratio",
                  value: localResult.ppiDI.toFixed(3),
                  sub: (localResult.ppiDI >= 0.8 && localResult.ppiDI <= 1.25) ? "Within 80% rule ✓" : "Outside 80% rule ✗",
                  color: (localResult.ppiDI >= 0.8 && localResult.ppiDI <= 1.25) ? "text-emerald-400" : "text-rose-400",
                  icon: <TrendingUp className="h-4 w-4" />,
                },
                {
                  label: "SPD Reduction",
                  value: `${spdImprovement >= 0 ? "-" : "+"}${(Math.abs(spdImprovement) * 100).toFixed(2)}%`,
                  sub: spdImprovement > 0 ? "Bias reduced ✓" : "Bias increased",
                  color: spdImprovement > 0 ? "text-emerald-400" : "text-rose-400",
                  icon: spdImprovement > 0 ? <ShieldCheck className="h-4 w-4" /> : <AlertTriangle className="h-4 w-4" />,
                },
                {
                  label: "Label Corrections",
                  value: localResult.totalFlips.toString(),
                  sub: `λ = ${localResult.lambda.toFixed(1)} · Strength ${(strength * 100).toFixed(0)}%`,
                  color: "text-primary",
                  icon: <Zap className="h-4 w-4" />,
                },
              ].map(({ label, value, sub, color, icon }) => (
                <div key={label} className="glass-card p-5 flex flex-col gap-3">
                  <div className={`flex items-center gap-2 ${color} opacity-80`}>{icon}</div>
                  <div>
                    <p className="text-[9px] font-black uppercase tracking-widest text-muted-foreground">{label}</p>
                    <p className={`text-2xl font-black font-display mt-0.5 ${color}`}>{value}</p>
                    <p className="text-[10px] text-muted-foreground mt-1">{sub}</p>
                  </div>
                </div>
              ))}
            </div>

            {/* Group Rate Chart */}
            <div className="glass-card p-7">
              <h3 className="font-display font-bold text-sm uppercase tracking-widest mb-6 text-foreground/80">
                Classical vs PPI-Corrected Positive Rates per Group
              </h3>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={rateChartData} barCategoryGap="30%">
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(var(--border))" />
                  <XAxis dataKey="group" tick={{ fontSize: 10 }} />
                  <YAxis tickFormatter={v => `${v}%`} tick={{ fontSize: 10 }} domain={[0, 100]} />
                  <Tooltip
                    formatter={(v: any, name: string) => [`${Number(v).toFixed(1)}%`, name]}
                    contentStyle={{ background: "hsl(var(--card))", border: "1px solid hsl(var(--border))", borderRadius: 12, fontSize: 11 }}
                  />
                  <ReferenceLine y={50} stroke="hsl(var(--muted-foreground))" strokeDasharray="4 4" opacity={0.4} />
                  <Bar dataKey="classical" name="Classical Rate" fill="hsl(var(--muted-foreground))" opacity={0.5} radius={[4, 4, 0, 0]} />
                  <Bar dataKey="ppi" name="PPI Rate" radius={[4, 4, 0, 0]}>
                    {rateChartData.map((entry, i) => (
                      <Cell key={i} fill={`hsl(${260 - i * 30},70%,65%)`} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Group Stats Table */}
            <div className="glass-card p-7 overflow-x-auto">
              <h3 className="font-display font-bold text-sm uppercase tracking-widest mb-5 text-foreground/80">
                PPI Group Statistics
              </h3>
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left border-b border-border">
                    {["Group", "N Total", "N Labeled", "Classical Rate", "PPI Rate", "95% CI Low", "95% CI High", "Rectifier", "Valid"].map(h => (
                      <th key={h} className="pb-3 pr-5 text-[10px] font-black uppercase tracking-widest text-muted-foreground whitespace-nowrap">{h}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {localResult.groupStats.map((g, i) => (
                    <tr key={i} className="border-b border-border/40 hover:bg-primary/5 transition-colors">
                      <td className="py-2.5 pr-5 font-mono text-[11px] text-foreground/90 max-w-[200px] truncate">{g.group}</td>
                      <td className="py-2.5 pr-5 text-muted-foreground">{g.n_total}</td>
                      <td className="py-2.5 pr-5 text-muted-foreground">{g.n_labeled}</td>
                      <td className="py-2.5 pr-5">{(g.classicalRate * 100).toFixed(1)}%</td>
                      <td className="py-2.5 pr-5 font-bold text-primary">{(g.ppiRate * 100).toFixed(1)}%</td>
                      <td className="py-2.5 pr-5 text-emerald-400/80">{(g.ppiCILow * 100).toFixed(1)}%</td>
                      <td className="py-2.5 pr-5 text-emerald-400/80">{(g.ppiCIHigh * 100).toFixed(1)}%</td>
                      <td className={`py-2.5 pr-5 font-mono ${g.rectifier >= 0 ? "text-emerald-400" : "text-rose-400"}`}>
                        {g.rectifier >= 0 ? "+" : ""}{(g.rectifier * 100).toFixed(2)}%
                      </td>
                      <td className="py-2.5">
                        <span className={`px-2 py-0.5 rounded-full text-[9px] font-black ${g.valid ? "bg-emerald-500/15 text-emerald-400" : "bg-rose-500/15 text-rose-400"}`}>
                          {g.valid ? "✓ Valid" : "⚠ Small N"}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Correction Log */}
            <div className="glass-card p-7">
              <h3 className="font-display font-bold text-sm uppercase tracking-widest mb-5 text-foreground/80">
                Correction Log — {localResult.totalFlips} Label Flips
              </h3>
              <div className="space-y-2 max-h-64 overflow-y-auto pr-2">
                {localResult.correctionLog.length === 0 && (
                  <p className="text-xs text-muted-foreground italic">No corrections were needed — all groups already match PPI targets.</p>
                )}
                {localResult.correctionLog.map((log, i) => (
                  <div key={i} className="flex items-start gap-3 p-3 rounded-xl bg-card border border-border">
                    <span className="text-[10px] font-black text-primary/60 shrink-0 mt-0.5">#{i + 1}</span>
                    <span className="text-[11px] font-mono text-foreground/80">{log}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Privileged / Unprivileged Group Summary */}
            <div className="grid md:grid-cols-2 gap-5">
              <div className="glass-card p-6 border-emerald-500/20">
                <p className="text-[9px] font-black uppercase tracking-widest text-muted-foreground mb-1">Privileged Group (Highest PPI Rate)</p>
                <p className="text-sm font-black text-emerald-400 font-mono break-all">{localResult.privilegedGroup}</p>
              </div>
              <div className="glass-card p-6 border-rose-500/20">
                <p className="text-[9px] font-black uppercase tracking-widest text-muted-foreground mb-1">Unprivileged Group (Lowest PPI Rate)</p>
                <p className="text-sm font-black text-rose-400 font-mono break-all">{localResult.unprivilegedGroup}</p>
              </div>
            </div>

            {/* Navigate to Results */}
            <div className="flex justify-end">
              <button
                onClick={() => navigate("/results")}
                className="px-8 py-4 rounded-2xl font-display font-black text-sm bg-gradient-to-r from-primary to-accent text-white hover:scale-[1.02] transition-all shadow-glow flex items-center gap-2"
              >
                View Full Results →
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <PageFooter />
    </div>
  );
}
