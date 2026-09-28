import { useData } from "@/context/DataContext";
import { PageHeader } from "@/components/PageHeader";
import {
  Brain, Play, Zap, ArrowRight, Activity, Settings2, Cpu, RefreshCw,
  ShieldCheck, Database, AlertTriangle, Rocket, CheckCircle2, TrendingUp,
  Target, Gauge, Clock, Award, ChevronRight
} from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useMemo, useEffect, useRef, useCallback } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell, RadarChart, Radar, PolarGrid, PolarAngleAxis } from "recharts";
import { useNavigate } from "react-router-dom";
import { PageFooter } from "@/components/PageFooter";
import { toast } from "sonner";
import { calculateFairnessMetrics } from "@/lib/metrics";
import { getPipelineInsights } from "@/lib/gemini";

type ModelId = "rf" | "linear" | "logistic" | "svm" | "knn" | "nn";
type DataSource = "raw" | "debiased";

interface ModelConfig {
  id: ModelId;
  name: string;
  type: "classification" | "regression" | "both";
  color: string;
  gradient: string;
  description: string;
  params: Record<string, { label: string; type: "number" | "select"; min?: number; max?: number; step?: number; options?: string[]; default: any }>;
}

const MODEL_DEFINITIONS: ModelConfig[] = [
  {
    id: "rf", name: "Random Forest", type: "both", color: "hsl(260,70%,65%)", gradient: "from-purple-500 to-indigo-600",
    description: "Ensemble of decision trees, robust to overfitting",
    params: {
      n_estimators: { label: "Estimators", type: "number", min: 10, max: 200, step: 10, default: 50 },
      max_depth: { label: "Max Depth", type: "number", min: 1, max: 20, step: 1, default: 10 },
      min_samples_split: { label: "Min Split", type: "number", min: 2, max: 20, step: 1, default: 2 }
    }
  },
  {
    id: "knn", name: "K-Nearest Neighbors", type: "both", color: "hsl(190,90%,50%)", gradient: "from-cyan-500 to-blue-600",
    description: "Instance-based learning via distance metrics",
    params: {
      n_neighbors: { label: "Neighbors (K)", type: "number", min: 1, max: 25, step: 1, default: 5 },
      weights: { label: "Weights", type: "select", options: ["uniform", "distance"], default: "uniform" }
    }
  },
  {
    id: "logistic", name: "Logistic Regression", type: "classification", color: "hsl(210,80%,55%)", gradient: "from-blue-500 to-indigo-600",
    description: "Linear classifier with sigmoid activation",
    params: {
      C: { label: "Inverse Reg (C)", type: "number", min: 0.1, max: 10, step: 0.1, default: 1.0 },
      penalty: { label: "Penalty", type: "select", options: ["l2", "none"], default: "l2" }
    }
  },
  {
    id: "svm", name: "SVM", type: "both", color: "hsl(340,75%,55%)", gradient: "from-rose-500 to-pink-600",
    description: "Support vector machine with kernel trick",
    params: {
      C: { label: "Regularization (C)", type: "number", min: 0.1, max: 10, step: 0.1, default: 1.0 },
      kernel: { label: "Kernel", type: "select", options: ["rbf", "linear", "poly"], default: "rbf" }
    }
  },
  {
    id: "nn", name: "Neural Network", type: "both", color: "hsl(280,80%,65%)", gradient: "from-violet-500 to-purple-600",
    description: "Multi-layer perceptron with backpropagation",
    params: {
      hidden_layers: { label: "Hidden Layers", type: "number", min: 1, max: 4, step: 1, default: 2 },
      epochs: { label: "Epochs", type: "number", min: 10, max: 100, step: 10, default: 50 },
      activation: { label: "Activation", type: "select", options: ["relu", "tanh", "sigmoid"], default: "relu" }
    }
  },
  {
    id: "linear", name: "Linear Regression", type: "regression", color: "hsl(30,80%,55%)", gradient: "from-orange-500 to-red-600",
    description: "Classic linear model for continuous targets",
    params: {
      fit_intercept: { label: "Fit Intercept", type: "select", options: ["True", "False"], default: "True" }
    }
  },
];

interface TrainingMetrics {
  accuracy: number; precision: number; recall: number; f1: number;
  latency: number; majorityBaseline: number;
}

interface ResultEntry {
  name: string; score: number; color: string;
  metrics?: Record<string, number>;
}

const MetricBar = ({ label, value, color = "var(--primary)", delay = 0 }: {
  label: string; value: number; color?: string; delay?: number;
}) => {
  // SPD (Bias) is a 0-1 fraction; show as % but cap bar and color appropriately
  const isBiasMetric = label === "SPD (Bias)";
  const displayPct = Math.min(Math.abs(value) * 100, 100);
  const barColor = isBiasMetric
    ? (displayPct > 10 ? "#f43f5e" : "#10b981")
    : color;
  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-[10px] font-bold">
        <span className="text-muted-foreground uppercase tracking-wider">{label}</span>
        <span style={{ color: barColor }}>
          {isBiasMetric ? `${displayPct.toFixed(1)}%` : `${(value * 100).toFixed(1)}%`}
        </span>
      </div>
      <div className="h-1.5 bg-muted rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={{ width: `${displayPct}%` }}
          transition={{ delay, duration: 0.8, ease: "easeOut" }}
          className="h-full rounded-full"
          style={{ background: barColor }}
        />
      </div>
    </div>
  );
};

export default function ModelPage() {
  const {
    dataset, debiasedDataset, targetColumn, setTargetColumn,
    sensitiveColumns, modelResults, setModelResults, datasetDescription,
    setBoostedMetrics
  } = useData();
  const navigate = useNavigate();

  const [taskType, setTaskType] = useState<"classification" | "regression">("classification");
  const [dataSource, setDataSource] = useState<DataSource>("raw");
  const [selectedModel, setSelectedModel] = useState<ModelId>("rf");
  const [hyperParams, setHyperParams] = useState<Record<string, any>>({});

  const [trainingState, setTrainingState] = useState<"idle" | "training" | "completed">(modelResults ? "completed" : "idle");
  const [progress, setProgress] = useState(modelResults ? 100 : 0);
  const [statusMsg, setStatusMsg] = useState("Initializing...");
  const [results, setResults] = useState<ResultEntry[] | null>(modelResults);
  const [rawMetricsFull, setRawMetricsFull] = useState<TrainingMetrics | null>(null);

  const [aiInsight, setAiInsight] = useState<string | null>(null);
  const [aiLoading, setAiLoading] = useState(false);

  const [boostState, setBoostState] = useState<"idle" | "boosting" | "completed">("idle");
  const [boostProgress, setBoostProgress] = useState(0);
  const [boostStatusMsg, setBoostStatusMsg] = useState("");
  const [boostedResults, setBoostedResults] = useState<{
    before: number;
    after: number;
    beforePrecision: number;
    afterPrecision: number;
    beforeRecall: number;
    afterRecall: number;
    beforeF1: number;
    afterF1: number;
    beforeBias: number;
    afterBias: number;
    logs: string[];
  } | null>(null);

  const workerRef = useRef<Worker | null>(null);

  // ── Setup worker ────────────────────────────────────────────────────────────
  const initWorker = useCallback(() => {
    if (workerRef.current) workerRef.current.terminate();
    workerRef.current = new Worker(new URL("../workers/mlWorker.ts", import.meta.url), { type: "module" });
    return workerRef.current;
  }, []);

  useEffect(() => () => { workerRef.current?.terminate(); }, []);

  // ── Auto-detect task type ───────────────────────────────────────────────────
  const aiDiagnosis = useMemo(() => {
    if (!dataset || !targetColumn) return null;
    const colStats = dataset.columnStats.find(c => c.name === targetColumn);
    if (!colStats) return { type: "classification" };
    const isClass = colStats.type !== "numeric" || colStats.unique < 15;
    return { type: isClass ? "classification" : "regression" };
  }, [dataset, targetColumn]);

  useEffect(() => {
    if (aiDiagnosis) setTaskType(aiDiagnosis.type as any);
  }, [aiDiagnosis]);

  // ── Load default hyperparams ────────────────────────────────────────────────
  useEffect(() => {
    const model = MODEL_DEFINITIONS.find(m => m.id === selectedModel);
    if (model) {
      const defaults: Record<string, any> = {};
      Object.entries(model.params).forEach(([key, val]) => { defaults[key] = val.default; });
      setHyperParams(defaults);
    }
  }, [selectedModel]);

  // ── Train ───────────────────────────────────────────────────────────────────
  const handleTrain = () => {
    if (!dataset || !targetColumn) return;
    setTrainingState("training");
    setProgress(0);
    setStatusMsg("Initializing pipeline...");
    setResults(null);
    setBoostedResults(null);
    setBoostState("idle");
    setAiInsight(null);

    const worker = initWorker();

    worker.onmessage = (e: MessageEvent) => {
      const { type, payload, value, status, message } = e.data;

      if (type === "PROGRESS") {
        setProgress(value);
        setStatusMsg(status);
      }

      if (type === "TRAIN_DONE") {
        const { rawMetrics, debMetrics }: { rawMetrics: TrainingMetrics & { predictionBiasSpd: number }; debMetrics: (TrainingMetrics & { predictionBiasSpd: number }) | null } = payload;
        const modelDef = MODEL_DEFINITIONS.find(m => m.id === selectedModel)!;
        const labelSpd = Math.abs(calculateFairnessMetrics(dataset.data, sensitiveColumns, targetColumn!).spd);

        const resArr: ResultEntry[] = [
          {
            name: `${modelDef.name} (Untreated)`,
            score: rawMetrics.accuracy,
            color: modelDef.color,
            metrics: {
              "Accuracy": rawMetrics.accuracy,
              "Precision": rawMetrics.precision,
              "Recall": rawMetrics.recall,
              "F1 Score": rawMetrics.f1,
              "SPD (Bias)": Math.abs(rawMetrics.predictionBiasSpd),
            }
          }
        ];

        if (debMetrics) {
          resArr.push({
            name: `${modelDef.name} (Debiased)`,
            score: debMetrics.accuracy,
            color: "hsl(150,65%,45%)",
            metrics: {
              "Accuracy": debMetrics.accuracy,
              "Precision": debMetrics.precision,
              "Recall": debMetrics.recall,
              "F1 Score": debMetrics.f1,
              "SPD (Bias)": Math.abs(debMetrics.predictionBiasSpd),
            }
          });
        } else {
          resArr.push({
            name: "Majority Class Baseline",
            score: rawMetrics.majorityBaseline,
            color: "hsl(215,15%,35%)",
            metrics: { "Accuracy": rawMetrics.majorityBaseline, "SPD (Bias)": labelSpd }
          });
        }

        setRawMetricsFull(rawMetrics);
        setResults(resArr);
        setModelResults(resArr);
        setProgress(100);
        setTrainingState("completed");
        toast.success(`✓ Model trained — ${(rawMetrics.accuracy * 100).toFixed(1)}% accuracy in ${rawMetrics.latency.toFixed(0)}ms`);

        if (datasetDescription) {
          setAiLoading(true);
          getPipelineInsights(datasetDescription, "Model Training", {
            model: modelDef.name, accuracy: rawMetrics.accuracy, biasSpd: Math.abs(rawMetrics.predictionBiasSpd), dataSource
          }).then(insight => { setAiInsight(insight); setAiLoading(false); }).catch(() => setAiLoading(false));
        }
      }

      if (type === "ERROR") {
        toast.error("Training failed: " + message);
        setTrainingState("idle");
        setProgress(0);
      }
    };

    worker.onerror = (e) => {
      toast.error("Worker error: " + e.message);
      setTrainingState("idle");
    };

    worker.postMessage({
      type: "TRAIN",
      payload: {
        rawData: dataset.data,
        debiasedData: debiasedDataset?.data || null,
        targetColumn,
        sensitiveColumns,
        selectedModel,
        hyperParams,
      }
    });
  };

  // ── Boost ───────────────────────────────────────────────────────────────────
  const handleBoostAccuracy = () => {
    const currentData = dataSource === "raw" ? dataset : debiasedDataset;
    if (!currentData || !targetColumn) return;

    const originalResult = results?.find(r =>
      dataSource === "raw" ? r.name.includes("Untreated") : r.name.includes("Debiased")
    );
    const originalAccuracy = originalResult?.score || 0;

    setBoostState("boosting");
    setBoostProgress(0);
    setBoostStatusMsg("Warming up boost engine...");
    setBoostedResults(null);

    const worker = initWorker();

    worker.onmessage = (e: MessageEvent) => {
      const { type, payload, value, status, message } = e.data;

      if (type === "BOOST_PROGRESS") {
        setBoostProgress(value);
        setBoostStatusMsg(status);
      }

      if (type === "BOOST_DONE") {
        const { boostedAccuracy, boostedPrecision, boostedRecall, boostedF1, boostedBiasSpd, logs } = payload;
        
        const originalPrecision = originalResult?.metrics?.["Precision"] || 0;
        const originalRecall = originalResult?.metrics?.["Recall"] || 0;
        const originalF1 = originalResult?.metrics?.["F1 Score"] || 0;
        const originalBias = originalResult?.metrics?.["SPD (Bias)"] || 0;

        setBoostedResults({
          before: originalAccuracy,
          after: boostedAccuracy,
          beforePrecision: originalPrecision,
          afterPrecision: boostedPrecision,
          beforeRecall: originalRecall,
          afterRecall: boostedRecall,
          beforeF1: originalF1,
          afterF1: boostedF1,
          beforeBias: originalBias,
          afterBias: boostedBiasSpd,
          logs
        });
        setBoostedMetrics({
          before: originalAccuracy,
          after: boostedAccuracy,
          precision: boostedPrecision,
          recall: boostedRecall,
          f1: boostedF1,
          logs,
        });
        setBoostProgress(100);
        setBoostState("completed");
        const gain = ((boostedAccuracy - originalAccuracy) * 100).toFixed(1);
        const sign = boostedAccuracy >= originalAccuracy ? "+" : "";
        toast.success(`🚀 Boost complete! ${sign}${gain}% accuracy → ${(boostedAccuracy * 100).toFixed(1)}%`);
      }

      if (type === "BOOST_ERROR") {
        toast.error("Boost failed: " + message);
        setBoostState("idle");
      }
    };

    worker.onerror = (e) => {
      toast.error("Worker error: " + e.message);
      setBoostState("idle");
    };

    worker.postMessage({
      type: "BOOST",
      payload: {
        data: currentData.data,
        targetColumn,
        sensitiveColumns,
        selectedModel,
        originalAccuracy,
      }
    });
  };

  if (!dataset) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-6 text-center">
        <Brain className="h-20 w-20 text-primary/10 animate-pulse" />
        <p className="text-muted-foreground font-display uppercase tracking-widest opacity-50 text-sm">No Dataset Loaded</p>
        <button onClick={() => navigate("/")} className="px-8 py-4 rounded-full bg-primary text-white text-xs font-black uppercase tracking-widest hover:scale-105 transition-all">
          Upload Dataset
        </button>
      </div>
    );
  }

  const currentModelDef = MODEL_DEFINITIONS.find(m => m.id === selectedModel)!;
  const isTraining = trainingState === "training";
  const isDone = trainingState === "completed";

  const radarData = rawMetricsFull ? [
    { subject: "Accuracy", A: rawMetricsFull.accuracy * 100 },
    { subject: "Precision", A: rawMetricsFull.precision * 100 },
    { subject: "Recall", A: rawMetricsFull.recall * 100 },
    { subject: "F1 Score", A: rawMetricsFull.f1 * 100 },
  ] : [];

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-24">
      <PageHeader
        title="Model Lab Pro"
        description="Real-time dual-mode training with bias-aware evaluation"
        icon={<Cpu className="h-5 w-5" />}
      />

      {/* ── Configuration Panel ──────────────────────────────────────────────── */}
      <div className="grid lg:grid-cols-3 gap-6">

        {/* Left: Data Source + Algorithm */}
        <div className="lg:col-span-2 space-y-5">

          {/* Data Source Selector */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-6">
            <h3 className="font-display font-bold text-sm uppercase tracking-widest mb-5 flex items-center gap-2 text-foreground/80">
              <Database className="h-4 w-4 text-primary" /> Training Context
            </h3>
            <div className="grid grid-cols-2 gap-4">
              <button
                onClick={() => setDataSource("raw")}
                className={`flex flex-col p-5 rounded-2xl border-2 transition-all text-left ${
                  dataSource === "raw"
                    ? "bg-rose-500/8 border-rose-500/50 shadow-[0_0_20px_rgba(244,63,94,0.12)]"
                    : "bg-card border-border opacity-60 hover:opacity-90"
                }`}
              >
                <div className="flex items-center gap-2 mb-2">
                  <AlertTriangle className={`h-4 w-4 ${dataSource === "raw" ? "text-rose-500" : "text-muted-foreground"}`} />
                  <span className="text-xs font-black uppercase tracking-wider">Raw Data</span>
                </div>
                <p className="text-[10px] text-muted-foreground leading-relaxed">Unfiltered dataset — may contain latent bias signals</p>
              </button>

              <button
                onClick={() => {
                  if (debiasedDataset) setDataSource("debiased");
                  else { toast.error("Run Fairness Lab first to generate debiased data"); navigate("/fairness"); }
                }}
                className={`flex flex-col p-5 rounded-2xl border-2 transition-all text-left relative overflow-hidden ${
                  dataSource === "debiased"
                    ? "bg-primary/8 border-primary/50 shadow-glow"
                    : "bg-card border-border opacity-60 hover:opacity-90"
                }`}
              >
                {!debiasedDataset && (
                  <div className="absolute top-2 right-2 px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-400 text-[8px] font-black uppercase tracking-widest border border-amber-500/30">
                    Run Fairness Lab First
                  </div>
                )}
                <div className="flex items-center gap-2 mb-2">
                  <ShieldCheck className={`h-4 w-4 ${dataSource === "debiased" ? "text-primary" : "text-muted-foreground"}`} />
                  <span className="text-xs font-black uppercase tracking-wider">AntiBias Certified</span>
                </div>
                <p className="text-[10px] text-muted-foreground leading-relaxed">Fairness-optimized via Optimal Transport alignment</p>
              </button>
            </div>

            {/* Target + Algorithm */}
            <div className="grid md:grid-cols-2 gap-5 mt-5">
              <div>
                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">Target Variable</label>
                <select
                  value={targetColumn ?? ""}
                  onChange={e => setTargetColumn(e.target.value)}
                  className="w-full px-4 py-3 rounded-xl bg-card border border-border text-sm outline-none focus:border-primary transition-colors"
                >
                  <option value="">Select target column...</option>
                  {dataset.columnStats.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">Algorithm</label>
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value as ModelId)}
                  className="w-full px-4 py-3 rounded-xl bg-card border border-border text-sm outline-none focus:border-primary transition-colors"
                >
                  {MODEL_DEFINITIONS.filter(m => m.type === "both" || m.type === taskType).map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
                <p className="text-[10px] text-muted-foreground mt-1.5 ml-1">{currentModelDef.description}</p>
              </div>
            </div>
          </motion.div>

          {/* Hyperparameters */}
          <motion.div initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.05 }} className="glass-card p-6">
            <h3 className="font-display font-bold text-sm uppercase tracking-widest mb-5 flex items-center gap-2 text-foreground/80">
              <Settings2 className="h-4 w-4 text-accent" /> Hyperparameters
            </h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(currentModelDef.params).map(([key, config]) => (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-[11px] font-bold text-foreground/70">{config.label}</label>
                    <span className="text-[10px] font-mono text-primary bg-primary/10 px-2 py-0.5 rounded-md border border-primary/20">{hyperParams[key]}</span>
                  </div>
                  {config.type === "number" ? (
                    <input
                      type="range" min={config.min} max={config.max} step={config.step}
                      value={hyperParams[key] ?? config.default}
                      onChange={e => setHyperParams(prev => ({ ...prev, [key]: Number(e.target.value) }))}
                      className="w-full h-1.5 bg-muted rounded-lg appearance-none cursor-pointer accent-primary"
                    />
                  ) : (
                    <select
                      value={hyperParams[key] ?? config.default}
                      onChange={e => setHyperParams(prev => ({ ...prev, [key]: e.target.value }))}
                      className="w-full px-3 py-2 rounded-lg bg-card border border-border text-xs outline-none"
                    >
                      {config.options?.map(opt => <option key={opt} value={opt}>{opt}</option>)}
                    </select>
                  )}
                </div>
              ))}
            </div>
          </motion.div>
        </div>

        {/* Right: Execution Panel */}
        <motion.div initial={{ opacity: 0, x: 16 }} animate={{ opacity: 1, x: 0 }} className="glass-card p-6 flex flex-col justify-between gap-6">
          <div className="space-y-5">
            <h3 className="font-display font-bold text-sm uppercase tracking-widest text-foreground/80">Execution Context</h3>

            {/* Data badge */}
            <div className="flex items-center gap-3 p-3.5 rounded-xl bg-card border border-border">
              <div className={`p-2 rounded-lg ${dataSource === 'raw' ? 'bg-rose-500/20 text-rose-400' : 'bg-primary/20 text-primary'}`}>
                <Database className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs font-bold">Dataset</p>
                <p className="text-[10px] text-muted-foreground">{debiasedDataset ? "Raw + Debiased available" : "Raw only"}</p>
              </div>
            </div>

            {/* Model badge */}
            <div className={`flex items-center gap-3 p-3.5 rounded-xl border bg-gradient-to-r ${currentModelDef.gradient} bg-opacity-5`} style={{ background: `${currentModelDef.color}12`, borderColor: `${currentModelDef.color}30` }}>
              <div className="p-2 rounded-lg" style={{ background: `${currentModelDef.color}25`, color: currentModelDef.color }}>
                <Brain className="h-4 w-4" />
              </div>
              <div>
                <p className="text-xs font-bold">{currentModelDef.name}</p>
                <p className="text-[10px] text-muted-foreground">{taskType} mode</p>
              </div>
            </div>

            {/* Dataset stats */}
            <div className="grid grid-cols-2 gap-3">
              {[
                { label: "Rows", value: dataset.rows.toLocaleString(), icon: <Gauge className="h-3 w-3" /> },
                { label: "Features", value: (dataset.columns - 1 - sensitiveColumns.length).toString(), icon: <Target className="h-3 w-3" /> },
              ].map(stat => (
                <div key={stat.label} className="p-3 rounded-lg bg-card border border-border text-center">
                  <div className="flex items-center justify-center gap-1 text-muted-foreground mb-1">{stat.icon}<span className="text-[9px] uppercase font-bold">{stat.label}</span></div>
                  <p className="text-lg font-display font-black">{stat.value}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Train Button */}
          <button
            id="train-model-btn"
            onClick={handleTrain}
            disabled={!targetColumn || isTraining}
            className={`w-full py-5 rounded-2xl font-display font-black text-base flex items-center justify-center gap-3 transition-all shadow-lg disabled:opacity-50 disabled:cursor-not-allowed ${
              dataSource === 'raw'
                ? 'bg-gradient-to-r from-rose-600 to-rose-500 text-white hover:scale-[1.02] hover:shadow-[0_8px_30px_rgba(244,63,94,0.4)]'
                : 'bg-gradient-to-r from-primary to-accent text-white hover:scale-[1.02] shadow-glow'
            }`}
          >
            {isTraining ? <RefreshCw className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
            {isTraining ? "TRAINING..." : debiasedDataset ? "TRAIN BOTH MODELS" : "TRAIN MODEL"}
          </button>
        </motion.div>
      </div>

      {/* ── Training Progress ─────────────────────────────────────────────────── */}
      <AnimatePresence>
        {isTraining && (
          <motion.div
            initial={{ opacity: 0, scale: 0.97 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.97 }}
            className="glass-card p-10 flex flex-col items-center justify-center relative overflow-hidden"
          >
            {/* Animated background grid */}
            <div className="absolute inset-0 opacity-5" style={{
              backgroundImage: "linear-gradient(var(--primary) 1px, transparent 1px), linear-gradient(90deg, var(--primary) 1px, transparent 1px)",
              backgroundSize: "40px 40px"
            }} />

            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 3, repeat: Infinity, ease: "linear" }}
              className="relative mb-6"
            >
              <div className="h-16 w-16 rounded-full border-2 border-primary/20 flex items-center justify-center">
                <div className="h-10 w-10 rounded-full border-2 border-t-primary border-r-primary/40 border-b-primary/20 border-l-transparent animate-spin" />
              </div>
              <Brain className="h-6 w-6 text-primary absolute inset-0 m-auto" />
            </motion.div>

            <h2 className="text-xl font-display font-black uppercase tracking-tighter mb-2 relative z-10">
              Training {currentModelDef.name}
            </h2>
            <p className="text-xs text-muted-foreground mb-6 font-mono relative z-10">{statusMsg}</p>

            {/* Real progress bar */}
            <div className="w-full max-w-md relative z-10">
              <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-gradient-to-r from-primary to-accent"
                  animate={{ width: `${progress}%` }}
                  transition={{ duration: 0.4, ease: "easeOut" }}
                />
              </div>
              <div className="flex justify-between mt-2">
                <p className="text-[10px] text-muted-foreground uppercase font-mono">{Math.round(progress)}% complete</p>
                <p className="text-[10px] text-muted-foreground font-mono">Main thread: free ✓</p>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Results ───────────────────────────────────────────────────────────── */}
      <AnimatePresence>
        {isDone && results && (
          <motion.div initial={{ opacity: 0, y: 24 }} animate={{ opacity: 1, y: 0 }} className="space-y-6">

            {/* Performance Report */}
            <div className="glass-card p-8">
              <div className="flex flex-col sm:flex-row justify-between items-start sm:items-center gap-4 mb-8">
                <div>
                  <h3 className="text-2xl font-display font-black tracking-tight flex items-center gap-3">
                    <Zap className="h-6 w-6 text-primary" /> Performance Report
                  </h3>
                  <p className="text-xs text-muted-foreground mt-1 uppercase tracking-widest">
                    {debiasedDataset ? "Raw vs. AntiBias-Certified comparison" : "Model vs. Majority Class Baseline"}
                  </p>
                </div>
                <button
                  onClick={() => navigate("/fairness")}
                  className="px-5 py-2.5 rounded-xl bg-accent text-white font-bold text-xs flex items-center gap-2 hover:scale-105 transition-all"
                >
                  Fairness Audit <ArrowRight className="h-3.5 w-3.5" />
                </button>
              </div>

              <div className="grid lg:grid-cols-2 gap-10 items-start">
                {/* Bar Chart */}
                <div>
                  <ResponsiveContainer width="100%" height={220}>
                    <BarChart data={results} layout="vertical" margin={{ left: 130, right: 30 }}>
                      <XAxis type="number" domain={[0, 1]} hide />
                      <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fontWeight: "bold" }} width={130} axisLine={false} tickLine={false} />
                      <Tooltip
                        cursor={{ fill: "rgba(255,255,255,0.04)" }}
                        contentStyle={{ borderRadius: 12, background: "#111", border: "1px solid rgba(255,255,255,0.1)", fontSize: 11 }}
                        formatter={(v: any) => [`${(Number(v) * 100).toFixed(1)}%`, "Accuracy"]}
                      />
                      <Bar dataKey="score" radius={[0, 8, 8, 0]} barSize={38}>
                        {results.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>

                  {/* Accuracy Labels */}
                  <div className={`grid grid-cols-${results.length} gap-4 mt-3 px-4`}>
                    {results.map((r, i) => (
                      <div key={i} className="text-center p-3 rounded-xl" style={{ background: `${r.color}12`, border: `1px solid ${r.color}25` }}>
                        <p className="text-2xl font-display font-black" style={{ color: r.color }}>
                          {(r.score * 100).toFixed(1)}%
                        </p>
                        <p className="text-[9px] text-muted-foreground uppercase font-bold tracking-wider mt-0.5">
                          {r.name.includes("Untreated") ? "Raw Accuracy" : r.name.includes("Debiased") ? "Fair Accuracy" : "Baseline"}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Metrics + Radar */}
                <div className="space-y-6">
                  {/* Radar Chart */}
                  {radarData.length > 0 && (
                    <div className="bg-card border border-border rounded-2xl p-4">
                      <p className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-3">Model Profile</p>
                      <ResponsiveContainer width="100%" height={180}>
                        <RadarChart data={radarData}>
                          <PolarGrid stroke="rgba(255,255,255,0.06)" />
                          <PolarAngleAxis dataKey="subject" tick={{ fontSize: 9, fill: "rgba(255,255,255,0.5)" }} />
                          <Radar name="Score" dataKey="A" stroke={currentModelDef.color} fill={currentModelDef.color} fillOpacity={0.15} />
                        </RadarChart>
                      </ResponsiveContainer>
                    </div>
                  )}

                  {/* Metric Bars */}
                  {(() => {
                    const activeResult = results.find(r =>
                      dataSource === 'raw' ? r.name.includes("Untreated") : r.name.includes("Debiased")
                    ) || results[0];
                    const metrics = activeResult?.metrics || {};
                    return (
                      <div className="space-y-3">
                        {Object.entries(metrics).map(([name, val], i) => (
                          <MetricBar
                            key={name} label={name} value={val}
                            color={name === "SPD (Bias)" ? (val > 0.1 ? "#f43f5e" : "#10b981") : currentModelDef.color}
                            delay={i * 0.1}
                          />
                        ))}
                      </div>
                    );
                  })()}
                </div>
              </div>

              {/* Latency badge */}
              {rawMetricsFull && (
                <div className="mt-6 flex items-center gap-6 pt-5 border-t border-border">
                  <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                    <Clock className="h-3 w-3" />
                    <span>Training latency: <strong className="text-foreground">{rawMetricsFull.latency.toFixed(0)}ms</strong></span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] text-muted-foreground">
                    <Award className="h-3 w-3" />
                    <span>Majority baseline: <strong className="text-foreground">{(rawMetricsFull.majorityBaseline * 100).toFixed(1)}%</strong></span>
                  </div>
                  <div className="flex items-center gap-2 text-[10px] text-emerald-400">
                    <CheckCircle2 className="h-3 w-3" />
                    <span>Trained on background thread — UI remained responsive</span>
                  </div>
                </div>
              )}

              {/* AI Insight */}
              <AnimatePresence>
                {(aiInsight || aiLoading) && (
                  <motion.div
                    initial={{ opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="mt-6 flex gap-4 p-5 rounded-xl border bg-primary/5 border-primary/20"
                  >
                    <div className="w-8 h-8 rounded-full flex items-center justify-center shrink-0 bg-primary/20 text-primary">
                      <Zap className={`h-4 w-4 ${aiLoading ? "animate-pulse" : ""}`} />
                    </div>
                    <div>
                      <h4 className="text-xs font-bold text-primary mb-1">AI Evaluator Insight</h4>
                      <p className="text-[11px] text-foreground/80 leading-relaxed">
                        {aiLoading ? "Analyzing model performance in context..." : aiInsight}
                      </p>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>

            {/* ── Auto-Boost Section ─────────────────────────────────────────── */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15 }}
              className="glass-card p-8 border-primary/20 relative overflow-hidden"
            >
              {/* BG decoration */}
              <div className="absolute top-0 right-0 p-6 opacity-[0.04] pointer-events-none">
                <Rocket className="h-40 w-40 text-primary" />
              </div>

              <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-5 mb-6 relative z-10">
                <div>
                  <h3 className="text-xl font-display font-black tracking-tight flex items-center gap-3 mb-1">
                    <TrendingUp className="h-5 w-5 text-primary" /> Auto-Boost Performance
                  </h3>
                  <p className="text-[10px] text-muted-foreground uppercase tracking-widest font-bold max-w-lg">
                    Runs feature engineering (Winsorization · Log transform · Polynomial interactions) + Hyperparameter Grid Search in a background worker
                  </p>
                </div>
                <button
                  id="boost-accuracy-btn"
                  onClick={handleBoostAccuracy}
                  disabled={boostState !== "idle"}
                  className={`px-7 py-3.5 rounded-xl font-black text-sm uppercase tracking-widest transition-all whitespace-nowrap flex items-center gap-2 ${
                    boostState === "boosting" ? "bg-primary/40 text-white cursor-wait" :
                    boostState === "completed" ? "bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 cursor-not-allowed" :
                    "bg-gradient-to-r from-primary to-accent text-white shadow-glow hover:scale-105"
                  }`}
                >
                  {boostState === "boosting" ? <><RefreshCw className="h-4 w-4 animate-spin" /> Boosting...</> :
                   boostState === "completed" ? <><CheckCircle2 className="h-4 w-4" /> Boost Applied</> :
                   <><Rocket className="h-4 w-4" /> Boost Accuracy</>}
                </button>
              </div>

              {/* Boost progress */}
              <AnimatePresence>
                {boostState === "boosting" && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="py-6"
                  >
                    <div className="flex items-center gap-3 mb-3">
                      <Activity className="h-4 w-4 text-primary animate-pulse" />
                      <p className="text-xs font-mono text-primary">{boostStatusMsg}</p>
                    </div>
                    <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                      <motion.div
                        className="h-full rounded-full bg-gradient-to-r from-primary via-accent to-primary"
                        animate={{ width: `${boostProgress}%` }}
                        transition={{ duration: 0.4, ease: "easeOut" }}
                      />
                    </div>
                    <p className="text-[10px] text-muted-foreground mt-2 font-mono">{Math.round(boostProgress)}% — running in background thread</p>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Boost Results */}
              <AnimatePresence>
                {boostState === "completed" && boostedResults && (
                  <motion.div
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="grid lg:grid-cols-2 gap-8 items-start pt-6 border-t border-border animate-fade-in"
                  >
                    {/* Before vs After Visual Chart */}
                    <div className="space-y-4">
                      <h4 className="text-[10px] uppercase font-black tracking-widest text-primary flex items-center gap-2">
                        <Zap className="h-3 w-3" /> Accuracy Shift Visualizer
                      </h4>
                      <ResponsiveContainer width="100%" height={190}>
                        <BarChart
                          data={[
                            { name: "Before", score: boostedResults.before },
                            { name: "After Boost", score: boostedResults.after }
                          ]}
                        >
                          <XAxis dataKey="name" tick={{ fontSize: 11, fontWeight: "bold" }} axisLine={false} tickLine={false} />
                          <YAxis domain={[0, 1]} hide />
                          <Tooltip
                            cursor={{ fill: "rgba(255,255,255,0.04)" }}
                            contentStyle={{ borderRadius: 12, background: "#111", border: "1px solid rgba(255,255,255,0.1)" }}
                            formatter={(v: any) => [`${(Number(v) * 100).toFixed(1)}%`, "Accuracy"]}
                          />
                          <Bar dataKey="score" radius={[8, 8, 0, 0]} barSize={56}>
                            <Cell fill="hsl(215,15%,35%)" />
                            <Cell fill="hsl(150,65%,45%)" />
                          </Bar>
                        </BarChart>
                      </ResponsiveContainer>

                      {/* Net Gain Card */}
                      <div className="text-center p-4 rounded-xl bg-primary/5 border border-primary/20">
                        <p className={`text-3xl font-display font-black ${boostedResults.after >= boostedResults.before ? "text-primary" : "text-rose-400"}`}>
                          {boostedResults.after >= boostedResults.before ? "+" : ""}{((boostedResults.after - boostedResults.before) * 100).toFixed(1)}%
                        </p>
                        <p className="text-[10px] text-muted-foreground uppercase font-bold tracking-widest mt-1">Net Accuracy Gain</p>
                      </div>
                    </div>

                    {/* Metrics Report Table */}
                    <div className="space-y-4">
                      <h4 className="text-[10px] uppercase font-black tracking-widest text-primary flex items-center gap-2">
                        <Activity className="h-3 w-3" /> Performance Metrics Shift
                      </h4>
                      <div className="overflow-hidden rounded-2xl border border-border bg-card">
                        <table className="w-full text-left border-collapse text-xs">
                          <thead>
                            <tr className="bg-muted/50 border-b border-border">
                              <th className="p-4 font-black text-muted-foreground uppercase">Metric</th>
                              <th className="p-4 font-black text-muted-foreground uppercase text-center">Before</th>
                              <th className="p-4 font-black text-muted-foreground uppercase text-center">After</th>
                              <th className="p-4 font-black text-muted-foreground uppercase text-center">Delta</th>
                            </tr>
                          </thead>
                          <tbody>
                            {[
                              { name: "Accuracy", before: boostedResults.before, after: boostedResults.after, higherBetter: true },
                              { name: "Precision", before: boostedResults.beforePrecision, after: boostedResults.afterPrecision, higherBetter: true },
                              { name: "Recall", before: boostedResults.beforeRecall, after: boostedResults.afterRecall, higherBetter: true },
                              { name: "F1-Score", before: boostedResults.beforeF1, after: boostedResults.afterF1, higherBetter: true },
                              { name: "SPD (Model Bias)", before: boostedResults.beforeBias, after: Math.abs(boostedResults.afterBias), higherBetter: false }
                            ].map((row, i) => {
                              const delta = row.after - row.before;
                              const isPos = delta >= 0;
                              const isBetter = row.higherBetter ? isPos : !isPos;
                              const colorClass = Math.abs(delta) < 0.0001 ? "text-muted-foreground" : isBetter ? "text-emerald-500 font-bold" : "text-rose-500 font-bold";
                              const deltaText = Math.abs(delta) < 0.0001 ? "0.0%" : `${isPos ? "+" : ""}${(delta * 100).toFixed(1)}%`;
                              
                              return (
                                <tr key={i} className="border-b border-border/30 last:border-0 hover:bg-muted/10 transition-colors">
                                  <td className="p-4 font-bold">{row.name}</td>
                                  <td className="p-4 text-center font-mono">{(row.before * 100).toFixed(1)}%</td>
                                  <td className="p-4 text-center font-mono font-bold text-primary">{(row.after * 100).toFixed(1)}%</td>
                                  <td className={`p-4 text-center font-mono ${colorClass}`}>{deltaText}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                    </div>

                    {/* Optimization Log */}
                    <div className="bg-card border border-border rounded-2xl p-6 lg:col-span-2">
                      <h4 className="text-[10px] uppercase font-black tracking-widest text-primary mb-4 flex items-center gap-2">
                        <CheckCircle2 className="h-3 w-3" /> Optimizations Applied
                      </h4>
                      <div className="grid md:grid-cols-2 gap-3">
                        {boostedResults.logs.map((log, idx) => (
                          <motion.div
                            key={idx}
                            initial={{ opacity: 0, x: -10 }}
                            animate={{ opacity: 1, x: 0 }}
                            transition={{ delay: idx * 0.05 }}
                            className="text-[10px] leading-relaxed font-mono text-foreground/80 flex items-start gap-2.5 p-3 rounded-lg bg-muted/30 border border-border/50"
                          >
                            <ChevronRight className="h-3 w-3 text-primary shrink-0 mt-0.5" />
                            <span>{log}</span>
                          </motion.div>
                        ))}
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>

          </motion.div>
        )}
      </AnimatePresence>

      {isDone && <PageFooter nextLabel="Results & Insights" nextUrl="/results" />}
    </div>
  );
}
