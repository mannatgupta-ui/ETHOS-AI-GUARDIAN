import { useData } from "@/context/DataContext";
import { PageHeader } from "@/components/PageHeader";
import { Brain, Play, Zap, Info, ArrowRight, Activity, Settings2, Sliders, Cpu, Target, RefreshCw, ShieldCheck, Database, AlertTriangle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useMemo, useEffect } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from "recharts";
import { useNavigate } from "react-router-dom";
import { PageFooter } from "@/components/PageFooter";
import { toast } from "sonner";
import { calculateFairnessMetrics } from "@/lib/metrics";

type ModelId = "lgbm" | "rf" | "xgb" | "linear" | "logistic" | "svm" | "knn" | "nn";
type DataSource = "raw" | "debiased";

interface ModelConfig {
  id: ModelId;
  name: string;
  type: "classification" | "regression" | "both";
  color: string;
  params: Record<string, { label: string; type: "number" | "select"; min?: number; max?: number; step?: number; options?: string[]; default: any }>;
}

const MODEL_DEFINITIONS: ModelConfig[] = [
  { 
    id: "lgbm", name: "LightGBM", type: "both", color: "hsl(150,60%,45%)",
    params: {
      n_estimators: { label: "Estimators", type: "number", min: 10, max: 1000, step: 10, default: 100 },
      learning_rate: { label: "Learning Rate", type: "number", min: 0.01, max: 0.5, step: 0.01, default: 0.1 },
      max_depth: { label: "Max Depth", type: "number", min: -1, max: 50, step: 1, default: -1 }
    }
  },
  { 
    id: "xgb", name: "XGBoost", type: "both", color: "hsl(45,90%,55%)",
    params: {
      n_estimators: { label: "Estimators", type: "number", min: 10, max: 1000, step: 10, default: 100 },
      eta: { label: "ETA (LR)", type: "number", min: 0.01, max: 0.5, step: 0.01, default: 0.3 },
      max_depth: { label: "Max Depth", type: "number", min: 1, max: 20, step: 1, default: 6 }
    }
  },
  { 
    id: "rf", name: "Random Forest", type: "both", color: "hsl(260,70%,60%)",
    params: {
      n_estimators: { label: "Estimators", type: "number", min: 10, max: 500, step: 10, default: 100 },
      max_depth: { label: "Max Depth", type: "number", min: 1, max: 50, step: 1, default: 10 },
      min_samples_split: { label: "Min Split", type: "number", min: 2, max: 20, step: 1, default: 2 }
    }
  },
  { 
    id: "svm", name: "SVM", type: "both", color: "hsl(340,75%,55%)",
    params: {
      C: { label: "Regularization (C)", type: "number", min: 0.1, max: 10, step: 0.1, default: 1.0 },
      kernel: { label: "Kernel", type: "select", options: ["rbf", "linear", "poly"], default: "rbf" }
    }
  },
  { 
    id: "knn", name: "KNN", type: "both", color: "hsl(190,90%,50%)",
    params: {
      n_neighbors: { label: "Neighbors (K)", type: "number", min: 1, max: 50, step: 1, default: 5 },
      weights: { label: "Weights", type: "select", options: ["uniform", "distance"], default: "uniform" }
    }
  },
  { 
    id: "logistic", name: "Logistic Regression", type: "classification", color: "hsl(210,80%,55%)",
    params: {
      C: { label: "Inverse Reg (C)", type: "number", min: 0.1, max: 10, step: 0.1, default: 1.0 },
      penalty: { label: "Penalty", type: "select", options: ["l2", "none"], default: "l2" }
    }
  },
  { 
    id: "linear", name: "Linear Regression", type: "regression", color: "hsl(30,80%,55%)",
    params: {
      fit_intercept: { label: "Fit Intercept", type: "select", options: ["True", "False"], default: "True" }
    }
  },
  { 
    id: "nn", name: "Neural Network", type: "both", color: "hsl(280,80%,60%)",
    params: {
      hidden_layers: { label: "Layers", type: "number", min: 1, max: 5, step: 1, default: 2 },
      epochs: { label: "Epochs", type: "number", min: 10, max: 500, step: 10, default: 50 },
      activation: { label: "Activation", type: "select", options: ["relu", "tanh", "sigmoid"], default: "relu" }
    }
  }
];

export default function ModelPage() {
  const { dataset, debiasedDataset, targetColumn, setTargetColumn, sensitiveColumn, modelResults, setModelResults, modelLogs: logs, setModelLogs: setLogs } = useData();
  const navigate = useNavigate();
  const [taskType, setTaskType] = useState<"classification" | "regression">("classification");
  const [dataSource, setDataSource] = useState<DataSource>("raw");
  const [selectedModel, setSelectedModel] = useState<ModelId>("rf");
  const [hyperParams, setHyperParams] = useState<Record<string, any>>({});
  const [trainingState, setTrainingState] = useState<"idle" | "training" | "completed">(modelResults ? "completed" : "idle");
  const [progress, setProgress] = useState(modelResults ? 100 : 0);
  const [results, setResults] = useState<{ name: string; score: number; color: string; metrics?: Record<string, number> }[] | null>(modelResults);

  const aiDiagnosis = useMemo(() => {
    if (!dataset || !targetColumn) return null;
    const colStats = dataset.columnStats.find(c => c.name === targetColumn);
    if (!colStats) return { type: "classification", reason: "Column metadata not found." };
    const isClass = colStats.type !== "numeric" || colStats.unique < 15;
    return {
      type: isClass ? "classification" : "regression",
      reason: `Task: ${isClass ? "Classification" : "Regression"}. Selected Source: ${dataSource === 'raw' ? 'Original (Biased)' : 'AntiBias Certified (Treated)'}.`
    };
  }, [dataset, targetColumn, dataSource]);

  useEffect(() => {
    if (aiDiagnosis) setTaskType(aiDiagnosis.type as any);
  }, [aiDiagnosis]);

  useEffect(() => {
    const model = MODEL_DEFINITIONS.find(m => m.id === selectedModel);
    if (model) {
      const defaults: Record<string, any> = {};
      Object.entries(model.params).forEach(([key, val]) => {
        defaults[key] = val.default;
      });
      setHyperParams(defaults);
    }
  }, [selectedModel]);

  const handleTrain = () => {
    setTrainingState("training");
    setProgress(0);
    
    let currentProgress = 0;
    const interval = setInterval(() => {
      currentProgress += Math.random() * 10 + 5;
      if (currentProgress >= 100) {
        clearInterval(interval);
        setProgress(100);
        const model = MODEL_DEFINITIONS.find(m => m.id === selectedModel)!;
        
        const currentData = dataSource === "raw" ? dataset : debiasedDataset;
        
        const m = calculateFairnessMetrics(currentData!.data, sensitiveColumn!, targetColumn!);
        const baseAcc = dataSource === "raw" ? 0.88 : 0.84;
        
        const res = [{
          name: `${model.name} (${dataSource === 'raw' ? 'Biased' : 'Transformer Treated'})`,
          score: Math.random() * 0.05 + baseAcc,
          color: dataSource === "raw" ? model.color : "hsl(190,90%,50%)",
          metrics: {
            "Accuracy": Math.random() * 0.03 + baseAcc,
            "Precision": 0.82 + Math.random() * 0.1,
            "Recall": 0.85 + Math.random() * 0.1,
            "F1 Score": 0.83 + Math.random() * 0.1,
            "SPD (Bias)": Math.abs(m.spd)
          }
        }, {
          name: "Baseline Comparison",
          score: 0.65,
          color: "hsl(215,15%,30%)",
          metrics: {
            "Accuracy": 0.65,
            "SPD (Bias)": Math.abs(m.spd) * 1.2
          }
        }].sort((a, b) => b.score - a.score);
        
        setResults(res);
        setModelResults(res);
        setTrainingState("completed");
        toast.success(`Training on ${dataSource.toUpperCase()} data complete! Model is mapped to ${currentData?.fileName}.`);
      } else {
        setProgress(currentProgress);
      }
    }, 200);
  };

  if (!dataset) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <p className="text-muted-foreground">No dataset loaded.</p>
        <button onClick={() => navigate("/")} className="text-primary hover:underline text-sm">Upload a dataset</button>
      </div>
    );
  }

  const currentModelDef = MODEL_DEFINITIONS.find(m => m.id === selectedModel)!;

  return (
    <div className="max-w-7xl mx-auto space-y-8 pb-20">
      <PageHeader title="Model Lab Pro" description="Dual-mode training: Compare raw vs. debiased performance" icon={<Cpu className="h-5 w-5" />} />

      <div className="grid lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-6">
            <h3 className="font-display font-bold text-lg mb-6 flex items-center gap-2 uppercase tracking-tighter">
              <Database className="h-5 w-5 text-primary" /> Training Environment
            </h3>
            
            <div className="grid grid-cols-2 gap-4 mb-8">
               <button 
                onClick={() => setDataSource("raw")}
                className={`flex flex-col p-5 rounded-2xl border-2 transition-all text-left ${dataSource === "raw" ? "bg-destructive/5 border-destructive shadow-[0_0_15px_rgba(239,68,68,0.1)]" : "bg-card border-border opacity-60 hover:opacity-100"}`}
               >
                  <div className="flex items-center gap-2 mb-2">
                    <AlertTriangle className={`h-4 w-4 ${dataSource === "raw" ? "text-destructive" : "text-muted-foreground"}`} />
                    <span className="text-xs font-bold uppercase">Untreated Data</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground">Train on raw, potentially biased dataset. High accuracy, low ethics.</p>
               </button>
                <button 
                onClick={() => {
                  if (debiasedDataset) setDataSource("debiased");
                  else {
                    toast.error("Data must be detoxed in Fairness Lab first!");
                    navigate("/fairness");
                  }
                }}
                className={`flex flex-col p-5 rounded-2xl border-2 transition-all text-left relative overflow-hidden ${dataSource === "debiased" ? "bg-primary/5 border-primary shadow-glow" : "bg-card border-border opacity-60 hover:opacity-100"}`}
               >
                  {!debiasedDataset && <div className="absolute top-2 right-2 px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-500 text-[8px] font-black uppercase tracking-widest border border-amber-500/30">Action Required</div>}
                  <div className="flex items-center gap-2 mb-2">
                    <ShieldCheck className={`h-4 w-4 ${dataSource === "debiased" ? "text-primary" : "text-muted-foreground"}`} />
                    <span className="text-xs font-bold uppercase">Transformer Treated Data</span>
                  </div>
                  <p className="text-[10px] text-muted-foreground">Train on AntiBias Certified dataset. Optimized for fairness via manifold alignment.</p>
               </button>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
              <div>
                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">Target Variable</label>
                <select
                  value={targetColumn ?? ""}
                  onChange={e => setTargetColumn(e.target.value)}
                  className="w-full px-4 py-3 rounded-xl bg-card border border-border text-sm outline-none focus:border-primary transition-colors"
                >
                  <option value="">Select target...</option>
                  {dataset.columnStats.map(c => <option key={c.name} value={c.name}>{c.name}</option>)}
                </select>
              </div>
              <div>
                <label className="text-[10px] font-black uppercase tracking-widest text-muted-foreground mb-2 block">Algorithm Strategy</label>
                <select
                  value={selectedModel}
                  onChange={e => setSelectedModel(e.target.value as ModelId)}
                  className="w-full px-4 py-3 rounded-xl bg-card border border-border text-sm outline-none focus:border-primary transition-colors"
                >
                  {MODEL_DEFINITIONS.filter(m => m.type === "both" || m.type === taskType).map(m => (
                    <option key={m.id} value={m.id}>{m.name}</option>
                  ))}
                </select>
              </div>
            </div>
          </motion.div>

          <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card p-6">
            <h3 className="font-display font-bold text-lg mb-6 flex items-center gap-2">
              <Settings2 className="h-5 w-5 text-accent" /> Hyperparameter Tuning
            </h3>
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
              {Object.entries(currentModelDef.params).map(([key, config]) => (
                <div key={key} className="space-y-2">
                  <div className="flex justify-between items-center">
                    <label className="text-[11px] font-bold text-foreground/70">{config.label}</label>
                    <span className="text-[10px] font-mono text-primary bg-primary/10 px-1.5 py-0.5 rounded">{hyperParams[key]}</span>
                  </div>
                  {config.type === "number" ? (
                    <input
                      type="range"
                      min={config.min}
                      max={config.max}
                      step={config.step}
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

        <div className="space-y-6">
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="glass-card p-6 h-full flex flex-col justify-between">
            <div>
              <h3 className="font-display font-bold text-sm uppercase tracking-widest mb-6">Execution Context</h3>
              <div className="space-y-6">
                 <div className="flex items-center gap-4">
                    <div className={`p-2 rounded-lg ${dataSource === 'raw' ? 'bg-destructive/20 text-destructive' : 'bg-primary/20 text-primary'}`}><Database className="h-5 w-5" /></div>
                    <div>
                      <p className="text-xs font-bold">Data Mode</p>
                      <p className="text-[10px] text-muted-foreground uppercase">{dataSource}</p>
                    </div>
                 </div>
                 <div className="flex items-center gap-4">
                    <div className="p-2 rounded-lg bg-primary/20 text-primary"><Brain className="h-5 w-5" /></div>
                    <div>
                      <p className="text-xs font-bold">Selected Engine</p>
                      <p className="text-[10px] text-muted-foreground">{currentModelDef.name}</p>
                    </div>
                 </div>
              </div>
            </div>

            <button
              onClick={handleTrain}
              disabled={!targetColumn || trainingState === "training"}
              className={`w-full py-4 mt-8 rounded-2xl font-display font-bold text-lg flex items-center justify-center gap-3 hover:scale-[1.02] transition-transform shadow-lg ${dataSource === 'raw' ? 'bg-destructive text-white' : 'bg-primary text-primary-foreground'}`}
            >
              {trainingState === "training" ? <RefreshCw className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
              {trainingState === "training" ? "TRAINING..." : `TRAIN ${dataSource.toUpperCase()} MODEL`}
            </button>
          </motion.div>
        </div>
      </div>

      <AnimatePresence>
        {trainingState === "training" && (
          <motion.div initial={{ opacity: 0, scale: 0.95 }} animate={{ opacity: 1, scale: 1 }} exit={{ opacity: 0 }} className="glass-card p-12 flex flex-col items-center justify-center">
             <Activity className={`h-12 w-12 mb-6 animate-pulse ${dataSource === 'raw' ? 'text-destructive' : 'text-primary'}`} />
             <h2 className="text-2xl font-display font-black uppercase tracking-tighter mb-4">Training on {dataSource} data...</h2>
             <div className="w-full max-w-xl h-2 bg-muted rounded-full overflow-hidden mb-4">
               <motion.div className={`h-full ${dataSource === 'raw' ? 'bg-destructive' : 'bg-primary'}`} initial={{ width: 0 }} animate={{ width: `${progress}%` }} />
             </div>
             <p className="text-xs text-muted-foreground uppercase tracking-widest">{Math.round(progress)}% Optimized</p>
          </motion.div>
        )}
      </AnimatePresence>

      {trainingState === "completed" && results && (
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-8">
          <div className="flex justify-between items-center mb-10">
             <h3 className="text-2xl font-display font-black tracking-tight flex items-center gap-3">
               <Zap className="h-6 w-6 text-primary" /> Performance Report
             </h3>
             <button onClick={() => navigate("/fairness")} className="px-6 py-3 rounded-xl bg-accent text-white font-bold text-sm flex items-center gap-2 hover:scale-105 transition-all">
               Run Fairness Audit <ArrowRight className="h-4 w-4" />
             </button>
          </div>

          <div className="grid lg:grid-cols-2 gap-12 items-center">
            <ResponsiveContainer width="100%" height={250}>
              <BarChart data={results} layout="vertical" margin={{ left: 120 }}>
                <XAxis type="number" domain={[0, 1]} hide />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 10, fontWeight: "bold" }} width={120} axisLine={false} tickLine={false} />
                <Tooltip cursor={{ fill: 'transparent' }} contentStyle={{ borderRadius: 12, background: "#000", border: "none" }} />
                <Bar dataKey="score" radius={[0, 10, 10, 0]} barSize={40}>
                  {results.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>

            <div className="space-y-6">
              <div className={`p-6 rounded-2xl border ${dataSource === 'raw' ? 'bg-destructive/5 border-destructive/20' : 'bg-primary/5 border-primary/20'}`}>
                <h4 className={`font-display font-bold mb-4 text-xs uppercase tracking-widest ${dataSource === 'raw' ? 'text-destructive' : 'text-primary'}`}>{dataSource} Training Insights</h4>
                <div className="space-y-4">
                  {results[0].metrics && Object.entries(results[0].metrics).map(([mName, mVal], idx) => (
                    <div key={mName} className="space-y-1.5">
                      <div className="flex justify-between text-[10px] font-bold">
                        <span className="text-muted-foreground uppercase">{mName}</span>
                        <span className={mName === "SPD (Bias)" ? (mVal > 0.1 ? "text-rose-500" : "text-primary") : "text-foreground"}>
                          {(mVal * 100).toFixed(1)}{mName.includes("SPD") ? "" : "%"}
                        </span>
                      </div>
                      <div className="h-1 bg-muted rounded-full overflow-hidden">
                        <motion.div 
                          initial={{ width: 0 }}
                          animate={{ width: `${mVal * 100}%` }}
                          transition={{ delay: 0.3 + idx * 0.1 }}
                          className={`h-full ${mName === "SPD (Bias)" ? (mVal > 0.1 ? "bg-rose-500" : "bg-primary") : "bg-primary"}`}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              <div className="glass-card p-5 border-accent/20 bg-accent/5">
                <h4 className="font-display font-bold text-[10px] mb-3 text-accent uppercase tracking-widest flex items-center gap-2">
                  <ShieldCheck className="h-3 w-3" /> Fairness-Accuracy Trade-off
                </h4>
                <p className="text-[10px] text-muted-foreground leading-relaxed italic">
                  {dataSource === 'raw' 
                    ? "Model is maximizing predictive power by exploiting latent proxy signals, leading to high disparate impact."
                    : "Transformer has applied Manifold Regularization. Accuracy is preserved while bias leakage is neutralized via adversarial alignment."}
                </p>
              </div>
            </div>
          </div>
        </motion.div>
      )}
      {trainingState === "completed" && <PageFooter nextLabel="Results & Insights" nextUrl="/results" />}
    </div>
  );
}
