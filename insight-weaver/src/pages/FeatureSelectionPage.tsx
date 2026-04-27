import { useData } from "@/context/DataContext";
import { PageHeader } from "@/components/PageHeader";
import { Filter, Play, CheckCircle2, ChevronRight, Binary, Fingerprint, Zap } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { PageFooter } from "@/components/PageFooter";
import { Shield, ShieldPlus, ShieldCheck, X } from "lucide-react";
import { calculateMutualInfo } from "@/lib/metrics";

export default function FeatureSelectionPage() {
  const { dataset, setDataset, safeList, toggleSafeColumn, selectionStep: step, setSelectionStep: setStep } = useData();

  const initialColumns = dataset?.columns || 0;

  const filter1Results = useMemo(() => {
    if (!dataset) return { kept: [], dropped: [] };
    const dropped = dataset.columnStats.filter(c => c.missingPct > 50 && !safeList.includes(c.name)).map(c => c.name);
    const kept = dataset.columnStats.filter(c => c.missingPct <= 50 || safeList.includes(c.name));
    return { kept, dropped };
  }, [dataset, safeList]);

  const filter2Results = useMemo(() => {
    const dropped = filter1Results.kept.filter(c => c.unique <= 1 && !safeList.includes(c.name)).map(c => c.name);
    const kept = filter1Results.kept.filter(c => c.unique > 1 || safeList.includes(c.name));
    return { kept, dropped };
  }, [filter1Results, safeList]);

  const filter3Results = useMemo(() => {
    // Separate safe features from the rest
    const safeFeatures = filter2Results.kept.filter(c => safeList.includes(c.name));
    const otherFeatures = filter2Results.kept.filter(c => !safeList.includes(c.name));
    
    // Drop bottom threshold of relevance scores for non-safe features
    const sortedOthers = [...otherFeatures].sort((a, b) => (b.relevanceScore || 0) - (a.relevanceScore || 0));
    
    // For large datasets, keep top 29 or top 75%
    const baseTargetCount = initialColumns > 100 ? 29 : Math.max(1, Math.floor((filter2Results.kept.length) * 0.75));
    // Final count should at least include all safe features
    const targetCount = Math.max(safeFeatures.length, baseTargetCount);
    
    const keptOthers = sortedOthers.slice(0, Math.max(0, targetCount - safeFeatures.length));
    const droppedOthers = sortedOthers.slice(Math.max(0, targetCount - safeFeatures.length));
    
    const kept = [...safeFeatures, ...keptOthers];
    const dropped = droppedOthers.map(c => c.name);
    
    // Calculate MI for UI display
    let avgMIKept = 0;
    let avgMIDropped = 0;
    if (dataset && dataset.headers.includes("hired") || (dataset && dataset.headers[dataset.headers.length-1])) {
      const target = dataset.headers.includes("hired") ? "hired" : dataset.headers[dataset.headers.length-1];
      const keptMI = kept.map(c => calculateMutualInfo(dataset.data, c.name, target));
      const droppedMI = droppedOthers.map(c => calculateMutualInfo(dataset.data, c.name, target));
      
      avgMIKept = keptMI.length ? keptMI.reduce((a, b) => a + b) / keptMI.length : 0;
      avgMIDropped = droppedMI.length ? droppedMI.reduce((a, b) => a + b) / droppedMI.length : 0;
    }

    return { kept, dropped, avgMIKept, avgMIDropped };
  }, [filter2Results, initialColumns, safeList, dataset]);

  const filter1Drops = filter1Results.dropped.length;
  const filter2Drops = filter2Results.dropped.length;
  const filter3Drops = filter3Results.dropped.length;
  const finalColumns = filter3Results.kept.length;

  const handleRunFilters = () => {
    if (step === 0) setStep(1);
    
    setTimeout(() => setStep(2), 1500);
    setTimeout(() => setStep(3), 3000);
    setTimeout(() => setStep(4), 4500);
  };

  const handleCommit = () => {
    if (!dataset) return;
    
    const keepKeys = filter3Results.kept;
    const newHeaders = keepKeys.map(c => c.name);
    
    const abbreviatedData = dataset.data.map(row => {
      const newRow: any = {};
      newHeaders.forEach(k => newRow[k] = row[k]);
      return newRow;
    });

    setDataset({
      ...dataset,
      columns: finalColumns,
      headers: newHeaders,
      columnStats: keepKeys,
      data: abbreviatedData
    });
    
    navigate("/eda");
  };

  if (!dataset) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <p className="text-muted-foreground">No dataset loaded.</p>
        <button onClick={() => navigate("/")} className="text-primary hover:underline text-sm">Upload a dataset</button>
      </div>
    );
  }

  return (
    <div className="max-w-5xl mx-auto">
      <PageHeader title="Feature Selection Pipeline" description="Automatically detecting and pruning statistical noise using multi-stage filtering" icon={<Filter className="h-5 w-5" />} />

      <div className="flex justify-between items-center bg-card border border-border p-5 rounded-2xl mb-8 shadow-xl">
        <div className="text-center">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Starting Features</p>
          <p className="text-3xl font-display font-medium text-foreground">{initialColumns}</p>
        </div>
        <div className="flex-1 px-8 flex items-center justify-center relative">
          <div className="w-full h-1 bg-muted rounded-full overflow-hidden absolute top-1/2 -translate-y-1/2 -z-10">
            <motion.div 
              className="h-full bg-primary"
              initial={{ width: "0%" }}
              animate={{ width: step > 0 ? `${(step / 4) * 100}%` : "0%" }}
              transition={{ duration: 1 }}
            />
          </div>
          <ChevronRight className="text-muted-foreground bg-card" />
        </div>
        <div className="text-center">
          <p className="text-xs text-muted-foreground uppercase tracking-wider mb-1">Final Kept</p>
          <p className="text-3xl font-display font-bold text-primary">{step === 4 ? finalColumns : "?"}</p>
        </div>
      </div>

      <div className="space-y-4 relative">
        <div className="absolute left-6 top-8 bottom-8 w-0.5 bg-muted"></div>

        {/* Filter 1 */}
        <div className={`relative flex gap-6 p-6 rounded-xl border transition-all duration-500 ${step >= 1 ? "bg-card border-border shadow-md" : "opacity-40 border-transparent grayscale"}`}>
          <div className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 z-10 transition-colors ${step >= 2 ? "bg-primary text-primary-foreground" : step === 1 ? "bg-primary/20 text-primary animate-pulse" : "bg-muted text-muted-foreground"}`}>
            {step >= 2 ? <CheckCircle2 className="h-6 w-6" /> : <Binary className="h-5 w-5" />}
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-display font-semibold mb-1">Filter 1: High Missingness</h3>
            <p className="text-sm text-foreground/70 mb-3">Eliminates variables where {">"}50% of records are null, preventing destructive mathematical imputation.</p>
            <AnimatePresence>
              {step >= 2 && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="flex flex-col gap-2 bg-destructive/10 border border-destructive/20 text-destructive rounded-lg overflow-hidden">
                  <div className="flex justify-between items-center px-4 py-2.5">
                    <span>Invalid variables dropped:</span>
                    <span className="font-bold text-lg">-{filter1Drops}</span>
                  </div>
                  {filter1Drops > 0 && (
                    <div className="px-4 pb-3 pt-1 border-t border-destructive/10">
                      <p className="text-[10px] uppercase tracking-wider mb-2 opacity-70">Dropped due to &gt;50% missingness:</p>
                      <div className="flex flex-wrap gap-1.5">
                        {filter1Results.dropped.map(name => (
                          <div key={name} className="group relative flex items-center">
                            <span className="px-2 py-0.5 rounded bg-destructive/20 text-[11px] font-medium pr-6">{name}</span>
                            <button 
                              onClick={() => toggleSafeColumn(name)}
                              className="absolute right-1 text-destructive/40 hover:text-primary transition-colors p-0.5"
                              title="Protect this variable"
                            >
                              <ShieldPlus className="h-3 w-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Filter 2 */}
        <div className={`relative flex gap-6 p-6 rounded-xl border transition-all duration-500 ${step >= 2 ? "bg-card border-border shadow-md" : "opacity-40 border-transparent grayscale"}`}>
          <div className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 z-10 transition-colors ${step >= 3 ? "bg-accent text-accent-foreground" : step === 2 ? "bg-accent/20 text-accent animate-pulse" : "bg-muted text-muted-foreground"}`}>
            {step >= 3 ? <CheckCircle2 className="h-6 w-6" /> : <Fingerprint className="h-5 w-5" />}
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-display font-semibold mb-1">Filter 2: Zero / Low Variance</h3>
            <p className="text-sm text-foreground/70 mb-3">Identifies invariant features (metadata flags where all users share the same value) that mathematically offer zero predictive power.</p>
            <AnimatePresence>
              {step >= 3 && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="flex flex-col gap-2 bg-accent/10 border border-accent/20 text-accent rounded-lg overflow-hidden">
                  <div className="flex justify-between items-center px-4 py-2.5">
                    <span>Static variables dropped:</span>
                    <span className="font-bold text-lg">-{filter2Drops}</span>
                  </div>
                  {filter2Drops > 0 && (
                    <div className="px-4 pb-3 pt-1 border-t border-accent/10">
                      <p className="text-[10px] uppercase tracking-wider mb-2 opacity-70">Dropped due to zero/low variance:</p>
                      <div className="flex flex-wrap gap-1.5">
                        {filter2Results.dropped.map(name => (
                          <div key={name} className="group relative flex items-center">
                            <span className="px-2 py-0.5 rounded bg-accent/20 text-[11px] font-medium pr-6">{name}</span>
                            <button 
                              onClick={() => toggleSafeColumn(name)}
                              className="absolute right-1 text-accent/40 hover:text-primary transition-colors p-0.5"
                              title="Protect this variable"
                            >
                              <ShieldPlus className="h-3 w-3" />
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Filter 3 */}
        <div className={`relative flex gap-6 p-6 rounded-xl border transition-all duration-500 ${step >= 3 ? "bg-card border-border shadow-md" : "opacity-40 border-transparent grayscale"}`}>
          <div className={`w-12 h-12 rounded-full flex items-center justify-center shrink-0 z-10 transition-colors ${step >= 4 ? "bg-[#ff0055] text-white" : step === 3 ? "bg-[#ff0055]/20 text-[#ff0055] animate-pulse" : "bg-muted text-muted-foreground"}`}>
            {step >= 4 ? <CheckCircle2 className="h-6 w-6" /> : <Zap className="h-5 w-5" />}
          </div>
          <div className="flex-1">
            <h3 className="text-lg font-display font-semibold mb-1">Filter 3: Transformer Contextual Pruning</h3>
            <p className="text-sm text-foreground/70 mb-3">Uses attention-based scoring to identify which features provide actual predictive signal vs. those that just mirror the noise or provide biased social context.</p>
            <AnimatePresence>
              {step >= 4 && (
                <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} className="flex flex-col gap-2 bg-gradient-to-r from-[#ff0055]/10 to-transparent border border-[#ff0055]/20 text-foreground text-sm rounded-lg overflow-hidden">
                  <div className="flex justify-between items-center px-4 py-3 border-b border-border/50">
                    <span className="text-[#ff0055]">Predictive noise variables dropped:</span>
                    <span className="font-bold text-lg text-[#ff0055]">-{filter3Drops}</span>
                  </div>
                  {filter3Drops > 0 && (
                    <div className="px-4 py-3 bg-[#ff0055]/5">
                      <p className="text-[10px] uppercase tracking-wider mb-2 text-[#ff0055]/70">Low Mutual Information scorers:</p>
                      <div className="flex flex-wrap gap-2 max-h-[150px] overflow-y-auto pr-2 custom-scrollbar">
                        {filter3Results.dropped.map(name => (
                          <div key={name} className="flex items-center gap-1.5 p-1 rounded-lg bg-[#ff0055]/5 border border-[#ff0055]/10 group">
                            <span className="px-2 py-0.5 text-[#ff0055] text-[11px] font-bold">{name}</span>
                            <button 
                              onClick={() => toggleSafeColumn(name)}
                              className="px-2 py-0.5 rounded bg-[#ff0055] text-white text-[9px] font-black uppercase tracking-tighter hover:scale-105 transition-all flex items-center gap-1 shadow-sm"
                            >
                              <ShieldPlus className="h-2.5 w-2.5" /> Keep
                            </button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                   {safeList.length > 0 && (
                    <div className="px-4 py-3 bg-primary/5 border-t border-primary/10 flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <ShieldCheck className="h-3.5 w-3.5 text-primary" />
                        <span className="text-[10px] font-black text-primary uppercase tracking-[0.2em]">Protected Features</span>
                      </div>
                      <div className="flex flex-wrap gap-1.5 justify-end">
                        {safeList.map(name => (
                          <span key={name} className="flex items-center gap-2 px-2.5 py-1 rounded-full bg-primary text-primary-foreground text-[9px] font-black uppercase tracking-widest shadow-sm">
                            {name}
                            <X className="h-3 w-3 cursor-pointer hover:rotate-90 transition-transform" onClick={() => toggleSafeColumn(name)} />
                          </span>
                        ))}
                      </div>
                    </div>
                  )}
                  <div className="flex justify-between text-xs px-4 py-2 bg-muted/20">
                    <span className="text-muted-foreground">Excluded features avg MI Score: <span className="font-mono text-foreground ml-1">{filter3Results.avgMIDropped.toFixed(5)}</span></span>
                    <span className="text-muted-foreground">Retained features avg MI Score: <span className="font-mono text-primary font-bold ml-1">{filter3Results.avgMIKept.toFixed(5)} ({((filter3Results.avgMIKept / (filter3Results.avgMIDropped || 0.0001)) || 1).toFixed(1)}x signal)</span></span>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>

      <div className="mt-10 flex justify-center pb-12">
        {step === 0 ? (
          <button onClick={handleRunFilters} className="flex items-center gap-2 px-8 py-4 rounded-xl bg-primary text-primary-foreground font-semibold hover:opacity-90 hover:-translate-y-1 transition-all shadow-[0_0_30px_hsl(var(--primary)/0.4)]">
            <Play className="h-5 w-5" /> Run Selection Pipeline
          </button>
        ) : step === 4 ? (
          <motion.button initial={{ scale: 0.9, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} onClick={handleCommit} className="flex flex-col items-center gap-1 px-8 py-3 rounded-xl bg-card border-2 border-primary text-primary font-bold hover:bg-primary hover:text-primary-foreground transition-all group">
            <span className="text-lg flex items-center gap-2">Commit Optimization <ChevronRight className="h-5 w-5 transition-transform group-hover:translate-x-1" /></span>
            <span className="text-xs opacity-70 font-medium">Proceed to EDA with {finalColumns} optimized features</span>
          </motion.button>
        ) : null}
      </div>
      <PageFooter nextLabel="EDA Dashboard" nextUrl="/eda" />
    </div>
  );
}
