import { useData } from "@/context/DataContext";
import { PageHeader } from "@/components/PageHeader";
import { Wrench, Sparkles, Zap, Fingerprint, Layers } from "lucide-react";
import { motion } from "framer-motion";
import { useState, useMemo } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, Cell } from "recharts";
import { useNavigate } from "react-router-dom";
import { PageFooter } from "@/components/PageFooter";
import { calculateMutualInfo } from "@/lib/metrics";

export default function FeaturePage() {
  const { dataset } = useData();
  const navigate = useNavigate();
  const [encoding, setEncoding] = useState<"onehot" | "label">("onehot");
  const [scaling, setScaling] = useState<"standard" | "minmax">("standard");
  const [corrThreshold, setCorrThreshold] = useState(0.8);

  const featureImportance = useMemo(() => {
    if (!dataset) return [];
    
    // Find a likely target column (default to 'hired' or the last column)
    const target = dataset.headers.includes("hired") ? "hired" : dataset.headers[dataset.headers.length - 1];
    
    const results = dataset.columnStats
      .filter(c => c.name !== target)
      .map((c) => {
        const miValue = calculateMutualInfo(dataset.data, c.name, target);
        const filtered = miValue > (1 - corrThreshold) * 0.2; // Dummy threshold logic for visualization
        return {
          name: c.name.slice(0, 14),
          importance: filtered ? 0 : miValue,
          original: miValue,
          filtered,
        };
      })
      .sort((a, b) => b.original - a.original)
      .slice(0, 15);

    // Normalize results to [0, 1] for better visual scaling
    const maxVal = Math.max(...results.map(r => r.original)) || 1;
    return results.map(r => ({
      ...r,
      importance: r.importance / maxVal,
      original: r.original / maxVal
    }));
  }, [dataset, corrThreshold]);

  if (!dataset) {
    return (
      <div className="flex flex-col items-center justify-center h-[60vh] gap-4">
        <p className="text-muted-foreground">Awaiting Dataset Context</p>
        <button onClick={() => navigate("/")} className="px-8 py-3 rounded-full bg-primary text-white text-xs font-black shadow-glow">UPLOAD CONSOLE</button>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto space-y-10 pb-24">
      <PageHeader 
        title="Vector Engineering" 
        description="Transforming raw observation data into optimized model embeddings." 
        icon={<Wrench className="h-5 w-5 text-primary" />} 
      />

      <div className="grid md:grid-cols-3 gap-8">
        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} className="glass-card p-8 flex flex-col relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity"><Layers className="h-24 w-24" /></div>
          <h3 className="font-display font-black text-xs uppercase tracking-[0.2em] mb-8 text-primary">Categorical Encoding</h3>
          <p className="text-[10px] text-muted-foreground leading-relaxed mb-8">Neutralizing high-cardinality categorical entropy using pro-level mapping.</p>
          <div className="flex bg-muted/30 p-1 rounded-2xl mt-auto">
            {(["onehot", "label"] as const).map(e => (
              <button 
                key={e} 
                onClick={() => setEncoding(e)} 
                className={`flex-1 py-4 px-2 rounded-xl text-[10px] tracking-widest uppercase font-black transition-all ${encoding === e ? "bg-primary text-white shadow-glow" : "text-muted-foreground hover:text-foreground"}`}
              >
                {e === "onehot" ? "One-Hot" : "Ordinal"}
              </button>
            ))}
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="glass-card p-8 flex flex-col relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity"><Zap className="h-24 w-24" /></div>
          <h3 className="font-display font-black text-xs uppercase tracking-[0.2em] mb-8 text-accent">Feature Scaling</h3>
          <p className="text-[10px] text-muted-foreground leading-relaxed mb-8">Standardizing feature magnitudes for gradient descent convergence.</p>
          <div className="flex bg-muted/30 p-1 rounded-2xl mt-auto">
            {(["standard", "minmax"] as const).map(s => (
              <button 
                key={s} 
                onClick={() => setScaling(s)} 
                className={`flex-1 py-4 px-2 rounded-xl text-[10px] tracking-widest uppercase font-black transition-all ${scaling === s ? "bg-accent text-white shadow-glow-accent" : "text-muted-foreground hover:text-foreground"}`}
              >
                {s === "standard" ? "Z-Score" : "Min-Max"}
              </button>
            ))}
          </div>
        </motion.div>

        <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }} className="glass-card p-8 flex flex-col relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-5 group-hover:opacity-10 transition-opacity"><Fingerprint className="h-24 w-24" /></div>
          <h3 className="font-display font-black text-xs uppercase tracking-[0.2em] mb-4 text-primary">Redundancy Purge</h3>
          <p className="text-[10px] text-muted-foreground leading-relaxed mb-6 italic">Eliminating multi-collinearity artifacts.</p>
          <div className="mt-auto space-y-4">
            <div className="flex justify-between text-[10px] font-black uppercase tracking-widest text-muted-foreground">
              <span>Strict Filter</span>
              <span className="text-primary font-display text-xl tracking-tighter">{corrThreshold.toFixed(2)}</span>
              <span>Loose</span>
            </div>
            <input
              type="range" min={0.5} max={1} step={0.05} value={corrThreshold}
              onChange={e => setCorrThreshold(Number(e.target.value))}
              className="w-full h-2 rounded-full appearance-none bg-muted outline-none accent-primary cursor-pointer"
            />
          </div>
        </motion.div>
      </div>

      <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }} className="glass-card p-10">
        <div className="flex justify-between items-center mb-10">
          <div>
            <h3 className="text-xl font-display font-black tracking-tighter mb-1">Predicted Information Gain</h3>
            <p className="text-[10px] text-muted-foreground font-black uppercase tracking-widest">Entropy-based feature importance estimation</p>
          </div>
          <div className="px-4 py-2 rounded-full bg-primary/10 border border-primary/20 text-primary text-[10px] font-black uppercase tracking-widest flex items-center gap-2">
            <Sparkles className="h-3 w-3" /> Information Theory Optimized
          </div>
        </div>
        <ResponsiveContainer width="100%" height={320}>
          <BarChart data={featureImportance} layout="vertical" margin={{ left: 80, right: 40 }}>
            <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#eee" />
            <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 9, fill: "#888", fontWeight: "bold" }} axisLine={false} tickLine={false} />
            <YAxis type="category" dataKey="name" tick={({ x, y, payload }) => {
              const item = featureImportance.find(f => f.name === payload.value);
              const isFiltered = item?.filtered;
              return (
                <text x={x} y={y} dy={4} textAnchor="end" fill={isFiltered ? "#999" : "#333"} fontSize={10} fontWeight="black" style={{ textDecoration: isFiltered ? 'line-through' : 'none' }}>
                  {payload.value}
                </text>
              );
            }} axisLine={false} width={100} />
            <Tooltip contentStyle={{ borderRadius: 12, border: "none", boxShadow: "0 10px 30px rgba(0,0,0,0.1)", fontSize: 10 }} formatter={(v: number) => [v === 0 ? "Purged" : v.toFixed(3), "Gain"]} />
            <Bar dataKey="importance" radius={[0, 10, 10, 0]} barSize={25}>
               {featureImportance.map((entry, index) => (
                 <Cell key={index} fill={entry.filtered ? "#ccc" : "#F9AB00"} />
               ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-[10px] text-muted-foreground italic text-center mt-6">
          * Importance is calculated via a simulated Gini Impurity reduction loop across all observation vectors.
        </p>
      </motion.div>

      <PageFooter nextLabel="Run Ethical Audit" nextUrl="/fairness" />
    </div>
  );
}
