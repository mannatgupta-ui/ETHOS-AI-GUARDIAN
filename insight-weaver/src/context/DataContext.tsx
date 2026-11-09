import React, { createContext, useContext, useState, ReactNode } from "react";
import type { PPIResult } from "@/lib/ppiDebias";

export interface ColumnStats {
  name: string;
  type: "numeric" | "categorical" | "datetime" | "boolean";
  missing: number;
  missingPct: number;
  unique: number;
  mean?: number;
  median?: number;
  std?: number;
  min?: number;
  max?: number;
  topValues?: { value: string; count: number }[];
  relevanceScore?: number;
}

export interface DatasetInfo {
  fileName: string;
  rows: number;
  columns: number;
  missingTotal: number;
  missingPct: number;
  duplicateRows: number;
  numericCols: number;
  categoricalCols: number;
  columnStats: ColumnStats[];
  data: Record<string, any>[];
  headers: string[];
}

export interface BoostedMetrics {
  before: number;
  after: number;
  precision: number;
  recall: number;
  f1: number;
  logs: string[];
}

interface DataContextType {
  dataset: DatasetInfo | null;
  setDataset: (d: DatasetInfo | null) => void;
  debiasedDataset: DatasetInfo | null;
  setDebiasedDataset: (d: DatasetInfo | null) => void;
  targetColumn: string | null;
  setTargetColumn: (c: string | null) => void;
  sensitiveColumn: string | null;
  setSensitiveColumn: (c: string | null) => void;
  sensitiveColumns: string[];
  setSensitiveColumns: (cols: string[]) => void;
  safeList: string[];
  setSafeList: (list: string[]) => void;
  toggleSafeColumn: (column: string) => void;
  darkMode: boolean;
  setDarkMode: (d: boolean) => void;
  modelResults: any | null;
  setModelResults: (r: any | null) => void;
  boostedMetrics: BoostedMetrics | null;
  setBoostedMetrics: (m: BoostedMetrics | null) => void;
  ppiResult: PPIResult | null;
  setPPIResult: (r: PPIResult | null) => void;
  ppiDebiasedDataset: DatasetInfo | null;
  setPPIDebiasedDataset: (d: DatasetInfo | null) => void;

  // PERSISTED ANALYSIS HISTORY
  fairnessLogs: string[];
  setFairnessLogs: (l: string[] | ((prev: string[]) => string[])) => void;
  scanComplete: boolean;
  setScanComplete: (b: boolean) => void;
  modelLogs: string[];
  setModelLogs: (l: string[] | ((prev: string[]) => string[])) => void;
  selectionStep: number;
  setSelectionStep: (s: number) => void;
  datasetDescription: string | null;
  setDatasetDescription: (desc: string | null) => void;
  boosterLogs: string[];
  setBoosterLogs: (l: string[] | ((prev: string[]) => string[])) => void;
}

const DataContext = createContext<DataContextType | undefined>(undefined);

export const useData = () => {
  const ctx = useContext(DataContext);
  if (!ctx) throw new Error("useData must be used within DataProvider");
  return ctx;
};

export const DataProvider = ({ children }: { children: ReactNode }) => {
  const [dataset, setDataset] = useState<DatasetInfo | null>(null);
  const [debiasedDataset, setDebiasedDataset] = useState<DatasetInfo | null>(null);
  const [targetColumn, setTargetColumn] = useState<string | null>(null);
  const [sensitiveColumns, setSensitiveColumnsState] = useState<string[]>([]);
  const [sensitiveColumn, setSensitiveColumnState] = useState<string | null>(null);
  const [datasetDescription, setDatasetDescription] = useState<string | null>(null);
  const [boosterLogs, setBoosterLogs] = useState<string[]>([]);

  const setSensitiveColumn = (c: string | null) => {
    setSensitiveColumnState(c);
    setSensitiveColumnsState(c ? [c] : []);
  };

  const setSensitiveColumns = (cols: string[]) => {
    setSensitiveColumnsState(cols);
    setSensitiveColumnState(cols[0] || null);
  };

  const [safeList, setSafeList] = useState<string[]>([]);
  const [darkMode, setDarkMode] = useState(true);
  const [modelResults, setModelResults] = useState<any | null>(null);
  const [boostedMetrics, setBoostedMetrics] = useState<BoostedMetrics | null>(null);
  const [ppiResult, setPPIResult] = useState<PPIResult | null>(null);
  const [ppiDebiasedDataset, setPPIDebiasedDataset] = useState<DatasetInfo | null>(null);

  // HISTORY STATE
  const [fairnessLogs, setFairnessLogs] = useState<string[]>([]);
  const [scanComplete, setScanComplete] = useState(false);
  const [modelLogs, setModelLogs] = useState<string[]>([]);
  const [selectionStep, setSelectionStep] = useState(0);

  const toggleSafeColumn = (column: string) => {
    setSafeList(prev => 
      prev.includes(column) 
        ? prev.filter(c => c !== column) 
        : [...prev, column]
    );
  };

  const handleSetDataset = (d: DatasetInfo | null) => {
    setDataset(d);
    // Reset history only on new dataset
    if (d) {
      setFairnessLogs([]);
      setScanComplete(false);
      setModelLogs([]);
      setSelectionStep(0);
      setDebiasedDataset(null);
      setModelResults(null);
      setBoostedMetrics(null);
      setPPIResult(null);
      setPPIDebiasedDataset(null);
      setSensitiveColumnsState([]);
      setSensitiveColumnState(null);
      setDatasetDescription(null);
      setBoosterLogs([]);
    }
  };

  return (
    <DataContext.Provider value={{ 
      dataset, setDataset: handleSetDataset, 
      debiasedDataset, setDebiasedDataset,
      targetColumn, setTargetColumn, 
      sensitiveColumn, setSensitiveColumn,
      sensitiveColumns, setSensitiveColumns,
      safeList, setSafeList, toggleSafeColumn,
      darkMode, setDarkMode,
      modelResults, setModelResults,
      boostedMetrics, setBoostedMetrics,
      ppiResult, setPPIResult,
      ppiDebiasedDataset, setPPIDebiasedDataset,
      fairnessLogs, setFairnessLogs,
      scanComplete, setScanComplete,
      modelLogs, setModelLogs,
      selectionStep, setSelectionStep,
      datasetDescription, setDatasetDescription,
      boosterLogs, setBoosterLogs
    }}>
      {children}
    </DataContext.Provider>
  );
};
