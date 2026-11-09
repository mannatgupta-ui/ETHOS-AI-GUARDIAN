import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { Toaster } from "@/components/ui/toaster";
import { TooltipProvider } from "@/components/ui/tooltip";
import { DataProvider, useData } from "@/context/DataContext";
import { DashboardLayout } from "@/components/DashboardLayout";
import UploadPage from "./pages/UploadPage";
import OverviewPage from "./pages/OverviewPage";
import FeatureSelectionPage from "./pages/FeatureSelectionPage";
import EdaPage from "./pages/EdaPage";
import ImputationPage from "./pages/ImputationPage";
import FeaturePage from "./pages/FeaturePage";
import ModelPage from "./pages/ModelPage";
import FairnessPage from "./pages/FairnessPage";
import ResultsPage from "./pages/ResultsPage";
import NotFound from "./pages/NotFound";
import PPIDebiasPage from "./pages/PPIDebiasPage";
import { useEffect } from "react";

const queryClient = new QueryClient();

function ThemeSync({ children }: { children: React.ReactNode }) {
  const { darkMode } = useData();
  useEffect(() => {
    document.documentElement.classList.toggle("dark", darkMode);
  }, [darkMode]);
  return <>{children}</>;
}

const App = () => (
  <QueryClientProvider client={queryClient}>
    <DataProvider>
      <ThemeSync>
        <TooltipProvider>
          <Toaster />
          <Sonner />
          <BrowserRouter>
            <DashboardLayout>
              <Routes>
                <Route path="/" element={<UploadPage />} />
                <Route path="/overview" element={<OverviewPage />} />
                <Route path="/selection" element={<FeatureSelectionPage />} />
                <Route path="/eda" element={<EdaPage />} />
                <Route path="/imputation" element={<ImputationPage />} />
                <Route path="/features" element={<FeaturePage />} />
                <Route path="/model" element={<ModelPage />} />
                <Route path="/fairness" element={<FairnessPage />} />
                <Route path="/ppi" element={<PPIDebiasPage />} />
                <Route path="/results" element={<ResultsPage />} />
                <Route path="*" element={<NotFound />} />
              </Routes>
            </DashboardLayout>
          </BrowserRouter>
        </TooltipProvider>
      </ThemeSync>
    </DataProvider>
  </QueryClientProvider>
);

export default App;
