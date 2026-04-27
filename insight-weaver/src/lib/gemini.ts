import { GoogleGenerativeAI } from "@google/generative-ai";

const API_KEY = import.meta.env.VITE_GEMINI_API_KEY;
const genAI = new GoogleGenerativeAI(API_KEY);

export async function generateBiasReport(metrics: any, fairnessStats: any) {
  try {
    const model = genAI.getGenerativeModel({ model: "gemini-1.5-flash" });

    const prompt = `
      You are an Ethical AI Auditor. Analyze the following model results and fairness metrics:
      
      MODEL PERFORMANCE:
      - Accuracy: ${(metrics.accuracy * 100).toFixed(2)}%
      - Precision: ${(metrics.precision * 100).toFixed(2)}%
      - F1 Score: ${(metrics.f1 * 100).toFixed(2)}%
      
      FAIRNESS METRICS:
      - Statistical Parity Difference (SPD): ${fairnessStats?.debiased?.spd?.toFixed(4) || "N/A"}
      - Disparate Impact (DI): ${fairnessStats?.debiased?.di?.toFixed(4) || "N/A"}
      - Wasserstein Distance: ${fairnessStats?.debiased?.wasserstein?.toFixed(4) || "N/A"}
      
      TASK:
      Provide a professional, concise "Bias Mitigation Report" (max 200 words). 
      Include:
      1. A "Verdict" on compliance (e.g., compliant with 80% rule).
      2. One specific technical observation about the accuracy-fairness trade-off.
      3. A recommendation for further optimization.
      
      Format with professional, technical language suitable for a high-stakes AI hackathon. Use markdown.
    `;

    const result = await model.generateContent(prompt);
    const response = await result.response;
    return response.text();
  } catch (error) {
    console.error("Gemini Error:", error);
    return "Error generating AI report. Please verify your API key and network connection.";
  }
}
