export interface SoilData {
  N: number;
  P: number;
  K: number;
  pH: number;
  EC: number;
  OC: number;
  S: number;
  Zn: number;
  Fe: number;
  Cu: number;
  Mn: number;
  B: number;
}

export interface SoilHealthBreakdown {
  score: number;
  weight: number;
  value: number;
}

export interface SoilHealthScore {
  overall_score: number;
  grade: 'A' | 'B' | 'C' | 'D';
  breakdown: Record<string, SoilHealthBreakdown>;
}

export interface CropRecommendation {
  name: string;
  category: string;
  suitabilityScore: number;
  matchedParameters: string[];
  potentialChallenges: string[];
  priority: 'Excellent' | 'Good' | 'Fair';
  plantingSeasons: string[];
  harvestMonths: number;
}

export interface PredictionResult {
  analysis_id: string;
  timestamp: string;
  suitability: 'High' | 'Medium' | 'Low';
  confidence: string;
  confidenceScore: number;
  probabilities: Record<string, number>;
  keyFactors: string[];
  deficiencies: string[];
  strengths: string[];
  recommendations: string[];
  summary: string;
  recommendedCrops: CropRecommendation[];
  shap_explanation: Record<string, number>;
  soil_health_score: SoilHealthScore;
  cached?: boolean;
}

export interface Toast {
  id: number;
  message: string;
  type: 'success' | 'error' | 'info';
}

export interface AnalysisHistory {
  id: number;
  date?: string;
  timestamp?: string;
  soilData?: Record<string, string>;
  soil_data?: Record<string, number>;
  result?: PredictionResult;
  suitability?: string;
  health_score?: number;
}

export interface FertilizerRecommendation {
  nutrientId: string;
  fertilizer: string;
  composition: string;
  dosage: string;
  cost: string;
  type: string;
  priority: 'High' | 'Medium';
}

export interface SoilParam {
  id: string;
  unit: string;
  range: { min: number; max: number };
  optimal: { min: number; max: number };
  placeholder: string;
  isPrimary?: boolean;
}

export type Lang = 'en' | 'fr';
export type Theme = 'dark' | 'light';
