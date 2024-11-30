import { useState } from 'react';
import api from '../config/api';
import { PredictionInput, PredictionResult } from '../types';

export const useLifeExpectancy = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResult | null>(null);

  const predict = async (data: PredictionInput) => {
    try {
      setLoading(true);
      setError(null);
      const response = await api.post('/predict/life-expectancy/', data);
      setResult(response.data);
    } catch (err) {
      setError('Failed to make prediction. Please try again.');
      console.error('Prediction error:', err);
    } finally {
      setLoading(false);
    }
  };

  return { predict, loading, error, result };
}; 