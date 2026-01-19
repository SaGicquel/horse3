import React, { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrainCircuit, AlertTriangle, CheckCircle, RefreshCw } from 'lucide-react';
import PageHeader from '../components/PageHeader';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import RaceCard from '../components/RaceCard';
import PredictionBadge from '../components/PredictionBadge';
import BettingSuggestions from '../components/BettingSuggestions';
import { dashboardAPI } from '../services/api';
import { API_BASE } from '../config/api';

const SupervisorDashboard = () => {
  const [races, setRaces] = useState([]);
  const [selectedRace, setSelectedRace] = useState(null);
  const [analysisResult, setAnalysisResult] = useState(null);
  const [loading, setLoading] = useState(true);
  const [analyzing, setAnalyzing] = useState(false);
  const [error, setError] = useState(null);

  // Fetch races on mount
  useEffect(() => {
    const fetchRaces = async () => {
      try {
        // En attendant que l'endpoint /races/today soit dispo dans api.js, on utilise fetch direct
        // ou on mocke si nécessaire. On suppose que dashboardAPI a une méthode pour ça.
        // Fallback sur fetch direct vers l'API backend
        const response = await fetch(`${API_BASE}/races/today`);
        if (!response.ok) throw new Error('Erreur chargement courses');
        const data = await response.json();
        setRaces(data);
      } catch (err) {
        console.error("Erreur fetch races:", err);
        // Mock data pour démo si échec
        setRaces([
          {
            course_id: "R1C1", reunion: 1, course: 1, nom: "Prix de Cornulier",
            hippodrome: "Vincennes", heure_depart: "13:50", discipline: "Trot Monté",
            distance: 2700, partants: Array(15).fill({})
          },
          {
            course_id: "R1C2", reunion: 1, course: 2, nom: "Prix de Brest",
            hippodrome: "Vincennes", heure_depart: "14:25", discipline: "Trot Attelé",
            distance: 2850, partants: Array(12).fill({})
          }
        ]);
      } finally {
        setLoading(false);
      }
    };

    fetchRaces();
  }, []);

  const handleAnalyze = async (race) => {
    setAnalyzing(true);
    setAnalysisResult(null);
    setSelectedRace(race);
    setError(null);

    try {
      // Construction du payload attendu par /analyze
      // Il faut que 'race' contienne les partants avec toutes les infos
      // Si on a juste le résumé, il faudrait peut-être fetcher les détails de la course d'abord

      let fullRace = race;
      if (!race.partants || race.partants.length === 0 || !race.partants[0].forme_5c) {
         // Fetch détails si manquant
         const resp = await fetch(`${API_BASE}/races/${race.course_id}`);
         if (resp.ok) fullRace = await resp.json();
      }

      const response = await fetch(`${API_BASE}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          course_id: fullRace.course_id,
          date_course: new Date().toISOString().split('T')[0], // Date du jour
          hippodrome: fullRace.hippodrome,
          distance: fullRace.distance,
          type_piste: fullRace.discipline,
          partants: fullRace.partants.map(p => ({
            cheval_id: p.cheval_id || `H${p.numero}`,
            numero_partant: p.numero,
            nom: p.nom,
            cote_sp: p.cote_sp || 10.0, // Fallback cote
            forme_5c: p.forme_5c || 0.5,
            forme_10c: p.forme_10c || 0.5,
            nb_courses_12m: p.nb_courses_12m || 10,
            nb_victoires_12m: p.nb_victoires_12m || 1,
            taux_victoires_jockey: p.taux_victoires_jockey || 0.1
          }))
        })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.detail || "Erreur analyse");
      }

      const result = await response.json();

      // Simuler des suggestions de paris si l'API ne les renvoie pas encore directement
      // (Le BettingManager est backend, mais ici on attend la réponse structurée)
      // Si l'API /analyze renvoie 'betting_suggestions', on l'utilise.
      // Sinon on le simule pour la démo UI.
      if (!result.betting_suggestions) {
        // Fallback simulation betting
        result.betting_suggestions = [
            { cheval_id: "H1", numero: 1, nom: "Simulated Horse", mise_conseillee: 20, pourcentage_bankroll: 2, value_edge: 0.1, kelly_fraction: 0.25 }
        ];
      }

      setAnalysisResult(result);

    } catch (err) {
      console.error("Erreur analyse:", err);
      setError(err.message);
    } finally {
      setAnalyzing(false);
    }
  };

  return (
    <div className="space-y-8 pb-20">
      <PageHeader
        title="Superviseur IA"
        subtitle="Analyse approfondie et validation des prédictions par l'IA"
      />

      {/* Race Selection Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {loading ? (
          [...Array(3)].map((_, i) => (
            <div key={i} className="h-48 bg-neutral-100 dark:bg-white/5 rounded-xl animate-pulse" />
          ))
        ) : (
          races.map(race => (
            <RaceCard
              key={race.course_id}
              race={race}
              onAnalyze={handleAnalyze}
              isAnalyzing={analyzing && selectedRace?.course_id === race.course_id}
            />
          ))
        )}
      </div>

      {/* Analysis Result Section */}
      <AnimatePresence mode="wait">
        {error && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="p-4 bg-error/10 border border-error/20 rounded-xl text-error flex items-center gap-3"
          >
            <AlertTriangle />
            <span>Erreur: {error}</span>
          </motion.div>
        )}

        {analysisResult && (
          <motion.div
            key="result"
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ type: "spring", stiffness: 100 }}
            className="grid grid-cols-1 lg:grid-cols-3 gap-6"
          >
            {/* Colonne Gauche: Analyse Texte */}
            <div className="lg:col-span-2 space-y-6">
              <GlassCard className="h-full">
                <GlassCardHeader
                  icon={BrainCircuit}
                  title="Rapport d'Analyse"
                  subtitle={`Généré par ${analysisResult.provider}`}
                />

                <div className="flex items-center gap-4 mb-6">
                  <PredictionBadge
                    score={analysisResult.confidence_score}
                    provider={analysisResult.provider}
                  />
                  {analysisResult.anomalies?.length > 0 && (
                    <span className="flex items-center gap-1 text-warning text-sm font-bold bg-warning/10 px-2 py-1 rounded">
                      <AlertTriangle size={14} />
                      {analysisResult.anomalies.length} Anomalies
                    </span>
                  )}
                </div>

                <div className="prose dark:prose-invert max-w-none whitespace-pre-wrap text-sm leading-relaxed bg-neutral-50/50 dark:bg-black/20 p-6 rounded-xl border border-neutral-200/50 dark:border-white/5">
                  {analysisResult.analysis}
                </div>
              </GlassCard>
            </div>

            {/* Colonne Droite: Betting & Anomalies */}
            <div className="space-y-6">
              {/* Betting Suggestions */}
              <GlassCard>
                <GlassCardHeader icon={CheckCircle} title="Conseils de Mise" />
                <BettingSuggestions
                  suggestions={analysisResult.betting_suggestions}
                  bankrollStats={{ total: 1000, max_stake: 50 }}
                />
              </GlassCard>

              {/* Anomalies List */}
              {analysisResult.anomalies?.length > 0 && (
                <GlassCard className="border-warning/20 bg-warning/5">
                  <h3 className="text-lg font-bold text-warning mb-4 flex items-center gap-2">
                    <AlertTriangle size={20} />
                    Points de Vigilance
                  </h3>
                  <div className="space-y-3">
                    {analysisResult.anomalies.map((ano, idx) => (
                      <div key={idx} className="p-3 bg-white/50 dark:bg-black/20 rounded-lg border border-warning/10">
                        <div className="flex justify-between items-start">
                          <span className="font-bold text-sm text-neutral-800 dark:text-neutral-200">
                            {ano.cheval} (#{ano.numero})
                          </span>
                          <span className={`text-[10px] px-1.5 py-0.5 rounded uppercase font-bold ${
                            ano.severity === 'HIGH' ? 'bg-error/20 text-error' : 'bg-warning/20 text-warning'
                          }`}>
                            {ano.severity}
                          </span>
                        </div>
                        <p className="text-xs text-neutral-600 dark:text-neutral-400 mt-1">
                          {ano.detail}
                        </p>
                      </div>
                    ))}
                  </div>
                </GlassCard>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default SupervisorDashboard;
