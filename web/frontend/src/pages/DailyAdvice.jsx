import React, { useState, useEffect } from 'react';
import { GlassCard } from '../components/GlassCard';
import { Button } from '../components/Button';
import { TrendingUp, AlertTriangle, Target, Euro, Percent } from 'lucide-react';

const DailyAdvice = () => {
  const [advice, setAdvice] = useState([]);
  const [loading, setLoading] = useState(true);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);

  useEffect(() => {
    fetchDailyAdvice(selectedDate);
  }, [selectedDate]);

  const fetchDailyAdvice = async (date) => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8001/daily-advice?date_str=${date}`);
      if (response.ok) {
        const data = await response.json();
        setAdvice(data);
      } else {
        setAdvice([]);
      }
    } catch (error) {
      console.error('Erreur récupération conseils:', error);
      setAdvice([]);
    } finally {
      setLoading(false);
    }
  };

  const getProfilBadge = (profil) => {
    const variants = {
      'SÛR': { color: 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300', icon: <Target className="w-3 h-3" /> },
      'Standard': { color: 'bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300', icon: <TrendingUp className="w-3 h-3" /> },
      'Ambitieux': { color: 'bg-orange-100 dark:bg-orange-900/30 text-orange-800 dark:text-orange-300', icon: <AlertTriangle className="w-3 h-3" /> }
    };
    const variant = variants[profil] || variants['Standard'];

    return (
      <span className={`${variant.color} px-2.5 py-0.5 rounded-full text-xs font-medium flex items-center gap-1 justify-center mx-auto w-fit`}>
        {variant.icon}
        {profil}
      </span>
    );
  };

  const getValueColor = (value) => {
    if (value >= 20) return 'text-green-600 dark:text-green-400 font-bold';
    if (value >= 10) return 'text-blue-600 dark:text-blue-400';
    if (value >= 5) return 'text-orange-600 dark:text-orange-400';
    return 'text-neutral-600 dark:text-neutral-400';
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col md:flex-row justify-between items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold text-neutral-900 dark:text-neutral-100">🎯 Conseils du Jour</h1>
          <p className="text-neutral-600 dark:text-neutral-400">Modèle Champion XGBoost - Calibré & Optimisé</p>
        </div>

        <div className="flex items-center gap-4">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="px-3 py-2 bg-white dark:bg-neutral-800 border border-neutral-300 dark:border-neutral-700 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent text-neutral-900 dark:text-neutral-100"
          />
          <Button onClick={() => fetchDailyAdvice(selectedDate)} variant="primary">
            Actualiser
          </Button>
        </div>
      </div>

      {/* Stats du jour */}
      {advice.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <GlassCard className="p-4">
            <div className="flex items-center gap-2">
              <Target className="w-5 h-5 text-blue-600 dark:text-blue-400" />
              <div>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">Paris conseillés</p>
                <p className="text-2xl font-bold text-neutral-900 dark:text-neutral-100">{advice.length}</p>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="p-4">
            <div className="flex items-center gap-2">
              <Percent className="w-5 h-5 text-green-600 dark:text-green-400" />
              <div>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">Value moyenne</p>
                <p className="text-2xl font-bold text-green-600 dark:text-green-400">
                  {(advice.reduce((sum, a) => sum + a.value, 0) / advice.length).toFixed(1)}%
                </p>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="p-4">
            <div className="flex items-center gap-2">
              <Euro className="w-5 h-5 text-orange-600 dark:text-orange-400" />
              <div>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">Mise totale</p>
                <p className="text-2xl font-bold text-orange-600 dark:text-orange-400">
                  {advice.reduce((sum, a) => sum + a.mise_conseillee, 0).toFixed(2)}€
                </p>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="p-4">
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-purple-600 dark:text-purple-400" />
              <div>
                <p className="text-sm text-neutral-600 dark:text-neutral-400">EV moyenne</p>
                <p className="text-2xl font-bold text-purple-600 dark:text-purple-400">
                  {(advice.reduce((sum, a) => sum + a.ev_pct, 0) / advice.length).toFixed(1)}%
                </p>
              </div>
            </div>
          </GlassCard>
        </div>
      )}

      {/* Liste des conseils */}
      {advice.length === 0 ? (
        <GlassCard className="p-8 text-center">
          <div className="text-neutral-500 dark:text-neutral-400">
            <Target className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg">Aucun conseil pour {selectedDate}</p>
            <p className="text-sm">Vérifiez qu'il y a des courses prévues ce jour-là</p>
          </div>
        </GlassCard>
      ) : (
        <div className="space-y-4">
          {advice.map((conseil, index) => (
            <GlassCard key={`${conseil.race_key}-${conseil.cheval_id}`} className="hover:shadow-lg transition-shadow p-6">
              <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-center">

                {/* Info cheval */}
                <div className="lg:col-span-3">
                  <div className="flex items-center gap-3">
                    <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center font-bold text-sm shadow-md">
                      {conseil.numero}
                    </div>
                    <div>
                      <h3 className="font-semibold text-neutral-900 dark:text-neutral-100 capitalize">
                        {conseil.nom}
                      </h3>
                      <p className="text-sm text-neutral-600 dark:text-neutral-400">{conseil.course}</p>
                    </div>
                  </div>
                </div>

                {/* Hippodrome */}
                <div className="lg:col-span-2">
                  <p className="text-sm font-medium text-neutral-700 dark:text-neutral-300">{conseil.hippodrome}</p>
                </div>

                {/* Probabilité */}
                <div className="lg:col-span-1 text-center">
                  <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-2">
                    <p className="text-xs text-blue-600 dark:text-blue-400">Probabilité</p>
                    <p className="font-bold text-blue-900 dark:text-blue-100">{conseil.p_final.toFixed(1)}%</p>
                  </div>
                </div>

                {/* Cote */}
                <div className="lg:col-span-1 text-center">
                  <div className="bg-neutral-100 dark:bg-neutral-800 rounded-lg p-2">
                    <p className="text-xs text-neutral-600 dark:text-neutral-400">Cote</p>
                    <p className="font-bold text-neutral-900 dark:text-neutral-100">{conseil.odds.toFixed(2)}</p>
                  </div>
                </div>

                {/* Value */}
                <div className="lg:col-span-1 text-center">
                  <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-2">
                    <p className="text-xs text-green-600 dark:text-green-400">Value</p>
                    <p className={`font-bold ${getValueColor(conseil.value)}`}>
                      {conseil.value > 0 ? '+' : ''}{conseil.value.toFixed(1)}%
                    </p>
                  </div>
                </div>

                {/* Mise */}
                <div className="lg:col-span-1 text-center">
                  <div className="bg-orange-50 dark:bg-orange-900/20 rounded-lg p-2">
                    <p className="text-xs text-orange-600 dark:text-orange-400">Mise</p>
                    <p className="font-bold text-orange-900 dark:text-orange-100">{conseil.mise_conseillee.toFixed(2)}€</p>
                  </div>
                </div>

                {/* EV */}
                <div className="lg:col-span-1 text-center">
                  <div className="bg-purple-50 dark:bg-purple-900/20 rounded-lg p-2">
                    <p className="text-xs text-purple-600 dark:text-purple-400">EV</p>
                    <p className="font-bold text-purple-900 dark:text-purple-100">{conseil.ev_pct.toFixed(1)}%</p>
                  </div>
                </div>

                {/* Profil */}
                <div className="lg:col-span-2 text-center">
                  {getProfilBadge(conseil.profil)}
                </div>

              </div>
            </GlassCard>
          ))}
        </div>
      )}

      {/* Footer note */}
      <GlassCard className="border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/10 p-4">
        <div className="flex items-start gap-3">
          <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
            <span className="text-white text-xs font-bold">!</span>
          </div>
          <div className="text-sm text-blue-800 dark:text-blue-200">
            <p className="font-semibold mb-1">ℹ️ Information importante</p>
            <p>
              Les conseils sont générés par le modèle champion XGBoost avec calibration Platt (ECE: 0.0112).
              Les mises utilisent le critère de Kelly fractionné (25%) pour optimiser le rapport risque/rendement.
              <strong> Pariez toujours de manière responsable.</strong>
            </p>
          </div>
        </div>
      </GlassCard>
    </div>
  );
};

export default DailyAdvice;