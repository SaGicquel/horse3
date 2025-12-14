import React, { useState, useEffect } from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../components/ui/card';
import { Button } from '../components/ui/button';
import { Wallet, TrendingUp, AlertTriangle, DollarSign, Percent, Target, Settings } from 'lucide-react';

const Portfolio = () => {
  const [portfolio, setPortfolio] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedDate, setSelectedDate] = useState(new Date().toISOString().split('T')[0]);
  const [showBankrollModal, setShowBankrollModal] = useState(false);
  const [newBankroll, setNewBankroll] = useState('');

  useEffect(() => {
    fetchPortfolio(selectedDate);
  }, [selectedDate]);

  const fetchPortfolio = async (date) => {
    try {
      setLoading(true);
      const response = await fetch(`http://localhost:8001/portfolio?date_str=${date}`);
      if (response.ok) {
        const data = await response.json();
        setPortfolio(data);
      } else {
        setPortfolio(null);
      }
    } catch (error) {
      console.error('Erreur récupération portefeuille:', error);
      setPortfolio(null);
    } finally {
      setLoading(false);
    }
  };

  const updateBankroll = async () => {
    try {
      const response = await fetch(`http://localhost:8001/update-bankroll?bankroll=${newBankroll}`, {
        method: 'POST'
      });
      if (response.ok) {
        setShowBankrollModal(false);
        setNewBankroll('');
        // Refresh portfolio
        fetchPortfolio(selectedDate);
      }
    } catch (error) {
      console.error('Erreur mise à jour bankroll:', error);
    }
  };

  const getRiskColor = (risque) => {
    if (risque <= 10) return 'text-green-600';
    if (risque <= 25) return 'text-orange-600';
    return 'text-red-600';
  };

  const getRiskBadge = (risque) => {
    if (risque <= 10) return { text: 'Risque Faible', color: 'bg-green-100 text-green-800' };
    if (risque <= 25) return { text: 'Risque Modéré', color: 'bg-orange-100 text-orange-800' };
    return { text: 'Risque Élevé', color: 'bg-red-100 text-red-800' };
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
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">💼 Portefeuille</h1>
          <p className="text-gray-600">Gestion des mises et du risque quotidien</p>
        </div>
        
        <div className="flex items-center gap-4">
          <input
            type="date"
            value={selectedDate}
            onChange={(e) => setSelectedDate(e.target.value)}
            className="px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
          <Button 
            variant="outline" 
            onClick={() => setShowBankrollModal(true)}
            className="flex items-center gap-2"
          >
            <Settings className="w-4 h-4" />
            Configurer
          </Button>
        </div>
      </div>

      {portfolio ? (
        <>
          {/* Vue d'ensemble du portefeuille */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
            
            {/* Bankroll de référence */}
            <Card className="border-blue-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-blue-600 flex items-center gap-2">
                  <Wallet className="w-4 h-4" />
                  Bankroll de Référence
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-blue-900">
                  {portfolio.bankroll_reference.toLocaleString('fr-FR')} €
                </div>
                <p className="text-xs text-blue-600 mt-1">Capital disponible</p>
              </CardContent>
            </Card>

            {/* Mise totale */}
            <Card className="border-orange-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-orange-600 flex items-center gap-2">
                  <DollarSign className="w-4 h-4" />
                  Mise Totale du Jour
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-orange-900">
                  {portfolio.mise_totale.toLocaleString('fr-FR')} €
                </div>
                <p className="text-xs text-orange-600 mt-1">{portfolio.nombre_paris} paris conseillés</p>
              </CardContent>
            </Card>

            {/* Risque */}
            <Card className="border-gray-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-gray-600 flex items-center gap-2">
                  <Percent className="w-4 h-4" />
                  Risque du Jour
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${getRiskColor(portfolio.risque_pct)}`}>
                  {portfolio.risque_pct.toFixed(1)} %
                </div>
                <div className="mt-2">
                  <span className={`px-2 py-1 rounded-full text-xs font-medium ${getRiskBadge(portfolio.risque_pct).color}`}>
                    {getRiskBadge(portfolio.risque_pct).text}
                  </span>
                </div>
              </CardContent>
            </Card>

            {/* Capital résiduel */}
            <Card className="border-green-200">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-medium text-green-600 flex items-center gap-2">
                  <TrendingUp className="w-4 h-4" />
                  Capital Résiduel
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold text-green-900">
                  {(portfolio.bankroll_reference - portfolio.mise_totale).toLocaleString('fr-FR')} €
                </div>
                <p className="text-xs text-green-600 mt-1">
                  {((1 - portfolio.risque_pct / 100) * 100).toFixed(1)}% disponible
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Alerte risque */}
          {portfolio.risque_pct > 25 && (
            <Card className="border-red-200 bg-red-50">
              <CardContent className="p-4">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-red-600 mt-0.5" />
                  <div>
                    <h3 className="font-semibold text-red-800">⚠️ Attention - Risque Élevé</h3>
                    <p className="text-red-700 text-sm mt-1">
                      Votre exposition représente {portfolio.risque_pct.toFixed(1)}% de votre bankroll. 
                      Considérez réduire les mises ou augmenter votre capital de référence.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}

          {/* Détail des paris */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Target className="w-5 h-5" />
                Détail des Paris ({portfolio.nombre_paris})
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="space-y-3">
                {portfolio.paris_details.map((pari, index) => (
                  <div key={`${pari.race_key}-${pari.cheval_id}`} 
                       className="flex items-center justify-between p-4 bg-gray-50 rounded-lg">
                    
                    <div className="flex items-center gap-4">
                      <div className="w-8 h-8 bg-blue-600 text-white rounded-full flex items-center justify-center text-sm font-bold">
                        {pari.numero}
                      </div>
                      <div>
                        <p className="font-medium capitalize">{pari.nom}</p>
                        <p className="text-sm text-gray-600">{pari.hippodrome} - {pari.course}</p>
                      </div>
                    </div>

                    <div className="flex items-center gap-6 text-sm">
                      <div className="text-center">
                        <p className="text-gray-600">Probabilité</p>
                        <p className="font-semibold">{pari.p_final.toFixed(1)}%</p>
                      </div>
                      <div className="text-center">
                        <p className="text-gray-600">Cote</p>
                        <p className="font-semibold">{pari.odds.toFixed(2)}</p>
                      </div>
                      <div className="text-center">
                        <p className="text-gray-600">Value</p>
                        <p className="font-semibold text-green-600">+{pari.value.toFixed(1)}%</p>
                      </div>
                      <div className="text-center">
                        <p className="text-gray-600">Mise</p>
                        <p className="font-bold text-lg">{pari.mise_conseillee.toFixed(2)} €</p>
                      </div>
                      <div className="text-center">
                        <span className={`px-2 py-1 rounded-full text-xs font-medium ${
                          pari.profil === 'SÛR' ? 'bg-green-100 text-green-800' :
                          pari.profil === 'Standard' ? 'bg-blue-100 text-blue-800' :
                          'bg-orange-100 text-orange-800'
                        }`}>
                          {pari.profil}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>

          {/* Conseils de gestion */}
          <Card className="border-blue-200 bg-blue-50">
            <CardContent className="p-4">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 bg-blue-600 rounded-full flex items-center justify-center flex-shrink-0 mt-1">
                  <span className="text-white text-xs font-bold">💡</span>
                </div>
                <div className="text-sm text-blue-800">
                  <p className="font-semibold mb-2">💡 Conseils de Gestion</p>
                  <ul className="space-y-1 text-xs">
                    <li>• <strong>Risque optimal :</strong> Maintenez votre exposition entre 5-15% de votre bankroll</li>
                    <li>• <strong>Kelly fractionné :</strong> Les mises utilisent 25% du Kelly pour limiter la variance</li>
                    <li>• <strong>Diversification :</strong> Répartissez vos paris sur plusieurs courses et hippodromes</li>
                    <li>• <strong>Suivi :</strong> Documentez tous vos paris pour analyser vos performances</li>
                  </ul>
                </div>
              </div>
            </CardContent>
          </Card>

        </>
      ) : (
        <Card>
          <CardContent className="p-8 text-center">
            <div className="text-gray-500">
              <Wallet className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Aucun portefeuille pour {selectedDate}</p>
              <p className="text-sm">Vérifiez qu'il y a des conseils générés pour cette date</p>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Modal configuration bankroll */}
      {showBankrollModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white rounded-lg p-6 max-w-md w-full">
            <h3 className="text-lg font-semibold mb-4">Configurer la Bankroll</h3>
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Bankroll de référence (€)
              </label>
              <input
                type="number"
                value={newBankroll}
                onChange={(e) => setNewBankroll(e.target.value)}
                placeholder="Ex: 1000"
                className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              />
              <p className="text-xs text-gray-500 mt-1">
                Capital total que vous consacrez aux paris hippiques
              </p>
            </div>
            <div className="flex gap-3">
              <Button 
                onClick={updateBankroll}
                disabled={!newBankroll || parseFloat(newBankroll) <= 0}
                className="flex-1"
              >
                Confirmer
              </Button>
              <Button 
                variant="outline" 
                onClick={() => {setShowBankrollModal(false); setNewBankroll('');}}
                className="flex-1"
              >
                Annuler
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Portfolio;