import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { Wallet, TrendingUp, Trophy, PiggyBank, PlusCircle, LogOut, RefreshCw, ShieldCheck, Clock3, AlertTriangle, Search, Edit3, X, Save, Trash2, LogIn } from 'lucide-react';
import { GlassCard, GlassCardHeader } from '../components/GlassCard';
import PageHeader from '../components/PageHeader';
import { SimulationToggle, SimulationBadge, useSimulationMode } from '../components/SimulationToggle';
import { betsAPI } from '../services/api';
import { useAuth } from '../context/AuthContext';

const statusStyles = {
  WIN: { label: 'Gagné', color: 'text-emerald-700 dark:text-emerald-300', bg: 'bg-emerald-500/15', dot: 'bg-emerald-500 dark:bg-emerald-400' },
  LOSE: { label: 'Perdu', color: 'text-rose-700 dark:text-rose-300', bg: 'bg-rose-500/15', dot: 'bg-rose-500 dark:bg-rose-400' },
  PENDING: { label: 'En cours', color: 'text-amber-700 dark:text-amber-300', bg: 'bg-amber-500/15', dot: 'bg-amber-500 dark:bg-amber-400' },
  VOID: { label: 'Remboursé', color: 'text-slate-700 dark:text-slate-200', bg: 'bg-neutral-100 dark:bg-white/10', dot: 'bg-slate-500 dark:bg-slate-200' },
};

const emptySummary = {
  total_bets: 0,
  pending_bets: 0,
  finished_bets: 0,
  total_stake: 0,
  pnl_net: 0,
  roi: 0,
  win_rate: 0,
  by_status: {},
  history: [],
  bets: []
};

const Input = ({ label, type = 'text', value, onChange, placeholder, step, min }) => (
  <label className="block space-y-2">
    <span className="text-xs uppercase tracking-wide text-neutral-600 dark:text-neutral-400">{label}</span>
    <input
      type={type}
      value={value}
      onChange={onChange}
      placeholder={placeholder}
      step={step}
      min={min}
      className="w-full rounded-xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-400/40 text-neutral-900 dark:text-white"
    />
  </label>
);

const StatusBadge = ({ status }) => {
  const style = statusStyles[status] || statusStyles.PENDING;
  return (
    <span className={`inline-flex items-center gap-2 rounded-full px-3 py-1 text-xs font-semibold ${style.bg} ${style.color}`}>
      <span className={`h-2 w-2 rounded-full ${style.dot}`} />
      {style.label}
    </span>
  );
};

const formatCurrency = (value) => `${value >= 0 ? '+' : ''}${value.toFixed(2)} €`;

const HistoryLine = ({ date, pnl }) => (
  <div className="flex items-center justify-between rounded-lg border border-neutral-200 dark:border-white/5 bg-neutral-50 dark:bg-white/5 px-3 py-2">
    <div className="flex items-center gap-3">
      <div className="h-10 w-10 rounded-xl bg-gradient-to-br from-purple-500/30 to-pink-500/30 flex items-center justify-center text-xs font-semibold text-purple-800 dark:text-white">
        {date.slice(5)}
      </div>
      <div>
        <div className="text-sm font-semibold text-neutral-900 dark:text-white">{date}</div>
        <div className="text-xs text-neutral-600 dark:text-neutral-400">PNL quotidien</div>
      </div>
    </div>
    <div className={`text-sm font-semibold ${pnl >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}`}>
      {formatCurrency(pnl)}
    </div>
  </div>
);

const StatsGrid = ({ summary }) => {
  const cards = [
    { title: 'PNL net', value: formatCurrency(summary.pnl_net), icon: Wallet, color: summary.pnl_net >= 0 ? 'text-emerald-600 dark:text-emerald-300' : 'text-rose-600 dark:text-rose-300' },
    { title: 'ROI', value: `${summary.roi.toFixed(2)} %`, icon: TrendingUp, color: 'text-cyan-600 dark:text-cyan-300' },
    { title: 'Win rate', value: `${summary.win_rate.toFixed(1)} %`, icon: Trophy, color: 'text-amber-600 dark:text-amber-200' },
    { title: 'Mise totale', value: `${summary.total_stake.toFixed(2)} €`, icon: PiggyBank, color: 'text-blue-600 dark:text-blue-200' },
  ];

  return (
    <div className="grid grid-cols-2 gap-3">
      {cards.map((card) => (
        <div key={card.title} className="rounded-2xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 p-4">
          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-wide text-neutral-600 dark:text-neutral-400">{card.title}</div>
            <card.icon size={18} className={card.color} />
          </div>
          <div className="mt-2 text-xl font-semibold text-neutral-900 dark:text-white">
            {card.value}
          </div>
        </div>
      ))}
    </div>
  );
};

export default function MesParis() {
  // Utiliser le contexte global d'auth
  const { user, token, isAuthenticated, logout } = useAuth();

  // Mode simulation (stocké dans localStorage)
  const { isSimulation, toggleSimulation } = useSimulationMode();

  const [summary, setSummary] = useState(emptySummary);
  const [betsLoading, setBetsLoading] = useState(false);

  // États pour la sélection multiple et suppression
  const [selectedBets, setSelectedBets] = useState(new Set());
  const [deleteModal, setDeleteModal] = useState({ open: false, betIds: [], single: false });
  const [statusMessage, setStatusMessage] = useState('');
  const [editingBet, setEditingBet] = useState(null); // Pari en cours d'édition
  const [editForm, setEditForm] = useState(null); // Formulaire d'édition
  const [isRefreshing, setIsRefreshing] = useState(false); // État du loader
  const [betForm, setBetForm] = useState({
    race_key: '',
    event_date: '',
    hippodrome: '',
    selection: '',
    bet_type: '',
    stake: 10,
    odds: 2,
    status: 'PENDING',
    notes: '',
    is_simulation: false,  // Mode simulation
  });

  // Charger les paris quand l'utilisateur est connecté
  useEffect(() => {
    if (token) {
      refreshSummary(token);
    } else {
      setSummary(emptySummary);
    }
  }, [token]);

  const refreshSummary = async (tokenValue = token) => {
    if (!tokenValue) return;
    setBetsLoading(true);
    setStatusMessage('');
    try {
      const data = await betsAPI.summary(tokenValue);
      setSummary({
        ...emptySummary,
        ...data,
        bets: data?.bets || [],
      });
    } catch (err) {
      setStatusMessage(err.message);
    } finally {
      setBetsLoading(false);
    }
  };

  const handleCreateBet = async () => {
    if (!token) {
      setStatusMessage('Connectez-vous pour enregistrer un pari.');
      return;
    }
    if (!betForm.selection || !betForm.stake || !betForm.odds) {
      setStatusMessage('Sélection, mise et cote sont requis.');
      return;
    }
    try {
      await betsAPI.create({
        ...betForm,
        stake: Number(betForm.stake),
        odds: Number(betForm.odds),
        status: betForm.status,
        event_date: betForm.event_date || null,
      }, token);
      setStatusMessage('Pari enregistré ✅');
      setBetForm((prev) => ({ ...prev, selection: '', notes: '' }));
      refreshSummary();
    } catch (err) {
      setStatusMessage(err.message);
    }
  };

  const updateBetStatus = async (betId, status) => {
    if (!token) return;
    setStatusMessage('');
    try {
      await betsAPI.update(betId, { status }, token);
      refreshSummary();
    } catch (err) {
      setStatusMessage(err.message);
    }
  };

  const refreshBetResult = async (betId) => {
    if (!token) return;
    setStatusMessage('Vérification du résultat...');
    try {
      await betsAPI.refreshResult(betId, token);
      refreshSummary();
      setStatusMessage('Résultat mis à jour (si disponible)');
    } catch (err) {
      setStatusMessage(err.message);
    }
  };

  const refreshAllPending = async () => {
    if (!token) return;
    const allBets = summary.bets || [];
    if (allBets.length === 0) {
      setStatusMessage('Aucun pari à vérifier.');
      return;
    }
    setIsRefreshing(true); // Activer le loader
    setStatusMessage(`Vérification de ${allBets.length} paris et mise à jour des cotes...`);
    try {
      let updated = 0;
      for (const b of allBets) {
        await betsAPI.refreshResult(b.id, token);
        updated++;
        // Mettre à jour le message de progression
        if (updated % 3 === 0) {
          setStatusMessage(`Vérification en cours... ${updated}/${allBets.length}`);
        }
      }
      refreshSummary();
      setStatusMessage(`✅ ${allBets.length} paris vérifiés, cotes mises à jour.`);
    } catch (err) {
      setStatusMessage(err.message);
    } finally {
      setIsRefreshing(false); // Désactiver le loader
    }
  };

  // Ouvrir le modal d'édition
  const handleEditBet = (bet) => {
    setEditingBet(bet);
    setEditForm({
      race_key: bet.race_key || '',
      event_date: bet.event_date || '',
      hippodrome: bet.hippodrome || '',
      selection: bet.selection || '',
      bet_type: bet.bet_type || '',
      stake: bet.stake || 10,
      odds: bet.odds || 2,
      status: bet.status || 'PENDING',
      notes: bet.notes || ''
    });
  };

  // Sauvegarder les modifications
  const handleSaveEdit = async () => {
    if (!token || !editingBet) return;
    setStatusMessage('Mise à jour du pari...');
    try {
      await betsAPI.update(editingBet.id, {
        ...editForm,
        stake: Number(editForm.stake),
        odds: Number(editForm.odds),
        event_date: editForm.event_date || null,
      }, token);
      setStatusMessage('Pari modifié ✅');
      setEditingBet(null);
      setEditForm(null);
      refreshSummary();
    } catch (err) {
      setStatusMessage(err.message);
    }
  };

  // Ouvrir la modale de confirmation de suppression
  const openDeleteModal = (betIds, single = false) => {
    setDeleteModal({ open: true, betIds: Array.isArray(betIds) ? betIds : [betIds], single });
  };

  // Fermer la modale de suppression
  const closeDeleteModal = () => {
    setDeleteModal({ open: false, betIds: [], single: false });
  };

  // Supprimer les paris confirmés
  const confirmDelete = async () => {
    if (!token || deleteModal.betIds.length === 0) return;
    const count = deleteModal.betIds.length;
    setStatusMessage(`Suppression de ${count} pari${count > 1 ? 's' : ''}...`);
    closeDeleteModal();
    try {
      for (const betId of deleteModal.betIds) {
        await betsAPI.delete(betId, token);
      }
      setStatusMessage(`${count} pari${count > 1 ? 's' : ''} supprimé${count > 1 ? 's' : ''} ✅`);
      setSelectedBets(new Set()); // Réinitialiser la sélection
      refreshSummary();
    } catch (err) {
      setStatusMessage(err.message);
    }
  };

  // Supprimer un pari (ouvre la modale)
  const handleDeleteBet = (betId) => {
    if (!token) return;
    openDeleteModal(betId, true);
  };

  // Supprimer les paris sélectionnés
  const handleDeleteSelected = () => {
    if (selectedBets.size === 0) return;
    openDeleteModal(Array.from(selectedBets), false);
  };

  // Toggle sélection d'un pari
  const toggleBetSelection = (betId) => {
    setSelectedBets(prev => {
      const newSet = new Set(prev);
      if (newSet.has(betId)) {
        newSet.delete(betId);
      } else {
        newSet.add(betId);
      }
      return newSet;
    });
  };

  // Sélectionner/Désélectionner tous les paris
  const toggleSelectAll = () => {
    if (selectedBets.size === displayedBets.length) {
      setSelectedBets(new Set());
    } else {
      setSelectedBets(new Set(displayedBets.map(b => b.id)));
    }
  };

  const displayedBets = useMemo(() => {
    const bets = summary?.bets || [];
    // Filtrer par mode simulation/réel
    const filtered = bets.filter(bet => {
      const betIsSimulation = bet.is_simulation === true;
      return isSimulation ? betIsSimulation : !betIsSimulation;
    });
    return [...filtered].sort((a, b) => {
      const aDate = new Date(a.created_at || a.event_date || 0).getTime();
      const bDate = new Date(b.created_at || b.event_date || 0).getTime();
      return bDate - aDate;
    });
  }, [summary, isSimulation]);

  return (
    <>
      {/* Loader HRP avec fond flou */}
      {isRefreshing && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* Fond flou */}
          <div className="absolute inset-0 bg-black/60 backdrop-blur-md" />
          {/* Logo qui pulse */}
          <div className="relative flex flex-col items-center gap-4">
            <img
              src="/logo.png"
              alt="HRP Logo"
              className="w-24 h-24 object-contain animate-pulse drop-shadow-2xl"
              style={{
                animation: 'pulse 1.5s cubic-bezier(0.4, 0, 0.6, 1) infinite'
              }}
            />
            <div className="text-white text-lg font-semibold animate-pulse">
              Mise à jour des courses et cotes...
            </div>
            <div className="text-white/70 text-sm">
              {statusMessage}
            </div>
          </div>
        </div>
      )}

      <div className="max-w-7xl mx-auto px-3 sm:px-6 py-6 sm:py-12 space-y-6">
        {/* Header unifié */}
        <PageHeader
          emoji="💰"
          title="Mes Paris"
          subtitle="Crée ton compte, enregistre tes mises et suis automatiquement tes performances."
        >
          {user && (
            <button
              onClick={logout}
              className="inline-flex items-center gap-2 rounded-xl bg-neutral-100 dark:bg-white/10 px-4 py-2 text-sm font-semibold text-neutral-900 dark:text-white hover:bg-neutral-200 dark:hover:bg-white/20"
            >
              <LogOut size={16} />
              Déconnexion
            </button>
          )}
        </PageHeader>

        {/* ========== SIMULATION TOGGLE ========== */}
        <div className="flex items-center justify-between p-3 rounded-xl bg-slate-800/30 border border-slate-700/30">
          <div className="flex items-center gap-3">
            {isSimulation && <SimulationBadge />}
            <span className="text-sm text-neutral-400">
              {isSimulation
                ? "Affichage des paris de simulation (tests)"
                : "Affichage des paris réels"}
            </span>
          </div>
          <SimulationToggle isSimulation={isSimulation} onToggle={toggleSimulation} />
        </div>

        <div className="grid xl:grid-cols-3 gap-6">
          <GlassCard className="xl:col-span-1">
            <GlassCardHeader
              icon={Wallet}
              title={user ? 'Profil connecté' : 'Connexion / Inscription'}
              subtitle={user ? 'Session active' : 'Crée ou rejoins ton espace paris'}
            />
            {user ? (
              <div className="space-y-3">
                <div className="rounded-2xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 p-4">
                  <div className="text-sm font-semibold text-neutral-900 dark:text-white">{user.display_name || user.email}</div>
                  <div className="text-xs text-neutral-600 dark:text-neutral-400">{user.email}</div>
                </div>
                <div className="flex gap-3">
                  <button
                    onClick={() => refreshSummary()}
                    className="flex-1 inline-flex items-center justify-center gap-2 rounded-xl bg-purple-500/20 px-3 py-2 text-sm font-semibold text-white hover:bg-purple-500/30"
                  >
                    <RefreshCw size={16} /> Rafraîchir
                  </button>
                  <button
                    onClick={logout}
                    className="inline-flex items-center justify-center gap-2 rounded-xl bg-neutral-100 dark:bg-white/10 px-3 py-2 text-sm font-semibold text-neutral-900 dark:text-white hover:bg-neutral-200 dark:hover:bg-white/20"
                  >
                    <LogOut size={16} /> Quitter
                  </button>
                </div>
              </div>
            ) : (
              <div className="space-y-4">
                <div className="text-center p-4">
                  <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-purple-500/20 to-pink-500/20 flex items-center justify-center">
                    <LogIn className="w-8 h-8 text-purple-500" />
                  </div>
                  <h3 className="text-lg font-semibold text-neutral-900 dark:text-white mb-2">
                    Connexion requise
                  </h3>
                  <p className="text-sm text-neutral-600 dark:text-neutral-400 mb-4">
                    Connecte-toi pour enregistrer et suivre tes paris.
                  </p>
                  <div className="flex flex-col gap-2">
                    <Link
                      to="/login"
                      className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 px-4 py-3 text-sm font-semibold text-white hover:opacity-90"
                    >
                      <LogIn size={16} />
                      Se connecter
                    </Link>
                    <Link
                      to="/register"
                      className="text-sm text-purple-600 dark:text-purple-400 hover:underline"
                    >
                      Créer un compte
                    </Link>
                  </div>
                </div>
              </div>
            )}
          </GlassCard>

          <GlassCard className="xl:col-span-1">
            <GlassCardHeader
              icon={TrendingUp}
              title="KPIs PnL"
              subtitle="Calculés automatiquement"
            />
            <StatsGrid summary={summary} />
            <div className="mt-4 grid grid-cols-2 gap-3">
              <div className="rounded-2xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 p-3">
                <div className="text-xs uppercase tracking-wide text-neutral-600 dark:text-neutral-400">Paris ouverts</div>
                <div className="mt-1 text-lg font-semibold text-neutral-900 dark:text-white">{summary.pending_bets}</div>
              </div>
              <div className="rounded-2xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 p-3">
                <div className="text-xs uppercase tracking-wide text-neutral-600 dark:text-neutral-400">Paris terminés</div>
                <div className="mt-1 text-lg font-semibold text-neutral-900 dark:text-white">{summary.finished_bets}</div>
              </div>
            </div>
          </GlassCard>

          <GlassCard className="xl:col-span-1">
            <GlassCardHeader
              icon={Clock3}
              title="Historique PnL"
              subtitle="Derniers jours"
            />
            <div className="space-y-3 max-h-72 overflow-auto pr-1">
              {summary.history.length === 0 && (
                <div className="rounded-xl border border-neutral-200 dark:border-white/5 bg-neutral-50 dark:bg-white/5 px-3 py-2 text-sm text-neutral-600 dark:text-neutral-400">
                  Pas encore d'historique. Ajoute un premier pari !
                </div>
              )}
              {summary.history.slice(-6).reverse().map((h) => (
                <HistoryLine key={h.date} date={h.date} pnl={h.pnl} />
              ))}
            </div>
          </GlassCard>
        </div>

        <GlassCard className="space-y-6">
          <div className="flex items-center justify-between">
            <GlassCardHeader
              icon={PlusCircle}
              title="Ajouter un pari"
              subtitle="Calcul PnL automatique selon le statut"
            />
            {statusMessage && (
              <div className="rounded-full bg-neutral-100 dark:bg-white/10 px-3 py-1 text-xs text-neutral-900 dark:text-white">{statusMessage}</div>
            )}
          </div>

          {!token && (
            <div className="flex items-center gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-3 py-2 text-sm text-amber-800 dark:text-amber-100">
              <AlertTriangle size={16} />
              Connecte-toi pour enregistrer tes paris.
            </div>
          )}

          <div className="grid lg:grid-cols-3 gap-4">
            <div className="lg:col-span-1 space-y-3">
              <Input
                label="Sélection / Cheval"
                value={betForm.selection}
                onChange={(e) => setBetForm({ ...betForm, selection: e.target.value })}
                placeholder="Nom du cheval / pari"
              />
              <Input
                label="Race key (optionnel)"
                value={betForm.race_key}
                onChange={(e) => setBetForm({ ...betForm, race_key: e.target.value })}
                placeholder="2025-02-01|R1|C3|VINCENNES"
              />
              <Input
                label="Hippodrome"
                value={betForm.hippodrome}
                onChange={(e) => setBetForm({ ...betForm, hippodrome: e.target.value })}
                placeholder="Vincennes"
              />
              <Input
                label="Type de pari"
                value={betForm.bet_type}
                onChange={(e) => setBetForm({ ...betForm, bet_type: e.target.value })}
                placeholder="Simple gagnant, placé..."
              />
            </div>

            <div className="lg:col-span-1 space-y-3">
              <Input
                label="Date de course"
                type="date"
                value={betForm.event_date}
                onChange={(e) => setBetForm({ ...betForm, event_date: e.target.value })}
              />
              <div className="grid grid-cols-2 gap-3">
                <Input
                  label="Mise (€)"
                  type="number"
                  min="0"
                  step="0.1"
                  value={betForm.stake}
                  onChange={(e) => setBetForm({ ...betForm, stake: e.target.value })}
                />
                <Input
                  label="Cote"
                  type="number"
                  min="1"
                  step="0.01"
                  value={betForm.odds}
                  onChange={(e) => setBetForm({ ...betForm, odds: e.target.value })}
                />
              </div>
              <label className="block space-y-2">
                <span className="text-xs uppercase tracking-wide text-neutral-600 dark:text-neutral-400">Statut</span>
                <select
                  value={betForm.status}
                  onChange={(e) => setBetForm({ ...betForm, status: e.target.value })}
                  className="w-full rounded-xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-400/40 text-neutral-900 dark:text-white"
                >
                  <option value="PENDING">En cours</option>
                  <option value="WIN">Gagné</option>
                  <option value="LOSE">Perdu</option>
                  <option value="VOID">Remboursé</option>
                </select>
              </label>
            </div>

            <div className="lg:col-span-1 space-y-3">
              <label className="block space-y-2 h-full">
                <span className="text-xs uppercase tracking-wide text-neutral-600 dark:text-neutral-400">Notes</span>
                <textarea
                  value={betForm.notes}
                  onChange={(e) => setBetForm({ ...betForm, notes: e.target.value })}
                  rows={6}
                  className="w-full rounded-xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-400/40 text-neutral-900 dark:text-white"
                  placeholder="Plan de mise, contexte, etc."
                />
              </label>
              <button
                onClick={handleCreateBet}
                disabled={!token}
                className="w-full inline-flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 px-4 py-3 text-sm font-semibold text-white hover:opacity-90 disabled:opacity-40"
              >
                <PlusCircle size={16} />
                Ajouter le pari
              </button>
            </div>
          </div>

          <div className="border-t border-neutral-200 dark:border-white/5 pt-4 space-y-3">
            <div className="flex items-center justify-between flex-wrap gap-2">
              <div className="text-sm font-semibold text-neutral-900 dark:text-white">
                {summary.total_bets} paris enregistrés
              </div>
              <div className="flex items-center gap-2 flex-wrap">
                {/* Bouton suppression sélection */}
                {selectedBets.size > 0 && (
                  <button
                    onClick={handleDeleteSelected}
                    className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-red-500 to-rose-500 px-3 py-1.5 text-xs font-semibold text-white hover:opacity-90"
                  >
                    <Trash2 size={14} />
                    Supprimer ({selectedBets.size})
                  </button>
                )}
                <button
                  onClick={refreshAllPending}
                  disabled={!token}
                  className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-blue-500 to-cyan-500 px-3 py-1.5 text-xs font-semibold text-white hover:opacity-90 disabled:opacity-40"
                >
                  <RefreshCw size={14} />
                  Actualiser tous les paris & cotes
                </button>
                {betsLoading && <div className="text-xs text-neutral-600 dark:text-neutral-400">Chargement...</div>}
              </div>
            </div>

            <div className="overflow-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="text-left text-xs uppercase text-neutral-600 dark:text-neutral-400">
                    <th className="py-2 pr-2 w-8">
                      <input
                        type="checkbox"
                        checked={displayedBets.length > 0 && selectedBets.size === displayedBets.length}
                        onChange={toggleSelectAll}
                        className="w-4 h-4 rounded border-gray-600 bg-neutral-100 dark:bg-white/10 text-purple-500 focus:ring-purple-400 focus:ring-offset-0 cursor-pointer"
                        title="Tout sélectionner"
                      />
                    </th>
                    <th className="py-2 pr-4">Date</th>
                    <th className="py-2 pr-4">Heure</th>
                    <th className="py-2 pr-4">Sélection</th>
                    <th className="py-2 pr-4">Mise</th>
                    <th className="py-2 pr-4">Cote</th>
                    <th className="py-2 pr-4">Statut</th>
                    <th className="py-2 pr-4">PNL</th>
                    <th className="py-2">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {displayedBets.map((bet) => (
                    <tr
                      key={bet.id}
                      className={`border-t border-neutral-200 dark:border-white/5 ${selectedBets.has(bet.id) ? 'bg-purple-50 dark:bg-purple-500/10' : ''}`}
                    >
                      <td className="py-3 pr-2">
                        <input
                          type="checkbox"
                          checked={selectedBets.has(bet.id)}
                          onChange={() => toggleBetSelection(bet.id)}
                          className="w-4 h-4 rounded border-gray-600 bg-neutral-100 dark:bg-white/10 text-purple-500 focus:ring-purple-400 focus:ring-offset-0 cursor-pointer"
                        />
                      </td>
                      <td className="py-3 pr-4 text-neutral-900 dark:text-white">
                        {bet.event_date || (bet.created_at ? bet.created_at.slice(0, 10) : '—')}
                      </td>
                      <td className="py-3 pr-4 text-neutral-900 dark:text-white">
                        {bet.race_time || '—'}
                      </td>
                      <td className="py-3 pr-4 text-neutral-900 dark:text-white">
                        <div className="font-semibold">{bet.selection}</div>
                        <div className="text-xs text-neutral-600 dark:text-neutral-400">
                          {bet.bet_type || '—'} {bet.hippodrome ? `• ${bet.hippodrome}` : ''}
                        </div>
                      </td>
                      <td className="py-3 pr-4 text-neutral-900 dark:text-white">{bet.stake.toFixed(2)} €</td>
                      <td className="py-3 pr-4 text-neutral-900 dark:text-white">{bet.odds.toFixed(2)}</td>
                      <td className="py-3 pr-4"><StatusBadge status={bet.status} /></td>
                      <td className="py-3 pr-4 font-semibold">
                        <span className={bet.pnl >= 0 ? 'text-emerald-600 dark:text-emerald-400' : 'text-rose-600 dark:text-rose-400'}>
                          {formatCurrency(bet.pnl)}
                        </span>
                      </td>
                      <td className="py-3 space-x-2 whitespace-nowrap">
                        <button
                          onClick={() => handleEditBet(bet)}
                          className="rounded-lg bg-purple-100 dark:bg-purple-500/20 px-2 py-1 text-xs font-semibold text-purple-900 dark:text-purple-100 hover:bg-purple-200 dark:hover:bg-purple-500/30"
                          title="Modifier"
                        >
                          <Edit3 size={14} />
                        </button>
                        <button
                          onClick={() => updateBetStatus(bet.id, 'WIN')}
                          className="rounded-lg bg-emerald-100 dark:bg-emerald-500/20 px-2 py-1 text-xs font-semibold text-emerald-900 dark:text-emerald-100 hover:bg-emerald-200 dark:hover:bg-emerald-500/30"
                        >
                          Win
                        </button>
                        <button
                          onClick={() => updateBetStatus(bet.id, 'LOSE')}
                          className="rounded-lg bg-rose-100 dark:bg-rose-500/20 px-2 py-1 text-xs font-semibold text-rose-900 dark:text-rose-100 hover:bg-rose-200 dark:hover:bg-rose-500/30"
                        >
                          Lose
                        </button>
                        <button
                          onClick={() => updateBetStatus(bet.id, 'PENDING')}
                          className="rounded-lg bg-amber-100 dark:bg-amber-500/20 px-2 py-1 text-xs font-semibold text-amber-900 dark:text-amber-100 hover:bg-amber-200 dark:hover:bg-amber-500/30"
                        >
                          En cours
                        </button>
                        <button
                          onClick={() => refreshBetResult(bet.id)}
                          className="rounded-lg bg-blue-100 dark:bg-blue-500/20 px-2 py-1 text-xs font-semibold text-blue-900 dark:text-blue-100 hover:bg-blue-200 dark:hover:bg-blue-500/30"
                        >
                          Refresh
                        </button>
                        <button
                          onClick={() => handleDeleteBet(bet.id)}
                          className="rounded-lg bg-red-100 dark:bg-red-500/20 px-2 py-1 text-xs font-semibold text-red-900 dark:text-red-100 hover:bg-red-200 dark:hover:bg-red-500/30"
                          title="Supprimer"
                        >
                          <Trash2 size={14} />
                        </button>
                      </td>
                    </tr>
                  ))}
                  {displayedBets.length === 0 && (
                    <tr>
                      <td colSpan={7} className="py-4 text-center text-sm text-neutral-600 dark:text-neutral-400">
                        Aucun pari enregistré pour l'instant.
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </GlassCard>

        {/* Modal d'édition */}
        {editingBet && editForm && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="w-full max-w-2xl rounded-2xl border border-neutral-200 dark:border-white/10 bg-white dark:bg-slate-900/95 p-6 shadow-2xl mx-4">
              <div className="flex items-center justify-between mb-6">
                <h2 className="text-xl font-bold text-neutral-900 dark:text-white">Modifier le pari</h2>
                <button
                  onClick={() => { setEditingBet(null); setEditForm(null); }}
                  className="rounded-lg bg-neutral-100 dark:bg-white/10 p-2 hover:bg-neutral-200 dark:hover:bg-white/20"
                >
                  <X size={20} className="text-white" />
                </button>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                <Input
                  label="Sélection / Cheval"
                  value={editForm.selection}
                  onChange={(e) => setEditForm({ ...editForm, selection: e.target.value })}
                  placeholder="Nom du cheval / pari"
                />
                <Input
                  label="Race key (optionnel)"
                  value={editForm.race_key}
                  onChange={(e) => setEditForm({ ...editForm, race_key: e.target.value })}
                  placeholder="2025-02-01|R1|C3|VINCENNES"
                />
                <Input
                  label="Hippodrome"
                  value={editForm.hippodrome}
                  onChange={(e) => setEditForm({ ...editForm, hippodrome: e.target.value })}
                  placeholder="Vincennes"
                />
                <Input
                  label="Type de pari"
                  value={editForm.bet_type}
                  onChange={(e) => setEditForm({ ...editForm, bet_type: e.target.value })}
                  placeholder="Simple gagnant, placé..."
                />
                <Input
                  label="Date de course"
                  type="date"
                  value={editForm.event_date}
                  onChange={(e) => setEditForm({ ...editForm, event_date: e.target.value })}
                />
                <div className="grid grid-cols-2 gap-3">
                  <Input
                    label="Mise (€)"
                    type="number"
                    min="0"
                    step="0.1"
                    value={editForm.stake}
                    onChange={(e) => setEditForm({ ...editForm, stake: e.target.value })}
                  />
                  <Input
                    label="Cote"
                    type="number"
                    min="1"
                    step="0.01"
                    value={editForm.odds}
                    onChange={(e) => setEditForm({ ...editForm, odds: e.target.value })}
                  />
                </div>
                <label className="block space-y-2">
                  <span className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-muted)' }}>Statut</span>
                  <select
                    value={editForm.status}
                    onChange={(e) => setEditForm({ ...editForm, status: e.target.value })}
                    className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-400/40"
                    style={{ color: 'var(--color-text)' }}
                  >
                    <option value="PENDING">En cours</option>
                    <option value="WIN">Gagné</option>
                    <option value="LOSE">Perdu</option>
                    <option value="VOID">Remboursé</option>
                  </select>
                </label>
                <label className="block space-y-2 md:col-span-2">
                  <span className="text-xs uppercase tracking-wide" style={{ color: 'var(--color-muted)' }}>Notes</span>
                  <textarea
                    value={editForm.notes}
                    onChange={(e) => setEditForm({ ...editForm, notes: e.target.value })}
                    rows={3}
                    className="w-full rounded-xl border border-white/10 bg-white/5 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-purple-400/40"
                    placeholder="Plan de mise, contexte, etc."
                    style={{ color: 'var(--color-text)' }}
                  />
                </label>
              </div>

              <div className="flex justify-end gap-3 mt-6">
                <button
                  onClick={() => { setEditingBet(null); setEditForm(null); }}
                  className="inline-flex items-center gap-2 rounded-xl bg-white/10 px-4 py-2 text-sm font-semibold text-white hover:bg-white/20"
                >
                  <X size={16} /> Annuler
                </button>
                <button
                  onClick={handleSaveEdit}
                  className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
                >
                  <Save size={16} /> Enregistrer
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Modal de confirmation de suppression */}
        {deleteModal.open && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <div className="w-full max-w-md rounded-2xl border border-white/10 bg-slate-900/95 p-6 shadow-2xl mx-4">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-3 rounded-full bg-red-500/20">
                  <AlertTriangle size={24} className="text-red-400" />
                </div>
                <h2 className="text-xl font-bold text-white">Confirmer la suppression</h2>
              </div>

              <p className="text-gray-300 mb-6">
                {deleteModal.single
                  ? "Êtes-vous sûr de vouloir supprimer ce pari ? Cette action est irréversible."
                  : `Êtes-vous sûr de vouloir supprimer ${deleteModal.betIds.length} paris ? Cette action est irréversible.`
                }
              </p>

              <div className="flex justify-end gap-3">
                <button
                  onClick={closeDeleteModal}
                  className="inline-flex items-center gap-2 rounded-xl bg-white/10 px-4 py-2 text-sm font-semibold text-white hover:bg-white/20"
                >
                  <X size={16} /> Annuler
                </button>
                <button
                  onClick={confirmDelete}
                  className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-red-500 to-rose-500 px-4 py-2 text-sm font-semibold text-white hover:opacity-90"
                >
                  <Trash2 size={16} /> Supprimer
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
