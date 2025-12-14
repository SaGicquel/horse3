/**
 * Register.jsx - Page d'inscription
 * Design glassmorphism cohérent avec l'application
 */

import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { UserPlus, Mail, Lock, User, AlertCircle, Loader2, CheckCircle } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { GlassCard } from '../components/GlassCard';

export default function Register() {
    const navigate = useNavigate();
    const { register, isAuthenticated, isLoading, error, clearError } = useAuth();

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
    const [displayName, setDisplayName] = useState('');
    const [localError, setLocalError] = useState('');
    const [submitting, setSubmitting] = useState(false);

    // Rediriger si déjà connecté
    useEffect(() => {
        if (isAuthenticated) {
            navigate('/dashboard', { replace: true });
        }
    }, [isAuthenticated, navigate]);

    // Effacer l'erreur quand les champs changent
    useEffect(() => {
        if (localError || error) {
            setLocalError('');
            clearError();
        }
    }, [email, password, displayName]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLocalError('');

        // Validation
        if (!email.trim()) {
            setLocalError('Email requis');
            return;
        }
        if (!password) {
            setLocalError('Mot de passe requis');
            return;
        }
        if (password.length < 6) {
            setLocalError('Le mot de passe doit contenir au moins 6 caractères');
            return;
        }

        setSubmitting(true);
        const result = await register(email.trim(), password, displayName);
        setSubmitting(false);

        if (result.success) {
            navigate('/dashboard', { replace: true });
        } else {
            setLocalError(result.error);
        }
    };

    const displayError = localError || error;

    // Indicateur de force du mot de passe
    const passwordStrength = password.length >= 8 ? 'fort' : password.length >= 6 ? 'ok' : 'faible';
    const strengthColor = {
        'fort': 'text-emerald-500',
        'ok': 'text-amber-500',
        'faible': 'text-rose-500',
    }[passwordStrength];

    return (
        <div className="min-h-screen flex items-center justify-center p-4">
            <motion.div
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.4, ease: 'easeOut' }}
                className="w-full max-w-md"
            >
                <GlassCard className="p-8">
                    {/* Header */}
                    <div className="text-center mb-8">
                        <motion.div
                            initial={{ scale: 0 }}
                            animate={{ scale: 1 }}
                            transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                            className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-emerald-500 to-teal-500 flex items-center justify-center"
                        >
                            <UserPlus className="w-8 h-8 text-white" />
                        </motion.div>
                        <h1 className="text-2xl font-bold text-neutral-900 dark:text-white">
                            Créer un compte
                        </h1>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-2">
                            Rejoins HorseRace Predictor gratuitement
                        </p>
                    </div>

                    {/* Formulaire */}
                    <form onSubmit={handleSubmit} className="space-y-5">
                        {/* Email */}
                        <div>
                            <label className="block text-xs uppercase tracking-wide text-neutral-600 dark:text-neutral-400 mb-2">
                                Email
                            </label>
                            <div className="relative">
                                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400" />
                                <input
                                    type="email"
                                    value={email}
                                    onChange={(e) => setEmail(e.target.value)}
                                    placeholder="you@example.com"
                                    className="w-full pl-11 pr-4 py-3 rounded-xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-emerald-400/50"
                                    disabled={submitting}
                                />
                            </div>
                        </div>

                        {/* Pseudo */}
                        <div>
                            <label className="block text-xs uppercase tracking-wide text-neutral-600 dark:text-neutral-400 mb-2">
                                Pseudo <span className="text-neutral-400">(optionnel)</span>
                            </label>
                            <div className="relative">
                                <User className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400" />
                                <input
                                    type="text"
                                    value={displayName}
                                    onChange={(e) => setDisplayName(e.target.value)}
                                    placeholder="Mon pseudo"
                                    className="w-full pl-11 pr-4 py-3 rounded-xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-emerald-400/50"
                                    disabled={submitting}
                                />
                            </div>
                        </div>

                        {/* Mot de passe */}
                        <div>
                            <label className="block text-xs uppercase tracking-wide text-neutral-600 dark:text-neutral-400 mb-2">
                                Mot de passe
                            </label>
                            <div className="relative">
                                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-5 h-5 text-neutral-400" />
                                <input
                                    type="password"
                                    value={password}
                                    onChange={(e) => setPassword(e.target.value)}
                                    placeholder="••••••••"
                                    className="w-full pl-11 pr-4 py-3 rounded-xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-emerald-400/50"
                                    disabled={submitting}
                                />
                            </div>
                            {password && (
                                <div className="mt-2 flex items-center gap-2 text-xs">
                                    <CheckCircle className={`w-4 h-4 ${strengthColor}`} />
                                    <span className={strengthColor}>
                                        Force : {passwordStrength}
                                    </span>
                                    <span className="text-neutral-400">
                                        (min. 6 caractères)
                                    </span>
                                </div>
                            )}
                        </div>

                        {/* Erreur */}
                        {displayError && (
                            <motion.div
                                initial={{ opacity: 0, y: -10 }}
                                animate={{ opacity: 1, y: 0 }}
                                className="flex items-center gap-2 p-3 rounded-xl bg-rose-500/10 border border-rose-500/30 text-rose-700 dark:text-rose-300 text-sm"
                            >
                                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                <span>{displayError}</span>
                            </motion.div>
                        )}

                        {/* Bouton Submit */}
                        <button
                            type="submit"
                            disabled={submitting || isLoading}
                            className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 text-white font-semibold hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                        >
                            {submitting ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Création...
                                </>
                            ) : (
                                <>
                                    <UserPlus className="w-5 h-5" />
                                    Créer mon compte
                                </>
                            )}
                        </button>
                    </form>

                    {/* Lien connexion */}
                    <div className="mt-6 text-center text-sm text-neutral-600 dark:text-neutral-400">
                        Déjà un compte ?{' '}
                        <Link
                            to="/login"
                            className="text-emerald-600 dark:text-emerald-400 font-medium hover:underline"
                        >
                            Se connecter
                        </Link>
                    </div>
                </GlassCard>

                {/* Retour au dashboard */}
                <div className="mt-4 text-center">
                    <Link
                        to="/dashboard"
                        className="text-sm text-neutral-500 dark:text-neutral-400 hover:text-neutral-700 dark:hover:text-neutral-200"
                    >
                        ← Retour au dashboard
                    </Link>
                </div>
            </motion.div>
        </div>
    );
}
