/**
 * Login.jsx - Page de connexion
 * Design glassmorphism cohérent avec l'application
 */

import { useState, useEffect } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { LogIn, Mail, Lock, AlertCircle, Loader2 } from 'lucide-react';
import { useAuth } from '../context/AuthContext';
import { GlassCard } from '../components/GlassCard';

export default function Login() {
    const navigate = useNavigate();
    const { login, isAuthenticated, isLoading, error, clearError } = useAuth();

    const [email, setEmail] = useState('');
    const [password, setPassword] = useState('');
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
    }, [email, password]);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLocalError('');

        if (!email.trim()) {
            setLocalError('Email requis');
            return;
        }
        if (!password) {
            setLocalError('Mot de passe requis');
            return;
        }

        setSubmitting(true);
        const result = await login(email.trim(), password);
        setSubmitting(false);

        if (result.success) {
            navigate('/dashboard', { replace: true });
        } else {
            setLocalError(result.error);
        }
    };

    const displayError = localError || error;

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
                            className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-gradient-to-br from-purple-500 to-pink-500 flex items-center justify-center"
                        >
                            <LogIn className="w-8 h-8 text-white" />
                        </motion.div>
                        <h1 className="text-2xl font-bold text-neutral-900 dark:text-white">
                            Connexion
                        </h1>
                        <p className="text-sm text-neutral-600 dark:text-neutral-400 mt-2">
                            Accède à ton espace HorseRace Predictor
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
                                    className="w-full pl-11 pr-4 py-3 rounded-xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-purple-400/50"
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
                                    className="w-full pl-11 pr-4 py-3 rounded-xl border border-neutral-200 dark:border-white/10 bg-neutral-50 dark:bg-white/5 text-neutral-900 dark:text-white placeholder-neutral-400 focus:outline-none focus:ring-2 focus:ring-purple-400/50"
                                    disabled={submitting}
                                />
                            </div>
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
                            className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-gradient-to-r from-purple-500 to-pink-500 text-white font-semibold hover:opacity-90 disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
                        >
                            {submitting ? (
                                <>
                                    <Loader2 className="w-5 h-5 animate-spin" />
                                    Connexion...
                                </>
                            ) : (
                                <>
                                    <LogIn className="w-5 h-5" />
                                    Se connecter
                                </>
                            )}
                        </button>
                    </form>

                    {/* Lien inscription */}
                    <div className="mt-6 text-center text-sm text-neutral-600 dark:text-neutral-400">
                        Pas encore de compte ?{' '}
                        <Link
                            to="/register"
                            className="text-purple-600 dark:text-purple-400 font-medium hover:underline"
                        >
                            Créer un compte
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
