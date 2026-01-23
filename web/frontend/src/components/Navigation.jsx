import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, ChevronDown, LogIn, LogOut, User } from 'lucide-react';
import ThemeToggle from './ThemeToggle';
import { useAuth } from '../context/AuthContext';

const Navigation = ({ currentPage, onNavigate }) => {
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuth();
  const [isScrolled, setIsScrolled] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState(null);

  // Détection du scroll pour l'animation
  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 50);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  const primaryLinks = [
    { id: 'dashboard', label: 'Dashboard', emoji: '🏠' },
    { id: 'supervisor', label: 'Superviseur', emoji: '🧠' },
    { id: 'courses', label: 'Courses', emoji: '🏇' },
    { id: 'conseils', label: 'Conseils', emoji: '💡' },
    { id: 'mesparis', label: 'Mes Paris', emoji: '🎟️' }
  ];

  const moreLinks = [
    { id: 'analytics', label: 'Analytics', emoji: '📈' },
    { id: 'conseils2', label: 'Conseils V2', emoji: '🎯' },
    { id: 'backtest', label: 'Backtest', emoji: '🧪' },
    { id: 'data', label: 'Data', emoji: '💾' },
    { id: 'settings', label: 'Settings', emoji: '⚙️' }
  ];

  const handleNavClick = (itemId) => {
    onNavigate(itemId);
    setActiveDropdown(null);
    setIsMobileMenuOpen(false);
  };

  // Composant Logo (réutilisé)
  const Logo = ({ className = "", showText = true }) => (
    <motion.div
      className={`flex items-center gap-2 cursor-pointer ${className}`}
      onClick={() => onNavigate('dashboard')}
      layout
    >
      <img src="/logoPSF.png" alt="Logo" className="h-8 w-8 object-contain" />
      {showText && (
        <span className="font-bold text-lg hidden xl:block text-gradient">
          HorseRace Predictor
        </span>
      )}
    </motion.div>
  );

  // Composant Actions (User + Theme)
  const Actions = ({ compact = false }) => (
    <motion.div className="flex items-center gap-2" layout>
      <ThemeToggle />

      {isAuthenticated ? (
        <div className="relative">
          <motion.button
            onClick={() => setActiveDropdown(activeDropdown === 'user' ? null : 'user')}
            className={`flex items-center gap-2 rounded-full font-medium transition-all ${
              compact
                ? 'p-1.5 hover:bg-neutral-200 dark:hover:bg-white/10'
                : 'px-3 py-1.5 bg-primary-500/10 text-primary-600 dark:text-primary-400 border border-primary-500/20'
            }`}
          >
            <User size={18} />
            {!compact && (
              <span className="max-w-[100px] truncate text-sm">
                {user?.display_name || 'Compte'}
              </span>
            )}
          </motion.button>

          <AnimatePresence>
            {activeDropdown === 'user' && (
              <motion.div
                initial={{ opacity: 0, y: 10, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, y: 10, scale: 0.95 }}
                className="absolute top-full right-0 mt-2 py-2 rounded-xl min-w-[180px] z-50 glass-panel"
              >
                <button
                  onClick={logout}
                  className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-rose-600 hover:bg-rose-500/10"
                >
                  <LogOut size={16} />
                  <span>Déconnexion</span>
                </button>
              </motion.div>
            )}
          </AnimatePresence>
        </div>
      ) : (
        <motion.button
          onClick={() => navigate('/login')}
          className={`flex items-center gap-2 rounded-full font-medium bg-gradient-to-r from-purple-500 to-pink-500 text-white shadow-lg shadow-purple-500/20 ${
            compact ? 'p-2' : 'px-4 py-1.5 text-sm'
          }`}
        >
          <LogIn size={16} />
          {!compact && <span>Connexion</span>}
        </motion.button>
      )}
    </motion.div>
  );

  return (
    <>
      {/* Container Principal Fixe - Pointer events none pour laisser passer le clic autour */}
      <header className="fixed top-0 left-0 right-0 z-50 flex justify-center pt-4 sm:pt-6 pointer-events-none">

        {/* 1. Éléments Flottants (Logo Gauche + Actions Droite) - Visibles SEULEMENT si non scrollé */}
        <AnimatePresence>
          {!isScrolled && (
            <>
              <motion.div
                className="absolute left-6 top-6 pointer-events-auto hidden lg:block"
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: -20, transition: { duration: 0.2 } }}
              >
                <Logo />
              </motion.div>

              <motion.div
                className="absolute right-6 top-6 pointer-events-auto hidden lg:block"
                initial={{ opacity: 0, x: 20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20, transition: { duration: 0.2 } }}
              >
                <Actions />
              </motion.div>
            </>
          )}
        </AnimatePresence>

        {/* 2. Gélule Centrale - Devient la barre principale au scroll */}
        <motion.nav
          layout
          className="pointer-events-auto backdrop-blur-xl bg-white/70 dark:bg-slate-900/70 border border-white/20 dark:border-white/10 shadow-xl shadow-black/5 rounded-full px-2 py-1.5 flex items-center gap-2 mx-4"
          initial={{ y: -100 }}
          animate={{ y: 0 }}
          transition={{ type: "spring", stiffness: 100, damping: 20 }}
        >
          {/* Logo intégré (Visible SEULEMENT au scroll) */}
          <AnimatePresence>
            {isScrolled && (
              <motion.div
                initial={{ width: 0, opacity: 0, paddingRight: 0 }}
                animate={{ width: "auto", opacity: 1, paddingRight: 8 }}
                exit={{ width: 0, opacity: 0, paddingRight: 0 }}
                className="overflow-hidden border-r border-neutral-200 dark:border-white/10 mr-1"
              >
                 <Logo className="pl-2" showText={false} />
              </motion.div>
            )}
          </AnimatePresence>

          {/* Liens de navigation */}
          <div className="flex items-center">
            {primaryLinks.map((item) => {
              const isActive = currentPage === item.id;
              return (
                <motion.button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  className={`relative px-3 py-1.5 rounded-full text-sm font-medium transition-colors ${
                    isActive
                      ? 'text-white'
                      : 'text-neutral-600 dark:text-neutral-400 hover:text-neutral-900 dark:hover:text-white hover:bg-neutral-100 dark:hover:bg-white/5'
                  }`}
                >
                  {isActive && (
                    <motion.div
                      layoutId="nav-pill"
                      className="absolute inset-0 bg-black/5 dark:bg-white/10 border border-black/5 dark:border-white/10 shadow-[0_0_10px_rgba(255,255,255,0.1)] rounded-full backdrop-blur-sm"
                      transition={{ type: "spring", bounce: 0.2, duration: 0.6 }}
                    />
                  )}
                  <span className="relative z-10 flex items-center gap-2">
                    <span>{item.emoji}</span>
                    <span className="hidden md:inline">{item.label}</span>
                  </span>
                </motion.button>
              );
            })}

            {/* Dropdown "Plus" */}
            <div className="relative ml-1">
              <button
                onClick={() => setActiveDropdown(activeDropdown === 'more' ? null : 'more')}
                className={`flex items-center justify-center w-8 h-8 rounded-full transition-colors ${
                  activeDropdown === 'more'
                    ? 'bg-neutral-100 dark:bg-white/10 text-neutral-900 dark:text-white'
                    : 'text-neutral-500 hover:bg-neutral-100 dark:hover:bg-white/5'
                }`}
              >
                <ChevronDown size={14} className={`transition-transform ${activeDropdown === 'more' ? 'rotate-180' : ''}`} />
              </button>
               <AnimatePresence>
                {activeDropdown === 'more' && (
                  <motion.div
                    initial={{ opacity: 0, y: 10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: 10, scale: 0.95 }}
                    className="absolute top-full left-1/2 -translate-x-1/2 mt-3 py-2 rounded-xl min-w-[200px] z-50 glass-panel overflow-hidden"
                  >
                    {moreLinks.map((item) => (
                      <button
                        key={item.id}
                        onClick={() => handleNavClick(item.id)}
                        className="w-full flex items-center gap-3 px-4 py-2.5 text-sm hover:bg-neutral-100 dark:hover:bg-white/5 text-left"
                      >
                        <span>{item.emoji}</span>
                        <span>{item.label}</span>
                      </button>
                    ))}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>

          {/* Actions intégrées (Visibles SEULEMENT au scroll ou Mobile) */}
          <AnimatePresence>
            {(isScrolled || window.innerWidth < 1024) && (
              <motion.div
                initial={{ width: 0, opacity: 0, paddingLeft: 0 }}
                animate={{ width: "auto", opacity: 1, paddingLeft: 8 }}
                exit={{ width: 0, opacity: 0, paddingLeft: 0 }}
                className="overflow-hidden border-l border-neutral-200 dark:border-white/10 ml-1 hidden lg:block"
              >
                <div className="pl-2">
                  <Actions compact />
                </div>
              </motion.div>
            )}
          </AnimatePresence>

          {/* Bouton Menu Mobile */}
          <button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="lg:hidden p-2 ml-2 rounded-full bg-neutral-100 dark:bg-white/10"
          >
            {isMobileMenuOpen ? <X size={18} /> : <Menu size={18} />}
          </button>
        </motion.nav>
      </header>

      {/* Menu Mobile Full Screen Overlay */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-40 bg-white/95 dark:bg-neutral-950/95 backdrop-blur-lg pt-24 px-6 lg:hidden"
          >
            <div className="flex flex-col gap-6">
              <div className="space-y-2">
                <p className="text-xs font-semibold uppercase text-neutral-400 mb-2">Navigation</p>
                {[...primaryLinks, ...moreLinks].map((item) => (
                  <motion.button
                    key={item.id}
                    onClick={() => handleNavClick(item.id)}
                    initial={{ x: -20, opacity: 0 }}
                    animate={{ x: 0, opacity: 1 }}
                    className="flex items-center gap-4 w-full p-4 rounded-xl bg-neutral-100 dark:bg-white/5 text-lg font-medium"
                  >
                    <span>{item.emoji}</span>
                    {item.label}
                  </motion.button>
                ))}
              </div>

              <div className="pt-6 border-t border-neutral-200 dark:border-white/10">
                 <div className="flex justify-between items-center">
                    <ThemeToggle />
                    {isAuthenticated ? (
                       <button onClick={logout} className="flex items-center gap-2 text-rose-500 font-medium">
                         <LogOut size={20} /> Déconnexion
                       </button>
                    ) : (
                       <button onClick={() => navigate('/login')} className="flex items-center gap-2 text-primary-500 font-medium">
                         <LogIn size={20} /> Connexion
                       </button>
                    )}
                 </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
};

export default Navigation;
