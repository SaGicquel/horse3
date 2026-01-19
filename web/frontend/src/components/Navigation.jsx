/**
 * Navigation principale de l'application
 * Design moderne avec dropdown et sidebar
 */

import { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { Menu, X, ChevronDown, LogIn, LogOut, User } from 'lucide-react';
import ThemeToggle from './ThemeToggle';
import { useAuth } from '../context/AuthContext';

const Navigation = ({ currentPage, onNavigate }) => {
  const navigate = useNavigate();
  const { user, isAuthenticated, logout } = useAuth();
  const [isDark, setIsDark] = useState(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);
  const [activeDropdown, setActiveDropdown] = useState(null);
  const dropdownRef = useRef(null);

  useEffect(() => {
    const checkTheme = () => {
      setIsDark(document.documentElement.classList.contains('dark'));
    };
    checkTheme();
    const observer = new MutationObserver(checkTheme);
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class'] });
    return () => observer.disconnect();
  }, []);

  // Fermer dropdown quand on clique ailleurs
  useEffect(() => {
    const handleClickOutside = (event) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target)) {
        setActiveDropdown(null);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Liens principaux (max 5 dans la gélule)
  const primaryLinks = [
    { id: 'dashboard', label: 'Dashboard', emoji: '🏠' },
    { id: 'supervisor', label: 'Superviseur', emoji: '🧠' },
    { id: 'courses', label: 'Courses', emoji: '🏇' },
    { id: 'conseils', label: 'Conseils', emoji: '💡' },
    { id: 'conseils2', label: 'Conseils V2', emoji: '🎯' },
    { id: 'mesparis', label: 'Mes Paris', emoji: '🎟️' }
  ];

  // Liens secondaires dans le dropdown "Plus"
  const moreLinks = [
    { id: 'analytics', label: 'Analytics', emoji: '📈' },
    { id: 'backtest', label: 'Backtest', emoji: '🧪' },
    { id: 'data', label: 'Data', emoji: '💾' },
    { id: 'settings', label: 'Settings', emoji: '⚙️' }
  ];

  const adminLinks = [];

  // Prefetch pages on hover pour un chargement quasi-instantané
  const prefetchPage = (itemId) => {
    const pageMap = {
      dashboard: () => import('../pages/Dashboard'),
      supervisor: () => import('../pages/SupervisorDashboard'),
      courses: () => import('../pages/Courses'),
      conseils: () => import('../pages/Conseils'),
      conseils2: () => import('../pages/Conseils2'),
      analytics: () => import('../pages/Analytics'),
      mesparis: () => import('../pages/MesParis'),
      backtest: () => import('../pages/Backtest'),
      data: () => import('../pages/Data'),
      settings: () => import('../pages/Settings'),
    };
    if (pageMap[itemId]) {
      pageMap[itemId]();
    }
  };

  const handleNavClick = (itemId) => {
    onNavigate(itemId);
    setActiveDropdown(null);
    setIsMobileMenuOpen(false);
  };

  const toggleDropdown = (groupId) => {
    setActiveDropdown(activeDropdown === groupId ? null : groupId);
  };

  return (
    <motion.header
      className="sticky top-0 z-50"
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ type: "spring", stiffness: 100, damping: 20 }}
    >
      <div className="flex items-center justify-between px-4 py-3 glass-nav">
        {/* Logo */}
        <motion.div
          className="flex items-center gap-3 cursor-pointer"
          onClick={() => onNavigate('dashboard')}
        >
          <motion.img
            src={isDark ? "/logoPSF.png" : "/logoPSFB.png"}
            alt="HRP Logo"
            className="h-10 w-10 object-contain"
            whileHover={{ rotate: [0, -5, 5, 0] }}
            transition={{ duration: 0.3 }}
          />
          <div className="hidden sm:block">
            <h1 className="text-lg font-bold text-gradient">
              HorseRace Predictor
            </h1>
          </div>
        </motion.div>

        {/* Navigation Desktop - Gélule centrale */}
        <nav className="hidden lg:flex items-center" ref={dropdownRef}>
          <div className="flex items-center gap-1 px-2 py-1.5 rounded-full bg-neutral-100/80 dark:bg-white/5 border border-neutral-200/50 dark:border-white/10 backdrop-blur-sm">
            {primaryLinks.map((item) => {
              const isActive = currentPage === item.id;
              return (
                <motion.button
                  key={item.id}
                  onClick={() => handleNavClick(item.id)}
                  onMouseEnter={() => prefetchPage(item.id)}
                  className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${isActive
                    ? 'bg-primary-500 text-white shadow-md'
                    : 'text-neutral-700 dark:text-neutral-400 hover:bg-neutral-200/80 dark:hover:bg-white/10 hover:text-primary-600 dark:hover:text-primary-400'
                    }`}
                >
                  <span>{item.emoji}</span>
                  <span>{item.label}</span>
                </motion.button>
              );
            })}

            {/* Dropdown "Plus" pour les liens secondaires */}
            <div className="relative">
              <motion.button
                onClick={() => toggleDropdown('more')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${moreLinks.some(link => link.id === currentPage) || activeDropdown === 'more'
                  ? 'bg-primary-500 text-white shadow-md'
                  : 'text-neutral-700 dark:text-neutral-400 hover:bg-neutral-200/80 dark:hover:bg-white/10 hover:text-primary-600 dark:hover:text-primary-400'
                  }`}
              >
                <span>⚡</span>
                <span>Plus</span>
                <ChevronDown
                  size={14}
                  className={`transition-transform ${activeDropdown === 'more' ? 'rotate-180' : ''}`}
                />
              </motion.button>

              <AnimatePresence>
                {activeDropdown === 'more' && (
                  <motion.div
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    transition={{ duration: 0.15 }}
                    className="absolute top-full left-0 mt-2 py-2 rounded-xl min-w-[180px] z-50 glass-panel"
                  >
                    {moreLinks.map((item) => {
                      const isActive = currentPage === item.id;
                      return (
                        <motion.button
                          key={item.id}
                          onClick={() => handleNavClick(item.id)}
                          onMouseEnter={() => prefetchPage(item.id)}
                          className={`w-full flex items-center gap-3 px-4 py-2.5 text-sm transition-all ${isActive
                            ? 'bg-primary-500/10 text-primary-600 dark:text-primary-400'
                            : 'text-neutral-700 dark:text-neutral-400 hover:bg-neutral-100 dark:hover:bg-white/5 hover:text-primary-600 dark:hover:text-primary-400'
                            }`}
                          whileHover={{ x: 4 }}
                        >
                          <span className="text-base">{item.emoji}</span>
                          <span>{item.label}</span>
                        </motion.button>
                      );
                    })}
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          </div>
        </nav>

        {/* Actions Desktop */}
        <div className="hidden lg:flex items-center gap-3">
          <ThemeToggle />

          {/* Auth Section */}
          {isAuthenticated ? (
            <div className="relative">
              <motion.button
                onClick={() => toggleDropdown('user')}
                className="flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium bg-primary-500/10 text-primary-600 dark:text-primary-400 border border-primary-500/20"
              >
                <User size={16} />
                <span className="max-w-[120px] truncate">
                  {user?.display_name || user?.email?.split('@')[0] || 'Compte'}
                </span>
                <ChevronDown
                  size={14}
                  className={`transition-transform ${activeDropdown === 'user' ? 'rotate-180' : ''}`}
                />
              </motion.button>

              <AnimatePresence>
                {activeDropdown === 'user' && (
                  <motion.div
                    initial={{ opacity: 0, y: -10, scale: 0.95 }}
                    animate={{ opacity: 1, y: 0, scale: 1 }}
                    exit={{ opacity: 0, y: -10, scale: 0.95 }}
                    transition={{ duration: 0.15 }}
                    className="absolute top-full right-0 mt-2 py-2 rounded-xl min-w-[180px] z-50 glass-panel"
                  >
                    <div className="px-4 py-2 border-b border-white/10">
                      <div className="text-sm font-medium text-neutral-900 dark:text-white truncate">
                        {user?.display_name || 'Utilisateur'}
                      </div>
                      <div className="text-xs text-neutral-500 truncate">
                        {user?.email}
                      </div>
                    </div>
                    <motion.button
                      onClick={() => {
                        logout();
                        setActiveDropdown(null);
                      }}
                      className="w-full flex items-center gap-3 px-4 py-2.5 text-sm text-rose-600 dark:text-rose-400 hover:bg-rose-500/10"
                      whileHover={{ x: 4 }}
                    >
                      <LogOut size={16} />
                      <span>Déconnexion</span>
                    </motion.button>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          ) : (
            <motion.button
              onClick={() => navigate('/login')}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium bg-gradient-to-r from-purple-500 to-pink-500 text-white"
            >
              <LogIn size={16} />
              <span>Connexion</span>
            </motion.button>
          )}
        </div>

        {/* Mobile Menu Button */}
        <div className="flex lg:hidden items-center gap-2">
          <ThemeToggle />
          <motion.button
            onClick={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            className="p-2 rounded-lg text-primary-600 dark:text-primary-400 bg-primary-500/10"
          >
            {isMobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
          </motion.button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden overflow-hidden glass-panel border-t-0 rounded-t-none"
          >
            <div className="p-4 space-y-3">
              <div className="space-y-2">
                {primaryLinks.map((item) => {
                  const isActive = currentPage === item.id;
                  return (
                    <motion.button
                      key={item.id}
                      onClick={() => handleNavClick(item.id)}
                      className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium ${isActive
                        ? 'bg-primary-500/10 text-primary-600 dark:text-primary-400 border border-primary-500/20'
                        : 'text-neutral-700 dark:text-neutral-400'
                        }`}
                    >
                      <span className="text-lg">{item.emoji}</span>
                      <span>{item.label}</span>
                    </motion.button>
                  );
                })}
              </div>

              <div className="pt-2 border-t border-white/10">
                <div className="text-xs font-semibold uppercase tracking-wider px-3 py-2 text-neutral-500 dark:text-neutral-400">
                  ⚡ Plus
                </div>
                <div className="space-y-2">
                  {moreLinks.map((item) => {
                    const isActive = currentPage === item.id;
                    return (
                      <motion.button
                        key={item.id}
                        onClick={() => handleNavClick(item.id)}
                        className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium ${isActive
                          ? 'bg-primary-500/10 text-primary-600 dark:text-primary-400 border border-primary-500/20'
                          : 'text-neutral-700 dark:text-neutral-400'
                          }`}
                      >
                        <span className="text-lg">{item.emoji}</span>
                        <span>{item.label}</span>
                      </motion.button>
                    );
                  })}
                </div>
              </div>

              {/* Auth Section Mobile */}
              <div className="pt-2 border-t border-white/10">
                {isAuthenticated ? (
                  <div className="space-y-2">
                    <div className="px-4 py-2">
                      <div className="text-sm font-medium text-neutral-900 dark:text-white">
                        {user?.display_name || 'Utilisateur'}
                      </div>
                      <div className="text-xs text-neutral-500">
                        {user?.email}
                      </div>
                    </div>
                    <motion.button
                      onClick={() => {
                        logout();
                        setIsMobileMenuOpen(false);
                      }}
                      className="w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium text-rose-600 dark:text-rose-400 hover:bg-rose-500/10"
                    >
                      <LogOut size={18} />
                      <span>Déconnexion</span>
                    </motion.button>
                  </div>
                ) : (
                  <motion.button
                    onClick={() => {
                      navigate('/login');
                      setIsMobileMenuOpen(false);
                    }}
                    className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-lg text-sm font-medium bg-gradient-to-r from-purple-500 to-pink-500 text-white"
                  >
                    <LogIn size={18} />
                    <span>Connexion</span>
                  </motion.button>
                )}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.header>
  );
};

export default Navigation;
