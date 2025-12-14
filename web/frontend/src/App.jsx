import { lazy, Suspense } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { BrowserRouter, Routes, Route, Navigate, useLocation, useNavigate, matchPath } from 'react-router-dom';
import Navigation from './components/Navigation';
import { CombinedBackground } from './components/AnimatedBackground';
import { SkeletonDashboard } from './components/Skeleton';
import BackendStatus from './components/BackendStatus';
import UserSettingsNotification from './components/UserSettingsNotification';
import { AuthProvider } from './context/AuthContext';

// Lazy loading des pages pour de meilleures performances
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Analytics = lazy(() => import('./pages/Analytics'));
const Backtest = lazy(() => import('./pages/Backtest'));
const Data = lazy(() => import('./pages/Data'));
// const Landing = lazy(() => import('./pages/Landing'));
const Courses = lazy(() => import('./pages/Courses'));
const Conseils = lazy(() => import('./pages/Conseils'));
const MesParis = lazy(() => import('./pages/MesParis'));
const Settings = lazy(() => import('./pages/Settings'));
const Calibration = lazy(() => import('./pages/Calibration'));
const ChevalProfile = lazy(() => import('./pages/ChevalProfile'));
const Hippodromes = lazy(() => import('./pages/Hippodromes'));
const Jockeys = lazy(() => import('./pages/Jockeys'));
const Entraineurs = lazy(() => import('./pages/Entraineurs'));
const Landing2 = lazy(() => import('./pages/landing2/Landing2'));
const Landing3 = lazy(() => import('./pages/landing3/Landing3'));
const Login = lazy(() => import('./pages/Login'));
const Register = lazy(() => import('./pages/Register'));

const appRoutes = [
  { id: 'dashboard', path: '/dashboard', element: <Dashboard /> },
  { id: 'courses', path: '/courses', element: <Courses /> },
  { id: 'conseils', path: '/conseils', element: <Conseils /> },
  { id: 'analytics', path: '/analytics', element: <Analytics /> },
  { id: 'mesparis', path: '/mes-paris', element: <MesParis /> },
  { id: 'backtest', path: '/backtest', element: <Backtest /> },
  { id: 'data', path: '/data', element: <Data /> },
  { id: 'settings', path: '/settings', element: <Settings /> },
  { id: 'calibration', path: '/calibration', element: <Calibration /> },
  { id: 'cheval', path: '/cheval', element: <ChevalProfile /> },
  { id: 'chevalDetail', path: '/cheval/:id', element: <ChevalProfile /> },
  { id: 'hippodromes', path: '/hippodromes', element: <Hippodromes /> },
  { id: 'hippodromeDetail', path: '/hippodrome/:id', element: <Hippodromes /> },
  { id: 'jockeys', path: '/jockeys', element: <Jockeys /> },
  { id: 'jockeyDetail', path: '/jockey/:id', element: <Jockeys /> },
  { id: 'entraineurs', path: '/entraineurs', element: <Entraineurs /> },
  { id: 'entraineurDetail', path: '/entraineur/:id', element: <Entraineurs /> },
  { id: 'landing2', path: '/landing2', element: <Landing2 /> },
  { id: 'landing3', path: '/landing3', element: <Landing3 /> },
  { id: 'login', path: '/login', element: <Login /> },
  { id: 'register', path: '/register', element: <Register /> },
];

const pathByPage = appRoutes.reduce((acc, route) => {
  acc[route.id] = route.path;
  return acc;
}, {});

// Variantes d'animation pour les transitions de page
const pageVariants = {
  initial: {
    opacity: 0,
    y: 20,
    scale: 0.98
  },
  animate: {
    opacity: 1,
    y: 0,
    scale: 1,
    transition: {
      type: "spring",
      stiffness: 100,
      damping: 20,
      mass: 1
    }
  },
  exit: {
    opacity: 0,
    y: -20,
    scale: 0.98,
    transition: {
      duration: 0.2
    }
  }
};

// Composant de chargement avec animation
const PageLoader = () => (
  <motion.div
    initial={{ opacity: 0 }}
    animate={{ opacity: 1 }}
    exit={{ opacity: 0 }}
    className="min-h-screen flex items-center justify-center bg-neutral-50 dark:bg-neutral-950"
  >
    <SkeletonDashboard />
  </motion.div>
);

const AppContent = () => {
  const location = useLocation();
  const navigate = useNavigate();

  const matchedRoute = appRoutes.find((route) =>
    matchPath({ path: route.path, end: true }, location.pathname)
  );
  const currentPage = matchedRoute?.id || 'dashboard';

  const handleNavigate = (pageId) => {
    const targetPath = pathByPage[pageId] || '/dashboard';
    if (location.pathname !== targetPath) {
      navigate(targetPath);
    }
  };

  if (location.pathname === '/') {
    return (
      <Suspense fallback={<PageLoader />}>
        {/* <Landing onEnter={() => navigate('/dashboard')} /> */}
        <Navigate to="/dashboard" replace />
      </Suspense>
    );
  }

  if (location.pathname === '/landing2') {
    return (
      <Suspense fallback={<PageLoader />}>
        <Landing2 />
      </Suspense>
    );
  }

  if (location.pathname === '/landing3') {
    return (
      <Suspense fallback={<PageLoader />}>
        <Landing3 />
      </Suspense>
    );
  }

  // Pages d'auth sans navigation
  if (location.pathname === '/login' || location.pathname === '/register') {
    return (
      <div className="min-h-screen relative">
        <CombinedBackground variant="minimal" />
        <Suspense fallback={<PageLoader />}>
          <Routes location={location}>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
          </Routes>
        </Suspense>
      </div>
    );
  }

  return (
    <div className="min-h-screen relative">
      {/* Backend smoke check - affiche une alerte si backend down */}
      <BackendStatus showWhenHealthy position="top" />

      {/* Background animé */}
      <CombinedBackground variant="minimal" />

      {/* Navigation */}
      <Navigation currentPage={currentPage} onNavigate={handleNavigate} />

      {/* Contenu principal avec transitions */}
      <main className="relative z-10 p-3 sm:p-6 lg:p-8">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={location.pathname}
            variants={pageVariants}
            initial="initial"
            animate="animate"
            exit="exit"
          >
            <Suspense fallback={<PageLoader />}>
              <Routes location={location}>
                {appRoutes.map((route) => (
                  <Route key={route.path} path={route.path} element={route.element} />
                ))}
                <Route path="/betting" element={<Navigate to="/conseils" replace />} />
                <Route path="/paris" element={<Navigate to="/conseils" replace />} />
                <Route path="/predictions" element={<Navigate to="/conseils" replace />} />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </Suspense>
          </motion.div>
        </AnimatePresence>
      </main>

      {/* Notifications des paramètres utilisateur */}
      <UserSettingsNotification />
    </div>
  );
};

function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <AppContent />
      </AuthProvider>
    </BrowserRouter>
  );
}

export default App;
