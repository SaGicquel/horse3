import React, { useState, useEffect, useRef } from 'react';
import { motion, useScroll, useTransform, useSpring, useMotionValue } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import logo from './imagelanding2.png';

// --- Components ---

const MouseParallax = ({ children, sensitivity = 50 }) => {
    const x = useSpring(useMotionValue(0), { stiffness: 400, damping: 90 });
    const y = useSpring(useMotionValue(0), { stiffness: 400, damping: 90 });
    const rotateX = useSpring(useMotionValue(0), { stiffness: 400, damping: 90 });
    const rotateY = useSpring(useMotionValue(0), { stiffness: 400, damping: 90 });
    const scale = useSpring(useMotionValue(1), { stiffness: 400, damping: 90 });

    const handleMouseMove = (e) => {
        const { clientX, clientY, currentTarget } = e;
        const { width, height, left, top } = currentTarget.getBoundingClientRect();

        // Calculate position relative to center [-1, 1]
        const mouseX = (clientX - left - width / 2) / (width / 2);
        const mouseY = (clientY - top - height / 2) / (height / 2);

        x.set(mouseX * sensitivity);
        y.set(mouseY * sensitivity);
        rotateX.set(-mouseY * 25); // Stronger tilt effect (25 deg)
        rotateY.set(mouseX * 25); // Stronger tilt effect (25 deg)
        scale.set(1 + Math.abs(mouseX * mouseY) * 0.1); // Subtle zoom on corners
    };

    const handleMouseLeave = () => {
        x.set(0);
        y.set(0);
        rotateX.set(0);
        rotateY.set(0);
        scale.set(1);
    }

    return (
        <motion.div
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            style={{ x, y, rotateX, rotateY, scale, perspective: 1200, transformStyle: "preserve-3d" }}
            className="w-full h-full flex items-center justify-center cursor-pointer"
        >
            <motion.div style={{ transformStyle: "preserve-3d" }} className="relative z-10 p-10">
                {children}
            </motion.div>
        </motion.div>
    );
};

const ScrollReveal = ({ children, delay = 0 }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-100px" }}
            transition={{ duration: 0.8, delay, ease: [0.22, 1, 0.36, 1] }}
        >
            {children}
        </motion.div>
    );
};

const FeatureCard = ({ title, desc, delay }) => (
    <ScrollReveal delay={delay}>
        <div className="p-8 rounded-2xl bg-white/5 backdrop-blur-lg border border-white/10 hover:bg-white/10 transition-colors duration-300">
            <div className="h-12 w-12 rounded-full bg-gradient-to-tr from-rose-500 to-orange-500 mb-6 flex items-center justify-center text-white font-bold text-xl">
                {title[0]}
            </div>
            <h3 className="text-2xl font-bold text-white mb-3">{title}</h3>
            <p className="text-gray-400 leading-relaxed">{desc}</p>
        </div>
    </ScrollReveal>
);

const RacetrackRing = () => {
    return (
        <div className="absolute inset-x-0 top-1/2 -translate-y-1/2 flex items-center justify-center pointer-events-none z-0 opacity-80 mix-blend-screen overflow-visible">
            <svg width="1000" height="800" viewBox="0 0 1000 800" className="w-[160%] max-w-none md:w-full h-auto overflow-visible">
                <defs>
                    <linearGradient id="laserGradient" gradientUnits="userSpaceOnUse" x1="0%" y1="0%" x2="100%" y2="0%">
                        <stop offset="0%" stopColor="#ec4899" stopOpacity="0" />
                        <stop offset="50%" stopColor="#ec4899" />
                        <stop offset="100%" stopColor="#fff" />
                    </linearGradient>
                    <filter id="glow" x="-50%" y="-50%" width="200%" height="200%">
                        <feGaussianBlur stdDeviation="4" result="coloredBlur" />
                        <feMerge>
                            <feMergeNode in="coloredBlur" />
                            <feMergeNode in="SourceGraphic" />
                        </feMerge>
                    </filter>
                </defs>

                {/* Track Path (Invisible guide or faint track) */}
                <path
                    d="M 500 550 A 450 150 0 1 1 499.9 550 L 500 800"
                    fill="none"
                    stroke="rgba(236, 72, 153, 0.1)"
                    strokeWidth="1"
                    className="opacity-30"
                />

                {/* Laser Beam Animation */}
                <motion.path
                    d="M 500 550 A 450 150 0 1 1 499.9 550 L 500 800"
                    fill="none"
                    stroke="#ec4899"
                    strokeWidth="2"
                    strokeLinecap="round"
                    filter="url(#glow)"
                    initial={{ pathLength: 0, opacity: 0 }}
                    animate={{
                        pathLength: [0, 1],
                        opacity: [0, 1, 1, 0],
                        pathOffset: [0, 0]
                    }}
                    transition={{
                        duration: 2.5,
                        repeat: Infinity,
                        ease: "easeInOut",
                        repeatDelay: 0.5
                    }}
                />
            </svg>
        </div>
    );
};

// --- Floating Navbar ---

const FloatingNav = () => {
    const scrollToSection = (id) => {
        const element = document.getElementById(id);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
        }
    };

    return (
        <motion.div
            initial={{ y: -100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ delay: 1, duration: 0.8, ease: "easeOut" }}
            className="fixed top-6 left-1/2 -translate-x-1/2 z-50 p-2 rounded-full border border-white/10 bg-black/60 backdrop-blur-md shadow-2xl flex items-center gap-2 md:gap-8 min-w-[320px] md:min-w-[600px] justify-between"
        >
            {/* Logo */}
            <div className="pl-4 pr-2">
                <img src={logo} alt="Logo" className="h-8 w-auto object-contain brightness-200" />
            </div>

            {/* Links - Hidden on very small screens, visible on md+ */}
            <div className="hidden md:flex items-center gap-6 text-sm font-medium text-neutral-300">
                <button onClick={() => scrollToSection('features')} className="hover:text-white transition-colors">Fonctionnalités</button>
                <button onClick={() => scrollToSection('pricing')} className="hover:text-white transition-colors">Tarifs</button>
                <button onClick={() => scrollToSection('faq')} className="hover:text-white transition-colors">FAQ</button>
            </div>

            {/* Actions */}
            <div className="flex items-center gap-3 pr-2">
                <a href="/login" className="text-sm font-medium text-neutral-300 hover:text-white transition-colors hidden sm:block">
                    Connexion
                </a>
                <button className="px-5 py-2.5 rounded-full bg-white text-black text-sm font-bold hover:bg-neutral-200 transition-colors whitespace-nowrap">
                    Créer un compte
                </button>
            </div>
        </motion.div>
    );
};

// --- Main Page ---

const Landing2 = () => {
    const navigate = useNavigate();
    const { scrollYProgress } = useScroll();
    const scaleX = useSpring(scrollYProgress, {
        stiffness: 100,
        damping: 30,
        restDelta: 0.001
    });

    return (
        <div className="min-h-screen bg-black text-white selection:bg-rose-500/30 overflow-x-hidden font-sans">
            {/* ProgressBar */}
            <motion.div
                className="fixed top-0 left-0 right-0 h-1 bg-rose-500 origin-left z-50"
                style={{ scaleX }}
            />

            <FloatingNav />

            {/* Hero Section */}
            <section className="h-screen w-full relative overflow-hidden flex flex-col items-center justify-center">
                {/* Background Gradients - Reduced opacity for deeper black feel */}
                <div className="absolute inset-0 z-0">
                    <div className="absolute top-[-20%] left-[-10%] w-[60%] h-[60%] bg-purple-900/10 rounded-full blur-[120px] mix-blend-screen animate-pulse-slow" />
                    <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-rose-900/10 rounded-full blur-[100px] mix-blend-screen" />
                </div>

                {/* Content */}
                <div className="relative z-10 w-full max-w-7xl mx-auto px-6 h-full flex flex-col items-center justify-center">

                    {/* Racetrack Ring Layer */}
                    <div className="absolute inset-0 z-0 flex items-center justify-center pointer-events-none">
                        <RacetrackRing />
                    </div>

                    <MouseParallax sensitivity={30}>
                        <div className="relative group cursor-pointer">
                            <motion.img
                                src={logo}
                                alt="Horse3 Logo"
                                className="w-full max-w-[500px] md:max-w-[700px] object-contain relative z-10 drop-shadow-2xl"
                                initial={{ opacity: 0, scale: 0.8, filter: "blur(10px)" }}
                                animate={{ opacity: 1, scale: 1, filter: "blur(0px)" }}
                                transition={{ duration: 1.2, ease: "easeOut" }}
                            />
                        </div>
                    </MouseParallax>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.8, duration: 0.8 }}
                        className="absolute bottom-12 flex flex-col items-center gap-4 text-center"
                    >
                        <p className="text-sm uppercase tracking-[0.3em] text-neutral-500">Découvrez l'expérience</p>
                        <motion.div
                            animate={{ y: [0, 10, 0] }}
                            transition={{ repeat: Infinity, duration: 2 }}
                            className="w-[1px] h-16 bg-gradient-to-b from-neutral-500 to-transparent"
                        />
                    </motion.div>
                </div>
            </section>

            {/* Intro Text Section */}
            <section className="py-32 px-6 relative z-10">
                <div className="max-w-4xl mx-auto text-center">
                    <ScrollReveal>
                        <h2 className="text-4xl md:text-6xl font-bold mb-8 bg-gradient-to-r from-white via-neutral-200 to-neutral-500 bg-clip-text text-transparent">
                            L'avenir des courses hippiques<br />commence ici.
                        </h2>
                    </ScrollReveal>
                    <ScrollReveal delay={0.2}>
                        <p className="text-xl md:text-2xl text-neutral-400 leading-relaxed font-light">
                            Une interface fluide, des données en temps réel et une expérience utilisateur repensée pour les passionnés exigeants.
                        </p>
                    </ScrollReveal>
                </div>
            </section>

            {/* Features Grid */}
            <section id="features" className="py-24 px-6 relative z-10 bg-neutral-900/30">
                <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-3 gap-8">
                    <FeatureCard
                        title="Vitesse"
                        desc="Optimisé pour la performance instantanée. Ne manquez jamais le départ."
                        delay={0}
                    />
                    <FeatureCard
                        title="Innovation"
                        desc="Des outils d'analyse prédictive nouvelle génération à portée de clic."
                        delay={0.2}
                    />
                    <FeatureCard
                        title="Intuitif"
                        desc="Un design épuré qui met l'information essentielle au premier plan."
                        delay={0.4}
                    />
                </div>
            </section>

            {/* Pricing Section */}
            <section id="pricing" className="py-32 px-6 relative z-10">
                <div className="max-w-4xl mx-auto text-center mb-16">
                    <ScrollReveal>
                        <h2 className="text-4xl font-bold mb-4">Des tarifs adaptés à votre ambition</h2>
                        <p className="text-neutral-400">Du parieur curieux au turfiste professionnel.</p>
                    </ScrollReveal>
                </div>

                <div className="max-w-7xl mx-auto grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
                    {/* Freemium */}
                    <ScrollReveal delay={0.1}>
                        <div className="h-full p-6 rounded-3xl bg-neutral-900/50 border border-white/5 hover:border-white/10 transition-colors flex flex-col">
                            <h3 className="text-xl font-bold mb-2 text-neutral-300">Freemium</h3>
                            <p className="text-3xl font-bold mb-6">Gratuit</p>
                            <div className="flex-grow">
                                <p className="text-sm text-neutral-500 mb-4 font-medium">Pour découvrir la qualité :</p>
                                <ul className="text-neutral-400 text-sm space-y-3 mb-8 text-left">
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> 3 paris gratuits / jour</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Value + explications simples</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Historique 3 jours</li>
                                    <li className="flex items-start opacity-50"><span className="mr-2">✕</span> Pas d'exotiques</li>
                                    <li className="flex items-start opacity-50"><span className="mr-2">✕</span> Pas de portfolio</li>
                                </ul>
                            </div>
                            <button className="w-full py-3 rounded-xl bg-white/10 hover:bg-white/20 transition-colors font-semibold text-sm">Découvrir</button>
                        </div>
                    </ScrollReveal>

                    {/* Standard */}
                    <ScrollReveal delay={0.2}>
                        <div className="h-full p-6 rounded-3xl bg-neutral-800/80 border border-rose-500/30 hover:border-rose-500/60 transition-colors relative flex flex-col shadow-2xl shadow-rose-900/10">
                            <div className="absolute top-0 right-0 bg-rose-500 text-white text-[10px] font-bold px-3 py-1 rounded-bl-xl rounded-tr-3xl tracking-wider">POPULAIRE</div>
                            <h3 className="text-xl font-bold mb-2 text-white">Standard</h3>
                            <p className="text-3xl font-bold mb-6">19€<span className="text-sm font-normal text-neutral-500">/mois</span></p>
                            <div className="flex-grow">
                                <p className="text-sm text-neutral-400 mb-4 font-medium">Le choix de 80% des joueurs :</p>
                                <ul className="text-neutral-300 text-sm space-y-3 mb-8 text-left">
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Tous les pronostics du jour</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Kelly + Mise suggérée</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> ROI/PNL personnel</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Historique complet</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Alertes variation de cotes</li>
                                </ul>
                            </div>
                            <button className="w-full py-3 rounded-xl bg-rose-600 hover:bg-rose-700 transition-colors font-semibold text-sm shadow-lg shadow-rose-600/20">Choisir Standard</button>
                        </div>
                    </ScrollReveal>

                    {/* Premium */}
                    <ScrollReveal delay={0.3}>
                        <div className="h-full p-6 rounded-3xl bg-neutral-900/50 border border-white/5 hover:border-white/10 transition-colors flex flex-col">
                            <h3 className="text-xl font-bold mb-2 text-rose-400">Premium</h3>
                            <p className="text-3xl font-bold mb-6">39€<span className="text-sm font-normal text-neutral-500">/mois</span></p>
                            <div className="flex-grow">
                                <p className="text-sm text-neutral-500 mb-4 font-medium">Pour les joueurs réguliers :</p>
                                <ul className="text-neutral-400 text-sm space-y-3 mb-8 text-left">
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Tout du Standard</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Algo Exotiques optimisés</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Portfolio complet + Bankroll</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Détection Steam/Drift live</li>
                                    <li className="flex items-start"><span className="mr-2 text-rose-500">✓</span> Accès API Pronos</li>
                                </ul>
                            </div>
                            <button className="w-full py-3 rounded-xl bg-white/10 hover:bg-white/20 transition-colors font-semibold text-sm">Passer Premium</button>
                        </div>
                    </ScrollReveal>

                    {/* Pro */}
                    <ScrollReveal delay={0.4}>
                        <div className="h-full p-6 rounded-3xl bg-neutral-900/50 border border-white/5 hover:border-white/10 transition-colors flex flex-col">
                            <h3 className="text-xl font-bold mb-2 text-amber-400">Pro</h3>
                            <p className="text-3xl font-bold mb-6">79€<span className="text-sm font-normal text-neutral-500">/mois</span></p>
                            <div className="flex-grow">
                                <p className="text-sm text-neutral-500 mb-4 font-medium">Turfistes intensifs / VIP :</p>
                                <ul className="text-neutral-400 text-sm space-y-3 mb-8 text-left">
                                    <li className="flex items-start"><span className="mr-2 text-amber-500">✓</span> Tout du Premium</li>
                                    <li className="flex items-start"><span className="mr-2 text-amber-500">✓</span> Export CSV/API Illimité</li>
                                    <li className="flex items-start"><span className="mr-2 text-amber-500">✓</span> Analyse Multi-courses</li>
                                    <li className="flex items-start"><span className="mr-2 text-amber-500">✓</span> Backtest personnalisé</li>
                                    <li className="flex items-start"><span className="mr-2 text-amber-500">✓</span> Priorité futurs modules</li>
                                </ul>
                            </div>
                            <button className="w-full py-3 rounded-xl bg-gradient-to-r from-amber-500 to-amber-700 hover:from-amber-400 hover:to-amber-600 text-black transition-all font-semibold text-sm shadow-lg shadow-amber-900/20">Devenir Pro</button>
                        </div>
                    </ScrollReveal>
                </div>
            </section>

            {/* FAQ Section */}
            <section id="faq" className="py-32 px-6 relative z-10 bg-neutral-900/30">
                <div className="max-w-3xl mx-auto">
                    <ScrollReveal>
                        <h2 className="text-4xl font-bold mb-12 text-center">Questions Fréquentes</h2>
                        <div className="space-y-6">
                            {[
                                { q: "Comment fonctionne l'IA ?", a: "Notre algorithme analyse des millions de données historiques pour prédire les issues les plus probables." },
                                { q: "Puis-je annuler à tout moment ?", a: "Oui, sans aucun engagement. Vous êtes libre." },
                                { q: "Les paiements sont-ils sécurisés ?", a: "Absolument. Nous utilisons Stripe pour traiter toutes les transactions." }
                            ].map((item, i) => (
                                <div key={i} className="p-6 rounded-2xl bg-black border border-white/5">
                                    <h4 className="font-bold mb-2 text-lg">{item.q}</h4>
                                    <p className="text-neutral-400">{item.a}</p>
                                </div>
                            ))}
                        </div>
                    </ScrollReveal>
                </div>
            </section>

            {/* CTA Section */}
            <section className="h-[80vh] flex items-center justify-center relative overflow-hidden">
                <div className="absolute inset-0 bg-rose-600/5" />
                <div className="text-center relative z-10 px-6">
                    <ScrollReveal>
                        <h2 className="text-5xl md:text-8xl font-black mb-8 tracking-tight">
                            PRET ?
                        </h2>
                        <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => navigate('/dashboard')}
                            className="px-12 py-6 bg-white text-black text-xl font-bold rounded-full hover:bg-neutral-200 transition-colors shadow-[0_0_40px_-10px_rgba(255,255,255,0.3)]"
                        >
                            Entrer dans l'arène
                        </motion.button>
                    </ScrollReveal>
                </div>
            </section>

            {/* Playground / Interactive Footer */}
            <section className="py-12 text-center text-neutral-600 text-sm">
                <p>© 2025 Horse3. Designed for the Future.</p>
            </section>
        </div>
    );
};

export default Landing2;
