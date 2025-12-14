import React, { useRef, useState, useEffect } from 'react';
import { motion, useScroll, useTransform, useSpring, useInView, useMotionValue } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { ArrowRight, ChevronDown, Rocket, Shield, TrendingUp, CheckCircle, HelpCircle, Star, Zap, Lock, BarChart3 } from 'lucide-react';

// --- Scroll Reveal Helper ---
const ScrollReveal = ({ children, delay = 0 }) => {
    return (
        <motion.div
            initial={{ opacity: 0, y: 50 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, margin: "-50px" }}
            transition={{ duration: 0.8, delay, ease: "easeOut" }}
        >
            {children}
        </motion.div>
    );
};

// --- Magnetic Button Helper ---
const MagneticButton = ({ children, className, onClick }) => {
    const ref = useRef(null);
    const [position, setPosition] = useState({ x: 0, y: 0 });

    const handleMouse = (e) => {
        const { clientX, clientY } = e;
        const { height, width, left, top } = ref.current.getBoundingClientRect();
        const middleX = clientX - (left + width / 2);
        const middleY = clientY - (top + height / 2);
        setPosition({ x: middleX * 0.2, y: middleY * 0.2 });
    };

    const reset = () => {
        setPosition({ x: 0, y: 0 });
    };

    const { x, y } = position;
    return (
        <motion.button
            ref={ref}
            className={className}
            animate={{ x, y }}
            transition={{ type: "spring", stiffness: 150, damping: 15, mass: 0.1 }}
            onMouseMove={handleMouse}
            onMouseLeave={reset}
            onClick={onClick}
        >
            {children}
        </motion.button>
    );
};

// --- 3D Tilt Container Helper ---
const TiltContainer = ({ children }) => {
    const x = useSpring(useMotionValue(0), { stiffness: 400, damping: 90 });
    const y = useSpring(useMotionValue(0), { stiffness: 400, damping: 90 });
    const rotateX = useSpring(useMotionValue(0), { stiffness: 400, damping: 90 });
    const rotateY = useSpring(useMotionValue(0), { stiffness: 400, damping: 90 });

    const handleMouseMove = (e) => {
        const { clientX, clientY, currentTarget } = e;
        const { width, height, left, top } = currentTarget.getBoundingClientRect();
        const mouseX = (clientX - left - width / 2) / (width / 2);
        const mouseY = (clientY - top - height / 2) / (height / 2);
        rotateX.set(-mouseY * 10);
        rotateY.set(mouseX * 10);
    };

    const handleMouseLeave = () => {
        rotateX.set(0);
        rotateY.set(0);
    };

    return (
        <motion.div
            onMouseMove={handleMouseMove}
            onMouseLeave={handleMouseLeave}
            style={{ rotateX, rotateY, transformStyle: "preserve-3d" }}
            className="w-full h-full perspective-1000"
        >
            {children}
        </motion.div>
    );
};

const Landing3 = () => {
    const navigate = useNavigate();
    const { scrollYProgress } = useScroll();
    const scaleX = useSpring(scrollYProgress, {
        stiffness: 100,
        damping: 30,
        restDelta: 0.001
    });

    const scrollToSection = (id) => {
        const element = document.getElementById(id);
        if (element) {
            element.scrollIntoView({ behavior: 'smooth' });
        }
    };

    return (
        <div className="bg-black text-white selection:bg-pink-500 selection:text-white overflow-x-hidden min-h-screen font-sans">
            {/* Scroll Progress Bar */}
            <motion.div
                className="fixed top-0 left-0 right-0 h-1 bg-gradient-to-r from-pink-500 to-violet-500 origin-[0%] z-50"
                style={{ scaleX }}
            />

            {/* 1) Navigation (Gellule) */}
            <nav className="fixed top-6 left-0 right-0 z-50 flex justify-center px-4">
                <motion.div
                    initial={{ y: -100, opacity: 0 }}
                    animate={{ y: 0, opacity: 1 }}
                    transition={{ delay: 0.5, duration: 0.8 }}
                    className="bg-black/80 backdrop-blur-md border border-white/10 rounded-full px-6 py-3 flex items-center gap-8 shadow-2xl"
                >
                    <div className="flex items-center gap-2 pr-4 border-r border-white/10">
                        <img src="/logoPSF.png" alt="Horse3" className="h-8 w-8 object-contain" />
                        <span className="font-display font-bold text-xl tracking-wider">HORSE3</span>
                    </div>

                    <div className="hidden md:flex gap-6 text-sm font-medium text-gray-400">
                        <button onClick={() => scrollToSection('features')} className="hover:text-white transition-colors">Fonctionnalités</button>
                        <button onClick={() => scrollToSection('pricing')} className="hover:text-white transition-colors">Tarifs</button>
                        <button onClick={() => scrollToSection('faq')} className="hover:text-white transition-colors">FAQ</button>
                    </div>

                    <div className="pl-4 border-l border-white/10">
                        <button
                            onClick={() => navigate('/register')}
                            className="bg-white text-black font-bold px-5 py-2 rounded-full text-sm hover:bg-gray-200 transition-colors"
                        >
                            Créer un compte
                        </button>
                    </div>
                </motion.div>
            </nav>

            {/* 2) Hero Section */}
            <section className="relative h-screen flex flex-col justify-center items-center overflow-hidden pt-20">
                {/* Background Effects */}
                <div className="absolute inset-0 z-0">
                    {/* Simplified gradient background similar to Landing2 but aligned with Landing3 style */}
                    <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-pink-600/20 rounded-full blur-[120px]" />
                    <div className="absolute bottom-[-10%] right-[-10%] w-[50%] h-[50%] bg-violet-600/20 rounded-full blur-[100px]" />
                </div>

                <div className="relative z-10 container mx-auto px-6 text-center max-w-5xl">
                    <motion.h1
                        initial={{ opacity: 0, y: 50 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8 }}
                        className="text-5xl md:text-8xl font-display font-bold tracking-tight mb-6 leading-[1.1]"
                    >
                        L'AVENIR DES COURSES<br />
                        <span className="bg-clip-text text-transparent bg-gradient-to-r from-pink-500 to-violet-500">
                            COMMENCE ICI
                        </span>
                    </motion.h1>

                    <motion.p
                        initial={{ opacity: 0, y: 30 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.2, duration: 0.8 }}
                        className="text-xl md:text-2xl text-gray-300 max-w-3xl mx-auto mb-12 font-light leading-relaxed"
                    >
                        Une interface fluide, des données en temps réel et une expérience utilisateur repensée.
                        Optimisez vos gains grâce à l'intelligence artificielle.
                    </motion.p>

                    <motion.div
                        initial={{ opacity: 0, scale: 0.9 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={{ delay: 0.4, duration: 0.5 }}
                        className="flex flex-col md:flex-row gap-4 justify-center"
                    >
                        <MagneticButton
                            onClick={() => navigate('/dashboard')}
                            className="bg-gradient-to-r from-pink-600 to-violet-600 text-white font-bold px-8 py-4 rounded-full text-lg hover:shadow-[0_0_30px_rgba(236,72,153,0.4)] transition-all flex items-center justify-center gap-2 group"
                        >
                            Entrer dans l'arène <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
                        </MagneticButton>
                        <MagneticButton
                            onClick={() => scrollToSection('pricing')}
                            className="bg-white/5 border border-white/10 hover:bg-white/10 text-white font-medium px-8 py-4 rounded-full text-lg transition-colors backdrop-blur-md"
                        >
                            Découvrir les offres
                        </MagneticButton>
                    </motion.div>
                </div>
            </section>

            {/* 3) Visual Sneak Peek */}
            <section className="py-20 px-6 relative z-10 -mt-20">
                <div className="container mx-auto">
                    <TiltContainer>
                        <motion.div
                            initial={{ opacity: 0, y: 100, rotateX: 20 }}
                            whileInView={{ opacity: 1, y: 0, rotateX: 0 }}
                            transition={{ duration: 1, ease: "easeOut" }}
                            viewport={{ once: true }}
                            className="relative rounded-[2rem] overflow-hidden border border-white/10 shadow-2xl shadow-violet-900/20 bg-black/40 backdrop-blur-xl aspect-video max-w-6xl mx-auto"
                        >
                            {/* Fake Interface Overlay / Mockup */}
                            <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-gray-900 to-black opacity-80">
                                <div className="text-center">
                                    <BarChart3 className="w-24 h-24 text-white/5 mx-auto mb-4" />
                                    <p className="text-white/20 font-display text-4xl font-bold uppercase tracking-widest">Dashboard Interface</p>
                                </div>
                            </div>

                            {/* Optional: Actual screenshot or image component here */}
                            {/* <img src="/dashboard-preview.png" alt="Dashboard" className="w-full h-full object-cover opacity-80" /> */}

                            {/* Decorative elements */}
                            <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-pink-500 via-violet-500 to-pink-500 opacity-50" />
                        </motion.div>
                    </TiltContainer>
                </div>
            </section>

            {/* 4) Benefices (Grid) */}
            <section id="features" className="py-32 px-6 bg-black relative z-10">
                <div className="container mx-auto">
                    <ScrollReveal>
                        <div className="text-center mb-20">
                            <h2 className="text-4xl md:text-6xl font-display font-bold mb-6">POURQUOI HORSE3 ?</h2>
                            <p className="text-gray-400 text-xl max-w-2xl mx-auto">La technologie au service de votre intuition pour des résultats concrets.</p>
                        </div>
                    </ScrollReveal>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <FeatureCard
                            title="Vitesse Éclair"
                            description="Optimisé pour la performance instantanée. Ne manquez jamais le départ d'une course grâce à notre flux live."
                            icon={<Zap className="w-8 h-8 text-yellow-400" />}
                            delay={0}
                        />
                        <FeatureCard
                            title="Innovation IA"
                            description="Des modèles prédictifs neuronaux analysent des milliers de paramètres pour dénicher la value invisible à l'œil nu."
                            icon={<Rocket className="w-8 h-8 text-pink-500" />}
                            delay={0.2}
                        />
                        <FeatureCard
                            title="Interface Intuitive"
                            description="Un design épuré qui met l'information essentielle au premier plan pour une prise de décision rapide."
                            icon={<LayoutIcon className="w-8 h-8 text-violet-500" />}
                            delay={0.4}
                        />
                    </div>
                </div>
            </section>

            {/* 5) Comment ca marche (3 Steps) */}
            <HowItWorks />

            {/* 6) Pricing / Offres */}
            <section id="pricing" className="py-32 px-6 bg-black relative">
                <div className="container mx-auto">
                    <ScrollReveal>
                        <div className="text-center mb-20">
                            <h2 className="text-4xl md:text-6xl font-display font-bold mb-6">NOS OFFRES</h2>
                            <p className="text-gray-400 text-xl">Du parieur curieux au turfiste professionnel.</p>
                        </div>
                    </ScrollReveal>

                    <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
                        <PricingCard
                            title="Freemium"
                            price="Gratuit"
                            features={["3 paris gratuits / jour", "Indicateurs Value", "Historique 3 jours"]}
                            unavailable={["Pas d'exotiques", "Pas de portfolio"]}
                            cta="Découvrir"
                            delay={0.1}
                        />
                        <PricingCard
                            title="Standard"
                            price="19€"
                            period="/mois"
                            features={["Tous les pronostics du jour", "Kelly + Mise suggérée", "ROI/PNL personnel", "Historique complet", "Alertes cotes"]}
                            cta="Choisir Standard"
                            featured={true}
                            delay={0.2}
                        />
                        <PricingCard
                            title="Premium"
                            price="39€"
                            period="/mois"
                            features={["Tout du Standard", "Algo Exotiques optimisés", "Portfolio complet", "Détection Steam/Drift", "Accès API Pronos"]}
                            cta="Passer Premium"
                            delay={0.3}
                        />
                        <PricingCard
                            title="Pro"
                            price="79€"
                            period="/mois"
                            features={["Tout du Premium", "Export CSV/API Illimité", "Analyse Multi-courses", "Backtest personnalisé", "Priorité futurs modules"]}
                            cta="Devenir Pro"
                            delay={0.4}
                        />
                    </div>
                </div>
            </section>

            {/* 7) Testimonials */}
            <section className="py-32 px-6 bg-neutral-900/20 relative overflow-hidden">
                {/* Background elements */}
                <div className="absolute top-0 right-0 w-[500px] h-[500px] bg-pink-600/10 rounded-full blur-[100px]" />

                <div className="container mx-auto relative z-10">
                    <ScrollReveal>
                        <h2 className="text-4xl font-display font-bold text-center mb-16">ILS ONT CHANGÉ DE DIMENSION</h2>
                    </ScrollReveal>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
                        <TestimonialCard
                            quote="Enfin un outil sérieux qui parle de mathématiques et non de superstition. Mon ROI est positif depuis 3 mois."
                            author="Thomas L."
                            role="Parieur régulier"
                            delay={0}
                        />
                        <TestimonialCard
                            quote="L'interface est incroyable. Avoir le calcul de mise Kelly intégré me fait gagner un temps précieux chaque matin."
                            author="Sarah M."
                            role="Turfiste passionnée"
                            delay={0.2}
                        />
                        <TestimonialCard
                            quote="Je suis passé à l'offre Pro pour l'API. C'est du niveau institutionnel. Bravo à l'équipe."
                            author="Marc D."
                            role="Trader sportif"
                            delay={0.4}
                        />
                    </div>
                </div>
            </section>

            {/* 8) FAQ */}
            <section id="faq" className="py-32 px-6 bg-black">
                <div className="container mx-auto max-w-3xl">
                    <ScrollReveal>
                        <h2 className="text-4xl font-display font-bold text-center mb-16">QUESTIONS FRÉQUENTES</h2>
                        <div className="space-y-4">
                            <FaqItem q="Comment fonctionne l'IA ?" a="Notre algorithme analyse des millions de données historiques, incluant la forme des chevaux, jockeys, météo et cotes pour prédire les issues les plus probables avec un taux de réussite élevé." />
                            <FaqItem q="Puis-je annuler à tout moment ?" a="Oui, absolument. L'abonnement est sans engagement. Vous pouvez l'arrêter en un clic depuis votre espace settings." />
                            <FaqItem q="Les paiements sont-ils sécurisés ?" a="Nous utilisons Stripe, le leader mondial des paiements en ligne. Vos données bancaires ne transitent jamais par nos serveurs." />
                        </div>
                        <div className="text-center mt-12">
                            <p className="text-gray-400 mb-4">D'autres questions ?</p>
                            <button className="text-pink-500 hover:text-pink-400 font-medium underline underline-offset-4">Contacter le support</button>
                        </div>
                    </ScrollReveal>
                </div>
            </section>

            {/* 9) Final CTA */}
            <section className="h-[70vh] flex flex-col items-center justify-center relative overflow-hidden bg-gradient-to-b from-black to-violet-950/30">
                <div className="absolute inset-0 bg-[url('/grid-pattern.svg')] opacity-10" />

                <div className="text-center relative z-10 px-6">
                    <ScrollReveal>
                        <h2 className="text-6xl md:text-9xl font-display font-black mb-8 tracking-tighter">
                            PRÊT ?
                        </h2>
                        <motion.button
                            whileHover={{ scale: 1.05 }}
                            whileTap={{ scale: 0.95 }}
                            onClick={() => navigate('/dashboard')}
                            className="px-16 py-8 bg-white text-black text-2xl font-bold rounded-full hover:bg-neutral-200 transition-colors shadow-[0_0_50px_-10px_rgba(255,255,255,0.4)]"
                        >
                            Entrer dans l'arène
                        </motion.button>
                    </ScrollReveal>
                </div>
            </section>

            {/* 10) Footer */}
            <Footer />
        </div>
    );
};

// --- Sub-components ---

const FeatureCard = ({ title, description, icon, delay }) => (
    <ScrollReveal delay={delay}>
        <div className="p-8 rounded-3xl bg-white/5 border border-white/10 hover:border-pink-500/30 hover:bg-white/10 transition-colors backdrop-blur-sm h-full">
            <div className="mb-6 p-4 rounded-2xl bg-white/5 w-fit">{icon}</div>
            <h3 className="text-2xl font-bold mb-4 text-white">{title}</h3>
            <p className="text-gray-400 leading-relaxed font-light">{description}</p>
        </div>
    </ScrollReveal>
);

// Icon helper
const LayoutIcon = ({ className }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className={className}><rect width="18" height="18" x="3" y="3" rx="2" ry="2" /><line x1="3" x2="21" y1="9" y2="9" /><line x1="9" x2="9" y1="21" y2="9" /></svg>
);

const StepByType = ({ number, title, description, color }) => (
    <div className="relative z-10 text-center group">
        <div className={`text-6xl font-display font-bold mb-6 opacity-20 group-hover:opacity-100 transition-opacity duration-500 ${color}`}>{number}</div>
        <h3 className="text-2xl font-bold mb-4">{title}</h3>
        <p className="text-gray-400">{description}</p>
    </div>
);
const HowItWorks = () => {
    const ref = useRef(null);
    const isInView = useInView(ref, { once: true, margin: "-20% 0px -20% 0px" });

    // Animation Timings (in seconds)
    const cardBorderDuration = 1.5;
    const lineDuration = 0.5;

    // Sequence:
    // 1. Card 1 Border: 0s -> 1.5s
    // 2. Line 1: 1.5s -> 2s
    // 3. Card 2 Border: 2s -> 3.5s
    // 4. Line 2: 3.5s -> 4s
    // 5. Card 3 Border: 4s -> 5.5s

    return (
        <section ref={ref} className="py-32 px-6 bg-gradient-to-b from-black to-gray-900 relative overflow-hidden">
            <div className="container mx-auto relative">
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={isInView ? { opacity: 1, y: 0 } : {}}
                    transition={{ duration: 0.8 }}
                >
                    <h2 className="text-4xl md:text-5xl font-display font-bold text-center mb-20">COMMENT ÇA MARCHE</h2>
                </motion.div>

                {/* Cards Container with Relative positioning for lines */}
                <div className="relative">

                    {/* Background Connecting Line (Gray) - Center of Col 1 to Center of Col 3 */}
                    <div className="hidden md:block absolute top-1/2 left-[16.66%] right-[16.66%] h-0.5 bg-white/5 -translate-y-1/2 z-0" />

                    {/* Active Laser Line 1: Col 1 Center to Col 2 Center */}
                    <div className="hidden md:block absolute top-1/2 left-[16.66%] w-[33.33%] h-0.5 -translate-y-1/2 z-0 overflow-hidden">
                        <motion.div
                            className="h-full w-full bg-gradient-to-r from-pink-500 to-violet-500 shadow-[0_0_10px_#ec4899]"
                            initial={{ x: "-100%" }}
                            animate={isInView ? { x: "0%" } : {}}
                            transition={{ duration: lineDuration, delay: cardBorderDuration, ease: "linear" }}
                        />
                    </div>

                    {/* Active Laser Line 2: Col 2 Center to Col 3 Center */}
                    <div className="hidden md:block absolute top-1/2 left-[50%] w-[33.33%] h-0.5 -translate-y-1/2 z-0 overflow-hidden">
                        <motion.div
                            className="h-full w-full bg-gradient-to-r from-violet-500 to-emerald-400 shadow-[0_0_10px_#8b5cf6]"
                            initial={{ x: "-100%" }}
                            animate={isInView ? { x: "0%" } : {}}
                            transition={{ duration: lineDuration, delay: cardBorderDuration + lineDuration + cardBorderDuration, ease: "linear" }}
                        />
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-12 relative z-10">

                        {/* Card 1 */}
                        <StepCard
                            number="01"
                            title="Analyse IA"
                            description="Nos algorithmes scannent les partants et calculent les probabilités réelles."
                            color="text-pink-500"
                            borderColor="#ec4899"
                            delay={0}
                            duration={cardBorderDuration}
                            isInView={isInView}
                        />

                        {/* Card 2 */}
                        <StepCard
                            number="02"
                            title="Optimisation Mise"
                            description="Le système Kelly Criterion vous suggère la mise idéale pour protéger votre bankroll."
                            color="text-violet-500"
                            borderColor="#8b5cf6"
                            delay={cardBorderDuration + lineDuration} // 2s
                            duration={cardBorderDuration}
                            isInView={isInView}
                        />

                        {/* Card 3 */}
                        <StepCard
                            number="03"
                            title="Profitez"
                            description="Placez vos paris avec confiance et suivez l'évolution de votre ROI en temps réel."
                            color="text-emerald-400"
                            borderColor="#34d399"
                            delay={cardBorderDuration + lineDuration + cardBorderDuration + lineDuration} // 4s
                            duration={cardBorderDuration}
                            isInView={isInView}
                        />
                    </div>
                </div>
            </div>
        </section>
    );
};

const StepCard = ({ number, title, description, color, borderColor, delay, duration, isInView }) => {
    // Note: rounded-3xl is 1.5rem = 24px
    return (
        <motion.div
            className="relative z-10 text-center group p-8 rounded-3xl bg-gray-900/80 border border-white/5 backdrop-blur-sm h-full"
            initial={{ opacity: 0, scale: 0.9 }}
            animate={isInView ? { opacity: 1, scale: 1 } : {}}
            transition={{ duration: 0.5, delay: delay }} // Fade in at start of its sequence
        >
            {/* Animated Border Overlay */}
            <svg className="absolute inset-0 w-full h-full pointer-events-none rounded-3xl overflow-visible">
                <motion.rect
                    x="2" y="2"
                    width="calc(100% - 4px)"
                    height="calc(100% - 4px)"
                    rx="22" ry="22" // Slightly less than 24 to fit inside
                    fill="none"
                    stroke={borderColor}
                    strokeWidth="2"
                    strokeLinecap="round"
                    initial={{ pathLength: 0 }}
                    animate={isInView ? { pathLength: 1 } : {}}
                    transition={{ duration: duration, delay: delay, ease: "linear" }}
                    style={{ filter: `drop-shadow(0 0 8px ${borderColor})` }}
                />
            </svg>

            <div className={`text-6xl font-display font-bold mb-6 opacity-40 group-hover:opacity-100 group-hover:scale-110 transition-all duration-500 ${color}`}>{number}</div>
            <h3 className="text-2xl font-bold mb-4 text-white group-hover:text-pink-100 transition-colors">{title}</h3>
            <p className="text-gray-400 group-hover:text-gray-300 transition-colors">{description}</p>
        </motion.div>
    );
};

const PricingCard = ({ title, price, period, features, unavailable, cta, featured, delay }) => (
    <ScrollReveal delay={delay}>
        <div className={`h-full p-8 rounded-3xl flex flex-col transition-all duration-300 relative ${featured ? 'bg-neutral-800/80 border-pink-500/50 shadow-2xl shadow-pink-900/20 scale-105 z-10' : 'bg-neutral-900/50 border border-white/5 hover:border-white/10'}`}>
            {featured && (
                <div className="absolute top-0 right-0 bg-pink-500 text-white text-[10px] font-bold px-3 py-1 rounded-bl-xl rounded-tr-3xl tracking-wider">POPULAIRE</div>
            )}
            <h3 className={`text-xl font-bold mb-2 ${featured ? 'text-white' : 'text-gray-400'}`}>{title}</h3>
            <div className="mb-6 flex items-baseline gap-1">
                <span className="text-4xl font-bold">{price}</span>
                {period && <span className="text-sm text-gray-500">{period}</span>}
            </div>

            <div className="flex-grow space-y-4 mb-8">
                {features.map((feat, i) => (
                    <div key={i} className="flex items-start gap-3 text-sm text-gray-300">
                        <CheckCircle className={`w-5 h-5 flex-shrink-0 ${featured ? 'text-pink-500' : 'text-gray-500'}`} />
                        <span>{feat}</span>
                    </div>
                ))}
                {unavailable && unavailable.map((feat, i) => (
                    <div key={i} className="flex items-start gap-3 text-sm text-gray-600">
                        <Lock className="w-5 h-5 flex-shrink-0" />
                        <span>{feat}</span>
                    </div>
                ))}
            </div>

            <button className={`w-full py-4 rounded-xl font-bold text-sm transition-all ${featured ? 'bg-pink-600 hover:bg-pink-700 text-white shadow-lg shadow-pink-600/20' : 'bg-white/10 hover:bg-white/20 text-white'}`}>
                {cta}
            </button>
        </div>
    </ScrollReveal>
);

const TestimonialCard = ({ quote, author, role, delay }) => (
    <ScrollReveal delay={delay}>
        <div className="p-8 rounded-2xl bg-black border border-white/10 relative">
            <div className="text-pink-500 text-4xl font-serif absolute top-4 left-6 opacity-30">"</div>
            <p className="text-lg text-gray-300 mb-6 relative z-10 italic leading-relaxed">
                {quote}
            </p>
            <div>
                <p className="font-bold text-white">{author}</p>
                <p className="text-sm text-gray-500">{role}</p>
            </div>
        </div>
    </ScrollReveal>
);

const FaqItem = ({ q, a }) => {
    const [isOpen, setIsOpen] = useState(false);
    return (
        <div className="border border-white/10 rounded-2xl bg-white/5 overflow-hidden transition-colors hover:bg-white/10">
            <button
                onClick={() => setIsOpen(!isOpen)}
                className="w-full flex items-center justify-between p-6 text-left"
            >
                <span className="font-bold text-lg">{q}</span>
                <ChevronDown className={`w-5 h-5 transition-transform ${isOpen ? 'rotate-180' : ''}`} />
            </button>
            <motion.div
                initial={false}
                animate={{ height: isOpen ? 'auto' : 0, opacity: isOpen ? 1 : 0 }}
                className="overflow-hidden bg-black/20"
            >
                <p className="p-6 pt-0 text-gray-400 leading-relaxed">{a}</p>
            </motion.div>
        </div>
    );
}

const Footer = () => (
    <footer className="py-12 bg-black border-t border-white/10 text-center md:text-left">
        <div className="container mx-auto px-6 grid grid-cols-1 md:grid-cols-4 gap-12">
            <div>
                <div className="flex items-center gap-2 mb-4 justify-center md:justify-start">
                    <img src="/logoPSF.png" alt="Logo" className="h-6 w-auto" />
                    <span className="font-display font-bold text-lg">HORSE3</span>
                </div>
                <p className="text-gray-500 text-sm">Le partenaire de confiance des parieurs modernes.</p>
            </div>

            {/* Columns... simplified for brevity but functional */}
            <div>
                <h4 className="font-bold mb-4">Produit</h4>
                <ul className="space-y-2 text-sm text-gray-500">
                    <li><a href="#" className="hover:text-white">Fonctionnalités</a></li>
                    <li><a href="#" className="hover:text-white">Tarifs</a></li>
                    <li><a href="#" className="hover:text-white">API</a></li>
                </ul>
            </div>
            <div>
                <h4 className="font-bold mb-4">Légal</h4>
                <ul className="space-y-2 text-sm text-gray-500">
                    <li><a href="#" className="hover:text-white">CGV / CGU</a></li>
                    <li><a href="#" className="hover:text-white">Confidentialité</a></li>
                    <li><a href="#" className="hover:text-white">Mentions légales</a></li>
                </ul>
            </div>
            <div>
                <h4 className="font-bold mb-4">Support</h4>
                <ul className="space-y-2 text-sm text-gray-500">
                    <li><a href="#" className="hover:text-white">Centre d'aide</a></li>
                    <li><a href="#" className="hover:text-white">Contact</a></li>
                </ul>
            </div>
        </div>
        <div className="text-center text-gray-600 text-xs mt-12 pt-8 border-t border-white/5">
            © {new Date().getFullYear()} HORSE3. All rights reserved.
        </div>
    </footer>
);

export default Landing3;
