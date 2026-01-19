/**
 * PageHeader - Composant de header unifié pour toutes les pages
 * Assure une cohérence visuelle sur l'ensemble de l'application
 */

import { motion } from 'framer-motion';

const PageHeader = ({
    emoji,
    title,
    subtitle,
    children // Pour les actions (boutons refresh, filtres, etc.)
}) => {
    return (
        <motion.header
            className="mb-6 sm:mb-8 flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4"
            initial={{ opacity: 0, y: -20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
        >
            <div>
                <motion.h1
                    className="text-2xl sm:text-3xl md:text-4xl font-bold text-neutral-900 dark:text-neutral-100 flex items-center gap-3"
                    initial={{ opacity: 0, x: -20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    {emoji && (
                        <motion.span
                            className="text-2xl sm:text-3xl"
                            animate={{ rotate: [0, 5, -5, 0] }}
                            transition={{ duration: 2, repeat: Infinity, repeatDelay: 4 }}
                        >
                            {emoji}
                        </motion.span>
                    )}
                    {title}
                </motion.h1>
                {subtitle && (
                    <motion.p
                        className="text-sm sm:text-base mt-1 text-neutral-700 dark:text-neutral-400"
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        transition={{ delay: 0.3 }}
                    >
                        {subtitle}
                    </motion.p>
                )}
            </div>

            {children && (
                <motion.div
                    className="flex items-center gap-3 flex-wrap"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.4 }}
                >
                    {children}
                </motion.div>
            )}
        </motion.header>
    );
};

export default PageHeader;
