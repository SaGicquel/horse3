/**
 * Composant Chat GPT spécialisé pour l'analyse des courses hippiques
 * 
 * Fonctionnalités :
 * - Interface de chat moderne avec messages
 * - Suggestions intelligentes basées sur les données disponibles
 * - Historique de conversation
 * - Indicateur de frappe en cours
 * - Mode sombre/clair automatique
 */

import { useState, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Send, Sparkles, Loader2, Bot, User, MessageSquare, TrendingUp, BarChart3, Users, Calendar, HelpCircle } from 'lucide-react';
import { API_BASE } from '../config/api';

const messageVariants = {
  initial: { opacity: 0, y: 20, scale: 0.95 },
  animate: { 
    opacity: 1, 
    y: 0, 
    scale: 1,
    transition: { type: "spring", damping: 25, stiffness: 300 }
  },
  exit: { opacity: 0, scale: 0.95 }
};

const typingVariants = {
  initial: { opacity: 0 },
  animate: { 
    opacity: 1,
    transition: { staggerChildren: 0.1 }
  }
};

const dotVariants = {
  animate: {
    y: [0, -5, 0],
    transition: { duration: 0.6, repeat: Infinity }
  }
};

const Chat = () => {
  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: 'Bonjour ! Je suis votre assistant spécialisé en analyse des courses hippiques. Je peux vous aider à analyser les performances des chevaux, comprendre les statistiques, identifier les tendances, et bien plus encore. Que souhaitez-vous savoir ?',
      timestamp: new Date()
    }
  ]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [showSuggestions, setShowSuggestions] = useState(true);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  const suggestions = [
    {
      icon: TrendingUp,
      title: 'Analyse de performance',
      question: 'Quels sont les chevaux les plus performants cette année ?',
      color: 'var(--color-primary)'
    },
    {
      icon: BarChart3,
      title: 'Statistiques par race',
      question: 'Quelles races de chevaux ont le meilleur taux de victoire ?',
      color: 'var(--color-secondary)'
    },
    {
      icon: Users,
      title: 'Analyse jockey/entraîneur',
      question: 'Quels sont les meilleurs jockeys et entraîneurs selon les statistiques ?',
      color: 'var(--color-primary-light)'
    },
    {
      icon: Calendar,
      title: 'Tendances temporelles',
      question: 'Comment évoluent les performances au fil des années ?',
      color: 'var(--color-success)'
    },
    {
      icon: HelpCircle,
      title: 'Conseils stratégiques',
      question: 'Quelles sont les meilleures stratégies pour analyser une course ?',
      color: 'var(--color-muted)'
    }
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSuggestionClick = (question) => {
    setInput(question);
    setShowSuggestions(false);
    inputRef.current?.focus();
  };

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMessage = {
      role: 'user',
      content: input.trim(),
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);
    setShowSuggestions(false);

    try {
      const response = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage.content,
          history: messages.map(m => ({
            role: m.role,
            content: m.content
          }))
        })
      });

      if (!response.ok) {
        throw new Error('Erreur lors de la communication avec le serveur');
      }

      const data = await response.json();
      
      const assistantMessage = {
        role: 'assistant',
        content: data.response || 'Je n\'ai pas pu générer de réponse. Veuillez réessayer.',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Erreur:', error);
      const errorMessage = {
        role: 'assistant',
        content: 'Désolé, une erreur est survenue. Veuillez vérifier que le serveur est bien démarré et que votre clé API OpenAI est configurée.',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="flex flex-col h-full max-h-[800px]">
      {/* Header */}
      <motion.div 
        className="flex items-center gap-3 p-4 border-b"
        style={{ 
          borderColor: 'rgba(var(--color-border-rgb), 0.1)',
          background: 'rgba(var(--color-card-rgb, 255, 255, 255), 0.55)',
          backdropFilter: 'blur(24px)',
          WebkitBackdropFilter: 'blur(24px)'
        }}
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <motion.div 
          className="flex h-10 w-10 items-center justify-center rounded-xl relative overflow-hidden"
          style={{
            background: 'linear-gradient(135deg, var(--color-primary), var(--color-primary-light))',
            color: 'white'
          }}
          whileHover={{ scale: 1.1, rotate: 5 }}
        >
          <Sparkles size={20} />
          <div className="absolute inset-0 bg-white/20 animate-shimmer" />
        </motion.div>
        <div className="flex-1">
          <h3 
            className="text-sm font-semibold"
            style={{ color: 'var(--color-text)' }}
          >
            Assistant IA - Analyse Hippique
          </h3>
          <p 
            className="text-xs flex items-center gap-1"
            style={{ color: 'var(--color-muted)' }}
          >
            <motion.span 
              className="w-2 h-2 rounded-full bg-green-500"
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
            Spécialisé dans l'analyse des courses et statistiques
          </p>
        </div>
      </motion.div>

      {/* Messages Area */}
      <div 
        className="flex-1 overflow-y-auto p-4 space-y-4"
        style={{ backgroundColor: 'var(--color-bg)' }}
      >
        <AnimatePresence mode="popLayout">
          {messages.map((message, index) => (
            <motion.div
              key={index}
              className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
              variants={messageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
              layout
            >
              {message.role === 'assistant' && (
                <motion.div 
                  className="flex h-8 w-8 items-center justify-center rounded-full flex-shrink-0"
                  style={{
                    background: 'linear-gradient(135deg, var(--color-primary), var(--color-primary-light))',
                    color: 'white'
                  }}
                  whileHover={{ scale: 1.1 }}
                >
                  <Bot size={16} />
                </motion.div>
              )}
              
              <motion.div 
                className={`max-w-[80%] rounded-2xl px-4 py-3 ${
                  message.role === 'user' 
                    ? 'rounded-br-sm' 
                    : 'rounded-bl-sm'
                }`}
                style={{
                  backgroundColor: message.role === 'user' 
                    ? 'var(--color-primary)' 
                    : 'var(--color-card)',
                  color: message.role === 'user' 
                    ? 'white' 
                    : 'var(--color-text)',
                  border: message.role === 'assistant' 
                    ? '1px solid var(--color-border)' 
                    : 'none',
                  boxShadow: message.role === 'user' 
                    ? '0 4px 15px rgba(var(--color-primary-rgb), 0.3)' 
                    : '0 2px 10px rgba(0,0,0,0.05)'
                }}
                whileHover={{ scale: 1.01 }}
              >
                <p className="text-sm whitespace-pre-wrap leading-relaxed">
                  {message.content}
                </p>
                <p 
                  className="text-xs mt-2 opacity-70"
                >
                  {message.timestamp.toLocaleTimeString('fr-FR', { 
                    hour: '2-digit', 
                    minute: '2-digit' 
                  })}
                </p>
              </motion.div>

              {message.role === 'user' && (
                <motion.div 
                  className="flex h-8 w-8 items-center justify-center rounded-full flex-shrink-0"
                  style={{
                    backgroundColor: 'var(--color-secondary)',
                    color: 'var(--color-primary)'
                  }}
                  whileHover={{ scale: 1.1 }}
                >
                  <User size={16} />
                </motion.div>
              )}
            </motion.div>
          ))}
        </AnimatePresence>

        {/* Loading indicator with typing animation */}
        <AnimatePresence>
          {isLoading && (
            <motion.div 
              className="flex gap-3 justify-start"
              variants={messageVariants}
              initial="initial"
              animate="animate"
              exit="exit"
            >
              <motion.div 
                className="flex h-8 w-8 items-center justify-center rounded-full flex-shrink-0"
                style={{
                  background: 'linear-gradient(135deg, var(--color-primary), var(--color-primary-light))',
                  color: 'white'
                }}
                animate={{ rotate: [0, 360] }}
                transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
              >
                <Bot size={16} />
              </motion.div>
              <div 
                className="rounded-2xl rounded-bl-sm px-4 py-3"
                style={{
                  backgroundColor: 'var(--color-card)',
                  border: '1px solid var(--color-border)'
                }}
              >
                <motion.div 
                  className="flex gap-1 items-center"
                  variants={typingVariants}
                  initial="initial"
                  animate="animate"
                >
                  <span className="text-sm mr-2" style={{ color: 'var(--color-muted)' }}>
                    L'assistant réfléchit
                  </span>
                  {[0, 1, 2].map((i) => (
                    <motion.span
                      key={i}
                      className="w-2 h-2 rounded-full"
                      style={{ backgroundColor: 'var(--color-primary)' }}
                      variants={dotVariants}
                      animate="animate"
                      transition={{ delay: i * 0.15 }}
                    />
                  ))}
                </motion.div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        <div ref={messagesEndRef} />
      </div>

      {/* Suggestions */}
      <AnimatePresence>
        {showSuggestions && messages.length === 1 && (
          <motion.div 
            className="p-4 border-t"
            style={{ 
              borderColor: 'var(--color-border)',
              backgroundColor: 'var(--color-card)'
            }}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 20 }}
          >
            <p 
              className="text-xs uppercase tracking-wider mb-3 flex items-center gap-2"
              style={{ color: 'var(--color-muted)' }}
            >
              <Sparkles size={12} />
              Suggestions
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              {suggestions.map((suggestion, index) => {
                const Icon = suggestion.icon;
                return (
                  <motion.button
                    key={index}
                    onClick={() => handleSuggestionClick(suggestion.question)}
                    className="flex items-center gap-3 p-3 rounded-xl text-left transition-all group relative overflow-hidden"
                    style={{
                      backgroundColor: 'var(--color-bg)',
                      border: '1px solid var(--color-border)',
                      color: 'var(--color-text)'
                    }}
                    initial={{ opacity: 0, y: 10 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: index * 0.1 }}
                    whileHover={{ 
                      scale: 1.02, 
                      borderColor: suggestion.color,
                      backgroundColor: 'var(--color-secondary)'
                    }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <div 
                      className="w-8 h-8 rounded-lg flex items-center justify-center flex-shrink-0 transition-transform group-hover:scale-110"
                      style={{ backgroundColor: `${suggestion.color}20` }}
                    >
                      <Icon size={16} style={{ color: suggestion.color }} />
                    </div>
                    <div className="flex-1 min-w-0">
                      <p className="text-xs font-medium truncate">{suggestion.title}</p>
                      <p 
                        className="text-xs truncate"
                        style={{ color: 'var(--color-muted)' }}
                      >
                        {suggestion.question}
                      </p>
                    </div>
                  </motion.button>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Input Area */}
      <motion.div 
        className="p-4 border-t backdrop-blur-md"
        style={{ 
          borderColor: 'var(--color-border)',
          backgroundColor: 'rgba(var(--color-card-rgb), 0.9)'
        }}
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <div className="flex gap-2">
          <motion.input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            placeholder="Posez votre question sur les courses hippiques..."
            className="flex-1 px-4 py-3 rounded-xl text-sm focus:outline-none transition-all"
            style={{
              backgroundColor: 'var(--color-bg)',
              border: '1px solid var(--color-border)',
              color: 'var(--color-text)'
            }}
            disabled={isLoading}
            whileFocus={{ 
              boxShadow: '0 0 0 3px rgba(var(--color-primary-rgb), 0.2)',
              borderColor: 'var(--color-primary)'
            }}
          />
          <motion.button
            onClick={handleSend}
            disabled={!input.trim() || isLoading}
            className="px-4 py-3 rounded-xl transition-all disabled:opacity-50 disabled:cursor-not-allowed relative overflow-hidden"
            style={{
              background: input.trim() && !isLoading 
                ? 'linear-gradient(135deg, var(--color-primary), var(--color-primary-light))' 
                : 'var(--color-muted)',
              color: 'white'
            }}
            whileHover={{ scale: input.trim() && !isLoading ? 1.05 : 1 }}
            whileTap={{ scale: input.trim() && !isLoading ? 0.95 : 1 }}
          >
            {isLoading ? (
              <Loader2 size={20} className="animate-spin" />
            ) : (
              <Send size={20} />
            )}
          </motion.button>
        </div>
        <p 
          className="text-xs mt-2 text-center"
          style={{ color: 'var(--color-muted)' }}
        >
          Appuyez sur Entrée pour envoyer
        </p>
      </motion.div>
    </div>
  );
};

export default Chat;

