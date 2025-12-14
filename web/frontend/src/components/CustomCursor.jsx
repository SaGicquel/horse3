import { useEffect, useState } from 'react';
import { motion, useSpring, useReducedMotion } from 'framer-motion';

export default function CustomCursor() {
  const [pos, setPos] = useState({ x: -100, y: -100 });
  const [hoverType, setHoverType] = useState(null);
  const [isTouch, setIsTouch] = useState(false);
  const shouldReduceMotion = useReducedMotion();

  // spring for smooth movement
  const springConfig = { damping: 24, stiffness: 300 };
  const springX = useSpring(pos.x, springConfig);
  const springY = useSpring(pos.y, springConfig);

  useEffect(() => {
    const touchDevice = 'ontouchstart' in window || (navigator?.maxTouchPoints || 0) > 0;
    setIsTouch(touchDevice);
  }, []);

  useEffect(() => {
    if (isTouch || shouldReduceMotion) return undefined;

    const move = (e) => {
      setPos({ x: e.clientX, y: e.clientY });
    };

    const onEnter = (e) => {
      const t = e.target.closest && e.target.closest('[data-cursor]');
      if (t) setHoverType(t.getAttribute('data-cursor'));
    };

    const onLeave = (e) => {
      const t = e.target.closest && e.target.closest('[data-cursor]');
      if (t) setHoverType(null);
    };

    window.addEventListener('mousemove', move);
    window.addEventListener('mouseover', onEnter, true);
    window.addEventListener('mouseout', onLeave, true);

    return () => {
      window.removeEventListener('mousemove', move);
      window.removeEventListener('mouseover', onEnter, true);
      window.removeEventListener('mouseout', onLeave, true);
    };
  }, [isTouch, shouldReduceMotion]);

  if (isTouch || shouldReduceMotion) {
    return null;
  }

  return (
    <div className="pointer-events-none fixed inset-0 z-50 select-none">
      <motion.div
        className={`custom-cursor cursor-dot ${hoverType ? 'cursor-hover' : ''}`}
        style={{ x: springX, y: springY }}
        animate={{ scale: hoverType === 'link' ? 1.8 : hoverType === 'action' ? 1.4 : 1 }}
        transition={{ type: 'spring', damping: 20, stiffness: 400 }}
      />

      <motion.div
        className={`custom-cursor cursor-halo ${hoverType ? 'cursor-hover' : ''}`}
        style={{ x: springX, y: springY }}
        animate={{ opacity: hoverType ? 0.5 : 0.2 }}
        transition={{ duration: 0.25 }}
      />
    </div>
  );
}
