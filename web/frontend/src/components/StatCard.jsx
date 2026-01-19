/**
 * StatCard - Carte de statistiques avec effet Glassmorphism parfait
 *
 * Caractéristiques :
 * - Effet glassmorphism moderne avec backdrop-blur
 * - Animations douces au hover
 * - Bordures subtiles avec reflets
 * - Support mode sombre avec contraste AA
 */

const StatCard = ({ title, value, evolution, icon: Icon, trend = 'up' }) => {
  const isPositive = trend === 'up';

  return (
    <div
      className="rounded-[20px] p-6 glass-card glass-shimmer"
      style={{
        background: 'rgba(var(--color-card-rgb), 0.5)',
        backdropFilter: 'blur(24px) saturate(180%)',
        WebkitBackdropFilter: 'blur(24px) saturate(180%)',
        border: '1px solid rgba(255, 255, 255, 0.18)',
        boxShadow: '0 8px 32px rgba(0, 0, 0, 0.08), inset 0 1px 0 rgba(255, 255, 255, 0.4)'
      }}
    >
      <div className="flex items-center justify-between">
        <div
          className="p-2.5 rounded-xl transition-all duration-300"
          style={isPositive ? {
            background: 'linear-gradient(135deg, rgba(var(--color-primary-rgb), 0.15), rgba(var(--color-primary-rgb), 0.05))',
            color: 'var(--color-primary)',
            border: '1px solid rgba(var(--color-primary-rgb), 0.2)',
            boxShadow: '0 4px 12px rgba(var(--color-primary-rgb), 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
          } : {
            background: 'linear-gradient(135deg, rgba(220, 38, 38, 0.15), rgba(220, 38, 38, 0.05))',
            color: 'var(--color-error)',
            border: '1px solid rgba(220, 38, 38, 0.2)',
            boxShadow: '0 4px 12px rgba(220, 38, 38, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.3)'
          }}
        >
          <Icon size={20} />
        </div>
        <div
          className="text-sm font-semibold font-['Inter'] px-3 py-1 rounded-full"
          style={{
            color: isPositive ? 'var(--color-primary)' : 'var(--color-error)',
            background: isPositive
              ? 'rgba(var(--color-primary-rgb), 0.1)'
              : 'rgba(220, 38, 38, 0.1)',
            border: `1px solid ${isPositive ? 'rgba(var(--color-primary-rgb), 0.2)' : 'rgba(220, 38, 38, 0.2)'}`
          }}
        >
          {evolution}
        </div>
      </div>
      <div className="mt-4">
        <p
          className="text-3xl font-bold font-['Inter']"
          style={{ color: 'var(--color-text)' }}
        >
          {value}
        </p>
        <p
          className="text-sm font-['Inter'] mt-1"
          style={{ color: 'var(--color-muted)' }}
        >
          {title}
        </p>
      </div>
    </div>
  );
};

export default StatCard;
