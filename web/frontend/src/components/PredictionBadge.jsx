import React from 'react';
import { ShieldCheck, ShieldAlert, Shield, Cpu, BrainCircuit } from 'lucide-react';

const PredictionBadge = ({ score, provider, showLabel = true }) => {
  let colorClass = 'bg-neutral-100 text-neutral-600 border-neutral-200';
  let Icon = Shield;
  let label = 'Incertain';

  if (score >= 0.8) {
    colorClass = 'bg-success/15 text-success border-success/30';
    Icon = ShieldCheck;
    label = 'Haute Confiance';
  } else if (score >= 0.5) {
    colorClass = 'bg-warning/15 text-warning border-warning/30';
    Icon = Shield;
    label = 'Confiance Moyenne';
  } else {
    colorClass = 'bg-error/15 text-error border-error/30';
    Icon = ShieldAlert;
    label = 'Faible Confiance';
  }

  const ProviderIcon = provider?.toLowerCase().includes('openai') || provider?.toLowerCase().includes('gemini')
    ? BrainCircuit
    : Cpu;

  return (
    <div className={`flex items-center gap-2 px-3 py-1.5 rounded-full border ${colorClass}`}>
      <Icon size={14} />
      {showLabel && <span className="text-xs font-bold">{label} ({Math.round(score * 100)}%)</span>}

      {provider && (
        <div className="flex items-center gap-1 pl-2 ml-2 border-l border-current/20">
          <ProviderIcon size={12} />
          <span className="text-[10px] uppercase tracking-wider opacity-80">{provider}</span>
        </div>
      )}
    </div>
  );
};

export default PredictionBadge;
