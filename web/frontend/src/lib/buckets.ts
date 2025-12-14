export type RiskBucket = 'SÛR' | 'ÉQUIL.' | 'RISQ.';

const BUCKET_CLASSES: Record<RiskBucket, string> = {
  'SÛR': 'bg-emerald-500/15 text-emerald-200 border border-emerald-400/30',
  'ÉQUIL.': 'bg-amber-500/15 text-amber-200 border border-amber-400/30',
  'RISQ.': 'bg-rose-500/15 text-rose-200 border border-rose-400/30',
};

const normalizeKey = (bucket?: string | null): RiskBucket | null => {
  if (!bucket) return null;

  const normalized = bucket
    .toString()
    .trim()
    .toUpperCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '') // remove accents
    .replace(/\.+$/, '') // drop trailing dot for matching
    .replace(/\s+/g, '');

  if (normalized === 'SUR') return 'SÛR';
  if (normalized === 'EQUIL' || normalized === 'EQUILIBRE') return 'ÉQUIL.';
  if (normalized === 'RISQ' || normalized === 'RISQUE') return 'RISQ.';

  // Already canonical?
  if (bucket === 'SÛR' || bucket === 'ÉQUIL.' || bucket === 'RISQ.') {
    return bucket as RiskBucket;
  }

  return null;
};

/**
 * Retourne les classes Tailwind pour un bucket de risque.
 * Normalise les variantes d'écriture (accents, points, majuscules).
 */
export const getBucketClassNames = (bucket?: string | null): string => {
  const key = normalizeKey(bucket);
  if (!key) {
    return 'bg-white/10 text-gray-300 border border-white/10';
  }
  return BUCKET_CLASSES[key];
};

/**
 * Retourne le libellé canonique du bucket (SÛR/ÉQUIL./RISQ.)
 * ou la valeur initiale trim si non reconnue.
 */
export const normalizeBucketLabel = (bucket?: string | null): string => {
  const key = normalizeKey(bucket);
  if (!key) {
    return (bucket ?? '').toString().trim();
  }
  return key;
};

export { BUCKET_CLASSES };
