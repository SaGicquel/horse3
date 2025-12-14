import { describe, expect, it } from 'vitest';
import { BUCKET_CLASSES, getBucketClassNames, normalizeBucketLabel } from './buckets';

describe('bucket class mapping', () => {
  it('returns tailored classes for known buckets', () => {
    expect(getBucketClassNames('SÛR')).toBe(BUCKET_CLASSES['SÛR']);
    expect(getBucketClassNames('ÉQUIL.')).toBe(BUCKET_CLASSES['ÉQUIL.']);
    expect(getBucketClassNames('RISQ.')).toBe(BUCKET_CLASSES['RISQ.']);
  });

  it('normalizes aliases and accents', () => {
    expect(getBucketClassNames('sur')).toBe(BUCKET_CLASSES['SÛR']);
    expect(getBucketClassNames('equilibre')).toBe(BUCKET_CLASSES['ÉQUIL.']);
    expect(getBucketClassNames('Risque')).toBe(BUCKET_CLASSES['RISQ.']);
  });

  it('falls back to neutral styling when unknown', () => {
    expect(getBucketClassNames('INCONNU')).toContain('bg-white/10');
  });
});

describe('bucket label normalization', () => {
  it('returns canonical labels', () => {
    expect(normalizeBucketLabel('sur')).toBe('SÛR');
    expect(normalizeBucketLabel('equil.')).toBe('ÉQUIL.');
    expect(normalizeBucketLabel('risque')).toBe('RISQ.');
  });

  it('preserves original value when not recognized', () => {
    expect(normalizeBucketLabel('Custom')).toBe('Custom');
    expect(normalizeBucketLabel(undefined)).toBe('');
  });
});
