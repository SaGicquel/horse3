import { describe, expect, it, vi } from 'vitest';

const loadEnv = async () => {
  vi.resetModules();
  return import('./env');
};

describe('env resolution', () => {
  it('prefers window.__APP_CONFIG__ over build-time env', async () => {
    vi.stubGlobal('window', {
      __APP_CONFIG__: { apiBaseUrl: 'https://runtime.example/api/' },
      location: { hostname: 'localhost', port: '5173' },
    } as any);
    vi.stubEnv('VITE_API_URL', 'https://env.example/api');

    const { env } = await loadEnv();
    expect(env.API_BASE_URL).toBe('https://runtime.example/api');
    expect(env.API_BASE).toBe('https://runtime.example');
  });

  it('falls back to VITE_API_URL when no runtime config is injected', async () => {
    vi.stubGlobal('window', { location: { hostname: 'localhost', port: '5173' } } as any);
    vi.stubEnv('VITE_API_URL', 'https://env-only.example/api');

    const { env } = await loadEnv();
    expect(env.API_BASE_URL).toBe('https://env-only.example/api');
    expect(env.API_BASE).toBe('https://env-only.example');
  });

  it('defaults to localhost API when nothing else is provided', async () => {
    vi.stubGlobal('window', undefined as any);
    vi.stubEnv('VITE_API_URL', '');

    const { env } = await loadEnv();
    expect(env.API_BASE_URL).toBe('http://localhost:8000/api');
    expect(env.API_BASE).toBe('http://localhost:8000');
  });
});
