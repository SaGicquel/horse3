import { afterEach, vi } from 'vitest';

// Reset mocks/stubs between tests to keep environments isolated
afterEach(() => {
  vi.restoreAllMocks();
  vi.unstubAllEnvs();
  vi.unstubAllGlobals();
});
