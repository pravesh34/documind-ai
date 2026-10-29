/**
 * Single source of truth for talking to the FastAPI backend.
 *
 * The Vite dev server proxies `/api` to http://localhost:8000, and in
 * production FastAPI serves the built frontend from the same origin, so
 * relative URLs work in both modes.
 */

export const API_BASE = '/api/v1';

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly requestId?: string;

  constructor(status: number, code: string, message: string, requestId?: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

type RequestOptions = Omit<RequestInit, 'body'> & { body?: unknown };

function errorMessage(payload: unknown, fallback: string): string {
  if (payload && typeof payload === 'object') {
    const record = payload as Record<string, unknown>;
    const error = record.error;
    if (typeof error === 'string') return error;
    if (error && typeof error === 'object') {
      const detail = (error as Record<string, unknown>).message;
      if (typeof detail === 'string') return detail;
    }
    if (typeof record.detail === 'string') return record.detail;
  }
  return fallback;
}

const API_KEY_STORAGE = 'documind.apiKey';

export function getApiKey(): string | null {
  try {
    return window.localStorage.getItem(API_KEY_STORAGE);
  } catch {
    return null;
  }
}

export function setApiKey(key: string | null): void {
  try {
    if (key) window.localStorage.setItem(API_KEY_STORAGE, key);
    else window.localStorage.removeItem(API_KEY_STORAGE);
  } catch {
    /* storage may be unavailable (private mode) */
  }
}

function authHeaders(): Record<string, string> {
  const key = getApiKey();
  return key ? { 'x-api-key': key } : {};
}

export async function apiFetch<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, headers, ...rest } = options;
  const isFormData = typeof FormData !== 'undefined' && body instanceof FormData;

  const response = await fetch(`${API_BASE}${path}`, {
    ...rest,
    headers: {
      Accept: 'application/json',
      ...authHeaders(),
      ...(isFormData || body === undefined ? {} : { 'Content-Type': 'application/json' }),
      ...(headers as Record<string, string> | undefined),
    },
    body: body === undefined ? undefined : isFormData ? (body as FormData) : JSON.stringify(body),
  });

  if (response.status === 204) {
    return undefined as unknown as T;
  }

  const contentType = response.headers.get('content-type') ?? '';
  const payload: unknown = contentType.includes('application/json')
    ? await response.json().catch(() => null)
    : await response.text();

  if (!response.ok) {
    const record = (payload ?? {}) as Record<string, unknown>;
    const error = (record.error ?? {}) as Record<string, unknown>;
    throw new ApiError(
      response.status,
      typeof error.code === 'string' ? error.code : `HTTP_${response.status}`,
      errorMessage(payload, `Request failed with status ${response.status}`),
      typeof record.requestId === 'string' ? record.requestId : undefined,
    );
  }

  return payload as T;
}

export function describeError(error: unknown, fallback: string): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}

/** Download a binary endpoint (used by the source export) with auth headers. */
export async function apiDownload(path: string, filename: string): Promise<void> {
  const response = await fetch(`${API_BASE}${path}`, { headers: authHeaders() });
  if (!response.ok) {
    throw new ApiError(
      response.status,
      `HTTP_${response.status}`,
      `Download failed with status ${response.status}`,
    );
  }

  const blob = await response.blob();
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}
