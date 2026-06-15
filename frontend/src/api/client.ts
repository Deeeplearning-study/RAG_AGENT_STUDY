const rawApiBaseUrl = import.meta.env.VITE_API_BASE_URL?.trim() ?? '';

export const apiBaseUrl = rawApiBaseUrl.replace(/\/+$/, '');

type ApiOptions = Omit<RequestInit, 'body'> & {
  body?: unknown;
};

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

async function parseResponse(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') ?? '';

  if (!contentType.includes('application/json')) {
    return response.text();
  }

  return response.json();
}

function resolveErrorMessage(payload: unknown, fallback: string) {
  if (typeof payload === 'string' && payload.trim().length > 0) {
    return payload;
  }

  if (
    payload &&
    typeof payload === 'object' &&
    'error' in payload &&
    payload.error &&
    typeof payload.error === 'object' &&
    'message' in payload.error &&
    typeof payload.error.message === 'string'
  ) {
    return payload.error.message;
  }

  if (
    payload &&
    typeof payload === 'object' &&
    'detail' in payload &&
    typeof payload.detail === 'string'
  ) {
    return payload.detail;
  }

  if (
    payload &&
    typeof payload === 'object' &&
    'message' in payload &&
    typeof payload.message === 'string'
  ) {
    return payload.message;
  }

  return fallback;
}

export async function apiFetch<T>(
  path: string,
  options: ApiOptions = {},
): Promise<T> {
  const headers = new Headers(options.headers);
  const isFormData =
    typeof FormData !== 'undefined' && options.body instanceof FormData;

  // FormData lets the browser set the multipart content-type (with boundary).
  if (options.body !== undefined && !isFormData) {
    headers.set('content-type', 'application/json');
  }

  let body: BodyInit | undefined;
  if (options.body === undefined) {
    body = undefined;
  } else if (isFormData) {
    body = options.body as FormData;
  } else {
    body = JSON.stringify(options.body);
  }

  const response = await fetch(`${apiBaseUrl}${path}`, {
    ...options,
    headers,
    body,
  });

  const payload = await parseResponse(response);

  if (!response.ok) {
    throw new ApiError(
      resolveErrorMessage(payload, 'API 요청을 처리하지 못했습니다.'),
      response.status,
    );
  }

  return payload as T;
}
