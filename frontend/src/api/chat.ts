import { apiFetch } from './client';
import type { ChatRequest, ChatResponse } from '../types/api';

export function sendChatMessage(request: ChatRequest) {
  return apiFetch<ChatResponse>('/api/chat', {
    method: 'POST',
    body: request,
  });
}
