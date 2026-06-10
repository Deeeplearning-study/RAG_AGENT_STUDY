import { FormEvent, KeyboardEvent, useRef, useState } from 'react';
import { SendHorizontal } from 'lucide-react';

type ChatInputProps = {
  disabled?: boolean;
  onSubmit: (message: string) => void;
};

export function ChatInput({ disabled = false, onSubmit }: ChatInputProps) {
  const [message, setMessage] = useState('');
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const trimmedMessage = message.trim();
  const canSubmit = trimmedMessage.length > 0 && !disabled;

  const submit = (event?: FormEvent<HTMLFormElement>) => {
    event?.preventDefault();

    if (!canSubmit) {
      return;
    }

    onSubmit(trimmedMessage);
    setMessage('');
    textareaRef.current?.focus();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form
      onSubmit={submit}
      className="border-t border-slate-200 dark:border-zinc-700 bg-white/95 dark:bg-zinc-900/95 px-4 py-3 backdrop-blur md:px-6"
    >
      <div className="mx-auto flex max-w-5xl items-end gap-2">
        <textarea
          ref={textareaRef}
          value={message}
          onChange={(event) => setMessage(event.target.value)}
          onKeyDown={handleKeyDown}
          disabled={disabled}
          rows={1}
          placeholder="문서에 대해 질문하세요"
          className="max-h-36 min-h-12 flex-1 resize-none rounded-md border border-slate-300 dark:border-zinc-600 bg-white dark:bg-zinc-800 px-4 py-3 text-sm leading-6 text-slate-900 dark:text-zinc-100 shadow-sm outline-none transition placeholder:text-slate-400 dark:placeholder:text-zinc-500 focus:border-teal-600 dark:focus:border-teal-500 focus:ring-2 focus:ring-teal-100 dark:focus:ring-teal-900/30 disabled:cursor-not-allowed disabled:bg-slate-100 dark:disabled:bg-zinc-700"
        />
        <button
          type="submit"
          disabled={!canSubmit}
          className="inline-flex h-12 w-12 shrink-0 items-center justify-center rounded-md bg-teal-700 dark:bg-teal-600 text-white shadow-sm transition hover:bg-teal-800 dark:hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-teal-300 disabled:cursor-not-allowed disabled:bg-slate-300 dark:disabled:bg-zinc-600"
          aria-label="질문 전송"
          title="질문 전송"
        >
          <SendHorizontal className="h-5 w-5" aria-hidden="true" />
        </button>
      </div>
    </form>
  );
}
