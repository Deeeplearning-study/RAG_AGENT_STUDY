import { useState } from 'react';
import { ChevronDown, ChevronUp, FileText } from 'lucide-react';
import type { Source } from '../types/api';

type SourceCardProps = {
  source: Source;
  index: number;
};

function formatPages(source: Source) {
  if (source.page_start === source.page_end) {
    return `${source.page_start}쪽`;
  }

  return `${source.page_start}-${source.page_end}쪽`;
}

export function SourceCard({ source, index }: SourceCardProps) {
  const [expanded, setExpanded] = useState(false);

  return (
    <article className="rounded-md border border-slate-200 bg-slate-50">
      <button
        type="button"
        onClick={() => setExpanded((current) => !current)}
        className="flex w-full items-start gap-3 px-3 py-3 text-left"
        aria-expanded={expanded}
      >
        <span className="mt-0.5 inline-flex h-8 w-8 shrink-0 items-center justify-center rounded-md bg-white text-teal-700 ring-1 ring-slate-200">
          <FileText className="h-4 w-4" aria-hidden="true" />
        </span>
        <span className="min-w-0 flex-1">
          <span className="block text-xs font-semibold text-slate-500">
            출처 {index + 1} · {formatPages(source)}
          </span>
          <span className="mt-0.5 block truncate text-sm font-medium text-slate-900">
            {source.file_name}
          </span>
        </span>
        <span className="mt-1 shrink-0 text-slate-500">
          {expanded ? (
            <ChevronUp className="h-4 w-4" aria-hidden="true" />
          ) : (
            <ChevronDown className="h-4 w-4" aria-hidden="true" />
          )}
        </span>
      </button>
      {expanded ? (
        <p className="border-t border-slate-200 px-3 pb-3 pt-2 text-sm leading-6 text-slate-700">
          {source.snippet}
        </p>
      ) : null}
    </article>
  );
}
