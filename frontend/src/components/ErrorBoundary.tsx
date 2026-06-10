import { Component, type ErrorInfo, type ReactNode } from 'react';
import { AlertTriangle } from 'lucide-react';

type ErrorBoundaryProps = {
  children: ReactNode;
};

type ErrorBoundaryState = {
  error: Error | null;
};

export class ErrorBoundary extends Component<
  ErrorBoundaryProps,
  ErrorBoundaryState
> {
  state: ErrorBoundaryState = {
    error: null,
  };

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Frontend render error', error, errorInfo);
  }

  render() {
    if (this.state.error) {
      return (
        <main className="flex min-h-screen items-center justify-center bg-slate-100 px-4 text-slate-950">
          <section className="w-full max-w-xl rounded-md border border-red-200 bg-white p-5 shadow-sm">
            <div className="flex items-start gap-3">
              <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-md bg-red-50 text-red-700">
                <AlertTriangle className="h-5 w-5" aria-hidden="true" />
              </span>
              <div>
                <h1 className="text-base font-semibold">화면 렌더링 오류</h1>
                <p className="mt-2 text-sm leading-6 text-slate-600">
                  브라우저에서 UI를 표시하는 중 오류가 발생했습니다. 새로고침 후에도
                  반복되면 아래 메시지를 확인해 주세요.
                </p>
                <pre className="mt-3 overflow-auto rounded-md bg-slate-950 p-3 text-xs leading-5 text-white">
                  {this.state.error.message}
                </pre>
              </div>
            </div>
          </section>
        </main>
      );
    }

    return this.props.children;
  }
}
