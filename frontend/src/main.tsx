import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import { ErrorBoundary } from './components/ErrorBoundary';
import './styles/index.css';
import User from './User';

function RootRouter() {
  if (window.location.pathname === '/test') {
    return <App />;
  }

  return <User />;
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <RootRouter />
    </ErrorBoundary>
  </StrictMode>,
);
