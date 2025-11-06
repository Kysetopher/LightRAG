import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import AppRouter from './AppRouter'
import { AppDataProvider } from '@/contexts/AppDataProvider'
import './i18n.ts';
import 'katex/dist/katex.min.css';



createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <AppDataProvider>
      <AppRouter />
    </AppDataProvider>
  </StrictMode>
)
