import { useState } from 'react';
import Sidebar from './components/Sidebar';
import ConsentManager from './components/ConsentManager';
import TrainingPurposeValidator from './components/TrainingPurposeValidator';
import TrainingControl from './components/TrainingControl';
import DataTransparencyDashboard from './components/DataTransparencyDashboard';

const PAGES = {
  consent: ConsentManager,
  purpose: TrainingPurposeValidator,
  training: TrainingControl,
  transparency: DataTransparencyDashboard,
};

export default function App() {
  const [page, setPage] = useState('consent');
  const Page = PAGES[page];

  return (
    <div className="flex h-screen overflow-hidden">
      <Sidebar active={page} onNavigate={setPage} />
      <main className="flex-1 overflow-y-auto p-6 lg:p-10">
        <Page />
      </main>
    </div>
  );
}
