import { useState } from 'react';
import { Layout } from './components/Layout';
import { Dashboard } from './pages/Dashboard';
import { Chat } from './pages/Chat';
import { Editor } from './pages/Editor';
import { Generator } from './pages/Generator';
import { TreeView } from './pages/TreeView';
import { Settings } from './pages/Settings';
import { GeneratorProvider } from './context/GeneratorContext';
import { ChatProvider } from './context/ChatContext';

function App() {
  const [currentPage, setCurrentPage] = useState('dashboard');

  const renderPage = () => {
    switch (currentPage) {
      case 'dashboard':
        return <Dashboard />;
      case 'chat':
        return <Chat />;
      case 'editor':
        return <Editor />;
      case 'generator':
        return <Generator />;
      case 'tree':
        return <TreeView />;
      case 'settings':
        return <Settings />;
      default:
        return <Dashboard />;
    }
  };

  return (
    <GeneratorProvider>
      <ChatProvider>
        <Layout currentPage={currentPage} onNavigate={setCurrentPage}>
          {renderPage()}
        </Layout>
      </ChatProvider>
    </GeneratorProvider>
  );
}

export default App;
