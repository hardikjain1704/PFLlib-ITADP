const NAV_ITEMS = [
  { key: 'consent', label: 'Consent Manager', icon: '🛡️' },
  { key: 'purpose', label: 'Purpose Validator', icon: '⚙️' },
  { key: 'training', label: 'Training Control', icon: '🚀' },
  { key: 'transparency', label: 'Data Transparency', icon: '📊' },
];

export default function Sidebar({ active, onNavigate }) {
  return (
    <aside className="hidden lg:flex flex-col w-72 bg-gradient-to-b from-brand-800 to-brand-900 text-white shadow-xl">
      {/* Logo area */}
      <div className="flex items-center gap-3 px-6 py-6 border-b border-white/10">
        <span className="text-3xl">🔒</span>
        <div>
          <h1 className="text-lg font-bold leading-tight">PFLlib</h1>
          <p className="text-xs text-brand-300">Privacy Compliance</p>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 mt-4 px-3 space-y-1">
        {NAV_ITEMS.map((item) => {
          const isActive = active === item.key;
          return (
            <button
              key={item.key}
              onClick={() => onNavigate(item.key)}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-150
                ${isActive
                  ? 'bg-white/15 text-white shadow-inner'
                  : 'text-brand-200 hover:bg-white/10 hover:text-white'}`}
            >
              <span className="text-xl">{item.icon}</span>
              {item.label}
            </button>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="px-6 py-4 text-xs text-brand-400 border-t border-white/10">
        Federated Learning Compliance Dashboard v1.0
      </div>
    </aside>
  );
}
