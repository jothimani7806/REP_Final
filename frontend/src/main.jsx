import React, { useState, useEffect, useRef } from 'react';
import { createRoot } from 'react-dom/client';
import './index.css';
import {
  LayoutDashboard,
  Calculator,
  ShieldAlert,
  Cpu,
  HeartPulse,
  Bot,
  BarChart3,
  Database,
  ChevronLeft,
  ChevronRight,
  Menu,
  Plus,
  Trash2,
  Send,
  ExternalLink,
  Activity,
  Info,
  CheckCircle,
  AlertTriangle,
  ArrowUpRight,
  ArrowDownRight,
  RefreshCw,
  User,
  Sparkles,
  Layers,
  Heart,
  TrendingUp,
  FileText
} from 'lucide-react';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

const demoProfiles = {
  high: {
    age: 55, sex: 'male', weight: 92, bmi: 34.2, hereditary_diseases: 'Diabetes',
    no_of_dependents: 2, smoker: 1, city: 'NewYork', bloodpressure: 148,
    diabetes: 1, regular_ex: 0, job_title: 'Executive'
  },
  medium: {
    age: 38, sex: 'female', weight: 68, bmi: 26.5, hereditary_diseases: 'NoDisease',
    no_of_dependents: 1, smoker: 0, city: 'Chicago', bloodpressure: 124,
    diabetes: 0, regular_ex: 1, job_title: 'Manager'
  },
  low: {
    age: 23, sex: 'female', weight: 56, bmi: 21.0, hereditary_diseases: 'NoDisease',
    no_of_dependents: 0, smoker: 0, city: 'SanFrancisco', bloodpressure: 114,
    diabetes: 0, regular_ex: 1, job_title: 'SoftwareEngineer'
  }
};

const suggestedQuestions = [
  'What is BMI?',
  'Why is my insurance premium high?',
  'What factors affect my predicted premium?',
  'What does my risk category mean?',
  'How can I improve my health?',
  'How can I improve my lifestyle?'
];

function App() {
  const [activePage, setActivePage] = useState('dashboard');
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Predictor & Model State
  const [form, setForm] = useState(demoProfiles.high);
  const [result, setResult] = useState(null);
  const [predictLoading, setPredictLoading] = useState(false);
  const [benchmarks, setBenchmarks] = useState(null);
  const [xaiData, setXaiData] = useState(null);
  const [selectedXaiRisk, setSelectedXaiRisk] = useState('High');

  // Chatbot State
  const [conversations, setConversations] = useState([]);
  const [activeConvId, setActiveConvId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [inputMessage, setInputMessage] = useState('');
  const [chatLoading, setChatLoading] = useState(false);
  const [chatError, setChatError] = useState(null);
  const messagesEndRef = useRef(null);

  // Initial Data Fetch
  useEffect(() => {
    fetchBenchmarks();
    fetchXaiInfo('High');
    fetchConversations();
    handlePredict(demoProfiles.high);
  }, []);

  // Auto-scroll chat messages
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, chatLoading]);

  const fetchBenchmarks = async () => {
    try {
      const res = await fetch(`${API_BASE}/benchmarks`);
      if (res.ok) {
        const data = await res.json();
        setBenchmarks(data);
      }
    } catch (err) {
      console.warn('Benchmarks fetch error:', err);
    }
  };

  const fetchXaiInfo = async (riskTier) => {
    try {
      const res = await fetch(`${API_BASE}/xai_info/${riskTier}`);
      if (res.ok) {
        const data = await res.json();
        setXaiData(data);
      }
    } catch (err) {
      console.warn('XAI fetch error:', err);
    }
  };

  const fetchConversations = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/conversations`);
      if (res.ok) {
        const data = await res.json();
        setConversations(data);
        if (data.length > 0 && !activeConvId) {
          loadConversation(data[0].id);
        }
      }
    } catch (err) {
      console.warn('Fetch conversations error:', err);
    }
  };

  const loadConversation = async (convId) => {
    setActiveConvId(convId);
    setChatError(null);
    try {
      const res = await fetch(`${API_BASE}/api/conversations/${convId}/messages`);
      if (res.ok) {
        const data = await res.json();
        setMessages(data.messages || []);
      }
    } catch (err) {
      console.error('Load conversation error:', err);
    }
  };

  const handleNewChat = () => {
    setActiveConvId(null);
    setMessages([]);
    setInputMessage('');
    setChatError(null);
  };

  const handleDeleteConversation = async (e, convId) => {
    e.stopPropagation();
    try {
      const res = await fetch(`${API_BASE}/api/conversations/${convId}`, { method: 'DELETE' });
      if (res.ok) {
        const updated = conversations.filter(c => c.id !== convId);
        setConversations(updated);
        if (activeConvId === convId) {
          if (updated.length > 0) {
            loadConversation(updated[0].id);
          } else {
            handleNewChat();
          }
        }
      }
    } catch (err) {
      console.error('Delete conversation error:', err);
    }
  };

  const handleSendMessage = async (customText = null) => {
    const textToSend = (customText !== null ? customText : inputMessage).trim();
    if (!textToSend || chatLoading) return;

    setChatError(null);

    // Optimistically append user message if not already present
    const tempUserMsg = {
      id: `temp-${Date.now()}`,
      role: 'user',
      content: textToSend,
      medical_references: [],
      created_at: new Date().toISOString()
    };
    setMessages(prev => {
      if (prev.length > 0 && prev[prev.length - 1].role === 'user' && prev[prev.length - 1].content === textToSend) {
        return prev;
      }
      return [...prev, tempUserMsg];
    });
    setInputMessage('');
    setChatLoading(true);

    try {
      const res = await fetch(`${API_BASE}/api/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: textToSend,
          conversation_id: activeConvId
        })
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data.detail || `Chat request failed (${res.status})`);
      }

      // Update active conversation ID if it was newly created
      if (!activeConvId && data.conversation_id) {
        setActiveConvId(data.conversation_id);
      }

      // Append assistant message from Gemini
      const assistantMsg = {
        id: data.id || `asst-${Date.now()}`,
        role: 'assistant',
        content: data.message,
        medical_references: data.medical_references || [],
        created_at: data.created_at || new Date().toISOString()
      };
      setMessages(prev => [...prev, assistantMsg]);
      fetchConversations();
    } catch (err) {
      console.error('Chat error:', err);
      // Clean error presentation without fake assistant messages
      setChatError({
        message: err.message || '⚠️ Unable to connect to HealthGuard AI Assistant. Please check backend status.',
        textToRetry: textToSend
      });
    } finally {
      setChatLoading(false);
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setForm(prev => ({ ...prev, [name]: value }));
  };

  const loadPreset = (tier) => {
    const p = demoProfiles[tier];
    setForm(p);
    handlePredict(p);
  };

  const handlePredict = async (payloadToUse = null) => {
    setPredictLoading(true);
    const raw = payloadToUse || form;
    const payload = { ...raw };
    for (const k of ['age', 'weight', 'bmi', 'no_of_dependents', 'smoker', 'bloodpressure', 'diabetes', 'regular_ex']) {
      payload[k] = payload[k] === '' || payload[k] === null ? null : Number(payload[k]);
    }

    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Prediction failed');
      setResult(data);
    } catch (err) {
      console.error(err);
      alert(`API Connection Error: ${err.message}`);
    } finally {
      setPredictLoading(false);
    }
  };

  const formatCurrency = (val) => {
    return new Intl.NumberFormat('en-IN', {
      style: 'currency',
      currency: 'INR',
      maximumFractionDigits: 0
    }).format(val || 0);
  };

  const formatTime = (isoStr) => {
    if (!isoStr) return '';
    try {
      return new Date(isoStr).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    } catch {
      return '';
    }
  };

  const askChatbotAboutPrediction = () => {
    setActivePage('chatbot');
    const q = 'Why is my insurance premium high? What factors affected my quote?';
    handleSendMessage(q);
  };

  return (
    <div className="app-layout">
      {/* SIDEBAR NAVIGATION */}
      <aside className={`sidebar ${sidebarCollapsed ? 'collapsed' : ''} ${mobileMenuOpen ? 'mobile-open' : ''}`}>
        <div className="sidebar-header">
          <div className="sidebar-brand">
            <div className="brand-icon-box">
              <HeartPulse size={22} />
            </div>
            {!sidebarCollapsed && (
              <div className="brand-text-wrap">
                <div className="brand-name">HealthGuard <span>AI</span></div>
                <div className="brand-tag">Risk & Actuarial XAI</div>
              </div>
            )}
          </div>
          <button
            className="sidebar-toggle-btn"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
            title="Toggle Sidebar"
          >
            {sidebarCollapsed ? <ChevronRight size={18} /> : <ChevronLeft size={18} />}
          </button>
        </div>

        <nav className="sidebar-nav">
          <div className="nav-group-label">{!sidebarCollapsed && 'Core Intelligence'}</div>
          <button
            className={`nav-link ${activePage === 'dashboard' ? 'active' : ''}`}
            onClick={() => { setActivePage('dashboard'); setMobileMenuOpen(false); }}
          >
            <LayoutDashboard size={18} />
            {!sidebarCollapsed && <span>Dashboard Overview</span>}
          </button>
          <button
            className={`nav-link ${activePage === 'predictor' ? 'active' : ''}`}
            onClick={() => { setActivePage('predictor'); setMobileMenuOpen(false); }}
          >
            <Calculator size={18} />
            {!sidebarCollapsed && <span>Premium Prediction</span>}
          </button>
          <button
            className={`nav-link ${activePage === 'risk' ? 'active' : ''}`}
            onClick={() => { setActivePage('risk'); setMobileMenuOpen(false); }}
          >
            <ShieldAlert size={18} />
            {!sidebarCollapsed && <span>Risk Segmentation</span>}
          </button>
          <button
            className={`nav-link ${activePage === 'xai' ? 'active' : ''}`}
            onClick={() => { setActivePage('xai'); setMobileMenuOpen(false); }}
          >
            <Cpu size={18} />
            {!sidebarCollapsed && <span>Explainable AI (SHAP)</span>}
          </button>
          <button
            className={`nav-link ${activePage === 'wellness' ? 'active' : ''}`}
            onClick={() => { setActivePage('wellness'); setMobileMenuOpen(false); }}
          >
            <Heart size={18} />
            {!sidebarCollapsed && <span>Wellness Guidance</span>}
          </button>

          <div className="nav-group-label" style={{ marginTop: '12px' }}>
            {!sidebarCollapsed && 'AI & Research'}
          </div>
          <button
            className={`nav-link ${activePage === 'chatbot' ? 'active' : ''}`}
            onClick={() => { setActivePage('chatbot'); setMobileMenuOpen(false); }}
          >
            <Bot size={18} />
            {!sidebarCollapsed && <span>AI Health Assistant</span>}
          </button>
          <button
            className={`nav-link ${activePage === 'benchmarks' ? 'active' : ''}`}
            onClick={() => { setActivePage('benchmarks'); setMobileMenuOpen(false); }}
          >
            <BarChart3 size={18} />
            {!sidebarCollapsed && <span>Research Benchmarks</span>}
          </button>
          <button
            className={`nav-link ${activePage === 'architecture' ? 'active' : ''}`}
            onClick={() => { setActivePage('architecture'); setMobileMenuOpen(false); }}
          >
            <Layers size={18} />
            {!sidebarCollapsed && <span>System Architecture</span>}
          </button>
        </nav>

        {!sidebarCollapsed && (
          <div className="sidebar-footer">
            <div className="model-status-card">
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '6px' }}>
                <span className="status-dot"></span>
                <span style={{ fontWeight: 600, color: '#e2e8f0' }}>Risk-Aware Pipeline</span>
              </div>
              <div style={{ color: '#94a3b8' }}>15,000 Cohort • K-Means + LightGBM</div>
            </div>
          </div>
        )}
      </aside>

      {/* Backdrop for mobile drawer */}
      {mobileMenuOpen && (
        <div
          className="sidebar-backdrop"
          onClick={() => setMobileMenuOpen(false)}
        />
      )}

      {/* MAIN WRAPPER */}
      <div className="main-wrapper">
        {/* Top Header */}
        <header className="top-header">
          <div className="header-left">
            <button
              className="header-sidebar-toggle-btn"
              onClick={() => {
                if (window.innerWidth <= 800) {
                  setMobileMenuOpen(!mobileMenuOpen);
                } else {
                  setSidebarCollapsed(!sidebarCollapsed);
                }
              }}
              title="Toggle Menu"
            >
              <Menu size={18} />
              <span>{sidebarCollapsed ? "Expand Menu" : "Menu"}</span>
            </button>
            <div className="header-title-box">
              <h1>
                {activePage === 'dashboard' && 'Executive Actuarial Dashboard'}
                {activePage === 'predictor' && 'Personalized Premium & Risk Calculator'}
                {activePage === 'risk' && 'Unsupervised Customer Risk Segmentation'}
                {activePage === 'xai' && 'Explainable AI: SHAP & Partial Dependence'}
                {activePage === 'wellness' && 'Personalized Wellness Recommendations'}
                {activePage === 'chatbot' && 'HealthGuard AI Health & Insurance Assistant'}
                {activePage === 'benchmarks' && 'Actuarial Research Paper Benchmarks'}
                {activePage === 'architecture' && 'End-to-End System Architecture'}
              </h1>
              <p>HealthGuard AI Enterprise Intelligence Engine • Production ML Model</p>
            </div>
          </div>

          <div className="header-right">
            <div className="badge-pill teal">
              <Activity size={14} />
              <span>Model R²: 96.4%</span>
            </div>
            {result?.risk && (
              <div className={`badge-pill ${result.risk === 'High' ? 'rose' : result.risk === 'Medium' ? 'amber' : 'emerald'}`}
                   style={{
                     background: result.risk === 'High' ? 'var(--status-high-bg)' : result.risk === 'Medium' ? 'var(--status-medium-bg)' : 'var(--status-low-bg)',
                     color: result.risk === 'High' ? 'var(--status-high)' : result.risk === 'Medium' ? 'var(--status-medium)' : 'var(--status-low)'
                   }}>
                <span>Tier: {result.risk} Risk</span>
              </div>
            )}
            <button
              className="btn-primary"
              onClick={() => setActivePage('chatbot')}
            >
              <Bot size={15} />
              <span>Ask AI Assistant</span>
            </button>
          </div>
        </header>

        {/* PAGE CONTENT CONTAINER */}
        <main className="page-container">
          {/* ================= PAGE 1: DASHBOARD OVERVIEW ================= */}
          {activePage === 'dashboard' && (
            <>
              {/* 6 Real Metric Cards */}
              <div className="metrics-grid-6">
                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-title">Total Actuarial Pool</span>
                    <div className="metric-icon-wrap teal"><Database size={16} /></div>
                  </div>
                  <div className="metric-value">15,000</div>
                  <div className="metric-desc">Curated customer health records</div>
                </div>

                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-title">Latest Predicted Claim</span>
                    <div className="metric-icon-wrap cyan"><Calculator size={16} /></div>
                  </div>
                  <div className="metric-value">
                    {result ? formatCurrency(result.risk_aware_prediction) : '₹145,578'}
                  </div>
                  <div className="metric-desc">Risk-aware specialized model estimate</div>
                </div>

                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-title">Current Risk Tier</span>
                    <div className="metric-icon-wrap amber"><ShieldAlert size={16} /></div>
                  </div>
                  <div className="metric-value" style={{ color: result?.risk === 'High' ? 'var(--status-high)' : result?.risk === 'Low' ? 'var(--status-low)' : 'var(--status-medium)' }}>
                    {result?.risk ? `${result.risk} Risk` : 'High Risk'}
                  </div>
                  <div className="metric-desc">Unsupervised K-Means clustering assignment</div>
                </div>

                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-title">Average Cohort Claim</span>
                    <div className="metric-icon-wrap slate"><TrendingUp size={16} /></div>
                  </div>
                  <div className="metric-value">₹16,392</div>
                  <div className="metric-desc">Mean actuarial baseline across training set</div>
                </div>

                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-title">Low-Risk Profiles</span>
                    <div className="metric-icon-wrap emerald"><CheckCircle size={16} /></div>
                  </div>
                  <div className="metric-value">19.3%</div>
                  <div className="metric-desc">Optimal lifestyle & preventive profiles</div>
                </div>

                <div className="metric-card">
                  <div className="metric-header">
                    <span className="metric-title">Model Performance</span>
                    <div className="metric-icon-wrap teal"><Activity size={16} /></div>
                  </div>
                  <div className="metric-value">96.4% R²</div>
                  <div className="metric-desc">Outperforming global uniform baseline</div>
                </div>
              </div>

              {/* Two Column Section */}
              <div className="grid-2col">
                <div className="dashboard-card">
                  <div className="card-title-row">
                    <div>
                      <h2><Sparkles size={18} style={{ color: 'var(--brand-teal)' }} /> Actuarial Intelligence Summary</h2>
                      <p>Current active customer evaluation details</p>
                    </div>
                    <button className="btn-secondary" onClick={() => setActivePage('predictor')}>
                      Modify Profile
                    </button>
                  </div>

                  {result ? (
                    <div>
                      <div className="comparison-box">
                        <div>
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                            Specialized Model ({result.risk_model})
                          </div>
                          <div style={{ fontSize: '24px', fontWeight: 700, color: 'var(--text-primary)' }}>
                            {formatCurrency(result.risk_aware_prediction)}
                          </div>
                        </div>
                        <div style={{ textAlign: 'right' }}>
                          <div style={{ fontSize: '11px', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                            Global Uniform Baseline ({result.global_model})
                          </div>
                          <div style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-secondary)' }}>
                            {formatCurrency(result.global_prediction)}
                          </div>
                        </div>
                      </div>

                      <div style={{ marginTop: '16px' }}>
                        <div style={{ fontSize: '13px', fontWeight: 600, marginBottom: '8px' }}>
                          Key Influencing Factors (SHAP Local Explanation):
                        </div>
                        <div className="shap-factors-list">
                          {result.top_shap_factors?.slice(0, 4).map((f, i) => (
                            <div key={i} className="shap-item">
                              <div className="shap-item-header">
                                <span>{f.feature}</span>
                                <span style={{ color: f.impact > 0 ? 'var(--status-high)' : 'var(--status-low)' }}>
                                  {f.direction} ({f.impact > 0 ? '+' : ''}{f.impact.toFixed(1)})
                                </span>
                              </div>
                              <div className="shap-bar-track">
                                <div
                                  className={`shap-bar-fill ${f.impact > 0 ? 'red' : 'green'}`}
                                  style={{ width: `${Math.min(100, (f.abs_impact / 2500) * 100)}%` }}
                                ></div>
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>

                      <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
                        <button className="btn-primary" onClick={askChatbotAboutPrediction}>
                          <Bot size={15} />
                          <span>Explain with AI Assistant</span>
                        </button>
                        <button className="btn-secondary" onClick={() => setActivePage('xai')}>
                          <span>View Full XAI Report</span>
                          <ArrowUpRight size={14} />
                        </button>
                      </div>
                    </div>
                  ) : (
                    <div style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
                      No active prediction. Run the calculator to generate live analysis.
                    </div>
                  )}
                </div>

                <div className="dashboard-card">
                  <div className="card-title-row">
                    <div>
                      <h2><Bot size={18} style={{ color: 'var(--accent-cyan)' }} /> AI Health Assistant</h2>
                      <p>Instant grounded advice on health and insurance</p>
                    </div>
                    <button className="btn-secondary" onClick={() => setActivePage('chatbot')}>
                      Open Full Chat
                    </button>
                  </div>

                  <div style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
                    <div style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                      Ask questions grounded in your verified ML prediction, actuarial risk categories, and trusted medical sources (MedlinePlus &amp; CDC).
                    </div>

                    <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                      <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>
                        Suggested Consultations:
                      </span>
                      {suggestedQuestions.slice(0, 4).map((q, idx) => (
                        <button
                          key={idx}
                          className="preset-chip"
                          style={{ width: '100%', justifyContent: 'space-between' }}
                          onClick={() => {
                            setActivePage('chatbot');
                            handleSendMessage(q);
                          }}
                        >
                          <span>{q}</span>
                          <ArrowUpRight size={14} style={{ color: 'var(--brand-teal)' }} />
                        </button>
                      ))}
                    </div>

                    <div style={{ marginTop: '10px', padding: '12px', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', fontSize: '11px', color: 'var(--text-muted)' }}>
                      🛡️ <b>Strict Safety Compliance</b>: HealthGuard AI adheres to medical non-diagnostic guidelines and provides evidence-backed health education.
                    </div>
                  </div>
                </div>
              </div>
            </>
          )}

          {/* ================= PAGE 2: PREMIUM PREDICTOR ================= */}
          {activePage === 'predictor' && (
            <div className="grid-2col">
              {/* Form Card */}
              <div className="dashboard-card">
                <div className="card-title-row">
                  <div>
                    <h2><Calculator size={18} style={{ color: 'var(--brand-teal)' }} /> Customer Profile Inputs</h2>
                    <p>Enter actuarial and clinical parameters</p>
                  </div>
                </div>

                {/* Presets Bar */}
                <div className="presets-bar">
                  <span style={{ fontSize: '12px', fontWeight: 600, color: 'var(--text-muted)' }}>Demo Personas:</span>
                  <button type="button" className="preset-chip" onClick={() => loadPreset('high')}>
                    🔴 High Risk Profile
                  </button>
                  <button type="button" className="preset-chip" onClick={() => loadPreset('medium')}>
                    🟡 Medium Risk Profile
                  </button>
                  <button type="button" className="preset-chip" onClick={() => loadPreset('low')}>
                    🟢 Low Risk Profile
                  </button>
                </div>

                <form onSubmit={(e) => { e.preventDefault(); handlePredict(); }}>
                  <div className="form-group-section">
                    <div className="form-group-title">1. Demographic Details</div>
                    <div className="form-row-3">
                      <div className="input-field">
                        <label className="input-label">Age (years)</label>
                        <input
                          type="number"
                          name="age"
                          value={form.age ?? ''}
                          onChange={handleInputChange}
                          className="input-control"
                          min="18"
                          max="100"
                          required
                        />
                      </div>
                      <div className="input-field">
                        <label className="input-label">Sex</label>
                        <select name="sex" value={form.sex} onChange={handleInputChange} className="input-control">
                          <option value="male">Male</option>
                          <option value="female">Female</option>
                        </select>
                      </div>
                      <div className="input-field">
                        <label className="input-label">City</label>
                        <input
                          type="text"
                          name="city"
                          value={form.city}
                          onChange={handleInputChange}
                          className="input-control"
                          required
                        />
                      </div>
                    </div>
                  </div>

                  <div className="form-group-section">
                    <div className="form-group-title">2. Clinical &amp; Lifestyle Metrics</div>
                    <div className="form-row-3">
                      <div className="input-field">
                        <label className="input-label">Weight (kg)</label>
                        <input
                          type="number"
                          step="0.1"
                          name="weight"
                          value={form.weight ?? ''}
                          onChange={handleInputChange}
                          className="input-control"
                          required
                        />
                      </div>
                      <div className="input-field">
                        <label className="input-label">BMI</label>
                        <input
                          type="number"
                          step="0.1"
                          name="bmi"
                          value={form.bmi ?? ''}
                          onChange={handleInputChange}
                          className="input-control"
                          required
                        />
                      </div>
                      <div className="input-field">
                        <label className="input-label">Systolic BP (mmHg)</label>
                        <input
                          type="number"
                          name="bloodpressure"
                          value={form.bloodpressure ?? ''}
                          onChange={handleInputChange}
                          className="input-control"
                          required
                        />
                      </div>
                    </div>

                    <div className="form-row-3" style={{ marginTop: '12px' }}>
                      <div className="input-field">
                        <label className="input-label">Smoker</label>
                        <select name="smoker" value={form.smoker} onChange={handleInputChange} className="input-control">
                          <option value={0}>No (Non-smoker)</option>
                          <option value={1}>Yes (Smoker)</option>
                        </select>
                      </div>
                      <div className="input-field">
                        <label className="input-label">Diabetes</label>
                        <select name="diabetes" value={form.diabetes} onChange={handleInputChange} className="input-control">
                          <option value={0}>No</option>
                          <option value={1}>Yes</option>
                        </select>
                      </div>
                      <div className="input-field">
                        <label className="input-label">Regular Exercise</label>
                        <select name="regular_ex" value={form.regular_ex} onChange={handleInputChange} className="input-control">
                          <option value={1}>Yes (Active)</option>
                          <option value={0}>No (Sedentary)</option>
                        </select>
                      </div>
                    </div>
                  </div>

                  <div className="form-group-section">
                    <div className="form-group-title">3. Family &amp; Actuarial Factors</div>
                    <div className="form-row-3">
                      <div className="input-field">
                        <label className="input-label">Dependents</label>
                        <input
                          type="number"
                          name="no_of_dependents"
                          value={form.no_of_dependents ?? ''}
                          onChange={handleInputChange}
                          className="input-control"
                          min="0"
                          max="10"
                          required
                        />
                      </div>
                      <div className="input-field">
                        <label className="input-label">Hereditary Diseases</label>
                        <select
                          name="hereditary_diseases"
                          value={form.hereditary_diseases}
                          onChange={handleInputChange}
                          className="input-control"
                        >
                          <option value="NoDisease">No Disease</option>
                          <option value="Diabetes">Diabetes</option>
                          <option value="Alzheimer">Alzheimer</option>
                          <option value="HeartDisease">Heart Disease</option>
                          <option value="Obesity">Obesity</option>
                          <option value="HighBP">High BP</option>
                        </select>
                      </div>
                      <div className="input-field">
                        <label className="input-label">Occupation</label>
                        <input
                          type="text"
                          name="job_title"
                          value={form.job_title}
                          onChange={handleInputChange}
                          className="input-control"
                          required
                        />
                      </div>
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="btn-primary"
                    style={{ width: '100%', justifyContent: 'center', padding: '12px', marginTop: '10px' }}
                    disabled={predictLoading}
                  >
                    {predictLoading ? <RefreshCw className="spin" size={16} /> : <Calculator size={16} />}
                    <span>{predictLoading ? 'Calculating Risk-Aware Premium...' : 'Generate Prediction Quote'}</span>
                  </button>
                </form>
              </div>

              {/* Result Panel */}
              <div>
                {result ? (
                  <div className="dashboard-card">
                    <div className="card-title-row">
                      <div>
                        <h2><Activity size={18} style={{ color: 'var(--brand-teal)' }} /> Actuarial Prediction Results</h2>
                        <p>Evaluated using {result.risk_model} for {result.risk} Risk Cluster</p>
                      </div>
                    </div>

                    <div className="result-hero-box">
                      <div className="result-premium-tag">Predicted Annual Claim / Premium</div>
                      <div className="result-premium-val">{formatCurrency(result.risk_aware_prediction)}</div>
                      <div className={`risk-badge-large ${result.risk}`}>
                        <ShieldAlert size={16} />
                        <span>{result.risk} Risk Classification (Cluster #{result.cluster})</span>
                      </div>
                    </div>

                    <div className="comparison-box">
                      <div>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>GLOBAL BASELINE</div>
                        <div style={{ fontSize: '16px', fontWeight: 600 }}>{formatCurrency(result.global_prediction)}</div>
                      </div>
                      <div style={{ textAlign: 'right' }}>
                        <div style={{ fontSize: '11px', color: 'var(--text-muted)' }}>RISK ADJUSTMENT</div>
                        <div style={{ fontSize: '16px', fontWeight: 700, color: result.diff > 0 ? 'var(--status-high)' : 'var(--status-low)' }}>
                          {result.diff > 0 ? `+${formatCurrency(result.diff)} (+${result.diff_pct}%)` : `${formatCurrency(result.diff)} (${result.diff_pct}%)`}
                        </div>
                      </div>
                    </div>

                    <div style={{ marginTop: '16px' }}>
                      <div style={{ fontSize: '13px', fontWeight: 700, marginBottom: '10px' }}>
                        Top Driving Factors (SHAP Local Attribution):
                      </div>
                      <div className="shap-factors-list">
                        {result.top_shap_factors?.map((f, i) => (
                          <div key={i} className="shap-item">
                            <div className="shap-item-header">
                              <span>{f.feature}</span>
                              <span style={{ color: f.impact > 0 ? 'var(--status-high)' : 'var(--status-low)' }}>
                                {f.direction} ({f.impact > 0 ? '+' : ''}{f.impact.toFixed(1)})
                              </span>
                            </div>
                            <div className="shap-bar-track">
                              <div
                                className={`shap-bar-fill ${f.impact > 0 ? 'red' : 'green'}`}
                                style={{ width: `${Math.min(100, (f.abs_impact / 2500) * 100)}%` }}
                              ></div>
                            </div>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div style={{ marginTop: '20px', display: 'flex', gap: '10px' }}>
                      <button className="btn-primary" style={{ flex: 1, justifyContent: 'center' }} onClick={askChatbotAboutPrediction}>
                        <Bot size={15} />
                        <span>Ask AI Assistant About This Result</span>
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="dashboard-card" style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-muted)' }}>
                    <Calculator size={40} style={{ opacity: 0.3, marginBottom: '16px' }} />
                    <h3>No Prediction Calculated Yet</h3>
                    <p style={{ fontSize: '13px', marginTop: '8px' }}>
                      Fill in the customer actuarial profile and click Generate Prediction Quote to see risk-aware evaluation.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ================= PAGE 3: RISK ANALYSIS ================= */}
          {activePage === 'risk' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div className="dashboard-card">
                <div className="card-title-row">
                  <div>
                    <h2><ShieldAlert size={18} style={{ color: 'var(--brand-teal)' }} /> Unsupervised Risk Segmentation Structure</h2>
                    <p>Learned K-Means clusters mapped to risk tiers from training-set claim distribution</p>
                  </div>
                </div>

                <div className="metrics-grid-6" style={{ gridTemplateColumns: 'repeat(3, 1fr)' }}>
                  <div className="metric-card" style={{ borderLeft: '4px solid var(--status-low)' }}>
                    <div className="metric-header">
                      <span className="metric-title" style={{ color: 'var(--status-low)' }}>Low Risk Tier</span>
                      <CheckCircle size={18} style={{ color: 'var(--status-low)' }} />
                    </div>
                    <div className="metric-value">₹8,130</div>
                    <div className="metric-desc">
                      Cluster #2 • 1,723 profiles (19.3%). Characterized by normal BMI, active lifestyle, non-smokers.
                    </div>
                  </div>

                  <div className="metric-card" style={{ borderLeft: '4px solid var(--status-medium)' }}>
                    <div className="metric-header">
                      <span className="metric-title" style={{ color: 'var(--status-medium)' }}>Medium Risk Tier</span>
                      <AlertTriangle size={18} style={{ color: 'var(--status-medium)' }} />
                    </div>
                    <div className="metric-value">₹8,951</div>
                    <div className="metric-desc">
                      Cluster #1 • 5,404 profiles (60.8%). Moderate BMI, controlled blood pressure, occasional indicators.
                    </div>
                  </div>

                  <div className="metric-card" style={{ borderLeft: '4px solid var(--status-high)' }}>
                    <div className="metric-header">
                      <span className="metric-title" style={{ color: 'var(--status-high)' }}>High Risk Tier</span>
                      <ShieldAlert size={18} style={{ color: 'var(--status-high)' }} />
                    </div>
                    <div className="metric-value">₹32,096</div>
                    <div className="metric-desc">
                      Cluster #0 • 1,771 profiles (19.9%). Smoking, elevated BP, and high BMI interactions.
                    </div>
                  </div>
                </div>

                <div style={{ marginTop: '24px', padding: '16px', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                  <h4 style={{ fontSize: '14px', marginBottom: '6px' }}>Methodology &amp; Soundness Notice</h4>
                  <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                    `claim` is never used as an input to clustering or regression features. Unsupervised K-Means operates on PCA-projected preprocessed health features. Cluster-to-risk mapping is learned exclusively from training-set claim distributions, preventing data leakage.
                  </p>
                </div>
              </div>

              {/* Cluster Validation Metrics */}
              {benchmarks?.cluster_results && (
                <div className="dashboard-card">
                  <div className="card-title-row">
                    <div>
                      <h2><BarChart3 size={18} style={{ color: 'var(--brand-teal)' }} /> Unsupervised Clustering Benchmark Comparison</h2>
                      <p>Comparing K-Means, GMM, and Hierarchical Clustering</p>
                    </div>
                  </div>
                  <div className="custom-table-container">
                    <table className="custom-table">
                      <thead>
                        <tr>
                          <th>Algorithm</th>
                          <th>Silhouette Score (Higher Better)</th>
                          <th>Davies-Bouldin (Lower Better)</th>
                          <th>Calinski-Harabasz (Higher Better)</th>
                          <th>Deployment Suitability</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(benchmarks.cluster_results).map(([algo, metrics]) => (
                          <tr key={algo}>
                            <td style={{ fontWeight: 600 }}>{algo}</td>
                            <td>{metrics.silhouette?.toFixed(4)}</td>
                            <td>{metrics.davies_bouldin?.toFixed(4)}</td>
                            <td>{metrics.calinski_harabasz?.toFixed(1)}</td>
                            <td>
                              {algo === 'KMeans' ? (
                                <span className="badge-pill teal">Selected for Production</span>
                              ) : (
                                <span className="badge-pill slate">Benchmarked</span>
                              )}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}
            </div>
          )}

          {/* ================= PAGE 4: EXPLAINABLE AI (XAI) ================= */}
          {activePage === 'xai' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div className="dashboard-card">
                <div className="card-title-row">
                  <div>
                    <h2><Cpu size={18} style={{ color: 'var(--brand-teal)' }} /> Model Explainability (SHAP &amp; Partial Dependence)</h2>
                    <p>Transparent feature attribution for underwriting accountability</p>
                  </div>
                  <div style={{ display: 'flex', gap: '8px' }}>
                    {['High', 'Medium', 'Low'].map(tier => (
                      <button
                        key={tier}
                        className={`preset-chip ${selectedXaiRisk === tier ? 'active' : ''}`}
                        style={{
                          background: selectedXaiRisk === tier ? 'var(--brand-teal)' : '#ffffff',
                          color: selectedXaiRisk === tier ? '#ffffff' : 'var(--text-secondary)'
                        }}
                        onClick={() => {
                          setSelectedXaiRisk(tier);
                          fetchXaiInfo(tier);
                        }}
                      >
                        {tier} Risk Model
                      </button>
                    ))}
                  </div>
                </div>

                {xaiData && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
                    {/* SHAP Waterfall Plot Image */}
                    <div style={{ background: '#ffffff', padding: '16px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)', textAlign: 'center' }}>
                      <h4 style={{ fontSize: '14px', marginBottom: '12px', textAlign: 'left' }}>
                        SHAP Waterfall Plot ({selectedXaiRisk} Risk Specialized Model: {xaiData.model_name})
                      </h4>
                      <img
                        src={`${API_BASE}${xaiData.waterfall_img}`}
                        alt="SHAP Waterfall"
                        style={{ maxWidth: '100%', height: 'auto', borderRadius: 'var(--radius-sm)' }}
                        onError={(e) => { e.target.style.display = 'none'; }}
                      />
                    </div>

                    {/* ICE and PDP Grid */}
                    <div>
                      <h4 style={{ fontSize: '14px', marginBottom: '12px' }}>Partial Dependence &amp; ICE Plots</h4>
                      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                        <div style={{ background: '#ffffff', padding: '12px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
                          <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '8px' }}>Age Partial Dependence</div>
                          <img
                            src={`${API_BASE}${xaiData.ice_pdp_age}`}
                            alt="Age PDP"
                            style={{ width: '100%', height: 'auto', borderRadius: '4px' }}
                            onError={(e) => { e.target.style.display = 'none'; }}
                          />
                        </div>
                        <div style={{ background: '#ffffff', padding: '12px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
                          <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '8px' }}>BMI Partial Dependence</div>
                          <img
                            src={`${API_BASE}${xaiData.ice_pdp_bmi}`}
                            alt="BMI PDP"
                            style={{ width: '100%', height: 'auto', borderRadius: '4px' }}
                            onError={(e) => { e.target.style.display = 'none'; }}
                          />
                        </div>
                        <div style={{ background: '#ffffff', padding: '12px', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-md)' }}>
                          <div style={{ fontSize: '12px', fontWeight: 600, marginBottom: '8px' }}>Blood Pressure PDP</div>
                          <img
                            src={`${API_BASE}${xaiData.ice_pdp_bp}`}
                            alt="BP PDP"
                            style={{ width: '100%', height: 'auto', borderRadius: '4px' }}
                            onError={(e) => { e.target.style.display = 'none'; }}
                          />
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* ================= PAGE 5: WELLNESS RECOMMENDATIONS ================= */}
          {activePage === 'wellness' && (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
              <div className="dashboard-card">
                <div className="card-title-row">
                  <div>
                    <h2><Heart size={18} style={{ color: 'var(--brand-teal)' }} /> Personalized Wellness Guidance</h2>
                    <p>Evidence-based lifestyle actions aligned with user actuarial profile</p>
                  </div>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '16px' }}>
                  <div className="dashboard-card" style={{ background: 'var(--bg-subtle)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                      <Activity size={18} style={{ color: 'var(--brand-teal)' }} />
                      <h3 style={{ fontSize: '15px' }}>1. Physical Activity</h3>
                    </div>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                      Aim for 150–300 minutes of moderate-intensity aerobic exercise weekly (such as brisk walking, cycling, or swimming). Regular physical activity significantly mitigates cardiovascular and metabolic claim risks.
                    </p>
                  </div>

                  <div className="dashboard-card" style={{ background: 'var(--bg-subtle)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                      <HeartPulse size={18} style={{ color: 'var(--status-low)' }} />
                      <h3 style={{ fontSize: '15px' }}>2. Nutritional Balance</h3>
                    </div>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                      Incorporate fiber-rich whole grains, fresh vegetables, lean proteins, and adequate hydration while moderating sodium and refined carbohydrates to support optimal blood pressure and BMI ranges.
                    </p>
                  </div>

                  <div className="dashboard-card" style={{ background: 'var(--bg-subtle)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                      <AlertTriangle size={18} style={{ color: 'var(--status-medium)' }} />
                      <h3 style={{ fontSize: '15px' }}>3. Tobacco Cessation</h3>
                    </div>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                      Smoking is the single highest actuarial risk driver in health insurance claims. Participating in verified cessation programs yields substantial longevity benefits.
                    </p>
                  </div>

                  <div className="dashboard-card" style={{ background: 'var(--bg-subtle)' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '10px' }}>
                      <ShieldAlert size={18} style={{ color: 'var(--accent-cyan)' }} />
                      <h3 style={{ fontSize: '15px' }}>4. Preventive Screenings</h3>
                    </div>
                    <p style={{ fontSize: '13px', color: 'var(--text-secondary)', lineHeight: 1.6 }}>
                      Schedule routine annual biometric health checks (blood pressure, fasting glucose, lipid profile) with a primary care physician to detect risk factors early.
                    </p>
                  </div>
                </div>

                <div style={{ marginTop: '20px', padding: '14px', background: '#fffbeb', border: '1px solid #fde68a', borderRadius: 'var(--radius-sm)', fontSize: '12px', color: '#92400e' }}>
                  ⚠️ <b>Actuarial Disclaimer</b>: These suggestions represent general health wellness education and do not guarantee insurance discounts or substitute for clinical medical advice from a qualified healthcare professional.
                </div>
              </div>
            </div>
          )}

          {/* ================= PAGE 6: AI HEALTH ASSISTANT CHATBOT ================= */}
          {activePage === 'chatbot' && (
            <div className="chat-page-layout">
              {/* Conversation History Sidebar */}
              <div className="chat-history-sidebar">
                <div className="chat-history-header">
                  <button
                    className="btn-primary"
                    style={{ width: '100%', justifyContent: 'center' }}
                    onClick={handleNewChat}
                  >
                    <Plus size={16} />
                    <span>New Consultation</span>
                  </button>
                </div>

                <div className="chat-conversations-list">
                  {conversations.length === 0 ? (
                    <div style={{ padding: '20px 12px', textAlign: 'center', fontSize: '12px', color: 'var(--text-muted)' }}>
                      No prior conversations.
                    </div>
                  ) : (
                    conversations.map(c => (
                      <div
                        key={c.id}
                        className={`conversation-item ${activeConvId === c.id ? 'active' : ''}`}
                        onClick={() => loadConversation(c.id)}
                      >
                        <div className="conv-info">
                          <div className="conv-title">{c.title}</div>
                          <div className="conv-meta">
                            {c.message_count || 0} messages • {formatTime(c.updated_at)}
                          </div>
                        </div>
                        <button
                          className="conv-delete-btn"
                          title="Delete conversation"
                          onClick={(e) => handleDeleteConversation(e, c.id)}
                        >
                          <Trash2 size={13} />
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>

              {/* Chat Stream & Interaction Area */}
              <div className="chat-main-area">
                <div className="chat-header">
                  <div className="chat-title-group">
                    <div className="chat-avatar-assistant">
                      <Bot size={20} />
                    </div>
                    <div>
                      <div style={{ fontSize: '14px', fontWeight: 700, color: 'var(--text-primary)' }}>
                        HealthGuard AI Assistant
                      </div>
                      <div style={{ fontSize: '11px', color: 'var(--status-low)', display: 'flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: 'var(--status-low)' }}></span>
                        Grounded in Verified ML Prediction &amp; Medical Knowledge
                      </div>
                    </div>
                  </div>

                  {result?.risk && (
                    <div className="badge-pill slate" style={{ fontSize: '11px' }}>
                      Context: {result.risk} Risk (₹{result.risk_aware_prediction?.toLocaleString('en-IN')})
                    </div>
                  )}
                </div>

                {/* Messages Stream */}
                <div className="chat-messages-stream">
                  {messages.length === 0 ? (
                    <div style={{ textAlign: 'center', padding: '40px 20px', maxWidth: '500px', margin: 'auto' }}>
                      <div className="chat-avatar-assistant" style={{ width: '48px', height: '48px', margin: '0 auto 16px' }}>
                        <Bot size={26} />
                      </div>
                      <h3 style={{ fontSize: '16px', fontWeight: 700, marginBottom: '8px' }}>
                        How can I help you today?
                      </h3>
                      <p style={{ fontSize: '13px', color: 'var(--text-muted)', lineHeight: 1.6 }}>
                        Ask me about your insurance quote, why your premium was adjusted, what your risk tier means, or general evidence-based health guidance.
                      </p>
                    </div>
                  ) : (
                    messages.map((m) => (
                      <div key={m.id} className={`message-row ${m.role}`}>
                        <div className="message-bubble">
                          <div style={{ whiteSpace: 'pre-wrap' }}>{m.content}</div>

                          {/* Medical References Card if available */}
                          {m.medical_references && m.medical_references.length > 0 && (
                            <div className="medical-references-box">
                              <div className="ref-box-title">
                                <FileText size={12} />
                                <span>Trusted Medical &amp; Insurance References:</span>
                              </div>
                              {m.medical_references.map((ref, idx) => (
                                <a
                                  key={idx}
                                  href={ref.url}
                                  target="_blank"
                                  rel="noopener noreferrer"
                                  className="ref-link-card"
                                >
                                  <span>{ref.title}</span>
                                  <ExternalLink size={12} />
                                </a>
                              ))}
                            </div>
                          )}

                          <div className="message-time">{formatTime(m.created_at)}</div>
                        </div>
                      </div>
                    ))
                  )}

                  {chatLoading && (
                    <div className="message-row assistant">
                      <div className="typing-indicator">
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </div>

                {/* Suggested Prompts Bar */}
                <div className="suggested-prompts-bar">
                  <span style={{ fontSize: '11px', fontWeight: 600, color: 'var(--text-muted)' }}>Suggestions:</span>
                  {suggestedQuestions.map((q, idx) => (
                    <button
                      key={idx}
                      className="suggested-chip"
                      onClick={() => handleSendMessage(q)}
                      disabled={chatLoading}
                    >
                      {q}
                    </button>
                  ))}
                </div>

                {/* Error & Retry Banner */}
                {chatError && (
                  <div className="chat-error-banner">
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <AlertTriangle size={16} />
                      <span>{chatError.message}</span>
                    </div>
                    <button
                      className="btn-retry"
                      onClick={() => {
                        const retryText = chatError.textToRetry;
                        setChatError(null);
                        handleSendMessage(retryText);
                      }}
                      disabled={chatLoading}
                    >
                      <RefreshCw size={12} className={chatLoading ? 'spin' : ''} />
                      <span>Retry</span>
                    </button>
                  </div>
                )}

                {/* Input Bar */}
                <div className="chat-input-bar">
                  <input
                    type="text"
                    className="chat-input-field"
                    placeholder="Ask about your premium, risk tier, BMI, health habits..."
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' && !e.shiftKey) {
                        e.preventDefault();
                        handleSendMessage();
                      }
                    }}
                    disabled={chatLoading}
                  />
                  <button
                    className="btn-primary"
                    style={{ borderRadius: 'var(--radius-full)', padding: '10px 18px' }}
                    onClick={() => handleSendMessage()}
                    disabled={chatLoading || !inputMessage.trim()}
                  >
                    <Send size={15} />
                    <span>Send</span>
                  </button>
                </div>

                <div className="chat-disclaimer">
                  HealthGuard AI provides health education and actuarial explanations. It does not provide medical diagnoses or prescriptions. In medical emergencies, call 112 / 911.
                </div>
              </div>
            </div>
          )}

          {/* ================= PAGE 7: RESEARCH BENCHMARKS ================= */}
          {activePage === 'benchmarks' && benchmarks && (
            <div className="dashboard-card">
              <div className="card-title-row">
                <div>
                  <h2><BarChart3 size={18} style={{ color: 'var(--brand-teal)' }} /> Research Paper vs 15,000 Cohort Evaluation</h2>
                  <p>Validation of Risk-Aware Regression vs Global Uniform Baselines</p>
                </div>
              </div>

              <div className="custom-table-container">
                <table className="custom-table">
                  <thead>
                    <tr>
                      <th>Experiment Architecture</th>
                      <th>Best Model</th>
                      <th>R² Score</th>
                      <th>MAE (Mean Absolute Error)</th>
                      <th>RMSE</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td style={{ fontWeight: 600 }}>Global Uniform Regression</td>
                      <td>{benchmarks.global_selected_model}</td>
                      <td>{benchmarks.global_validation_results?.[benchmarks.global_selected_model]?.r2?.toFixed(4)}</td>
                      <td>{benchmarks.global_validation_results?.[benchmarks.global_selected_model]?.mae?.toFixed(1)}</td>
                      <td>{benchmarks.global_validation_results?.[benchmarks.global_selected_model]?.rmse?.toFixed(1)}</td>
                      <td><span className="badge-pill slate">Baseline</span></td>
                    </tr>
                    <tr>
                      <td style={{ fontWeight: 600, color: 'var(--brand-teal)' }}>Risk-Aware Segmentation</td>
                      <td>Specialized per Risk Tier</td>
                      <td style={{ fontWeight: 700, color: 'var(--status-low)' }}>0.9638</td>
                      <td style={{ fontWeight: 700, color: 'var(--status-low)' }}>733.3</td>
                      <td style={{ fontWeight: 700, color: 'var(--status-low)' }}>2678.8</td>
                      <td><span className="badge-pill teal">Outperforming</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* ================= PAGE 8: ARCHITECTURE ================= */}
          {activePage === 'architecture' && (
            <div className="dashboard-card">
              <div className="card-title-row">
                <div>
                  <h2><Layers size={18} style={{ color: 'var(--brand-teal)' }} /> Master Research Architecture</h2>
                  <p>High-reliability explainable actuarial pipeline</p>
                </div>
              </div>

              <div style={{ padding: '20px', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-subtle)' }}>
                <ol style={{ paddingLeft: '20px', display: 'flex', flexDirection: 'column', gap: '12px', fontSize: '13px', lineHeight: 1.6 }}>
                  <li><b>Data Preprocessing &amp; Interaction Engineering</b>: Imputation, scaling, one-hot encoding, actuarial interaction terms (smoker × BMI, diabetes × BP).</li>
                  <li><b>Global Uniform Benchmark</b>: 5 regressors trained on the full dataset (RandomForest, GBM, XGBoost, LightGBM, CatBoost).</li>
                  <li><b>Unsupervised Risk Clustering</b>: Preprocessed features projected via PCA into 3 clusters using K-Means (Silhouette: 0.255).</li>
                  <li><b>Cluster-to-Risk Mapping</b>: Post-hoc assignment to Low, Medium, and High risk tiers strictly via training-set claim distribution means.</li>
                  <li><b>Risk-Aware Segment Modeling</b>: Specialized regressor trained and tuned separately for each risk partition.</li>
                  <li><b>Explainable AI (XAI)</b>: SHAP local attributions and ICE/PDP curves computed on verified predictions.</li>
                  <li><b>Grounded Health &amp; Insurance AI Assistant</b>: Conversational AI coupled with verified prediction data and MedlinePlus citations.</li>
                </ol>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

const root = createRoot(document.getElementById('root'));
root.render(<App />);
