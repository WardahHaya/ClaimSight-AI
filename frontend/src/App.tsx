import React, { useEffect, useRef, useState } from 'react';
import {
  CheckCircle,
  ChevronDown,
  ChevronUp,
  Eye,
  FileText,
  Image as ImageIcon,
  MessageSquare,
  Plus,
  Send,
  Shield,
  Trash2,
  X,
} from 'lucide-react';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  image_analysis?: string | null;
  image_data_url?: string | null;
  pipeline_route?: string[] | null;
  response_mode?: string | null;
}

interface Session {
  id: string;
  name: string;
}

const API_BASE =
  import.meta.env.VITE_API_BASE ??
  (typeof window !== 'undefined' && window.location.port === '5173' ? 'http://127.0.0.1:8000' : '');

const apiUrl = (path: string) => `${API_BASE}${path}`;

const backendLabel =
  typeof window !== 'undefined' ? (API_BASE || window.location.origin) : API_BASE || 'http://127.0.0.1:8000';

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([]);
  const [activeSession, setActiveSession] = useState<string>('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputText, setInputText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [expandedRoutes, setExpandedRoutes] = useState<Record<number, boolean>>({});

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const savedSessions = localStorage.getItem('claimsight_sessions');
    if (savedSessions) {
      try {
        const parsed = JSON.parse(savedSessions) as Session[];
        if (parsed.length > 0) {
          setSessions(parsed);
          setActiveSession(parsed[0].id);
          return;
        }
      } catch (error) {
        console.error('Failed to parse saved sessions', error);
      }
    }

    const defaultId = `session_${Math.random().toString(36).slice(2, 11)}`;
    const defaultSession = { id: defaultId, name: 'Claims Triage Session 1' };
    setSessions([defaultSession]);
    setActiveSession(defaultId);
    localStorage.setItem('claimsight_sessions', JSON.stringify([defaultSession]));
  }, []);

  useEffect(() => {
    if (!activeSession) {
      return;
    }

    const fetchHistory = async () => {
      try {
        const response = await fetch(apiUrl(`/history/${activeSession}`));
        if (!response.ok) {
          return;
        }
        const data = await response.json();
        const formatted: Message[] = data.history.map((item: any) => ({
          role: item.role,
          content: item.content,
          image_analysis: item.image_analysis,
          image_data_url: item.image_data_url,
          pipeline_route: item.pipeline_route,
          response_mode: item.response_mode,
        }));
        setMessages(formatted);
      } catch (error) {
        console.error('Failed to load session history', error);
      }
    };

    fetchHistory();
  }, [activeSession]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const saveSessions = (updatedSessions: Session[]) => {
    setSessions(updatedSessions);
    localStorage.setItem('claimsight_sessions', JSON.stringify(updatedSessions));
  };

  const resetImage = () => {
    setSelectedImage(null);
    setImagePreview(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  const createNewSession = () => {
    const newId = `session_${Math.random().toString(36).slice(2, 11)}`;
    const newSession = { id: newId, name: `Claims Triage Session ${sessions.length + 1}` };
    const updatedSessions = [newSession, ...sessions];
    saveSessions(updatedSessions);
    setActiveSession(newId);
    setMessages([]);
    resetImage();
  };

  const handleDeleteSession = async (sessionId: string, event: React.MouseEvent) => {
    event.stopPropagation();
    try {
      await fetch(apiUrl(`/reset/${sessionId}`), { method: 'POST' });
    } catch (error) {
      console.error('Failed to reset session in database', error);
    }

    const updatedSessions = sessions.filter((session) => session.id !== sessionId);
    if (updatedSessions.length === 0) {
      const fallbackId = `session_${Math.random().toString(36).slice(2, 11)}`;
      const fallbackSession = { id: fallbackId, name: 'Claims Triage Session 1' };
      saveSessions([fallbackSession]);
      setActiveSession(fallbackId);
      setMessages([]);
    } else {
      saveSessions(updatedSessions);
      if (activeSession === sessionId) {
        setActiveSession(updatedSessions[0].id);
      }
    }
    resetImage();
  };

  const loadImageFile = (file: File) => {
    if (!file.type.startsWith('image/')) {
      return;
    }
    setSelectedImage(file);
    const reader = new FileReader();
    reader.onloadend = () => setImagePreview(reader.result as string);
    reader.readAsDataURL(file);
  };

  const handleImageChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) {
      loadImageFile(file);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files?.[0];
    if (file) {
      loadImageFile(file);
    }
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    if (!inputText.trim() && !selectedImage) {
      return;
    }

    const userText = inputText.trim();
    const localImagePreview = imagePreview;
    const imageToSend = selectedImage;

    setInputText('');
    setIsLoading(true);
    resetImage();

    const userMessage: Message = {
      role: 'user',
      content: userText || 'Uploaded a vehicle damage photo for analysis.',
      image_data_url: localImagePreview,
    };
    setMessages((previous) => [...previous, userMessage]);

    const formData = new FormData();
    formData.append('session_id', activeSession);
    formData.append('query', userText);
    if (imageToSend) {
      formData.append('image', imageToSend);
    }

    try {
      const response = await fetch(apiUrl('/chat'), {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => null);
        throw new Error(payload?.detail || 'The backend returned an error.');
      }

      const data = await response.json();
      const assistantMessage: Message = {
        role: 'assistant',
        content: data.response,
        image_analysis: data.image_analysis,
        pipeline_route: data.pipeline_route,
        response_mode: data.response_mode,
      };
      setMessages((previous) => [...previous, assistantMessage]);
    } catch (error) {
      console.error(error);
      const assistantMessage: Message = {
        role: 'assistant',
        content:
          error instanceof Error
            ? error.message
            : 'Failed to connect to the triage backend. Make sure the FastAPI server is running on localhost:8000.',
        response_mode: 'safe',
      };
      setMessages((previous) => [...previous, assistantMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSuggestionClick = (suggestionText: string) => {
    setInputText(suggestionText);
  };

  const toggleRoute = (index: number) => {
    setExpandedRoutes((previous) => ({
      ...previous,
      [index]: !previous[index],
    }));
  };

  const getRouteBadge = (message: Message) => {
    const routeText = (message.pipeline_route || []).join(' ');

    if (message.response_mode === 'direct' || routeText.includes('Generate Direct Response')) {
      return (
        <span
          className="status-badge"
          style={{
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            color: '#3b82f6',
            borderColor: 'rgba(59, 130, 246, 0.2)',
          }}
        >
          Answered Directly
        </span>
      );
    }

    if (message.response_mode === 'safe' || routeText.includes('Generate Safe Response')) {
      return (
        <span
          className="status-badge"
          style={{
            backgroundColor: 'rgba(244, 63, 94, 0.1)',
            color: '#f43f5e',
            borderColor: 'rgba(244, 63, 94, 0.2)',
          }}
        >
          Insufficient Info
        </span>
      );
    }

    if (routeText.includes('Retry')) {
      const retryMatches = routeText.match(/Retry Query Rewrite #[0-9]+/g);
      const retryText = retryMatches ? `Retried (${retryMatches.length}x)` : 'Retried Query';
      return (
        <span
          className="status-badge"
          style={{
            backgroundColor: 'rgba(245, 158, 11, 0.1)',
            color: '#f59e0b',
            borderColor: 'rgba(245, 158, 11, 0.2)',
          }}
        >
          {retryText}
        </span>
      );
    }

    if (message.response_mode === 'rag' || routeText.includes('Generate Grounded Response')) {
      return (
        <span
          className="status-badge"
          style={{
            backgroundColor: 'rgba(16, 185, 129, 0.1)',
            color: '#10b981',
            borderColor: 'rgba(16, 185, 129, 0.2)',
          }}
        >
          Answered from Documents
        </span>
      );
    }

    return null;
  };

  return (
    <div className="app-container" onDragOver={handleDragOver} onDrop={handleDrop}>
      {isDragging && (
        <div className="drag-overlay" onDragLeave={handleDragLeave}>
          <ImageIcon size={48} style={{ color: '#3b82f6' }} />
          <h3>Drop vehicle damage photo here</h3>
          <p style={{ color: '#94a3b8', fontSize: '13px' }}>Supports JPG, PNG, and WEBP</p>
        </div>
      )}

      <div className="sidebar">
        <div className="sidebar-header">
          <div className="logo-icon">CS</div>
          <div className="logo-text">
            <h1>ClaimSight AI</h1>
            <span>Triage Assistant</span>
          </div>
        </div>

        <div className="sidebar-content">
          <button className="btn-new-session" onClick={createNewSession}>
            <Plus size={16} />
            New Claims Session
          </button>

          <div>
            <h3 className="section-title">Active Sessions</h3>
            <div className="session-list">
              {sessions.map((session) => (
                <div
                  key={session.id}
                  className={`session-item ${activeSession === session.id ? 'active' : ''}`}
                  onClick={() => {
                    setActiveSession(session.id);
                    resetImage();
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', overflow: 'hidden' }}>
                    <MessageSquare size={14} style={{ flexShrink: 0 }} />
                    <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                      {session.name}
                    </span>
                  </div>
                  <button
                    className="btn-delete-session"
                    onClick={(event) => handleDeleteSession(session.id, event)}
                    title="Delete session"
                  >
                    <Trash2 size={14} />
                  </button>
                </div>
              ))}
            </div>
          </div>

          <div style={{ marginTop: 'auto' }}>
            <div className="disclaimer-card">
              <h3>
                <Shield size={14} />
                Cyber & Ethical Policy
              </h3>
              <p style={{ marginBottom: '8px' }}>
                Uploaded damage photos stay local to your machine and are only sent to the configured LLM provider for the vision step.
              </p>
              <p>
                <strong>Important:</strong> AI triage does not replace a licensed adjuster or your active policy contract.
              </p>
            </div>
          </div>
        </div>
      </div>

      <div className="main-chat">
        <div className="chat-header">
          <div className="active-session-title">
            <h2>{sessions.find((session) => session.id === activeSession)?.name || 'Claims Assistant'}</h2>
            <span className="status-badge">
              <CheckCircle size={10} /> Active
            </span>
          </div>
          <div style={{ display: 'flex', gap: '8px', color: '#94a3b8', fontSize: '13px', alignItems: 'center' }}>
            <span
              style={{
                backgroundColor: 'rgba(255,255,255,0.05)',
                padding: '6px 12px',
                borderRadius: '6px',
                border: '1px solid var(--border-color)',
              }}
            >
              Backend: <strong>{backendLabel}</strong>
            </span>
          </div>
        </div>

        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-screen">
              <div className="welcome-icon">
                <Shield size={36} />
              </div>
              <h2>ClaimSight AI Triage Assistant</h2>
              <p>
                Upload a vehicle-damage photo and ask coverage or claim-handling questions. The app will fuse
                image analysis with a document-grounded RAG workflow and show the exact path taken.
              </p>

              <div className="welcome-suggestions">
                <div
                  className="suggestion-card"
                  onClick={() =>
                    handleSuggestionClick(
                      'My front bumper is dented from sliding into a guardrail in the snow. Is this covered under collision or comprehensive?'
                    )
                  }
                >
                  <h4>1. Slide into Guardrail</h4>
                  <p>Collision versus comprehensive, deductible flow, and likely next steps.</p>
                </div>
                <div
                  className="suggestion-card"
                  onClick={() =>
                    handleSuggestionClick(
                      'A tree branch fell onto my roof during a windstorm and cracked my windshield. Do I need a police report?'
                    )
                  }
                >
                  <h4>2. Fallen Tree Branch</h4>
                  <p>Comprehensive coverage, documentation, and reporting requirements.</p>
                </div>
                <div
                  className="suggestion-card"
                  onClick={() =>
                    handleSuggestionClick(
                      'What is the process and timeline to file my auto claim, and does my policy cover rental cars?'
                    )
                  }
                >
                  <h4>3. Timelines & Rental Cars</h4>
                  <p>Claims intake, insurer response expectations, and rental reimbursement.</p>
                </div>
                <div
                  className="suggestion-card"
                  onClick={() =>
                    handleSuggestionClick(
                      'My older sedan has minor hood scratches. Will the insurer likely pay for new OEM parts?'
                    )
                  }
                >
                  <h4>4. Parts Guidance</h4>
                  <p>Repair parts, actual cash value, and older-vehicle tradeoffs.</p>
                </div>
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={index} className={`message-wrapper ${message.role}`}>
                <div className="message-bubble">
                  <div style={{ whiteSpace: 'pre-line' }}>{message.content}</div>

                  {message.role === 'user' && message.image_data_url && (
                    <div style={{ marginTop: '12px' }}>
                      <img
                        src={message.image_data_url}
                        alt="Uploaded vehicle damage"
                        style={{
                          width: '220px',
                          maxWidth: '100%',
                          borderRadius: '10px',
                          border: '1px solid rgba(255,255,255,0.16)',
                          display: 'block',
                        }}
                      />
                    </div>
                  )}

                  {message.role === 'assistant' && message.image_analysis && (
                    <div
                      style={{
                        marginTop: '16px',
                        backgroundColor: 'rgba(59, 130, 246, 0.05)',
                        border: '1px solid rgba(59, 130, 246, 0.2)',
                        borderRadius: '8px',
                        padding: '12px',
                      }}
                    >
                      <h4
                        style={{
                          display: 'flex',
                          alignItems: 'center',
                          gap: '6px',
                          fontSize: '13px',
                          color: '#60a5fa',
                          marginBottom: '6px',
                        }}
                      >
                        <Eye size={14} /> Damage Photo Analysis (Vision LLM)
                      </h4>
                      <p style={{ fontSize: '12px', color: '#94a3b8', lineHeight: '1.5' }}>{message.image_analysis}</p>
                    </div>
                  )}
                </div>

                {message.role === 'assistant' && message.pipeline_route && (
                  <div className="pipeline-visualizer">
                    <div className="pipeline-header" onClick={() => toggleRoute(index)}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                        <FileText size={12} />
                        <span>RAG Execution Trace</span>
                        {getRouteBadge(message)}
                      </div>
                      <div>{expandedRoutes[index] ? <ChevronUp size={14} /> : <ChevronDown size={14} />}</div>
                    </div>

                    {expandedRoutes[index] && (
                      <div className="pipeline-steps">
                        {message.pipeline_route.map((step, stepIndex) => {
                          let stepClass = '';
                          if (step.includes('Evaluate Relevance: YES') || step.includes('Response')) {
                            stepClass = 'success';
                          } else if (step.includes('Evaluate Relevance: NO') || step.includes('Safe')) {
                            stepClass = 'failure';
                          }
                          return (
                            <div key={stepIndex} className={`pipeline-step ${stepClass}`}>
                              {step}
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))
          )}

          {isLoading && (
            <div className="message-wrapper assistant">
              <div className="message-bubble typing-bubble">
                <div className="dot"></div>
                <div className="dot"></div>
                <div className="dot"></div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {imagePreview && (
          <div className="upload-preview-container">
            <div className="upload-preview-card">
              <img src={imagePreview} alt="Damage upload preview" />
              <button className="btn-remove-preview" onClick={resetImage}>
                <X size={10} />
              </button>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <span style={{ fontSize: '13px', fontWeight: '500' }}>{selectedImage?.name}</span>
              <span style={{ fontSize: '11px', color: '#94a3b8' }}>
                {(selectedImage?.size || 0) > 1024 * 1024
                  ? `${((selectedImage?.size || 0) / (1024 * 1024)).toFixed(2)} MB`
                  : `${((selectedImage?.size || 0) / 1024).toFixed(1)} KB`}
              </span>
            </div>
          </div>
        )}

        <div className="chat-input-container">
          <form className="chat-input-form" onSubmit={handleSubmit}>
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleImageChange}
              accept="image/*"
              style={{ display: 'none' }}
            />
            <button
              type="button"
              className="input-action-btn"
              onClick={() => fileInputRef.current?.click()}
              title="Upload vehicle damage photo"
              disabled={isLoading}
            >
              <ImageIcon size={20} />
            </button>

            <input
              type="text"
              className="chat-text-input"
              value={inputText}
              onChange={(event) => setInputText(event.target.value)}
              placeholder={imagePreview ? 'Ask a question about this photo...' : 'Type your auto insurance question...'}
              disabled={isLoading}
            />

            <button
              type="submit"
              className="input-action-btn submit-btn"
              disabled={isLoading || (!inputText.trim() && !selectedImage)}
            >
              <Send size={16} />
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
