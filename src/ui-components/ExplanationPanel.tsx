/**
 * Explanation Panel Component
 * Displays agent explanations of their inventions and discoveries
 * using their mathematical and physics knowledge
 * 
 * Features:
 * - UNLIMITED explanation storage
 * - Web search integration for scientific terms
 * - Google and Wikipedia links for deeper research
 */

import React, { useState, useMemo } from 'react';
import type { Agent, Invention } from '../types';
import type { PhysicsConcept } from '../science-system/physics';
import type { MathConcept } from '../science-system/mathematics';
import type { AgentExplanation, WebSearchResult, InternetKnowledge, InternetLearningResult } from '../explanation-system';
import { 
  explainInvention, 
  explainPhysicsDiscovery, 
  explainMathDiscovery,
  formatExplanationForDisplay,
  generateWebEnhancedExplanation,
  searchForInventionTerms,
  searchForPhysicsTerms,
  searchForMathTerms,
  openGoogleSearch,
  openWikipedia,
  // Internet Learning System
  calculateAgentIntelligence,
  canAccessInternet,
  getAvailableInternetKnowledge,
  learnFromInternet,
  autoEvolveWithKnowledge,
  generateTheory,
  INTELLIGENCE_THRESHOLDS,
  INTERNET_KNOWLEDGE_DATABASE
} from '../explanation-system';

interface ExplanationPanelProps {
  agents: Agent[];
  unlockedMath: MathConcept[];
  unlockedPhysics: PhysicsConcept[];
  currentTick: number;
  selectedAgentId: number | null;
  // Optional: Persisted internet learning state from parent
  persistedInternetKnowledge?: Set<string>;
  onInternetKnowledgeChange?: (knowledge: Set<string>) => void;
  onLearningComplete?: (count: number) => void;
}

const SOPHISTICATION_COLORS: Record<number, string> = {
  1: '#666',
  2: '#888',
  3: '#4caf50',
  4: '#8bc34a',
  5: '#cddc39',
  6: '#ffeb3b',
  7: '#ff9800',
  8: '#ff5722',
  9: '#e91e63',
  10: '#9c27b0',
};

export const ExplanationPanel: React.FC<ExplanationPanelProps> = ({
  agents,
  unlockedMath,
  unlockedPhysics,
  currentTick,
  selectedAgentId,
  persistedInternetKnowledge,
  onInternetKnowledgeChange,
  onLearningComplete
}) => {
  const [selectedSubject, setSelectedSubject] = useState<'inventions' | 'physics' | 'math'>('inventions');
  const [explanations, setExplanations] = useState<AgentExplanation[]>([]); // UNLIMITED storage
  const [webSearchResults, setWebSearchResults] = useState<Map<string, WebSearchResult>>(new Map());
  const [showWebSearch, setShowWebSearch] = useState<boolean>(true);
  const [searchMode, setSearchMode] = useState<'basic' | 'enhanced'>('enhanced');
  
  // Internet Learning State - use persisted state if provided
  const [internetKnowledgeLearned, setInternetKnowledgeLearnedLocal] = useState<Set<string>>(
    persistedInternetKnowledge || new Set()
  );
  const [learningResults, setLearningResults] = useState<InternetLearningResult[]>([]);
  const [autoEvolveEnabled, setAutoEvolveEnabled] = useState<boolean>(true);
  const [theories, setTheories] = useState<Array<{ theoryName: string; description: string; searchUrl: string; potentialImpact: number }>>([]);

  // Sync with persisted state
  const setInternetKnowledgeLearned = (updater: Set<string> | ((prev: Set<string>) => Set<string>)) => {
    const newValue = typeof updater === 'function' 
      ? updater(internetKnowledgeLearned) 
      : updater;
    setInternetKnowledgeLearnedLocal(newValue);
    onInternetKnowledgeChange?.(newValue);
  };

  // Sync from persisted state when it changes
  React.useEffect(() => {
    if (persistedInternetKnowledge && persistedInternetKnowledge.size !== internetKnowledgeLearned.size) {
      setInternetKnowledgeLearnedLocal(persistedInternetKnowledge);
    }
  }, [persistedInternetKnowledge]);

  // Get the selected agent or most creative agent
  const focusAgent = useMemo(() => {
    if (selectedAgentId !== null) {
      return agents.find(a => a.id === selectedAgentId);
    }
    // Default to most creative agent with inventions
    return agents
      .filter(a => a.inventions.length > 0)
      .sort((a, b) => b.genes.creativity - a.genes.creativity)[0];
  }, [agents, selectedAgentId]);

  // Calculate agent intelligence and internet access
  const agentIntelligence = useMemo(() => {
    if (!focusAgent) return { intelligence: 0, canAccess: false, level: 'INSUFFICIENT' };
    return canAccessInternet(focusAgent, unlockedMath, unlockedPhysics);
  }, [focusAgent, unlockedMath, unlockedPhysics]);

  // Get available internet knowledge
  const availableInternetKnowledge = useMemo(() => {
    if (!focusAgent) return [];
    return getAvailableInternetKnowledge(focusAgent, unlockedMath, unlockedPhysics, internetKnowledgeLearned);
  }, [focusAgent, unlockedMath, unlockedPhysics, internetKnowledgeLearned]);
  // Handle learning from internet
  const handleLearnFromInternet = (knowledge: InternetKnowledge) => {
    if (!focusAgent) return;
    
    const result = learnFromInternet(focusAgent, knowledge, unlockedMath, unlockedPhysics);
    setLearningResults(prev => [result, ...prev]);
    
    if (result.success && result.knowledgeGained) {
      const newKnowledge = new Set(internetKnowledgeLearned).add(result.knowledgeGained!.id);
      setInternetKnowledgeLearned(newKnowledge);
      
      // Notify parent of learning
      onLearningComplete?.(1);
      
      // Auto-generate a theory if at innovation level
      if (agentIntelligence.intelligence >= INTELLIGENCE_THRESHOLDS.INNOVATION) {
        const knowledgeList = Array.from(newKnowledge)
          .map(id => [...INTERNET_KNOWLEDGE_DATABASE.mathematics, ...INTERNET_KNOWLEDGE_DATABASE.physics, 
                      ...INTERNET_KNOWLEDGE_DATABASE.technology, ...INTERNET_KNOWLEDGE_DATABASE.transcendent]
            .find(k => k.id === id))
          .filter(Boolean) as InternetKnowledge[];
        
        const theory = generateTheory(focusAgent, unlockedMath, unlockedPhysics, knowledgeList);
        if (theory) {
          setTheories(prev => [theory, ...prev]);
        }
      }
    }
  };

  // Auto-evolve effect
  const handleAutoEvolve = () => {
    if (!focusAgent || !autoEvolveEnabled) return;
    
    const evolution = autoEvolveWithKnowledge(focusAgent, unlockedMath, unlockedPhysics, internetKnowledgeLearned);
    if (evolution) {
      setLearningResults(prev => [{
        success: true,
        knowledgeGained: null,
        searchQuery: evolution.suggestedSearch,
        searchUrl: `https://www.google.com/search?q=${encodeURIComponent(evolution.suggestedSearch)}`,
        agentIntelligence: agentIntelligence.intelligence,
        requiredIntelligence: 0,
        insightGained: evolution.newInsight,
        bonusesApplied: { evolutionBonus: evolution.evolutionBonus },
        timestamp: Date.now()
      }, ...prev]);
    }
  };
  // Generate explanation for selected item
  const generateExplanation = (type: 'invention' | 'physics' | 'math', item: Invention | PhysicsConcept | MathConcept) => {
    if (!focusAgent) return;

    if (searchMode === 'enhanced') {
      // Use web-enhanced explanation
      const { explanation, webSearch } = generateWebEnhancedExplanation(
        focusAgent,
        { type, item },
        unlockedMath,
        unlockedPhysics,
        currentTick
      );
      
      setExplanations(prev => [explanation, ...prev]); // NO LIMIT - store all explanations
      setWebSearchResults(prev => new Map(prev).set(explanation.id, webSearch));
    } else {
      // Basic explanation without web enhancement
      let explanation: AgentExplanation;
      
      if (type === 'invention') {
        explanation = explainInvention(focusAgent, item as Invention, unlockedMath, unlockedPhysics, currentTick);
      } else if (type === 'physics') {
        explanation = explainPhysicsDiscovery(focusAgent, item as PhysicsConcept, unlockedMath, unlockedPhysics, currentTick);
      } else {
        explanation = explainMathDiscovery(focusAgent, item as MathConcept, unlockedMath, unlockedPhysics, currentTick);
      }

      setExplanations(prev => [explanation, ...prev]); // NO LIMIT - store all explanations
      
      // Still generate web search data for reference
      let webSearch: WebSearchResult;
      if (type === 'invention') {
        webSearch = searchForInventionTerms(item as Invention, unlockedMath, unlockedPhysics);
      } else if (type === 'physics') {
        webSearch = searchForPhysicsTerms(item as PhysicsConcept, unlockedMath);
      } else {
        webSearch = searchForMathTerms(item as MathConcept, unlockedPhysics);
      }
      setWebSearchResults(prev => new Map(prev).set(explanation.id, webSearch));
    }
  };
  // Get items to explain based on selection - NO LIMIT
  const availableItems = useMemo(() => {
    if (selectedSubject === 'inventions') {
      return focusAgent?.inventions || [];
    } else if (selectedSubject === 'physics') {
      return unlockedPhysics; // All physics discoveries
    } else {
      return unlockedMath; // All math discoveries
    }
  }, [selectedSubject, focusAgent, unlockedPhysics, unlockedMath]);

  if (!focusAgent && agents.length > 0) {
    return (
      <div style={{
        padding: '10px',
        background: '#151a30',
        borderRadius: '8px',
        border: '1px solid #333',
        marginBottom: '12px'
      }}>
        <h3>🧠 Agent Explanations</h3>
        <p style={{ opacity: 0.7, fontSize: '0.85em' }}>
          Select an agent or wait for one to make inventions to see explanations.
        </p>
      </div>
    );
  }

  return (
    <div style={{
      padding: '10px',
      background: '#151a30',
      borderRadius: '8px',
      border: '1px solid #333',
      marginBottom: '12px'
    }}>
      <h3>🧠 Agent Explanations</h3>
        {/* Agent info */}
      {focusAgent && (
        <div style={{ 
          fontSize: '0.85em', 
          marginBottom: '10px',
          padding: '8px',
          background: '#1a2040',
          borderRadius: '4px'
        }}>
          <strong>Agent {focusAgent.id}</strong> explaining
          <span style={{ marginLeft: '10px', opacity: 0.8 }}>
            🎨 Creativity: {(focusAgent.genes.creativity * 100).toFixed(0)}% |
            🔍 Curiosity: {(focusAgent.genes.curiosity * 100).toFixed(0)}%
          </span>
        </div>
      )}

      {/* Intelligence & Internet Access Status */}
      {focusAgent && (
        <div style={{
          marginBottom: '10px',
          padding: '10px',
          background: agentIntelligence.canAccess ? '#1a3020' : '#2a1a20',
          borderRadius: '6px',
          border: `1px solid ${agentIntelligence.canAccess ? '#4caf50' : '#f44336'}`
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
            <span style={{ fontWeight: 'bold' }}>
              🧠 Intelligence: {agentIntelligence.intelligence}
            </span>
            <span style={{
              padding: '2px 8px',
              borderRadius: '4px',
              fontSize: '0.8em',
              background: agentIntelligence.canAccess ? '#4caf50' : '#666',
              color: 'white'
            }}>
              {agentIntelligence.level}
            </span>
          </div>
          
          {/* Intelligence Progress Bar */}
          <div style={{ background: '#333', borderRadius: '4px', height: '8px', marginBottom: '8px' }}>
            <div style={{
              width: `${Math.min(100, (agentIntelligence.intelligence / INTELLIGENCE_THRESHOLDS.SINGULARITY) * 100)}%`,
              height: '100%',
              borderRadius: '4px',
              background: `linear-gradient(90deg, #4caf50, #ff9800, #e91e63, #9c27b0)`,
              transition: 'width 0.3s'
            }} />
          </div>
          
          {/* Threshold indicators */}
          <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.7em', opacity: 0.7 }}>
            <span>Basic({INTELLIGENCE_THRESHOLDS.BASIC_SEARCH})</span>
            <span>Advanced({INTELLIGENCE_THRESHOLDS.ADVANCED_SEARCH})</span>
            <span>Synthesis({INTELLIGENCE_THRESHOLDS.SYNTHESIS})</span>
            <span>Innovation({INTELLIGENCE_THRESHOLDS.INNOVATION})</span>
            <span>Transcendent({INTELLIGENCE_THRESHOLDS.TRANSCENDENT})</span>
            <span>Singularity({INTELLIGENCE_THRESHOLDS.SINGULARITY})</span>
          </div>
          
          {agentIntelligence.canAccess && (
            <div style={{ marginTop: '8px', fontSize: '0.85em', color: '#4caf50' }}>
              🌐 Internet Access ENABLED - Agent can learn from web searches!
            </div>
          )}
          {!agentIntelligence.canAccess && (
            <div style={{ marginTop: '8px', fontSize: '0.85em', color: '#f44336' }}>
              🔒 Intelligence too low for internet access. Need {INTELLIGENCE_THRESHOLDS.BASIC_SEARCH} (currently {agentIntelligence.intelligence})
            </div>
          )}
        </div>
      )}

      {/* Internet Learning Section */}
      {agentIntelligence.canAccess && availableInternetKnowledge.length > 0 && (
        <div style={{
          marginBottom: '12px',
          padding: '10px',
          background: '#0d1a2a',
          borderRadius: '6px',
          border: '1px solid #2196f3'
        }}>
          <div style={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center',
            marginBottom: '10px'
          }}>
            <span style={{ fontWeight: 'bold', color: '#4fc3f7' }}>
              🌐 Internet Learning ({availableInternetKnowledge.length} available)
            </span>
            <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
              <label style={{ display: 'flex', alignItems: 'center', gap: '4px', fontSize: '0.8em' }}>
                <input
                  type="checkbox"
                  checked={autoEvolveEnabled}
                  onChange={(e) => setAutoEvolveEnabled(e.target.checked)}
                />
                Auto-Evolve
              </label>
              <button
                onClick={handleAutoEvolve}
                style={{
                  padding: '4px 8px',
                  background: '#ff9800',
                  border: 'none',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '0.8em'
                }}
              >
                ⚡ Evolve Now
              </button>
            </div>
          </div>
          
          <div style={{ fontSize: '0.8em', opacity: 0.7, marginBottom: '8px' }}>
            Click to learn from the internet (opens Google search):
          </div>
          
          <div style={{ 
            display: 'flex', 
            flexWrap: 'wrap', 
            gap: '6px',
            maxHeight: '120px',
            overflowY: 'auto'
          }}>
            {availableInternetKnowledge.map((knowledge) => (
              <button
                key={knowledge.id}
                onClick={() => handleLearnFromInternet(knowledge)}
                style={{
                  padding: '4px 8px',
                  background: knowledge.complexity >= 9 ? '#9c27b0' : 
                             knowledge.complexity >= 7 ? '#ff5722' : 
                             knowledge.complexity >= 5 ? '#ff9800' : '#4caf50',
                  border: 'none',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '0.8em'
                }}
                title={`Complexity: ${knowledge.complexity} | Search: ${knowledge.searchTerms.join(', ')}`}
              >
                {knowledge.name} ({knowledge.complexity})
              </button>
            ))}
          </div>
          
          {/* Learned Knowledge */}
          {internetKnowledgeLearned.size > 0 && (
            <div style={{ marginTop: '10px', fontSize: '0.8em' }}>
              <span style={{ opacity: 0.7 }}>✅ Learned: </span>
              <span style={{ color: '#4caf50' }}>
                {Array.from(internetKnowledgeLearned).length} concepts
              </span>
            </div>
          )}
        </div>
      )}

      {/* Theories Generated */}
      {theories.length > 0 && (
        <div style={{
          marginBottom: '12px',
          padding: '10px',
          background: '#1a0a2a',
          borderRadius: '6px',
          border: '1px solid #9c27b0'
        }}>
          <div style={{ fontWeight: 'bold', color: '#ce93d8', marginBottom: '8px' }}>
            💡 Generated Theories ({theories.length})
          </div>
          <div style={{ maxHeight: '100px', overflowY: 'auto' }}>
            {theories.slice(0, 3).map((theory, idx) => (
              <div key={idx} style={{ 
                marginBottom: '6px', 
                padding: '6px', 
                background: '#2a1a3a',
                borderRadius: '4px',
                fontSize: '0.8em'
              }}>
                <strong>{theory.theoryName}</strong>
                <span style={{ marginLeft: '8px', opacity: 0.7 }}>
                  Impact: {theory.potentialImpact.toFixed(1)}
                </span>
                <button
                  onClick={() => window.open(theory.searchUrl, '_blank')}
                  style={{
                    marginLeft: '8px',
                    padding: '2px 6px',
                    background: '#9c27b0',
                    border: 'none',
                    borderRadius: '3px',
                    color: 'white',
                    cursor: 'pointer',
                    fontSize: '0.85em'
                  }}
                >
                  🔍 Research
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Learning Results Log */}
      {learningResults.length > 0 && (
        <div style={{
          marginBottom: '12px',
          padding: '10px',
          background: '#0a1a1a',
          borderRadius: '6px',
          border: '1px solid #00bcd4',
          maxHeight: '150px',
          overflowY: 'auto'
        }}>
          <div style={{ fontWeight: 'bold', color: '#4dd0e1', marginBottom: '8px' }}>
            📚 Learning Log ({learningResults.length})
          </div>
          {learningResults.slice(0, 5).map((result, idx) => (
            <div key={idx} style={{
              marginBottom: '6px',
              padding: '6px',
              background: result.success ? '#1a2a2a' : '#2a1a1a',
              borderRadius: '4px',
              fontSize: '0.75em',
              borderLeft: `2px solid ${result.success ? '#4caf50' : '#f44336'}`
            }}>
              <div>{result.insightGained}</div>
              {result.success && Object.keys(result.bonusesApplied).length > 0 && (
                <div style={{ marginTop: '4px', opacity: 0.7 }}>
                  Bonuses: {Object.entries(result.bonusesApplied).map(([k, v]) => `${k}: +${((v as number - 1) * 100).toFixed(0)}%`).join(', ')}
                </div>
              )}
              <button
                onClick={() => window.open(result.searchUrl, '_blank')}
                style={{
                  marginTop: '4px',
                  padding: '2px 6px',
                  background: '#4285f4',
                  border: 'none',
                  borderRadius: '3px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '0.9em'
                }}
              >
                🔍 Open Search
              </button>
            </div>
          ))}
        </div>
      )}{/* Subject selector */}
      <div style={{ 
        display: 'flex', 
        gap: '8px', 
        marginBottom: '10px',
        flexWrap: 'wrap'
      }}>
        <button
          onClick={() => setSelectedSubject('inventions')}
          style={{
            padding: '6px 12px',
            background: selectedSubject === 'inventions' ? '#4caf50' : '#333',
            border: 'none',
            borderRadius: '4px',
            color: 'white',
            cursor: 'pointer'
          }}
        >
          🔧 Inventions ({focusAgent?.inventions.length || 0})
        </button>
        <button
          onClick={() => setSelectedSubject('physics')}
          style={{
            padding: '6px 12px',
            background: selectedSubject === 'physics' ? '#2196f3' : '#333',
            border: 'none',
            borderRadius: '4px',
            color: 'white',
            cursor: 'pointer'
          }}
        >
          ⚛️ Physics ({unlockedPhysics.length})
        </button>
        <button
          onClick={() => setSelectedSubject('math')}
          style={{
            padding: '6px 12px',
            background: selectedSubject === 'math' ? '#ff9800' : '#333',
            border: 'none',
            borderRadius: '4px',
            color: 'white',
            cursor: 'pointer'
          }}
        >
          📐 Math ({unlockedMath.length})
        </button>
      </div>

      {/* Web Search Controls */}
      <div style={{ 
        display: 'flex', 
        gap: '10px', 
        marginBottom: '10px',
        alignItems: 'center',
        padding: '8px',
        background: '#1a2040',
        borderRadius: '4px'
      }}>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
          <input
            type="checkbox"
            checked={showWebSearch}
            onChange={(e) => setShowWebSearch(e.target.checked)}
            style={{ cursor: 'pointer' }}
          />
          <span style={{ fontSize: '0.85em' }}>🌐 Show Web Search</span>
        </label>
        <span style={{ opacity: 0.5 }}>|</span>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
          <input
            type="radio"
            name="searchMode"
            checked={searchMode === 'basic'}
            onChange={() => setSearchMode('basic')}
            style={{ cursor: 'pointer' }}
          />
          <span style={{ fontSize: '0.85em' }}>Basic</span>
        </label>
        <label style={{ display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer' }}>
          <input
            type="radio"
            name="searchMode"
            checked={searchMode === 'enhanced'}
            onChange={() => setSearchMode('enhanced')}
            style={{ cursor: 'pointer' }}
          />
          <span style={{ fontSize: '0.85em' }}>🔬 Enhanced (with scientific terms)</span>
        </label>
        <span style={{ 
          marginLeft: 'auto', 
          fontSize: '0.75em', 
          opacity: 0.6 
        }}>
          📝 {explanations.length} explanations stored
        </span>
      </div>

      {/* Items to explain */}
      {availableItems.length > 0 && (
        <div style={{ marginBottom: '12px' }}>
          <div style={{ fontSize: '0.8em', opacity: 0.8, marginBottom: '6px' }}>
            Click to generate explanation:
          </div>
          <div style={{ 
            display: 'flex', 
            flexWrap: 'wrap', 
            gap: '6px',
            maxHeight: '80px',
            overflowY: 'auto'
          }}>
            {availableItems.map((item: any) => (
              <button
                key={item.id}
                onClick={() => generateExplanation(
                  selectedSubject === 'inventions' ? 'invention' : 
                  selectedSubject === 'physics' ? 'physics' : 'math',
                  item
                )}
                style={{
                  padding: '4px 8px',
                  background: '#2a3050',
                  border: '1px solid #444',
                  borderRadius: '4px',
                  color: 'white',
                  cursor: 'pointer',
                  fontSize: '0.8em'
                }}
                title={item.description || item.effect?.type || ''}
              >
                {item.name}
              </button>
            ))}
          </div>
        </div>
      )}      {/* Generated explanations */}
      {explanations.length > 0 && (
        <div>
          <div style={{ 
            fontSize: '0.85em', 
            fontWeight: 'bold', 
            marginBottom: '8px',
            borderBottom: '1px solid #333',
            paddingBottom: '4px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center'
          }}>
            <span>📝 Generated Explanations ({explanations.length})</span>
            <button
              onClick={() => {
                setExplanations([]);
                setWebSearchResults(new Map());
              }}
              style={{
                padding: '2px 8px',
                background: '#8b0000',
                border: 'none',
                borderRadius: '3px',
                color: 'white',
                cursor: 'pointer',
                fontSize: '0.85em'
              }}
            >
              Clear All
            </button>
          </div>
          <div style={{ 
            maxHeight: '400px', 
            overflowY: 'auto',
            display: 'flex',
            flexDirection: 'column',
            gap: '10px'
          }}>
            {explanations.map((exp, idx) => (
              <div
                key={exp.id + idx}
                style={{
                  padding: '10px',
                  background: '#1a2040',
                  borderRadius: '6px',
                  borderLeft: `3px solid ${SOPHISTICATION_COLORS[exp.sophisticationLevel] || '#666'}`,
                  fontSize: '0.85em'
                }}
              >
                {/* Header */}
                <div style={{ 
                  display: 'flex', 
                  justifyContent: 'space-between',
                  marginBottom: '6px',
                  opacity: 0.9
                }}>
                  <span>
                    <strong>{exp.subject.name}</strong>
                    <span style={{ 
                      marginLeft: '8px',
                      fontSize: '0.85em',
                      opacity: 0.7 
                    }}>
                      ({exp.subject.type})
                    </span>
                  </span>
                  <span>
                    {'⭐'.repeat(Math.min(5, Math.ceil(exp.sophisticationLevel / 2)))}
                  </span>
                </div>

                {/* Content */}
                <div style={{ 
                  lineHeight: 1.5,
                  marginBottom: '8px',
                  color: '#e0e0e0'
                }}>
                  {exp.content}
                </div>                {/* Stats */}
                <div style={{ 
                  display: 'flex',
                  gap: '12px',
                  fontSize: '0.8em',
                  opacity: 0.7
                }}>
                  <span>📊 Level: {exp.sophisticationLevel}/10</span>
                  <span>🎯 Accuracy: {Math.round(exp.accuracy * 100)}%</span>
                  <span>📐 Math: {exp.mathConceptsUsed.length}</span>
                  <span>⚛️ Physics: {exp.physicsConceptsUsed.length}</span>
                </div>

                {/* Web Search Results */}
                {showWebSearch && webSearchResults.has(exp.id) && (
                  <div style={{
                    marginTop: '10px',
                    padding: '8px',
                    background: '#0d1520',
                    borderRadius: '4px',
                    fontSize: '0.8em'
                  }}>
                    <div style={{ 
                      fontWeight: 'bold', 
                      marginBottom: '6px',
                      color: '#4fc3f7'
                    }}>
                      🌐 Web Search Resources
                    </div>
                    
                    {/* Suggested Terms */}
                    <div style={{ marginBottom: '6px' }}>
                      <span style={{ opacity: 0.7 }}>Related terms: </span>
                      {webSearchResults.get(exp.id)!.suggestedTerms.slice(0, 5).map((term, i) => (
                        <button
                          key={i}
                          onClick={() => openGoogleSearch(term)}
                          style={{
                            marginRight: '4px',
                            marginBottom: '4px',
                            padding: '2px 6px',
                            background: '#2a3a5a',
                            border: '1px solid #3a4a6a',
                            borderRadius: '3px',
                            color: '#8ab4f8',
                            cursor: 'pointer',
                            fontSize: '0.9em'
                          }}
                          title={`Search Google for "${term}"`}
                        >
                          {term}
                        </button>
                      ))}
                    </div>

                    {/* Search Buttons */}
                    <div style={{ 
                      display: 'flex', 
                      gap: '8px',
                      marginTop: '8px'
                    }}>
                      <button
                        onClick={() => {
                          const ws = webSearchResults.get(exp.id)!;
                          window.open(ws.searchUrl, '_blank', 'noopener,noreferrer');
                        }}
                        style={{
                          padding: '4px 10px',
                          background: '#4285f4',
                          border: 'none',
                          borderRadius: '4px',
                          color: 'white',
                          cursor: 'pointer',
                          display: 'flex',
                          alignItems: 'center',
                          gap: '4px',
                          fontSize: '0.85em'
                        }}
                      >
                        🔍 Search Google
                      </button>
                      {webSearchResults.get(exp.id)!.wikipediaUrl && (
                        <button
                          onClick={() => {
                            const ws = webSearchResults.get(exp.id)!;
                            window.open(ws.wikipediaUrl, '_blank', 'noopener,noreferrer');
                          }}
                          style={{
                            padding: '4px 10px',
                            background: '#666',
                            border: 'none',
                            borderRadius: '4px',
                            color: 'white',
                            cursor: 'pointer',
                            display: 'flex',
                            alignItems: 'center',
                            gap: '4px',
                            fontSize: '0.85em'
                          }}
                        >
                          📖 Wikipedia
                        </button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Empty state */}
      {explanations.length === 0 && availableItems.length > 0 && (
        <div style={{ 
          textAlign: 'center', 
          padding: '20px',
          opacity: 0.6,
          fontSize: '0.85em'
        }}>
          Click on an item above to have the agent explain it using their scientific knowledge.
        </div>
      )}

      {availableItems.length === 0 && (
        <div style={{ 
          textAlign: 'center', 
          padding: '20px',
          opacity: 0.6,
          fontSize: '0.85em'
        }}>
          No {selectedSubject} available yet. Wait for discoveries!
        </div>
      )}
    </div>
  );
};

export default ExplanationPanel;
