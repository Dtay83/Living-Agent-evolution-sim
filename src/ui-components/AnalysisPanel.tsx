/**
 * ANALYSIS PANEL COMPONENT
 * 
 * Upload exported JSON files and get AI-powered recommendations
 * for achieving agent sentience.
 */

import React, { useState, useRef } from 'react';
import type { AnalysisResult, Pattern, Recommendation, ConfigChange } from '../analysis-system';
import { analyzeExport } from '../analysis-system';

interface AnalysisPanelProps {
  isOpen: boolean;
  onClose: () => void;
}

export const AnalysisPanel: React.FC<AnalysisPanelProps> = ({ isOpen, onClose }) => {
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [fileName, setFileName] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setError(null);
    setFileName(file.name);

    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const result = analyzeExport(data);
      setAnalysisResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze file');
      setAnalysisResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    const file = e.dataTransfer.files?.[0];
    if (!file || !file.name.endsWith('.json')) {
      setError('Please drop a JSON file');
      return;
    }

    setIsLoading(true);
    setError(null);
    setFileName(file.name);

    try {
      const text = await file.text();
      const data = JSON.parse(text);
      const result = analyzeExport(data);
      setAnalysisResult(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to analyze file');
      setAnalysisResult(null);
    } finally {
      setIsLoading(false);
    }
  };

  const clearAnalysis = () => {
    setAnalysisResult(null);
    setFileName(null);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  if (!isOpen) return null;

  return (
    <div style={styles.overlay}>
      <div style={styles.panel}>
        {/* Header */}
        <div style={styles.header}>
          <h2 style={styles.title}>🔬 Sentience Analysis</h2>
          <button onClick={onClose} style={styles.closeButton}>✕</button>
        </div>

        {/* Upload Area */}
        {!analysisResult && (
          <div
            style={styles.dropZone}
            onDragOver={handleDragOver}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".json"
              onChange={handleFileUpload}
              style={{ display: 'none' }}
            />
            {isLoading ? (
              <div style={styles.loading}>
                <span style={styles.spinner}>⏳</span>
                <p>Analyzing data...</p>
              </div>
            ) : (
              <>
                <span style={styles.uploadIcon}>📤</span>
                <p style={styles.uploadText}>Drop exported JSON here</p>
                <p style={styles.uploadSubtext}>or click to browse</p>
                <p style={styles.supportedFormats}>
                  Supports: complete-export, evolution-data, conversations, invention-history
                </p>
              </>
            )}
          </div>
        )}

        {error && (
          <div style={styles.error}>
            ⚠️ {error}
          </div>
        )}

        {/* Analysis Results */}
        {analysisResult && (
          <div style={styles.results}>
            {/* File Info */}
            <div style={styles.fileInfo}>
              <span>📄 {fileName}</span>
              <button onClick={clearAnalysis} style={styles.clearButton}>
                Analyze Different File
              </button>
            </div>

            {/* Sentience Progress */}
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>🧠 Sentience Progress</h3>
              <div style={styles.progressContainer}>
                <div style={styles.overallScore}>
                  <div style={styles.scoreCircle}>
                    <span style={styles.scoreNumber}>
                      {analysisResult.sentinenceProgress.overallScore.toFixed(0)}
                    </span>
                    <span style={styles.scoreLabel}>/ 100</span>
                  </div>
                  <p style={styles.milestone}>
                    Next: {analysisResult.sentinenceProgress.nextMilestone}
                  </p>
                </div>
                
                <div style={styles.pillars}>
                  {Object.entries(analysisResult.sentinenceProgress.pillars).map(([name, value]) => (
                    <div key={name} style={styles.pillar}>
                      <span style={styles.pillarName}>{formatPillarName(name)}</span>
                      <div style={styles.pillarBar}>
                        <div 
                          style={{
                            ...styles.pillarFill,
                            width: `${Math.min(100, value)}%`,
                            backgroundColor: getPillarColor(value),
                          }}
                        />
                      </div>
                      <span style={styles.pillarValue}>{value.toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Patterns Detected */}
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>🔍 Patterns Detected</h3>
              <div style={styles.patternList}>
                {analysisResult.patterns.length === 0 ? (
                  <p style={styles.noData}>No significant patterns detected</p>
                ) : (
                  analysisResult.patterns.map((pattern, i) => (
                    <PatternCard key={i} pattern={pattern} />
                  ))
                )}
              </div>
            </div>

            {/* Recommendations */}
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>💡 Recommendations</h3>
              <div style={styles.recommendationList}>
                {analysisResult.recommendations.length === 0 ? (
                  <p style={styles.noData}>No recommendations - simulation looks healthy!</p>
                ) : (
                  analysisResult.recommendations.map((rec, i) => (
                    <RecommendationCard key={i} recommendation={rec} />
                  ))
                )}
              </div>
            </div>

            {/* Metrics Summary */}
            <div style={styles.section}>
              <h3 style={styles.sectionTitle}>📊 Key Metrics</h3>
              <div style={styles.metricsGrid}>
                <MetricBox 
                  label="Survival Rate" 
                  value={`${(analysisResult.metrics.survivalRate * 100).toFixed(1)}%`}
                />
                <MetricBox 
                  label="Population Stability" 
                  value={`${(analysisResult.metrics.populationStability * 100).toFixed(1)}%`}
                />
                <MetricBox 
                  label="Genetic Diversity" 
                  value={`${(analysisResult.metrics.geneticDiversity * 100).toFixed(1)}%`}
                />
                <MetricBox 
                  label="Invention Rate" 
                  value={`${analysisResult.metrics.inventionRate.toFixed(2)}/100 ticks`}
                />
                <MetricBox 
                  label="Communication" 
                  value={`${(analysisResult.metrics.communicationFrequency * 100).toFixed(2)}/tick`}
                />
                <MetricBox 
                  label="Consciousness" 
                  value={`${analysisResult.metrics.consciousnessScore.toFixed(1)}`}
                />
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// Sub-components
const PatternCard: React.FC<{ pattern: Pattern }> = ({ pattern }) => {
  const icon = pattern.type === 'positive' ? '✅' : pattern.type === 'negative' ? '⚠️' : 'ℹ️';
  const bgColor = pattern.type === 'positive' ? '#1a3a1a' : pattern.type === 'negative' ? '#3a1a1a' : '#1a2a3a';
  
  return (
    <div style={{ ...styles.patternCard, backgroundColor: bgColor }}>
      <div style={styles.patternHeader}>
        <span style={styles.patternIcon}>{icon}</span>
        <span style={styles.patternCategory}>{pattern.category}</span>
        <span style={{ ...styles.significance, color: getSignificanceColor(pattern.significance) }}>
          {pattern.significance}
        </span>
      </div>
      <p style={styles.patternDescription}>{pattern.description}</p>
      <p style={styles.patternEvidence}>📋 {pattern.evidence}</p>
    </div>
  );
};

const RecommendationCard: React.FC<{ recommendation: Recommendation }> = ({ recommendation }) => {
  const [expanded, setExpanded] = useState(false);
  
  return (
    <div style={styles.recCard}>
      <div style={styles.recHeader} onClick={() => setExpanded(!expanded)}>
        <span style={styles.priority}>P{recommendation.priority}</span>
        <span style={styles.recCategory}>{recommendation.category}</span>
        <span style={styles.recTitle}>{recommendation.title}</span>
        <span style={styles.expandIcon}>{expanded ? '▼' : '▶'}</span>
      </div>
      
      {expanded && (
        <div style={styles.recBody}>
          <p style={styles.recDescription}>{recommendation.description}</p>
          <p style={styles.recImpact}>💫 Expected: {recommendation.expectedImpact}</p>
          
          {recommendation.configChanges && recommendation.configChanges.length > 0 && (
            <div style={styles.configChanges}>
              <p style={styles.configTitle}>Suggested CONFIG changes:</p>
              {recommendation.configChanges.map((change, i) => (
                <div key={i} style={styles.configChange}>
                  <code style={styles.configParam}>{change.parameter}</code>
                  <span style={styles.configArrow}>→</span>
                  <code style={styles.configValue}>{String(change.suggestedValue)}</code>
                  <span style={styles.configReason}>({change.reason})</span>
                </div>
              ))}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

const MetricBox: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={styles.metricBox}>
    <span style={styles.metricValue}>{value}</span>
    <span style={styles.metricLabel}>{label}</span>
  </div>
);

// Helper functions
function formatPillarName(name: string): string {
  return name.replace(/([A-Z])/g, ' $1').replace(/^./, s => s.toUpperCase());
}

function getPillarColor(value: number): string {
  if (value >= 70) return '#4ade80';
  if (value >= 40) return '#fbbf24';
  return '#f87171';
}

function getSignificanceColor(sig: string): string {
  switch (sig) {
    case 'critical': return '#ef4444';
    case 'high': return '#f97316';
    case 'medium': return '#eab308';
    default: return '#a3a3a3';
  }
}

// Styles
const styles: Record<string, React.CSSProperties> = {
  overlay: {
    position: 'fixed',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.8)',
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1000,
  },
  panel: {
    backgroundColor: '#1a1a2e',
    borderRadius: '12px',
    width: '90%',
    maxWidth: '800px',
    maxHeight: '90vh',
    overflow: 'hidden',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: '0 25px 50px -12px rgba(0, 0, 0, 0.5)',
  },
  header: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '16px 20px',
    borderBottom: '1px solid #333',
    backgroundColor: '#16162a',
  },
  title: {
    margin: 0,
    color: '#fff',
    fontSize: '1.3rem',
  },
  closeButton: {
    background: 'none',
    border: 'none',
    color: '#888',
    fontSize: '1.5rem',
    cursor: 'pointer',
    padding: '4px 8px',
  },
  dropZone: {
    margin: '20px',
    padding: '40px',
    border: '2px dashed #444',
    borderRadius: '8px',
    textAlign: 'center',
    cursor: 'pointer',
    transition: 'border-color 0.2s',
  },
  uploadIcon: {
    fontSize: '3rem',
    display: 'block',
    marginBottom: '10px',
  },
  uploadText: {
    color: '#fff',
    fontSize: '1.1rem',
    margin: '5px 0',
  },
  uploadSubtext: {
    color: '#888',
    fontSize: '0.9rem',
    margin: '5px 0',
  },
  supportedFormats: {
    color: '#666',
    fontSize: '0.8rem',
    marginTop: '15px',
  },
  loading: {
    color: '#fff',
  },
  spinner: {
    fontSize: '2rem',
    display: 'block',
    animation: 'spin 1s linear infinite',
  },
  error: {
    margin: '0 20px 20px',
    padding: '12px',
    backgroundColor: '#3a1a1a',
    borderRadius: '6px',
    color: '#f87171',
  },
  results: {
    flex: 1,
    overflow: 'auto',
    padding: '0 20px 20px',
  },
  fileInfo: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '12px',
    backgroundColor: '#16162a',
    borderRadius: '6px',
    marginBottom: '15px',
    color: '#888',
  },
  clearButton: {
    background: 'none',
    border: '1px solid #444',
    color: '#888',
    padding: '6px 12px',
    borderRadius: '4px',
    cursor: 'pointer',
  },
  section: {
    marginBottom: '20px',
  },
  sectionTitle: {
    color: '#fff',
    fontSize: '1.1rem',
    marginBottom: '12px',
    borderBottom: '1px solid #333',
    paddingBottom: '8px',
  },
  progressContainer: {
    backgroundColor: '#16162a',
    borderRadius: '8px',
    padding: '16px',
  },
  overallScore: {
    textAlign: 'center',
    marginBottom: '20px',
  },
  scoreCircle: {
    display: 'inline-block',
    padding: '20px 30px',
    backgroundColor: '#1a2a3a',
    borderRadius: '50%',
    marginBottom: '10px',
  },
  scoreNumber: {
    fontSize: '2.5rem',
    fontWeight: 'bold',
    color: '#4ade80',
  },
  scoreLabel: {
    color: '#888',
    fontSize: '1rem',
  },
  milestone: {
    color: '#fbbf24',
    fontSize: '0.9rem',
    margin: 0,
  },
  pillars: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  pillar: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
  },
  pillarName: {
    width: '120px',
    color: '#aaa',
    fontSize: '0.85rem',
  },
  pillarBar: {
    flex: 1,
    height: '8px',
    backgroundColor: '#333',
    borderRadius: '4px',
    overflow: 'hidden',
  },
  pillarFill: {
    height: '100%',
    borderRadius: '4px',
    transition: 'width 0.3s',
  },
  pillarValue: {
    width: '45px',
    textAlign: 'right',
    color: '#fff',
    fontSize: '0.85rem',
  },
  patternList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '10px',
  },
  patternCard: {
    padding: '12px',
    borderRadius: '6px',
  },
  patternHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '8px',
  },
  patternIcon: {
    fontSize: '1.1rem',
  },
  patternCategory: {
    color: '#888',
    fontSize: '0.8rem',
    textTransform: 'uppercase',
    letterSpacing: '0.5px',
  },
  significance: {
    marginLeft: 'auto',
    fontSize: '0.75rem',
    textTransform: 'uppercase',
    fontWeight: 'bold',
  },
  patternDescription: {
    color: '#fff',
    margin: '0 0 6px',
    fontSize: '0.95rem',
  },
  patternEvidence: {
    color: '#888',
    margin: 0,
    fontSize: '0.85rem',
  },
  noData: {
    color: '#666',
    fontStyle: 'italic',
    textAlign: 'center',
    padding: '20px',
  },
  recommendationList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  recCard: {
    backgroundColor: '#16162a',
    borderRadius: '6px',
    overflow: 'hidden',
  },
  recHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    padding: '12px',
    cursor: 'pointer',
  },
  priority: {
    backgroundColor: '#4f46e5',
    color: '#fff',
    padding: '2px 8px',
    borderRadius: '4px',
    fontSize: '0.8rem',
    fontWeight: 'bold',
  },
  recCategory: {
    color: '#888',
    fontSize: '0.8rem',
    textTransform: 'uppercase',
  },
  recTitle: {
    flex: 1,
    color: '#fff',
    fontSize: '0.95rem',
  },
  expandIcon: {
    color: '#666',
    fontSize: '0.8rem',
  },
  recBody: {
    padding: '0 12px 12px',
    borderTop: '1px solid #333',
  },
  recDescription: {
    color: '#aaa',
    fontSize: '0.9rem',
    margin: '12px 0',
  },
  recImpact: {
    color: '#4ade80',
    fontSize: '0.85rem',
    margin: '8px 0',
  },
  configChanges: {
    backgroundColor: '#0a0a1a',
    borderRadius: '4px',
    padding: '12px',
    marginTop: '10px',
  },
  configTitle: {
    color: '#888',
    fontSize: '0.8rem',
    margin: '0 0 10px',
  },
  configChange: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '6px',
    flexWrap: 'wrap',
  },
  configParam: {
    color: '#f472b6',
    fontSize: '0.85rem',
  },
  configArrow: {
    color: '#666',
  },
  configValue: {
    color: '#4ade80',
    fontSize: '0.85rem',
    fontWeight: 'bold',
  },
  configReason: {
    color: '#666',
    fontSize: '0.8rem',
  },
  metricsGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '10px',
  },
  metricBox: {
    backgroundColor: '#16162a',
    borderRadius: '6px',
    padding: '12px',
    textAlign: 'center',
  },
  metricValue: {
    display: 'block',
    color: '#4ade80',
    fontSize: '1.2rem',
    fontWeight: 'bold',
    marginBottom: '4px',
  },
  metricLabel: {
    color: '#888',
    fontSize: '0.75rem',
    textTransform: 'uppercase',
  },
};

export default AnalysisPanel;
