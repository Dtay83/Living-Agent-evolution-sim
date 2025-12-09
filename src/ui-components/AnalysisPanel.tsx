/**
 * ANALYSIS PANEL COMPONENT
 * 
 * Upload exported JSON files and get AI-powered recommendations
 * for achieving agent sentience.
 */

import React, { useState, useRef } from 'react';
import type { 
  AnalysisResult, 
  Pattern, 
  Recommendation, 
  ConfigChange,
  TrendAnalysis,
  SentinencePrediction,
  AdvancedMetrics,
  BottleneckAnalysis,
  TrendLine
} from '../analysis-system';
import { analyzeExport, analyzeCompleteExportAdvanced } from '../analysis-system';

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
      // Use advanced analysis for complete exports
      const result = data.worldState 
        ? analyzeCompleteExportAdvanced(data) 
        : analyzeExport(data);
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
      // Use advanced analysis for complete exports
      const result = data.worldState 
        ? analyzeCompleteExportAdvanced(data) 
        : analyzeExport(data);
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

            {/* Trend Analysis */}
            {analysisResult.trendAnalysis && (
              <div style={styles.section}>
                <h3 style={styles.sectionTitle}>📈 Trend Analysis</h3>
                <TrendAnalysisPanel trends={analysisResult.trendAnalysis} />
              </div>
            )}

            {/* Sentience Predictions */}
            {analysisResult.predictions && (
              <div style={styles.section}>
                <h3 style={styles.sectionTitle}>🔮 Sentience Prediction</h3>
                <PredictionPanel predictions={analysisResult.predictions} />
              </div>
            )}

            {/* Advanced Metrics */}
            {analysisResult.advancedMetrics && (
              <div style={styles.section}>
                <h3 style={styles.sectionTitle}>🧪 Advanced Metrics</h3>
                <AdvancedMetricsPanel metrics={analysisResult.advancedMetrics} />
              </div>
            )}
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

// ============================================
// TREND ANALYSIS PANEL
// ============================================

const TrendAnalysisPanel: React.FC<{ trends: TrendAnalysis }> = ({ trends }) => {
  const getDirectionIcon = (dir: 'up' | 'down' | 'flat') => {
    if (dir === 'up') return '📈';
    if (dir === 'down') return '📉';
    return '➡️';
  };

  const getTrajectoryColor = (trajectory: string) => {
    switch (trajectory) {
      case 'improving': return '#4ade80';
      case 'declining': return '#f87171';
      case 'volatile': return '#fbbf24';
      default: return '#a3a3a3';
    }
  };

  return (
    <div style={styles.trendPanel}>
      {/* Overall Trajectory */}
      <div style={styles.trajectoryHeader}>
        <span style={styles.trajectoryLabel}>Overall Trajectory:</span>
        <span style={{ 
          ...styles.trajectoryValue, 
          color: getTrajectoryColor(trends.overallTrajectory) 
        }}>
          {trends.overallTrajectory.toUpperCase()}
        </span>
        <span style={styles.confidenceBadge}>
          {trends.confidenceScore.toFixed(0)}% confidence
        </span>
      </div>

      {/* Trend Grid */}
      <div style={styles.trendGrid}>
        <TrendItem 
          label="Population" 
          trend={trends.population} 
          icon={getDirectionIcon(trends.population.direction)}
        />
        <TrendItem 
          label="Curiosity" 
          trend={trends.curiosity} 
          icon={getDirectionIcon(trends.curiosity.direction)}
        />
        <TrendItem 
          label="Creativity" 
          trend={trends.creativity} 
          icon={getDirectionIcon(trends.creativity.direction)}
        />
        <TrendItem 
          label="Social" 
          trend={trends.social} 
          icon={getDirectionIcon(trends.social.direction)}
        />
        <TrendItem 
          label="Consciousness" 
          trend={trends.consciousnessScore} 
          icon={getDirectionIcon(trends.consciousnessScore.direction)}
        />
        <TrendItem 
          label="Invention Rate" 
          trend={trends.inventionRate} 
          icon={getDirectionIcon(trends.inventionRate.direction)}
        />
      </div>
    </div>
  );
};

const TrendItem: React.FC<{ label: string; trend: TrendLine; icon: string }> = ({ label, trend, icon }) => (
  <div style={styles.trendItem}>
    <div style={styles.trendItemHeader}>
      <span style={styles.trendIcon}>{icon}</span>
      <span style={styles.trendLabel}>{label}</span>
    </div>
    <div style={styles.trendStats}>
      <span style={styles.trendStat}>
        Slope: <code>{trend.slope.toFixed(4)}</code>
      </span>
      <span style={styles.trendStat}>
        R²: <code>{(trend.rSquared * 100).toFixed(1)}%</code>
      </span>
      <span style={styles.trendStat}>
        Volatility: <code>{(trend.volatility * 100).toFixed(0)}%</code>
      </span>
    </div>
    <div style={styles.projectedValue}>
      Projected (+100 ticks): <strong>{trend.projectedValue.toFixed(2)}</strong>
    </div>
  </div>
);

// ============================================
// PREDICTION PANEL
// ============================================

const PredictionPanel: React.FC<{ predictions: SentinencePrediction }> = ({ predictions }) => {
  const getConfidenceColor = (conf: string) => {
    switch (conf) {
      case 'high': return '#4ade80';
      case 'medium': return '#fbbf24';
      default: return '#f87171';
    }
  };

  return (
    <div style={styles.predictionPanel}>
      {/* Main Prediction */}
      <div style={styles.mainPrediction}>
        <div style={styles.predictionHeader}>
          <span style={styles.predictionIcon}>🎯</span>
          <span style={styles.predictionTitle}>Estimated Time to Sentience</span>
        </div>
        <div style={styles.predictionValue}>
          {predictions.estimatedTicksToSentience !== null ? (
            <>
              <span style={styles.tickCount}>{predictions.estimatedTicksToSentience}</span>
              <span style={styles.tickLabel}>ticks</span>
            </>
          ) : (
            <span style={styles.unknownPrediction}>Cannot determine</span>
          )}
        </div>
        <div style={styles.probabilityRow}>
          <span>Success Probability:</span>
          <span style={{ 
            color: predictions.probabilityOfSuccess > 60 ? '#4ade80' : 
                   predictions.probabilityOfSuccess > 30 ? '#fbbf24' : '#f87171',
            fontWeight: 'bold'
          }}>
            {predictions.probabilityOfSuccess.toFixed(1)}%
          </span>
          <span style={{ 
            ...styles.confidenceTag, 
            backgroundColor: getConfidenceColor(predictions.confidence) 
          }}>
            {predictions.confidence} confidence
          </span>
        </div>
      </div>

      {/* Bottlenecks */}
      {predictions.bottlenecks.length > 0 && (
        <div style={styles.bottleneckSection}>
          <h4 style={styles.subSectionTitle}>⚠️ Bottlenecks Blocking Sentience</h4>
          <div style={styles.bottleneckList}>
            {predictions.bottlenecks.map((bottleneck, i) => (
              <BottleneckItem key={i} bottleneck={bottleneck} />
            ))}
          </div>
        </div>
      )}

      {/* Scenario Analysis */}
      <div style={styles.scenarioSection}>
        <h4 style={styles.subSectionTitle}>📊 Scenario Analysis</h4>
        <div style={styles.scenarioGrid}>
          <ScenarioCard 
            title="Best Case" 
            icon="🌟" 
            scenario={predictions.scenarioAnalysis.bestCase}
            color="#4ade80"
          />
          <ScenarioCard 
            title="Likely Case" 
            icon="📈" 
            scenario={predictions.scenarioAnalysis.likelyCase}
            color="#fbbf24"
          />
          <ScenarioCard 
            title="Worst Case" 
            icon="⚡" 
            scenario={predictions.scenarioAnalysis.worstCase}
            color="#f87171"
          />
        </div>
      </div>

      {/* Optimal Path */}
      {predictions.optimalPath.length > 0 && (
        <div style={styles.optimalPathSection}>
          <h4 style={styles.subSectionTitle}>🛤️ Optimal Path to Sentience</h4>
          <div style={styles.pathSteps}>
            {predictions.optimalPath.map((step, i) => (
              <div key={i} style={styles.pathStep}>
                <span style={styles.stepNumber}>{i + 1}</span>
                <div style={styles.stepContent}>
                  <span style={styles.stepAction}>{step.action}</span>
                  <span style={styles.stepOutcome}>{step.expectedOutcome}</span>
                  <span style={styles.stepTick}>Target: Tick {step.tick}</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const BottleneckItem: React.FC<{ bottleneck: BottleneckAnalysis }> = ({ bottleneck }) => (
  <div style={{
    ...styles.bottleneckItem,
    borderLeft: `3px solid ${bottleneck.blockingSentience ? '#f87171' : '#fbbf24'}`
  }}>
    <div style={styles.bottleneckHeader}>
      <span style={styles.bottleneckName}>{bottleneck.pillar}</span>
      {bottleneck.blockingSentience && (
        <span style={styles.blockingTag}>BLOCKING</span>
      )}
    </div>
    <div style={styles.bottleneckProgress}>
      <div style={styles.progressBarBg}>
        <div style={{
          ...styles.progressBarFill,
          width: `${(bottleneck.currentValue / bottleneck.requiredValue) * 100}%`,
        }} />
      </div>
      <span style={styles.progressText}>
        {bottleneck.currentValue.toFixed(1)}% / {bottleneck.requiredValue}% (gap: {bottleneck.gap.toFixed(1)})
      </span>
    </div>
    {bottleneck.estimatedTicksToResolve && (
      <span style={styles.resolveTime}>
        Est. {bottleneck.estimatedTicksToResolve} ticks to resolve
      </span>
    )}
  </div>
);

const ScenarioCard: React.FC<{ 
  title: string; 
  icon: string; 
  scenario: { ticksToSentience: number | null; finalScore: number; description: string };
  color: string;
}> = ({ title, icon, scenario, color }) => (
  <div style={{ ...styles.scenarioCard, borderTop: `3px solid ${color}` }}>
    <div style={styles.scenarioHeader}>
      <span>{icon}</span>
      <span style={{ color }}>{title}</span>
    </div>
    <div style={styles.scenarioTicks}>
      {scenario.ticksToSentience !== null ? (
        <>{scenario.ticksToSentience} ticks</>
      ) : (
        <span style={{ color: '#888' }}>N/A</span>
      )}
    </div>
    <div style={styles.scenarioScore}>
      Score: {scenario.finalScore.toFixed(1)}
    </div>
    <p style={styles.scenarioDesc}>{scenario.description}</p>
  </div>
);

// ============================================
// ADVANCED METRICS PANEL
// ============================================

const AdvancedMetricsPanel: React.FC<{ metrics: AdvancedMetrics }> = ({ metrics }) => {
  return (
    <div style={styles.advancedPanel}>
      {/* System Health */}
      <div style={styles.advancedSection}>
        <h4 style={styles.advancedTitle}>🏥 System Health</h4>
        <div style={styles.advancedGrid}>
          <AdvancedMetricItem label="Carrying Capacity" value={metrics.carryingCapacity.toFixed(0)} />
          <AdvancedMetricItem label="System Entropy" value={(metrics.systemEntropy * 100).toFixed(1) + '%'} />
          <AdvancedMetricItem label="Resource Efficiency" value={(metrics.resourceEfficiency * 100).toFixed(1) + '%'} />
          <AdvancedMetricItem label="Population Variance" value={metrics.populationVariance.toFixed(2)} />
        </div>
      </div>

      {/* Evolution Metrics */}
      <div style={styles.advancedSection}>
        <h4 style={styles.advancedTitle}>🧬 Evolution Dynamics</h4>
        <div style={styles.advancedGrid}>
          <AdvancedMetricItem label="Evolutionary Pressure" value={(metrics.evolutionaryPressure * 100).toFixed(1) + '%'} />
          <AdvancedMetricItem label="Adaptation Rate" value={(metrics.adaptationRate * 100).toFixed(1) + '%'} />
        </div>
      </div>

      {/* Consciousness Metrics */}
      <div style={styles.advancedSection}>
        <h4 style={styles.advancedTitle}>🧠 Consciousness Indicators</h4>
        <div style={styles.advancedGrid}>
          <AdvancedMetricItem label="Emergence Rate" value={(metrics.consciousnessEmergenceRate * 100).toFixed(1) + '%'} />
          <AdvancedMetricItem label="Collective Intelligence" value={(metrics.collectiveIntelligence * 100).toFixed(1) + '%'} />
          <AdvancedMetricItem label="Knowledge Transfer" value={(metrics.knowledgeTransferEfficiency * 100).toFixed(1) + '%'} />
        </div>
      </div>

      {/* Gene Correlations */}
      {metrics.geneCorrelations.length > 0 && (
        <div style={styles.advancedSection}>
          <h4 style={styles.advancedTitle}>🔗 Gene Correlations</h4>
          <div style={styles.correlationList}>
            {metrics.geneCorrelations.slice(0, 5).map((corr, i) => (
              <div key={i} style={styles.correlationItem}>
                <span style={styles.correlationGenes}>
                  {corr.gene1} ↔ {corr.gene2}
                </span>
                <span style={{
                  ...styles.correlationValue,
                  color: corr.correlation > 0 ? '#4ade80' : '#f87171'
                }}>
                  {corr.correlation > 0 ? '+' : ''}{corr.correlation.toFixed(3)}
                </span>
                <span style={styles.correlationSig}>{corr.significance}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Anomalies */}
      {metrics.anomalies.length > 0 && (
        <div style={styles.advancedSection}>
          <h4 style={styles.advancedTitle}>⚡ Detected Anomalies</h4>
          <div style={styles.anomalyList}>
            {metrics.anomalies.slice(0, 5).map((anomaly, i) => (
              <div key={i} style={styles.anomalyItem}>
                <span style={styles.anomalyTick}>Tick {anomaly.tick}</span>
                <span style={styles.anomalyType}>{anomaly.type.replace('_', ' ')}</span>
                <span style={styles.anomalyDesc}>{anomaly.description}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

const AdvancedMetricItem: React.FC<{ label: string; value: string }> = ({ label, value }) => (
  <div style={styles.advancedMetricItem}>
    <span style={styles.advancedMetricValue}>{value}</span>
    <span style={styles.advancedMetricLabel}>{label}</span>
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

  // Trend Analysis Styles
  trendPanel: {
    backgroundColor: '#16162a',
    borderRadius: '8px',
    padding: '16px',
  },
  trajectoryHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '16px',
    flexWrap: 'wrap',
  },
  trajectoryLabel: {
    color: '#888',
    fontSize: '0.9rem',
  },
  trajectoryValue: {
    fontWeight: 'bold',
    fontSize: '1.1rem',
  },
  confidenceBadge: {
    backgroundColor: '#333',
    color: '#aaa',
    padding: '4px 10px',
    borderRadius: '12px',
    fontSize: '0.8rem',
  },
  trendGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '12px',
  },
  trendItem: {
    backgroundColor: '#1a2a3a',
    borderRadius: '6px',
    padding: '12px',
  },
  trendItemHeader: {
    display: 'flex',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '8px',
  },
  trendIcon: {
    fontSize: '1.2rem',
  },
  trendLabel: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: '0.9rem',
  },
  trendStats: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    marginBottom: '6px',
  },
  trendStat: {
    color: '#888',
    fontSize: '0.75rem',
  },
  projectedValue: {
    color: '#4ade80',
    fontSize: '0.8rem',
    marginTop: '4px',
  },

  // Prediction Panel Styles
  predictionPanel: {
    backgroundColor: '#16162a',
    borderRadius: '8px',
    padding: '16px',
  },
  mainPrediction: {
    textAlign: 'center',
    marginBottom: '20px',
    padding: '20px',
    backgroundColor: '#1a2a3a',
    borderRadius: '8px',
  },
  predictionHeader: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '8px',
    marginBottom: '12px',
  },
  predictionIcon: {
    fontSize: '1.5rem',
  },
  predictionTitle: {
    color: '#fff',
    fontSize: '1rem',
  },
  predictionValue: {
    marginBottom: '12px',
  },
  tickCount: {
    fontSize: '3rem',
    fontWeight: 'bold',
    color: '#4ade80',
  },
  tickLabel: {
    color: '#888',
    fontSize: '1rem',
    marginLeft: '8px',
  },
  unknownPrediction: {
    color: '#f87171',
    fontSize: '1.2rem',
  },
  probabilityRow: {
    display: 'flex',
    justifyContent: 'center',
    alignItems: 'center',
    gap: '12px',
    color: '#aaa',
    fontSize: '0.9rem',
  },
  confidenceTag: {
    padding: '2px 8px',
    borderRadius: '4px',
    fontSize: '0.75rem',
    color: '#000',
    fontWeight: 'bold',
  },
  bottleneckSection: {
    marginBottom: '16px',
  },
  subSectionTitle: {
    color: '#fff',
    fontSize: '0.95rem',
    margin: '0 0 12px',
  },
  bottleneckList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  bottleneckItem: {
    backgroundColor: '#1a1a2e',
    borderRadius: '4px',
    padding: '10px',
  },
  bottleneckHeader: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: '6px',
  },
  bottleneckName: {
    color: '#fff',
    fontWeight: 'bold',
    fontSize: '0.9rem',
  },
  blockingTag: {
    backgroundColor: '#f87171',
    color: '#000',
    padding: '2px 6px',
    borderRadius: '3px',
    fontSize: '0.7rem',
    fontWeight: 'bold',
  },
  bottleneckProgress: {
    marginBottom: '4px',
  },
  progressBarBg: {
    height: '6px',
    backgroundColor: '#333',
    borderRadius: '3px',
    overflow: 'hidden',
    marginBottom: '4px',
  },
  progressBarFill: {
    height: '100%',
    backgroundColor: '#fbbf24',
    borderRadius: '3px',
  },
  progressText: {
    color: '#888',
    fontSize: '0.75rem',
  },
  resolveTime: {
    color: '#4ade80',
    fontSize: '0.75rem',
  },
  scenarioSection: {
    marginBottom: '16px',
  },
  scenarioGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(3, 1fr)',
    gap: '10px',
  },
  scenarioCard: {
    backgroundColor: '#1a1a2e',
    borderRadius: '6px',
    padding: '12px',
    textAlign: 'center',
  },
  scenarioHeader: {
    display: 'flex',
    justifyContent: 'center',
    gap: '6px',
    marginBottom: '8px',
    fontSize: '0.85rem',
    fontWeight: 'bold',
  },
  scenarioTicks: {
    fontSize: '1.2rem',
    fontWeight: 'bold',
    color: '#fff',
    marginBottom: '4px',
  },
  scenarioScore: {
    color: '#888',
    fontSize: '0.8rem',
    marginBottom: '8px',
  },
  scenarioDesc: {
    color: '#666',
    fontSize: '0.7rem',
    margin: 0,
    lineHeight: 1.3,
  },
  optimalPathSection: {},
  pathSteps: {
    display: 'flex',
    flexDirection: 'column',
    gap: '8px',
  },
  pathStep: {
    display: 'flex',
    gap: '12px',
    backgroundColor: '#1a2a3a',
    borderRadius: '6px',
    padding: '10px',
  },
  stepNumber: {
    width: '24px',
    height: '24px',
    backgroundColor: '#4f46e5',
    borderRadius: '50%',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    color: '#fff',
    fontWeight: 'bold',
    fontSize: '0.8rem',
    flexShrink: 0,
  },
  stepContent: {
    flex: 1,
  },
  stepAction: {
    display: 'block',
    color: '#fff',
    fontSize: '0.9rem',
    fontWeight: 'bold',
    marginBottom: '4px',
  },
  stepOutcome: {
    display: 'block',
    color: '#888',
    fontSize: '0.8rem',
    marginBottom: '2px',
  },
  stepTick: {
    color: '#4ade80',
    fontSize: '0.75rem',
  },

  // Advanced Metrics Styles
  advancedPanel: {
    backgroundColor: '#16162a',
    borderRadius: '8px',
    padding: '16px',
  },
  advancedSection: {
    marginBottom: '16px',
  },
  advancedTitle: {
    color: '#fff',
    fontSize: '0.9rem',
    margin: '0 0 10px',
    paddingBottom: '6px',
    borderBottom: '1px solid #333',
  },
  advancedGrid: {
    display: 'grid',
    gridTemplateColumns: 'repeat(2, 1fr)',
    gap: '8px',
  },
  advancedMetricItem: {
    backgroundColor: '#1a2a3a',
    borderRadius: '4px',
    padding: '10px',
    textAlign: 'center',
  },
  advancedMetricValue: {
    display: 'block',
    color: '#4ade80',
    fontSize: '1.1rem',
    fontWeight: 'bold',
  },
  advancedMetricLabel: {
    color: '#888',
    fontSize: '0.7rem',
    textTransform: 'uppercase',
  },
  correlationList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  correlationItem: {
    display: 'flex',
    alignItems: 'center',
    gap: '10px',
    backgroundColor: '#1a2a3a',
    borderRadius: '4px',
    padding: '8px 10px',
  },
  correlationGenes: {
    flex: 1,
    color: '#fff',
    fontSize: '0.85rem',
  },
  correlationValue: {
    fontWeight: 'bold',
    fontSize: '0.9rem',
  },
  correlationSig: {
    color: '#888',
    fontSize: '0.7rem',
    textTransform: 'uppercase',
  },
  anomalyList: {
    display: 'flex',
    flexDirection: 'column',
    gap: '6px',
  },
  anomalyItem: {
    display: 'flex',
    flexWrap: 'wrap',
    gap: '8px',
    backgroundColor: '#3a1a1a',
    borderRadius: '4px',
    padding: '8px 10px',
  },
  anomalyTick: {
    color: '#f87171',
    fontSize: '0.8rem',
    fontWeight: 'bold',
  },
  anomalyType: {
    color: '#fbbf24',
    fontSize: '0.8rem',
    textTransform: 'capitalize',
  },
  anomalyDesc: {
    color: '#888',
    fontSize: '0.75rem',
    width: '100%',
  },
};

export default AnalysisPanel;
