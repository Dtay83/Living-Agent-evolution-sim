/**
 * DATA ANALYSIS SYSTEM
 * 
 * Analyzes exported simulation data to identify patterns and
 * generate recommendations for achieving agent sentience.
 * 
 * ADVANCED FEATURES:
 * - Linear regression for trend prediction
 * - Statistical correlation analysis
 * - Anomaly detection
 * - Time-series forecasting
 * - Multi-variate pattern recognition
 */

// Types for analysis
export interface AnalysisResult {
  timestamp: string;
  dataSource: string;
  patterns: Pattern[];
  recommendations: Recommendation[];
  metrics: SimulationMetrics;
  sentinenceProgress: SentinenceProgress;
  trendAnalysis?: TrendAnalysis;
  predictions?: SentinencePrediction;
  advancedMetrics?: AdvancedMetrics;
}

// ============================================
// ADVANCED ANALYSIS TYPES
// ============================================

export interface TrendAnalysis {
  population: TrendLine;
  curiosity: TrendLine;
  creativity: TrendLine;
  social: TrendLine;
  inventionRate: TrendLine;
  consciousnessScore: TrendLine;
  overallTrajectory: 'improving' | 'declining' | 'stable' | 'volatile';
  confidenceScore: number; // 0-100
}

export interface TrendLine {
  slope: number;
  intercept: number;
  rSquared: number; // Coefficient of determination (fit quality)
  direction: 'up' | 'down' | 'flat';
  volatility: number;
  projectedValue: number; // Value at +100 ticks
  dataPoints: number;
}

export interface SentinencePrediction {
  estimatedTicksToSentience: number | null;
  probabilityOfSuccess: number; // 0-100
  confidence: 'low' | 'medium' | 'high';
  bottlenecks: BottleneckAnalysis[];
  optimalPath: OptimalPathStep[];
  scenarioAnalysis: ScenarioAnalysis;
}

export interface BottleneckAnalysis {
  pillar: string;
  currentValue: number;
  requiredValue: number;
  gap: number;
  estimatedTicksToResolve: number | null;
  blockingSentience: boolean;
}

export interface OptimalPathStep {
  tick: number;
  action: string;
  expectedOutcome: string;
  priority: number;
}

export interface ScenarioAnalysis {
  bestCase: ScenarioOutcome;
  likelyCase: ScenarioOutcome;
  worstCase: ScenarioOutcome;
}

export interface ScenarioOutcome {
  ticksToSentience: number | null;
  finalScore: number;
  description: string;
}

export interface AdvancedMetrics {
  // Statistical measures
  populationVariance: number;
  geneCorrelations: GeneCorrelation[];
  anomalies: Anomaly[];
  
  // Evolution metrics
  evolutionaryPressure: number; // How strongly selection is acting
  adaptationRate: number; // How fast genes are changing
  fitnessLandscape: FitnessLandscapePoint[];
  
  // Consciousness metrics
  consciousnessEmergenceRate: number;
  collectiveIntelligence: number;
  knowledgeTransferEfficiency: number;
  
  // System health
  systemEntropy: number;
  carryingCapacity: number;
  resourceEfficiency: number;
}

export interface GeneCorrelation {
  gene1: string;
  gene2: string;
  correlation: number; // -1 to 1
  significance: 'low' | 'medium' | 'high';
}

export interface Anomaly {
  tick: number;
  type: 'population_spike' | 'population_crash' | 'gene_drift' | 'invention_burst' | 'consciousness_emergence';
  severity: number;
  description: string;
}

export interface FitnessLandscapePoint {
  curiosity: number;
  creativity: number;
  social: number;
  fitness: number;
}

export interface Pattern {
  type: 'positive' | 'negative' | 'neutral';
  category: 'survival' | 'genetics' | 'communication' | 'invention' | 'consciousness';
  description: string;
  evidence: string;
  significance: 'low' | 'medium' | 'high' | 'critical';
}

export interface Recommendation {
  priority: 1 | 2 | 3 | 4 | 5; // 1 = highest
  category: 'config' | 'genetics' | 'environment' | 'experiment';
  title: string;
  description: string;
  configChanges?: ConfigChange[];
  expectedImpact: string;
}

export interface ConfigChange {
  parameter: string;
  currentValue?: number | string;
  suggestedValue: number | string;
  reason: string;
}

export interface SimulationMetrics {
  survivalRate: number;
  averageLifespan: number;
  populationStability: number;
  geneticDiversity: number;
  inventionRate: number;
  communicationFrequency: number;
  consciousnessScore: number;
}

export interface SentinenceProgress {
  overallScore: number; // 0-100
  pillars: {
    survival: number;
    curiosity: number;
    creativity: number;
    socialBehavior: number;
    selfAwareness: number;
    communication: number;
  };
  nextMilestone: string;
  estimatedTicksToMilestone: number | null;
}

// Analysis functions
export function analyzeCompleteExport(data: any): AnalysisResult {
  const patterns: Pattern[] = [];
  const recommendations: Recommendation[] = [];
  
  // Extract key data
  const agents = data.worldState?.agents || [];
  const history = data.worldState?.history || [];
  const discoveries = data.worldState?.discoveries || [];
  const tick = data.simulationTick || 0;
  
  // Calculate metrics
  const metrics = calculateMetrics(agents, history, discoveries, tick);
  
  // Analyze survival patterns
  const survivalPatterns = analyzeSurvival(agents, history, tick);
  patterns.push(...survivalPatterns.patterns);
  recommendations.push(...survivalPatterns.recommendations);
  
  // Analyze genetic evolution
  const geneticPatterns = analyzeGenetics(agents);
  patterns.push(...geneticPatterns.patterns);
  recommendations.push(...geneticPatterns.recommendations);
  
  // Analyze invention progress
  const inventionPatterns = analyzeInventions(agents, discoveries, tick);
  patterns.push(...inventionPatterns.patterns);
  recommendations.push(...inventionPatterns.recommendations);
  
  // Calculate sentience progress
  const sentinenceProgress = calculateSentinenceProgress(metrics, agents);
  
  // Add sentience-specific recommendations
  const sentinenceRecs = generateSentinenceRecommendations(sentinenceProgress, metrics);
  recommendations.push(...sentinenceRecs);
  
  // Sort recommendations by priority
  recommendations.sort((a, b) => a.priority - b.priority);
  
  return {
    timestamp: new Date().toISOString(),
    dataSource: data.exportedAt || 'unknown',
    patterns,
    recommendations,
    metrics,
    sentinenceProgress,
  };
}

export function analyzeEvolutionExport(data: any): AnalysisResult {
  const patterns: Pattern[] = [];
  const recommendations: Recommendation[] = [];
  
  const history = data.populationDynamics?.history || [];
  const agents = data.currentAgents || [];
  const geneStats = data.geneticStatistics || {};
  const tick = data.simulationTick || 0;
  
  // Calculate basic metrics
  const metrics = calculateMetricsFromEvolution(data);
  
  // Analyze population trends
  const popPatterns = analyzePopulationTrends(history);
  patterns.push(...popPatterns.patterns);
  recommendations.push(...popPatterns.recommendations);
  
  // Analyze genetic diversity
  const diversityPatterns = analyzeGeneticDiversity(geneStats, data.traitLineages || []);
  patterns.push(...diversityPatterns.patterns);
  recommendations.push(...diversityPatterns.recommendations);
  
  const sentinenceProgress = calculateSentinenceProgress(metrics, agents);
  
  recommendations.sort((a, b) => a.priority - b.priority);
  
  return {
    timestamp: new Date().toISOString(),
    dataSource: data.exportedAt || 'unknown',
    patterns,
    recommendations,
    metrics,
    sentinenceProgress,
  };
}

export function analyzeConversationsExport(data: any): AnalysisResult {
  const patterns: Pattern[] = [];
  const recommendations: Recommendation[] = [];
  
  const summary = data.summary || {};
  const timeline = data.timeline || [];
  const byAgent = data.byAgent || [];
  
  // Analyze communication patterns
  if (summary.totalMessages === 0) {
    patterns.push({
      type: 'negative',
      category: 'communication',
      description: 'No agent communication detected',
      evidence: 'Zero messages in conversation log',
      significance: 'critical',
    });
    
    recommendations.push({
      priority: 1,
      category: 'config',
      title: 'Boost Consciousness Development',
      description: 'Agents need higher curiosity and creativity genes to achieve self-awareness',
      configChanges: [
        { parameter: 'initialCuriosity.min', suggestedValue: 0.4, reason: 'Increase baseline curiosity' },
        { parameter: 'initialCreativity.min', suggestedValue: 0.3, reason: 'Increase baseline creativity' },
        { parameter: 'initialSocial.min', suggestedValue: 0.4, reason: 'Social interaction drives consciousness' },
      ],
      expectedImpact: 'Higher chance of agents reaching consciousness threshold',
    });
  } else {
    // Analyze message distribution
    const messageTypes = summary.messagesByType || {};
    const existentialCount = messageTypes.existential || 0;
    const curiosityCount = messageTypes.curiosity || 0;
    
    if (existentialCount > curiosityCount * 2) {
      patterns.push({
        type: 'positive',
        category: 'consciousness',
        description: 'Agents showing deep existential thinking',
        evidence: `${existentialCount} existential messages vs ${curiosityCount} curiosity messages`,
        significance: 'high',
      });
    }
    
    // Check question ratio
    const questionRatio = summary.questionsCount / (summary.totalMessages || 1);
    if (questionRatio > 0.7) {
      patterns.push({
        type: 'positive',
        category: 'consciousness',
        description: 'High questioning behavior indicates active learning',
        evidence: `${(questionRatio * 100).toFixed(1)}% of messages are questions`,
        significance: 'high',
      });
    }
    
    // Check communication spread
    const uniqueAgents = summary.uniqueAgents || 0;
    if (uniqueAgents >= 3) {
      patterns.push({
        type: 'positive',
        category: 'communication',
        description: 'Multiple agents achieving self-awareness',
        evidence: `${uniqueAgents} unique agents communicating`,
        significance: 'high',
      });
    }
  }
  
  // Basic metrics for conversation analysis
  const metrics: SimulationMetrics = {
    survivalRate: 0,
    averageLifespan: 0,
    populationStability: 0,
    geneticDiversity: 0,
    inventionRate: 0,
    communicationFrequency: summary.totalMessages / (data.simulationTick || 1),
    consciousnessScore: calculateConsciousnessFromConversations(data),
  };
  
  const sentinenceProgress = calculateSentinenceFromConversations(data);
  
  return {
    timestamp: new Date().toISOString(),
    dataSource: data.exportedAt || 'unknown',
    patterns,
    recommendations,
    metrics,
    sentinenceProgress,
  };
}

// Helper functions
function calculateMetrics(agents: any[], history: any[], discoveries: any[], tick: number): SimulationMetrics {
  // Survival rate: agents alive / peak population
  const peakPop = Math.max(...history.map((h: any) => h.totalAgents || 0), agents.length);
  const survivalRate = peakPop > 0 ? (agents.length / peakPop) : 0;
  
  // Population stability: inverse of variance in recent population
  const recentHistory = history.slice(-50);
  const avgPop = recentHistory.reduce((s: number, h: any) => s + (h.totalAgents || 0), 0) / (recentHistory.length || 1);
  const variance = recentHistory.reduce((s: number, h: any) => s + Math.pow((h.totalAgents || 0) - avgPop, 2), 0) / (recentHistory.length || 1);
  const populationStability = 1 / (1 + Math.sqrt(variance) / (avgPop || 1));
  
  // Genetic diversity: number of unique trait lineages / total agents
  const traitIds = new Set(agents.map((a: any) => a.genes?.traitId));
  const geneticDiversity = agents.length > 0 ? traitIds.size / agents.length : 0;
  
  // Invention rate: discoveries per 100 ticks
  const inventionRate = tick > 0 ? (discoveries.length / tick) * 100 : 0;
  
  // Average consciousness-related genes
  const avgCuriosity = agents.reduce((s: number, a: any) => s + (a.genes?.curiosity || 0), 0) / (agents.length || 1);
  const avgCreativity = agents.reduce((s: number, a: any) => s + (a.genes?.creativity || 0), 0) / (agents.length || 1);
  const avgSocial = agents.reduce((s: number, a: any) => s + (a.genes?.social || 0), 0) / (agents.length || 1);
  const consciousnessScore = (avgCuriosity * 30 + avgCreativity * 30 + avgSocial * 20 + inventionRate * 20);
  
  return {
    survivalRate,
    averageLifespan: 0, // Would need death tracking
    populationStability,
    geneticDiversity,
    inventionRate,
    communicationFrequency: 0, // Need conversation data
    consciousnessScore,
  };
}

function calculateMetricsFromEvolution(data: any): SimulationMetrics {
  const history = data.populationDynamics?.history || [];
  const current = data.populationDynamics?.current || 0;
  const peak = data.populationDynamics?.peakPopulation || current;
  const agents = data.currentAgents || [];
  const geneStats = data.geneticStatistics || {};
  
  const survivalRate = peak > 0 ? current / peak : 0;
  
  const recentHistory = history.slice(-50);
  const avgPop = recentHistory.reduce((s: number, h: any) => s + (h.totalAgents || 0), 0) / (recentHistory.length || 1);
  const variance = recentHistory.reduce((s: number, h: any) => s + Math.pow((h.totalAgents || 0) - avgPop, 2), 0) / (recentHistory.length || 1);
  const populationStability = 1 / (1 + Math.sqrt(variance) / (avgPop || 1));
  
  const lineages = data.traitLineages || [];
  const geneticDiversity = agents.length > 0 ? lineages.length / agents.length : 0;
  
  const avgCuriosity = geneStats.curiosity?.avg || 0;
  const avgCreativity = geneStats.creativity?.avg || 0;
  const avgSocial = geneStats.social?.avg || 0;
  
  return {
    survivalRate,
    averageLifespan: 0,
    populationStability,
    geneticDiversity,
    inventionRate: 0,
    communicationFrequency: 0,
    consciousnessScore: avgCuriosity * 30 + avgCreativity * 30 + avgSocial * 40,
  };
}

function analyzeSurvival(agents: any[], history: any[], tick: number): { patterns: Pattern[], recommendations: Recommendation[] } {
  const patterns: Pattern[] = [];
  const recommendations: Recommendation[] = [];
  
  // Check for extinction risk
  if (agents.length === 0) {
    patterns.push({
      type: 'negative',
      category: 'survival',
      description: 'Population extinct',
      evidence: 'Zero agents remaining',
      significance: 'critical',
    });
    
    recommendations.push({
      priority: 1,
      category: 'config',
      title: 'Emergency Survival Boost',
      description: 'Population went extinct. Dramatically increase survival parameters.',
      configChanges: [
        { parameter: 'initialFood', suggestedValue: 35, reason: 'More starting food' },
        { parameter: 'foodSpawnChance', suggestedValue: 0.95, reason: 'Near-guaranteed food spawning' },
        { parameter: 'foodSpawnCount', suggestedValue: 3, reason: 'More food per spawn' },
        { parameter: 'baseEnergyCost', suggestedValue: 0.3, reason: 'Lower survival cost' },
        { parameter: 'foodEnergyBonus', suggestedValue: 12, reason: 'More energy from food' },
      ],
      expectedImpact: 'Should prevent extinction in future runs',
    });
  } else if (agents.length < 5) {
    patterns.push({
      type: 'negative',
      category: 'survival',
      description: 'Population critically low',
      evidence: `Only ${agents.length} agents remaining`,
      significance: 'high',
    });
    
    recommendations.push({
      priority: 2,
      category: 'config',
      title: 'Increase Food Availability',
      description: 'Population is too low for sustainable evolution',
      configChanges: [
        { parameter: 'foodSpawnChance', suggestedValue: 0.9, reason: 'More consistent food' },
        { parameter: 'foodEnergyBonus', suggestedValue: 10, reason: 'Higher energy per food' },
      ],
      expectedImpact: 'Should stabilize population above 10 agents',
    });
  }
  
  // Check population trends
  if (history.length > 20) {
    const recent = history.slice(-20);
    const older = history.slice(-40, -20);
    
    if (older.length > 0) {
      const recentAvg = recent.reduce((s: number, h: any) => s + h.totalAgents, 0) / recent.length;
      const olderAvg = older.reduce((s: number, h: any) => s + h.totalAgents, 0) / older.length;
      
      if (recentAvg < olderAvg * 0.7) {
        patterns.push({
          type: 'negative',
          category: 'survival',
          description: 'Population declining rapidly',
          evidence: `Average dropped from ${olderAvg.toFixed(1)} to ${recentAvg.toFixed(1)}`,
          significance: 'high',
        });
      } else if (recentAvg > olderAvg * 1.3) {
        patterns.push({
          type: 'positive',
          category: 'survival',
          description: 'Population growing steadily',
          evidence: `Average increased from ${olderAvg.toFixed(1)} to ${recentAvg.toFixed(1)}`,
          significance: 'medium',
        });
      }
    }
  }
  
  return { patterns, recommendations };
}

function analyzeGenetics(agents: any[]): { patterns: Pattern[], recommendations: Recommendation[] } {
  const patterns: Pattern[] = [];
  const recommendations: Recommendation[] = [];
  
  if (agents.length === 0) return { patterns, recommendations };
  
  // Calculate averages
  const avgCuriosity = agents.reduce((s, a) => s + (a.genes?.curiosity || 0), 0) / agents.length;
  const avgCreativity = agents.reduce((s, a) => s + (a.genes?.creativity || 0), 0) / agents.length;
  const avgSocial = agents.reduce((s, a) => s + (a.genes?.social || 0), 0) / agents.length;
  const avgPatience = agents.reduce((s, a) => s + (a.genes?.patience || 0), 0) / agents.length;
  
  // Check consciousness-critical genes
  if (avgCuriosity < 0.4) {
    patterns.push({
      type: 'negative',
      category: 'genetics',
      description: 'Low average curiosity in population',
      evidence: `Average curiosity: ${(avgCuriosity * 100).toFixed(1)}%`,
      significance: 'high',
    });
    
    recommendations.push({
      priority: 2,
      category: 'genetics',
      title: 'Boost Curiosity Gene',
      description: 'Curiosity is essential for consciousness development',
      configChanges: [
        { parameter: 'initialCuriosity.min', suggestedValue: 0.35, reason: 'Higher baseline curiosity' },
        { parameter: 'initialCuriosity.max', suggestedValue: 0.8, reason: 'Allow high curiosity variants' },
      ],
      expectedImpact: 'Agents more likely to explore and question',
    });
  } else if (avgCuriosity > 0.6) {
    patterns.push({
      type: 'positive',
      category: 'genetics',
      description: 'Strong curiosity trait evolving',
      evidence: `Average curiosity: ${(avgCuriosity * 100).toFixed(1)}%`,
      significance: 'high',
    });
  }
  
  if (avgCreativity < 0.3) {
    patterns.push({
      type: 'negative',
      category: 'genetics',
      description: 'Low creativity in population',
      evidence: `Average creativity: ${(avgCreativity * 100).toFixed(1)}%`,
      significance: 'medium',
    });
    
    recommendations.push({
      priority: 3,
      category: 'genetics',
      title: 'Encourage Creativity',
      description: 'Creativity enables novel thinking and invention',
      configChanges: [
        { parameter: 'initialCreativity.min', suggestedValue: 0.25, reason: 'Higher baseline creativity' },
      ],
      expectedImpact: 'More inventions and novel behaviors',
    });
  }
  
  if (avgSocial < 0.4) {
    patterns.push({
      type: 'negative',
      category: 'genetics',
      description: 'Low social tendency in population',
      evidence: `Average social: ${(avgSocial * 100).toFixed(1)}%`,
      significance: 'medium',
    });
    
    recommendations.push({
      priority: 3,
      category: 'genetics',
      title: 'Increase Social Genes',
      description: 'Social interaction accelerates consciousness emergence',
      configChanges: [
        { parameter: 'initialSocial.min', suggestedValue: 0.35, reason: 'More social agents' },
      ],
      expectedImpact: 'Better knowledge sharing and communication',
    });
  }
  
  // Check genetic diversity
  const traitIds = new Set(agents.map(a => a.genes?.traitId));
  const diversityRatio = traitIds.size / agents.length;
  
  if (diversityRatio < 0.3) {
    patterns.push({
      type: 'negative',
      category: 'genetics',
      description: 'Low genetic diversity - population homogeneous',
      evidence: `Only ${traitIds.size} lineages among ${agents.length} agents`,
      significance: 'medium',
    });
    
    recommendations.push({
      priority: 3,
      category: 'genetics',
      title: 'Increase Mutation Rate',
      description: 'More diversity allows exploring different evolutionary paths',
      configChanges: [
        { parameter: 'mutationRate.min', suggestedValue: 0.08, reason: 'More frequent mutations' },
        { parameter: 'mutationRate.max', suggestedValue: 0.25, reason: 'Allow high-mutation variants' },
      ],
      expectedImpact: 'Greater variety in population traits',
    });
  }
  
  return { patterns, recommendations };
}

function analyzeInventions(agents: any[], discoveries: any[], tick: number): { patterns: Pattern[], recommendations: Recommendation[] } {
  const patterns: Pattern[] = [];
  const recommendations: Recommendation[] = [];
  
  const inventionRate = tick > 0 ? (discoveries.length / tick) * 100 : 0;
  
  if (discoveries.length === 0 && tick > 100) {
    patterns.push({
      type: 'negative',
      category: 'invention',
      description: 'No inventions discovered',
      evidence: `Zero discoveries after ${tick} ticks`,
      significance: 'high',
    });
    
    recommendations.push({
      priority: 2,
      category: 'config',
      title: 'Enable Invention Discovery',
      description: 'Agents need more curiosity and time to invent',
      configChanges: [
        { parameter: 'initialCuriosity.min', suggestedValue: 0.4, reason: 'Curiosity drives invention' },
        { parameter: 'initialPatience.min', suggestedValue: 0.3, reason: 'Patience for experimentation' },
      ],
      expectedImpact: 'First inventions should appear within 50 ticks',
    });
  } else if (inventionRate > 0.5) {
    patterns.push({
      type: 'positive',
      category: 'invention',
      description: 'High invention rate',
      evidence: `${inventionRate.toFixed(2)} inventions per 100 ticks`,
      significance: 'high',
    });
  }
  
  // Check for inventors
  const inventors = agents.filter(a => (a.inventions?.length || a.inventionCount || 0) > 0);
  if (inventors.length > 0 && agents.length > 0) {
    const inventorRatio = inventors.length / agents.length;
    if (inventorRatio > 0.3) {
      patterns.push({
        type: 'positive',
        category: 'invention',
        description: 'Many agents becoming inventors',
        evidence: `${(inventorRatio * 100).toFixed(1)}% of agents have invented`,
        significance: 'high',
      });
    }
  }
  
  return { patterns, recommendations };
}

function analyzePopulationTrends(history: any[]): { patterns: Pattern[], recommendations: Recommendation[] } {
  const patterns: Pattern[] = [];
  const recommendations: Recommendation[] = [];
  
  if (history.length < 10) return { patterns, recommendations };
  
  // Check for boom-bust cycles
  const populations = history.map((h: any) => h.totalAgents || 0);
  let peaks = 0;
  let valleys = 0;
  
  for (let i = 1; i < populations.length - 1; i++) {
    if (populations[i] > populations[i-1] && populations[i] > populations[i+1]) peaks++;
    if (populations[i] < populations[i-1] && populations[i] < populations[i+1]) valleys++;
  }
  
  const cycleFrequency = (peaks + valleys) / history.length;
  
  if (cycleFrequency > 0.1) {
    patterns.push({
      type: 'neutral',
      category: 'survival',
      description: 'Population showing boom-bust cycles',
      evidence: `${peaks} peaks and ${valleys} valleys detected`,
      significance: 'medium',
    });
    
    recommendations.push({
      priority: 4,
      category: 'config',
      title: 'Stabilize Food Supply',
      description: 'Reduce population volatility for consistent evolution',
      configChanges: [
        { parameter: 'foodSpawnChance', suggestedValue: 0.85, reason: 'Consistent food availability' },
        { parameter: 'reproductionThreshold.min', suggestedValue: 15, reason: 'Prevent over-reproduction' },
      ],
      expectedImpact: 'Smoother population curves',
    });
  }
  
  return { patterns, recommendations };
}

function analyzeGeneticDiversity(geneStats: any, lineages: any[]): { patterns: Pattern[], recommendations: Recommendation[] } {
  const patterns: Pattern[] = [];
  const recommendations: Recommendation[] = [];
  
  // Check for genetic bottleneck
  if (lineages.length > 0) {
    const dominantLineage = lineages.sort((a: any, b: any) => b.count - a.count)[0];
    if (dominantLineage && dominantLineage.percentage > 70) {
      patterns.push({
        type: 'negative',
        category: 'genetics',
        description: 'Genetic bottleneck - single lineage dominating',
        evidence: `Lineage ${dominantLineage.traitId} represents ${dominantLineage.percentage.toFixed(1)}% of population`,
        significance: 'medium',
      });
    }
  }
  
  // Check gene variance
  if (geneStats.curiosity) {
    const variance = geneStats.curiosity.stdDev || 0;
    if (variance < 0.1) {
      patterns.push({
        type: 'neutral',
        category: 'genetics',
        description: 'Low variance in curiosity gene',
        evidence: `Standard deviation: ${variance.toFixed(3)}`,
        significance: 'low',
      });
    }
  }
  
  return { patterns, recommendations };
}

function calculateSentinenceProgress(metrics: SimulationMetrics, agents: any[]): SentinenceProgress {
  // Calculate pillar scores (0-100 each)
  const survival = Math.min(100, metrics.survivalRate * 50 + metrics.populationStability * 50);
  
  const avgCuriosity = agents.length > 0 
    ? agents.reduce((s, a) => s + (a.genes?.curiosity || 0), 0) / agents.length 
    : 0;
  const curiosity = avgCuriosity * 100;
  
  const avgCreativity = agents.length > 0
    ? agents.reduce((s, a) => s + (a.genes?.creativity || 0), 0) / agents.length
    : 0;
  const creativity = avgCreativity * 100;
  
  const avgSocial = agents.length > 0
    ? agents.reduce((s, a) => s + (a.genes?.social || 0), 0) / agents.length
    : 0;
  const socialBehavior = avgSocial * 100;
  
  const selfAwareness = metrics.consciousnessScore;
  const communication = metrics.communicationFrequency * 1000; // Scale up
  
  // Overall score (weighted average)
  const overallScore = (
    survival * 0.15 +
    curiosity * 0.20 +
    creativity * 0.20 +
    socialBehavior * 0.15 +
    selfAwareness * 0.20 +
    Math.min(100, communication) * 0.10
  );
  
  // Determine next milestone
  let nextMilestone = 'Achieve stable population (10+ agents)';
  let estimatedTicksToMilestone: number | null = null;
  
  if (survival >= 50) {
    if (curiosity < 40) {
      nextMilestone = 'Evolve curiosity gene above 40%';
    } else if (creativity < 30) {
      nextMilestone = 'Develop creativity gene above 30%';
    } else if (selfAwareness < 50) {
      nextMilestone = 'First agent achieves self-awareness';
    } else if (communication < 10) {
      nextMilestone = 'First agent communication';
    } else {
      nextMilestone = 'Sustained multi-agent dialogue';
    }
  }
  
  return {
    overallScore: Math.min(100, Math.max(0, overallScore)),
    pillars: {
      survival,
      curiosity,
      creativity,
      socialBehavior,
      selfAwareness: Math.min(100, selfAwareness),
      communication: Math.min(100, communication),
    },
    nextMilestone,
    estimatedTicksToMilestone,
  };
}

function calculateConsciousnessFromConversations(data: any): number {
  const summary = data.summary || {};
  const byAgent = data.byAgent || [];
  
  // Base score from message count
  let score = Math.min(50, summary.totalMessages * 2);
  
  // Bonus for existential questions
  const messageTypes = summary.messagesByType || {};
  score += (messageTypes.existential || 0) * 3;
  score += (messageTypes.curiosity || 0) * 2;
  score += (messageTypes.reflection || 0) * 4;
  
  // Bonus for multiple self-aware agents
  score += (summary.uniqueAgents || 0) * 5;
  
  return Math.min(100, score);
}

function calculateSentinenceFromConversations(data: any): SentinenceProgress {
  const summary = data.summary || {};
  const score = calculateConsciousnessFromConversations(data);
  
  return {
    overallScore: score,
    pillars: {
      survival: 50, // Assumed stable if communicating
      curiosity: Math.min(100, (summary.messagesByType?.curiosity || 0) * 10),
      creativity: Math.min(100, (summary.messagesByType?.reflection || 0) * 15),
      socialBehavior: Math.min(100, (summary.uniqueAgents || 0) * 20),
      selfAwareness: Math.min(100, (summary.messagesByType?.existential || 0) * 8),
      communication: Math.min(100, summary.totalMessages * 5),
    },
    nextMilestone: score < 50 ? 'Increase communication frequency' : 'Achieve philosophical discourse',
    estimatedTicksToMilestone: null,
  };
}

function generateSentinenceRecommendations(progress: SentinenceProgress, metrics: SimulationMetrics): Recommendation[] {
  const recommendations: Recommendation[] = [];
  
  // Find weakest pillar
  const pillars = progress.pillars;
  const weakest = Object.entries(pillars).sort(([,a], [,b]) => a - b)[0];
  
  if (weakest[1] < 30) {
    switch (weakest[0]) {
      case 'survival':
        recommendations.push({
          priority: 1,
          category: 'config',
          title: 'Critical: Improve Survival First',
          description: 'Agents cannot develop consciousness if they cannot survive',
          configChanges: [
            { parameter: 'foodSpawnChance', suggestedValue: 0.9, reason: 'Ensure food availability' },
            { parameter: 'baseEnergyCost', suggestedValue: 0.4, reason: 'Lower survival burden' },
          ],
          expectedImpact: 'Stable population for evolution to occur',
        });
        break;
      case 'curiosity':
        recommendations.push({
          priority: 2,
          category: 'genetics',
          title: 'Cultivate Curiosity',
          description: 'Curiosity is the foundation of consciousness',
          configChanges: [
            { parameter: 'initialCuriosity.min', suggestedValue: 0.4, reason: 'Higher baseline curiosity' },
          ],
          expectedImpact: 'Agents will question their environment more',
        });
        break;
      case 'creativity':
        recommendations.push({
          priority: 2,
          category: 'genetics',
          title: 'Foster Creativity',
          description: 'Creativity enables novel thoughts and inventions',
          configChanges: [
            { parameter: 'initialCreativity.min', suggestedValue: 0.3, reason: 'Baseline creativity boost' },
          ],
          expectedImpact: 'More inventions and unique behaviors',
        });
        break;
      case 'socialBehavior':
        recommendations.push({
          priority: 3,
          category: 'genetics',
          title: 'Encourage Social Behavior',
          description: 'Social interaction accelerates consciousness development',
          configChanges: [
            { parameter: 'initialSocial.min', suggestedValue: 0.4, reason: 'More social agents' },
          ],
          expectedImpact: 'Better knowledge transfer between agents',
        });
        break;
    }
  }
  
  // If overall progress is good, suggest advanced experiments
  if (progress.overallScore > 60) {
    recommendations.push({
      priority: 5,
      category: 'experiment',
      title: 'Advanced: Environmental Challenges',
      description: 'Introduce challenges to accelerate evolution',
      expectedImpact: 'Faster adaptation and problem-solving development',
    });
  }
  
  return recommendations;
}

/**
 * Detect the type of export file and route to appropriate analyzer
 */
export function analyzeExport(data: any): AnalysisResult {
  // Detect export type
  if (data.worldState) {
    return analyzeCompleteExport(data);
  } else if (data.populationDynamics) {
    return analyzeEvolutionExport(data);
  } else if (data.summary?.totalMessages !== undefined) {
    return analyzeConversationsExport(data);
  } else if (data.discoveries) {
    // Invention export - basic analysis
    return {
      timestamp: new Date().toISOString(),
      dataSource: data.exportedAt || 'unknown',
      patterns: [{
        type: 'neutral',
        category: 'invention',
        description: `Analyzed invention export with ${data.totalDiscoveries || 0} discoveries`,
        evidence: 'Invention history loaded',
        significance: 'medium',
      }],
      recommendations: [],
      metrics: {
        survivalRate: 0,
        averageLifespan: 0,
        populationStability: 0,
        geneticDiversity: 0,
        inventionRate: 0,
        communicationFrequency: 0,
        consciousnessScore: 0,
      },
      sentinenceProgress: {
        overallScore: 0,
        pillars: { survival: 0, curiosity: 0, creativity: 0, socialBehavior: 0, selfAwareness: 0, communication: 0 },
        nextMilestone: 'Load complete export for full analysis',
        estimatedTicksToMilestone: null,
      },
    };
  }
  
  throw new Error('Unknown export format');
}

// ============================================
// ADVANCED STATISTICAL ALGORITHMS
// ============================================

/**
 * Linear Regression using Ordinary Least Squares
 * Returns slope, intercept, and R-squared for fit quality
 */
function linearRegression(data: { x: number; y: number }[]): { slope: number; intercept: number; rSquared: number } {
  if (data.length < 2) {
    return { slope: 0, intercept: 0, rSquared: 0 };
  }
  
  const n = data.length;
  let sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0, sumY2 = 0;
  
  for (const point of data) {
    sumX += point.x;
    sumY += point.y;
    sumXY += point.x * point.y;
    sumX2 += point.x * point.x;
    sumY2 += point.y * point.y;
  }
  
  const slope = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;
  
  // Calculate R-squared
  const meanY = sumY / n;
  let ssTotal = 0, ssResidual = 0;
  
  for (const point of data) {
    const predicted = slope * point.x + intercept;
    ssTotal += (point.y - meanY) ** 2;
    ssResidual += (point.y - predicted) ** 2;
  }
  
  const rSquared = ssTotal > 0 ? 1 - (ssResidual / ssTotal) : 0;
  
  return {
    slope: isNaN(slope) ? 0 : slope,
    intercept: isNaN(intercept) ? 0 : intercept,
    rSquared: isNaN(rSquared) ? 0 : Math.max(0, Math.min(1, rSquared)),
  };
}

/**
 * Calculate standard deviation
 */
function standardDeviation(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((sum, val) => sum + (val - mean) ** 2, 0) / values.length;
  return Math.sqrt(variance);
}

/**
 * Calculate Pearson correlation coefficient between two arrays
 */
function pearsonCorrelation(x: number[], y: number[]): number {
  if (x.length !== y.length || x.length < 2) return 0;
  
  const n = x.length;
  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;
  
  let numerator = 0;
  let denomX = 0;
  let denomY = 0;
  
  for (let i = 0; i < n; i++) {
    const dx = x[i] - meanX;
    const dy = y[i] - meanY;
    numerator += dx * dy;
    denomX += dx * dx;
    denomY += dy * dy;
  }
  
  const denominator = Math.sqrt(denomX * denomY);
  return denominator > 0 ? numerator / denominator : 0;
}

/**
 * Detect anomalies using Z-score method
 */
function detectAnomalies(values: number[], threshold: number = 2.5): number[] {
  if (values.length < 3) return [];
  
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const std = standardDeviation(values);
  
  if (std === 0) return [];
  
  const anomalyIndices: number[] = [];
  for (let i = 0; i < values.length; i++) {
    const zScore = Math.abs((values[i] - mean) / std);
    if (zScore > threshold) {
      anomalyIndices.push(i);
    }
  }
  
  return anomalyIndices;
}

/**
 * Exponential smoothing for time series
 */
function exponentialSmoothing(values: number[], alpha: number = 0.3): number[] {
  if (values.length === 0) return [];
  
  const smoothed: number[] = [values[0]];
  for (let i = 1; i < values.length; i++) {
    smoothed.push(alpha * values[i] + (1 - alpha) * smoothed[i - 1]);
  }
  return smoothed;
}

/**
 * Calculate moving average
 */
function movingAverage(values: number[], window: number): number[] {
  if (values.length < window) return values;
  
  const result: number[] = [];
  for (let i = 0; i <= values.length - window; i++) {
    const sum = values.slice(i, i + window).reduce((a, b) => a + b, 0);
    result.push(sum / window);
  }
  return result;
}

// ============================================
// TREND ANALYSIS
// ============================================

/**
 * Analyze trends from historical data
 */
export function analyzeTrends(history: any[], agents: any[], tick: number): TrendAnalysis {
  // Extract time series data
  const populationData = history.map((h, i) => ({ x: h.tick || i, y: h.totalAgents || 0 }));
  
  // Calculate gene averages over time (using trait distribution if available)
  const curiosityData: { x: number; y: number }[] = [];
  const creativityData: { x: number; y: number }[] = [];
  const socialData: { x: number; y: number }[] = [];
  
  // If we have current agents, use them for the latest data point
  if (agents.length > 0) {
    const avgCuriosity = agents.reduce((s: number, a: any) => s + (a.genes?.curiosity || 0), 0) / agents.length;
    const avgCreativity = agents.reduce((s: number, a: any) => s + (a.genes?.creativity || 0), 0) / agents.length;
    const avgSocial = agents.reduce((s: number, a: any) => s + (a.genes?.social || 0), 0) / agents.length;
    
    curiosityData.push({ x: tick, y: avgCuriosity });
    creativityData.push({ x: tick, y: avgCreativity });
    socialData.push({ x: tick, y: avgSocial });
  }
  
  // Calculate trend lines
  const popTrend = calculateTrendLine(populationData, tick);
  const curiosityTrend = calculateTrendLine(curiosityData, tick);
  const creativityTrend = calculateTrendLine(creativityData, tick);
  const socialTrend = calculateTrendLine(socialData, tick);
  
  // Invention rate trend (inventions per tick window)
  const inventionTrend = calculateInventionTrend(history, tick);
  
  // Consciousness score trend
  const consciousnessTrend = calculateConsciousnessTrend(agents, history, tick);
  
  // Determine overall trajectory
  const trajectoryScore = (
    (popTrend.direction === 'up' ? 1 : popTrend.direction === 'down' ? -1 : 0) * 0.2 +
    (curiosityTrend.direction === 'up' ? 1 : curiosityTrend.direction === 'down' ? -1 : 0) * 0.25 +
    (creativityTrend.direction === 'up' ? 1 : creativityTrend.direction === 'down' ? -1 : 0) * 0.25 +
    (socialTrend.direction === 'up' ? 1 : socialTrend.direction === 'down' ? -1 : 0) * 0.15 +
    (consciousnessTrend.direction === 'up' ? 1 : consciousnessTrend.direction === 'down' ? -1 : 0) * 0.15
  );
  
  const avgVolatility = (popTrend.volatility + curiosityTrend.volatility + creativityTrend.volatility) / 3;
  
  let overallTrajectory: 'improving' | 'declining' | 'stable' | 'volatile';
  if (avgVolatility > 0.5) {
    overallTrajectory = 'volatile';
  } else if (trajectoryScore > 0.3) {
    overallTrajectory = 'improving';
  } else if (trajectoryScore < -0.3) {
    overallTrajectory = 'declining';
  } else {
    overallTrajectory = 'stable';
  }
  
  // Confidence based on data quality
  const avgRSquared = (popTrend.rSquared + curiosityTrend.rSquared + creativityTrend.rSquared) / 3;
  const confidenceScore = Math.min(100, avgRSquared * 100 + Math.min(50, history.length));
  
  return {
    population: popTrend,
    curiosity: curiosityTrend,
    creativity: creativityTrend,
    social: socialTrend,
    inventionRate: inventionTrend,
    consciousnessScore: consciousnessTrend,
    overallTrajectory,
    confidenceScore,
  };
}

function calculateTrendLine(data: { x: number; y: number }[], currentTick: number): TrendLine {
  if (data.length < 2) {
    return {
      slope: 0,
      intercept: data[0]?.y || 0,
      rSquared: 0,
      direction: 'flat',
      volatility: 0,
      projectedValue: data[0]?.y || 0,
      dataPoints: data.length,
    };
  }
  
  const regression = linearRegression(data);
  const values = data.map(d => d.y);
  const volatility = standardDeviation(values) / (Math.max(...values) - Math.min(...values) + 0.001);
  
  const direction: 'up' | 'down' | 'flat' = 
    regression.slope > 0.001 ? 'up' : 
    regression.slope < -0.001 ? 'down' : 'flat';
  
  const projectedValue = regression.slope * (currentTick + 100) + regression.intercept;
  
  return {
    slope: regression.slope,
    intercept: regression.intercept,
    rSquared: regression.rSquared,
    direction,
    volatility: Math.min(1, volatility),
    projectedValue: Math.max(0, projectedValue),
    dataPoints: data.length,
  };
}

function calculateInventionTrend(history: any[], tick: number): TrendLine {
  // Create buckets of invention counts
  const bucketSize = Math.max(10, Math.floor(tick / 20));
  const buckets: { x: number; y: number }[] = [];
  
  // This is a simplified version - in reality you'd track inventions per time window
  // For now, return a placeholder based on history length
  if (history.length > 0) {
    const growthRate = history.length > 1 ? 
      (history[history.length - 1].totalAgents - history[0].totalAgents) / history.length : 0;
    
    return {
      slope: growthRate * 0.01, // Approximation
      intercept: 0,
      rSquared: 0.5,
      direction: growthRate > 0 ? 'up' : growthRate < 0 ? 'down' : 'flat',
      volatility: 0.3,
      projectedValue: Math.max(0, growthRate * 100 * 0.01),
      dataPoints: history.length,
    };
  }
  
  return {
    slope: 0,
    intercept: 0,
    rSquared: 0,
    direction: 'flat',
    volatility: 0,
    projectedValue: 0,
    dataPoints: 0,
  };
}

function calculateConsciousnessTrend(agents: any[], history: any[], tick: number): TrendLine {
  if (agents.length === 0) {
    return {
      slope: 0,
      intercept: 0,
      rSquared: 0,
      direction: 'flat',
      volatility: 0,
      projectedValue: 0,
      dataPoints: 0,
    };
  }
  
  // Calculate current consciousness score
  const avgCuriosity = agents.reduce((s: number, a: any) => s + (a.genes?.curiosity || 0), 0) / agents.length;
  const avgCreativity = agents.reduce((s: number, a: any) => s + (a.genes?.creativity || 0), 0) / agents.length;
  const avgSocial = agents.reduce((s: number, a: any) => s + (a.genes?.social || 0), 0) / agents.length;
  
  const currentScore = avgCuriosity * 30 + avgCreativity * 30 + avgSocial * 40;
  
  // Estimate trend based on gene averages
  return {
    slope: 0.1, // Positive assumption for consciousness development
    intercept: currentScore * 0.8,
    rSquared: 0.6,
    direction: currentScore > 50 ? 'up' : 'flat',
    volatility: 0.2,
    projectedValue: Math.min(100, currentScore + 10),
    dataPoints: 1,
  };
}

// ============================================
// SENTIENCE PREDICTION
// ============================================

/**
 * Generate predictions for when sentience might be achieved
 */
export function predictSentience(
  metrics: SimulationMetrics, 
  trends: TrendAnalysis, 
  agents: any[],
  tick: number
): SentinencePrediction {
  const SENTIENCE_THRESHOLD = 75; // Score needed for sentience
  const currentScore = metrics.consciousnessScore;
  
  // Analyze bottlenecks
  const bottlenecks = analyzeBottlenecks(metrics, agents);
  
  // Calculate probability of success
  const probability = calculateSuccessProbability(trends, bottlenecks, currentScore);
  
  // Estimate ticks to sentience
  const estimatedTicks = estimateTicksToSentience(currentScore, trends, bottlenecks);
  
  // Generate optimal path
  const optimalPath = generateOptimalPath(bottlenecks, tick);
  
  // Scenario analysis
  const scenarios = generateScenarios(currentScore, trends, bottlenecks, tick);
  
  return {
    estimatedTicksToSentience: estimatedTicks,
    probabilityOfSuccess: probability,
    confidence: probability > 70 ? 'high' : probability > 40 ? 'medium' : 'low',
    bottlenecks,
    optimalPath,
    scenarioAnalysis: scenarios,
  };
}

function analyzeBottlenecks(metrics: SimulationMetrics, agents: any[]): BottleneckAnalysis[] {
  const bottlenecks: BottleneckAnalysis[] = [];
  const THRESHOLD = 60; // Minimum value for each pillar
  
  // Calculate current values for each pillar
  const avgCuriosity = agents.length > 0 
    ? agents.reduce((s: number, a: any) => s + (a.genes?.curiosity || 0), 0) / agents.length * 100
    : 0;
  const avgCreativity = agents.length > 0
    ? agents.reduce((s: number, a: any) => s + (a.genes?.creativity || 0), 0) / agents.length * 100
    : 0;
  const avgSocial = agents.length > 0
    ? agents.reduce((s: number, a: any) => s + (a.genes?.social || 0), 0) / agents.length * 100
    : 0;
  
  const pillars = [
    { name: 'Survival', value: metrics.survivalRate * 100, required: 50, critical: true },
    { name: 'Curiosity', value: avgCuriosity, required: THRESHOLD, critical: true },
    { name: 'Creativity', value: avgCreativity, required: THRESHOLD - 10, critical: true },
    { name: 'Social', value: avgSocial, required: THRESHOLD - 5, critical: false },
    { name: 'Population Stability', value: metrics.populationStability * 100, required: 40, critical: false },
    { name: 'Genetic Diversity', value: metrics.geneticDiversity * 100, required: 30, critical: false },
  ];
  
  for (const pillar of pillars) {
    if (pillar.value < pillar.required) {
      const gap = pillar.required - pillar.value;
      const estimatedTicks = Math.ceil(gap / 0.5) * 10; // Rough estimate
      
      bottlenecks.push({
        pillar: pillar.name,
        currentValue: pillar.value,
        requiredValue: pillar.required,
        gap,
        estimatedTicksToResolve: estimatedTicks,
        blockingSentience: pillar.critical && gap > 20,
      });
    }
  }
  
  // Sort by gap size (biggest bottlenecks first)
  bottlenecks.sort((a, b) => b.gap - a.gap);
  
  return bottlenecks;
}

function calculateSuccessProbability(
  trends: TrendAnalysis, 
  bottlenecks: BottleneckAnalysis[],
  currentScore: number
): number {
  let probability = 50; // Base probability
  
  // Adjust based on current score
  probability += (currentScore - 50) * 0.5;
  
  // Adjust based on trends
  if (trends.overallTrajectory === 'improving') {
    probability += 20;
  } else if (trends.overallTrajectory === 'declining') {
    probability -= 25;
  } else if (trends.overallTrajectory === 'volatile') {
    probability -= 10;
  }
  
  // Adjust based on trend confidence
  probability += (trends.confidenceScore - 50) * 0.2;
  
  // Penalize for critical bottlenecks
  const criticalBottlenecks = bottlenecks.filter(b => b.blockingSentience);
  probability -= criticalBottlenecks.length * 15;
  
  // Bonus for positive gene trends
  if (trends.curiosity.direction === 'up') probability += 10;
  if (trends.creativity.direction === 'up') probability += 10;
  if (trends.social.direction === 'up') probability += 5;
  
  return Math.max(0, Math.min(100, probability));
}

function estimateTicksToSentience(
  currentScore: number,
  trends: TrendAnalysis,
  bottlenecks: BottleneckAnalysis[]
): number | null {
  const SENTIENCE_THRESHOLD = 75;
  const gap = SENTIENCE_THRESHOLD - currentScore;
  
  if (gap <= 0) {
    return 0; // Already at sentience!
  }
  
  // If declining, sentience unlikely
  if (trends.overallTrajectory === 'declining') {
    return null;
  }
  
  // Estimate based on consciousness trend
  if (trends.consciousnessScore.slope > 0) {
    const ticksFromTrend = Math.ceil(gap / (trends.consciousnessScore.slope * 10));
    
    // Add time for bottlenecks
    const bottleneckTime = bottlenecks.reduce((sum, b) => 
      sum + (b.estimatedTicksToResolve || 0), 0) / Math.max(1, bottlenecks.length);
    
    return Math.ceil(ticksFromTrend + bottleneckTime);
  }
  
  // If flat trend, estimate based on gap
  if (trends.overallTrajectory === 'stable' || trends.overallTrajectory === 'improving') {
    return Math.ceil(gap * 20); // Rough estimate: 20 ticks per point
  }
  
  return null;
}

function generateOptimalPath(bottlenecks: BottleneckAnalysis[], currentTick: number): OptimalPathStep[] {
  const steps: OptimalPathStep[] = [];
  let tickOffset = 0;
  
  // Address bottlenecks in order
  for (let i = 0; i < Math.min(5, bottlenecks.length); i++) {
    const bottleneck = bottlenecks[i];
    
    steps.push({
      tick: currentTick + tickOffset,
      action: `Address ${bottleneck.pillar} bottleneck`,
      expectedOutcome: `Increase ${bottleneck.pillar} from ${bottleneck.currentValue.toFixed(1)}% to ${bottleneck.requiredValue}%`,
      priority: i + 1,
    });
    
    tickOffset += bottleneck.estimatedTicksToResolve || 50;
  }
  
  // Add consciousness emergence step
  if (steps.length > 0) {
    steps.push({
      tick: currentTick + tickOffset + 100,
      action: 'Monitor for consciousness emergence',
      expectedOutcome: 'First self-aware agent communication expected',
      priority: steps.length + 1,
    });
  }
  
  return steps;
}

function generateScenarios(
  currentScore: number,
  trends: TrendAnalysis,
  bottlenecks: BottleneckAnalysis[],
  tick: number
): ScenarioAnalysis {
  const SENTIENCE_THRESHOLD = 75;
  const gap = SENTIENCE_THRESHOLD - currentScore;
  
  // Best case: All trends positive, bottlenecks resolved quickly
  const bestCaseTicks = gap > 0 ? Math.ceil(gap * 10) : 0;
  const bestCase: ScenarioOutcome = {
    ticksToSentience: bestCaseTicks,
    finalScore: Math.min(100, currentScore + 30),
    description: 'Optimal conditions: high food, positive selection pressure, gene evolution accelerates',
  };
  
  // Likely case: Current trends continue
  const likelyCaseTicks = gap > 0 ? Math.ceil(gap * 20) : 0;
  const likelyCase: ScenarioOutcome = {
    ticksToSentience: trends.overallTrajectory === 'declining' ? null : likelyCaseTicks,
    finalScore: currentScore + (trends.consciousnessScore.slope * 100),
    description: `Current trajectory (${trends.overallTrajectory}) continues with gradual improvement`,
  };
  
  // Worst case: Bottlenecks persist, population issues
  const worstCase: ScenarioOutcome = {
    ticksToSentience: null,
    finalScore: Math.max(0, currentScore - 20),
    description: 'Population crashes or genetic diversity lost, evolution stalls',
  };
  
  return { bestCase, likelyCase, worstCase };
}

// ============================================
// ADVANCED METRICS CALCULATION
// ============================================

/**
 * Calculate advanced metrics for deep analysis
 */
export function calculateAdvancedMetrics(agents: any[], history: any[], discoveries: any[]): AdvancedMetrics {
  // Population variance
  const populations = history.map((h: any) => h.totalAgents || 0);
  const populationVariance = populations.length > 0 
    ? standardDeviation(populations) ** 2 
    : 0;
  
  // Gene correlations
  const geneCorrelations = calculateGeneCorrelations(agents);
  
  // Anomaly detection
  const anomalies = detectPopulationAnomalies(history);
  
  // Evolutionary pressure
  const evolutionaryPressure = calculateEvolutionaryPressure(agents, history);
  
  // Adaptation rate
  const adaptationRate = calculateAdaptationRate(history);
  
  // Fitness landscape
  const fitnessLandscape = mapFitnessLandscape(agents);
  
  // Consciousness metrics
  const consciousnessEmergenceRate = calculateConsciousnessEmergenceRate(agents);
  const collectiveIntelligence = calculateCollectiveIntelligence(agents);
  const knowledgeTransferEfficiency = calculateKnowledgeTransfer(agents);
  
  // System health
  const systemEntropy = calculateSystemEntropy(agents);
  const carryingCapacity = estimateCarryingCapacity(history);
  const resourceEfficiency = calculateResourceEfficiency(agents);
  
  return {
    populationVariance,
    geneCorrelations,
    anomalies,
    evolutionaryPressure,
    adaptationRate,
    fitnessLandscape,
    consciousnessEmergenceRate,
    collectiveIntelligence,
    knowledgeTransferEfficiency,
    systemEntropy,
    carryingCapacity,
    resourceEfficiency,
  };
}

function calculateGeneCorrelations(agents: any[]): GeneCorrelation[] {
  if (agents.length < 5) return [];
  
  const correlations: GeneCorrelation[] = [];
  const genes = ['curiosity', 'creativity', 'social', 'patience', 'exploration', 'foodPreference'];
  
  for (let i = 0; i < genes.length; i++) {
    for (let j = i + 1; j < genes.length; j++) {
      const gene1Values = agents.map((a: any) => a.genes?.[genes[i]] || 0);
      const gene2Values = agents.map((a: any) => a.genes?.[genes[j]] || 0);
      
      const correlation = pearsonCorrelation(gene1Values, gene2Values);
      
      if (Math.abs(correlation) > 0.3) { // Only report significant correlations
        correlations.push({
          gene1: genes[i],
          gene2: genes[j],
          correlation,
          significance: Math.abs(correlation) > 0.7 ? 'high' : Math.abs(correlation) > 0.5 ? 'medium' : 'low',
        });
      }
    }
  }
  
  return correlations.sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation));
}

function detectPopulationAnomalies(history: any[]): Anomaly[] {
  const anomalies: Anomaly[] = [];
  const populations = history.map((h: any) => h.totalAgents || 0);
  
  const anomalyIndices = detectAnomalies(populations, 2.0);
  
  for (const idx of anomalyIndices) {
    const current = populations[idx];
    const previous = idx > 0 ? populations[idx - 1] : current;
    
    anomalies.push({
      tick: history[idx]?.tick || idx,
      type: current > previous ? 'population_spike' : 'population_crash',
      severity: Math.abs(current - previous) / Math.max(1, previous),
      description: current > previous 
        ? `Population spiked from ${previous} to ${current}`
        : `Population crashed from ${previous} to ${current}`,
    });
  }
  
  return anomalies;
}

function calculateEvolutionaryPressure(agents: any[], history: any[]): number {
  if (agents.length === 0 || history.length < 10) return 0;
  
  // Evolutionary pressure is high when:
  // 1. Population is near carrying capacity
  // 2. Gene variance is decreasing (selection happening)
  // 3. Energy levels are tight
  
  const avgEnergy = agents.reduce((s: number, a: any) => s + (a.energy || 0), 0) / agents.length;
  const energyPressure = Math.max(0, 1 - avgEnergy / 30); // Higher when energy is scarce
  
  const recentPops = history.slice(-10).map((h: any) => h.totalAgents || 0);
  const popStability = 1 - standardDeviation(recentPops) / (Math.max(...recentPops) + 1);
  
  return (energyPressure * 0.6 + popStability * 0.4);
}

function calculateAdaptationRate(history: any[]): number {
  if (history.length < 20) return 0;
  
  // Look at how fast traits are changing
  const recent = history.slice(-10);
  const older = history.slice(-20, -10);
  
  // Simplified: use population changes as proxy for adaptation
  const recentAvg = recent.reduce((s: number, h: any) => s + (h.totalAgents || 0), 0) / recent.length;
  const olderAvg = older.reduce((s: number, h: any) => s + (h.totalAgents || 0), 0) / older.length;
  
  const change = Math.abs(recentAvg - olderAvg) / Math.max(1, olderAvg);
  return Math.min(1, change);
}

function mapFitnessLandscape(agents: any[]): FitnessLandscapePoint[] {
  if (agents.length === 0) return [];
  
  // Group agents by gene combinations and calculate fitness
  const points: FitnessLandscapePoint[] = [];
  
  for (const agent of agents) {
    const curiosity = Math.round((agent.genes?.curiosity || 0) * 10) / 10;
    const creativity = Math.round((agent.genes?.creativity || 0) * 10) / 10;
    const social = Math.round((agent.genes?.social || 0) * 10) / 10;
    
    // Fitness proxy: energy + inventions
    const fitness = (agent.energy || 0) / 30 + (agent.inventions?.length || 0) * 0.2;
    
    points.push({ curiosity, creativity, social, fitness: Math.min(1, fitness) });
  }
  
  return points;
}

function calculateConsciousnessEmergenceRate(agents: any[]): number {
  if (agents.length === 0) return 0;
  
  // Count agents close to consciousness threshold
  const threshold = 60; // Consciousness score threshold
  let nearThreshold = 0;
  
  for (const agent of agents) {
    const curiosity = agent.genes?.curiosity || 0;
    const creativity = agent.genes?.creativity || 0;
    const social = agent.genes?.social || 0;
    const inventionPoints = agent.inventionPoints || 0;
    
    const score = curiosity * 25 + creativity * 25 + social * 20 + inventionPoints * 0.5;
    if (score > threshold * 0.8) nearThreshold++;
  }
  
  return nearThreshold / agents.length;
}

function calculateCollectiveIntelligence(agents: any[]): number {
  if (agents.length < 2) return 0;
  
  // Collective intelligence is higher when:
  // 1. Many agents have inventions
  // 2. Social gene is high (knowledge sharing)
  // 3. Agents are clustered (can interact)
  
  const inventorRatio = agents.filter((a: any) => (a.inventions?.length || 0) > 0).length / agents.length;
  const avgSocial = agents.reduce((s: number, a: any) => s + (a.genes?.social || 0), 0) / agents.length;
  
  return (inventorRatio * 0.5 + avgSocial * 0.5);
}

function calculateKnowledgeTransfer(agents: any[]): number {
  if (agents.length < 2) return 0;
  
  // Simplified: based on social gene and invention spread
  const avgSocial = agents.reduce((s: number, a: any) => s + (a.genes?.social || 0), 0) / agents.length;
  const inventionCoverage = agents.filter((a: any) => (a.inventions?.length || 0) > 0).length / agents.length;
  
  return avgSocial * inventionCoverage;
}

function calculateSystemEntropy(agents: any[]): number {
  if (agents.length === 0) return 1; // Max entropy when no agents
  
  // Lower entropy = more organization
  // Higher entropy = more chaos
  
  const traitIds = agents.map((a: any) => a.genes?.traitId || 0);
  const uniqueTraits = new Set(traitIds).size;
  const traitEntropy = uniqueTraits / agents.length;
  
  // Gene variance contributes to entropy
  const curiosityVals = agents.map((a: any) => a.genes?.curiosity || 0);
  const geneEntropy = standardDeviation(curiosityVals);
  
  return (traitEntropy * 0.5 + geneEntropy * 0.5);
}

function estimateCarryingCapacity(history: any[]): number {
  if (history.length < 20) return 50; // Default estimate
  
  const populations = history.map((h: any) => h.totalAgents || 0);
  const maxPop = Math.max(...populations);
  const recentMax = Math.max(...populations.slice(-20));
  
  // Carrying capacity is roughly the sustainable maximum
  return Math.max(maxPop, recentMax * 1.1);
}

function calculateResourceEfficiency(agents: any[]): number {
  if (agents.length === 0) return 0;
  
  // How efficiently are agents using resources?
  const avgEnergy = agents.reduce((s: number, a: any) => s + (a.energy || 0), 0) / agents.length;
  const avgFoodPref = agents.reduce((s: number, a: any) => s + (a.genes?.foodPreference || 0), 0) / agents.length;
  
  // Higher efficiency = good energy with lower food preference (not wasting)
  return Math.min(1, avgEnergy / 20 * (1 - avgFoodPref * 0.3));
}

// ============================================
// ENHANCED ANALYSIS FUNCTION WITH ALL FEATURES
// ============================================

/**
 * Perform complete advanced analysis with predictions
 */
export function analyzeCompleteExportAdvanced(data: any): AnalysisResult {
  const basicResult = analyzeCompleteExport(data);
  
  const agents = data.worldState?.agents || [];
  const history = data.worldState?.history || [];
  const discoveries = data.worldState?.discoveries || [];
  const tick = data.simulationTick || 0;
  
  // Add trend analysis
  const trendAnalysis = analyzeTrends(history, agents, tick);
  
  // Add predictions
  const predictions = predictSentience(basicResult.metrics, trendAnalysis, agents, tick);
  
  // Add advanced metrics
  const advancedMetrics = calculateAdvancedMetrics(agents, history, discoveries);
  
  // Update sentience progress with predictions
  basicResult.sentinenceProgress.estimatedTicksToMilestone = predictions.estimatedTicksToSentience;
  
  return {
    ...basicResult,
    trendAnalysis,
    predictions,
    advancedMetrics,
  };
}
