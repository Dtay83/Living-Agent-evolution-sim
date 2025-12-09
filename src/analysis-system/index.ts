/**
 * DATA ANALYSIS SYSTEM
 * 
 * Analyzes exported simulation data to identify patterns and
 * generate recommendations for achieving agent sentience.
 */

// Types for analysis
export interface AnalysisResult {
  timestamp: string;
  dataSource: string;
  patterns: Pattern[];
  recommendations: Recommendation[];
  metrics: SimulationMetrics;
  sentinenceProgress: SentinenceProgress;
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
