/**
 * Scientific Discovery Logger - EXPANDED EDITION
 * Tracks and organizes discoveries by era for historical analysis
 * 
 * Now tracks:
 * - All 17 physics categories
 * - Collaborative discoveries
 * - Curiosity-driven metrics
 * - Discovery momentum
 * - Category breakthroughs
 * - Agent contribution rankings
 * - Scientific acceleration
 */

import type { ScientificEra, ScientificDiscovery, ProgressionMetrics } from './eras';
import type { ScienceState, CombinedScienceBonuses } from './index';
import type { PhysicsConcept } from './physics';

/**
 * All physics categories for tracking
 */
const PHYSICS_CATEGORIES = [
  'mechanics', 'thermodynamics', 'electromagnetism', 'quantum', 'relativity',
  'unified', 'cosmology', 'particle', 'plasma', 'condensed_matter', 'exotic',
  'transcendent', 'nuclear', 'astrophysics', 'biophysics', 'information_physics',
  'chaos_complexity'
] as const;

type PhysicsCategory = typeof PHYSICS_CATEGORIES[number];

/**
 * Extended discovery with additional tracking fields
 */
export interface ExtendedDiscovery extends ScientificDiscovery {
  wasCollaborative?: boolean;
  collaboratorCount?: number;
  curiosityLevel?: number;        // Curiosity level of discoverer at time of discovery
  momentumAtDiscovery?: number;   // Discovery momentum when this was found
  complexityRank?: number;        // How complex relative to era
  breakthroughType?: 'normal' | 'category_first' | 'era_defining' | 'transcendent';
}

/**
 * Category-specific metrics
 */
export interface CategoryMetrics {
  category: PhysicsCategory;
  totalDiscovered: number;
  firstDiscoveryTick?: number;
  lastDiscoveryTick?: number;
  averageComplexity: number;
  highestComplexity: number;
  collaborativeCount: number;
  topContributorId?: number;
  concepts: string[];           // IDs of discovered concepts in this category
}

/**
 * Agent contribution tracking
 */
export interface AgentContribution {
  agentId: number;
  totalDiscoveries: number;
  physicsDiscoveries: number;
  mathDiscoveries: number;
  collaborativeDiscoveries: number;
  soloDiscoveries: number;
  categoriesContributed: Set<string>;
  highestComplexityDiscovered: number;
  breakthroughsAchieved: number;
  firstDiscoveryTick?: number;
  lastDiscoveryTick?: number;
  averageCuriosityAtDiscovery: number;
  discoveryTicks: number[];     // Ticks when discoveries were made
}

/**
 * Enhanced era log with expanded metrics
 */
export interface EraLog {
  era: ScientificEra;
  discoveries: ExtendedDiscovery[];
  metrics: {
    startTick: number;
    endTick?: number;
    duration?: number;
    totalDiscoveries: number;
    physicsDiscovered: number;
    mathDiscovered: number;
    inventionsDiscovered: number;
    evolutionsDiscovered: number;
    averageDiscoveryRate: number;
    progressionSpeed: 'dormant' | 'slow' | 'moderate' | 'fast' | 'explosive' | 'singularity';
    // NEW expanded metrics
    collaborativeDiscoveries: number;
    soloDiscoveries: number;
    collaborationRate: number;
    peakMomentum: number;
    averageMomentum: number;
    categoryBreakthroughs: number;   // First discoveries in new categories
    transcendentDiscoveries: number;
    averageComplexity: number;
    highestComplexityReached: number;
    uniqueContributors: number;
    mostProductiveAgentId?: number;
    scientificAcceleration: number;
  };
  // Category breakdown
  categoryMetrics: Map<PhysicsCategory, CategoryMetrics>;
  // Agent contributions for this era
  agentContributions: Map<number, AgentContribution>;
  // Bonus snapshots at era transitions
  bonusSnapshotStart?: Partial<CombinedScienceBonuses>;
  bonusSnapshotEnd?: Partial<CombinedScienceBonuses>;
}

/**
 * Discovery milestone tracking
 */
export interface DiscoveryMilestone {
  id: string;
  name: string;
  description: string;
  achievedAt?: number;
  achievedBy?: number;
  requirement: {
    type: 'count' | 'category' | 'complexity' | 'rate' | 'collaboration' | 'era';
    target: number | string;
  };
}

/**
 * Global science statistics
 */
export interface GlobalScienceStats {
  totalPhysicsDiscovered: number;
  totalMathDiscovered: number;
  totalCollaborativeDiscoveries: number;
  categoriesUnlocked: number;
  totalCategories: number;
  highestComplexityEverReached: number;
  fastestEra: { level: number; duration: number } | null;
  slowestEra: { level: number; duration: number } | null;
  mostProductiveEra: { level: number; discoveries: number } | null;
  allTimeTopContributor: { agentId: number; discoveries: number } | null;
  discoveryDistribution: {
    physics: number;
    mathematics: number;
    inventions: number;
    evolutions: number;
  };
  categoryDistribution: Record<PhysicsCategory, number>;
  collaborationTrend: number[];  // Collaboration rate per era
  complexityTrend: number[];     // Average complexity per era
  momentumHistory: { tick: number; momentum: number }[];
}

/**
 * Enhanced science log
 */
export interface ScienceLog {
  eras: Map<number, EraLog>;
  allDiscoveries: ExtendedDiscovery[];
  globalMetrics: ProgressionMetrics;
  // NEW tracking
  globalStats: GlobalScienceStats;
  agentLifetimeContributions: Map<number, AgentContribution>;
  milestones: DiscoveryMilestone[];
  achievedMilestones: string[];
  categoryFirstDiscoveries: Map<PhysicsCategory, ExtendedDiscovery>;
  discoveryTimeline: { tick: number; discoveryId: string; category: string }[];
}

/**
 * Default milestones for tracking progress
 */
const DEFAULT_MILESTONES: DiscoveryMilestone[] = [
  { id: 'first_physics', name: 'First Steps', description: 'Make first physics discovery', requirement: { type: 'count', target: 1 } },
  { id: 'ten_physics', name: 'Curious Minds', description: 'Discover 10 physics concepts', requirement: { type: 'count', target: 10 } },
  { id: 'fifty_physics', name: 'Scientific Foundation', description: 'Discover 50 physics concepts', requirement: { type: 'count', target: 50 } },
  { id: 'hundred_physics', name: 'Knowledge Revolution', description: 'Discover 100 physics concepts', requirement: { type: 'count', target: 100 } },
  { id: 'all_basic_categories', name: 'Renaissance', description: 'Discover concepts in all basic categories', requirement: { type: 'category', target: 'basic' } },
  { id: 'quantum_unlock', name: 'Quantum Leap', description: 'Make first quantum physics discovery', requirement: { type: 'category', target: 'quantum' } },
  { id: 'relativity_unlock', name: 'Einsteinian', description: 'Make first relativity discovery', requirement: { type: 'category', target: 'relativity' } },
  { id: 'exotic_unlock', name: 'Beyond Known Physics', description: 'Make first exotic physics discovery', requirement: { type: 'category', target: 'exotic' } },
  { id: 'transcendent_unlock', name: 'Transcendence', description: 'Make first transcendent physics discovery', requirement: { type: 'category', target: 'transcendent' } },
  { id: 'complexity_10', name: 'Deep Understanding', description: 'Discover a concept with complexity 10+', requirement: { type: 'complexity', target: 10 } },
  { id: 'complexity_15', name: 'Profound Insight', description: 'Discover a concept with complexity 15+', requirement: { type: 'complexity', target: 15 } },
  { id: 'first_collab', name: 'Collaboration', description: 'Make first collaborative discovery', requirement: { type: 'collaboration', target: 1 } },
  { id: 'ten_collab', name: 'Research Team', description: 'Make 10 collaborative discoveries', requirement: { type: 'collaboration', target: 10 } },
  { id: 'era_5', name: 'Advanced Civilization', description: 'Reach Era 5', requirement: { type: 'era', target: 5 } },
  { id: 'era_10', name: 'Technological Singularity', description: 'Reach Era 10', requirement: { type: 'era', target: 10 } },
  { id: 'fast_era', name: 'Scientific Explosion', description: 'Complete an era with explosive progression', requirement: { type: 'rate', target: 0.5 } },
];

/**
 * Initialize science logging system
 */
export function initializeScienceLog(): ScienceLog {
  const categoryDistribution: Record<PhysicsCategory, number> = {} as Record<PhysicsCategory, number>;
  for (const cat of PHYSICS_CATEGORIES) {
    categoryDistribution[cat] = 0;
  }
  return {
    eras: new Map(),
    allDiscoveries: [],
    globalMetrics: {
      totalDiscoveries: 0,
      discoveriesPerEra: {},
      averageDiscoveryRate: 0,
      currentEraLevel: 0,
      ticksSinceLastEra: 0,
      scientificAcceleration: 0,
      // Extended metrics
      collaborativeDiscoveries: 0,
      categoryBreakthroughs: 0,
      peakMomentum: 1.0,
      averageComplexity: 0,
      physicsCategories: 0,
      mathCategories: 0,
      transcendentDiscoveries: 0
    },
    globalStats: {
      totalPhysicsDiscovered: 0,
      totalMathDiscovered: 0,
      totalCollaborativeDiscoveries: 0,
      categoriesUnlocked: 0,
      totalCategories: PHYSICS_CATEGORIES.length,
      highestComplexityEverReached: 0,
      fastestEra: null,
      slowestEra: null,
      mostProductiveEra: null,
      allTimeTopContributor: null,
      discoveryDistribution: {
        physics: 0,
        mathematics: 0,
        inventions: 0,
        evolutions: 0
      },
      categoryDistribution,
      collaborationTrend: [],
      complexityTrend: [],
      momentumHistory: []
    },
    agentLifetimeContributions: new Map(),
    milestones: [...DEFAULT_MILESTONES],
    achievedMilestones: [],
    categoryFirstDiscoveries: new Map(),
    discoveryTimeline: []
  };
}

/**
 * Initialize category metrics for an era
 */
function initializeCategoryMetrics(): Map<PhysicsCategory, CategoryMetrics> {
  const metrics = new Map<PhysicsCategory, CategoryMetrics>();
  
  for (const category of PHYSICS_CATEGORIES) {
    metrics.set(category, {
      category,
      totalDiscovered: 0,
      averageComplexity: 0,
      highestComplexity: 0,
      collaborativeCount: 0,
      concepts: []
    });
  }
  
  return metrics;
}

/**
 * Get or create agent contribution tracking
 */
function getOrCreateAgentContribution(
  contributions: Map<number, AgentContribution>,
  agentId: number
): AgentContribution {
  let contribution = contributions.get(agentId);
  
  if (!contribution) {
    contribution = {
      agentId,
      totalDiscoveries: 0,
      physicsDiscoveries: 0,
      mathDiscoveries: 0,
      collaborativeDiscoveries: 0,
      soloDiscoveries: 0,
      categoriesContributed: new Set(),
      highestComplexityDiscovered: 0,
      breakthroughsAchieved: 0,
      averageCuriosityAtDiscovery: 0,
      discoveryTicks: []
    };
    contributions.set(agentId, contribution);
  }
  
  return contribution;
}

/**
 * Determine breakthrough type for a discovery
 */
function determineBreakthroughType(
  discovery: ScientificDiscovery,
  concept: PhysicsConcept | undefined,
  log: ScienceLog,
  eraLog: EraLog
): ExtendedDiscovery['breakthroughType'] {
  if (!concept) return 'normal';
  
  // Check if this is first in category
  const category = concept.category as PhysicsCategory;
  if (!log.categoryFirstDiscoveries.has(category)) {
    return 'category_first';
  }
  
  // Check if transcendent
  if (concept.category === 'transcendent') {
    return 'transcendent';
  }
  
  // Check if era-defining (high complexity relative to era)
  const eraBaseComplexity = eraLog.era.level * 2;
  if (concept.complexity >= eraBaseComplexity + 5) {
    return 'era_defining';
  }
  
  return 'normal';
}

/**
 * Check and award milestones
 */
function checkMilestones(
  log: ScienceLog,
  discovery: ExtendedDiscovery,
  concept: PhysicsConcept | undefined,
  scienceState: ScienceState,
  tick: number
): string[] {
  const newlyAchieved: string[] = [];
  
  for (const milestone of log.milestones) {
    if (log.achievedMilestones.includes(milestone.id)) continue;
    
    let achieved = false;
    
    switch (milestone.requirement.type) {
      case 'count':
        achieved = log.globalStats.totalPhysicsDiscovered >= (milestone.requirement.target as number);
        break;
        
      case 'category':
        if (milestone.requirement.target === 'basic') {
          const basicCategories = ['mechanics', 'thermodynamics', 'electromagnetism'];
          achieved = basicCategories.every(cat => 
            log.categoryFirstDiscoveries.has(cat as PhysicsCategory)
          );
        } else {
          achieved = log.categoryFirstDiscoveries.has(milestone.requirement.target as PhysicsCategory);
        }
        break;
        
      case 'complexity':
        achieved = log.globalStats.highestComplexityEverReached >= (milestone.requirement.target as number);
        break;
        
      case 'collaboration':
        achieved = log.globalStats.totalCollaborativeDiscoveries >= (milestone.requirement.target as number);
        break;
        
      case 'era':
        achieved = scienceState.currentEra.level >= (milestone.requirement.target as number);
        break;
        
      case 'rate':
        // Check if any era achieved this rate
        for (const [, eraLog] of log.eras) {
          if (eraLog.metrics.averageDiscoveryRate >= (milestone.requirement.target as number)) {
            achieved = true;
            break;
          }
        }
        break;
    }
    
    if (achieved) {
      milestone.achievedAt = tick;
      milestone.achievedBy = discovery.discoveredBy;
      log.achievedMilestones.push(milestone.id);
      newlyAchieved.push(milestone.id);
    }
  }
  
  return newlyAchieved;
}

/**
 * Add discoveries to the log - ENHANCED
 */
export function logDiscoveries(
  log: ScienceLog,
  discoveries: ScientificDiscovery[],
  currentEra: ScientificEra,
  tick: number,
  scienceState: ScienceState,
  conceptLookup?: Map<string, PhysicsConcept>
): {
  log: ScienceLog;
  newMilestones: string[];
  categoryBreakthroughs: PhysicsCategory[];
} {
  const newMilestones: string[] = [];
  const categoryBreakthroughs: PhysicsCategory[] = [];
  
  // Get or create era log
  let eraLog = log.eras.get(currentEra.level);
  
  if (!eraLog) {
    eraLog = {
      era: currentEra,
      discoveries: [],
      metrics: {
        startTick: currentEra.startTick,
        totalDiscoveries: 0,
        physicsDiscovered: 0,
        mathDiscovered: 0,
        inventionsDiscovered: 0,
        evolutionsDiscovered: 0,
        averageDiscoveryRate: 0,
        progressionSpeed: 'dormant',
        collaborativeDiscoveries: 0,
        soloDiscoveries: 0,
        collaborationRate: 0,
        peakMomentum: scienceState.discoveryMomentum,
        averageMomentum: scienceState.discoveryMomentum,
        categoryBreakthroughs: 0,
        transcendentDiscoveries: 0,
        averageComplexity: 0,
        highestComplexityReached: 0,
        uniqueContributors: 0,
        scientificAcceleration: 0
      },
      categoryMetrics: initializeCategoryMetrics(),
      agentContributions: new Map()
    };
    log.eras.set(currentEra.level, eraLog);
  }
  
  // Process each discovery
  for (const discovery of discoveries) {
    // Look up the concept for additional info
    const conceptId = discovery.id.replace(/^(physics|math)_/, '').replace(/_\d+$/, '');
    const concept = conceptLookup?.get(conceptId);
    
    // Create extended discovery
    const extendedDiscovery: ExtendedDiscovery = {
      ...discovery,
      wasCollaborative: (discovery as ExtendedDiscovery).wasCollaborative || false,
      collaboratorCount: (discovery as ExtendedDiscovery).collaboratorCount || 0,
      momentumAtDiscovery: scienceState.discoveryMomentum,
      complexityRank: concept?.complexity || discovery.significance * 10,
      breakthroughType: determineBreakthroughType(discovery, concept, log, eraLog)
    };
    
    // Add to logs
    eraLog.discoveries.push(extendedDiscovery);
    log.allDiscoveries.push(extendedDiscovery);
    log.discoveryTimeline.push({
      tick,
      discoveryId: discovery.id,
      category: discovery.category
    });
    
    // Update basic metrics
    eraLog.metrics.totalDiscoveries++;
    
    // Category-specific updates
    switch (discovery.category) {
      case 'physics':
        eraLog.metrics.physicsDiscovered++;
        log.globalStats.totalPhysicsDiscovered++;
        log.globalStats.discoveryDistribution.physics++;
        
        // Update physics category metrics
        if (concept) {
          const category = concept.category as PhysicsCategory;
          const catMetrics = eraLog.categoryMetrics.get(category);
          
          if (catMetrics) {
            // Check for category first
            if (!log.categoryFirstDiscoveries.has(category)) {
              log.categoryFirstDiscoveries.set(category, extendedDiscovery);
              eraLog.metrics.categoryBreakthroughs++;
              categoryBreakthroughs.push(category);
              log.globalStats.categoriesUnlocked++;
            }
            
            catMetrics.totalDiscovered++;
            catMetrics.concepts.push(concept.id);
            catMetrics.highestComplexity = Math.max(catMetrics.highestComplexity, concept.complexity);
            
            if (!catMetrics.firstDiscoveryTick) {
              catMetrics.firstDiscoveryTick = tick;
            }
            catMetrics.lastDiscoveryTick = tick;
            
            // Update average complexity
            const totalComplexity = catMetrics.averageComplexity * (catMetrics.totalDiscovered - 1) + concept.complexity;
            catMetrics.averageComplexity = totalComplexity / catMetrics.totalDiscovered;
            
            if (extendedDiscovery.wasCollaborative) {
              catMetrics.collaborativeCount++;
            }
            
            // Update global category distribution
            log.globalStats.categoryDistribution[category]++;
          }
          
          // Update complexity tracking
          if (concept.complexity > log.globalStats.highestComplexityEverReached) {
            log.globalStats.highestComplexityEverReached = concept.complexity;
          }
          if (concept.complexity > eraLog.metrics.highestComplexityReached) {
            eraLog.metrics.highestComplexityReached = concept.complexity;
          }
          
          // Track transcendent discoveries
          if (concept.category === 'transcendent') {
            eraLog.metrics.transcendentDiscoveries++;
          }
        }
        break;
        
      case 'mathematics':
        eraLog.metrics.mathDiscovered++;
        log.globalStats.totalMathDiscovered++;
        log.globalStats.discoveryDistribution.mathematics++;
        break;
        
      case 'invention':
        eraLog.metrics.inventionsDiscovered++;
        log.globalStats.discoveryDistribution.inventions++;
        break;
        
      case 'evolution':
        eraLog.metrics.evolutionsDiscovered++;
        log.globalStats.discoveryDistribution.evolutions++;
        break;
    }
    
    // Collaboration tracking
    if (extendedDiscovery.wasCollaborative) {
      eraLog.metrics.collaborativeDiscoveries++;
      log.globalStats.totalCollaborativeDiscoveries++;
    } else {
      eraLog.metrics.soloDiscoveries++;
    }
    
    // Agent contribution tracking
    if (discovery.discoveredBy !== undefined) {
      // Era-level tracking
      const eraContribution = getOrCreateAgentContribution(
        eraLog.agentContributions, 
        discovery.discoveredBy
      );
      
      // Lifetime tracking
      const lifetimeContribution = getOrCreateAgentContribution(
        log.agentLifetimeContributions,
        discovery.discoveredBy
      );
      
      // Update both
      for (const contribution of [eraContribution, lifetimeContribution]) {
        contribution.totalDiscoveries++;
        contribution.discoveryTicks.push(tick);
        
        if (!contribution.firstDiscoveryTick) {
          contribution.firstDiscoveryTick = tick;
        }
        contribution.lastDiscoveryTick = tick;
        
        if (discovery.category === 'physics') {
          contribution.physicsDiscoveries++;
          if (concept) {
            contribution.categoriesContributed.add(concept.category);
            contribution.highestComplexityDiscovered = Math.max(
              contribution.highestComplexityDiscovered,
              concept.complexity
            );
          }
        } else if (discovery.category === 'mathematics') {
          contribution.mathDiscoveries++;
        }
        
        if (extendedDiscovery.wasCollaborative) {
          contribution.collaborativeDiscoveries++;
        } else {
          contribution.soloDiscoveries++;
        }
        
        if (extendedDiscovery.breakthroughType !== 'normal') {
          contribution.breakthroughsAchieved++;
        }
      }
    }
    
    // Update momentum tracking
    eraLog.metrics.peakMomentum = Math.max(
      eraLog.metrics.peakMomentum,
      scienceState.discoveryMomentum
    );
    
    // Check milestones
    const achieved = checkMilestones(log, extendedDiscovery, concept, scienceState, tick);
    newMilestones.push(...achieved);
  }
  
  // Recalculate era metrics
  const duration = tick - eraLog.metrics.startTick;
  if (duration > 0) {
    eraLog.metrics.averageDiscoveryRate = eraLog.metrics.totalDiscoveries / duration;
    
    // Update collaboration rate
    if (eraLog.metrics.totalDiscoveries > 0) {
      eraLog.metrics.collaborationRate = 
        eraLog.metrics.collaborativeDiscoveries / eraLog.metrics.totalDiscoveries;
    }
    
    // Determine progression speed
    const rate = eraLog.metrics.averageDiscoveryRate;
    if (rate > 1.0) {
      eraLog.metrics.progressionSpeed = 'singularity';
    } else if (rate > 0.5) {
      eraLog.metrics.progressionSpeed = 'explosive';
    } else if (rate > 0.2) {
      eraLog.metrics.progressionSpeed = 'fast';
    } else if (rate > 0.05) {
      eraLog.metrics.progressionSpeed = 'moderate';
    } else if (rate > 0.01) {
      eraLog.metrics.progressionSpeed = 'slow';
    } else {
      eraLog.metrics.progressionSpeed = 'dormant';
    }
  }
  
  // Update unique contributors
  eraLog.metrics.uniqueContributors = eraLog.agentContributions.size;
  
  // Find most productive agent
  let maxDiscoveries = 0;
  for (const [agentId, contribution] of eraLog.agentContributions) {
    if (contribution.totalDiscoveries > maxDiscoveries) {
      maxDiscoveries = contribution.totalDiscoveries;
      eraLog.metrics.mostProductiveAgentId = agentId;
    }
  }
  
  // Update average complexity
  if (eraLog.metrics.physicsDiscovered > 0) {
    let totalComplexity = 0;
    for (const disc of eraLog.discoveries) {
      if (disc.category === 'physics') {
        totalComplexity += disc.complexityRank || disc.significance * 10;
      }
    }
    eraLog.metrics.averageComplexity = totalComplexity / eraLog.metrics.physicsDiscovered;
  }
  
  // Update global all-time top contributor
  let topContributor = log.globalStats.allTimeTopContributor;
  for (const [agentId, contribution] of log.agentLifetimeContributions) {
    if (!topContributor || contribution.totalDiscoveries > topContributor.discoveries) {
      topContributor = { agentId, discoveries: contribution.totalDiscoveries };
    }
  }
  log.globalStats.allTimeTopContributor = topContributor;
  
  // Add momentum to history (sample every 100 ticks)
  if (tick % 100 === 0) {
    log.globalStats.momentumHistory.push({
      tick,
      momentum: scienceState.discoveryMomentum
    });
  }
  
  return { log, newMilestones, categoryBreakthroughs };
}

/**
 * Finalize an era when advancing - ENHANCED
 */
export function finalizeEra(
  log: ScienceLog,
  eraLevel: number,
  endTick: number,
  scienceState: ScienceState
): ScienceLog {
  const eraLog = log.eras.get(eraLevel);
  
  if (eraLog) {
    eraLog.metrics.endTick = endTick;
    eraLog.metrics.duration = endTick - eraLog.metrics.startTick;
    
    // Capture end-of-era bonuses
    // (Would need getScienceBonuses imported for full implementation)
    
    // Update global era statistics
    const duration = eraLog.metrics.duration;
    const discoveries = eraLog.metrics.totalDiscoveries;
    
    // Track fastest era
    if (!log.globalStats.fastestEra || 
        (duration > 0 && duration < log.globalStats.fastestEra.duration)) {
      log.globalStats.fastestEra = { level: eraLevel, duration };
    }
    
    // Track slowest era
    if (!log.globalStats.slowestEra || duration > log.globalStats.slowestEra.duration) {
      log.globalStats.slowestEra = { level: eraLevel, duration };
    }
    
    // Track most productive era
    if (!log.globalStats.mostProductiveEra || 
        discoveries > log.globalStats.mostProductiveEra.discoveries) {
      log.globalStats.mostProductiveEra = { level: eraLevel, discoveries };
    }
    
    // Add to trends
    log.globalStats.collaborationTrend.push(eraLog.metrics.collaborationRate);
    log.globalStats.complexityTrend.push(eraLog.metrics.averageComplexity);
  }
  
  return log;
}

/**
 * Generate a detailed progress report - ENHANCED
 */
export function generateProgressReport(log: ScienceLog, scienceState: ScienceState): string {
  const lines: string[] = [];
  
  lines.push('═'.repeat(70));
  lines.push('                    SCIENTIFIC PROGRESS REPORT');
  lines.push('═'.repeat(70));
  lines.push('');
  
  // Current state
  lines.push('CURRENT STATE');
  lines.push('─'.repeat(70));
  lines.push(`Era: ${scienceState.currentEra.name} (Level ${scienceState.currentEra.level})`);
  lines.push(`Discovery Momentum: ${scienceState.discoveryMomentum.toFixed(2)}x`);
  lines.push('');
  
  // Discovery totals
  lines.push('DISCOVERY TOTALS');
  lines.push('─'.repeat(70));
  lines.push(`Total Discoveries: ${log.allDiscoveries.length}`);
  lines.push(`  • Physics: ${log.globalStats.totalPhysicsDiscovered}`);
  lines.push(`  • Mathematics: ${log.globalStats.totalMathDiscovered}`);
  lines.push(`  • Inventions: ${log.globalStats.discoveryDistribution.inventions}`);
  lines.push(`  • Evolutions: ${log.globalStats.discoveryDistribution.evolutions}`);
  lines.push('');
  lines.push(`Collaborative Discoveries: ${log.globalStats.totalCollaborativeDiscoveries}`);
  lines.push(`Categories Unlocked: ${log.globalStats.categoriesUnlocked}/${log.globalStats.totalCategories}`);
  lines.push(`Highest Complexity Reached: ${log.globalStats.highestComplexityEverReached}`);
  lines.push('');
  
  // Category breakdown
  lines.push('PHYSICS CATEGORY BREAKDOWN');
  lines.push('─'.repeat(70));
  
  const categoryEntries = Array.from(log.categoryFirstDiscoveries.entries())
    .sort((a, b) => {
      const aCount = log.globalStats.categoryDistribution[a[0]] || 0;
      const bCount = log.globalStats.categoryDistribution[b[0]] || 0;
      return bCount - aCount;
    });
  
  for (const [category, firstDiscovery] of categoryEntries) {
    const count = log.globalStats.categoryDistribution[category] || 0;
    const firstTick = firstDiscovery.discoveredAt;
    lines.push(`${category.toUpperCase().padEnd(20)} ${String(count).padStart(4)} discoveries (first at tick ${firstTick})`);
  }
  
  // Show locked categories
  const lockedCategories = PHYSICS_CATEGORIES.filter(cat => !log.categoryFirstDiscoveries.has(cat));
  if (lockedCategories.length > 0) {
    lines.push('');
    lines.push(`Locked Categories: ${lockedCategories.join(', ')}`);
  }
  lines.push('');
  
  // Era history
  lines.push('ERA HISTORY');
  lines.push('─'.repeat(70));
  
  const sortedEras = Array.from(log.eras.values()).sort((a, b) => a.era.level - b.era.level);
  
  for (const eraLog of sortedEras) {
    const status = eraLog.metrics.endTick ? '✓' : '▶';
    const duration = eraLog.metrics.duration ? `${eraLog.metrics.duration} ticks` : '(ongoing)';
    
    lines.push(`${status} ${eraLog.era.name} (Level ${eraLog.era.level})`);
    lines.push(`    Duration: ${duration}`);
    lines.push(`    Discoveries: ${eraLog.metrics.totalDiscoveries} (${eraLog.metrics.physicsDiscovered} physics, ${eraLog.metrics.mathDiscovered} math)`);
    lines.push(`    Collaboration: ${(eraLog.metrics.collaborationRate * 100).toFixed(1)}% | Contributors: ${eraLog.metrics.uniqueContributors}`);
    lines.push(`    Rate: ${eraLog.metrics.averageDiscoveryRate.toFixed(4)}/tick | Speed: ${eraLog.metrics.progressionSpeed.toUpperCase()}`);
    lines.push(`    Avg Complexity: ${eraLog.metrics.averageComplexity.toFixed(1)} | Peak: ${eraLog.metrics.highestComplexityReached}`);
    if (eraLog.metrics.categoryBreakthroughs > 0) {
      lines.push(`    Category Breakthroughs: ${eraLog.metrics.categoryBreakthroughs}`);
    }
    if (eraLog.metrics.transcendentDiscoveries > 0) {
      lines.push(`    ★ Transcendent Discoveries: ${eraLog.metrics.transcendentDiscoveries}`);
    }
    lines.push('');
  }
  
  // Milestones
  lines.push('MILESTONES');
  lines.push('─'.repeat(70));
  
  const achieved = log.milestones.filter(m => log.achievedMilestones.includes(m.id));
  const pending = log.milestones.filter(m => !log.achievedMilestones.includes(m.id));
  
  if (achieved.length > 0) {
    lines.push('Achieved:');
    for (const milestone of achieved) {
      lines.push(`  ✓ ${milestone.name}: ${milestone.description} (tick ${milestone.achievedAt})`);
    }
  }
  
  if (pending.length > 0) {
    lines.push('');
    lines.push('Pending:');
    for (const milestone of pending.slice(0, 5)) {
      lines.push(`  ○ ${milestone.name}: ${milestone.description}`);
    }
    if (pending.length > 5) {
      lines.push(`  ... and ${pending.length - 5} more`);
    }
  }
  lines.push('');
  
  // Top contributors
  lines.push('TOP CONTRIBUTORS (All-Time)');
  lines.push('─'.repeat(70));
  
  const topContributors = Array.from(log.agentLifetimeContributions.values())
    .sort((a, b) => b.totalDiscoveries - a.totalDiscoveries)
    .slice(0, 5);
  
  for (let i = 0; i < topContributors.length; i++) {
    const contrib = topContributors[i];
    lines.push(`${i + 1}. Agent #${contrib.agentId}: ${contrib.totalDiscoveries} discoveries`);
    lines.push(`     Physics: ${contrib.physicsDiscoveries} | Math: ${contrib.mathDiscoveries} | Collab: ${contrib.collaborativeDiscoveries}`);
    lines.push(`     Categories: ${contrib.categoriesContributed.size} | Breakthroughs: ${contrib.breakthroughsAchieved}`);
  }
  
  lines.push('');
  lines.push('═'.repeat(70));
  
  return lines.join('\n');
}

/**
 * Generate a compact summary for UI display
 */
export function generateCompactSummary(log: ScienceLog, scienceState: ScienceState): {
  era: string;
  eraLevel: number;
  totalDiscoveries: number;
  physicsCount: number;
  mathCount: number;
  categoriesUnlocked: number;
  momentum: number;
  progressionSpeed: string;
  topMilestone: string | null;
  recentBreakthroughs: string[];
} {
  const currentEraLog = log.eras.get(scienceState.currentEra.level);
  
  // Get recent category breakthroughs (last 5)
  const recentBreakthroughs = Array.from(log.categoryFirstDiscoveries.entries())
    .sort((a, b) => (b[1].discoveredAt || 0) - (a[1].discoveredAt || 0))
    .slice(0, 5)
    .map(([cat]) => cat);
  
  // Get most recent pending milestone
  const pendingMilestones = log.milestones.filter(m => !log.achievedMilestones.includes(m.id));
  const topMilestone = pendingMilestones.length > 0 ? pendingMilestones[0].name : null;
  
  return {
    era: scienceState.currentEra.name,
    eraLevel: scienceState.currentEra.level,
    totalDiscoveries: log.allDiscoveries.length,
    physicsCount: log.globalStats.totalPhysicsDiscovered,
    mathCount: log.globalStats.totalMathDiscovered,
    categoriesUnlocked: log.globalStats.categoriesUnlocked,
    momentum: scienceState.discoveryMomentum,
    progressionSpeed: currentEraLog?.metrics.progressionSpeed || 'dormant',
    topMilestone,
    recentBreakthroughs
  };
}

/**
 * Export science log to JSON for persistence - ENHANCED
 */
export function exportScienceLog(log: ScienceLog): string {
  const exportData = {
    eras: Array.from(log.eras.entries()).map(([level, eraLog]) => ({
      level,
      era: eraLog.era,
      discoveries: eraLog.discoveries,
      metrics: eraLog.metrics,
      categoryMetrics: Array.from(eraLog.categoryMetrics.entries()),
      agentContributions: Array.from(eraLog.agentContributions.entries()).map(([id, contrib]) => ({
        ...contrib,
        categoriesContributed: Array.from(contrib.categoriesContributed)
      }))
    })),
    allDiscoveries: log.allDiscoveries,
    globalMetrics: log.globalMetrics,
    globalStats: log.globalStats,
    agentLifetimeContributions: Array.from(log.agentLifetimeContributions.entries()).map(([id, contrib]) => ({
      ...contrib,
      categoriesContributed: Array.from(contrib.categoriesContributed)
    })),
    milestones: log.milestones,
    achievedMilestones: log.achievedMilestones,
    categoryFirstDiscoveries: Array.from(log.categoryFirstDiscoveries.entries()),
    discoveryTimeline: log.discoveryTimeline
  };
  
  return JSON.stringify(exportData, null, 2);
}

/**
 * Import science log from JSON - ENHANCED
 */
export function importScienceLog(jsonData: string): ScienceLog {
  const data = JSON.parse(jsonData);
  
  // Initialize with defaults
  const log = initializeScienceLog();
  
  // Restore global data
  log.allDiscoveries = data.allDiscoveries || [];
  log.globalMetrics = data.globalMetrics || log.globalMetrics;
  log.globalStats = { ...log.globalStats, ...data.globalStats };
  log.milestones = data.milestones || DEFAULT_MILESTONES;
  log.achievedMilestones = data.achievedMilestones || [];
  log.discoveryTimeline = data.discoveryTimeline || [];
  
  // Restore category first discoveries
  if (data.categoryFirstDiscoveries) {
    for (const [category, discovery] of data.categoryFirstDiscoveries) {
      log.categoryFirstDiscoveries.set(category as PhysicsCategory, discovery);
    }
  }
  
  // Restore agent lifetime contributions
  if (data.agentLifetimeContributions) {
    for (const contrib of data.agentLifetimeContributions) {
      log.agentLifetimeContributions.set(contrib.agentId, {
        ...contrib,
        categoriesContributed: new Set(contrib.categoriesContributed)
      });
    }
  }
  
  // Restore eras
  for (const eraData of data.eras || []) {
    const categoryMetrics = new Map<PhysicsCategory, CategoryMetrics>();
    if (eraData.categoryMetrics) {
      for (const [cat, metrics] of eraData.categoryMetrics) {
        categoryMetrics.set(cat as PhysicsCategory, metrics);
      }
    }
    
    const agentContributions = new Map<number, AgentContribution>();
    if (eraData.agentContributions) {
      for (const contrib of eraData.agentContributions) {
        agentContributions.set(contrib.agentId, {
          ...contrib,
          categoriesContributed: new Set(contrib.categoriesContributed)
        });
      }
    }
    
    log.eras.set(eraData.level, {
      era: eraData.era,
      discoveries: eraData.discoveries,
      metrics: eraData.metrics,
      categoryMetrics,
      agentContributions
    });
  }
  
  return log;
}

/**
 * Get discovery statistics for visualization
 */
export function getDiscoveryVisualizationData(log: ScienceLog): {
  timelineData: { tick: number; cumulative: number; category: string }[];
  categoryPieData: { category: string; count: number; percentage: number }[];
  eraBarData: { era: string; physics: number; math: number; inventions: number }[];
  momentumLineData: { tick: number; momentum: number }[];
  collaborationData: { era: string; collaborative: number; solo: number }[];
} {
  // Timeline data
  const timelineData: { tick: number; cumulative: number; category: string }[] = [];
  let cumulative = 0;
  for (const entry of log.discoveryTimeline) {
    cumulative++;
    timelineData.push({
      tick: entry.tick,
      cumulative,
      category: entry.category
    });
  }
  
  // Category pie data
  const totalPhysics = log.globalStats.totalPhysicsDiscovered;
  const categoryPieData = PHYSICS_CATEGORIES
    .map(cat => ({
      category: cat,
      count: log.globalStats.categoryDistribution[cat] || 0,
      percentage: totalPhysics > 0 ? 
        ((log.globalStats.categoryDistribution[cat] || 0) / totalPhysics) * 100 : 0
    }))
    .filter(d => d.count > 0)
    .sort((a, b) => b.count - a.count);
  
  // Era bar data
  const eraBarData = Array.from(log.eras.values())
    .sort((a, b) => a.era.level - b.era.level)
    .map(eraLog => ({
      era: eraLog.era.name,
      physics: eraLog.metrics.physicsDiscovered,
      math: eraLog.metrics.mathDiscovered,
      inventions: eraLog.metrics.inventionsDiscovered
    }));
  
  // Collaboration data
  const collaborationData = Array.from(log.eras.values())
    .sort((a, b) => a.era.level - b.era.level)
    .map(eraLog => ({
      era: eraLog.era.name,
      collaborative: eraLog.metrics.collaborativeDiscoveries,
      solo: eraLog.metrics.soloDiscoveries
    }));
  
  return {
    timelineData,
    categoryPieData,
    eraBarData,
    momentumLineData: log.globalStats.momentumHistory,
    collaborationData
  };
}