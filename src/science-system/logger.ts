/**
 * Scientific Discovery Logger
 * Tracks and organizes discoveries by era for historical analysis
 */

import type { ScientificEra, ScientificDiscovery, ProgressionMetrics } from './eras';
import type { ScienceState } from './index';

export interface EraLog {
  era: ScientificEra;
  discoveries: ScientificDiscovery[];
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
    progressionSpeed: 'slow' | 'moderate' | 'fast' | 'explosive';
  };
}

export interface ScienceLog {
  eras: Map<number, EraLog>;           // Era level -> EraLog
  allDiscoveries: ScientificDiscovery[];
  globalMetrics: ProgressionMetrics;
}

/**
 * Initialize science logging system
 */
export function initializeScienceLog(): ScienceLog {
  return {
    eras: new Map(),
    allDiscoveries: [],
    globalMetrics: {
      totalDiscoveries: 0,
      discoveriesPerEra: {},
      averageDiscoveryRate: 0,
      currentEraLevel: 0,
      ticksSinceLastEra: 0,
      scientificAcceleration: 0
    }
  };
}

/**
 * Add discoveries to the log
 */
export function logDiscoveries(
  log: ScienceLog,
  discoveries: ScientificDiscovery[],
  currentEra: ScientificEra,
  tick: number
): ScienceLog {
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
        progressionSpeed: 'slow'
      }
    };
    log.eras.set(currentEra.level, eraLog);
  }
  
  // Add discoveries
  eraLog.discoveries.push(...discoveries);
  log.allDiscoveries.push(...discoveries);
  
  // Update metrics
  eraLog.metrics.totalDiscoveries += discoveries.length;
  
  for (const discovery of discoveries) {
    switch (discovery.category) {
      case 'physics':
        eraLog.metrics.physicsDiscovered++;
        break;
      case 'mathematics':
        eraLog.metrics.mathDiscovered++;
        break;
      case 'invention':
        eraLog.metrics.inventionsDiscovered++;
        break;
      case 'evolution':
        eraLog.metrics.evolutionsDiscovered++;
        break;
    }
  }
  
  // Calculate rate
  const duration = tick - eraLog.metrics.startTick;
  if (duration > 0) {
    eraLog.metrics.averageDiscoveryRate = eraLog.metrics.totalDiscoveries / duration;
    
    // Determine progression speed
    if (eraLog.metrics.averageDiscoveryRate > 0.5) {
      eraLog.metrics.progressionSpeed = 'explosive';
    } else if (eraLog.metrics.averageDiscoveryRate > 0.2) {
      eraLog.metrics.progressionSpeed = 'fast';
    } else if (eraLog.metrics.averageDiscoveryRate > 0.05) {
      eraLog.metrics.progressionSpeed = 'moderate';
    } else {
      eraLog.metrics.progressionSpeed = 'slow';
    }
  }
  
  return log;
}

/**
 * Finalize an era when advancing
 */
export function finalizeEra(
  log: ScienceLog,
  eraLevel: number,
  endTick: number
): ScienceLog {
  const eraLog = log.eras.get(eraLevel);
  
  if (eraLog) {
    eraLog.metrics.endTick = endTick;
    eraLog.metrics.duration = endTick - eraLog.metrics.startTick;
  }
  
  return log;
}

/**
 * Generate a summary report of scientific progress
 */
export function generateProgressReport(log: ScienceLog, scienceState: ScienceState): string {
  const lines: string[] = [];
  
  lines.push('='.repeat(60));
  lines.push('SCIENTIFIC PROGRESS REPORT');
  lines.push('='.repeat(60));
  lines.push('');
  
  lines.push(`Current Era: ${scienceState.currentEra.name} (Level ${scienceState.currentEra.level})`);
  lines.push(`Total Discoveries: ${log.allDiscoveries.length}`);
  lines.push(`Physics Concepts Unlocked: ${scienceState.unlockedPhysics.length}`);
  lines.push(`Mathematics Concepts Unlocked: ${scienceState.unlockedMath.length}`);
  lines.push('');
  
  lines.push('ERA HISTORY:');
  lines.push('-'.repeat(60));
  
  const sortedEras = Array.from(log.eras.values()).sort((a, b) => a.era.level - b.era.level);
  
  for (const eraLog of sortedEras) {
    const status = eraLog.metrics.endTick ? 'Completed' : 'Current';
    const duration = eraLog.metrics.duration || '(ongoing)';
    
    lines.push(`${eraLog.era.name} (Level ${eraLog.era.level}) - ${status}`);
    lines.push(`  Duration: ${duration} ticks`);
    lines.push(`  Discoveries: ${eraLog.metrics.totalDiscoveries}`);
    lines.push(`  - Physics: ${eraLog.metrics.physicsDiscovered}`);
    lines.push(`  - Mathematics: ${eraLog.metrics.mathDiscovered}`);
    lines.push(`  - Inventions: ${eraLog.metrics.inventionsDiscovered}`);
    lines.push(`  Discovery Rate: ${eraLog.metrics.averageDiscoveryRate.toFixed(4)} per tick`);
    lines.push(`  Progression Speed: ${eraLog.metrics.progressionSpeed.toUpperCase()}`);
    lines.push('');
  }
  
  lines.push('='.repeat(60));
  
  return lines.join('\n');
}

/**
 * Export era data to JSON for persistence
 */
export function exportScienceLog(log: ScienceLog): string {
  const exportData = {
    eras: Array.from(log.eras.entries()).map(([level, eraLog]) => ({
      level,
      ...eraLog
    })),
    allDiscoveries: log.allDiscoveries,
    globalMetrics: log.globalMetrics
  };
  
  return JSON.stringify(exportData, null, 2);
}

/**
 * Import era data from JSON
 */
export function importScienceLog(jsonData: string): ScienceLog {
  const data = JSON.parse(jsonData);
  
  const log: ScienceLog = {
    eras: new Map(),
    allDiscoveries: data.allDiscoveries || [],
    globalMetrics: data.globalMetrics || {
      totalDiscoveries: 0,
      discoveriesPerEra: {},
      averageDiscoveryRate: 0,
      currentEraLevel: 0,
      ticksSinceLastEra: 0,
      scientificAcceleration: 0
    }
  };
  
  for (const eraData of data.eras || []) {
    log.eras.set(eraData.level, eraData);
  }
  
  return log;
}
