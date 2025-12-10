/**
 * DATA EXPORT UTILITIES
 * 
 * Functions for exporting simulation data as downloadable JSON files:
 * - Invention history (all discoveries with timestamps)
 * - Evolution data (population dynamics, trait distribution, genetic trends)
 * - Scientific era logs (physics/mathematics discoveries, era progression)
 * - Agent conversations (self-aware agent communications and questions)
 */

import type { WorldState, Agent, DiscoveryEvent, HistoryPoint } from '../types';
import type { ScienceState } from '../science-system';
import type { PhysicsConcept } from '../science-system/physics';
import type { MathConcept } from '../science-system/mathematics';
import type { CommunicationLog, AgentMessage, MessageType } from '../communication-system';

/**
 * Generic function to trigger a JSON file download in the browser
 */
function downloadJSON(data: any, filename: string): void {
  const jsonStr = JSON.stringify(data, null, 2);
  const blob = new Blob([jsonStr], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  // Clean up the URL object
  setTimeout(() => URL.revokeObjectURL(url), 100);
}

/**
 * Export complete invention history with all discoveries
 */
export function exportInventionHistory(world: WorldState): void {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  
  const inventionData = {
    exportedAt: new Date().toISOString(),
    simulationTick: world.tick,
    totalDiscoveries: world.discoveries.length,
    discoveries: world.discoveries.map(d => ({
      tick: d.tick,
      agentId: d.agentId,
      inventionName: d.invention.name,
      inventionType: d.invention.type,
      inventionId: d.invention.id,
      effect: d.invention.effect,
      description: d.invention.description,
      requirements: d.invention.requirements,
    })),
    inventionsByAgent: world.agents.map(agent => ({
      agentId: agent.id,
      inventionCount: agent.inventions.length,
      inventionPoints: agent.inventionPoints,
      genes: {
        curiosity: agent.genes.curiosity,
        creativity: agent.genes.creativity,
        social: agent.genes.social,
        patience: agent.genes.patience,
      },
      inventions: agent.inventions.map(inv => ({
        name: inv.name,
        type: inv.type,
        effect: inv.effect,
        discoveredAt: inv.discoveredAt,
      })),
    })),
    statistics: {
      totalAgents: world.agents.length,
      agentsWithInventions: world.agents.filter(a => a.inventions.length > 0).length,
      averageInventionsPerAgent: world.agents.length > 0
        ? world.agents.reduce((sum, a) => sum + a.inventions.length, 0) / world.agents.length
        : 0,
      averageInventionPoints: world.agents.length > 0
        ? world.agents.reduce((sum, a) => sum + a.inventionPoints, 0) / world.agents.length
        : 0,
      mostInventiveAgent: world.agents.reduce((max, a) => 
        a.inventions.length > max.inventionCount 
          ? { agentId: a.id, inventionCount: a.inventions.length }
          : max,
        { agentId: -1, inventionCount: 0 }
      ),
    },
  };
  
  downloadJSON(inventionData, `invention-history_tick-${world.tick}_${timestamp}.json`);
}

/**
 * Export evolution and population data
 */
export function exportEvolutionData(world: WorldState): void {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
    // Calculate genetic statistics
  const geneStats = {
    foodPreference: { min: Number.MAX_VALUE, max: 0, avg: 0, stdDev: 0 },
    exploration: { min: Number.MAX_VALUE, max: 0, avg: 0, stdDev: 0 },
    reproductionThreshold: { min: Number.MAX_VALUE, max: 0, avg: 0, stdDev: 0 },
    mutationRate: { min: Number.MAX_VALUE, max: 0, avg: 0, stdDev: 0 },
    curiosity: { min: Number.MAX_VALUE, max: 0, avg: 0, stdDev: 0 },
    social: { min: Number.MAX_VALUE, max: 0, avg: 0, stdDev: 0 },
    creativity: { min: Number.MAX_VALUE, max: 0, avg: 0, stdDev: 0 },
    patience: { min: Number.MAX_VALUE, max: 0, avg: 0, stdDev: 0 },
  };
  
  // Calculate min, max, avg for each gene
  if (world.agents.length > 0) {
    for (const agent of world.agents) {
      for (const gene of Object.keys(geneStats) as Array<keyof typeof geneStats>) {
        const value = agent.genes[gene] as number;
        geneStats[gene].min = Math.min(geneStats[gene].min, value);
        geneStats[gene].max = Math.max(geneStats[gene].max, value);
        geneStats[gene].avg += value;
      }
    }
    
    // Calculate averages
    for (const gene of Object.keys(geneStats) as Array<keyof typeof geneStats>) {
      geneStats[gene].avg /= world.agents.length;
    }
    
    // Calculate standard deviation
    for (const agent of world.agents) {
      for (const gene of Object.keys(geneStats) as Array<keyof typeof geneStats>) {
        const value = agent.genes[gene] as number;
        const diff = value - geneStats[gene].avg;
        geneStats[gene].stdDev += diff * diff;
      }
    }
    
    for (const gene of Object.keys(geneStats) as Array<keyof typeof geneStats>) {
      geneStats[gene].stdDev = Math.sqrt(geneStats[gene].stdDev / world.agents.length);
    }
  }
  
  const evolutionData = {
    exportedAt: new Date().toISOString(),
    simulationTick: world.tick,
    populationDynamics: {
      current: world.agents.length,
      history: world.history.map(h => ({
        tick: h.tick,
        totalAgents: h.totalAgents,
        traitDistribution: h.byTrait,
      })),
      peakPopulation: Math.max(...world.history.map(h => h.totalAgents), 0),
      extinctionEvents: world.history.filter((h, i) => 
        i > 0 && h.totalAgents === 0 && world.history[i-1].totalAgents > 0
      ).length,
    },
    geneticStatistics: geneStats,
    traitLineages: Object.entries(
      world.agents.reduce((acc, a) => {
        acc[a.genes.traitId] = (acc[a.genes.traitId] || 0) + 1;
        return acc;
      }, {} as Record<number, number>)
    ).map(([traitId, count]) => ({
      traitId: parseInt(traitId),
      count,
      percentage: world.agents.length > 0 ? (count / world.agents.length) * 100 : 0,
    })),
    currentAgents: world.agents.map(agent => ({
      id: agent.id,
      position: { x: agent.x, y: agent.y },
      energy: agent.energy,
      genes: agent.genes,
      inventionCount: agent.inventions.length,
      inventionPoints: agent.inventionPoints,
    })),
    gridSize: {
      width: world.gridWidth,
      height: world.gridHeight,
    },
  };
  
  downloadJSON(evolutionData, `evolution-data_tick-${world.tick}_${timestamp}.json`);
}

/**
 * Export scientific discovery logs and era progression
 */
export function exportScienceLogs(world: WorldState): void {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  
  if (!world.scienceState) {
    console.warn('No science data available to export');
    return;
  }
  
  const scienceData = {
    exportedAt: new Date().toISOString(),
    simulationTick: world.tick,
    currentEra: {
      level: world.scienceState.currentEra.level,
      name: world.scienceState.currentEra.name,
      minDiscoveries: world.scienceState.currentEra.minDiscoveries,
      minPhysicsConcepts: world.scienceState.currentEra.minPhysicsConcepts,
      minMathConcepts: world.scienceState.currentEra.minMathConcepts,
    },
    progressToNextEra: {
      currentDiscoveries: world.scienceState.totalDiscoveries,
      requiredDiscoveries: world.scienceState.currentEra.minDiscoveries,
      percentComplete: (world.scienceState.totalDiscoveries / world.scienceState.currentEra.minDiscoveries) * 100,
    },    physicsDiscoveries: {
      total: world.scienceState.physics.length,
      concepts: world.scienceState.physics.map((p: PhysicsConcept) => ({
        id: p.id,
        name: p.name,
        category: p.category,
        complexity: p.complexity,
        aiBonus: p.aiBonus,
        discoveredAt: p.discoveredAt,
        discoveredBy: p.discoveredBy,
      })),
    },
    mathematicsDiscoveries: {
      total: world.scienceState.mathematics.length,
      concepts: world.scienceState.mathematics.map((m: MathConcept) => ({
        id: m.id,
        name: m.name,
        category: m.category,
        complexity: m.complexity,
        aiBonus: m.aiBonus,
        discoveredAt: m.discoveredAt,
        discoveredBy: m.discoveredBy,
      })),
    },
    statistics: {
      totalDiscoveries: world.scienceState.totalDiscoveries,
      physicsCount: world.scienceState.physics.length,
      mathematicsCount: world.scienceState.mathematics.length,
      averagePhysicsComplexity: world.scienceState.physics.length > 0
        ? world.scienceState.physics.reduce((sum: number, p: PhysicsConcept) => sum + p.complexity, 0) / world.scienceState.physics.length
        : 0,
      averageMathComplexity: world.scienceState.mathematics.length > 0
        ? world.scienceState.mathematics.reduce((sum: number, m: MathConcept) => sum + m.complexity, 0) / world.scienceState.mathematics.length
        : 0,
    },
  };
  
  downloadJSON(scienceData, `science-logs_tick-${world.tick}_${timestamp}.json`);
}

/**
 * Export complete world state (comprehensive export including all systems)
 */
export function exportCompleteData(world: WorldState): void {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  
  const completeData = {
    exportedAt: new Date().toISOString(),
    simulationTick: world.tick,
    worldState: {
      gridSize: {
        width: world.gridWidth,
        height: world.gridHeight,
      },
      agents: world.agents,
      discoveries: world.discoveries,
      history: world.history,
      scienceState: world.scienceState,
    },
    metadata: {
      totalAgents: world.agents.length,
      totalDiscoveries: world.discoveries.length,
      totalScientificDiscoveries: world.scienceState
        ? world.scienceState.physics.length + world.scienceState.mathematics.length
        : 0,
      currentEra: world.scienceState?.currentEra.name || 'Unknown',
    },
  };
    downloadJSON(completeData, `complete-export_tick-${world.tick}_${timestamp}.json`);
}

/**
 * Export agent conversation history
 * Captures all communications from self-aware agents including questions and statements
 */
export function exportConversations(
  communicationLog: CommunicationLog,
  tick: number,
  agents?: Agent[]
): void {
  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
  
  // Group messages by agent
  const messagesByAgent: Record<number, AgentMessage[]> = {};
  for (const msg of communicationLog.messages) {
    if (!messagesByAgent[msg.agentId]) {
      messagesByAgent[msg.agentId] = [];
    }
    messagesByAgent[msg.agentId].push(msg);
  }

  // Count messages by type
  const messagesByType: Record<string, number> = {};
  for (const msg of communicationLog.messages) {
    messagesByType[msg.type] = (messagesByType[msg.type] || 0) + 1;
  }

  // Extract questions vs statements
  const questions = communicationLog.messages.filter(m => m.isQuestion);
  const statements = communicationLog.messages.filter(m => !m.isQuestion);

  const conversationData = {
    exportedAt: new Date().toISOString(),
    simulationTick: tick,
    summary: {
      totalMessages: communicationLog.totalMessages,
      uniqueAgents: Object.keys(messagesByAgent).length,
      firstCommunicationTick: communicationLog.firstCommunicationTick,
      questionsCount: questions.length,
      statementsCount: statements.length,
      messagesByType,
    },
    timeline: communicationLog.messages.map(msg => ({
      tick: msg.tick,
      agentId: msg.agentId,
      type: msg.type,
      content: msg.content,
      isQuestion: msg.isQuestion,
      consciousnessLevel: msg.consciousnessLevel,
      context: msg.context,
    })),
    byAgent: Object.entries(messagesByAgent).map(([agentId, messages]) => {
      const agent = agents?.find(a => a.id === parseInt(agentId));
      return {
        agentId: parseInt(agentId),
        messageCount: messages.length,
        firstMessageTick: messages[0]?.tick,
        lastMessageTick: messages[messages.length - 1]?.tick,
        genes: agent ? {
          curiosity: agent.genes.curiosity,
          creativity: agent.genes.creativity,
          social: agent.genes.social,
        } : null,
        messages: messages.map(m => ({
          tick: m.tick,
          type: m.type,
          content: m.content,
          isQuestion: m.isQuestion,
        })),
      };
    }),
    questions: questions.map(q => ({
      tick: q.tick,
      agentId: q.agentId,
      type: q.type,
      content: q.content,
    })),
    statements: statements.map(s => ({
      tick: s.tick,
      agentId: s.agentId,
      type: s.type,
      content: s.content,
    })),
    analysis: {
      mostTalkativeAgent: Object.entries(messagesByAgent)
        .sort(([, a], [, b]) => b.length - a.length)[0] || null,
      mostCommonMessageType: Object.entries(messagesByType)
        .sort(([, a], [, b]) => b - a)[0]?.[0] || null,
      averageTicksBetweenMessages: communicationLog.messages.length > 1
        ? (communicationLog.messages[communicationLog.messages.length - 1].tick - 
           communicationLog.messages[0].tick) / (communicationLog.messages.length - 1)
        : 0,
    },
  };

  downloadJSON(conversationData, `conversations_tick-${tick}_${timestamp}.json`);
}
