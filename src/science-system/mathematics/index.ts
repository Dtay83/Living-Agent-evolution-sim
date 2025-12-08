/**
 * Mathematics Concepts System
 * Provides mathematical knowledge that aids agent reasoning and optimization
 */

export interface MathConcept {
  id: string;
  name: string;
  category: 'arithmetic' | 'geometry' | 'algebra' | 'calculus' | 'statistics' | 'topology' | 'abstract';
  complexity: number;         // 1-10: How advanced this concept is
  discoveredAt?: number;      // Tick when discovered
  discoveredBy?: number;      // Agent ID who discovered
  prerequisiteIds: string[];  // Required prior knowledge
  description: string;
  aiBonus: {
    decisionQuality?: number;   // Multiplier for Q-value updates
    explorationBonus?: number;  // Additive to exploration gene
    patternRecognition?: number; // Improves state recognition
    optimizationPower?: number;  // Improves overall efficiency
  };
}

/**
 * Progressive mathematics concepts from basic to advanced
 * Unlimited system - new concepts can be generated procedurally
 */
export const MATH_CONCEPTS: MathConcept[] = [
  // ARITHMETIC (Level 1-2)
  {
    id: 'counting',
    name: 'Counting',
    category: 'arithmetic',
    complexity: 1,
    prerequisiteIds: [],
    description: 'Basic enumeration and quantity',
    aiBonus: { decisionQuality: 1.05 }
  },
  {
    id: 'addition_subtraction',
    name: 'Addition & Subtraction',
    category: 'arithmetic',
    complexity: 1,
    prerequisiteIds: ['counting'],
    description: 'Combining and removing quantities',
    aiBonus: { decisionQuality: 1.08, optimizationPower: 1.05 }
  },
  {
    id: 'multiplication_division',
    name: 'Multiplication & Division',
    category: 'arithmetic',
    complexity: 2,
    prerequisiteIds: ['addition_subtraction'],
    description: 'Scaling and partitioning',
    aiBonus: { decisionQuality: 1.10, optimizationPower: 1.08 }
  },
  
  // GEOMETRY (Level 2-4)
  {
    id: 'basic_shapes',
    name: 'Basic Shapes',
    category: 'geometry',
    complexity: 2,
    prerequisiteIds: ['counting'],
    description: 'Recognition of geometric forms',
    aiBonus: { patternRecognition: 1.10 }
  },
  {
    id: 'pythagorean_theorem',
    name: 'Pythagorean Theorem',
    category: 'geometry',
    complexity: 3,
    prerequisiteIds: ['basic_shapes', 'multiplication_division'],
    description: 'Right triangle relationships',
    aiBonus: { decisionQuality: 1.12, patternRecognition: 1.15 }
  },
  {
    id: 'trigonometry',
    name: 'Trigonometry',
    category: 'geometry',
    complexity: 4,
    prerequisiteIds: ['pythagorean_theorem'],
    description: 'Angular relationships and periodic functions',
    aiBonus: { decisionQuality: 1.15, patternRecognition: 1.20, optimizationPower: 1.12 }
  },
  
  // ALGEBRA (Level 3-5)
  {
    id: 'variables',
    name: 'Variables & Equations',
    category: 'algebra',
    complexity: 3,
    prerequisiteIds: ['multiplication_division'],
    description: 'Symbolic representation',
    aiBonus: { decisionQuality: 1.15, explorationBonus: 0.05 }
  },
  {
    id: 'functions',
    name: 'Functions',
    category: 'algebra',
    complexity: 4,
    prerequisiteIds: ['variables'],
    description: 'Input-output relationships',
    aiBonus: { decisionQuality: 1.20, patternRecognition: 1.25 }
  },
  {
    id: 'polynomials',
    name: 'Polynomials',
    category: 'algebra',
    complexity: 4,
    prerequisiteIds: ['functions'],
    description: 'Higher-degree equations',
    aiBonus: { decisionQuality: 1.22, optimizationPower: 1.15 }
  },
  {
    id: 'exponentials_logarithms',
    name: 'Exponentials & Logarithms',
    category: 'algebra',
    complexity: 5,
    prerequisiteIds: ['polynomials'],
    description: 'Growth and scaling functions',
    aiBonus: { decisionQuality: 1.25, patternRecognition: 1.30, optimizationPower: 1.20 }
  },
  
  // CALCULUS (Level 5-7)
  {
    id: 'limits',
    name: 'Limits',
    category: 'calculus',
    complexity: 5,
    prerequisiteIds: ['exponentials_logarithms', 'trigonometry'],
    description: 'Behavior as values approach infinity',
    aiBonus: { decisionQuality: 1.28, optimizationPower: 1.25 }
  },
  {
    id: 'derivatives',
    name: 'Derivatives',
    category: 'calculus',
    complexity: 6,
    prerequisiteIds: ['limits'],
    description: 'Rates of change',
    aiBonus: { decisionQuality: 1.32, optimizationPower: 1.30, explorationBonus: 0.08 }
  },
  {
    id: 'integrals',
    name: 'Integrals',
    category: 'calculus',
    complexity: 6,
    prerequisiteIds: ['derivatives'],
    description: 'Accumulation and area',
    aiBonus: { decisionQuality: 1.35, optimizationPower: 1.35 }
  },
  {
    id: 'differential_equations',
    name: 'Differential Equations',
    category: 'calculus',
    complexity: 7,
    prerequisiteIds: ['integrals'],
    description: 'Equations involving rates of change',
    aiBonus: { decisionQuality: 1.40, patternRecognition: 1.40, optimizationPower: 1.40 }
  },
  
  // STATISTICS (Level 5-7)
  {
    id: 'probability',
    name: 'Probability Theory',
    category: 'statistics',
    complexity: 5,
    prerequisiteIds: ['multiplication_division'],
    description: 'Likelihood and chance',
    aiBonus: { decisionQuality: 1.30, explorationBonus: 0.10 }
  },
  {
    id: 'distributions',
    name: 'Probability Distributions',
    category: 'statistics',
    complexity: 6,
    prerequisiteIds: ['probability', 'integrals'],
    description: 'Patterns in random data',
    aiBonus: { decisionQuality: 1.35, patternRecognition: 1.45 }
  },
  {
    id: 'bayesian_inference',
    name: 'Bayesian Inference',
    category: 'statistics',
    complexity: 7,
    prerequisiteIds: ['distributions'],
    description: 'Updating beliefs with evidence',
    aiBonus: { decisionQuality: 1.45, patternRecognition: 1.50, optimizationPower: 1.45 }
  },
  
  // TOPOLOGY (Level 8-9)
  {
    id: 'set_theory',
    name: 'Set Theory',
    category: 'topology',
    complexity: 7,
    prerequisiteIds: ['functions'],
    description: 'Collections and membership',
    aiBonus: { decisionQuality: 1.40, patternRecognition: 1.40 }
  },
  {
    id: 'topology_basics',
    name: 'Topology',
    category: 'topology',
    complexity: 8,
    prerequisiteIds: ['set_theory'],
    description: 'Properties preserved under continuous deformation',
    aiBonus: { decisionQuality: 1.50, patternRecognition: 1.55, optimizationPower: 1.50 }
  },
  
  // ABSTRACT ALGEBRA (Level 9-10)
  {
    id: 'group_theory',
    name: 'Group Theory',
    category: 'abstract',
    complexity: 9,
    prerequisiteIds: ['set_theory', 'polynomials'],
    description: 'Abstract algebraic structures',
    aiBonus: { decisionQuality: 1.55, patternRecognition: 1.60, optimizationPower: 1.55 }
  },
  {
    id: 'category_theory',
    name: 'Category Theory',
    category: 'abstract',
    complexity: 10,
    prerequisiteIds: ['group_theory', 'topology_basics'],
    description: 'Universal properties and morphisms',
    aiBonus: { decisionQuality: 1.65, patternRecognition: 1.70, optimizationPower: 1.65, explorationBonus: 0.15 }
  }
];

/**
 * Generate advanced math concepts procedurally
 * Enables unlimited scientific progression
 */
export function generateAdvancedMathConcept(level: number): MathConcept {
  const categories: MathConcept['category'][] = ['arithmetic', 'geometry', 'algebra', 'calculus', 'statistics', 'topology', 'abstract'];
  const category = categories[level % categories.length];
  
  const complexity = Math.min(10 + Math.floor(level / 10), 100);
  
  return {
    id: `advanced_math_${level}`,
    name: `Advanced ${category.charAt(0).toUpperCase() + category.slice(1)} ${level}`,
    category,
    complexity,
    prerequisiteIds: level > 0 ? [`advanced_math_${level - 1}`] : ['category_theory'],
    description: `Cutting-edge ${category} research level ${level}`,
    aiBonus: {
      decisionQuality: 1.65 + (level * 0.03),
      patternRecognition: 1.70 + (level * 0.04),
      optimizationPower: 1.65 + (level * 0.03),
      explorationBonus: 0.15 + (level * 0.01)
    }
  };
}

/**
 * Calculate total mathematics bonuses for an agent
 */
export function calculateMathBonuses(unlockedMath: MathConcept[]): {
  decisionQuality: number;
  explorationBonus: number;
  patternRecognition: number;
  optimizationPower: number;
} {
  let decisionMult = 1.0;
  let explorationAdd = 0.0;
  let patternMult = 1.0;
  let optimizationMult = 1.0;
  
  for (const concept of unlockedMath) {
    if (concept.aiBonus.decisionQuality) {
      decisionMult *= concept.aiBonus.decisionQuality;
    }
    if (concept.aiBonus.explorationBonus) {
      explorationAdd += concept.aiBonus.explorationBonus;
    }
    if (concept.aiBonus.patternRecognition) {
      patternMult *= concept.aiBonus.patternRecognition;
    }
    if (concept.aiBonus.optimizationPower) {
      optimizationMult *= concept.aiBonus.optimizationPower;
    }
  }
  
  return {
    decisionQuality: decisionMult,
    explorationBonus: Math.min(explorationAdd, 0.3), // Cap at 30% bonus
    patternRecognition: patternMult,
    optimizationPower: optimizationMult
  };
}
