/**
 * Mathematics Concepts System
 * Provides mathematical knowledge that aids agent reasoning and optimization
 * UNLIMITED EVOLUTION - No caps on discovery or bonuses
 */

export interface MathConcept {
  id: string;
  name: string;
  category: 'arithmetic' | 'geometry' | 'algebra' | 'calculus' | 'statistics' | 'topology' | 'abstract' | 'number_theory' | 'logic' | 'optimization' | 'ai_math' | 'transcendent';
  complexity: number;         // 1-100+: How advanced this concept is (NO CAP)
  discoveredAt?: number;      // Tick when discovered
  discoveredBy?: number;      // Agent ID who discovered
  prerequisiteIds: string[];  // Required prior knowledge
  description: string;
  aiBonus: {
    decisionQuality?: number;   // Multiplier for Q-value updates
    explorationBonus?: number;  // Additive to exploration gene
    patternRecognition?: number; // Improves state recognition
    optimizationPower?: number;  // Improves overall efficiency
    computationalSpeed?: number; // NEW: Processing speed multiplier
    abstractionLevel?: number;   // NEW: Higher-level thinking
  };
}

/**
 * Progressive mathematics concepts from basic to transcendent
 * UNLIMITED system - new concepts generated procedurally beyond base set
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
  {
    id: 'modular_arithmetic',
    name: 'Modular Arithmetic',
    category: 'arithmetic',
    complexity: 3,
    prerequisiteIds: ['multiplication_division'],
    description: 'Cyclic number systems and remainders',
    aiBonus: { decisionQuality: 1.12, patternRecognition: 1.10 }
  },
  
  // GEOMETRY (Level 2-5)
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
  {
    id: 'analytic_geometry',
    name: 'Analytic Geometry',
    category: 'geometry',
    complexity: 4,
    prerequisiteIds: ['trigonometry', 'variables'],
    description: 'Coordinate systems and geometric algebra',
    aiBonus: { decisionQuality: 1.18, patternRecognition: 1.22 }
  },
  {
    id: 'non_euclidean_geometry',
    name: 'Non-Euclidean Geometry',
    category: 'geometry',
    complexity: 6,
    prerequisiteIds: ['analytic_geometry'],
    description: 'Curved spaces and alternative axioms',
    aiBonus: { decisionQuality: 1.25, patternRecognition: 1.30, abstractionLevel: 1.15 }
  },
  
  // ALGEBRA (Level 3-6)
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
  {
    id: 'linear_algebra',
    name: 'Linear Algebra',
    category: 'algebra',
    complexity: 6,
    prerequisiteIds: ['exponentials_logarithms'],
    description: 'Vectors, matrices, and transformations',
    aiBonus: { decisionQuality: 1.30, computationalSpeed: 1.20, optimizationPower: 1.25 }
  },
  {
    id: 'abstract_algebra',
    name: 'Abstract Algebra',
    category: 'algebra',
    complexity: 8,
    prerequisiteIds: ['linear_algebra', 'group_theory'],
    description: 'Rings, fields, and algebraic structures',
    aiBonus: { decisionQuality: 1.45, abstractionLevel: 1.30, optimizationPower: 1.35 }
  },
  
  // CALCULUS (Level 5-8)
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
  {
    id: 'multivariable_calculus',
    name: 'Multivariable Calculus',
    category: 'calculus',
    complexity: 7,
    prerequisiteIds: ['differential_equations', 'linear_algebra'],
    description: 'Calculus in multiple dimensions',
    aiBonus: { decisionQuality: 1.45, optimizationPower: 1.45, computationalSpeed: 1.25 }
  },
  {
    id: 'tensor_calculus',
    name: 'Tensor Calculus',
    category: 'calculus',
    complexity: 9,
    prerequisiteIds: ['multivariable_calculus'],
    description: 'Multi-dimensional array mathematics',
    aiBonus: { decisionQuality: 1.55, computationalSpeed: 1.40, abstractionLevel: 1.35 }
  },
  
  // STATISTICS (Level 5-8)
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
  {
    id: 'information_theory',
    name: 'Information Theory',
    category: 'statistics',
    complexity: 8,
    prerequisiteIds: ['bayesian_inference'],
    description: 'Entropy, compression, and communication',
    aiBonus: { decisionQuality: 1.52, patternRecognition: 1.55, computationalSpeed: 1.30 }
  },
  {
    id: 'stochastic_processes',
    name: 'Stochastic Processes',
    category: 'statistics',
    complexity: 8,
    prerequisiteIds: ['bayesian_inference', 'differential_equations'],
    description: 'Random processes evolving over time',
    aiBonus: { decisionQuality: 1.50, patternRecognition: 1.52, explorationBonus: 0.12 }
  },
  
  // NUMBER THEORY (Level 6-9) - NEW CATEGORY
  {
    id: 'prime_numbers',
    name: 'Prime Numbers',
    category: 'number_theory',
    complexity: 4,
    prerequisiteIds: ['modular_arithmetic'],
    description: 'Fundamental building blocks of integers',
    aiBonus: { patternRecognition: 1.25, computationalSpeed: 1.15 }
  },
  {
    id: 'cryptographic_math',
    name: 'Cryptographic Mathematics',
    category: 'number_theory',
    complexity: 7,
    prerequisiteIds: ['prime_numbers', 'exponentials_logarithms'],
    description: 'Mathematical foundations of encryption',
    aiBonus: { decisionQuality: 1.40, computationalSpeed: 1.35, patternRecognition: 1.40 }
  },
  {
    id: 'algebraic_number_theory',
    name: 'Algebraic Number Theory',
    category: 'number_theory',
    complexity: 9,
    prerequisiteIds: ['cryptographic_math', 'abstract_algebra'],
    description: 'Algebraic structures within number systems',
    aiBonus: { decisionQuality: 1.55, abstractionLevel: 1.40, patternRecognition: 1.50 }
  },
  
  // LOGIC (Level 6-9) - NEW CATEGORY
  {
    id: 'propositional_logic',
    name: 'Propositional Logic',
    category: 'logic',
    complexity: 4,
    prerequisiteIds: ['variables'],
    description: 'Boolean logic and truth tables',
    aiBonus: { decisionQuality: 1.25, computationalSpeed: 1.20 }
  },
  {
    id: 'predicate_logic',
    name: 'Predicate Logic',
    category: 'logic',
    complexity: 6,
    prerequisiteIds: ['propositional_logic', 'set_theory'],
    description: 'Quantifiers and formal reasoning',
    aiBonus: { decisionQuality: 1.38, abstractionLevel: 1.25, patternRecognition: 1.35 }
  },
  {
    id: 'modal_logic',
    name: 'Modal Logic',
    category: 'logic',
    complexity: 8,
    prerequisiteIds: ['predicate_logic'],
    description: 'Possibility, necessity, and belief',
    aiBonus: { decisionQuality: 1.50, abstractionLevel: 1.40, explorationBonus: 0.15 }
  },
  {
    id: 'godel_incompleteness',
    name: 'Gödel Incompleteness',
    category: 'logic',
    complexity: 10,
    prerequisiteIds: ['modal_logic'],
    description: 'Limits of formal systems and self-reference',
    aiBonus: { decisionQuality: 1.60, abstractionLevel: 1.55, explorationBonus: 0.18 }
  },
  
  // TOPOLOGY (Level 7-9)
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
  {
    id: 'manifold_theory',
    name: 'Manifold Theory',
    category: 'topology',
    complexity: 9,
    prerequisiteIds: ['topology_basics', 'multivariable_calculus'],
    description: 'Smooth curved spaces',
    aiBonus: { decisionQuality: 1.55, patternRecognition: 1.58, abstractionLevel: 1.45 }
  },
  
  // OPTIMIZATION (Level 7-10) - NEW CATEGORY
  {
    id: 'convex_optimization',
    name: 'Convex Optimization',
    category: 'optimization',
    complexity: 7,
    prerequisiteIds: ['multivariable_calculus', 'linear_algebra'],
    description: 'Finding optimal solutions in convex spaces',
    aiBonus: { optimizationPower: 1.50, computationalSpeed: 1.35, decisionQuality: 1.42 }
  },
  {
    id: 'dynamic_programming',
    name: 'Dynamic Programming',
    category: 'optimization',
    complexity: 7,
    prerequisiteIds: ['functions', 'probability'],
    description: 'Optimal substructure and memoization',
    aiBonus: { optimizationPower: 1.55, computationalSpeed: 1.45, decisionQuality: 1.45 }
  },
  {
    id: 'game_theory',
    name: 'Game Theory',
    category: 'optimization',
    complexity: 8,
    prerequisiteIds: ['probability', 'convex_optimization'],
    description: 'Strategic decision making',
    aiBonus: { decisionQuality: 1.55, explorationBonus: 0.15, optimizationPower: 1.45 }
  },
  {
    id: 'control_theory',
    name: 'Control Theory',
    category: 'optimization',
    complexity: 9,
    prerequisiteIds: ['differential_equations', 'convex_optimization'],
    description: 'Feedback systems and stability',
    aiBonus: { optimizationPower: 1.60, decisionQuality: 1.52, computationalSpeed: 1.40 }
  },
  
  // AI MATH (Level 8-10) - NEW CATEGORY
  {
    id: 'neural_math',
    name: 'Neural Network Mathematics',
    category: 'ai_math',
    complexity: 8,
    prerequisiteIds: ['linear_algebra', 'derivatives', 'probability'],
    description: 'Backpropagation and gradient flow',
    aiBonus: { decisionQuality: 1.55, computationalSpeed: 1.50, optimizationPower: 1.50 }
  },
  {
    id: 'reinforcement_learning_theory',
    name: 'RL Theory',
    category: 'ai_math',
    complexity: 9,
    prerequisiteIds: ['dynamic_programming', 'stochastic_processes'],
    description: 'Bellman equations and policy optimization',
    aiBonus: { decisionQuality: 1.62, explorationBonus: 0.18, optimizationPower: 1.58 }
  },
  {
    id: 'computational_complexity',
    name: 'Computational Complexity',
    category: 'ai_math',
    complexity: 9,
    prerequisiteIds: ['predicate_logic', 'dynamic_programming'],
    description: 'P vs NP and complexity classes',
    aiBonus: { computationalSpeed: 1.55, decisionQuality: 1.55, abstractionLevel: 1.45 }
  },
  {
    id: 'algorithmic_information',
    name: 'Algorithmic Information Theory',
    category: 'ai_math',
    complexity: 10,
    prerequisiteIds: ['information_theory', 'computational_complexity'],
    description: 'Kolmogorov complexity and compression',
    aiBonus: { decisionQuality: 1.65, computationalSpeed: 1.60, patternRecognition: 1.60 }
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
  },
  {
    id: 'homotopy_type_theory',
    name: 'Homotopy Type Theory',
    category: 'abstract',
    complexity: 10,
    prerequisiteIds: ['category_theory', 'topology_basics'],
    description: 'Foundations unifying logic, topology, and computation',
    aiBonus: { decisionQuality: 1.70, abstractionLevel: 1.60, patternRecognition: 1.68 }
  },
  
  // TRANSCENDENT MATHEMATICS (Level 10+) - NEW CATEGORY
  {
    id: 'mathematical_universe',
    name: 'Mathematical Universe Hypothesis',
    category: 'transcendent',
    complexity: 11,
    prerequisiteIds: ['homotopy_type_theory', 'godel_incompleteness'],
    description: 'Reality as mathematical structure',
    aiBonus: { decisionQuality: 1.75, abstractionLevel: 1.70, explorationBonus: 0.20, patternRecognition: 1.75 }
  },
  {
    id: 'omega_math',
    name: 'Transfinite Mathematics',
    category: 'transcendent',
    complexity: 12,
    prerequisiteIds: ['mathematical_universe', 'set_theory'],
    description: 'Infinite ordinals and cardinals beyond countable',
    aiBonus: { decisionQuality: 1.80, abstractionLevel: 1.80, patternRecognition: 1.80, computationalSpeed: 1.70 }
  },
  {
    id: 'hypercomputation',
    name: 'Hypercomputation Theory',
    category: 'transcendent',
    complexity: 15,
    prerequisiteIds: ['omega_math', 'algorithmic_information'],
    description: 'Computation beyond Turing machines',
    aiBonus: { decisionQuality: 2.0, computationalSpeed: 2.0, abstractionLevel: 2.0, optimizationPower: 2.0, explorationBonus: 0.25 }
  }
];

// Track procedurally generated concepts for unlimited evolution
let generatedMathLevel = 0;

/**
 * Generate advanced math concepts procedurally
 * Enables UNLIMITED scientific progression - NO CAPS
 */
export function generateAdvancedMathConcept(level: number): MathConcept {
  const categories: MathConcept['category'][] = ['arithmetic', 'geometry', 'algebra', 'calculus', 'statistics', 'topology', 'abstract', 'number_theory', 'logic', 'optimization', 'ai_math', 'transcendent'];
  const category = categories[level % categories.length];
  
  // Complexity scales infinitely
  const complexity = 15 + level;
  
  // Bonuses grow without limit
  const baseBonus = 2.0 + (level * 0.1);
  
  generatedMathLevel = Math.max(generatedMathLevel, level);
  
  return {
    id: `advanced_math_${level}`,
    name: `${getCategoryPrefix(category)} ${getRomanNumeral(level)}`,
    category,
    complexity,
    prerequisiteIds: level > 0 ? [`advanced_math_${level - 1}`] : ['hypercomputation'],
    description: `${category.charAt(0).toUpperCase() + category.slice(1).replace('_', ' ')} beyond current understanding - Level ${level}`,
    aiBonus: {
      decisionQuality: baseBonus,
      patternRecognition: baseBonus + 0.05,
      optimizationPower: baseBonus,
      explorationBonus: 0.25 + (level * 0.02),
      computationalSpeed: baseBonus,
      abstractionLevel: baseBonus + 0.1
    }
  };
}

function getCategoryPrefix(category: string): string {
  const prefixes: Record<string, string> = {
    'arithmetic': 'Hypernumerics',
    'geometry': 'Dimensional Geometry',
    'algebra': 'Metaalgebra',
    'calculus': 'Infinitesimal Analysis',
    'statistics': 'Probability Manifolds',
    'topology': 'Topological Singularity',
    'abstract': 'Abstract Foundation',
    'number_theory': 'Prime Structures',
    'logic': 'Metamathematical Logic',
    'optimization': 'Universal Optimization',
    'ai_math': 'Cognitive Mathematics',
    'transcendent': 'Transcendent Theory'
  };
  return prefixes[category] || 'Advanced Mathematics';
}

function getRomanNumeral(num: number): string {
  if (num <= 0) return 'I';
  if (num > 100) return `${num}`;
  const numerals = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX'];
  return num < numerals.length ? numerals[num] : `${num}`;
}

/**
 * Calculate total mathematics bonuses for an agent
 * NO CAPS - bonuses can grow infinitely
 */
export function calculateMathBonuses(unlockedMath: MathConcept[]): {
  decisionQuality: number;
  explorationBonus: number;
  patternRecognition: number;
  optimizationPower: number;
  computationalSpeed: number;
  abstractionLevel: number;
} {
  let decisionMult = 1.0;
  let explorationAdd = 0.0;
  let patternMult = 1.0;
  let optimizationMult = 1.0;
  let computationalMult = 1.0;
  let abstractionMult = 1.0;
  
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
    if (concept.aiBonus.computationalSpeed) {
      computationalMult *= concept.aiBonus.computationalSpeed;
    }
    if (concept.aiBonus.abstractionLevel) {
      abstractionMult *= concept.aiBonus.abstractionLevel;
    }
  }
  
  // NO CAPS - unlimited evolution!
  return {
    decisionQuality: decisionMult,
    explorationBonus: explorationAdd, // REMOVED CAP
    patternRecognition: patternMult,
    optimizationPower: optimizationMult,
    computationalSpeed: computationalMult,
    abstractionLevel: abstractionMult
  };
}

/**
 * Get all available math concepts including procedurally generated ones
 */
export function getAllMathConcepts(generatedLevels: number = 0): MathConcept[] {
  const baseConcepts = [...MATH_CONCEPTS];
  for (let i = 0; i < generatedLevels; i++) {
    baseConcepts.push(generateAdvancedMathConcept(i));
  }
  return baseConcepts;
}

/**
 * Get current generated math level for saving state
 */
export function getGeneratedMathLevel(): number {
  return generatedMathLevel;
}

/**
 * Set generated math level when loading state
 */
export function setGeneratedMathLevel(level: number): void {
  generatedMathLevel = level;
}
