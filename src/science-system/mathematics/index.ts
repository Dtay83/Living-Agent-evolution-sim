/**
 * Mathematics Concepts System - EXPANDED EDITION
 * Provides comprehensive mathematical knowledge that aids agent reasoning and optimization
 * UNLIMITED EVOLUTION - No caps on discovery or bonuses
 * 
 * Categories:
 * - arithmetic, geometry, algebra, calculus, statistics, topology, abstract
 * - number_theory, logic, optimization, ai_math, transcendent
 * - NEW: combinatorics, analysis, differential_geometry, computation, foundations, applied_math
 * 
 * Bonus Types:
 * - decisionQuality, explorationBonus, patternRecognition, optimizationPower
 * - computationalSpeed, abstractionLevel
 * - NEW: curiosity, cooperationBonus, informationProcessing, proofIntuition, abstractThinking
 */

export interface MathConcept {
  id: string;
  name: string;
  category: 
    | 'arithmetic' 
    | 'geometry' 
    | 'algebra' 
    | 'calculus' 
    | 'statistics' 
    | 'topology' 
    | 'abstract' 
    | 'number_theory' 
    | 'logic' 
    | 'optimization' 
    | 'ai_math' 
    | 'transcendent'
    | 'combinatorics'
    | 'analysis'
    | 'differential_geometry'
    | 'computation'
    | 'foundations'
    | 'applied_math';
  complexity: number;         // 1-100+: How advanced this concept is (NO CAP)
  discoveredAt?: number;      // Tick when discovered
  discoveredBy?: number;      // Agent ID who discovered
  prerequisiteIds: string[];  // Required prior knowledge
  description: string;
  flavorText?: string;        // Optional deeper insight into the concept
  aiBonus: {
    decisionQuality?: number;     // Multiplier for Q-value updates
    explorationBonus?: number;    // Additive to exploration gene
    patternRecognition?: number;  // Improves state recognition
    optimizationPower?: number;   // Improves overall efficiency
    computationalSpeed?: number;  // Processing speed multiplier
    abstractionLevel?: number;    // Higher-level thinking
    // NEW BONUS TYPES
    curiosity?: number;           // Drive to explore mathematical structures
    cooperationBonus?: number;    // Benefits from collaborative proof/discovery
    informationProcessing?: number; // Faster reasoning and deduction
    proofIntuition?: number;      // Ability to sense correct proof paths
    abstractThinking?: number;    // Capacity for abstract reasoning
  };
}

/**
 * All mathematics categories for tracking
 */
export const MATH_CATEGORIES = [
  'arithmetic', 'geometry', 'algebra', 'calculus', 'statistics', 'topology',
  'abstract', 'number_theory', 'logic', 'optimization', 'ai_math', 'transcendent',
  'combinatorics', 'analysis', 'differential_geometry', 'computation', 'foundations',
  'applied_math'
] as const;

export type MathCategory = typeof MATH_CATEGORIES[number];

/**
 * Progressive mathematics concepts from basic to transcendent
 * UNLIMITED system - new concepts generated procedurally beyond base set
 */
export const MATH_CONCEPTS: MathConcept[] = [
  
  // ============================================================================
  // ARITHMETIC (Level 1-4)
  // The foundation of all mathematics - numbers and basic operations
  // ============================================================================
  {
    id: 'counting',
    name: 'Counting',
    category: 'arithmetic',
    complexity: 1,
    prerequisiteIds: [],
    description: 'Basic enumeration and quantity',
    flavorText: 'One, two, three... The first abstraction. Numbers represent things.',
    aiBonus: { decisionQuality: 1.05, curiosity: 1.02 }
  },
  {
    id: 'ordering',
    name: 'Ordering & Comparison',
    category: 'arithmetic',
    complexity: 1,
    prerequisiteIds: ['counting'],
    description: 'Greater than, less than, equal to',
    flavorText: 'Which is more? Comparison is the basis of optimization.',
    aiBonus: { decisionQuality: 1.06, patternRecognition: 1.03 }
  },
  {
    id: 'addition_subtraction',
    name: 'Addition & Subtraction',
    category: 'arithmetic',
    complexity: 1,
    prerequisiteIds: ['ordering'],
    description: 'Combining and removing quantities',
    flavorText: 'Put together, take apart. The fundamental operations.',
    aiBonus: { decisionQuality: 1.08, optimizationPower: 1.05 }
  },
  {
    id: 'multiplication_division',
    name: 'Multiplication & Division',
    category: 'arithmetic',
    complexity: 2,
    prerequisiteIds: ['addition_subtraction'],
    description: 'Scaling and partitioning',
    flavorText: 'Repeated addition, fair sharing. Scaling up and down.',
    aiBonus: { decisionQuality: 1.10, optimizationPower: 1.08 }
  },
  {
    id: 'fractions',
    name: 'Fractions & Ratios',
    category: 'arithmetic',
    complexity: 2,
    prerequisiteIds: ['multiplication_division'],
    description: 'Parts of wholes and proportional relationships',
    flavorText: 'Not everything is whole. Between integers lie infinite rationals.',
    aiBonus: { decisionQuality: 1.08, patternRecognition: 1.06 }
  },
  {
    id: 'negative_numbers',
    name: 'Negative Numbers',
    category: 'arithmetic',
    complexity: 2,
    prerequisiteIds: ['addition_subtraction'],
    description: 'Numbers less than zero',
    flavorText: 'Debt, below sea level, opposite directions. Numbers extend both ways.',
    aiBonus: { abstractThinking: 1.05, decisionQuality: 1.06 }
  },
  {
    id: 'exponents',
    name: 'Exponents & Powers',
    category: 'arithmetic',
    complexity: 3,
    prerequisiteIds: ['multiplication_division'],
    description: 'Repeated multiplication and powers',
    flavorText: 'Square, cube, and beyond. Numbers grow explosively.',
    aiBonus: { decisionQuality: 1.10, computationalSpeed: 1.08 }
  },
  {
    id: 'roots',
    name: 'Roots & Radicals',
    category: 'arithmetic',
    complexity: 3,
    prerequisiteIds: ['exponents'],
    description: 'Inverse of exponentiation',
    flavorText: 'What multiplied by itself gives this? Reversing the explosive growth.',
    aiBonus: { decisionQuality: 1.08, patternRecognition: 1.08 }
  },
  {
    id: 'modular_arithmetic',
    name: 'Modular Arithmetic',
    category: 'arithmetic',
    complexity: 3,
    prerequisiteIds: ['multiplication_division'],
    description: 'Cyclic number systems and remainders',
    flavorText: 'Clock arithmetic. After 12 comes 1 again. Numbers wrap around.',
    aiBonus: { decisionQuality: 1.12, patternRecognition: 1.10 }
  },
  {
    id: 'place_value',
    name: 'Place Value & Bases',
    category: 'arithmetic',
    complexity: 3,
    prerequisiteIds: ['exponents'],
    description: 'Positional notation and different number bases',
    flavorText: 'Binary, decimal, hexadecimal. The same numbers, different representations.',
    aiBonus: { computationalSpeed: 1.12, informationProcessing: 1.08 }
  },
  {
    id: 'estimation',
    name: 'Estimation & Approximation',
    category: 'arithmetic',
    complexity: 4,
    prerequisiteIds: ['fractions', 'roots'],
    description: 'Quick calculation and reasonable bounds',
    flavorText: 'Close enough is often good enough. Speed versus precision.',
    aiBonus: { computationalSpeed: 1.15, decisionQuality: 1.10 }
  },
  
  // ============================================================================
  // GEOMETRY (Level 2-7)
  // Shapes, spaces, and spatial reasoning
  // ============================================================================
  {
    id: 'basic_shapes',
    name: 'Basic Shapes',
    category: 'geometry',
    complexity: 2,
    prerequisiteIds: ['counting'],
    description: 'Recognition of geometric forms',
    flavorText: 'Circle, square, triangle. The building blocks of visual mathematics.',
    aiBonus: { patternRecognition: 1.10, curiosity: 1.05 }
  },
  {
    id: 'measurement',
    name: 'Measurement',
    category: 'geometry',
    complexity: 2,
    prerequisiteIds: ['basic_shapes', 'multiplication_division'],
    description: 'Length, area, volume',
    flavorText: 'How big? How far? Assigning numbers to space.',
    aiBonus: { patternRecognition: 1.08, decisionQuality: 1.06 }
  },
  {
    id: 'angles',
    name: 'Angles',
    category: 'geometry',
    complexity: 2,
    prerequisiteIds: ['basic_shapes'],
    description: 'Measurement of rotation and direction',
    flavorText: 'The space between two lines. Degrees, radians, turns.',
    aiBonus: { patternRecognition: 1.10, decisionQuality: 1.05 }
  },
  {
    id: 'symmetry',
    name: 'Symmetry',
    category: 'geometry',
    complexity: 3,
    prerequisiteIds: ['basic_shapes', 'angles'],
    description: 'Reflective and rotational invariance',
    flavorText: 'The same from different views. Nature loves symmetry.',
    aiBonus: { patternRecognition: 1.15, abstractThinking: 1.08 }
  },
  {
    id: 'pythagorean_theorem',
    name: 'Pythagorean Theorem',
    category: 'geometry',
    complexity: 3,
    prerequisiteIds: ['measurement', 'exponents'],
    description: 'Right triangle relationships',
    flavorText: 'a² + b² = c². The most famous equation in geometry.',
    aiBonus: { decisionQuality: 1.12, patternRecognition: 1.15 }
  },
  {
    id: 'congruence_similarity',
    name: 'Congruence & Similarity',
    category: 'geometry',
    complexity: 3,
    prerequisiteIds: ['symmetry', 'fractions'],
    description: 'Same shape, same size vs same shape, different size',
    flavorText: 'Identical twins and scaled copies. Preserving ratios.',
    aiBonus: { patternRecognition: 1.12, abstractThinking: 1.06 }
  },
  {
    id: 'trigonometry',
    name: 'Trigonometry',
    category: 'geometry',
    complexity: 4,
    prerequisiteIds: ['pythagorean_theorem', 'angles'],
    description: 'Angular relationships and periodic functions',
    flavorText: 'Sine, cosine, tangent. The mathematics of circles and waves.',
    aiBonus: { decisionQuality: 1.15, patternRecognition: 1.20, optimizationPower: 1.12 }
  },
  {
    id: 'coordinate_geometry',
    name: 'Coordinate Geometry',
    category: 'geometry',
    complexity: 4,
    prerequisiteIds: ['pythagorean_theorem', 'negative_numbers'],
    description: 'Points, lines, and shapes on a grid',
    flavorText: 'Descartes\' gift: algebra meets geometry. Every point has coordinates.',
    aiBonus: { patternRecognition: 1.18, computationalSpeed: 1.10 }
  },
  {
    id: 'analytic_geometry',
    name: 'Analytic Geometry',
    category: 'geometry',
    complexity: 5,
    prerequisiteIds: ['coordinate_geometry', 'trigonometry', 'variables'],
    description: 'Equations of curves and surfaces',
    flavorText: 'Circles, ellipses, parabolas, hyperbolas. Conic sections united by algebra.',
    aiBonus: { decisionQuality: 1.18, patternRecognition: 1.22 }
  },
  {
    id: 'vectors_geometry',
    name: 'Vector Geometry',
    category: 'geometry',
    complexity: 5,
    prerequisiteIds: ['analytic_geometry'],
    description: 'Directed magnitudes in space',
    flavorText: 'Direction and magnitude combined. Arrows that add and scale.',
    aiBonus: { patternRecognition: 1.20, computationalSpeed: 1.15, decisionQuality: 1.15 }
  },
  {
    id: 'projective_geometry',
    name: 'Projective Geometry',
    category: 'geometry',
    complexity: 6,
    prerequisiteIds: ['analytic_geometry'],
    description: 'Geometry of projection and perspective',
    flavorText: 'Parallel lines meet at infinity. The artist\'s vanishing point made rigorous.',
    aiBonus: { abstractThinking: 1.15, patternRecognition: 1.22 }
  },
  {
    id: 'non_euclidean_geometry',
    name: 'Non-Euclidean Geometry',
    category: 'geometry',
    complexity: 6,
    prerequisiteIds: ['projective_geometry'],
    description: 'Curved spaces and alternative axioms',
    flavorText: 'What if parallel lines could meet? Hyperbolic and spherical worlds.',
    aiBonus: { decisionQuality: 1.25, patternRecognition: 1.30, abstractionLevel: 1.15, curiosity: 1.12 }
  },
  {
    id: 'computational_geometry',
    name: 'Computational Geometry',
    category: 'geometry',
    complexity: 7,
    prerequisiteIds: ['vectors_geometry', 'algorithms'],
    description: 'Algorithms for geometric problems',
    flavorText: 'Convex hulls, Voronoi diagrams, triangulations. Geometry meets computation.',
    aiBonus: { computationalSpeed: 1.25, patternRecognition: 1.25, optimizationPower: 1.20 }
  },
  
  // ============================================================================
  // ALGEBRA (Level 3-9)
  // Variables, equations, and abstract structures
  // ============================================================================
  {
    id: 'variables',
    name: 'Variables & Expressions',
    category: 'algebra',
    complexity: 3,
    prerequisiteIds: ['multiplication_division'],
    description: 'Symbolic representation of unknowns',
    flavorText: 'Let x be... The power of naming the unknown.',
    aiBonus: { decisionQuality: 1.15, explorationBonus: 0.05, abstractThinking: 1.08 }
  },
  {
    id: 'equations',
    name: 'Equations',
    category: 'algebra',
    complexity: 3,
    prerequisiteIds: ['variables'],
    description: 'Statements of equality to solve',
    flavorText: 'Find x. The eternal quest of algebra.',
    aiBonus: { decisionQuality: 1.12, optimizationPower: 1.08 }
  },
  {
    id: 'inequalities',
    name: 'Inequalities',
    category: 'algebra',
    complexity: 4,
    prerequisiteIds: ['equations', 'ordering'],
    description: 'Statements of comparison',
    flavorText: 'Greater than, less than, within bounds. Constraints and feasibility.',
    aiBonus: { optimizationPower: 1.12, decisionQuality: 1.10 }
  },
  {
    id: 'functions',
    name: 'Functions',
    category: 'algebra',
    complexity: 4,
    prerequisiteIds: ['equations'],
    description: 'Input-output relationships',
    flavorText: 'Feed in x, get out f(x). The mathematical machine.',
    aiBonus: { decisionQuality: 1.20, patternRecognition: 1.25, abstractThinking: 1.10 }
  },
  {
    id: 'linear_functions',
    name: 'Linear Functions',
    category: 'algebra',
    complexity: 4,
    prerequisiteIds: ['functions', 'coordinate_geometry'],
    description: 'Straight line relationships',
    flavorText: 'y = mx + b. The simplest function, the straightest line.',
    aiBonus: { decisionQuality: 1.15, computationalSpeed: 1.10 }
  },
  {
    id: 'quadratic_functions',
    name: 'Quadratic Functions',
    category: 'algebra',
    complexity: 4,
    prerequisiteIds: ['linear_functions', 'exponents'],
    description: 'Parabolic curves and square terms',
    flavorText: 'ax² + bx + c. Curves that rise and fall, or fall and rise.',
    aiBonus: { patternRecognition: 1.18, decisionQuality: 1.15 }
  },
  {
    id: 'polynomials',
    name: 'Polynomials',
    category: 'algebra',
    complexity: 5,
    prerequisiteIds: ['quadratic_functions'],
    description: 'Higher-degree equations',
    flavorText: 'Cubic, quartic, quintic... More wiggles, more roots.',
    aiBonus: { decisionQuality: 1.22, optimizationPower: 1.15 }
  },
  {
    id: 'exponentials_logarithms',
    name: 'Exponentials & Logarithms',
    category: 'algebra',
    complexity: 5,
    prerequisiteIds: ['polynomials', 'exponents'],
    description: 'Growth and scaling functions',
    flavorText: 'Explosive growth and its inverse. e ≈ 2.71828...',
    aiBonus: { decisionQuality: 1.25, patternRecognition: 1.30, optimizationPower: 1.20 }
  },
  {
    id: 'sequences_series',
    name: 'Sequences & Series',
    category: 'algebra',
    complexity: 5,
    prerequisiteIds: ['functions'],
    description: 'Ordered lists and their sums',
    flavorText: 'Arithmetic, geometric, Fibonacci. Patterns that unfold.',
    aiBonus: { patternRecognition: 1.25, curiosity: 1.10 }
  },
  {
    id: 'systems_of_equations',
    name: 'Systems of Equations',
    category: 'algebra',
    complexity: 5,
    prerequisiteIds: ['linear_functions'],
    description: 'Multiple equations, multiple unknowns',
    flavorText: 'Where do the lines cross? Finding the intersection.',
    aiBonus: { decisionQuality: 1.20, computationalSpeed: 1.15 }
  },
  {
    id: 'matrices',
    name: 'Matrices',
    category: 'algebra',
    complexity: 6,
    prerequisiteIds: ['systems_of_equations', 'vectors_geometry'],
    description: 'Rectangular arrays of numbers',
    flavorText: 'Numbers in rows and columns. Transformations in disguise.',
    aiBonus: { computationalSpeed: 1.20, patternRecognition: 1.18 }
  },
  {
    id: 'linear_algebra',
    name: 'Linear Algebra',
    category: 'algebra',
    complexity: 6,
    prerequisiteIds: ['matrices'],
    description: 'Vectors, matrices, and transformations',
    flavorText: 'The mathematics of high dimensions. Eigenvalues emerge.',
    aiBonus: { decisionQuality: 1.30, computationalSpeed: 1.20, optimizationPower: 1.25 }
  },
  {
    id: 'eigenvalues',
    name: 'Eigenvalues & Eigenvectors',
    category: 'algebra',
    complexity: 7,
    prerequisiteIds: ['linear_algebra'],
    description: 'Special vectors under linear transformations',
    flavorText: 'Vectors that only scale, never rotate. The DNA of matrices.',
    aiBonus: { patternRecognition: 1.30, decisionQuality: 1.28, abstractThinking: 1.18 }
  },
  {
    id: 'polynomial_rings',
    name: 'Polynomial Rings',
    category: 'algebra',
    complexity: 8,
    prerequisiteIds: ['polynomials', 'group_theory'],
    description: 'Algebraic structures of polynomials',
    flavorText: 'Polynomials form a ring. Addition and multiplication, but division fails.',
    aiBonus: { abstractThinking: 1.25, decisionQuality: 1.25 }
  },
  {
    id: 'abstract_algebra',
    name: 'Abstract Algebra',
    category: 'algebra',
    complexity: 8,
    prerequisiteIds: ['polynomial_rings', 'group_theory'],
    description: 'Rings, fields, and algebraic structures',
    flavorText: 'The algebra of algebras. Structure over calculation.',
    aiBonus: { decisionQuality: 1.45, abstractionLevel: 1.30, optimizationPower: 1.35, abstractThinking: 1.30 }
  },
  {
    id: 'galois_theory',
    name: 'Galois Theory',
    category: 'algebra',
    complexity: 9,
    prerequisiteIds: ['abstract_algebra'],
    description: 'Symmetries of polynomial roots',
    flavorText: 'Why there\'s no quintic formula. Galois died at 20, leaving this treasure.',
    aiBonus: { abstractThinking: 1.40, patternRecognition: 1.35, curiosity: 1.20 }
  },
  
  // ============================================================================
  // CALCULUS (Level 5-10)
  // Continuous change, limits, and infinitesimals
  // ============================================================================
  {
    id: 'limits',
    name: 'Limits',
    category: 'calculus',
    complexity: 5,
    prerequisiteIds: ['exponentials_logarithms', 'trigonometry'],
    description: 'Behavior as values approach a target',
    flavorText: 'Getting closer and closer, without arriving. The foundation of calculus.',
    aiBonus: { decisionQuality: 1.28, optimizationPower: 1.25, curiosity: 1.08 }
  },
  {
    id: 'continuity',
    name: 'Continuity',
    category: 'calculus',
    complexity: 5,
    prerequisiteIds: ['limits'],
    description: 'Functions without jumps or breaks',
    flavorText: 'You can draw it without lifting your pen. Smoothness formalized.',
    aiBonus: { patternRecognition: 1.18, abstractThinking: 1.12 }
  },
  {
    id: 'derivatives',
    name: 'Derivatives',
    category: 'calculus',
    complexity: 6,
    prerequisiteIds: ['limits'],
    description: 'Rates of change and slopes',
    flavorText: 'How fast is it changing? The tangent line at every point.',
    aiBonus: { decisionQuality: 1.32, optimizationPower: 1.30, explorationBonus: 0.08 }
  },
  {
    id: 'derivative_rules',
    name: 'Derivative Rules',
    category: 'calculus',
    complexity: 6,
    prerequisiteIds: ['derivatives'],
    description: 'Product, quotient, and chain rules',
    flavorText: 'Shortcuts for differentiation. The calculus cookbook.',
    aiBonus: { computationalSpeed: 1.20, decisionQuality: 1.18 }
  },
  {
    id: 'optimization_calculus',
    name: 'Optimization with Calculus',
    category: 'calculus',
    complexity: 6,
    prerequisiteIds: ['derivative_rules'],
    description: 'Finding maxima and minima',
    flavorText: 'Where derivatives equal zero, extrema live. Find the best.',
    aiBonus: { optimizationPower: 1.35, decisionQuality: 1.25 }
  },
  {
    id: 'integrals',
    name: 'Integrals',
    category: 'calculus',
    complexity: 6,
    prerequisiteIds: ['derivatives'],
    description: 'Accumulation and area',
    flavorText: 'Add up infinitely many infinitely thin slices. Area under the curve.',
    aiBonus: { decisionQuality: 1.35, optimizationPower: 1.35 }
  },
  {
    id: 'fundamental_theorem',
    name: 'Fundamental Theorem of Calculus',
    category: 'calculus',
    complexity: 7,
    prerequisiteIds: ['integrals'],
    description: 'Derivatives and integrals are inverses',
    flavorText: 'The most important theorem. Differentiation and integration are opposites.',
    aiBonus: { abstractThinking: 1.20, decisionQuality: 1.30, proofIntuition: 1.15 }
  },
  {
    id: 'differential_equations',
    name: 'Differential Equations',
    category: 'calculus',
    complexity: 7,
    prerequisiteIds: ['fundamental_theorem'],
    description: 'Equations involving rates of change',
    flavorText: 'The language of physics. How things evolve in time.',
    aiBonus: { decisionQuality: 1.40, patternRecognition: 1.40, optimizationPower: 1.40 }
  },
  {
    id: 'partial_derivatives',
    name: 'Partial Derivatives',
    category: 'calculus',
    complexity: 7,
    prerequisiteIds: ['derivative_rules', 'linear_algebra'],
    description: 'Derivatives with multiple variables',
    flavorText: 'Hold everything else constant, change just one thing.',
    aiBonus: { decisionQuality: 1.35, computationalSpeed: 1.22 }
  },
  {
    id: 'multivariable_calculus',
    name: 'Multivariable Calculus',
    category: 'calculus',
    complexity: 8,
    prerequisiteIds: ['partial_derivatives'],
    description: 'Calculus in multiple dimensions',
    flavorText: 'Gradients, divergence, curl. Calculus goes 3D and beyond.',
    aiBonus: { decisionQuality: 1.45, optimizationPower: 1.45, computationalSpeed: 1.25 }
  },
  {
    id: 'vector_calculus',
    name: 'Vector Calculus',
    category: 'calculus',
    complexity: 8,
    prerequisiteIds: ['multivariable_calculus', 'vectors_geometry'],
    description: 'Calculus of vector fields',
    flavorText: 'Stokes, Gauss, Green. The great theorems unify it all.',
    aiBonus: { patternRecognition: 1.40, abstractThinking: 1.28, decisionQuality: 1.42 }
  },
  {
    id: 'tensor_calculus',
    name: 'Tensor Calculus',
    category: 'calculus',
    complexity: 9,
    prerequisiteIds: ['vector_calculus'],
    description: 'Multi-dimensional array mathematics',
    flavorText: 'Scalars, vectors, matrices, and beyond. Einstein\'s tool for relativity.',
    aiBonus: { decisionQuality: 1.55, computationalSpeed: 1.40, abstractionLevel: 1.35, abstractThinking: 1.35 }
  },
  {
    id: 'calculus_of_variations',
    name: 'Calculus of Variations',
    category: 'calculus',
    complexity: 9,
    prerequisiteIds: ['differential_equations', 'multivariable_calculus'],
    description: 'Optimizing functions of functions',
    flavorText: 'What path minimizes travel time? Euler and Lagrange found the answer.',
    aiBonus: { optimizationPower: 1.55, decisionQuality: 1.48, proofIntuition: 1.22 }
  },
  {
    id: 'fractional_calculus',
    name: 'Fractional Calculus',
    category: 'calculus',
    complexity: 10,
    prerequisiteIds: ['tensor_calculus'],
    description: 'Non-integer order derivatives',
    flavorText: 'What\'s a half-derivative? Calculus between the integers.',
    aiBonus: { abstractThinking: 1.40, curiosity: 1.25, decisionQuality: 1.50 }
  },
  
  // ============================================================================
  // STATISTICS (Level 4-10)
  // Probability, data, and inference
  // ============================================================================
  {
    id: 'descriptive_stats',
    name: 'Descriptive Statistics',
    category: 'statistics',
    complexity: 4,
    prerequisiteIds: ['addition_subtraction', 'multiplication_division'],
    description: 'Mean, median, mode, range',
    flavorText: 'Summarize the data. What\'s typical? How spread out?',
    aiBonus: { decisionQuality: 1.15, patternRecognition: 1.12 }
  },
  {
    id: 'probability_basics',
    name: 'Basic Probability',
    category: 'statistics',
    complexity: 5,
    prerequisiteIds: ['fractions', 'descriptive_stats'],
    description: 'Likelihood and chance',
    flavorText: 'What are the odds? The mathematics of uncertainty.',
    aiBonus: { decisionQuality: 1.25, explorationBonus: 0.08 }
  },
  {
    id: 'probability',
    name: 'Probability Theory',
    category: 'statistics',
    complexity: 5,
    prerequisiteIds: ['probability_basics'],
    description: 'Formal probability axioms',
    flavorText: 'Kolmogorov\'s axioms. Probability as measure.',
    aiBonus: { decisionQuality: 1.30, explorationBonus: 0.10 }
  },
  {
    id: 'conditional_probability',
    name: 'Conditional Probability',
    category: 'statistics',
    complexity: 5,
    prerequisiteIds: ['probability'],
    description: 'Probability given information',
    flavorText: 'If we know A, what\'s the chance of B? Updating beliefs.',
    aiBonus: { decisionQuality: 1.28, informationProcessing: 1.15 }
  },
  {
    id: 'combinatorics_basic',
    name: 'Basic Combinatorics',
    category: 'statistics',
    complexity: 5,
    prerequisiteIds: ['probability'],
    description: 'Counting permutations and combinations',
    flavorText: 'How many ways? n! and n choose k.',
    aiBonus: { computationalSpeed: 1.15, patternRecognition: 1.18 }
  },
  {
    id: 'random_variables',
    name: 'Random Variables',
    category: 'statistics',
    complexity: 6,
    prerequisiteIds: ['probability', 'functions'],
    description: 'Numerical outcomes of random processes',
    flavorText: 'Assign numbers to chance. Expected value emerges.',
    aiBonus: { decisionQuality: 1.32, patternRecognition: 1.25 }
  },
  {
    id: 'distributions',
    name: 'Probability Distributions',
    category: 'statistics',
    complexity: 6,
    prerequisiteIds: ['random_variables', 'integrals'],
    description: 'Patterns in random data',
    flavorText: 'Normal, Poisson, exponential. The shapes of chance.',
    aiBonus: { decisionQuality: 1.35, patternRecognition: 1.45 }
  },
  {
    id: 'central_limit_theorem',
    name: 'Central Limit Theorem',
    category: 'statistics',
    complexity: 7,
    prerequisiteIds: ['distributions'],
    description: 'Averages tend toward normal distribution',
    flavorText: 'The most important theorem in statistics. Everything becomes normal.',
    aiBonus: { patternRecognition: 1.40, decisionQuality: 1.35, proofIntuition: 1.15 }
  },
  {
    id: 'bayesian_inference',
    name: 'Bayesian Inference',
    category: 'statistics',
    complexity: 7,
    prerequisiteIds: ['conditional_probability', 'distributions'],
    description: 'Updating beliefs with evidence',
    flavorText: 'Prior × Likelihood ∝ Posterior. Learning from data.',
    aiBonus: { decisionQuality: 1.45, patternRecognition: 1.50, optimizationPower: 1.45, informationProcessing: 1.25 }
  },
  {
    id: 'hypothesis_testing',
    name: 'Hypothesis Testing',
    category: 'statistics',
    complexity: 7,
    prerequisiteIds: ['central_limit_theorem'],
    description: 'Statistical significance and p-values',
    flavorText: 'Is it real or chance? Making decisions under uncertainty.',
    aiBonus: { decisionQuality: 1.38, proofIntuition: 1.18 }
  },
  {
    id: 'regression',
    name: 'Regression Analysis',
    category: 'statistics',
    complexity: 7,
    prerequisiteIds: ['hypothesis_testing', 'linear_algebra'],
    description: 'Fitting models to data',
    flavorText: 'Find the line of best fit. Prediction from patterns.',
    aiBonus: { patternRecognition: 1.45, optimizationPower: 1.40, decisionQuality: 1.40 }
  },
  {
    id: 'information_theory',
    name: 'Information Theory',
    category: 'statistics',
    complexity: 8,
    prerequisiteIds: ['bayesian_inference'],
    description: 'Entropy, compression, and communication',
    flavorText: 'Shannon\'s entropy. Information measured in bits.',
    aiBonus: { decisionQuality: 1.52, patternRecognition: 1.55, computationalSpeed: 1.30, informationProcessing: 1.35 }
  },
  {
    id: 'stochastic_processes',
    name: 'Stochastic Processes',
    category: 'statistics',
    complexity: 8,
    prerequisiteIds: ['bayesian_inference', 'differential_equations'],
    description: 'Random processes evolving over time',
    flavorText: 'Random walks, Markov chains, Brownian motion. Randomness in time.',
    aiBonus: { decisionQuality: 1.50, patternRecognition: 1.52, explorationBonus: 0.12 }
  },
  {
    id: 'monte_carlo',
    name: 'Monte Carlo Methods',
    category: 'statistics',
    complexity: 8,
    prerequisiteIds: ['stochastic_processes'],
    description: 'Simulation-based computation',
    flavorText: 'Roll the dice a million times. Approximate the unapproximable.',
    aiBonus: { computationalSpeed: 1.35, optimizationPower: 1.40, explorationBonus: 0.15 }
  },
  {
    id: 'measure_theory',
    name: 'Measure Theory',
    category: 'statistics',
    complexity: 9,
    prerequisiteIds: ['information_theory', 'topology_basics'],
    description: 'Rigorous foundations of probability',
    flavorText: 'Lebesgue\'s measure. Making probability mathematically rigorous.',
    aiBonus: { abstractThinking: 1.40, proofIntuition: 1.30, decisionQuality: 1.52 }
  },
  {
    id: 'extreme_value_theory',
    name: 'Extreme Value Theory',
    category: 'statistics',
    complexity: 9,
    prerequisiteIds: ['measure_theory'],
    description: 'Statistics of rare events',
    flavorText: 'The largest, the smallest, the rarest. Tail risks quantified.',
    aiBonus: { decisionQuality: 1.55, patternRecognition: 1.50, curiosity: 1.18 }
  },
  
  // ============================================================================
  // COMBINATORICS (Level 5-9) - NEW CATEGORY
  // Counting, arrangements, and discrete structures
  // ============================================================================
  {
    id: 'permutations',
    name: 'Permutations',
    category: 'combinatorics',
    complexity: 5,
    prerequisiteIds: ['combinatorics_basic'],
    description: 'Ordered arrangements',
    flavorText: 'How many ways to arrange n things? n! possibilities.',
    aiBonus: { computationalSpeed: 1.18, patternRecognition: 1.20 }
  },
  {
    id: 'combinations',
    name: 'Combinations',
    category: 'combinatorics',
    complexity: 5,
    prerequisiteIds: ['permutations'],
    description: 'Unordered selections',
    flavorText: 'Choose k from n, order doesn\'t matter. The binomial coefficients.',
    aiBonus: { patternRecognition: 1.22, decisionQuality: 1.15 }
  },
  {
    id: 'generating_functions',
    name: 'Generating Functions',
    category: 'combinatorics',
    complexity: 6,
    prerequisiteIds: ['combinations', 'sequences_series'],
    description: 'Power series for counting',
    flavorText: 'Encode sequences as functions. Transform counting into algebra.',
    aiBonus: { computationalSpeed: 1.25, patternRecognition: 1.28, abstractThinking: 1.15 }
  },
  {
    id: 'recurrence_relations',
    name: 'Recurrence Relations',
    category: 'combinatorics',
    complexity: 6,
    prerequisiteIds: ['generating_functions'],
    description: 'Sequences defined by previous terms',
    flavorText: 'F(n) = F(n-1) + F(n-2). The Fibonacci pattern.',
    aiBonus: { patternRecognition: 1.30, computationalSpeed: 1.22 }
  },
  {
    id: 'inclusion_exclusion',
    name: 'Inclusion-Exclusion',
    category: 'combinatorics',
    complexity: 6,
    prerequisiteIds: ['combinations', 'set_theory'],
    description: 'Counting with overlaps',
    flavorText: 'Add, subtract, add, subtract. Correct for overcounting.',
    aiBonus: { decisionQuality: 1.25, patternRecognition: 1.25, informationProcessing: 1.15 }
  },
  {
    id: 'graph_theory',
    name: 'Graph Theory',
    category: 'combinatorics',
    complexity: 7,
    prerequisiteIds: ['inclusion_exclusion'],
    description: 'Vertices, edges, and connectivity',
    flavorText: 'Nodes and links. The mathematics of networks.',
    aiBonus: { patternRecognition: 1.35, cooperationBonus: 1.15, informationProcessing: 1.20 }
  },
  {
    id: 'extremal_combinatorics',
    name: 'Extremal Combinatorics',
    category: 'combinatorics',
    complexity: 8,
    prerequisiteIds: ['graph_theory'],
    description: 'Maximum and minimum configurations',
    flavorText: 'How few edges guarantee a triangle? Turán\'s question.',
    aiBonus: { optimizationPower: 1.35, patternRecognition: 1.38, proofIntuition: 1.20 }
  },
  {
    id: 'ramsey_theory',
    name: 'Ramsey Theory',
    category: 'combinatorics',
    complexity: 9,
    prerequisiteIds: ['extremal_combinatorics'],
    description: 'Order from chaos in large structures',
    flavorText: 'In large enough structures, order is inevitable. Complete disorder is impossible.',
    aiBonus: { patternRecognition: 1.45, abstractThinking: 1.35, curiosity: 1.20 }
  },
  {
    id: 'probabilistic_combinatorics',
    name: 'Probabilistic Method',
    category: 'combinatorics',
    complexity: 9,
    prerequisiteIds: ['ramsey_theory', 'probability'],
    description: 'Existence proofs via probability',
    flavorText: 'It exists because it\'s probably there. Erdős\'s revolutionary method.',
    aiBonus: { proofIntuition: 1.35, explorationBonus: 0.15, curiosity: 1.22 }
  },
  
  // ============================================================================
  // NUMBER THEORY (Level 4-11)
  // Properties of integers and prime numbers
  // ============================================================================
  {
    id: 'divisibility',
    name: 'Divisibility',
    category: 'number_theory',
    complexity: 4,
    prerequisiteIds: ['multiplication_division'],
    description: 'When one number divides another evenly',
    flavorText: 'No remainder. The fundamental relation between integers.',
    aiBonus: { patternRecognition: 1.15, computationalSpeed: 1.10 }
  },
  {
    id: 'prime_numbers',
    name: 'Prime Numbers',
    category: 'number_theory',
    complexity: 4,
    prerequisiteIds: ['divisibility'],
    description: 'Fundamental building blocks of integers',
    flavorText: 'Divisible only by 1 and themselves. The atoms of arithmetic.',
    aiBonus: { patternRecognition: 1.25, computationalSpeed: 1.15, curiosity: 1.10 }
  },
  {
    id: 'fundamental_theorem_arithmetic',
    name: 'Fundamental Theorem of Arithmetic',
    category: 'number_theory',
    complexity: 5,
    prerequisiteIds: ['prime_numbers'],
    description: 'Unique prime factorization',
    flavorText: 'Every integer factors uniquely into primes. The backbone of number theory.',
    aiBonus: { patternRecognition: 1.28, proofIntuition: 1.12 }
  },
  {
    id: 'gcd_lcm',
    name: 'GCD & LCM',
    category: 'number_theory',
    complexity: 5,
    prerequisiteIds: ['fundamental_theorem_arithmetic'],
    description: 'Greatest common divisor and least common multiple',
    flavorText: 'What do numbers share? Where do they meet?',
    aiBonus: { computationalSpeed: 1.18, optimizationPower: 1.12 }
  },
  {
    id: 'euclidean_algorithm',
    name: 'Euclidean Algorithm',
    category: 'number_theory',
    complexity: 5,
    prerequisiteIds: ['gcd_lcm'],
    description: 'Fast GCD computation',
    flavorText: 'The oldest algorithm still in daily use. 2300 years and counting.',
    aiBonus: { computationalSpeed: 1.22, decisionQuality: 1.15 }
  },
  {
    id: 'diophantine_equations',
    name: 'Diophantine Equations',
    category: 'number_theory',
    complexity: 6,
    prerequisiteIds: ['euclidean_algorithm'],
    description: 'Integer solutions to polynomial equations',
    flavorText: 'Find integers x and y such that... The ancient quest.',
    aiBonus: { patternRecognition: 1.30, proofIntuition: 1.18, curiosity: 1.12 }
  },
  {
    id: 'quadratic_residues',
    name: 'Quadratic Residues',
    category: 'number_theory',
    complexity: 6,
    prerequisiteIds: ['modular_arithmetic', 'prime_numbers'],
    description: 'Squares modulo primes',
    flavorText: 'Is a number a perfect square mod p? Legendre knew.',
    aiBonus: { patternRecognition: 1.28, abstractThinking: 1.15 }
  },
  {
    id: 'cryptographic_math',
    name: 'Cryptographic Mathematics',
    category: 'number_theory',
    complexity: 7,
    prerequisiteIds: ['quadratic_residues', 'exponentials_logarithms'],
    description: 'Mathematical foundations of encryption',
    flavorText: 'RSA, discrete logs, elliptic curves. Security from hard problems.',
    aiBonus: { decisionQuality: 1.40, computationalSpeed: 1.35, patternRecognition: 1.40, informationProcessing: 1.25 }
  },
  {
    id: 'prime_distribution',
    name: 'Prime Number Distribution',
    category: 'number_theory',
    complexity: 8,
    prerequisiteIds: ['cryptographic_math'],
    description: 'How primes are distributed among integers',
    flavorText: 'The prime number theorem. Primes thin out logarithmically.',
    aiBonus: { patternRecognition: 1.42, curiosity: 1.20, abstractThinking: 1.25 }
  },
  {
    id: 'analytic_number_theory',
    name: 'Analytic Number Theory',
    category: 'number_theory',
    complexity: 9,
    prerequisiteIds: ['prime_distribution', 'complex_analysis'],
    description: 'Number theory via analysis',
    flavorText: 'The Riemann zeta function. Where analysis meets primes.',
    aiBonus: { decisionQuality: 1.52, patternRecognition: 1.48, abstractThinking: 1.35 }
  },
  {
    id: 'algebraic_number_theory',
    name: 'Algebraic Number Theory',
    category: 'number_theory',
    complexity: 9,
    prerequisiteIds: ['analytic_number_theory', 'abstract_algebra'],
    description: 'Algebraic structures within number systems',
    flavorText: 'Number fields, rings of integers. Algebra illuminates arithmetic.',
    aiBonus: { decisionQuality: 1.55, abstractionLevel: 1.40, patternRecognition: 1.50, abstractThinking: 1.38 }
  },
  {
    id: 'elliptic_curves',
    name: 'Elliptic Curves',
    category: 'number_theory',
    complexity: 10,
    prerequisiteIds: ['algebraic_number_theory'],
    description: 'Cubic curves and their remarkable structure',
    flavorText: 'y² = x³ + ax + b. The key to Fermat\'s Last Theorem.',
    aiBonus: { patternRecognition: 1.55, decisionQuality: 1.55, abstractThinking: 1.42, curiosity: 1.22 }
  },
  {
    id: 'langlands_program',
    name: 'Langlands Program',
    category: 'number_theory',
    complexity: 11,
    prerequisiteIds: ['elliptic_curves', 'representation_theory'],
    description: 'Grand unified theory of mathematics',
    flavorText: 'Connecting number theory, geometry, and representation theory. The Rosetta Stone.',
    aiBonus: { abstractThinking: 1.55, patternRecognition: 1.58, cooperationBonus: 1.25, proofIntuition: 1.40 }
  },
  
  // ============================================================================
  // LOGIC (Level 4-11)
  // Formal reasoning and metamathematics
  // ============================================================================
  {
    id: 'propositional_logic',
    name: 'Propositional Logic',
    category: 'logic',
    complexity: 4,
    prerequisiteIds: ['variables'],
    description: 'Boolean logic and truth tables',
    flavorText: 'AND, OR, NOT, IF-THEN. The building blocks of reasoning.',
    aiBonus: { decisionQuality: 1.25, computationalSpeed: 1.20, informationProcessing: 1.15 }
  },
  {
    id: 'logical_proofs',
    name: 'Proof Techniques',
    category: 'logic',
    complexity: 5,
    prerequisiteIds: ['propositional_logic'],
    description: 'Direct proof, contraposition, contradiction',
    flavorText: 'How to prove things. The methods of mathematical argument.',
    aiBonus: { proofIntuition: 1.25, decisionQuality: 1.22 }
  },
  {
    id: 'induction',
    name: 'Mathematical Induction',
    category: 'logic',
    complexity: 5,
    prerequisiteIds: ['logical_proofs'],
    description: 'Proof by induction',
    flavorText: 'Prove it for 1, prove it carries forward. Dominos fall forever.',
    aiBonus: { proofIntuition: 1.30, patternRecognition: 1.20, abstractThinking: 1.12 }
  },
  {
    id: 'predicate_logic',
    name: 'Predicate Logic',
    category: 'logic',
    complexity: 6,
    prerequisiteIds: ['induction', 'set_theory'],
    description: 'Quantifiers and formal reasoning',
    flavorText: 'For all x, there exists y. Making quantification precise.',
    aiBonus: { decisionQuality: 1.38, abstractionLevel: 1.25, patternRecognition: 1.35, informationProcessing: 1.22 }
  },
  {
    id: 'formal_systems',
    name: 'Formal Systems',
    category: 'logic',
    complexity: 7,
    prerequisiteIds: ['predicate_logic'],
    description: 'Axioms, rules, and theorems',
    flavorText: 'Mathematics as a game with symbols. Hilbert\'s dream.',
    aiBonus: { abstractThinking: 1.28, proofIntuition: 1.25 }
  },
  {
    id: 'model_theory',
    name: 'Model Theory',
    category: 'logic',
    complexity: 8,
    prerequisiteIds: ['formal_systems'],
    description: 'Interpretations of formal languages',
    flavorText: 'What structures satisfy these axioms? Semantics meets syntax.',
    aiBonus: { abstractThinking: 1.35, decisionQuality: 1.42, patternRecognition: 1.38 }
  },
  {
    id: 'modal_logic',
    name: 'Modal Logic',
    category: 'logic',
    complexity: 8,
    prerequisiteIds: ['model_theory'],
    description: 'Possibility, necessity, and belief',
    flavorText: 'What could be, what must be. Reasoning about possibilities.',
    aiBonus: { decisionQuality: 1.50, abstractionLevel: 1.40, explorationBonus: 0.15, curiosity: 1.18 }
  },
  {
    id: 'proof_theory',
    name: 'Proof Theory',
    category: 'logic',
    complexity: 9,
    prerequisiteIds: ['formal_systems'],
    description: 'Mathematical study of proofs',
    flavorText: 'Proofs as mathematical objects. Studying reasoning itself.',
    aiBonus: { proofIntuition: 1.45, abstractThinking: 1.40, informationProcessing: 1.30 }
  },
  {
    id: 'recursion_theory',
    name: 'Recursion Theory',
    category: 'logic',
    complexity: 9,
    prerequisiteIds: ['proof_theory'],
    description: 'Computability and recursive functions',
    flavorText: 'What can be computed? Turing and Church drew the boundaries.',
    aiBonus: { computationalSpeed: 1.45, informationProcessing: 1.40, abstractThinking: 1.35 }
  },
  {
    id: 'godel_incompleteness',
    name: 'Gödel Incompleteness',
    category: 'logic',
    complexity: 10,
    prerequisiteIds: ['recursion_theory'],
    description: 'Limits of formal systems and self-reference',
    flavorText: 'True but unprovable. Gödel shattered Hilbert\'s dream.',
    aiBonus: { decisionQuality: 1.60, abstractionLevel: 1.55, explorationBonus: 0.18, curiosity: 1.25, proofIntuition: 1.35 }
  },
  {
    id: 'type_theory',
    name: 'Type Theory',
    category: 'logic',
    complexity: 10,
    prerequisiteIds: ['godel_incompleteness'],
    description: 'Types as propositions, programs as proofs',
    flavorText: 'The Curry-Howard correspondence. Logic is computation.',
    aiBonus: { informationProcessing: 1.50, computationalSpeed: 1.45, proofIntuition: 1.40, abstractThinking: 1.42 }
  },
  {
    id: 'constructive_logic',
    name: 'Constructive Logic',
    category: 'logic',
    complexity: 10,
    prerequisiteIds: ['type_theory'],
    description: 'Logic where existence requires construction',
    flavorText: 'Don\'t just prove it exists—build it. Brouwer\'s intuitionistic vision.',
    aiBonus: { proofIntuition: 1.50, informationProcessing: 1.45, decisionQuality: 1.55 }
  },
  
  // ============================================================================
  // ANALYSIS (Level 6-11) - NEW CATEGORY
  // Rigorous study of limits, continuity, and convergence
  // ============================================================================
  {
    id: 'real_analysis_intro',
    name: 'Real Analysis Basics',
    category: 'analysis',
    complexity: 6,
    prerequisiteIds: ['limits', 'logical_proofs'],
    description: 'Rigorous foundations of calculus',
    flavorText: 'Epsilon-delta definitions. Making calculus honest.',
    aiBonus: { proofIntuition: 1.20, abstractThinking: 1.18 }
  },
  {
    id: 'sequences_convergence',
    name: 'Sequences & Convergence',
    category: 'analysis',
    complexity: 6,
    prerequisiteIds: ['real_analysis_intro'],
    description: 'Limits of sequences and series',
    flavorText: 'Does the sequence approach a limit? Cauchy knew how to check.',
    aiBonus: { patternRecognition: 1.25, proofIntuition: 1.22 }
  },
  {
    id: 'metric_spaces',
    name: 'Metric Spaces',
    category: 'analysis',
    complexity: 7,
    prerequisiteIds: ['sequences_convergence'],
    description: 'Abstract distance functions',
    flavorText: 'Generalize distance. Any function satisfying the triangle inequality.',
    aiBonus: { abstractThinking: 1.28, patternRecognition: 1.28 }
  },
  {
    id: 'uniform_convergence',
    name: 'Uniform Convergence',
    category: 'analysis',
    complexity: 7,
    prerequisiteIds: ['sequences_convergence'],
    description: 'Convergence independent of point',
    flavorText: 'Pointwise isn\'t enough. Uniform convergence preserves continuity.',
    aiBonus: { proofIntuition: 1.28, decisionQuality: 1.25 }
  },
  {
    id: 'compactness',
    name: 'Compactness',
    category: 'analysis',
    complexity: 8,
    prerequisiteIds: ['metric_spaces'],
    description: 'Finite subcover property',
    flavorText: 'Closed and bounded in ℝⁿ. The key to many existence proofs.',
    aiBonus: { abstractThinking: 1.32, proofIntuition: 1.30 }
  },
  {
    id: 'complex_analysis',
    name: 'Complex Analysis',
    category: 'analysis',
    complexity: 8,
    prerequisiteIds: ['uniform_convergence', 'multivariable_calculus'],
    description: 'Calculus of complex functions',
    flavorText: 'Functions of complex variables. Surprisingly rigid and beautiful.',
    aiBonus: { patternRecognition: 1.45, decisionQuality: 1.42, abstractThinking: 1.35 }
  },
  {
    id: 'fourier_analysis',
    name: 'Fourier Analysis',
    category: 'analysis',
    complexity: 8,
    prerequisiteIds: ['complex_analysis'],
    description: 'Decomposition into frequencies',
    flavorText: 'Any function as a sum of sines and cosines. Hearing the shape of a signal.',
    aiBonus: { patternRecognition: 1.50, computationalSpeed: 1.35, informationProcessing: 1.30 }
  },
  {
    id: 'functional_analysis',
    name: 'Functional Analysis',
    category: 'analysis',
    complexity: 9,
    prerequisiteIds: ['compactness', 'linear_algebra'],
    description: 'Analysis of infinite-dimensional spaces',
    flavorText: 'Banach spaces, Hilbert spaces. Linear algebra goes infinite.',
    aiBonus: { abstractThinking: 1.45, decisionQuality: 1.48, optimizationPower: 1.40 }
  },
  {
    id: 'harmonic_analysis',
    name: 'Harmonic Analysis',
    category: 'analysis',
    complexity: 9,
    prerequisiteIds: ['fourier_analysis', 'group_theory'],
    description: 'Fourier analysis on groups',
    flavorText: 'Frequencies on abstract groups. Symmetry meets analysis.',
    aiBonus: { patternRecognition: 1.52, abstractThinking: 1.42 }
  },
  {
    id: 'spectral_theory',
    name: 'Spectral Theory',
    category: 'analysis',
    complexity: 10,
    prerequisiteIds: ['functional_analysis', 'eigenvalues'],
    description: 'Eigenvalues of operators',
    flavorText: 'The spectrum of an operator. Quantum mechanics speaks this language.',
    aiBonus: { decisionQuality: 1.55, patternRecognition: 1.55, abstractThinking: 1.48 }
  },
  {
    id: 'operator_algebras',
    name: 'Operator Algebras',
    category: 'analysis',
    complexity: 11,
    prerequisiteIds: ['spectral_theory'],
    description: 'Algebras of bounded operators',
    flavorText: 'C*-algebras and von Neumann algebras. The mathematics of quantum theory.',
    aiBonus: { abstractThinking: 1.55, decisionQuality: 1.58, cooperationBonus: 1.18 }
  },
  
  // ============================================================================
  // TOPOLOGY (Level 6-11)
  // Shapes, continuity, and invariants
  // ============================================================================
  {
    id: 'set_theory',
    name: 'Set Theory',
    category: 'topology',
    complexity: 6,
    prerequisiteIds: ['functions'],
    description: 'Collections and membership',
    flavorText: 'The empty set, unions, intersections. The foundation of modern mathematics.',
    aiBonus: { decisionQuality: 1.40, patternRecognition: 1.40, abstractThinking: 1.25 }
  },
  {
    id: 'relations',
    name: 'Relations',
    category: 'topology',
    complexity: 6,
    prerequisiteIds: ['set_theory'],
    description: 'Ordered pairs and correspondences',
    flavorText: 'Equivalence, order, functions. How things relate.',
    aiBonus: { patternRecognition: 1.35, abstractThinking: 1.22 }
  },
  {
    id: 'topology_basics',
    name: 'Point-Set Topology',
    category: 'topology',
    complexity: 8,
    prerequisiteIds: ['metric_spaces', 'set_theory'],
    description: 'Open sets, closed sets, continuity',
    flavorText: 'What is nearness without distance? Topology abstracts continuity.',
    aiBonus: { decisionQuality: 1.50, patternRecognition: 1.55, optimizationPower: 1.50, abstractThinking: 1.35 }
  },
  {
    id: 'connectedness',
    name: 'Connectedness',
    category: 'topology',
    complexity: 8,
    prerequisiteIds: ['topology_basics'],
    description: 'One piece or many?',
    flavorText: 'Can you get there from here without jumping? The topology of unity.',
    aiBonus: { patternRecognition: 1.42, cooperationBonus: 1.12 }
  },
  {
    id: 'quotient_spaces',
    name: 'Quotient Spaces',
    category: 'topology',
    complexity: 8,
    prerequisiteIds: ['topology_basics', 'relations'],
    description: 'Spaces formed by identification',
    flavorText: 'Glue edges together. The Möbius strip, the torus emerge.',
    aiBonus: { abstractThinking: 1.38, patternRecognition: 1.40 }
  },
  {
    id: 'fundamental_group',
    name: 'Fundamental Group',
    category: 'topology',
    complexity: 9,
    prerequisiteIds: ['quotient_spaces', 'group_theory'],
    description: 'Loops up to deformation',
    flavorText: 'Which loops can be shrunk to a point? The first algebraic invariant.',
    aiBonus: { patternRecognition: 1.52, abstractThinking: 1.45, proofIntuition: 1.28 }
  },
  {
    id: 'manifold_theory',
    name: 'Manifold Theory',
    category: 'topology',
    complexity: 9,
    prerequisiteIds: ['fundamental_group', 'multivariable_calculus'],
    description: 'Smooth curved spaces',
    flavorText: 'Locally like ℝⁿ, globally something else. The Earth looks flat up close.',
    aiBonus: { decisionQuality: 1.55, patternRecognition: 1.58, abstractionLevel: 1.45 }
  },
  {
    id: 'homology',
    name: 'Homology Theory',
    category: 'topology',
    complexity: 10,
    prerequisiteIds: ['fundamental_group', 'linear_algebra'],
    description: 'Holes and cycles',
    flavorText: 'Count the holes of each dimension. Algebraic invariants for shapes.',
    aiBonus: { patternRecognition: 1.58, abstractThinking: 1.52, proofIntuition: 1.35 }
  },
  {
    id: 'cohomology',
    name: 'Cohomology Theory',
    category: 'topology',
    complexity: 10,
    prerequisiteIds: ['homology'],
    description: 'Dual theory to homology',
    flavorText: 'Forms and integration. The dual perspective reveals more structure.',
    aiBonus: { abstractThinking: 1.55, patternRecognition: 1.55, decisionQuality: 1.52 }
  },
  {
    id: 'homotopy_theory',
    name: 'Homotopy Theory',
    category: 'topology',
    complexity: 11,
    prerequisiteIds: ['cohomology'],
    description: 'Continuous deformations',
    flavorText: 'When are two maps the same? Homotopy groups count the ways they differ.',
    aiBonus: { abstractThinking: 1.60, patternRecognition: 1.60, proofIntuition: 1.45 }
  },
  
  // ============================================================================
  // DIFFERENTIAL GEOMETRY (Level 8-11) - NEW CATEGORY
  // Calculus on curved spaces
  // ============================================================================
  {
    id: 'curves_surfaces',
    name: 'Curves & Surfaces',
    category: 'differential_geometry',
    complexity: 8,
    prerequisiteIds: ['multivariable_calculus', 'vectors_geometry'],
    description: 'Parametric curves and surfaces',
    flavorText: 'Describing curves with calculus. Curvature, torsion, the Frenet frame.',
    aiBonus: { patternRecognition: 1.40, computationalSpeed: 1.25 }
  },
  {
    id: 'riemannian_geometry',
    name: 'Riemannian Geometry',
    category: 'differential_geometry',
    complexity: 9,
    prerequisiteIds: ['manifold_theory', 'tensor_calculus'],
    description: 'Geometry on curved manifolds',
    flavorText: 'Metrics, geodesics, curvature tensors. The geometry of general relativity.',
    aiBonus: { patternRecognition: 1.50, abstractThinking: 1.45, decisionQuality: 1.48 }
  },
  {
    id: 'connections',
    name: 'Connections & Parallel Transport',
    category: 'differential_geometry',
    complexity: 9,
    prerequisiteIds: ['riemannian_geometry'],
    description: 'Moving vectors along curves',
    flavorText: 'What does it mean for a vector to stay constant? The connection tells us.',
    aiBonus: { abstractThinking: 1.48, patternRecognition: 1.45 }
  },
  {
    id: 'curvature_tensors',
    name: 'Curvature',
    category: 'differential_geometry',
    complexity: 10,
    prerequisiteIds: ['connections'],
    description: 'Measuring the bend of space',
    flavorText: 'Riemann, Ricci, scalar curvature. How space curves at each point.',
    aiBonus: { patternRecognition: 1.55, abstractThinking: 1.52, decisionQuality: 1.50 }
  },
  {
    id: 'lie_groups_diff',
    name: 'Lie Groups',
    category: 'differential_geometry',
    complexity: 10,
    prerequisiteIds: ['riemannian_geometry', 'group_theory'],
    description: 'Smooth groups',
    flavorText: 'Continuous symmetries form smooth manifolds. The rotation group SO(3).',
    aiBonus: { abstractThinking: 1.55, cooperationBonus: 1.18, patternRecognition: 1.52 }
  },
  {
    id: 'symplectic_geometry',
    name: 'Symplectic Geometry',
    category: 'differential_geometry',
    complexity: 10,
    prerequisiteIds: ['riemannian_geometry'],
    description: 'Geometry of phase space',
    flavorText: 'The natural geometry of classical mechanics. Area-preserving.',
    aiBonus: { optimizationPower: 1.50, abstractThinking: 1.50, decisionQuality: 1.52 }
  },
  {
    id: 'gauge_theory_math',
    name: 'Gauge Theory',
    category: 'differential_geometry',
    complexity: 11,
    prerequisiteIds: ['lie_groups_diff', 'connections'],
    description: 'Fiber bundles and connections',
    flavorText: 'The mathematics of the Standard Model. Physics shapes geometry.',
    aiBonus: { abstractThinking: 1.60, patternRecognition: 1.58, cooperationBonus: 1.22 }
  },
  
  // ============================================================================
  // OPTIMIZATION (Level 7-11)
  // Finding the best solution
  // ============================================================================
  {
    id: 'linear_programming',
    name: 'Linear Programming',
    category: 'optimization',
    complexity: 7,
    prerequisiteIds: ['systems_of_equations', 'inequalities'],
    description: 'Optimizing linear objectives with linear constraints',
    flavorText: 'The simplex method. Industrial optimization solved.',
    aiBonus: { optimizationPower: 1.40, decisionQuality: 1.35, computationalSpeed: 1.25 }
  },
  {
    id: 'convex_optimization',
    name: 'Convex Optimization',
    category: 'optimization',
    complexity: 7,
    prerequisiteIds: ['linear_programming', 'multivariable_calculus'],
    description: 'Finding optimal solutions in convex spaces',
    flavorText: 'Local = global. Convexity makes optimization tractable.',
    aiBonus: { optimizationPower: 1.50, computationalSpeed: 1.35, decisionQuality: 1.42 }
  },
  {
    id: 'dynamic_programming',
    name: 'Dynamic Programming',
    category: 'optimization',
    complexity: 7,
    prerequisiteIds: ['functions', 'probability'],
    description: 'Optimal substructure and memoization',
    flavorText: 'Bellman\'s principle: optimal paths have optimal subpaths.',
    aiBonus: { optimizationPower: 1.55, computationalSpeed: 1.45, decisionQuality: 1.45 }
  },
  {
    id: 'gradient_methods',
    name: 'Gradient Methods',
    category: 'optimization',
    complexity: 8,
    prerequisiteIds: ['derivatives', 'convex_optimization'],
    description: 'Optimization by following gradients',
    flavorText: 'Go downhill. Gradient descent finds local minima.',
    aiBonus: { optimizationPower: 1.55, computationalSpeed: 1.40, decisionQuality: 1.45 }
  },
  {
    id: 'game_theory',
    name: 'Game Theory',
    category: 'optimization',
    complexity: 8,
    prerequisiteIds: ['probability', 'convex_optimization'],
    description: 'Strategic decision making',
    flavorText: 'Nash equilibrium. What\'s optimal when others are also optimizing?',
    aiBonus: { decisionQuality: 1.55, explorationBonus: 0.15, optimizationPower: 1.45, cooperationBonus: 1.20 }
  },
  {
    id: 'constrained_optimization',
    name: 'Constrained Optimization',
    category: 'optimization',
    complexity: 8,
    prerequisiteIds: ['convex_optimization', 'multivariable_calculus'],
    description: 'Optimization with constraints',
    flavorText: 'Lagrange multipliers. The price of constraints.',
    aiBonus: { optimizationPower: 1.52, decisionQuality: 1.48 }
  },
  {
    id: 'control_theory',
    name: 'Control Theory',
    category: 'optimization',
    complexity: 9,
    prerequisiteIds: ['differential_equations', 'convex_optimization'],
    description: 'Feedback systems and stability',
    flavorText: 'PID controllers, Kalman filters. Making systems behave.',
    aiBonus: { optimizationPower: 1.60, decisionQuality: 1.52, computationalSpeed: 1.40 }
  },
  {
    id: 'stochastic_optimization',
    name: 'Stochastic Optimization',
    category: 'optimization',
    complexity: 9,
    prerequisiteIds: ['stochastic_processes', 'gradient_methods'],
    description: 'Optimization under uncertainty',
    flavorText: 'Optimize the expected. SGD powers modern machine learning.',
    aiBonus: { optimizationPower: 1.58, explorationBonus: 0.15, decisionQuality: 1.50 }
  },
  {
    id: 'combinatorial_optimization',
    name: 'Combinatorial Optimization',
    category: 'optimization',
    complexity: 9,
    prerequisiteIds: ['graph_theory', 'dynamic_programming'],
    description: 'Optimization over discrete structures',
    flavorText: 'Traveling salesman, knapsack, scheduling. NP-hard but essential.',
    aiBonus: { optimizationPower: 1.55, computationalSpeed: 1.45, patternRecognition: 1.40 }
  },
  {
    id: 'semidefinite_programming',
    name: 'Semidefinite Programming',
    category: 'optimization',
    complexity: 10,
    prerequisiteIds: ['convex_optimization', 'linear_algebra'],
    description: 'Optimization over positive semidefinite matrices',
    flavorText: 'Beyond LP. Relaxations that unlock hard problems.',
    aiBonus: { optimizationPower: 1.62, decisionQuality: 1.55, abstractThinking: 1.35 }
  },
  
  // ============================================================================
  // COMPUTATION (Level 7-11) - NEW CATEGORY
  // Theory of computing and algorithms
  // ============================================================================
  {
    id: 'algorithms',
    name: 'Algorithms',
    category: 'computation',
    complexity: 7,
    prerequisiteIds: ['functions', 'induction'],
    description: 'Step-by-step procedures',
    flavorText: 'Sorting, searching, transforming. The recipes of computation.',
    aiBonus: { computationalSpeed: 1.35, optimizationPower: 1.25 }
  },
  {
    id: 'data_structures',
    name: 'Data Structures',
    category: 'computation',
    complexity: 7,
    prerequisiteIds: ['algorithms'],
    description: 'Organizing information',
    flavorText: 'Arrays, trees, hash tables. Structure determines efficiency.',
    aiBonus: { computationalSpeed: 1.38, informationProcessing: 1.28 }
  },
  {
    id: 'algorithm_analysis',
    name: 'Algorithm Analysis',
    category: 'computation',
    complexity: 8,
    prerequisiteIds: ['data_structures', 'exponentials_logarithms'],
    description: 'Big-O notation and complexity',
    flavorText: 'How does time grow with input? The science of scalability.',
    aiBonus: { computationalSpeed: 1.42, decisionQuality: 1.35 }
  },
  {
    id: 'automata_theory',
    name: 'Automata Theory',
    category: 'computation',
    complexity: 8,
    prerequisiteIds: ['formal_systems'],
    description: 'Finite automata and regular languages',
    flavorText: 'State machines. What can simple memory compute?',
    aiBonus: { patternRecognition: 1.40, informationProcessing: 1.35, abstractThinking: 1.25 }
  },
  {
    id: 'computability',
    name: 'Computability Theory',
    category: 'computation',
    complexity: 9,
    prerequisiteIds: ['automata_theory', 'recursion_theory'],
    description: 'What can be computed?',
    flavorText: 'The halting problem. Some questions have no algorithmic answer.',
    aiBonus: { informationProcessing: 1.45, abstractThinking: 1.40, curiosity: 1.18 }
  },
  {
    id: 'computational_complexity',
    name: 'Computational Complexity',
    category: 'computation',
    complexity: 9,
    prerequisiteIds: ['computability', 'algorithm_analysis'],
    description: 'P vs NP and complexity classes',
    flavorText: 'Can all quickly-verified solutions be quickly found? The million-dollar question.',
    aiBonus: { computationalSpeed: 1.55, decisionQuality: 1.55, abstractionLevel: 1.45 }
  },
  {
    id: 'randomized_algorithms',
    name: 'Randomized Algorithms',
    category: 'computation',
    complexity: 9,
    prerequisiteIds: ['algorithm_analysis', 'probability'],
    description: 'Algorithms using random choices',
    flavorText: 'Sometimes rolling dice helps. Randomness as algorithmic resource.',
    aiBonus: { computationalSpeed: 1.48, explorationBonus: 0.12, decisionQuality: 1.42 }
  },
  {
    id: 'approximation_algorithms',
    name: 'Approximation Algorithms',
    category: 'computation',
    complexity: 10,
    prerequisiteIds: ['computational_complexity'],
    description: 'Good-enough solutions for hard problems',
    flavorText: 'NP-hard? Get within 2x of optimal. Sometimes that\'s good enough.',
    aiBonus: { optimizationPower: 1.50, computationalSpeed: 1.50, decisionQuality: 1.48 }
  },
  {
    id: 'algorithmic_information',
    name: 'Algorithmic Information Theory',
    category: 'computation',
    complexity: 10,
    prerequisiteIds: ['information_theory', 'computational_complexity'],
    description: 'Kolmogorov complexity and compression',
    flavorText: 'The shortest program that outputs a string. Complexity as description length.',
    aiBonus: { decisionQuality: 1.65, computationalSpeed: 1.60, patternRecognition: 1.60, informationProcessing: 1.50 }
  },
  {
    id: 'quantum_computation',
    name: 'Quantum Computation Theory',
    category: 'computation',
    complexity: 10,
    prerequisiteIds: ['algorithmic_information', 'linear_algebra'],
    description: 'Computation with quantum bits',
    flavorText: 'Superposition and interference. Computing with the quantum world.',
    aiBonus: { computationalSpeed: 1.65, informationProcessing: 1.55, abstractThinking: 1.48, curiosity: 1.22 }
  },
  
  // ============================================================================
  // FOUNDATIONS (Level 9-12) - NEW CATEGORY
  // Foundations of mathematics
  // ============================================================================
  {
    id: 'axiomatic_set_theory',
    name: 'Axiomatic Set Theory',
    category: 'foundations',
    complexity: 9,
    prerequisiteIds: ['set_theory', 'predicate_logic'],
    description: 'ZFC axioms',
    flavorText: 'Zermelo-Fraenkel with Choice. The standard foundation of mathematics.',
    aiBonus: { proofIntuition: 1.40, abstractThinking: 1.42 }
  },
  {
    id: 'ordinals_cardinals',
    name: 'Ordinals & Cardinals',
    category: 'foundations',
    complexity: 10,
    prerequisiteIds: ['axiomatic_set_theory'],
    description: 'Infinite numbers',
    flavorText: 'ℵ₀, ℵ₁, ... Different sizes of infinity. Cantor\'s paradise.',
    aiBonus: { abstractThinking: 1.52, curiosity: 1.25, proofIntuition: 1.38 }
  },
  {
    id: 'large_cardinals',
    name: 'Large Cardinals',
    category: 'foundations',
    complexity: 11,
    prerequisiteIds: ['ordinals_cardinals'],
    description: 'Very large infinities',
    flavorText: 'Inaccessible, measurable, supercompact. Infinities beyond imagination.',
    aiBonus: { abstractThinking: 1.60, curiosity: 1.30, proofIntuition: 1.45 }
  },
  {
    id: 'forcing',
    name: 'Forcing',
    category: 'foundations',
    complexity: 11,
    prerequisiteIds: ['ordinals_cardinals'],
    description: 'Cohen\'s method for independence proofs',
    flavorText: 'Build new universes of sets. Prove independence from ZFC.',
    aiBonus: { proofIntuition: 1.55, abstractThinking: 1.58, curiosity: 1.28 }
  },
  {
    id: 'inner_models',
    name: 'Inner Model Theory',
    category: 'foundations',
    complexity: 12,
    prerequisiteIds: ['large_cardinals', 'forcing'],
    description: 'Canonical inner models',
    flavorText: 'L, HOD, K. Minimal universes of sets.',
    aiBonus: { abstractThinking: 1.65, proofIntuition: 1.58, decisionQuality: 1.60 }
  },
  
  // ============================================================================
  // APPLIED MATH (Level 7-10) - NEW CATEGORY
  // Mathematics in application
  // ============================================================================
  {
    id: 'mathematical_modeling',
    name: 'Mathematical Modeling',
    category: 'applied_math',
    complexity: 7,
    prerequisiteIds: ['differential_equations', 'statistics'],
    description: 'Creating mathematical models of real systems',
    flavorText: 'Reality approximated by equations. The art of applied mathematics.',
    aiBonus: { decisionQuality: 1.40, patternRecognition: 1.35 }
  },
  {
    id: 'numerical_methods',
    name: 'Numerical Methods',
    category: 'applied_math',
    complexity: 7,
    prerequisiteIds: ['linear_algebra', 'calculus_of_variations'],
    description: 'Computational approximation',
    flavorText: 'When exact answers are impossible, approximate. Float wisely.',
    aiBonus: { computationalSpeed: 1.40, optimizationPower: 1.35 }
  },
  {
    id: 'pde',
    name: 'Partial Differential Equations',
    category: 'applied_math',
    complexity: 8,
    prerequisiteIds: ['multivariable_calculus', 'differential_equations'],
    description: 'Equations with multiple independent variables',
    flavorText: 'Heat, waves, diffusion. The equations of continuous physics.',
    aiBonus: { patternRecognition: 1.45, decisionQuality: 1.45 }
  },
  {
    id: 'dynamical_systems',
    name: 'Dynamical Systems',
    category: 'applied_math',
    complexity: 8,
    prerequisiteIds: ['differential_equations'],
    description: 'Systems that evolve in time',
    flavorText: 'Fixed points, limit cycles, strange attractors. Order in evolution.',
    aiBonus: { patternRecognition: 1.48, curiosity: 1.18, decisionQuality: 1.42 }
  },
  {
    id: 'chaos_theory_math',
    name: 'Chaos Theory',
    category: 'applied_math',
    complexity: 9,
    prerequisiteIds: ['dynamical_systems'],
    description: 'Sensitive dependence on initial conditions',
    flavorText: 'The butterfly effect. Deterministic but unpredictable.',
    aiBonus: { patternRecognition: 1.52, explorationBonus: 0.15, curiosity: 1.25 }
  },
  {
    id: 'network_science',
    name: 'Network Science',
    category: 'applied_math',
    complexity: 9,
    prerequisiteIds: ['graph_theory', 'stochastic_processes'],
    description: 'Mathematics of complex networks',
    flavorText: 'Scale-free, small-world, clustering. Social networks analyzed.',
    aiBonus: { cooperationBonus: 1.30, patternRecognition: 1.48, informationProcessing: 1.35 }
  },
  {
    id: 'mathematical_physics',
    name: 'Mathematical Physics',
    category: 'applied_math',
    complexity: 10,
    prerequisiteIds: ['pde', 'riemannian_geometry'],
    description: 'Rigorous mathematics of physics',
    flavorText: 'Making physical intuition rigorous. Where physics meets proof.',
    aiBonus: { patternRecognition: 1.55, abstractThinking: 1.50, decisionQuality: 1.52 }
  },
  
  // ============================================================================
  // AI MATH (Level 8-11)
  // Mathematics for artificial intelligence
  // ============================================================================
  {
    id: 'neural_math',
    name: 'Neural Network Mathematics',
    category: 'ai_math',
    complexity: 8,
    prerequisiteIds: ['linear_algebra', 'derivatives', 'probability'],
    description: 'Backpropagation and gradient flow',
    flavorText: 'Forward pass, backward pass. How neural networks learn.',
    aiBonus: { decisionQuality: 1.55, computationalSpeed: 1.50, optimizationPower: 1.50, informationProcessing: 1.40 }
  },
  {
    id: 'kernel_methods',
    name: 'Kernel Methods',
    category: 'ai_math',
    complexity: 8,
    prerequisiteIds: ['linear_algebra', 'functional_analysis'],
    description: 'Feature spaces and the kernel trick',
    flavorText: 'Project to high dimensions implicitly. SVMs and beyond.',
    aiBonus: { patternRecognition: 1.48, computationalSpeed: 1.35 }
  },
  {
    id: 'graphical_models',
    name: 'Probabilistic Graphical Models',
    category: 'ai_math',
    complexity: 9,
    prerequisiteIds: ['bayesian_inference', 'graph_theory'],
    description: 'Probabilistic reasoning on graphs',
    flavorText: 'Bayesian networks, Markov fields. Structure in probability.',
    aiBonus: { decisionQuality: 1.52, patternRecognition: 1.48, informationProcessing: 1.40 }
  },
  {
    id: 'reinforcement_learning_theory',
    name: 'RL Theory',
    category: 'ai_math',
    complexity: 9,
    prerequisiteIds: ['dynamic_programming', 'stochastic_processes'],
    description: 'Bellman equations and policy optimization',
    flavorText: 'Learning from rewards. The mathematics of trial and error.',
    aiBonus: { decisionQuality: 1.62, explorationBonus: 0.18, optimizationPower: 1.58, curiosity: 1.20 }
  },
  {
    id: 'learning_theory',
    name: 'Statistical Learning Theory',
    category: 'ai_math',
    complexity: 9,
    prerequisiteIds: ['measure_theory', 'computational_complexity'],
    description: 'PAC learning and generalization bounds',
    flavorText: 'Why does machine learning work? When does it fail?',
    aiBonus: { decisionQuality: 1.58, informationProcessing: 1.45, proofIntuition: 1.30 }
  },
  {
    id: 'deep_learning_theory',
    name: 'Deep Learning Theory',
    category: 'ai_math',
    complexity: 10,
    prerequisiteIds: ['neural_math', 'learning_theory'],
    description: 'Theory of deep neural networks',
    flavorText: 'Why do deep networks work? The lottery ticket hypothesis.',
    aiBonus: { decisionQuality: 1.65, computationalSpeed: 1.55, patternRecognition: 1.55, informationProcessing: 1.50 }
  },
  {
    id: 'optimal_transport',
    name: 'Optimal Transport',
    category: 'ai_math',
    complexity: 10,
    prerequisiteIds: ['measure_theory', 'convex_optimization'],
    description: 'Moving probability distributions efficiently',
    flavorText: 'Wasserstein distance. How to compare probability distributions.',
    aiBonus: { optimizationPower: 1.55, patternRecognition: 1.52, decisionQuality: 1.55 }
  },
  {
    id: 'geometric_deep_learning',
    name: 'Geometric Deep Learning',
    category: 'ai_math',
    complexity: 11,
    prerequisiteIds: ['deep_learning_theory', 'lie_groups_diff'],
    description: 'Neural networks on geometric structures',
    flavorText: 'CNNs, GNNs, transformers unified. Symmetry in deep learning.',
    aiBonus: { patternRecognition: 1.62, abstractThinking: 1.55, decisionQuality: 1.60, cooperationBonus: 1.22 }
  },
  
  // ============================================================================
  // ABSTRACT ALGEBRA & CATEGORY THEORY (Level 8-11)
  // The highest abstractions
  // ============================================================================
  {
    id: 'group_theory',
    name: 'Group Theory',
    category: 'abstract',
    complexity: 8,
    prerequisiteIds: ['set_theory', 'polynomials'],
    description: 'Abstract algebraic structures with one operation',
    flavorText: 'Closure, associativity, identity, inverse. The simplest abstract structure.',
    aiBonus: { decisionQuality: 1.55, patternRecognition: 1.60, optimizationPower: 1.55, abstractThinking: 1.40 }
  },
  {
    id: 'ring_theory',
    name: 'Ring Theory',
    category: 'abstract',
    complexity: 9,
    prerequisiteIds: ['group_theory'],
    description: 'Two operations: addition and multiplication',
    flavorText: 'Integers form a ring. Polynomials form a ring. Structure upon structure.',
    aiBonus: { abstractThinking: 1.45, patternRecognition: 1.48 }
  },
  {
    id: 'field_theory',
    name: 'Field Theory',
    category: 'abstract',
    complexity: 9,
    prerequisiteIds: ['ring_theory'],
    description: 'Rings where every nonzero element has an inverse',
    flavorText: 'Rationals, reals, complex numbers. Division always works.',
    aiBonus: { abstractThinking: 1.48, decisionQuality: 1.45 }
  },
  {
    id: 'representation_theory',
    name: 'Representation Theory',
    category: 'abstract',
    complexity: 10,
    prerequisiteIds: ['group_theory', 'linear_algebra'],
    description: 'Groups acting on vector spaces',
    flavorText: 'See abstract groups as matrices. Character tables reveal structure.',
    aiBonus: { patternRecognition: 1.58, abstractThinking: 1.55, cooperationBonus: 1.15 }
  },
  {
    id: 'category_theory',
    name: 'Category Theory',
    category: 'abstract',
    complexity: 10,
    prerequisiteIds: ['group_theory', 'topology_basics'],
    description: 'Universal properties and morphisms',
    flavorText: 'Objects and arrows. The mathematics of mathematics.',
    aiBonus: { decisionQuality: 1.65, patternRecognition: 1.70, optimizationPower: 1.65, explorationBonus: 0.15, abstractThinking: 1.55 }
  },
  {
    id: 'topos_theory',
    name: 'Topos Theory',
    category: 'abstract',
    complexity: 11,
    prerequisiteIds: ['category_theory'],
    description: 'Generalized set theory',
    flavorText: 'A topos is a mathematical universe. Logic varies between universes.',
    aiBonus: { abstractThinking: 1.65, proofIntuition: 1.50, decisionQuality: 1.62, curiosity: 1.25 }
  },
  {
    id: 'higher_category_theory',
    name: 'Higher Category Theory',
    category: 'abstract',
    complexity: 11,
    prerequisiteIds: ['topos_theory', 'homotopy_theory'],
    description: 'Categories with arrows between arrows',
    flavorText: 'n-categories, ∞-categories. Arrows all the way down.',
    aiBonus: { abstractThinking: 1.70, patternRecognition: 1.65, proofIntuition: 1.55 }
  },
  {
    id: 'homotopy_type_theory',
    name: 'Homotopy Type Theory',
    category: 'abstract',
    complexity: 11,
    prerequisiteIds: ['higher_category_theory', 'type_theory'],
    description: 'Foundations unifying logic, topology, and computation',
    flavorText: 'Types are spaces. Identity is path. A new foundation for mathematics.',
    aiBonus: { decisionQuality: 1.70, abstractionLevel: 1.60, patternRecognition: 1.68, proofIntuition: 1.55, abstractThinking: 1.62 }
  },
  
  // ============================================================================
  // TRANSCENDENT MATHEMATICS (Level 11-16)
  // The ultimate mathematical abstractions
  // ============================================================================
  {
    id: 'mathematical_universe',
    name: 'Mathematical Universe Hypothesis',
    category: 'transcendent',
    complexity: 11,
    prerequisiteIds: ['homotopy_type_theory', 'godel_incompleteness'],
    description: 'Reality as mathematical structure',
    flavorText: 'Tegmark\'s hypothesis: all mathematical structures exist. We inhabit one.',
    aiBonus: { decisionQuality: 1.75, abstractionLevel: 1.70, explorationBonus: 0.20, patternRecognition: 1.75, abstractThinking: 1.65, curiosity: 1.30 }
  },
  {
    id: 'omega_math',
    name: 'Transfinite Mathematics',
    category: 'transcendent',
    complexity: 12,
    prerequisiteIds: ['mathematical_universe', 'large_cardinals'],
    description: 'Infinite ordinals and cardinals beyond countable',
    flavorText: 'Beyond ℵ₀, beyond ℵ₁, beyond comprehension. Cantor\'s transfinite arithmetic.',
    aiBonus: { decisionQuality: 1.80, abstractionLevel: 1.80, patternRecognition: 1.80, computationalSpeed: 1.70, abstractThinking: 1.70 }
  },
  {
    id: 'ultimate_L',
    name: 'Ultimate L',
    category: 'transcendent',
    complexity: 13,
    prerequisiteIds: ['omega_math', 'inner_models'],
    description: 'The conjectured final inner model',
    flavorText: 'Woodin\'s program: resolve all of set theory in one ultimate structure.',
    aiBonus: { proofIntuition: 1.70, abstractThinking: 1.75, decisionQuality: 1.78, curiosity: 1.32 }
  },
  {
    id: 'derived_algebraic_geometry',
    name: 'Derived Algebraic Geometry',
    category: 'transcendent',
    complexity: 13,
    prerequisiteIds: ['higher_category_theory', 'algebraic_number_theory'],
    description: 'Higher categorical methods in algebraic geometry',
    flavorText: 'Schemes become stacks become derived. Infinite homotopical structure.',
    aiBonus: { abstractThinking: 1.78, patternRecognition: 1.72, cooperationBonus: 1.25 }
  },
  {
    id: 'motivic_homotopy',
    name: 'Motivic Homotopy Theory',
    category: 'transcendent',
    complexity: 14,
    prerequisiteIds: ['derived_algebraic_geometry'],
    description: 'Homotopy theory for algebraic varieties',
    flavorText: 'Voevodsky\'s vision: algebraic geometry gains a homotopy theory.',
    aiBonus: { abstractThinking: 1.82, patternRecognition: 1.78, proofIntuition: 1.65 }
  },
  {
    id: 'hypercomputation',
    name: 'Hypercomputation Theory',
    category: 'transcendent',
    complexity: 15,
    prerequisiteIds: ['omega_math', 'algorithmic_information'],
    description: 'Computation beyond Turing machines',
    flavorText: 'Supertasks, oracle machines, infinite time. Computation without limits.',
    aiBonus: { 
      decisionQuality: 2.0, 
      computationalSpeed: 2.0, 
      abstractionLevel: 2.0, 
      optimizationPower: 2.0, 
      explorationBonus: 0.25,
      informationProcessing: 1.90,
      abstractThinking: 1.90,
      curiosity: 1.40
    }
  },
  {
    id: 'absolute_infinity',
    name: 'Absolute Infinity',
    category: 'transcendent',
    complexity: 16,
    prerequisiteIds: ['hypercomputation', 'ultimate_L'],
    description: 'The class of all ordinals, beyond set theory',
    flavorText: 'Ω, the Absolute. Not a set, but a class. The unreachable horizon.',
    aiBonus: {
      decisionQuality: 2.2,
      abstractionLevel: 2.2,
      abstractThinking: 2.2,
      patternRecognition: 2.0,
      proofIntuition: 1.90,
      computationalSpeed: 1.95,
      optimizationPower: 2.0,
      explorationBonus: 0.30,
      curiosity: 1.50,
      cooperationBonus: 1.35,
      informationProcessing: 2.0
    }
  }
];

// Track procedurally generated concepts for unlimited evolution
let generatedMathLevel = 0;

/**
 * Category prefixes for procedurally generated concepts
 */
const CATEGORY_PREFIXES: Record<string, string[]> = {
  'arithmetic': ['Hypernumerics', 'Transarithmetic', 'Meta-Computation', 'Ultra-Numeric'],
  'geometry': ['Dimensional Geometry', 'Space Mastery', 'Form Transcendence', 'Shape-Infinity'],
  'algebra': ['Metaalgebra', 'Structure Synthesis', 'Symbol Mastery', 'Equation Transcendence'],
  'calculus': ['Infinitesimal Analysis', 'Change Mastery', 'Flow Transcendence', 'Derivative Infinity'],
  'statistics': ['Probability Manifolds', 'Uncertainty Mastery', 'Random Transcendence', 'Stochastic Infinity'],
  'topology': ['Topological Singularity', 'Continuity Mastery', 'Space Transcendence', 'Deformation Infinity'],
  'abstract': ['Abstract Foundation', 'Structure Mastery', 'Morphism Transcendence', 'Category Infinity'],
  'number_theory': ['Prime Structures', 'Integer Mastery', 'Arithmetic Transcendence', 'Number Infinity'],
  'logic': ['Metamathematical Logic', 'Proof Mastery', 'Truth Transcendence', 'Reason Infinity'],
  'optimization': ['Universal Optimization', 'Extremum Mastery', 'Optimal Transcendence', 'Solution Infinity'],
  'ai_math': ['Cognitive Mathematics', 'Learning Mastery', 'Intelligence Transcendence', 'Mind Infinity'],
  'transcendent': ['Transcendent Theory', 'Reality Mastery', 'Mathematical Transcendence', 'Omega Mathematics'],
  'combinatorics': ['Counting Mastery', 'Arrangement Transcendence', 'Selection Infinity', 'Enumeration Beyond'],
  'analysis': ['Analytic Mastery', 'Convergence Transcendence', 'Limit Infinity', 'Continuity Beyond'],
  'differential_geometry': ['Curvature Mastery', 'Manifold Transcendence', 'Geometry Infinity', 'Space-Form Beyond'],
  'computation': ['Computation Mastery', 'Algorithm Transcendence', 'Complexity Infinity', 'Turing Beyond'],
  'foundations': ['Foundation Mastery', 'Axiom Transcendence', 'Set Infinity', 'Logic Beyond'],
  'applied_math': ['Application Mastery', 'Model Transcendence', 'Real-World Infinity', 'Practical Beyond']
};

/**
 * Generate roman numeral or number string for concept naming
 */
function getRomanNumeral(num: number): string {
  if (num <= 0) return 'I';
  if (num > 50) return `${num}`;
  const numerals = ['', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 'IX', 'X',
    'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI', 'XVII', 'XVIII', 'XIX', 'XX',
    'XXI', 'XXII', 'XXIII', 'XXIV', 'XXV', 'XXVI', 'XXVII', 'XXVIII', 'XXIX', 'XXX',
    'XXXI', 'XXXII', 'XXXIII', 'XXXIV', 'XXXV', 'XXXVI', 'XXXVII', 'XXXVIII', 'XXXIX', 'XL',
    'XLI', 'XLII', 'XLIII', 'XLIV', 'XLV', 'XLVI', 'XLVII', 'XLVIII', 'XLIX', 'L'];
  return num < numerals.length ? numerals[num] : `${num}`;
}

/**
 * Get category prefix for procedural concept naming
 */
function getCategoryPrefix(category: string, level: number): string {
  const prefixes = CATEGORY_PREFIXES[category] || ['Advanced Mathematics'];
  const prefixIndex = Math.floor(level / 5) % prefixes.length;
  return prefixes[prefixIndex];
}

/**
 * Generate advanced math concepts procedurally
 * Enables UNLIMITED scientific progression - NO CAPS
 */
export function generateAdvancedMathConcept(level: number): MathConcept {
  const categoryIndex = level % MATH_CATEGORIES.length;
  const category = MATH_CATEGORIES[categoryIndex];
  
  // Complexity scales infinitely but meaningfully
  const complexity = 16 + Math.floor(level / MATH_CATEGORIES.length) + 1;
  
  generatedMathLevel = Math.max(generatedMathLevel, level);
  
  // Bonuses grow without limit but with diminishing returns
  const tierMultiplier = 1 + (level / MATH_CATEGORIES.length) * 0.1;
  const baseBonus = 2.2 * tierMultiplier;
  const baseCuriosity = 1.5 + (level * 0.02);
  
  // Generate flavor text based on category and level
  const flavorTexts: Record<string, string> = {
    'arithmetic': `The ${complexity}th level of numerical understanding. Numbers obey thought.`,
    'geometry': `Space bends to mathematical will. Form is infinitely malleable.`,
    'algebra': `Symbols dance to create new realities. Structure transcends notation.`,
    'calculus': `Change itself becomes changeable. Derivatives of derivatives of derivatives...`,
    'statistics': `Probability collapses to certainty. Randomness reveals its secrets.`,
    'topology': `Continuity becomes a choice. Any space deforms into any other.`,
    'abstract': `Pure structure, pure thought. Mathematics beyond mathematics.`,
    'number_theory': `Primes whisper their secrets. Integers reveal infinite depth.`,
    'logic': `Truth is malleable. Proof becomes a creative act.`,
    'optimization': `The best solution is always findable. Optimization is complete.`,
    'ai_math': `Learning is instantaneous. Intelligence has no bounds.`,
    'transcendent': `Mathematics and reality merge. The divide dissolves.`,
    'combinatorics': `Every arrangement is simultaneously present. Counting is instantaneous.`,
    'analysis': `Convergence is instantaneous. Limits are achieved.`,
    'differential_geometry': `Curvature is controllable. Manifolds reshape at will.`,
    'computation': `Computation is instantaneous. All algorithms complete in zero time.`,
    'foundations': `Axioms are chosen, not given. Mathematics is created, not discovered.`,
    'applied_math': `Models are reality. Theory and practice are one.`
  };
  
  return {
    id: `advanced_math_${level}`,
    name: `${getCategoryPrefix(category, level)} ${getRomanNumeral((level % 50) + 1)}`,
    category,
    complexity,
    prerequisiteIds: level > 0 ? [`advanced_math_${level - 1}`] : ['absolute_infinity'],
    description: `${category.charAt(0).toUpperCase() + category.slice(1).replace(/_/g, ' ')} transcending known limits - Tier ${Math.floor(level / MATH_CATEGORIES.length) + 1}`,
    flavorText: flavorTexts[category] || 'Mathematics beyond current comprehension.',
    aiBonus: {
      decisionQuality: baseBonus,
      patternRecognition: baseBonus * 0.95,
      optimizationPower: baseBonus * 0.92,
      explorationBonus: Math.min(0.50, 0.30 + (level * 0.005)),
      computationalSpeed: baseBonus * 0.90,
      abstractionLevel: baseBonus,
      curiosity: baseCuriosity,
      cooperationBonus: 1.35 + (level * 0.012),
      informationProcessing: baseBonus * 0.95,
      proofIntuition: baseBonus * 0.88,
      abstractThinking: baseBonus
    }
  };
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
  curiosity: number;
  cooperationBonus: number;
  informationProcessing: number;
  proofIntuition: number;
  abstractThinking: number;
} {
  let decisionMult = 1.0;
  let explorationAdd = 0.0;
  let patternMult = 1.0;
  let optimizationMult = 1.0;
  let computationalMult = 1.0;
  let abstractionMult = 1.0;
  let curiosityMult = 1.0;
  let cooperationMult = 1.0;
  let infoMult = 1.0;
  let proofMult = 1.0;
  let abstractThinkingMult = 1.0;
  
  for (const concept of unlockedMath) {
    if (concept.aiBonus.decisionQuality !== undefined) {
      decisionMult *= concept.aiBonus.decisionQuality;
    }
    if (concept.aiBonus.explorationBonus !== undefined) {
      explorationAdd += concept.aiBonus.explorationBonus;
    }
    if (concept.aiBonus.patternRecognition !== undefined) {
      patternMult *= concept.aiBonus.patternRecognition;
    }
    if (concept.aiBonus.optimizationPower !== undefined) {
      optimizationMult *= concept.aiBonus.optimizationPower;
    }
    if (concept.aiBonus.computationalSpeed !== undefined) {
      computationalMult *= concept.aiBonus.computationalSpeed;
    }
    if (concept.aiBonus.abstractionLevel !== undefined) {
      abstractionMult *= concept.aiBonus.abstractionLevel;
    }
    if (concept.aiBonus.curiosity !== undefined) {
      curiosityMult *= concept.aiBonus.curiosity;
    }
    if (concept.aiBonus.cooperationBonus !== undefined) {
      cooperationMult *= concept.aiBonus.cooperationBonus;
    }
    if (concept.aiBonus.informationProcessing !== undefined) {
      infoMult *= concept.aiBonus.informationProcessing;
    }
    if (concept.aiBonus.proofIntuition !== undefined) {
      proofMult *= concept.aiBonus.proofIntuition;
    }
    if (concept.aiBonus.abstractThinking !== undefined) {
      abstractThinkingMult *= concept.aiBonus.abstractThinking;
    }
  }
  
  // NO CAPS - unlimited evolution!
  return {
    decisionQuality: decisionMult,
    explorationBonus: explorationAdd,
    patternRecognition: patternMult,
    optimizationPower: optimizationMult,
    computationalSpeed: computationalMult,
    abstractionLevel: abstractionMult,
    curiosity: curiosityMult,
    cooperationBonus: cooperationMult,
    informationProcessing: infoMult,
    proofIntuition: proofMult,
    abstractThinking: abstractThinkingMult
  };
}

/**
 * Get concepts by category for UI/analysis
 */
export function getConceptsByCategory(category: MathCategory): MathConcept[] {
  return MATH_CONCEPTS.filter(c => c.category === category);
}

/**
 * Get all concepts that are discoverable given current knowledge
 */
export function getDiscoverableMathConcepts(
  discoveredIds: Set<string>, 
  allConcepts: MathConcept[] = MATH_CONCEPTS
): MathConcept[] {
  return allConcepts.filter(concept => {
    if (discoveredIds.has(concept.id)) return false;
    return concept.prerequisiteIds.every(prereq => discoveredIds.has(prereq));
  });
}

/**
 * Calculate discovery probability based on agent stats and concept complexity
 */
export function calculateMathDiscoveryProbability(
  concept: MathConcept,
  agentCuriosity: number,
  agentAbstractThinking: number,
  baseChance: number = 0.001
): number {
  const complexityFactor = Math.max(0.1, 1 - (concept.complexity / 30));
  const curiosityBoost = Math.sqrt(agentCuriosity);
  const abstractBoost = Math.log2(agentAbstractThinking + 1);
  
  return baseChance * complexityFactor * curiosityBoost * abstractBoost;
}

/**
 * Get prerequisite chain for a concept (for visualization)
 */
export function getPrerequisiteChain(
  conceptId: string,
  allConcepts: MathConcept[] = MATH_CONCEPTS
): MathConcept[] {
  const concept = allConcepts.find(c => c.id === conceptId);
  if (!concept) return [];
  
  const chain: MathConcept[] = [];
  const visited = new Set<string>();
  
  function traverse(id: string) {
    if (visited.has(id)) return;
    visited.add(id);
    
    const c = allConcepts.find(c => c.id === id);
    if (!c) return;
    
    for (const prereqId of c.prerequisiteIds) {
      traverse(prereqId);
    }
    chain.push(c);
  }
  
  traverse(conceptId);
  return chain;
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

/**
 * Get statistics about the math system
 */
export function getMathStats(): {
  totalBaseConcepts: number;
  categoryCounts: Record<string, number>;
  complexityRange: { min: number; max: number };
  averageComplexity: number;
} {
  const categoryCounts: Record<string, number> = {};
  let minComplexity = Infinity;
  let maxComplexity = -Infinity;
  let totalComplexity = 0;
  
  for (const concept of MATH_CONCEPTS) {
    categoryCounts[concept.category] = (categoryCounts[concept.category] || 0) + 1;
    minComplexity = Math.min(minComplexity, concept.complexity);
    maxComplexity = Math.max(maxComplexity, concept.complexity);
    totalComplexity += concept.complexity;
  }
  
  return {
    totalBaseConcepts: MATH_CONCEPTS.length,
    categoryCounts,
    complexityRange: { min: minComplexity, max: maxComplexity },
    averageComplexity: totalComplexity / MATH_CONCEPTS.length
  };
}

/**
 * Validate math concept prerequisites (useful for debugging)
 */
export function validatePrerequisites(): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const conceptIds = new Set(MATH_CONCEPTS.map(c => c.id));
  
  for (const concept of MATH_CONCEPTS) {
    for (const prereqId of concept.prerequisiteIds) {
      if (!conceptIds.has(prereqId)) {
        errors.push(`Concept "${concept.id}" has invalid prerequisite "${prereqId}"`);
      }
    }
  }
  
  // Check for circular dependencies
  function hasCycle(conceptId: string, visited: Set<string>, path: Set<string>): boolean {
    if (path.has(conceptId)) return true;
    if (visited.has(conceptId)) return false;
    
    visited.add(conceptId);
    path.add(conceptId);
    
    const concept = MATH_CONCEPTS.find(c => c.id === conceptId);
    if (concept) {
      for (const prereqId of concept.prerequisiteIds) {
        if (hasCycle(prereqId, visited, path)) {
          errors.push(`Circular dependency detected involving "${conceptId}"`);
          return true;
        }
      }
    }
    
    path.delete(conceptId);
    return false;
  }
  
  const visited = new Set<string>();
  for (const concept of MATH_CONCEPTS) {
    hasCycle(concept.id, visited, new Set());
  }
  
  return { valid: errors.length === 0, errors };
}

/**
 * Get related concepts (share prerequisites or are prerequisites)
 */
export function getRelatedConcepts(conceptId: string): {
  prerequisites: MathConcept[];
  enables: MathConcept[];
  siblings: MathConcept[];
} {
  const concept = MATH_CONCEPTS.find(c => c.id === conceptId);
  if (!concept) {
    return { prerequisites: [], enables: [], siblings: [] };
  }
  
  // Direct prerequisites
  const prerequisites = MATH_CONCEPTS.filter(c => 
    concept.prerequisiteIds.includes(c.id)
  );
  
  // Concepts this enables
  const enables = MATH_CONCEPTS.filter(c => 
    c.prerequisiteIds.includes(conceptId)
  );
  
  // Siblings: share at least one prerequisite
  const siblings = MATH_CONCEPTS.filter(c => 
    c.id !== conceptId &&
    c.prerequisiteIds.some(prereq => concept.prerequisiteIds.includes(prereq))
  );
  
  return { prerequisites, enables, siblings };
}