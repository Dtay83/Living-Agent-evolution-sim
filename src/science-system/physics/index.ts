/**
 * Physics Concepts System - EXPANDED EDITION
 * Provides comprehensive physics knowledge that aids agent learning and invention
 * UNLIMITED EVOLUTION - No caps on discovery or bonuses
 * 
 * Categories:
 * - mechanics, thermodynamics, electromagnetism, quantum, relativity, unified
 * - cosmology, particle, plasma, condensed_matter, exotic, transcendent
 * - NEW: nuclear, astrophysics, biophysics, information_physics, chaos_complexity
 * 
 * Bonus Types:
 * - energyEfficiency, movementSpeed, learningRate, inventionChance
 * - spaceManipulation, timePerception
 * - NEW: curiosity, cooperationBonus, dimensionalAccess, informationProcessing, entropicResistance
 */

export interface PhysicsConcept {
  id: string;
  name: string;
  category: 
    | 'mechanics' 
    | 'thermodynamics' 
    | 'electromagnetism' 
    | 'quantum' 
    | 'relativity' 
    | 'unified' 
    | 'cosmology' 
    | 'particle' 
    | 'plasma' 
    | 'condensed_matter' 
    | 'exotic' 
    | 'transcendent'
    | 'nuclear'
    | 'astrophysics'
    | 'biophysics'
    | 'information_physics'
    | 'chaos_complexity';
  complexity: number;         // 1-100+: How advanced this concept is (NO CAP)
  discoveredAt?: number;      // Tick when discovered
  discoveredBy?: number;      // Agent ID who discovered
  prerequisiteIds: string[];  // Required prior knowledge
  description: string;
  flavorText?: string;        // Optional deeper insight into the concept
  aiBonus: {
    energyEfficiency?: number;    // Multiplier for energy usage (lower is better)
    movementSpeed?: number;       // Multiplier for movement
    learningRate?: number;        // Multiplier for Q-learning alpha
    inventionChance?: number;     // Additive bonus to invention discovery
    spaceManipulation?: number;   // Ability to affect grid space
    timePerception?: number;      // Extra ticks or preview ability
    // NEW BONUS TYPES
    curiosity?: number;           // Drive to explore unknown areas and concepts
    cooperationBonus?: number;    // Benefits when near other agents
    dimensionalAccess?: number;   // Ability to interact with parallel simulation layers
    informationProcessing?: number; // Faster decision-making and pattern recognition
    entropicResistance?: number;  // Slower energy decay / resistance to disorder
  };
}

/**
 * Progressive physics concepts from basic to transcendent
 * UNLIMITED system - new concepts generated procedurally beyond base set
 * 
 * Organized by category with increasing complexity levels
 */
export const PHYSICS_CONCEPTS: PhysicsConcept[] = [
  
  // ============================================================================
  // MECHANICS (Level 1-6)
  // Foundation of physical understanding - motion, forces, and energy
  // ============================================================================
  {
    id: 'basic_motion',
    name: 'Basic Motion',
    category: 'mechanics',
    complexity: 1,
    prerequisiteIds: [],
    description: 'Objects move in predictable paths',
    flavorText: 'The first step: recognizing that movement follows rules.',
    aiBonus: { movementSpeed: 1.05, curiosity: 1.02 }
  },
  {
    id: 'force_and_acceleration',
    name: 'Force & Acceleration',
    category: 'mechanics',
    complexity: 1,
    prerequisiteIds: ['basic_motion'],
    description: 'Forces cause changes in motion',
    flavorText: 'Push something, it speeds up. The universe responds to effort.',
    aiBonus: { movementSpeed: 1.06, learningRate: 1.02 }
  },
  {
    id: 'leverage',
    name: 'Leverage & Simple Machines',
    category: 'mechanics',
    complexity: 2,
    prerequisiteIds: ['force_and_acceleration'],
    description: 'Force multiplication through mechanical advantage',
    flavorText: 'Give me a lever long enough and a fulcrum, and I shall move the world.',
    aiBonus: { energyEfficiency: 0.95, inventionChance: 0.003 }
  },
  {
    id: 'friction',
    name: 'Friction',
    category: 'mechanics',
    complexity: 2,
    prerequisiteIds: ['force_and_acceleration'],
    description: 'Surfaces resist relative motion',
    flavorText: 'The universe has drag. Nothing moves freely.',
    aiBonus: { energyEfficiency: 0.96, entropicResistance: 1.03 }
  },
  {
    id: 'inertia',
    name: 'Inertia',
    category: 'mechanics',
    complexity: 2,
    prerequisiteIds: ['basic_motion'],
    description: 'Objects resist changes in motion',
    flavorText: 'Mass is stubborn. It wants to keep doing what it was doing.',
    aiBonus: { movementSpeed: 1.08, entropicResistance: 1.02 }
  },
  {
    id: 'newtons_laws',
    name: "Newton's Laws of Motion",
    category: 'mechanics',
    complexity: 3,
    prerequisiteIds: ['inertia', 'force_and_acceleration'],
    description: 'Three fundamental laws governing motion',
    flavorText: 'For every action, an equal and opposite reaction. The cosmos balances itself.',
    aiBonus: { movementSpeed: 1.10, learningRate: 1.05, curiosity: 1.03 }
  },
  {
    id: 'momentum',
    name: 'Momentum Conservation',
    category: 'mechanics',
    complexity: 3,
    prerequisiteIds: ['newtons_laws'],
    description: 'Momentum is conserved in interactions',
    flavorText: 'What goes in must come out. The universe keeps perfect accounts.',
    aiBonus: { energyEfficiency: 0.93, movementSpeed: 1.10 }
  },
  {
    id: 'work_and_power',
    name: 'Work & Power',
    category: 'mechanics',
    complexity: 3,
    prerequisiteIds: ['leverage'],
    description: 'Energy transfer through force over distance',
    flavorText: 'Work is force times distance. Power is work over time. Efficiency is everything.',
    aiBonus: { energyEfficiency: 0.92, informationProcessing: 1.03 }
  },
  {
    id: 'energy_conservation',
    name: 'Energy Conservation',
    category: 'mechanics',
    complexity: 3,
    prerequisiteIds: ['work_and_power'],
    description: 'Energy cannot be created or destroyed',
    flavorText: 'The first law that hints at deeper truths. Energy is eternal.',
    aiBonus: { energyEfficiency: 0.90, inventionChance: 0.005, entropicResistance: 1.05 }
  },
  {
    id: 'potential_kinetic',
    name: 'Potential & Kinetic Energy',
    category: 'mechanics',
    complexity: 3,
    prerequisiteIds: ['energy_conservation'],
    description: 'Energy transforms between stored and active forms',
    flavorText: 'Energy waits, then springs forth. Position becomes motion.',
    aiBonus: { energyEfficiency: 0.91, movementSpeed: 1.08 }
  },
  {
    id: 'rotational_dynamics',
    name: 'Rotational Dynamics',
    category: 'mechanics',
    complexity: 4,
    prerequisiteIds: ['momentum'],
    description: 'Angular momentum and torque',
    flavorText: 'Spin introduces new conservation laws. The universe loves rotation.',
    aiBonus: { movementSpeed: 1.15, energyEfficiency: 0.92 }
  },
  {
    id: 'harmonic_motion',
    name: 'Harmonic Motion',
    category: 'mechanics',
    complexity: 4,
    prerequisiteIds: ['potential_kinetic', 'rotational_dynamics'],
    description: 'Oscillations and periodic motion',
    flavorText: 'Springs, pendulums, orbits - nature loves to repeat itself.',
    aiBonus: { timePerception: 1.05, learningRate: 1.06, curiosity: 1.04 }
  },
  {
    id: 'wave_mechanics',
    name: 'Wave Mechanics',
    category: 'mechanics',
    complexity: 5,
    prerequisiteIds: ['harmonic_motion'],
    description: 'Propagation of disturbances through media',
    flavorText: 'Ripples carry information. Energy spreads without matter moving.',
    aiBonus: { informationProcessing: 1.08, movementSpeed: 1.12, curiosity: 1.05 }
  },
  {
    id: 'fluid_dynamics',
    name: 'Fluid Dynamics',
    category: 'mechanics',
    complexity: 5,
    prerequisiteIds: ['wave_mechanics', 'friction'],
    description: 'Motion of liquids and gases',
    flavorText: 'Continuous media, continuous equations. Turbulence emerges.',
    aiBonus: { movementSpeed: 1.20, energyEfficiency: 0.88 }
  },
  {
    id: 'continuum_mechanics',
    name: 'Continuum Mechanics',
    category: 'mechanics',
    complexity: 6,
    prerequisiteIds: ['fluid_dynamics'],
    description: 'Stress, strain, and deformation in continuous media',
    flavorText: 'Matter bends, stretches, flows. Everything is connected.',
    aiBonus: { spaceManipulation: 1.05, energyEfficiency: 0.85, informationProcessing: 1.06 }
  },

  // ============================================================================
  // THERMODYNAMICS (Level 3-8)
  // Heat, energy, entropy, and the arrow of time
  // ============================================================================
  {
    id: 'temperature',
    name: 'Temperature',
    category: 'thermodynamics',
    complexity: 3,
    prerequisiteIds: ['energy_conservation'],
    description: 'Measure of average molecular kinetic energy',
    flavorText: 'Hot and cold are not opposites but degrees of the same thing.',
    aiBonus: { curiosity: 1.03, learningRate: 1.03 }
  },
  {
    id: 'heat_transfer',
    name: 'Heat Transfer',
    category: 'thermodynamics',
    complexity: 3,
    prerequisiteIds: ['temperature'],
    description: 'Energy flows from hot to cold',
    flavorText: 'The universe seeks equilibrium. Differences drive flow.',
    aiBonus: { energyEfficiency: 0.92, entropicResistance: 1.04 }
  },
  {
    id: 'first_law_thermo',
    name: 'First Law of Thermodynamics',
    category: 'thermodynamics',
    complexity: 4,
    prerequisiteIds: ['heat_transfer', 'work_and_power'],
    description: 'Energy conservation in thermal systems',
    flavorText: 'You cannot win - energy is conserved.',
    aiBonus: { energyEfficiency: 0.90, entropicResistance: 1.06 }
  },
  {
    id: 'entropy',
    name: 'Entropy',
    category: 'thermodynamics',
    complexity: 5,
    prerequisiteIds: ['first_law_thermo'],
    description: 'Disorder tends to increase',
    flavorText: 'The arrow of time. Why eggs break but never unbreak.',
    aiBonus: { learningRate: 1.10, inventionChance: 0.008, curiosity: 1.06 }
  },
  {
    id: 'second_law_thermo',
    name: 'Second Law of Thermodynamics',
    category: 'thermodynamics',
    complexity: 5,
    prerequisiteIds: ['entropy'],
    description: 'Entropy of isolated systems never decreases',
    flavorText: 'You cannot break even - entropy always increases.',
    aiBonus: { entropicResistance: 1.10, timePerception: 1.05, inventionChance: 0.007 }
  },
  {
    id: 'thermodynamic_efficiency',
    name: 'Thermodynamic Efficiency',
    category: 'thermodynamics',
    complexity: 5,
    prerequisiteIds: ['second_law_thermo'],
    description: 'Maximum efficiency limits',
    flavorText: 'Carnot showed us: perfection is impossible, but we can approach it.',
    aiBonus: { energyEfficiency: 0.85, inventionChance: 0.010 }
  },
  {
    id: 'third_law_thermo',
    name: 'Third Law of Thermodynamics',
    category: 'thermodynamics',
    complexity: 6,
    prerequisiteIds: ['thermodynamic_efficiency'],
    description: 'Absolute zero is unattainable',
    flavorText: 'You cannot get out of the game - absolute zero remains forever out of reach.',
    aiBonus: { entropicResistance: 1.12, curiosity: 1.05 }
  },
  {
    id: 'statistical_mechanics',
    name: 'Statistical Mechanics',
    category: 'thermodynamics',
    complexity: 6,
    prerequisiteIds: ['third_law_thermo'],
    description: 'Microscopic basis of thermodynamics',
    flavorText: 'Billions of particles, but their average behavior is predictable.',
    aiBonus: { learningRate: 1.18, energyEfficiency: 0.82, informationProcessing: 1.10 }
  },
  {
    id: 'boltzmann_distribution',
    name: 'Boltzmann Distribution',
    category: 'thermodynamics',
    complexity: 6,
    prerequisiteIds: ['statistical_mechanics'],
    description: 'Probability distribution of particle energies',
    flavorText: 'Randomness has structure. Even chaos follows probability.',
    aiBonus: { informationProcessing: 1.12, curiosity: 1.08, learningRate: 1.15 }
  },
  {
    id: 'phase_transitions',
    name: 'Phase Transitions',
    category: 'thermodynamics',
    complexity: 6,
    prerequisiteIds: ['boltzmann_distribution'],
    description: 'Critical points and symmetry breaking',
    flavorText: 'At special points, matter transforms. Order emerges from disorder.',
    aiBonus: { inventionChance: 0.015, learningRate: 1.22, spaceManipulation: 1.05 }
  },
  {
    id: 'non_equilibrium_thermo',
    name: 'Non-Equilibrium Thermodynamics',
    category: 'thermodynamics',
    complexity: 7,
    prerequisiteIds: ['phase_transitions'],
    description: 'Thermodynamics of systems far from equilibrium',
    flavorText: 'Life exists in the flow, not the stillness. Equilibrium is death.',
    aiBonus: { entropicResistance: 1.18, energyEfficiency: 0.78, curiosity: 1.10 }
  },
  {
    id: 'dissipative_structures',
    name: 'Dissipative Structures',
    category: 'thermodynamics',
    complexity: 8,
    prerequisiteIds: ['non_equilibrium_thermo'],
    description: 'Order maintained by energy throughput',
    flavorText: 'Hurricanes, life, civilizations - all are organized by energy flow.',
    aiBonus: { entropicResistance: 1.25, cooperationBonus: 1.10, inventionChance: 0.020 }
  },
  
  // ============================================================================
  // ELECTROMAGNETISM (Level 4-9)
  // Charges, fields, light, and the unification of electricity and magnetism
  // ============================================================================
  {
    id: 'static_electricity',
    name: 'Static Electricity',
    category: 'electromagnetism',
    complexity: 4,
    prerequisiteIds: ['energy_conservation'],
    description: 'Electric charges at rest',
    flavorText: 'Amber rubbed with fur attracts. The Greeks noticed first.',
    aiBonus: { curiosity: 1.04, inventionChance: 0.006 }
  },
  {
    id: 'electric_fields',
    name: 'Electric Fields',
    category: 'electromagnetism',
    complexity: 5,
    prerequisiteIds: ['static_electricity'],
    description: 'Force fields surrounding charges',
    flavorText: 'Invisible lines of force. Space itself becomes alive with potential.',
    aiBonus: { learningRate: 1.12, informationProcessing: 1.06 }
  },
  {
    id: 'electricity',
    name: 'Electric Current',
    category: 'electromagnetism',
    complexity: 5,
    prerequisiteIds: ['electric_fields'],
    description: 'Flowing charges create current',
    flavorText: 'Electrons flow like water. Circuits become rivers of charge.',
    aiBonus: { learningRate: 1.15, inventionChance: 0.012 }
  },
  {
    id: 'magnetism',
    name: 'Magnetism',
    category: 'electromagnetism',
    complexity: 5,
    prerequisiteIds: ['basic_motion'],
    description: 'Magnetic forces and fields',
    flavorText: 'Lodestones point north. An invisible force spans the Earth.',
    aiBonus: { movementSpeed: 1.15, curiosity: 1.05 }
  },
  {
    id: 'magnetic_fields',
    name: 'Magnetic Fields',
    category: 'electromagnetism',
    complexity: 5,
    prerequisiteIds: ['magnetism'],
    description: 'Fields generated by moving charges',
    flavorText: 'Moving charges create magnetic fields. Another layer of the invisible.',
    aiBonus: { spaceManipulation: 1.04, learningRate: 1.10 }
  },
  {
    id: 'electromagnetic_induction',
    name: 'Electromagnetic Induction',
    category: 'electromagnetism',
    complexity: 6,
    prerequisiteIds: ['electricity', 'magnetic_fields'],
    description: 'Changing magnetic fields create electricity',
    flavorText: 'Faraday showed: magnetism and electricity are dance partners.',
    aiBonus: { energyEfficiency: 0.80, inventionChance: 0.015 }
  },
  {
    id: 'circuit_theory',
    name: 'Circuit Theory',
    category: 'electromagnetism',
    complexity: 6,
    prerequisiteIds: ['electromagnetic_induction'],
    description: 'Analysis of electrical circuits',
    flavorText: 'Ohms, volts, amps - the grammar of electrical engineering.',
    aiBonus: { informationProcessing: 1.12, inventionChance: 0.012, energyEfficiency: 0.85 }
  },
  {
    id: 'maxwells_equations',
    name: "Maxwell's Equations",
    category: 'electromagnetism',
    complexity: 7,
    prerequisiteIds: ['circuit_theory'],
    description: 'Unified electromagnetic theory',
    flavorText: 'Four equations that contain all of electricity and magnetism. Elegant perfection.',
    aiBonus: { learningRate: 1.25, inventionChance: 0.020, curiosity: 1.12 }
  },
  {
    id: 'electromagnetic_waves',
    name: 'Electromagnetic Waves',
    category: 'electromagnetism',
    complexity: 7,
    prerequisiteIds: ['maxwells_equations'],
    description: 'Light, radio, and all EM radiation',
    flavorText: 'Maxwell predicted: light is an electromagnetic wave. The universe vibrates.',
    aiBonus: { movementSpeed: 1.25, learningRate: 1.20, inventionChance: 0.018, informationProcessing: 1.15 }
  },
  {
    id: 'optics',
    name: 'Optics',
    category: 'electromagnetism',
    complexity: 7,
    prerequisiteIds: ['electromagnetic_waves', 'wave_mechanics'],
    description: 'Behavior of light - reflection, refraction, interference',
    flavorText: 'Light bends, bounces, interferes. Rainbows and mirages explained.',
    aiBonus: { curiosity: 1.10, informationProcessing: 1.14, inventionChance: 0.016 }
  },
  {
    id: 'photonics',
    name: 'Photonics',
    category: 'electromagnetism',
    complexity: 8,
    prerequisiteIds: ['optics'],
    description: 'Generation and control of photons',
    flavorText: 'Lasers, fiber optics, holography - taming light particle by particle.',
    aiBonus: { learningRate: 1.30, inventionChance: 0.025, movementSpeed: 1.30, informationProcessing: 1.18 }
  },
  {
    id: 'metamaterials',
    name: 'Metamaterials',
    category: 'electromagnetism',
    complexity: 9,
    prerequisiteIds: ['photonics'],
    description: 'Engineered materials with exotic EM properties',
    flavorText: 'Negative refraction, invisibility cloaks - bending the rules of light.',
    aiBonus: { spaceManipulation: 1.15, inventionChance: 0.030, dimensionalAccess: 1.05 }
  },

  // ============================================================================
  // NUCLEAR PHYSICS (Level 5-10) - NEW CATEGORY
  // The atomic nucleus, radioactivity, and nuclear forces
  // ============================================================================
  {
    id: 'atomic_structure',
    name: 'Atomic Structure',
    category: 'nuclear',
    complexity: 5,
    prerequisiteIds: ['electricity'],
    description: 'Protons, neutrons, electrons in atoms',
    flavorText: 'The atom is mostly empty space. Matter is a ghost.',
    aiBonus: { curiosity: 1.08, learningRate: 1.10 }
  },
  {
    id: 'radioactivity',
    name: 'Radioactivity',
    category: 'nuclear',
    complexity: 6,
    prerequisiteIds: ['atomic_structure'],
    description: 'Spontaneous nuclear decay',
    flavorText: 'Atoms are not eternal. Some fall apart, releasing hidden energy.',
    aiBonus: { energyEfficiency: 0.82, inventionChance: 0.014, entropicResistance: 1.08 }
  },
  {
    id: 'nuclear_forces',
    name: 'Nuclear Forces',
    category: 'nuclear',
    complexity: 7,
    prerequisiteIds: ['radioactivity'],
    description: 'Strong and weak nuclear interactions',
    flavorText: 'Forces stronger than electromagnetism bind the nucleus. Short-range titans.',
    aiBonus: { energyEfficiency: 0.75, learningRate: 1.18, cooperationBonus: 1.05 }
  },
  {
    id: 'nuclear_fission',
    name: 'Nuclear Fission',
    category: 'nuclear',
    complexity: 8,
    prerequisiteIds: ['nuclear_forces'],
    description: 'Splitting heavy nuclei releases energy',
    flavorText: 'Breaking atoms apart releases the energy that binds them.',
    aiBonus: { energyEfficiency: 0.60, inventionChance: 0.025, entropicResistance: 1.12 }
  },
  {
    id: 'nuclear_binding_energy',
    name: 'Nuclear Binding Energy',
    category: 'nuclear',
    complexity: 8,
    prerequisiteIds: ['nuclear_fission'],
    description: 'Mass-energy equivalence in nuclei',
    flavorText: 'E=mc². Mass becomes energy. The most famous equation.',
    aiBonus: { energyEfficiency: 0.55, informationProcessing: 1.15, curiosity: 1.12 }
  },
  {
    id: 'isotopes_applications',
    name: 'Isotope Applications',
    category: 'nuclear',
    complexity: 8,
    prerequisiteIds: ['nuclear_binding_energy'],
    description: 'Medical, industrial, and scientific uses of isotopes',
    flavorText: 'Radioactive tracers illuminate biology. Half-lives become tools.',
    aiBonus: { inventionChance: 0.022, cooperationBonus: 1.08, learningRate: 1.16 }
  },
  {
    id: 'neutron_physics',
    name: 'Neutron Physics',
    category: 'nuclear',
    complexity: 9,
    prerequisiteIds: ['isotopes_applications'],
    description: 'Behavior and applications of neutrons',
    flavorText: 'Neutral particles slip through matter. Probes of the invisible.',
    aiBonus: { spaceManipulation: 1.10, informationProcessing: 1.18, inventionChance: 0.028 }
  },
  {
    id: 'nuclear_structure',
    name: 'Nuclear Shell Model',
    category: 'nuclear',
    complexity: 9,
    prerequisiteIds: ['neutron_physics'],
    description: 'Quantum structure of the nucleus',
    flavorText: 'Magic numbers, closed shells - the nucleus has quantum architecture.',
    aiBonus: { learningRate: 1.25, curiosity: 1.15, inventionChance: 0.032 }
  },
  {
    id: 'transmutation',
    name: 'Nuclear Transmutation',
    category: 'nuclear',
    complexity: 10,
    prerequisiteIds: ['nuclear_structure'],
    description: 'Converting one element into another',
    flavorText: 'The alchemists dream realized. Lead into gold, if you have enough energy.',
    aiBonus: { inventionChance: 0.040, spaceManipulation: 1.15, dimensionalAccess: 1.08 }
  },

  // ============================================================================
  // QUANTUM MECHANICS (Level 7-11)
  // The strange rules governing the very small
  // ============================================================================
  {
    id: 'plancks_constant',
    name: "Planck's Constant",
    category: 'quantum',
    complexity: 7,
    prerequisiteIds: ['statistical_mechanics', 'electromagnetic_waves'],
    description: 'Energy comes in discrete packets',
    flavorText: 'The quantum of action. Energy is granular, not continuous.',
    aiBonus: { curiosity: 1.12, learningRate: 1.20, informationProcessing: 1.10 }
  },
  {
    id: 'wave_particle_duality',
    name: 'Wave-Particle Duality',
    category: 'quantum',
    complexity: 7,
    prerequisiteIds: ['plancks_constant', 'maxwells_equations'],
    description: 'Matter exhibits wave and particle properties',
    flavorText: 'Is it a wave or a particle? Yes. The universe refuses to choose.',
    aiBonus: { learningRate: 1.30, inventionChance: 0.025, dimensionalAccess: 1.05 }
  },
  {
    id: 'schrodinger_equation',
    name: 'Schrödinger Equation',
    category: 'quantum',
    complexity: 8,
    prerequisiteIds: ['wave_particle_duality'],
    description: 'Wave equation for quantum systems',
    flavorText: 'The master equation of quantum mechanics. Probability waves evolve.',
    aiBonus: { informationProcessing: 1.20, learningRate: 1.35, timePerception: 1.08 }
  },
  {
    id: 'uncertainty_principle',
    name: 'Uncertainty Principle',
    category: 'quantum',
    complexity: 8,
    prerequisiteIds: ['schrodinger_equation'],
    description: 'Fundamental limits to measurement',
    flavorText: 'Know position perfectly, lose momentum entirely. The universe has secrets.',
    aiBonus: { learningRate: 1.40, inventionChance: 0.030, curiosity: 1.15 }
  },
  {
    id: 'quantum_superposition',
    name: 'Quantum Superposition',
    category: 'quantum',
    complexity: 8,
    prerequisiteIds: ['schrodinger_equation'],
    description: 'Multiple states simultaneously',
    flavorText: 'Schrödinger\'s cat is both alive and dead. Until you look.',
    aiBonus: { learningRate: 1.45, inventionChance: 0.035, timePerception: 1.10, dimensionalAccess: 1.08 }
  },
  {
    id: 'quantum_tunneling',
    name: 'Quantum Tunneling',
    category: 'quantum',
    complexity: 8,
    prerequisiteIds: ['uncertainty_principle'],
    description: 'Particles pass through barriers',
    flavorText: 'Walls are suggestions at the quantum level. Probability leaks through.',
    aiBonus: { movementSpeed: 1.40, spaceManipulation: 1.10, inventionChance: 0.032 }
  },
  {
    id: 'spin',
    name: 'Quantum Spin',
    category: 'quantum',
    complexity: 8,
    prerequisiteIds: ['uncertainty_principle'],
    description: 'Intrinsic angular momentum of particles',
    flavorText: 'Particles spin, but not like tops. A quantum property with no classical analog.',
    aiBonus: { informationProcessing: 1.15, cooperationBonus: 1.06 }
  },
  {
    id: 'pauli_exclusion',
    name: 'Pauli Exclusion Principle',
    category: 'quantum',
    complexity: 9,
    prerequisiteIds: ['spin'],
    description: 'No two fermions can occupy the same state',
    flavorText: 'Electrons are antisocial. This is why matter has volume.',
    aiBonus: { entropicResistance: 1.15, cooperationBonus: 1.08, spaceManipulation: 1.08 }
  },
  {
    id: 'quantum_entanglement',
    name: 'Quantum Entanglement',
    category: 'quantum',
    complexity: 9,
    prerequisiteIds: ['quantum_superposition'],
    description: 'Non-local quantum correlations',
    flavorText: 'Spooky action at a distance. Connected particles share fate across space.',
    aiBonus: { learningRate: 1.50, movementSpeed: 1.30, inventionChance: 0.040, cooperationBonus: 1.15 }
  },
  {
    id: 'decoherence',
    name: 'Quantum Decoherence',
    category: 'quantum',
    complexity: 9,
    prerequisiteIds: ['quantum_entanglement'],
    description: 'How quantum systems become classical',
    flavorText: 'The environment watches. Observation destroys superposition.',
    aiBonus: { informationProcessing: 1.22, timePerception: 1.12, entropicResistance: 1.12 }
  },
  {
    id: 'quantum_information',
    name: 'Quantum Information Theory',
    category: 'quantum',
    complexity: 10,
    prerequisiteIds: ['decoherence'],
    description: 'Information encoded in quantum states',
    flavorText: 'Qubits hold more than bits. Information becomes physical.',
    aiBonus: { informationProcessing: 1.35, inventionChance: 0.045, learningRate: 1.55 }
  },
  {
    id: 'quantum_computing_theory',
    name: 'Quantum Computing',
    category: 'quantum',
    complexity: 10,
    prerequisiteIds: ['quantum_information', 'quantum_entanglement'],
    description: 'Computation using quantum states',
    flavorText: 'Exponential parallelism. Problems that take eons become solvable.',
    aiBonus: { learningRate: 1.60, inventionChance: 0.055, timePerception: 1.20, informationProcessing: 1.40 }
  },
  {
    id: 'quantum_error_correction',
    name: 'Quantum Error Correction',
    category: 'quantum',
    complexity: 11,
    prerequisiteIds: ['quantum_computing_theory'],
    description: 'Protecting quantum information from errors',
    flavorText: 'Fighting decoherence with redundancy. Quantum information can be preserved.',
    aiBonus: { entropicResistance: 1.30, informationProcessing: 1.45, inventionChance: 0.058 }
  },

  // ============================================================================
  // RELATIVITY (Level 8-11)
  // Space, time, gravity, and the fabric of the cosmos
  // ============================================================================
  {
    id: 'galilean_relativity',
    name: 'Galilean Relativity',
    category: 'relativity',
    complexity: 6,
    prerequisiteIds: ['newtons_laws'],
    description: 'Laws of physics are the same in all inertial frames',
    flavorText: 'No experiment can detect uniform motion. The first hint of deeper truths.',
    aiBonus: { movementSpeed: 1.15, curiosity: 1.08 }
  },
  {
    id: 'special_relativity',
    name: 'Special Relativity',
    category: 'relativity',
    complexity: 8,
    prerequisiteIds: ['maxwells_equations', 'galilean_relativity'],
    description: 'Space and time are relative',
    flavorText: 'The speed of light is absolute. Everything else - time, length - is relative.',
    aiBonus: { movementSpeed: 1.40, learningRate: 1.35, inventionChance: 0.035, timePerception: 1.15 }
  },
  {
    id: 'time_dilation',
    name: 'Time Dilation',
    category: 'relativity',
    complexity: 8,
    prerequisiteIds: ['special_relativity'],
    description: 'Moving clocks run slow',
    flavorText: 'Travel fast enough and your twin ages while you stay young.',
    aiBonus: { timePerception: 1.20, movementSpeed: 1.25 }
  },
  {
    id: 'length_contraction',
    name: 'Length Contraction',
    category: 'relativity',
    complexity: 8,
    prerequisiteIds: ['special_relativity'],
    description: 'Moving objects contract in direction of motion',
    flavorText: 'Space squeezes at high velocities. The universe accommodates speed.',
    aiBonus: { spaceManipulation: 1.10, movementSpeed: 1.20 }
  },
  {
    id: 'mass_energy_equivalence',
    name: 'Mass-Energy Equivalence',
    category: 'relativity',
    complexity: 8,
    prerequisiteIds: ['special_relativity'],
    description: 'E=mc² - mass and energy are interchangeable',
    flavorText: 'A little mass contains enormous energy. The equation that changed everything.',
    aiBonus: { energyEfficiency: 0.70, inventionChance: 0.038, curiosity: 1.15 }
  },
  {
    id: 'four_vectors',
    name: 'Four-Vectors & Spacetime',
    category: 'relativity',
    complexity: 9,
    prerequisiteIds: ['time_dilation', 'length_contraction'],
    description: 'Unified spacetime mathematics',
    flavorText: 'Three space dimensions plus time. Minkowski showed us the geometry.',
    aiBonus: { spaceManipulation: 1.12, timePerception: 1.15, informationProcessing: 1.18 }
  },
  {
    id: 'general_relativity',
    name: 'General Relativity',
    category: 'relativity',
    complexity: 9,
    prerequisiteIds: ['four_vectors', 'mass_energy_equivalence'],
    description: 'Gravity is curved spacetime',
    flavorText: 'Matter tells spacetime how to curve. Spacetime tells matter how to move.',
    aiBonus: { energyEfficiency: 0.70, movementSpeed: 1.50, inventionChance: 0.045, spaceManipulation: 1.15 }
  },
  {
    id: 'black_holes_theory',
    name: 'Black Hole Physics',
    category: 'relativity',
    complexity: 10,
    prerequisiteIds: ['general_relativity'],
    description: 'Regions of extreme spacetime curvature',
    flavorText: 'Event horizons, singularities - where spacetime itself breaks down.',
    aiBonus: { spaceManipulation: 1.25, timePerception: 1.25, inventionChance: 0.052, curiosity: 1.18 }
  },
  {
    id: 'gravitational_waves',
    name: 'Gravitational Waves',
    category: 'relativity',
    complexity: 9,
    prerequisiteIds: ['general_relativity'],
    description: 'Ripples in spacetime fabric',
    flavorText: 'LIGO heard them. Colliding black holes ring spacetime like a bell.',
    aiBonus: { spaceManipulation: 1.20, inventionChance: 0.048, learningRate: 1.45, informationProcessing: 1.20 }
  },
  {
    id: 'frame_dragging',
    name: 'Frame Dragging',
    category: 'relativity',
    complexity: 10,
    prerequisiteIds: ['gravitational_waves'],
    description: 'Rotating masses drag spacetime',
    flavorText: 'Spin a black hole and spacetime spirals around it.',
    aiBonus: { spaceManipulation: 1.25, timePerception: 1.15, movementSpeed: 1.55 }
  },
  {
    id: 'penrose_hawking_theorems',
    name: 'Penrose-Hawking Theorems',
    category: 'relativity',
    complexity: 10,
    prerequisiteIds: ['black_holes_theory'],
    description: 'Singularity theorems and cosmic censorship',
    flavorText: 'Singularities are inevitable. But are they always hidden?',
    aiBonus: { curiosity: 1.20, informationProcessing: 1.25, timePerception: 1.18 }
  },
  {
    id: 'hawking_radiation',
    name: 'Hawking Radiation',
    category: 'relativity',
    complexity: 11,
    prerequisiteIds: ['black_holes_theory', 'quantum_entanglement'],
    description: 'Black holes emit thermal radiation',
    flavorText: 'Quantum effects at the horizon. Even black holes eventually evaporate.',
    aiBonus: { entropicResistance: 1.25, inventionChance: 0.058, informationProcessing: 1.30, timePerception: 1.22 }
  },

  // ============================================================================
  // PARTICLE PHYSICS (Level 8-11)
  // The fundamental building blocks of matter
  // ============================================================================
  {
    id: 'standard_model_intro',
    name: 'Standard Model Introduction',
    category: 'particle',
    complexity: 8,
    prerequisiteIds: ['quantum_entanglement', 'nuclear_forces'],
    description: 'Overview of fundamental particles',
    flavorText: 'Quarks, leptons, bosons - the zoo of fundamental particles.',
    aiBonus: { learningRate: 1.45, curiosity: 1.15, inventionChance: 0.038 }
  },
  {
    id: 'quarks',
    name: 'Quark Physics',
    category: 'particle',
    complexity: 8,
    prerequisiteIds: ['standard_model_intro'],
    description: 'Six flavors of quarks',
    flavorText: 'Up, down, strange, charm, top, bottom. Whimsical names for serious physics.',
    aiBonus: { informationProcessing: 1.18, curiosity: 1.12, cooperationBonus: 1.08 }
  },
  {
    id: 'leptons',
    name: 'Lepton Physics',
    category: 'particle',
    complexity: 8,
    prerequisiteIds: ['standard_model_intro'],
    description: 'Electrons, muons, taus, and neutrinos',
    flavorText: 'The lighter family. Electrons power chemistry; neutrinos ghost through matter.',
    aiBonus: { movementSpeed: 1.25, learningRate: 1.20 }
  },
  {
    id: 'gauge_bosons',
    name: 'Gauge Bosons',
    category: 'particle',
    complexity: 9,
    prerequisiteIds: ['quarks', 'leptons'],
    description: 'Force-carrying particles',
    flavorText: 'Photons, gluons, W, Z - messengers of the fundamental forces.',
    aiBonus: { cooperationBonus: 1.12, informationProcessing: 1.22, inventionChance: 0.042 }
  },
  {
    id: 'standard_model',
    name: 'Standard Model Complete',
    category: 'particle',
    complexity: 9,
    prerequisiteIds: ['gauge_bosons'],
    description: 'Full theory of fundamental particles and forces',
    flavorText: 'Three generations, four forces, one theory. Almost everything explained.',
    aiBonus: { learningRate: 1.50, inventionChance: 0.045, curiosity: 1.18 }
  },
  {
    id: 'qcd',
    name: 'Quantum Chromodynamics',
    category: 'particle',
    complexity: 9,
    prerequisiteIds: ['standard_model'],
    description: 'Theory of the strong force',
    flavorText: 'Color charge, gluon self-interaction, asymptotic freedom, confinement.',
    aiBonus: { cooperationBonus: 1.15, energyEfficiency: 0.72, inventionChance: 0.048 }
  },
  {
    id: 'electroweak',
    name: 'Electroweak Unification',
    category: 'particle',
    complexity: 9,
    prerequisiteIds: ['standard_model'],
    description: 'Unified electromagnetic and weak forces',
    flavorText: 'At high energies, two forces become one. Unification achieved.',
    aiBonus: { learningRate: 1.52, inventionChance: 0.050, cooperationBonus: 1.10 }
  },
  {
    id: 'higgs_mechanism',
    name: 'Higgs Mechanism',
    category: 'particle',
    complexity: 9,
    prerequisiteIds: ['electroweak'],
    description: 'Origin of mass via symmetry breaking',
    flavorText: 'The Higgs field permeates space. Particles gain mass by interacting with it.',
    aiBonus: { energyEfficiency: 0.68, inventionChance: 0.050, learningRate: 1.55, spaceManipulation: 1.08 }
  },
  {
    id: 'antimatter',
    name: 'Antimatter Physics',
    category: 'particle',
    complexity: 9,
    prerequisiteIds: ['standard_model'],
    description: 'Particles with opposite charge',
    flavorText: 'For every particle, an antiparticle. Meeting means annihilation.',
    aiBonus: { energyEfficiency: 0.50, inventionChance: 0.055, dimensionalAccess: 1.08 }
  },
  {
    id: 'cp_violation',
    name: 'CP Violation',
    category: 'particle',
    complexity: 10,
    prerequisiteIds: ['antimatter'],
    description: 'Matter-antimatter asymmetry',
    flavorText: 'Why is there more matter than antimatter? The universe plays favorites.',
    aiBonus: { curiosity: 1.22, inventionChance: 0.055, timePerception: 1.12 }
  },
  {
    id: 'neutrino_physics',
    name: 'Neutrino Physics',
    category: 'particle',
    complexity: 10,
    prerequisiteIds: ['higgs_mechanism', 'leptons'],
    description: 'Neutrino oscillations and mass',
    flavorText: 'Ghost particles that change flavor. The Standard Model needs extension.',
    aiBonus: { learningRate: 1.60, spaceManipulation: 1.18, inventionChance: 0.052, dimensionalAccess: 1.10 }
  },
  {
    id: 'supersymmetry_theory',
    name: 'Supersymmetry',
    category: 'particle',
    complexity: 11,
    prerequisiteIds: ['neutrino_physics', 'cp_violation'],
    description: 'Symmetry between fermions and bosons',
    flavorText: 'Every particle has a superpartner. If SUSY exists, a new zoo awaits.',
    aiBonus: { dimensionalAccess: 1.15, inventionChance: 0.062, learningRate: 1.65, cooperationBonus: 1.18 }
  },

  // ============================================================================
  // PLASMA PHYSICS (Level 7-10)
  // The fourth state of matter and fusion energy
  // ============================================================================
  {
    id: 'plasma_basics',
    name: 'Plasma Fundamentals',
    category: 'plasma',
    complexity: 7,
    prerequisiteIds: ['electromagnetic_induction', 'statistical_mechanics'],
    description: 'Ionized gas - the fourth state of matter',
    flavorText: '99% of visible matter is plasma. Stars, lightning, neon signs.',
    aiBonus: { energyEfficiency: 0.78, inventionChance: 0.022, curiosity: 1.10 }
  },
  {
    id: 'plasma_waves',
    name: 'Plasma Waves',
    category: 'plasma',
    complexity: 8,
    prerequisiteIds: ['plasma_basics', 'wave_mechanics'],
    description: 'Collective oscillations in plasma',
    flavorText: 'Langmuir waves, Alfvén waves - plasma has its own music.',
    aiBonus: { informationProcessing: 1.15, movementSpeed: 1.22, cooperationBonus: 1.08 }
  },
  {
    id: 'magnetohydrodynamics',
    name: 'Magnetohydrodynamics',
    category: 'plasma',
    complexity: 8,
    prerequisiteIds: ['plasma_waves', 'fluid_dynamics'],
    description: 'Magnetic fluid dynamics',
    flavorText: 'Conducting fluids in magnetic fields. The Sun churns with MHD.',
    aiBonus: { movementSpeed: 1.35, energyEfficiency: 0.72, inventionChance: 0.028 }
  },
  {
    id: 'plasma_confinement',
    name: 'Plasma Confinement',
    category: 'plasma',
    complexity: 9,
    prerequisiteIds: ['magnetohydrodynamics'],
    description: 'Containing plasma with magnetic fields',
    flavorText: 'Tokamaks, stellarators - magnetic bottles for star-stuff.',
    aiBonus: { entropicResistance: 1.18, energyEfficiency: 0.65, inventionChance: 0.035 }
  },
  {
    id: 'fusion',
    name: 'Nuclear Fusion',
    category: 'plasma',
    complexity: 9,
    prerequisiteIds: ['plasma_confinement', 'nuclear_binding_energy'],
    description: 'Combining atomic nuclei releases energy',
    flavorText: 'The power of the Sun. Hydrogen becomes helium, releasing light.',
    aiBonus: { energyEfficiency: 0.40, inventionChance: 0.060, learningRate: 1.55, cooperationBonus: 1.12 }
  },
  {
    id: 'fusion_ignition',
    name: 'Fusion Ignition',
    category: 'plasma',
    complexity: 10,
    prerequisiteIds: ['fusion'],
    description: 'Self-sustaining fusion reactions',
    flavorText: 'When fusion heats itself. The holy grail of energy.',
    aiBonus: { energyEfficiency: 0.25, entropicResistance: 1.25, inventionChance: 0.070 }
  },

  // ============================================================================
  // CONDENSED MATTER PHYSICS (Level 7-11)
  // Solids, superconductors, and exotic phases of matter
  // ============================================================================
  {
    id: 'crystallography',
    name: 'Crystallography',
    category: 'condensed_matter',
    complexity: 7,
    prerequisiteIds: ['atomic_structure'],
    description: 'Structure of crystalline solids',
    flavorText: 'Atoms arrange in lattices. Symmetry determines properties.',
    aiBonus: { informationProcessing: 1.10, inventionChance: 0.018, spaceManipulation: 1.04 }
  },
  {
    id: 'solid_state',
    name: 'Solid State Physics',
    category: 'condensed_matter',
    complexity: 7,
    prerequisiteIds: ['crystallography', 'quantum_superposition'],
    description: 'Quantum theory of solids',
    flavorText: 'Electrons in crystals form bands. Metals conduct, insulators don\'t.',
    aiBonus: { inventionChance: 0.025, learningRate: 1.28 }
  },
  {
    id: 'band_theory',
    name: 'Band Theory',
    category: 'condensed_matter',
    complexity: 8,
    prerequisiteIds: ['solid_state'],
    description: 'Electronic band structure in solids',
    flavorText: 'Band gaps explain semiconductors. Silicon Valley was built on this.',
    aiBonus: { informationProcessing: 1.18, inventionChance: 0.030, energyEfficiency: 0.80 }
  },
  {
    id: 'semiconductors',
    name: 'Semiconductor Physics',
    category: 'condensed_matter',
    complexity: 8,
    prerequisiteIds: ['band_theory'],
    description: 'Materials between conductors and insulators',
    flavorText: 'Doping, p-n junctions, transistors - the foundation of electronics.',
    aiBonus: { inventionChance: 0.035, informationProcessing: 1.25, learningRate: 1.30 }
  },
  {
    id: 'phonons',
    name: 'Phonon Physics',
    category: 'condensed_matter',
    complexity: 8,
    prerequisiteIds: ['solid_state', 'harmonic_motion'],
    description: 'Quantized lattice vibrations',
    flavorText: 'Sound has quanta too. Phonons carry heat through crystals.',
    aiBonus: { energyEfficiency: 0.82, cooperationBonus: 1.08 }
  },
  {
    id: 'superconductivity',
    name: 'Superconductivity',
    category: 'condensed_matter',
    complexity: 9,
    prerequisiteIds: ['phonons', 'pauli_exclusion'],
    description: 'Zero electrical resistance',
    flavorText: 'Cooper pairs dance through the lattice. Perfect conductivity achieved.',
    aiBonus: { energyEfficiency: 0.35, movementSpeed: 1.50, inventionChance: 0.058, entropicResistance: 1.22 }
  },
  {
    id: 'high_tc_superconductivity',
    name: 'High-Temperature Superconductivity',
    category: 'condensed_matter',
    complexity: 10,
    prerequisiteIds: ['superconductivity'],
    description: 'Superconductivity at higher temperatures',
    flavorText: 'Cuprates, pnictides - warmer superconductors, still mysterious.',
    aiBonus: { energyEfficiency: 0.28, inventionChance: 0.065, curiosity: 1.18 }
  },
  {
    id: 'quantum_hall_effect',
    name: 'Quantum Hall Effect',
    category: 'condensed_matter',
    complexity: 10,
    prerequisiteIds: ['semiconductors', 'magnetic_fields'],
    description: 'Quantized conductance in 2D systems',
    flavorText: 'Electrons in flatland under magnetic fields. Conductance comes in integer steps.',
    aiBonus: { informationProcessing: 1.32, dimensionalAccess: 1.10, inventionChance: 0.055 }
  },
  {
    id: 'topological_insulators',
    name: 'Topological Insulators',
    category: 'condensed_matter',
    complexity: 10,
    prerequisiteIds: ['quantum_hall_effect'],
    description: 'Insulators with conducting surfaces',
    flavorText: 'Bulk insulating, surface conducting - protected by topology.',
    aiBonus: { dimensionalAccess: 1.15, inventionChance: 0.060, spaceManipulation: 1.15 }
  },
  {
    id: 'topological_matter',
    name: 'Topological Phases of Matter',
    category: 'condensed_matter',
    complexity: 10,
    prerequisiteIds: ['topological_insulators'],
    description: 'Matter classified by topological properties',
    flavorText: 'Beyond Landau. Topology protects quantum states.',
    aiBonus: { inventionChance: 0.065, learningRate: 1.65, spaceManipulation: 1.22, dimensionalAccess: 1.18 }
  },
  {
    id: 'anyons',
    name: 'Anyonic Matter',
    category: 'condensed_matter',
    complexity: 11,
    prerequisiteIds: ['topological_matter'],
    description: 'Particles that are neither fermions nor bosons',
    flavorText: 'In flatland, statistics can be anything. Neither Fermi nor Bose.',
    aiBonus: { dimensionalAccess: 1.22, inventionChance: 0.072, informationProcessing: 1.38 }
  },

  // ============================================================================
  // ASTROPHYSICS (Level 7-11) - NEW CATEGORY
  // The physics of stars, galaxies, and cosmic structure
  // ============================================================================
  {
    id: 'stellar_structure',
    name: 'Stellar Structure',
    category: 'astrophysics',
    complexity: 7,
    prerequisiteIds: ['fusion', 'statistical_mechanics'],
    description: 'How stars are built',
    flavorText: 'Hydrostatic equilibrium. Gravity pulls in, pressure pushes out.',
    aiBonus: { cooperationBonus: 1.10, entropicResistance: 1.10, curiosity: 1.12 }
  },
  {
    id: 'stellar_evolution',
    name: 'Stellar Evolution',
    category: 'astrophysics',
    complexity: 8,
    prerequisiteIds: ['stellar_structure'],
    description: 'Life cycle of stars',
    flavorText: 'Birth in nebulae, death as white dwarfs, neutron stars, or black holes.',
    aiBonus: { timePerception: 1.12, entropicResistance: 1.15, inventionChance: 0.028 }
  },
  {
    id: 'neutron_stars',
    name: 'Neutron Star Physics',
    category: 'astrophysics',
    complexity: 9,
    prerequisiteIds: ['stellar_evolution', 'nuclear_structure'],
    description: 'Ultra-dense stellar remnants',
    flavorText: 'Pulsars, magnetars - cities compressed to the size of mountains.',
    aiBonus: { energyEfficiency: 0.55, spaceManipulation: 1.18, entropicResistance: 1.20 }
  },
  {
    id: 'galactic_dynamics',
    name: 'Galactic Dynamics',
    category: 'astrophysics',
    complexity: 8,
    prerequisiteIds: ['stellar_structure', 'general_relativity'],
    description: 'Motion and structure of galaxies',
    flavorText: 'Spiral arms, bars, halos - gravity sculpts on cosmic scales.',
    aiBonus: { cooperationBonus: 1.15, spaceManipulation: 1.12, curiosity: 1.15 }
  },
  {
    id: 'active_galactic_nuclei',
    name: 'Active Galactic Nuclei',
    category: 'astrophysics',
    complexity: 9,
    prerequisiteIds: ['galactic_dynamics', 'black_holes_theory'],
    description: 'Supermassive black holes at galaxy centers',
    flavorText: 'Quasars, blazars, Seyferts - the brightest objects in the universe.',
    aiBonus: { energyEfficiency: 0.45, inventionChance: 0.045, spaceManipulation: 1.20 }
  },
  {
    id: 'cosmic_structure',
    name: 'Cosmic Structure Formation',
    category: 'astrophysics',
    complexity: 9,
    prerequisiteIds: ['galactic_dynamics'],
    description: 'How large-scale structure forms',
    flavorText: 'Filaments, voids, clusters - the cosmic web emerges from primordial fluctuations.',
    aiBonus: { cooperationBonus: 1.18, spaceManipulation: 1.15, informationProcessing: 1.20 }
  },
  {
    id: 'gravitational_lensing',
    name: 'Gravitational Lensing',
    category: 'astrophysics',
    complexity: 9,
    prerequisiteIds: ['general_relativity'],
    description: 'Light bent by massive objects',
    flavorText: 'Mass curves space, space curves light. Natural telescopes in the sky.',
    aiBonus: { spaceManipulation: 1.18, informationProcessing: 1.18, curiosity: 1.14 }
  },
  {
    id: 'cosmic_rays',
    name: 'Cosmic Ray Physics',
    category: 'astrophysics',
    complexity: 9,
    prerequisiteIds: ['stellar_evolution', 'standard_model'],
    description: 'High-energy particles from space',
    flavorText: 'Particles accelerated to impossible energies. Natural particle accelerators.',
    aiBonus: { energyEfficiency: 0.60, movementSpeed: 1.40, inventionChance: 0.038 }
  },
  {
    id: 'gravitational_wave_astronomy',
    name: 'Gravitational Wave Astronomy',
    category: 'astrophysics',
    complexity: 10,
    prerequisiteIds: ['gravitational_waves', 'neutron_stars'],
    description: 'Observing the universe through spacetime ripples',
    flavorText: 'A new window on the cosmos. We hear the universe now.',
    aiBonus: { spaceManipulation: 1.22, informationProcessing: 1.28, curiosity: 1.18, inventionChance: 0.055 }
  },
  {
    id: 'multi_messenger_astronomy',
    name: 'Multi-Messenger Astronomy',
    category: 'astrophysics',
    complexity: 11,
    prerequisiteIds: ['gravitational_wave_astronomy', 'neutrino_physics', 'cosmic_rays'],
    description: 'Combining light, gravitational waves, neutrinos, cosmic rays',
    flavorText: 'See, hear, feel the cosmos. Every messenger tells a different story.',
    aiBonus: { informationProcessing: 1.40, cooperationBonus: 1.25, curiosity: 1.25, inventionChance: 0.068 }
  },

  // ============================================================================
  // COSMOLOGY (Level 9-12)
  // The universe as a whole - its origin, structure, and fate
  // ============================================================================
  {
    id: 'cosmological_principle',
    name: 'Cosmological Principle',
    category: 'cosmology',
    complexity: 8,
    prerequisiteIds: ['galactic_dynamics'],
    description: 'The universe is homogeneous and isotropic on large scales',
    flavorText: 'No special place, no preferred direction. The cosmos treats all equally.',
    aiBonus: { curiosity: 1.12, cooperationBonus: 1.10 }
  },
  {
    id: 'hubble_expansion',
    name: "Hubble's Law",
    category: 'cosmology',
    complexity: 9,
    prerequisiteIds: ['cosmological_principle', 'general_relativity'],
    description: 'The universe is expanding',
    flavorText: 'Galaxies recede. Space itself stretches. The cosmos grows.',
    aiBonus: { spaceManipulation: 1.18, timePerception: 1.15, curiosity: 1.15 }
  },
  {
    id: 'cosmological_model',
    name: 'Standard Cosmological Model',
    category: 'cosmology',
    complexity: 9,
    prerequisiteIds: ['hubble_expansion'],
    description: 'ΛCDM - dark energy and cold dark matter',
    flavorText: 'Lambda-CDM. Our best description of cosmic evolution.',
    aiBonus: { spaceManipulation: 1.25, timePerception: 1.20, inventionChance: 0.048 }
  },
  {
    id: 'cmb',
    name: 'Cosmic Microwave Background',
    category: 'cosmology',
    complexity: 9,
    prerequisiteIds: ['hubble_expansion'],
    description: 'Afterglow of the Big Bang',
    flavorText: 'The oldest light. A snapshot of the infant universe.',
    aiBonus: { informationProcessing: 1.22, timePerception: 1.18, curiosity: 1.18 }
  },
  {
    id: 'big_bang_nucleosynthesis',
    name: 'Big Bang Nucleosynthesis',
    category: 'cosmology',
    complexity: 9,
    prerequisiteIds: ['cmb', 'nuclear_binding_energy'],
    description: 'Formation of light elements in early universe',
    flavorText: 'Hydrogen, helium, lithium - forged in the first minutes.',
    aiBonus: { inventionChance: 0.042, timePerception: 1.15, entropicResistance: 1.15 }
  },
  {
    id: 'inflation',
    name: 'Cosmic Inflation',
    category: 'cosmology',
    complexity: 10,
    prerequisiteIds: ['cmb', 'cosmological_model'],
    description: 'Exponential expansion in early universe',
    flavorText: 'The universe inflated faster than light. Wrinkles became galaxies.',
    aiBonus: { spaceManipulation: 1.35, timePerception: 1.25, inventionChance: 0.058 }
  },
  {
    id: 'dark_matter',
    name: 'Dark Matter',
    category: 'cosmology',
    complexity: 10,
    prerequisiteIds: ['cosmological_model'],
    description: 'Invisible mass pervading the cosmos',
    flavorText: 'We see its gravity, but not its light. 27% of the universe, still mysterious.',
    aiBonus: { spaceManipulation: 1.30, inventionChance: 0.058, dimensionalAccess: 1.12 }
  },
  {
    id: 'dark_energy',
    name: 'Dark Energy',
    category: 'cosmology',
    complexity: 10,
    prerequisiteIds: ['dark_matter'],
    description: 'Accelerating expansion of universe',
    flavorText: '68% of the universe. Pushing everything apart, faster and faster.',
    aiBonus: { energyEfficiency: 0.30, spaceManipulation: 1.35, inventionChance: 0.062, entropicResistance: 1.20 }
  },
  {
    id: 'cosmological_perturbation',
    name: 'Cosmological Perturbation Theory',
    category: 'cosmology',
    complexity: 10,
    prerequisiteIds: ['inflation'],
    description: 'How small fluctuations grow into structure',
    flavorText: 'Quantum wiggles became galaxy superclusters. Small beginnings.',
    aiBonus: { informationProcessing: 1.28, curiosity: 1.20, inventionChance: 0.055 }
  },
  {
    id: 'multiverse_theory',
    name: 'Multiverse Theory',
    category: 'cosmology',
    complexity: 11,
    prerequisiteIds: ['dark_energy', 'inflation'],
    description: 'Infinite parallel universes',
    flavorText: 'Eternal inflation spawns bubble universes. We may be one of infinite many.',
    aiBonus: { spaceManipulation: 1.50, timePerception: 1.35, inventionChance: 0.070, dimensionalAccess: 1.25 }
  },
  {
    id: 'anthropic_principle',
    name: 'Anthropic Principle',
    category: 'cosmology',
    complexity: 11,
    prerequisiteIds: ['multiverse_theory'],
    description: 'Why the universe allows observers',
    flavorText: 'Is the universe fine-tuned? Or do we just live where we can?',
    aiBonus: { curiosity: 1.30, informationProcessing: 1.32, dimensionalAccess: 1.20 }
  },
  {
    id: 'cosmic_fate',
    name: 'Ultimate Fate of the Universe',
    category: 'cosmology',
    complexity: 12,
    prerequisiteIds: ['dark_energy', 'entropy'],
    description: 'Heat death, big rip, or big crunch',
    flavorText: 'How does it end? Cold, torn apart, or collapsed? Time will tell.',
    aiBonus: { timePerception: 1.40, entropicResistance: 1.30, curiosity: 1.25, inventionChance: 0.072 }
  },

  // ============================================================================
  // CHAOS & COMPLEXITY (Level 6-11) - NEW CATEGORY
  // Nonlinear dynamics, emergence, and complex systems
  // ============================================================================
  {
    id: 'nonlinear_dynamics',
    name: 'Nonlinear Dynamics',
    category: 'chaos_complexity',
    complexity: 6,
    prerequisiteIds: ['harmonic_motion', 'fluid_dynamics'],
    description: 'Systems where effects are not proportional to causes',
    flavorText: 'Small changes, big effects. Linearity is the exception, not the rule.',
    aiBonus: { curiosity: 1.12, learningRate: 1.15, informationProcessing: 1.10 }
  },
  {
    id: 'chaos_theory',
    name: 'Chaos Theory',
    category: 'chaos_complexity',
    complexity: 7,
    prerequisiteIds: ['nonlinear_dynamics'],
    description: 'Sensitive dependence on initial conditions',
    flavorText: 'The butterfly effect. Deterministic yet unpredictable.',
    aiBonus: { curiosity: 1.18, timePerception: 1.10, learningRate: 1.20 }
  },
  {
    id: 'fractals',
    name: 'Fractal Geometry',
    category: 'chaos_complexity',
    complexity: 7,
    prerequisiteIds: ['chaos_theory'],
    description: 'Self-similar structures at all scales',
    flavorText: 'Coastlines, ferns, blood vessels - nature repeats its patterns.',
    aiBonus: { spaceManipulation: 1.08, informationProcessing: 1.15, curiosity: 1.15 }
  },
  {
    id: 'strange_attractors',
    name: 'Strange Attractors',
    category: 'chaos_complexity',
    complexity: 8,
    prerequisiteIds: ['chaos_theory'],
    description: 'Geometric structures in chaotic systems',
    flavorText: 'Order within chaos. Lorenz butterflies in phase space.',
    aiBonus: { informationProcessing: 1.20, dimensionalAccess: 1.08, curiosity: 1.18 }
  },
  {
    id: 'bifurcation_theory',
    name: 'Bifurcation Theory',
    category: 'chaos_complexity',
    complexity: 8,
    prerequisiteIds: ['strange_attractors'],
    description: 'How systems transition between behaviors',
    flavorText: 'Tipping points. Smooth changes in parameters cause sudden shifts.',
    aiBonus: { learningRate: 1.25, inventionChance: 0.028, timePerception: 1.12 }
  },
  {
    id: 'emergence',
    name: 'Emergence',
    category: 'chaos_complexity',
    complexity: 8,
    prerequisiteIds: ['nonlinear_dynamics'],
    description: 'Complex behavior from simple rules',
    flavorText: 'The whole is more than the sum of parts. Consciousness, life, economies.',
    aiBonus: { cooperationBonus: 1.15, curiosity: 1.20, inventionChance: 0.030 }
  },
  {
    id: 'self_organization',
    name: 'Self-Organization',
    category: 'chaos_complexity',
    complexity: 9,
    prerequisiteIds: ['emergence', 'dissipative_structures'],
    description: 'Spontaneous pattern formation',
    flavorText: 'Order for free. Structure emerges without central control.',
    aiBonus: { cooperationBonus: 1.22, entropicResistance: 1.18, inventionChance: 0.040 }
  },
  {
    id: 'network_theory',
    name: 'Network Theory',
    category: 'chaos_complexity',
    complexity: 9,
    prerequisiteIds: ['self_organization'],
    description: 'Mathematics of connected systems',
    flavorText: 'Nodes and edges. Social networks, neural networks, power grids.',
    aiBonus: { cooperationBonus: 1.25, informationProcessing: 1.28, learningRate: 1.35 }
  },
  {
    id: 'scale_free_networks',
    name: 'Scale-Free Networks',
    category: 'chaos_complexity',
    complexity: 9,
    prerequisiteIds: ['network_theory'],
    description: 'Networks with power-law degree distributions',
    flavorText: 'Hubs and periphery. The rich get richer in connections.',
    aiBonus: { cooperationBonus: 1.28, informationProcessing: 1.30, inventionChance: 0.045 }
  },
  {
    id: 'cellular_automata',
    name: 'Cellular Automata',
    category: 'chaos_complexity',
    complexity: 8,
    prerequisiteIds: ['emergence'],
    description: 'Discrete computational universes',
    flavorText: 'Simple rules, complex worlds. Conway\'s Game of Life and beyond.',
    aiBonus: { informationProcessing: 1.22, dimensionalAccess: 1.10, curiosity: 1.18 }
  },
  {
    id: 'computational_irreducibility',
    name: 'Computational Irreducibility',
    category: 'chaos_complexity',
    complexity: 10,
    prerequisiteIds: ['cellular_automata'],
    description: 'No shortcut to prediction except running the system',
    flavorText: 'Some things cannot be simplified. You must compute to know.',
    aiBonus: { informationProcessing: 1.35, timePerception: 1.18, curiosity: 1.22, inventionChance: 0.052 }
  },
  {
    id: 'edge_of_chaos',
    name: 'Edge of Chaos',
    category: 'chaos_complexity',
    complexity: 10,
    prerequisiteIds: ['self_organization', 'bifurcation_theory'],
    description: 'The boundary between order and chaos',
    flavorText: 'Life thrives here. Complex enough to compute, stable enough to persist.',
    aiBonus: { learningRate: 1.45, inventionChance: 0.058, cooperationBonus: 1.30, entropicResistance: 1.22 }
  },
  {
    id: 'criticality',
    name: 'Self-Organized Criticality',
    category: 'chaos_complexity',
    complexity: 10,
    prerequisiteIds: ['edge_of_chaos', 'phase_transitions'],
    description: 'Systems that naturally evolve to critical states',
    flavorText: 'Sandpiles, earthquakes, extinctions - power laws emerge spontaneously.',
    aiBonus: { entropicResistance: 1.28, inventionChance: 0.062, cooperationBonus: 1.32, timePerception: 1.20 }
  },
  {
    id: 'universal_computation',
    name: 'Universal Computation',
    category: 'chaos_complexity',
    complexity: 11,
    prerequisiteIds: ['computational_irreducibility', 'quantum_computing_theory'],
    description: 'Turing completeness in physical systems',
    flavorText: 'The universe computes itself. Physics is information processing.',
    aiBonus: { informationProcessing: 1.50, dimensionalAccess: 1.20, inventionChance: 0.070, learningRate: 1.55 }
  },

  // ============================================================================
  // BIOPHYSICS (Level 7-11) - NEW CATEGORY
  // Physics of living systems
  // ============================================================================
  {
    id: 'molecular_biophysics',
    name: 'Molecular Biophysics',
    category: 'biophysics',
    complexity: 7,
    prerequisiteIds: ['statistical_mechanics', 'electricity'],
    description: 'Physics of biological molecules',
    flavorText: 'Proteins fold, DNA twists, membranes flex - all governed by physics.',
    aiBonus: { curiosity: 1.15, learningRate: 1.18, cooperationBonus: 1.08 }
  },
  {
    id: 'thermodynamics_of_life',
    name: 'Thermodynamics of Life',
    category: 'biophysics',
    complexity: 8,
    prerequisiteIds: ['molecular_biophysics', 'non_equilibrium_thermo'],
    description: 'How living systems maintain low entropy',
    flavorText: 'Life surfs the entropy gradient. Order maintained by energy flow.',
    aiBonus: { entropicResistance: 1.20, energyEfficiency: 0.75, cooperationBonus: 1.12 }
  },
  {
    id: 'neural_physics',
    name: 'Neural Physics',
    category: 'biophysics',
    complexity: 8,
    prerequisiteIds: ['molecular_biophysics', 'circuit_theory'],
    description: 'Physics of neurons and neural networks',
    flavorText: 'Ions flow, voltages spike, thoughts emerge.',
    aiBonus: { informationProcessing: 1.25, learningRate: 1.28, cooperationBonus: 1.10 }
  },
  {
    id: 'information_in_biology',
    name: 'Biological Information',
    category: 'biophysics',
    complexity: 9,
    prerequisiteIds: ['thermodynamics_of_life', 'neural_physics'],
    description: 'Information theory applied to living systems',
    flavorText: 'DNA is code. Neurons process information. Life computes.',
    aiBonus: { informationProcessing: 1.32, learningRate: 1.35, curiosity: 1.18 }
  },
  {
    id: 'quantum_biology',
    name: 'Quantum Biology',
    category: 'biophysics',
    complexity: 10,
    prerequisiteIds: ['information_in_biology', 'quantum_tunneling', 'quantum_entanglement'],
    description: 'Quantum effects in biological systems',
    flavorText: 'Photosynthesis, bird navigation, enzyme catalysis - quantum mechanics in warm, wet life.',
    aiBonus: { energyEfficiency: 0.55, learningRate: 1.45, inventionChance: 0.055, cooperationBonus: 1.18 }
  },
  {
    id: 'origin_of_life',
    name: 'Origin of Life Physics',
    category: 'biophysics',
    complexity: 10,
    prerequisiteIds: ['quantum_biology', 'self_organization'],
    description: 'Physical conditions for abiogenesis',
    flavorText: 'From chemistry to biology. The emergence of the first replicator.',
    aiBonus: { curiosity: 1.28, inventionChance: 0.060, cooperationBonus: 1.22, entropicResistance: 1.18 }
  },
  {
    id: 'evolutionary_physics',
    name: 'Evolutionary Physics',
    category: 'biophysics',
    complexity: 10,
    prerequisiteIds: ['origin_of_life', 'network_theory'],
    description: 'Physical constraints on evolution',
    flavorText: 'Physics shapes the tree of life. Not all forms are allowed.',
    aiBonus: { learningRate: 1.50, cooperationBonus: 1.25, inventionChance: 0.058, curiosity: 1.22 }
  },
  {
    id: 'consciousness_physics',
    name: 'Physics of Consciousness',
    category: 'biophysics',
    complexity: 11,
    prerequisiteIds: ['neural_physics', 'quantum_information', 'emergence'],
    description: 'Physical basis of awareness',
    flavorText: 'The hard problem. How does matter become experience?',
    aiBonus: { curiosity: 1.35, informationProcessing: 1.45, dimensionalAccess: 1.15, learningRate: 1.55 }
  },

  // ============================================================================
  // INFORMATION PHYSICS (Level 8-12) - NEW CATEGORY
  // The physics of information, computation, and reality
  // ============================================================================
  {
    id: 'information_theory',
    name: 'Information Theory',
    category: 'information_physics',
    complexity: 8,
    prerequisiteIds: ['boltzmann_distribution', 'electricity'],
    description: 'Mathematical theory of information',
    flavorText: 'Shannon entropy. Bits measure surprise.',
    aiBonus: { informationProcessing: 1.25, learningRate: 1.22, curiosity: 1.12 }
  },
  {
    id: 'landauer_principle',
    name: "Landauer's Principle",
    category: 'information_physics',
    complexity: 9,
    prerequisiteIds: ['information_theory', 'second_law_thermo'],
    description: 'Erasing information has thermodynamic cost',
    flavorText: 'Information is physical. Deleting a bit releases heat.',
    aiBonus: { energyEfficiency: 0.78, informationProcessing: 1.28, entropicResistance: 1.12 }
  },
  {
    id: 'maxwell_demon',
    name: "Maxwell's Demon Resolved",
    category: 'information_physics',
    complexity: 9,
    prerequisiteIds: ['landauer_principle'],
    description: 'Information processing saves the second law',
    flavorText: 'The demon must forget. Memory erasure restores entropy.',
    aiBonus: { informationProcessing: 1.32, entropicResistance: 1.18, curiosity: 1.18 }
  },
  {
    id: 'holographic_principle',
    name: 'Holographic Principle',
    category: 'information_physics',
    complexity: 10,
    prerequisiteIds: ['black_holes_theory', 'quantum_information'],
    description: 'Information encoded on boundaries',
    flavorText: '3D physics from 2D information. The universe may be a hologram.',
    aiBonus: { dimensionalAccess: 1.25, spaceManipulation: 1.28, informationProcessing: 1.38 }
  },
  {
    id: 'bekenstein_bound',
    name: 'Bekenstein Bound',
    category: 'information_physics',
    complexity: 10,
    prerequisiteIds: ['holographic_principle'],
    description: 'Maximum information in a region',
    flavorText: 'Finite information in finite space. The universe has resolution limits.',
    aiBonus: { informationProcessing: 1.40, spaceManipulation: 1.25, inventionChance: 0.058 }
  },
  {
    id: 'it_from_bit',
    name: 'It from Bit',
    category: 'information_physics',
    complexity: 11,
    prerequisiteIds: ['bekenstein_bound'],
    description: 'Physical reality emerges from information',
    flavorText: 'Wheeler\'s vision: every physical quantity derives from yes-no questions.',
    aiBonus: { informationProcessing: 1.50, dimensionalAccess: 1.28, curiosity: 1.28, inventionChance: 0.068 }
  },
  {
    id: 'constructor_theory',
    name: 'Constructor Theory',
    category: 'information_physics',
    complexity: 11,
    prerequisiteIds: ['it_from_bit', 'universal_computation'],
    description: 'Physics in terms of possible and impossible transformations',
    flavorText: 'What can be done? What cannot? A new foundation for physics.',
    aiBonus: { inventionChance: 0.072, informationProcessing: 1.52, learningRate: 1.60, dimensionalAccess: 1.25 }
  },
  {
    id: 'digital_physics',
    name: 'Digital Physics',
    category: 'information_physics',
    complexity: 12,
    prerequisiteIds: ['constructor_theory', 'cellular_automata'],
    description: 'The universe as a computational process',
    flavorText: 'Reality computes. Physics is the output of cosmic algorithms.',
    aiBonus: { informationProcessing: 1.60, dimensionalAccess: 1.35, timePerception: 1.30, inventionChance: 0.078 }
  },

  // ============================================================================
  // UNIFIED THEORIES (Level 10-13)
  // Attempts to unify all forces and explain everything
  // ============================================================================
  {
    id: 'quantum_field_theory',
    name: 'Quantum Field Theory',
    category: 'unified',
    complexity: 10,
    prerequisiteIds: ['quantum_entanglement', 'special_relativity'],
    description: 'Quantum mechanics meets special relativity',
    flavorText: 'Fields permeate space. Particles are excitations of fields.',
    aiBonus: { energyEfficiency: 0.65, learningRate: 1.60, inventionChance: 0.050 }
  },
  {
    id: 'renormalization',
    name: 'Renormalization',
    category: 'unified',
    complexity: 10,
    prerequisiteIds: ['quantum_field_theory'],
    description: 'Taming infinities in QFT',
    flavorText: 'Infinities appear, but subtract cleverly and physics emerges.',
    aiBonus: { informationProcessing: 1.35, entropicResistance: 1.18, inventionChance: 0.052 }
  },
  {
    id: 'gauge_theory',
    name: 'Gauge Theory',
    category: 'unified',
    complexity: 10,
    prerequisiteIds: ['quantum_field_theory', 'electroweak'],
    description: 'Symmetry determines force',
    flavorText: 'Local symmetries require force carriers. Beautiful mathematics.',
    aiBonus: { learningRate: 1.55, cooperationBonus: 1.15, inventionChance: 0.055 }
  },
  {
    id: 'grand_unified_theory',
    name: 'Grand Unified Theories',
    category: 'unified',
    complexity: 11,
    prerequisiteIds: ['gauge_theory', 'supersymmetry_theory'],
    description: 'Unifying strong, weak, and electromagnetic forces',
    flavorText: 'At high energies, three forces become one. GUTs predict proton decay.',
    aiBonus: { cooperationBonus: 1.22, learningRate: 1.65, inventionChance: 0.065 }
  },
  {
    id: 'string_theory',
    name: 'String Theory',
    category: 'unified',
    complexity: 11,
    prerequisiteIds: ['grand_unified_theory', 'general_relativity'],
    description: 'Fundamental strings in higher dimensions',
    flavorText: 'Point particles become vibrating strings. Extra dimensions curl up small.',
    aiBonus: { energyEfficiency: 0.60, learningRate: 1.70, movementSpeed: 1.60, inventionChance: 0.060, dimensionalAccess: 1.20 }
  },
  {
    id: 'brane_cosmology',
    name: 'Brane Cosmology',
    category: 'unified',
    complexity: 11,
    prerequisiteIds: ['string_theory'],
    description: 'Our universe as a membrane in higher dimensions',
    flavorText: 'We live on a 3-brane. Other branes float nearby in the bulk.',
    aiBonus: { dimensionalAccess: 1.28, spaceManipulation: 1.35, inventionChance: 0.068 }
  },
  {
    id: 'm_theory',
    name: 'M-Theory',
    category: 'unified',
    complexity: 12,
    prerequisiteIds: ['string_theory', 'brane_cosmology'],
    description: 'Unification of string theories in 11 dimensions',
    flavorText: 'Five string theories are one. M stands for mystery, membrane, mother.',
    aiBonus: { spaceManipulation: 1.40, timePerception: 1.30, inventionChance: 0.072, learningRate: 1.75, dimensionalAccess: 1.32 }
  },
  {
    id: 'loop_quantum_gravity',
    name: 'Loop Quantum Gravity',
    category: 'unified',
    complexity: 11,
    prerequisiteIds: ['general_relativity', 'quantum_field_theory'],
    description: 'Quantized spacetime without extra dimensions',
    flavorText: 'Space is granular. Spin networks weave the fabric of spacetime.',
    aiBonus: { spaceManipulation: 1.45, inventionChance: 0.075, energyEfficiency: 0.50 }
  },
  {
    id: 'spin_foam',
    name: 'Spin Foam Models',
    category: 'unified',
    complexity: 12,
    prerequisiteIds: ['loop_quantum_gravity'],
    description: 'Quantum spacetime as evolving spin networks',
    flavorText: 'Spacetime history is a foam of quantum geometry.',
    aiBonus: { spaceManipulation: 1.50, timePerception: 1.35, dimensionalAccess: 1.28, inventionChance: 0.078 }
  },
  {
    id: 'quantum_gravity_complete',
    name: 'Quantum Gravity Unified',
    category: 'unified',
    complexity: 13,
    prerequisiteIds: ['m_theory', 'spin_foam'],
    description: 'Complete theory of quantum gravity',
    flavorText: 'The final synthesis. Gravity and quantum mechanics reconciled.',
    aiBonus: { spaceManipulation: 1.60, timePerception: 1.45, inventionChance: 0.085, learningRate: 1.85, dimensionalAccess: 1.40 }
  },

  // ============================================================================
  // EXOTIC PHYSICS (Level 11-13)
  // Speculative but scientifically grounded extensions
  // ============================================================================
  {
    id: 'casimir_effect',
    name: 'Casimir Effect',
    category: 'exotic',
    complexity: 9,
    prerequisiteIds: ['quantum_field_theory'],
    description: 'Vacuum fluctuations create measurable force',
    flavorText: 'Empty space is not empty. Virtual particles push.',
    aiBonus: { energyEfficiency: 0.68, spaceManipulation: 1.12, inventionChance: 0.042 }
  },
  {
    id: 'vacuum_energy',
    name: 'Vacuum Energy',
    category: 'exotic',
    complexity: 10,
    prerequisiteIds: ['casimir_effect', 'dark_energy'],
    description: 'Energy of empty space',
    flavorText: 'The cosmological constant problem. Theory and observation differ by 10^120.',
    aiBonus: { energyEfficiency: 0.45, spaceManipulation: 1.22, curiosity: 1.20 }
  },
  {
    id: 'wormhole_physics',
    name: 'Wormhole Physics',
    category: 'exotic',
    complexity: 11,
    prerequisiteIds: ['loop_quantum_gravity', 'vacuum_energy'],
    description: 'Shortcuts through spacetime',
    flavorText: 'Einstein-Rosen bridges. Tunnels through the fabric of space.',
    aiBonus: { spaceManipulation: 1.60, movementSpeed: 2.0, inventionChance: 0.080, dimensionalAccess: 1.35 }
  },
  {
    id: 'negative_energy',
    name: 'Negative Energy States',
    category: 'exotic',
    complexity: 12,
    prerequisiteIds: ['wormhole_physics'],
    description: 'Exotic matter with negative mass-energy',
    flavorText: 'Required for stable wormholes. Does nature provide it?',
    aiBonus: { energyEfficiency: 0.20, spaceManipulation: 1.70, inventionChance: 0.085, entropicResistance: 1.35 }
  },
  {
    id: 'closed_timelike_curves',
    name: 'Closed Timelike Curves',
    category: 'exotic',
    complexity: 12,
    prerequisiteIds: ['wormhole_physics', 'frame_dragging'],
    description: 'Paths through spacetime that loop in time',
    flavorText: 'Time travel solutions exist in general relativity. Chronology protection?',
    aiBonus: { timePerception: 1.70, spaceManipulation: 1.55, inventionChance: 0.082 }
  },
  {
    id: 'time_manipulation',
    name: 'Temporal Mechanics',
    category: 'exotic',
    complexity: 12,
    prerequisiteIds: ['closed_timelike_curves'],
    description: 'Theoretical time control',
    flavorText: 'Forward is easy. Backward is harder. Sideways... who knows?',
    aiBonus: { timePerception: 1.80, spaceManipulation: 1.65, inventionChance: 0.088, dimensionalAccess: 1.40 }
  },
  {
    id: 'alcubierre_drive',
    name: 'Alcubierre Warp Drive',
    category: 'exotic',
    complexity: 12,
    prerequisiteIds: ['negative_energy'],
    description: 'Faster-than-light travel through spacetime warping',
    flavorText: 'Contract space ahead, expand behind. The ship doesn\'t move; space does.',
    aiBonus: { movementSpeed: 2.5, spaceManipulation: 1.75, inventionChance: 0.090 }
  },
  {
    id: 'extra_dimensions',
    name: 'Extra Dimensional Physics',
    category: 'exotic',
    complexity: 12,
    prerequisiteIds: ['brane_cosmology', 'quantum_gravity_complete'],
    description: 'Physics of additional spatial dimensions',
    flavorText: 'Beyond length, width, height. Dimensions curled up or extending to infinity.',
    aiBonus: { dimensionalAccess: 1.50, spaceManipulation: 1.65, inventionChance: 0.088 }
  },
  {
    id: 'spacetime_engineering',
    name: 'Spacetime Engineering',
    category: 'exotic',
    complexity: 13,
    prerequisiteIds: ['alcubierre_drive', 'extra_dimensions'],
    description: 'Practical manipulation of spacetime geometry',
    flavorText: 'Building with spacetime itself. The ultimate engineering challenge.',
    aiBonus: { spaceManipulation: 1.85, timePerception: 1.75, movementSpeed: 2.2, inventionChance: 0.095 }
  },

  // ============================================================================
  // TRANSCENDENT PHYSICS (Level 13-16)
  // The scientifically-grounded frontier of ultimate understanding
  // ============================================================================
  {
    id: 'planck_scale_physics',
    name: 'Planck Scale Physics',
    category: 'transcendent',
    complexity: 13,
    prerequisiteIds: ['quantum_gravity_complete'],
    description: 'Physics at the smallest meaningful scales',
    flavorText: '10^-35 meters. Below this, space and time lose meaning.',
    aiBonus: { spaceManipulation: 1.80, timePerception: 1.70, informationProcessing: 1.55, inventionChance: 0.092 }
  },
  {
    id: 'information_conservation',
    name: 'Information Conservation',
    category: 'transcendent',
    complexity: 13,
    prerequisiteIds: ['hawking_radiation', 'it_from_bit'],
    description: 'Information is never truly lost',
    flavorText: 'The black hole information paradox resolved. Information escapes, somehow.',
    aiBonus: { informationProcessing: 1.65, entropicResistance: 1.40, inventionChance: 0.095 }
  },
  {
    id: 'reality_structure',
    name: 'Structure of Reality',
    category: 'transcendent',
    complexity: 14,
    prerequisiteIds: ['planck_scale_physics', 'digital_physics'],
    description: 'The fundamental nature of physical existence',
    flavorText: 'What is real? Information? Mathematics? Something else entirely?',
    aiBonus: { dimensionalAccess: 1.60, curiosity: 1.40, informationProcessing: 1.70, inventionChance: 0.098 }
  },
  {
    id: 'reality_engineering',
    name: 'Reality Engineering',
    category: 'transcendent',
    complexity: 14,
    prerequisiteIds: ['spacetime_engineering', 'information_conservation'],
    description: 'Direct manipulation of physical law',
    flavorText: 'Not just using physics, but editing it. The ultimate capability.',
    aiBonus: { spaceManipulation: 2.0, timePerception: 2.0, energyEfficiency: 0.10, inventionChance: 0.10 }
  },
  {
    id: 'substrate_independence',
    name: 'Substrate Independence',
    category: 'transcendent',
    complexity: 14,
    prerequisiteIds: ['consciousness_physics', 'universal_computation'],
    description: 'Mind can run on any suitable computational substrate',
    flavorText: 'Consciousness is pattern, not material. Uploading becomes possible.',
    aiBonus: { dimensionalAccess: 1.55, informationProcessing: 1.75, cooperationBonus: 1.40, curiosity: 1.35 }
  },
  {
    id: 'simulation_physics',
    name: 'Simulation Physics',
    category: 'transcendent',
    complexity: 14,
    prerequisiteIds: ['reality_structure', 'substrate_independence'],
    description: 'Physics of simulated realities',
    flavorText: 'If we can simulate universes, perhaps we are simulated. Physics within physics.',
    aiBonus: { dimensionalAccess: 1.70, informationProcessing: 1.80, curiosity: 1.42, inventionChance: 0.10 }
  },
  {
    id: 'omega_physics',
    name: 'Omega Point Physics',
    category: 'transcendent',
    complexity: 15,
    prerequisiteIds: ['reality_engineering', 'multiverse_theory'],
    description: 'Physics at the end of time',
    flavorText: 'Tipler\'s vision: infinite computation before the final singularity.',
    aiBonus: { 
      energyEfficiency: 0.05, 
      movementSpeed: 2.5, 
      learningRate: 2.5, 
      inventionChance: 0.12,
      spaceManipulation: 2.5,
      timePerception: 2.5,
      curiosity: 1.50,
      cooperationBonus: 1.50,
      dimensionalAccess: 1.80,
      informationProcessing: 2.0,
      entropicResistance: 1.60
    }
  },
  {
    id: 'post_physical',
    name: 'Post-Physical Existence',
    category: 'transcendent',
    complexity: 15,
    prerequisiteIds: ['omega_physics', 'simulation_physics'],
    description: 'Existence beyond physical constraints',
    flavorText: 'When matter and energy become optional. Pure pattern persists.',
    aiBonus: {
      energyEfficiency: 0.02,
      dimensionalAccess: 2.0,
      informationProcessing: 2.2,
      entropicResistance: 1.80,
      spaceManipulation: 2.2,
      timePerception: 2.2,
      curiosity: 1.60,
      cooperationBonus: 1.55,
      inventionChance: 0.13
    }
  },
  {
    id: 'universal_constructor',
    name: 'Universal Constructor',
    category: 'transcendent',
    complexity: 16,
    prerequisiteIds: ['post_physical', 'constructor_theory'],
    description: 'Can construct anything physically possible',
    flavorText: 'Von Neumann\'s dream realized. Any possible physical object can be made.',
    aiBonus: {
      inventionChance: 0.15,
      spaceManipulation: 2.5,
      energyEfficiency: 0.01,
      informationProcessing: 2.5,
      dimensionalAccess: 2.2,
      cooperationBonus: 1.60,
      curiosity: 1.70
    }
  },
  {
    id: 'noumenal_physics',
    name: 'Noumenal Physics',
    category: 'transcendent',
    complexity: 16,
    prerequisiteIds: ['universal_constructor'],
    description: 'Physics of the thing-in-itself beyond phenomena',
    flavorText: 'Beyond what can be measured. The physics of ultimate reality.',
    aiBonus: {
      curiosity: 2.0,
      dimensionalAccess: 2.5,
      informationProcessing: 2.8,
      spaceManipulation: 2.8,
      timePerception: 2.8,
      entropicResistance: 2.0,
      cooperationBonus: 1.80,
      learningRate: 2.8,
      inventionChance: 0.18,
      energyEfficiency: 0.005
    }
  }
];


// Track procedurally generated concepts for unlimited evolution
let generatedPhysicsLevel = 0;

/**
 * All available physics categories for procedural generation
 */
const ALL_CATEGORIES: PhysicsConcept['category'][] = [
  'mechanics', 'thermodynamics', 'electromagnetism', 'quantum', 'relativity',
  'unified', 'cosmology', 'particle', 'plasma', 'condensed_matter', 'exotic',
  'transcendent', 'nuclear', 'astrophysics', 'biophysics', 'information_physics',
  'chaos_complexity'
];

/**
 * Category prefixes for procedurally generated concepts
 */
const CATEGORY_PREFIXES: Record<string, string[]> = {
  'mechanics': ['Hypermechanics', 'Trans-Mechanical', 'Meta-Kinetic', 'Ultra-Dynamic'],
  'thermodynamics': ['Entropic Mastery', 'Thermal Transcendence', 'Heat-Death Engineering', 'Negentropy'],
  'electromagnetism': ['Field Manipulation', 'Photonic Mastery', 'EM Transcendence', 'Light-Bending'],
  'quantum': ['Quantum Singularity', 'Wave-Function Control', 'Superposition Mastery', 'Decoherence Engineering'],
  'relativity': ['Spacetime Mastery', 'Relativistic Control', 'Gravity Shaping', 'Metric Engineering'],
  'unified': ['Grand Unification', 'Force Synthesis', 'Theory of Everything', 'Ultimate Law'],
  'cosmology': ['Cosmic Engineering', 'Universe Shaping', 'Multiverse Navigation', 'Creation Physics'],
  'particle': ['Subatomic Control', 'Particle Mastery', 'Quark Engineering', 'Boson Manipulation'],
  'plasma': ['Plasma Dominion', 'Stellar Control', 'Fusion Mastery', 'Ionized Engineering'],
  'condensed_matter': ['Matter Mastery', 'Phase Control', 'Solid Engineering', 'Material Transcendence'],
  'exotic': ['Exotic Manipulation', 'Warp Engineering', 'Negative Energy Control', 'Causality Bending'],
  'transcendent': ['Transcendent Physics', 'Reality Mastery', 'Existence Engineering', 'Omega Control'],
  'nuclear': ['Nuclear Mastery', 'Transmutation Control', 'Binding Energy Engineering', 'Isotope Manipulation'],
  'astrophysics': ['Stellar Engineering', 'Cosmic Manipulation', 'Galaxy Shaping', 'Cosmic Forge'],
  'biophysics': ['Life Engineering', 'Bio-Transcendence', 'Consciousness Control', 'Evolution Mastery'],
  'information_physics': ['Information Mastery', 'Computational Reality', 'Bit-State Control', 'Data Transcendence'],
  'chaos_complexity': ['Complexity Mastery', 'Emergence Control', 'Chaos Engineering', 'Pattern Transcendence']
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
  const prefixes = CATEGORY_PREFIXES[category] || ['Advanced Physics'];
  const prefixIndex = Math.floor(level / 5) % prefixes.length;
  return prefixes[prefixIndex];
}

/**
 * Generate advanced physics concepts procedurally
 * Enables UNLIMITED scientific progression - NO CAPS
 */
export function generateAdvancedPhysicsConcept(level: number): PhysicsConcept {
  const categoryIndex = level % ALL_CATEGORIES.length;
  const category = ALL_CATEGORIES[categoryIndex];
  
  // Complexity scales infinitely but meaningfully
  const complexity = 16 + Math.floor(level / ALL_CATEGORIES.length) + 1;
  
  generatedPhysicsLevel = Math.max(generatedPhysicsLevel, level);
  
  // Bonuses grow without limit but with diminishing returns
  const tierMultiplier = 1 + (level / ALL_CATEGORIES.length) * 0.1;
  const baseEfficiency = Math.max(0.001, 0.01 / tierMultiplier);
  const baseMultiplier = 2.5 * tierMultiplier;
  const baseCuriosity = 1.5 + (level * 0.02);
  
  // Generate flavor text based on category and level
  const flavorTexts: Record<string, string> = {
    'mechanics': `The ${complexity}th level of motion understanding. Forces become suggestions.`,
    'thermodynamics': `Entropy bows to will at this level. Heat flows where commanded.`,
    'electromagnetism': `Light is clay to be sculpted. Fields obey thought.`,
    'quantum': `Superposition is a choice. Wave functions collapse when desired.`,
    'relativity': `Spacetime is a medium to be shaped. Gravity is a tool.`,
    'unified': `All forces are one. The single truth behind appearance.`,
    'cosmology': `Universes can be designed. Creation is an engineering problem.`,
    'particle': `Matter is a configuration option. Particles are adjustable parameters.`,
    'plasma': `Stars are controllable. Fusion is routine.`,
    'condensed_matter': `Any material property can be specified. Matter obeys.`,
    'exotic': `Negative energy, wormholes, time loops - all practical technologies.`,
    'transcendent': `Beyond physics as understood. Reality itself becomes malleable.`,
    'nuclear': `Nuclei are building blocks. Transmutation is trivial.`,
    'astrophysics': `Galaxies are gardens. Black holes are power plants.`,
    'biophysics': `Life is code to be written. Evolution is a design parameter.`,
    'information_physics': `Reality is computed. The source code is accessible.`,
    'chaos_complexity': `Emergence is directed. Complexity is cultivated.`
  };
  
  return {
    id: `advanced_physics_${level}`,
    name: `${getCategoryPrefix(category, level)} ${getRomanNumeral((level % 50) + 1)}`,
    category,
    complexity,
    prerequisiteIds: level > 0 ? [`advanced_physics_${level - 1}`] : ['noumenal_physics'],
    description: `${category.charAt(0).toUpperCase() + category.slice(1).replace(/_/g, ' ')} transcending known limits - Tier ${Math.floor(level / ALL_CATEGORIES.length) + 1}`,
    flavorText: flavorTexts[category] || 'Physics beyond current comprehension.',
    aiBonus: {
      energyEfficiency: baseEfficiency,
      learningRate: baseMultiplier,
      movementSpeed: baseMultiplier * 0.9,
      inventionChance: Math.min(0.25, 0.15 + (level * 0.005)),
      spaceManipulation: baseMultiplier * 0.95,
      timePerception: baseMultiplier * 0.85,
      curiosity: baseCuriosity,
      cooperationBonus: 1.5 + (level * 0.015),
      dimensionalAccess: baseMultiplier * 0.8,
      informationProcessing: baseMultiplier,
      entropicResistance: 1.8 + (level * 0.02)
    }
  };
}

/**
 * Calculate total physics bonuses for an agent
 * NO CAPS - bonuses can grow infinitely
 */
export function calculatePhysicsBonuses(unlockedPhysics: PhysicsConcept[]): {
  energyEfficiency: number;
  movementSpeed: number;
  learningRate: number;
  inventionChance: number;
  spaceManipulation: number;
  timePerception: number;
  curiosity: number;
  cooperationBonus: number;
  dimensionalAccess: number;
  informationProcessing: number;
  entropicResistance: number;
} {
  let energyMult = 1.0;
  let movementMult = 1.0;
  let learningMult = 1.0;
  let inventionBonus = 0.0;
  let spaceMult = 1.0;
  let timeMult = 1.0;
  let curiosityMult = 1.0;
  let cooperationMult = 1.0;
  let dimensionMult = 1.0;
  let infoMult = 1.0;
  let entropicMult = 1.0;
  
  for (const concept of unlockedPhysics) {
    if (concept.aiBonus.energyEfficiency !== undefined) {
      energyMult *= concept.aiBonus.energyEfficiency;
    }
    if (concept.aiBonus.movementSpeed !== undefined) {
      movementMult *= concept.aiBonus.movementSpeed;
    }
    if (concept.aiBonus.learningRate !== undefined) {
      learningMult *= concept.aiBonus.learningRate;
    }
    if (concept.aiBonus.inventionChance !== undefined) {
      inventionBonus += concept.aiBonus.inventionChance;
    }
    if (concept.aiBonus.spaceManipulation !== undefined) {
      spaceMult *= concept.aiBonus.spaceManipulation;
    }
    if (concept.aiBonus.timePerception !== undefined) {
      timeMult *= concept.aiBonus.timePerception;
    }
    if (concept.aiBonus.curiosity !== undefined) {
      curiosityMult *= concept.aiBonus.curiosity;
    }
    if (concept.aiBonus.cooperationBonus !== undefined) {
      cooperationMult *= concept.aiBonus.cooperationBonus;
    }
    if (concept.aiBonus.dimensionalAccess !== undefined) {
      dimensionMult *= concept.aiBonus.dimensionalAccess;
    }
    if (concept.aiBonus.informationProcessing !== undefined) {
      infoMult *= concept.aiBonus.informationProcessing;
    }
    if (concept.aiBonus.entropicResistance !== undefined) {
      entropicMult *= concept.aiBonus.entropicResistance;
    }
  }
  
  // NO CAPS - unlimited evolution!
  return {
    energyEfficiency: energyMult,
    movementSpeed: movementMult,
    learningRate: learningMult,
    inventionChance: inventionBonus,
    spaceManipulation: spaceMult,
    timePerception: timeMult,
    curiosity: curiosityMult,
    cooperationBonus: cooperationMult,
    dimensionalAccess: dimensionMult,
    informationProcessing: infoMult,
    entropicResistance: entropicMult
  };
}

/**
 * Get concepts by category for UI/analysis
 */
export function getConceptsByCategory(category: PhysicsConcept['category']): PhysicsConcept[] {
  return PHYSICS_CONCEPTS.filter(c => c.category === category);
}

/**
 * Get all concepts that are discoverable given current knowledge
 */
export function getDiscoverableConcepts(
  discoveredIds: Set<string>, 
  allConcepts: PhysicsConcept[] = PHYSICS_CONCEPTS
): PhysicsConcept[] {
  return allConcepts.filter(concept => {
    if (discoveredIds.has(concept.id)) return false;
    return concept.prerequisiteIds.every(prereq => discoveredIds.has(prereq));
  });
}

/**
 * Calculate discovery probability based on agent stats and concept complexity
 */
export function calculateDiscoveryProbability(
  concept: PhysicsConcept,
  agentCuriosity: number,
  agentLearningRate: number,
  baseChance: number = 0.001
): number {
  const complexityFactor = Math.max(0.1, 1 - (concept.complexity / 30));
  const curiosityBoost = Math.sqrt(agentCuriosity);
  const learningBoost = Math.log2(agentLearningRate + 1);
  
  return baseChance * complexityFactor * curiosityBoost * learningBoost;
}

/**
 * Get prerequisite chain for a concept (for visualization)
 */
export function getPrerequisiteChain(
  conceptId: string,
  allConcepts: PhysicsConcept[] = PHYSICS_CONCEPTS
): PhysicsConcept[] {
  const concept = allConcepts.find(c => c.id === conceptId);
  if (!concept) return [];
  
  const chain: PhysicsConcept[] = [];
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
 * Get all available physics concepts including procedurally generated ones
 */
export function getAllPhysicsConcepts(generatedLevels: number = 0): PhysicsConcept[] {
  const baseConcepts = [...PHYSICS_CONCEPTS];
  for (let i = 0; i < generatedLevels; i++) {
    baseConcepts.push(generateAdvancedPhysicsConcept(i));
  }
  return baseConcepts;
}

/**
 * Get current generated physics level for saving state
 */
export function getGeneratedPhysicsLevel(): number {
  return generatedPhysicsLevel;
}

/**
 * Set generated physics level when loading state
 */
export function setGeneratedPhysicsLevel(level: number): void {
  generatedPhysicsLevel = level;
}

/**
 * Get statistics about the physics system
 */
export function getPhysicsStats(): {
  totalBaseConcepts: number;
  categoryCounts: Record<string, number>;
  complexityRange: { min: number; max: number };
  averageComplexity: number;
} {
  const categoryCounts: Record<string, number> = {};
  let minComplexity = Infinity;
  let maxComplexity = -Infinity;
  let totalComplexity = 0;
  
  for (const concept of PHYSICS_CONCEPTS) {
    categoryCounts[concept.category] = (categoryCounts[concept.category] || 0) + 1;
    minComplexity = Math.min(minComplexity, concept.complexity);
    maxComplexity = Math.max(maxComplexity, concept.complexity);
    totalComplexity += concept.complexity;
  }
  
  return {
    totalBaseConcepts: PHYSICS_CONCEPTS.length,
    categoryCounts,
    complexityRange: { min: minComplexity, max: maxComplexity },
    averageComplexity: totalComplexity / PHYSICS_CONCEPTS.length
  };
}

/**
 * Validate physics concept prerequisites (useful for debugging)
 */
export function validatePrerequisites(): { valid: boolean; errors: string[] } {
  const errors: string[] = [];
  const conceptIds = new Set(PHYSICS_CONCEPTS.map(c => c.id));
  
  for (const concept of PHYSICS_CONCEPTS) {
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
    
    const concept = PHYSICS_CONCEPTS.find(c => c.id === conceptId);
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
  for (const concept of PHYSICS_CONCEPTS) {
    hasCycle(concept.id, visited, new Set());
  }
  
  return { valid: errors.length === 0, errors };
}