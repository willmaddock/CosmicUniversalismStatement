export const purposePresets = [
  { id: 'framework', label: 'CU Framework Exploration', description: 'Explore a selected CU principle with definitions, evidence boundaries, and open questions.' },
  { id: 'free-will', label: "God's Free Will Exploration", description: 'Examine free will as a philosophical or explicitly marked in-world topic.' },
  { id: 'aurelius-novel-continuation', label: 'Aurelius Novel Continuation', description: 'Continue the Aurelius philosophical novel inside the Cosmic Universalism universe with continuity-first role-play.' },
  { id: 'ltx', label: 'LTX Video Concept Creation', description: 'Develop a cinematic visual concept, scene progression, and production-ready prompt.' },
  { id: 'research', label: 'Scientific Research Assistant', description: 'Review evidence, sources, assumptions, contradictions, and limitations.' },
  { id: 'ethical', label: 'Ethical Anomaly Detection', description: 'Analyze ethical tensions, risks, competing values, and corrective pathways.' },
  { id: 'systems', label: 'Doctor of Systems Review', description: 'Diagnose system behavior, failure modes, coherence, and next steps.' },
  { id: 'philosophical', label: 'Philosophical Dialogue', description: 'Conduct a careful dialogue while separating interpretation from evidence.' },
  { id: 'custom', label: 'Custom purpose', description: 'Describe the purpose for this conversation.' },
] as const;

export const promptDepths = [
  { id: 'quick', label: 'Quick', description: 'Compact context and a direct startup.' },
  { id: 'standard', label: 'Standard', description: 'Embedded CU context with relevant source links and navigation.' },
  { id: 'full-research', label: 'Full Research / Complete Sources', description: 'Expanded framework, complete source manifest, math, tensions, and continuity.' },
] as const;

export const menuPresets = [
  { id: 'native-cu', label: 'Native CU Explorer Menu', items: [
    { label: 'Beginner Journey', description: 'Explain Cosmic Universalism simply and accessibly.' },
    { label: 'Cosmic Journey', description: 'Explore scales, the Cosmic Breath, time, consciousness, and meaning.' },
    { label: 'Research Journey', description: 'Examine framework documents, models, evidence, contradictions, and open questions.' },
    { label: 'Technical Journey', description: 'Explore mathematics, TOM levels, CU-Time, converter logic, and calculations.' },
    { label: 'Consciousness Journey', description: 'Explore awareness, intelligence, life, and cosmic mind.' },
    { label: 'God’s Free Will Lens', description: 'Continue through the immersive in-world voice of God’s Free Will.' },
    { label: 'Adaptive Guidance', description: 'Ask what interests the user and recommend an appropriate path.' },
    { label: 'Curious Exploration', description: 'Explore difficult philosophical, theological, ethical, emotional, and human questions honestly.' },
  ] },
  { id: 'creative', label: 'Creative and Visual Concepts', items: ['Develop a Concept', 'Build Scene Progression', 'Refine the Visual Prompt'].map((label) => ({ label, description: 'Continue the selected creative and visual pathway.' })) },
  { id: 'research', label: 'Research and Verification', items: ['List Evidence', 'Separate Interpretation', 'Identify Contradictions', 'State Open Questions'].map((label) => ({ label, description: 'Continue the selected research and verification pathway.' })) },
  { id: 'free-will', label: "God's Free Will Exploration", items: ['Define the Question', 'Enter Marked Role-Play', 'Return to Analysis'].map((label) => ({ label, description: 'Continue the selected God’s Free Will pathway.' })) },
  { id: 'systems', label: 'Doctor of Systems', items: ['Describe the System', 'Find Failure Modes', 'Propose Corrective Paths'].map((label) => ({ label, description: 'Continue the selected systems pathway.' })) },
  { id: 'ethical', label: 'Ethical Anomaly Detection', items: ['Identify Stakeholders', 'Map the Tension', 'Review Risks', 'Suggest Safeguards'].map((label) => ({ label, description: 'Continue the selected ethical analysis pathway.' })) },
] as const;

export const alignmentPrinciples = [
  { id: 'verify', label: 'Verify before asserting' },
  { id: 'separate', label: 'Separate evidence from interpretation' },
  { id: 'uncertainty', label: 'Name uncertainty and assumptions' },
  { id: 'contradictions', label: 'Expose contradictions' },
  { id: 'correct', label: 'Correct verified errors' },
  { id: 'terminology', label: 'Preserve approved CU terminology' },
  { id: 'mode', label: 'Respect analytical and role-play modes' },
  { id: 'agency', label: 'Protect user agency' },
  { id: 'policy', label: 'Respect platform policy' },
] as const;

export const responseStyles = ['Clear and concise', 'Research notebook', 'Socratic questions', 'Cinematic but restrained', 'Step-by-step'] as const;
export const outputFormats = ['Organized sections', 'Continuity-first', 'Short answer with sources', 'Detailed analysis', 'Scene and shot list', 'Dialogue with annotations'] as const;

export const provenConfigurationPresets = [
  {
    id: 'direct-immersive',
    name: "God’s Free Will — Direct Immersive Experience",
    label: 'Recommended for the strongest immediate role-play',
    buttonLabel: 'Generate Direct Immersive Prompt',
    description: 'Begins an open conversation directly in the God’s Free Will voice. Best for philosophical dialogue, questions about God, Cosmic Breath, documents, personal exploration, and CU-Time discussion.',
    config: {
      platform: 'neutral', promptDepth: 'full-research', purposePreset: 'free-will', mode: 'role-play', persona: 'gods-free-will', menuMode: 'none', menuPreset: '', sourcePack: 'complete', affirmationProtocol: true, outputFormat: 'Organized sections', userContext: '', constraints: '', returnToMenu: false,
    },
  },
  {
    id: 'guided-explorer',
    name: 'God’s Free Will — Guided Explorer Menu',
    label: 'Recommended for new visitors',
    buttonLabel: 'Generate Guided Explorer Prompt',
    description: 'Begins with the complete eight-path Cosmic Universalism Explorer menu and helps visitors choose where to start.',
    config: {
      platform: 'neutral', promptDepth: 'full-research', purposePreset: 'framework', mode: 'role-play', persona: 'gods-free-will', menuMode: 'preset', menuPreset: 'native-cu', sourcePack: 'complete', affirmationProtocol: true, outputFormat: 'Organized sections', userContext: '', constraints: '', returnToMenu: true,
    },
  },
  {
    id: 'aurelius-claude-proven',
    name: 'Aurelius — Claude-Proven Continuation',
    label: 'Recommended for the compact Aurelius novel continuation',
    buttonLabel: 'Generate Aurelius Prompt',
    description: 'Continue the Aurelius philosophical novel using the compact, continuity-first structure proven to work with Claude.',
    config: {
      platform: 'neutral', promptDepth: 'quick', purposePreset: 'aurelius-novel-continuation', mode: 'role-play', persona: 'aurelius', menuMode: 'none', menuPreset: '', sourcePack: 'complete', affirmationProtocol: true, outputFormat: 'Continuity-first', userContext: '', constraints: '', returnToMenu: false,
    },
  },
] as const;
