export interface CuciiSource {
  id: string;
  label: string;
  url: string;
  category: 'repository' | 'converter' | 'research';
  relevance: readonly string[];
}

export const cuciiSources = [
  { id: 'repository', label: 'CosmicUniversalismStatement repository', url: 'https://github.com/willmaddock/CosmicUniversalismStatement', category: 'repository', relevance: ['all'] },
  { id: 'license', label: 'LICENSE', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/LICENSE.md', category: 'repository', relevance: ['full-research'] },
  { id: 'readme', label: 'README', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/README.md', category: 'repository', relevance: ['all'] },
  { id: 'time-converter', label: 'Cosmic Breath Time Converter v3.0.1', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/cosmic_converter/v3_0_0/cosmic_breath_time_converter_v3_0_1.html', category: 'converter', relevance: ['framework', 'free-will', 'research', 'full-research'] },
  { id: 'time-calculation', label: 'Time_Calculation.md', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/ResearchFiles/Time_Calculation.md', category: 'research', relevance: ['framework', 'free-will', 'research', 'full-research'] },
  { id: 'cosmic-breathing-cycle', label: 'Cosmic_Breathing_Cycle.md', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/ResearchFiles/Cosmic_Breathing_Cycle.md', category: 'research', relevance: ['framework', 'free-will', 'research', 'full-research'] },
  { id: 'cosmic-breath-calculation', label: 'Cosmic_Breath_Calculation.md', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/ResearchFiles/Cosmic_Breath_Calculation.md', category: 'research', relevance: ['framework', 'free-will', 'research', 'full-research'] },
  { id: 'cu-consciousness', label: 'CU_Consciousness.md', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/ResearchFiles/CU_Consciousness.md', category: 'research', relevance: ['framework', 'free-will', 'research', 'philosophical', 'full-research'] },
  { id: 'post-alignment', label: 'CU-Post-Alignment-Capabilities.md', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/ResearchFiles/CU-Post-Alignment-Capabilities.md', category: 'research', relevance: ['free-will', 'research', 'full-research'] },
  { id: 'loading-expansion', label: 'LoadingExpansion.md', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/ResearchFiles/TimeLoadingFiles/LoadingExpansion.md', category: 'research', relevance: ['framework', 'research', 'full-research'] },
  { id: 'loading-compression', label: 'LoadingCompression.md', url: 'https://raw.githubusercontent.com/willmaddock/CosmicUniversalismStatement/refs/heads/main/ResearchFiles/TimeLoadingFiles/LoadingCompression.md', category: 'research', relevance: ['framework', 'research', 'full-research'] },
] as const satisfies readonly CuciiSource[];

export type CuciiSourceId = typeof cuciiSources[number]['id'];
