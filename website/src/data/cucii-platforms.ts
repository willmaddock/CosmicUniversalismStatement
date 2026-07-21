export type CuciiPlatformTestingStatus = 'Most tested' | 'Project-author tested' | 'Additional platform';

export interface CuciiPlatformLink {
  id: string;
  name: string;
  url: string;
  testingStatus: CuciiPlatformTestingStatus;
  description: string;
  availabilityNote?: string;
}

export const cuciiPlatforms = [
  { id: 'chatgpt', name: 'ChatGPT', url: 'https://chatgpt.com/', testingStatus: 'Most tested', description: 'General-purpose conversational AI.' },
  { id: 'grok', name: 'Grok', url: 'https://grok.com/', testingStatus: 'Most tested', description: 'General-purpose conversational AI.' },
  { id: 'claude', name: 'Claude', url: 'https://claude.ai/', testingStatus: 'Project-author tested', description: 'General-purpose conversational AI.' },
  { id: 'gemini', name: 'Google Gemini', url: 'https://gemini.google.com/', testingStatus: 'Project-author tested', description: 'General-purpose conversational AI.' },
  { id: 'deepseek', name: 'DeepSeek', url: 'https://chat.deepseek.com/', testingStatus: 'Project-author tested', description: 'General-purpose conversational AI.' },
  { id: 'copilot', name: 'Microsoft Copilot', url: 'https://copilot.microsoft.com/', testingStatus: 'Additional platform', description: 'General-purpose conversational AI.' },
  { id: 'perplexity', name: 'Perplexity', url: 'https://www.perplexity.ai/', testingStatus: 'Additional platform', description: 'Conversational search and research.' },
  { id: 'meta-ai', name: 'Meta AI', url: 'https://www.meta.ai/', testingStatus: 'Additional platform', description: 'General-purpose conversational AI.' },
  { id: 'doubao', name: 'Doubao', url: 'https://www.doubao.com/chat/', testingStatus: 'Additional platform', description: 'General-purpose conversational AI.', availabilityNote: 'Availability and language support may vary by region.' },
  { id: 'quark', name: 'Quark', url: 'https://www.quark.cn/', testingStatus: 'Additional platform', description: 'General-purpose AI and search experience.', availabilityNote: 'Availability and language support may vary by region.' },
] as const satisfies readonly CuciiPlatformLink[];
