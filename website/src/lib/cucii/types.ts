export type CuciiPlatform = 'chatgpt' | 'grok' | 'neutral' | 'custom';
export type CuciiMode = 'analytical' | 'role-play' | 'hybrid';
export type CuciiMenuMode = 'none' | 'preset' | 'custom';
export type CuciiPersona = 'none' | 'gods-free-will' | 'aurelius' | 'custom';
export type CuciiPromptDepth = 'quick' | 'standard' | 'full-research';
export type CuciiSourcePack = 'complete';

export interface CuciiMenuItem {
  id: string;
  label: string;
  description: string;
  requestedAction?: string;
  outputFormat?: string;
}

export interface CuciiPromptConfig {
  platform: CuciiPlatform;
  promptDepth: CuciiPromptDepth;
  purposePreset: string;
  customPurpose?: string;
  mode: CuciiMode;
  persona: CuciiPersona;
  customPersona?: string;
  menuMode: CuciiMenuMode;
  menuPreset?: string;
  menuItems: CuciiMenuItem[];
  principles: string[];
  responseStyle: string[];
  outputFormat: string;
  sourcePack: CuciiSourcePack;
  affirmationProtocol: boolean;
  userContext?: string;
  constraints?: string;
  returnToMenu: boolean;
}

export interface GeneratedCuciiPrompt {
  text: string;
  filenameStem: string;
  platform: CuciiPlatform;
  version: '1.1';
  generatedAt: string;
  menuItemCount: number;
  sourceIds: string[];
  warnings: string[];
}

export type ValidationError = { code: string; field: string; message: string };
export type ValidationResult<T> =
  | { ok: true; value: T }
  | { ok: false; errors: ValidationError[] };

export interface CuciiConversation {
  id: string;
  title: string;
  platform: 'chatgpt' | 'grok' | 'claude';
  url: string;
  mode: CuciiMode;
  rolePlay: string;
  menuMode: string;
  purpose: string;
  description: string;
  disclosure: string;
  externalLinkLabel: string;
  lastReviewed: string;
}
