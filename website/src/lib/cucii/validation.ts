import type { CuciiMenuItem, CuciiPromptConfig, ValidationResult } from './types';

const clean = (value: string | undefined) => (value ?? '').replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F\u007F]/g, '').trim();

export function normalizeCuciiConfig(config: CuciiPromptConfig): CuciiPromptConfig {
  return {
    ...config,
    promptDepth: clean(config.promptDepth),
    purposePreset: clean(config.purposePreset),
    customPurpose: clean(config.customPurpose),
    customPersona: clean(config.customPersona),
    userContext: clean(config.userContext),
    constraints: clean(config.constraints),
    principles: config.principles.map(clean).filter(Boolean),
    responseStyle: config.responseStyle.map(clean).filter(Boolean),
    outputFormat: clean(config.outputFormat),
    sourcePack: config.sourcePack ?? 'complete',
    affirmationProtocol: config.affirmationProtocol ?? false,
    menuPreset: clean(config.menuPreset),
    menuItems: config.menuItems.map((item) => ({
      ...item,
      label: clean(item.label),
      description: clean(item.description),
      requestedAction: clean(item.requestedAction),
      outputFormat: clean(item.outputFormat),
    })),
  };
}

export function validateCuciiConfig(input: CuciiPromptConfig): ValidationResult<CuciiPromptConfig> {
  const config = normalizeCuciiConfig(input);
  const errors = [] as Array<{ code: string; field: string; message: string }>;
  if (!['chatgpt', 'grok', 'neutral', 'custom'].includes(config.platform)) errors.push({ code: 'PLATFORM_REQUIRED', field: 'platform', message: 'Choose a platform target.' });
  if (config.sourcePack && config.sourcePack !== 'complete') errors.push({ code: 'SOURCE_PACK_REQUIRED', field: 'sourcePack', message: 'Choose the Complete CU Repository Sources pack.' });
  if (!['quick', 'standard', 'full-research'].includes(config.promptDepth)) errors.push({ code: 'PROMPT_DEPTH_REQUIRED', field: 'promptDepth', message: 'Choose a prompt depth.' });
  if (!config.purposePreset) errors.push({ code: 'PURPOSE_REQUIRED', field: 'purposePreset', message: 'Choose a purpose.' });
  if (config.purposePreset === 'custom' && (config.customPurpose ?? '').length < 3) errors.push({ code: 'CUSTOM_PURPOSE_REQUIRED', field: 'customPurpose', message: 'Enter at least three characters for a custom purpose.' });
  if (config.mode === 'role-play' && config.persona === 'none') errors.push({ code: 'PERSONA_REQUIRED', field: 'persona', message: 'Choose a persona for role-play mode.' });
  if (config.mode !== 'analytical' && config.persona === 'custom' && (config.customPersona ?? '').length < 3) errors.push({ code: 'CUSTOM_PERSONA_REQUIRED', field: 'customPersona', message: 'Enter a custom persona or choose another persona.' });
  if (config.menuMode === 'preset' && !config.menuPreset) errors.push({ code: 'MENU_PRESET_REQUIRED', field: 'menuPreset', message: 'Choose a preset menu.' });
  if (config.menuMode === 'custom') {
    if (config.menuItems.length < 1) errors.push({ code: 'MENU_COUNT', field: 'menuItems', message: 'A custom menu must contain at least one menu choice.' });
    if (config.menuItems.length > 12) errors.push({ code: 'MENU_COUNT', field: 'menuItems', message: 'Maximum of 12 menu choices reached.' });
    const labels = new Set<string>();
    config.menuItems.forEach((item, index) => {
      const field = `menuItems.${index}`;
      if (!item.label || item.label.length > 60) errors.push({ code: 'MENU_LABEL', field, message: 'Each menu label is required and must be 60 characters or fewer.' });
      if (!item.description || item.description.length > 180) errors.push({ code: 'MENU_DESCRIPTION', field, message: 'Each menu description is required and must be 180 characters or fewer.' });
      const key = item.label.toLowerCase();
      if (key && labels.has(key)) errors.push({ code: 'MENU_DUPLICATE', field, message: 'Menu labels must be unique.' });
      if (key) labels.add(key);
      if ((item.requestedAction ?? '').length > 240) errors.push({ code: 'MENU_ACTION', field, message: 'Requested actions must be 240 characters or fewer.' });
      if ((item.outputFormat ?? '').length > 120) errors.push({ code: 'MENU_OUTPUT', field, message: 'Output formats must be 120 characters or fewer.' });
    });
  }
  return errors.length ? { ok: false, errors } : { ok: true, value: config };
}

export function menuItem(label = '', description = ''): CuciiMenuItem {
  return { id: `menu-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`, label, description };
}
