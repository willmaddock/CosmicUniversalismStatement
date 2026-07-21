import { describe, expect, it } from 'vitest';
import { cuciiPlatforms } from '../../../data/cucii-platforms';

describe('CUCII platform registry', () => {
  it('contains exactly eleven unique HTTPS platforms', () => {
    expect(cuciiPlatforms).toHaveLength(11);
    expect(new Set(cuciiPlatforms.map((platform) => platform.id)).size).toBe(11);
    cuciiPlatforms.forEach((platform) => expect(platform.url).toMatch(/^https:\/\//));
  });

  it('preserves the tested-platform classifications', () => {
    expect(cuciiPlatforms.filter((platform) => platform.testingStatus === 'Most tested').map((platform) => platform.name)).toEqual(['ChatGPT', 'Grok']);
    expect(cuciiPlatforms.filter((platform) => platform.testingStatus === 'Project-author tested').map((platform) => platform.name)).toEqual(['Claude', 'Google Gemini', 'DeepSeek', 'Alexa+']);
  });

  it('registers Alexa+ with the approved external destination', () => {
    expect(cuciiPlatforms.find((platform) => platform.id === 'alexa-plus')).toMatchObject({ name: 'Alexa+', url: 'https://alexa.com/', testingStatus: 'Project-author tested' });
  });

  it('keeps every link an explicit external destination', () => {
    cuciiPlatforms.forEach((platform) => {
      expect(platform.url.startsWith('https://')).toBe(true);
      expect(platform.url).not.toContain('prompt');
    });
  });
});
