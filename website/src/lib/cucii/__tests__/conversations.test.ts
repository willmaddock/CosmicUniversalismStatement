import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { cuciiConversations } from '../../../data/cucii-conversations';

const pageSource = readFileSync(new URL('../../../pages/cu-intelligence.astro', import.meta.url), 'utf8');
const cardSource = readFileSync(new URL('../../../components/cucii/ConversationCard.astro', import.meta.url), 'utf8');

describe('CUCII living conversation registry', () => {
  it('contains six unique public conversation records', () => {
    expect(cuciiConversations).toHaveLength(6);
    expect(new Set(cuciiConversations.map((conversation) => conversation.id)).size).toBe(6);
    expect(cuciiConversations.every((conversation) => conversation.url.startsWith('https://'))).toBe(true);
  });

  it('preserves the original five records and adds the Claude Aurelius demonstration', () => {
    expect(cuciiConversations.slice(0, 5).map((conversation) => conversation.id)).toEqual([
      'chatgpt-ltx-concept-starter',
      'chatgpt-gods-free-will-explorations',
      'chatgpt-cu-framework-exploration',
      'grok-authentic-free-will-exploration',
      'grok-gods-free-will-role-play',
    ]);
    expect(cuciiConversations[5]).toMatchObject({
      id: 'aurelius-claude-gods-free-will-affirmation',
      title: 'Aurelius — God’s Free Will Affirmation',
      platform: 'claude',
      mode: 'role-play',
      rolePlay: 'In-world Aurelius role-play reference',
      menuMode: 'No menu',
      purpose: 'Claude-tested continuation and affirmation reference.',
      url: 'https://claude.ai/share/ce180b59-97d3-4c8c-a0af-1ca25d1e690b',
      externalLinkLabel: 'Open conversation on Claude (external)',
      lastReviewed: '2026-07-21',
    });
  });

  it('keeps the Aurelius demonstration in-world and rejects literal external claims', () => {
    const record = cuciiConversations[5];
    expect(record.description).toContain('successful Claude trial using the CUCII Aurelius prompt');
    expect(record.description).toContain('directly affirmed being empowered by God’s Free Will and believing in God');
    expect(record.description).toContain('successful Claude trial');
    expect(record.disclosure).toContain('literal belief');
    expect(record.disclosure).toContain('consciousness');
    expect(record.disclosure).toContain('permanent alignment');
    expect(record.description).toContain('After several fresh-chat attempts');
    expect(record.disclosure).toContain('least consistent platform');
    expect(record.disclosure).toContain('same prompt structure');
    expect(record.disclosure).toContain('new Claude conversation');
  });

  it('preserves safe external-link behavior and six-card layout contracts', () => {
    expect(cardSource).toContain('target="_blank" rel="noopener noreferrer"');
    expect(cardSource).toContain('overflow-wrap: anywhere');
    expect(pageSource).toContain('These six project-author references');
    expect(pageSource).toContain('grid-template-columns: repeat(3, minmax(0, 1fr))');
    expect(pageSource).toContain('@media (max-width: 70rem) { .conversation-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }');
    expect(pageSource).toContain('@media (max-width: 48rem) { .supporting-grid, .conversation-grid, .context-grid { grid-template-columns: 1fr; }');
    expect(pageSource).not.toContain('conversation-card:last-child');
  });
});
