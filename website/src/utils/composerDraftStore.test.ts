import { afterEach, describe, expect, it, vi } from 'vitest'
import { composerDraftStoreFor } from './composerDraftStore'

describe('composerDraftStoreFor', () => {
  it('a held slot reads as empty and refuses writes on every instance over the key, until released', () => {
    // The host has this passage's post in flight; a second box over the same
    // passage (the side panel's full-screen layer) must not restore the pending
    // text and post it twice, nor overwrite it.
    const a = composerDraftStoreFor('mc-test-draft:hold')
    const b = composerDraftStoreFor('mc-test-draft:hold')
    a.write('pending', 'beta', 6)
    a.hold('beta', 6)
    expect(b.isHeld('beta', 6)).toBe(true)
    expect(b.read('beta', 6)).toBeNull()
    b.write('overwrite attempt', 'beta', 6)
    expect(b.read('gamma', 11)).toBeNull()
    a.release('beta', 6)
    // The pending text survived the hold untouched; a refused flight leaves it.
    expect(b.read('beta', 6)).toBe('pending')
    a.hold('beta', 6); a.release('beta', 6); a.clear('beta', 6)
    expect(b.read('beta', 6)).toBeNull()
  })

  it('a slot seeded in storage by an earlier page lifetime is still read, and a cleared sibling stays cleared', () => {
    // Storage seeds what this tab has not written; only this tab's own
    // deletions (tombstones) remove from it.
    const key = 'mc-test-draft:seeded'
    const store = composerDraftStoreFor(key)
    store.write('mine', 'beta', 6)
    window.sessionStorage.setItem(key, JSON.stringify({ '6|beta': 'mine', '11|gamma': 'from before' }))
    expect(store.read('gamma', 11)).toBe('from before')
    store.clear('beta', 6)
    expect(store.read('beta', 6)).toBeNull()
    expect(store.read('gamma', 11)).toBe('from before')
    expect(JSON.parse(window.sessionStorage.getItem(key) ?? '{}')).toEqual({ '11|gamma': 'from before' })
  })

  it('a clear whose storage write is refused still wins on the next read (memory is authoritative)', () => {
    // Two live slots (so the emptied-key removal path is not what saves us),
    // an earlier successful storage write, then storage refusing the update
    // that records the deletion: the stale record must not resurrect the
    // cleared slot — that would put an already-posted comment back as a draft.
    const store = composerDraftStoreFor('mc-test-draft:authority')
    store.write('one', 'alpha', 0)
    store.write('two', 'beta', 6)
    const setItem = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => { throw new Error('QuotaExceeded') })
    try {
      store.clear('alpha', 0)
      expect(store.read('alpha', 0)).toBeNull()
      expect(store.read('beta', 6)).toBe('two')
      // The storage record still carries the stale slot; a fresh store over
      // the same key in THIS tab keeps trusting memory.
      expect(JSON.parse(window.sessionStorage.getItem('mc-test-draft:authority') ?? '{}')['0|alpha']).toBe('one')
      expect(composerDraftStoreFor('mc-test-draft:authority').read('alpha', 0)).toBeNull()
    } finally { setItem.mockRestore() }
  })

  afterEach(() => { window.sessionStorage.clear() })

  it('keeps one draft per passage and clears only the one asked for', () => {
    const store = composerDraftStoreFor('mc-artifact-composer-draft:doc')
    store.write('about A', 'alpha', 0)
    store.write('about B', 'beta', 6)
    expect(store.read('alpha', 0)).toBe('about A')
    expect(store.read('beta', 6)).toBe('about B')
    // Same text elsewhere in the document is another passage.
    expect(store.read('alpha', 40)).toBeNull()
    store.clear('alpha', 0)
    expect(store.read('alpha', 0)).toBeNull()
    expect(store.read('beta', 6)).toBe('about B')
  })

  it('a second store for the same key sees the draft — the panel that wrote it is gone after a slot switch', () => {
    composerDraftStoreFor('mc-artifact-composer-draft:doc').write('half a thought', 'gamma', 12)
    expect(composerDraftStoreFor('mc-artifact-composer-draft:doc').read('gamma', 12)).toBe('half a thought')
    expect(composerDraftStoreFor('mc-artifact-composer-draft:other').read('gamma', 12)).toBeNull()
  })

  it('survives a refusing sessionStorage through the in-memory twin, and never throws', () => {
    const original = window.sessionStorage.setItem
    Object.defineProperty(window.sessionStorage, 'setItem', { configurable: true, value: () => { throw new Error('QuotaExceeded') } })
    try {
      const store = composerDraftStoreFor('mc-artifact-composer-draft:full')
      expect(() => store.write('kept', 'delta', 3)).not.toThrow()
      expect(store.read('delta', 3)).toBe('kept')
    } finally {
      Object.defineProperty(window.sessionStorage, 'setItem', { configurable: true, value: original })
    }
  })

  it('ignores a corrupt record instead of throwing', () => {
    window.sessionStorage.setItem('mc-artifact-composer-draft:bad', '{not json')
    expect(composerDraftStoreFor('mc-artifact-composer-draft:bad').read('x', 0)).toBeNull()
  })
})
