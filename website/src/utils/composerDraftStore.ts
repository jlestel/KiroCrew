import { safeSetSessionItem } from './safeStorage'

/**
 * Where a comment composer's typed-but-unsent text lives between teardowns the
 * toolbar cannot guard (a chat-slot switch replaces the whole side panel; a
 * page navigation unmounts the artifact page). One store per host document
 * (`key`), one slot per passage inside it (`anchor` + `start`), so drafts on two
 * passages of one file coexist and neither can overwrite or clear the other.
 *
 * Two layers: `sessionStorage` (survives a reload of this tab) and an in-memory
 * snapshot per key (survives a refusing storage — quota, private mode — and is
 * what a fresh instance over the same key in the SAME tab reads). The snapshot
 * also carries TOMBSTONES: a slot this tab cleared stays cleared even when the
 * storage write that recorded the deletion was refused, so a stale storage
 * record can never resurrect a draft of a comment that was already posted. A
 * later write to the same slot lifts its tombstone.
 *
 * A slot can be HELD while the host has that draft's post in flight: a held
 * slot reads as empty and refuses writes, so a second composer instance over
 * the same passage (the side panel's full-screen layer, opened mid-post) does
 * not restore the pending text and re-post it. `release` ends the hold; the
 * flight's own settle then clears the slot (success) or leaves it (refusal).
 *
 * Every operation is non-throwing: a full or refusing storage is a mundane
 * condition, not a crash.
 */
export interface ComposerDraftStore {
  read: (anchor: string, start: number) => string | null
  write: (text: string, anchor: string, start: number) => void
  clear: (anchor: string, start: number) => void
  /** Mark the passage's post as in flight (see module doc). */
  hold: (anchor: string, start: number) => void
  release: (anchor: string, start: number) => void
  /** Whether a post for this passage is in flight (on any instance over this key). */
  isHeld: (anchor: string, start: number) => boolean
}

type Slots = Record<string, string>
interface Snapshot { slots: Slots; gone: Set<string> }

const composerDraftMemory = new Map<string, Snapshot>()
/** `${key}\n${slotKey}` of every slot whose post is in flight, tab-wide. */
const heldSlots = new Set<string>()

const slotKey = (anchor: string, start: number) => `${start}|${anchor}`

function parse(raw: string | null): Slots {
  if (!raw) return {}
  try {
    const parsed = JSON.parse(raw) as unknown
    if (!parsed || typeof parsed !== 'object') return {}
    const out: Slots = {}
    for (const [k, v] of Object.entries(parsed as Record<string, unknown>)) if (typeof v === 'string') out[k] = v
    return out
  } catch { return {} }
}

export function composerDraftStoreFor(key: string): ComposerDraftStore {
  // Storage seeds what this tab has not yet written; the snapshot's slots win
  // over it and its tombstones remove from it.
  const load = (): Slots => {
    let fromSession: Slots = {}
    try { fromSession = parse(window.sessionStorage.getItem(key)) } catch { /* unavailable */ }
    const mem = composerDraftMemory.get(key)
    if (!mem) return fromSession
    const out: Slots = {}
    for (const [k, v] of Object.entries(fromSession)) if (!mem.gone.has(k)) out[k] = v
    return { ...out, ...mem.slots }
  }
  const save = (slots: Slots, gone: Set<string>) => {
    composerDraftMemory.set(key, { slots, gone })
    if (Object.keys(slots).length === 0) {
      try { window.sessionStorage.removeItem(key) } catch { /* unavailable */ }
      return
    }
    safeSetSessionItem(key, JSON.stringify(slots))
  }
  const gone = () => new Set(composerDraftMemory.get(key)?.gone ?? [])
  const heldKey = (anchor: string, start: number) => `${key}\n${slotKey(anchor, start)}`
  return {
    read: (anchor, start) => heldSlots.has(heldKey(anchor, start)) ? null : load()[slotKey(anchor, start)] ?? null,
    write: (text, anchor, start) => {
      if (heldSlots.has(heldKey(anchor, start))) return
      const slots = load(); const k = slotKey(anchor, start)
      slots[k] = text
      const g = gone(); g.delete(k)
      save(slots, g)
    },
    clear: (anchor, start) => {
      const slots = load(); const k = slotKey(anchor, start)
      delete slots[k]
      const g = gone(); g.add(k)
      save(slots, g)
    },
    hold: (anchor, start) => { heldSlots.add(heldKey(anchor, start)) },
    release: (anchor, start) => { heldSlots.delete(heldKey(anchor, start)) },
    isHeld: (anchor, start) => heldSlots.has(heldKey(anchor, start)),
  }
}
