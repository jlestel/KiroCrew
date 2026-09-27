import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor, fireEvent } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import type { ReactNode } from 'react'

/**
 * The crewmate's Perpetual mode switch on its detail page.
 *
 * What is pinned: the two layers under the switch -- the facts that decide the
 * press in body copy and as the switch's own description, the rest muted below
 * them; the switch's position is the registry READ, never the press
 * (a refused press leaves it where the backend is); pending disables rather
 * than flips; each verdict renders its plain-words reason, including the
 * crewmate's own stop words; a thread never opened shows the switch disabled
 * with the reason instead of pressing into a 409; and a failed read never
 * renders as "nothing wakes this crewmate".
 */

const H = vi.hoisted(() => ({
  members: vi.fn(),
  autonudgeList: vi.fn(),
  memberPerpetualSet: vi.fn(),
}))

vi.mock('../../api/client', () => ({
  api: {
    members: H.members,
    autonudgeList: H.autonudgeList,
    memberPerpetualSet: H.memberPerpetualSet,
  },
}))

import CrewPerpetualSection from './CrewPerpetualSection'

function wrap(node: ReactNode) {
  const qc = new QueryClient({ defaultOptions: { queries: { retry: false }, mutations: { retry: false } } })
  return render(<QueryClientProvider client={qc}>{node}</QueryClientProvider>)
}

const ROW = { name: 'Radar', slug: 'radar', slot_key: 'member-radar', running: false }
const now = Math.floor(Date.now() / 1000)
const LOOP = {
  id: 'lp1', slot_key: 'member-radar', message: 'Perpetual mode wake.', banner: 'Perpetual mode',
  idle_secs: 3600, max_cycles: 0, max_runtime_secs: 0, cycle_count: 7, active: true,
  created_ts: now - 30_000, last_fire_ts: now - 600, next_due_ts: now + 3000, gate: false,
}

beforeEach(() => {
  H.members.mockReset(); H.autonudgeList.mockReset(); H.memberPerpetualSet.mockReset()
  H.members.mockResolvedValue({ members: [ROW] })
})
afterEach(cleanup)

const switchEl = () => screen.getByTestId('crew-perpetual-switch').querySelector('[role="switch"]') as HTMLElement

describe('CrewPerpetualSection', () => {
  it('reads ON from the registry and shows the wake readouts', async () => {
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [LOOP] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await waitFor(() => expect(switchEl().getAttribute('aria-checked')).toBe('true'))
    expect(screen.getByTestId('crew-perpetual-status').getAttribute('data-state')).toBe('on')
    const interval = screen.getByTestId('crew-perpetual-interval')
    expect(interval.textContent).toMatch(/1\s?h/)
    expect(interval.parentElement?.className).toContain('flex-col')
    expect(interval.parentElement?.className).toContain('sm:flex-row')
    expect(interval.previousElementSibling?.className).toContain('sm:w-24')
    const title = screen.getByRole('heading', { name: /Perpetual mode/i })
    expect(title.className).toContain('min-w-0')
    expect(title.parentElement?.className).toContain('flex-wrap')
    expect(screen.getByTestId('crew-perpetual-cycles').textContent).toMatch(/7/)
    expect(screen.getByTestId('crew-perpetual-next').textContent).toMatch(/Due in/)
  })

  it('states the facts that decide the press in body copy, and only those, as the switch\'s description', async () => {
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [LOOP] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await waitFor(() => expect(switchEl().getAttribute('aria-checked')).toBe('true'))

    // Two layers, not one muted paragraph: a label/value list carries the three
    // facts a reader needs before pressing (when it starts, what it costs, and
    // where to review what it will act on), and the muted lines below carry
    // the rest.
    const facts = screen.getByTestId('crew-perpetual-facts')
    expect(facts.tagName).toBe('DL')
    expect(facts.className).not.toContain('text-muted')
    expect(facts.querySelectorAll('dt')).toHaveLength(3)
    expect(facts.querySelectorAll('dd')).toHaveLength(3)
    // The switch is described by the primary facts ALONE -- the secondary lines
    // are page text in reading order, not a paragraph read out on every focus.
    // The id must resolve to exactly this list: a duplicate id would make the
    // description ambiguous, and an id nothing carries would describe nothing.
    expect(switchEl().getAttribute('aria-describedby')).toBe('crew-perpetual-facts')
    expect(document.querySelectorAll('#crew-perpetual-facts')).toHaveLength(1)
    expect(document.getElementById('crew-perpetual-facts')).toBe(facts)

    const firstWake = screen.getByTestId('crew-perpetual-fact-first-wake')
    // Timing: one interval after the press, stated with the interval this
    // crewmate has saved (1 h here) and hedged, since a wake can run long.
    expect(firstWake.textContent).toMatch(/^About 1\s?h after you turn it on\./)
    // Scope: what that wake actually does, in the reader's own terms.
    expect(firstWake.textContent).toMatch(
      /It picks up the goals and instructions you gave it in its own chat, then ends the turn when nothing is due\./,
    )
    // Cost: one model turn per wake AND what the saved interval works out to,
    // so spend is a number on screen rather than an inference. 3600s → 24/day.
    expect(screen.getByTestId('crew-perpetual-fact-each-wake').textContent).toBe(
      'One model turn per wake, about 24 a day at this interval.',
    )
    // Review: the thing an uncapped loop will act on is the crewmate's own
    // chat, so the third fact is a link to that chat, addressed by the roster
    // name (not the slug) the section was given.
    const reviewChat = screen.getByTestId('crew-perpetual-review-chat')
    expect(reviewChat.tagName).toBe('A')
    expect(reviewChat.textContent).toBe("Review this crewmate's chat")
    expect(reviewChat).toHaveAttribute('href', '/members?member=Radar')
    // Targets a new browsing context and sends no referrer; the attributes are
    // what is checked here, not what a browser does with them.
    expect(reviewChat).toHaveAttribute('target', '_blank')
    expect(reviewChat).toHaveAttribute('rel', 'noreferrer')
    // The link is the THIRD fact's value: inside the third `dd`, which shares
    // its row with the third `dt` ("Before turning on"), so a screen reader
    // pairs the label with the link rather than reading a stray anchor.
    const dts = [...facts.querySelectorAll('dt')]
    const dds = [...facts.querySelectorAll('dd')]
    expect(dds[2].contains(reviewChat)).toBe(true)
    expect(dds[0].contains(reviewChat)).toBe(false)
    expect(dds[1].contains(reviewChat)).toBe(false)
    expect(dds[2].parentElement).toBe(dts[2].parentElement)
    expect(dts[2].nextElementSibling).toBe(dds[2])
    // Each fact wears its own label, so the list reads as label/value to a
    // screen reader instead of three sentences in a row.
    const labels = dts.map((dt) => dt.textContent)
    expect(labels).toEqual(['First wake', 'Each wake', 'Before turning on'])

    // The secondary layer is muted and keeps every fact that does NOT decide
    // the press: no cap on count or duration, what OFF does and does not stop,
    // and that scheduled jobs are separate work either way.
    const limits = screen.getByTestId('crew-perpetual-limits')
    expect(limits.className).toContain('text-muted')
    expect(limits.textContent).toMatch(/Nothing caps how many times it wakes or how long it keeps going\./)
    expect(limits.textContent).toMatch(/Turning it off starts no new wakes, and work already running finishes\./)
    expect(limits.textContent).toMatch(/Scheduled jobs run separately either way\./)
    // The cadence is measured from the END of a wake (never a frequency), the
    // interval is the crewmate's own to retune, and the two saving controls on
    // this screen are told apart.
    const cadence = screen.getByTestId('crew-perpetual-cadence')
    expect(cadence.className).toContain('text-muted')
    expect(cadence.textContent).toMatch(/Each later interval starts when the current wake ends/)
    expect(cadence.textContent).toMatch(/the crewmate can retune that interval from inside a wake/)
    expect(cadence.textContent).not.toMatch(/Save changes/)
    // Neither muted line is inside the list the switch is described by, so
    // neither is read out as part of the switch's description.
    expect(facts.contains(limits)).toBe(false)
    expect(facts.contains(cadence)).toBe(false)
    const all = `${facts.textContent} ${limits.textContent} ${cadence.textContent}`
    expect(all).not.toMatch(/how often/)
    // There is no Perpetual interval editor on this page, so nothing here
    // offers the reader an interval to set or save.
    expect(all).not.toMatch(/\byou (can )?(set|change|save)s? (the |its )?interval\b/i)
  })

  it('states the first-wake timing and the per-day cost from the default interval when no loop exists yet', async () => {
    // Nothing armed: the facts still have to answer "when does it start and
    // what does it cost", so they fall back to the interval an arm begins on
    // (1 h, the backend's own default) rather than going blank or reading 0.
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await screen.findByTestId('crew-perpetual-facts')
    expect(screen.getByTestId('crew-perpetual-fact-first-wake').textContent).toMatch(/^About 1\s?h after you turn it on\./)
    expect(screen.getByTestId('crew-perpetual-fact-each-wake').textContent).toMatch(/about 24 a day/)
  })

  it('states the per-day cost from the interval the crewmate retuned itself', async () => {
    // The crewmate owns the cadence (`monitor_update`), so a retuned interval
    // must move the estimate with it: 15 minutes is 96 wakes a day.
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [{ ...LOOP, idle_secs: 900 }] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await screen.findByTestId('crew-perpetual-facts')
    expect(screen.getByTestId('crew-perpetual-fact-each-wake').textContent).toMatch(/about 96 a day/)
    expect(screen.getByTestId('crew-perpetual-fact-first-wake').textContent).toMatch(/^About 15\s?m/)
  })

  it('withholds the explainer when the switch is not offered, so nothing describes a press that cannot happen', async () => {
    // A gateway with no nudge service: the block says the mode is off for this
    // install, and the facts about turning it on are not rendered at all --
    // including as the switch's description, which would dangle.
    H.autonudgeList.mockResolvedValue({ enabled: false, loops: [] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await screen.findByTestId('crew-perpetual-reason')
    expect(screen.queryByTestId('crew-perpetual-facts')).toBeNull()
    expect(screen.queryByTestId('crew-perpetual-limits')).toBeNull()
    expect(screen.queryByTestId('crew-perpetual-cadence')).toBeNull()
  })

  it('reads OFF with the coded reason and the crewmate\'s own words', async () => {
    H.autonudgeList.mockResolvedValue({
      enabled: true,
      loops: [{ ...LOOP, active: false, stopped_reason: 'autonudge_stop', stopped_detail: 'standing duty is over' }],
    })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await waitFor(() => expect(switchEl().getAttribute('aria-checked')).toBe('false'))
    expect(screen.getByTestId('crew-perpetual-status').getAttribute('data-state')).toBe('off')
    expect(screen.getByTestId('crew-perpetual-reason').textContent).toMatch(/Stopped by the crewmate itself/)
    expect(screen.getByTestId('crew-perpetual-detail').textContent).toBe('standing duty is over')
  })

  it('reads OFF as "turned off by you" for the owner\'s manual pause', async () => {
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [{ ...LOOP, active: false, stopped_reason: 'manual' }] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    expect((await screen.findByTestId('crew-perpetual-reason')).textContent).toMatch(/Turned off by you/)
    expect(screen.queryByTestId('crew-perpetual-detail')).toBeNull()
  })

  it('reads "nothing wakes it" when no loop was ever armed, switch OFF', async () => {
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    expect((await screen.findByTestId('crew-perpetual-reason')).textContent).toMatch(/never been turned on/)
    expect(switchEl().getAttribute('aria-checked')).toBe('false')
    expect(switchEl().getAttribute('aria-disabled')).toBeNull()
  })

  it('a press asks the server, disables while pending, then re-reads the registry', async () => {
    let answer: (v: unknown) => void = () => {}
    H.memberPerpetualSet.mockImplementation(() => new Promise((res) => { answer = res }))
    H.autonudgeList.mockResolvedValueOnce({ enabled: true, loops: [] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await screen.findByTestId('crew-perpetual-reason')
    fireEvent.click(switchEl())
    // The mutation runs its function on the next tick, so the call is awaited.
    await waitFor(() => expect(H.memberPerpetualSet).toHaveBeenCalledWith('radar', 'Radar', true))
    // Pending: disabled and announced, NOT flipped -- the switch holds no truth of its own.
    await waitFor(() => expect(screen.getByTestId('crew-perpetual-control').getAttribute('aria-busy')).toBe('true'))
    expect(switchEl().getAttribute('aria-checked')).toBe('false')
    expect(switchEl().getAttribute('aria-disabled')).toBe('true')
    expect(screen.getByText(/^Saving/)).toBeTruthy()
    // The answer lands; the registry now holds the armed loop and the switch flips from THAT read.
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [LOOP] })
    answer({ ok: true, loop: LOOP })
    await waitFor(() => expect(switchEl().getAttribute('aria-checked')).toBe('true'))
    expect(screen.getByTestId('crew-perpetual-control').getAttribute('aria-busy')).toBeNull()
    // Success is STATED where "Saving…" was, briefly: the press happens beside a
    // Save footer that stays disabled, so the card changing is not the only cue.
    expect(screen.getByTestId('crew-perpetual-save-state').textContent).toBe('Saved')
    await waitFor(() => expect(screen.queryByTestId('crew-perpetual-save-state')).toBeNull(), { timeout: 4000 })
  })

  it('a refused press shows the plain-words reason and the switch stays where the backend is', async () => {
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [] })
    const err = Object.assign(new Error('this member is running a structured monitor; stop it from its own thread'), {
      status: 409,
      body: JSON.stringify({ error: 'x', code: 'structured_monitor_not_convertible' }),
    })
    H.memberPerpetualSet.mockRejectedValue(err)
    wrap(<CrewPerpetualSection crew="Radar" />)
    await screen.findByTestId('crew-perpetual-reason')
    fireEvent.click(switchEl())
    expect((await screen.findByTestId('crew-perpetual-error')).textContent).toMatch(/watch task it set up in its chat.*stop the task there first/)
    expect(switchEl().getAttribute('aria-checked')).toBe('false')
    expect(switchEl().getAttribute('aria-disabled')).toBeNull()
    // No hand-off by default: the host's schedules pane may hold a draft this
    // section cannot see, and the hand-off would unmount it.
    expect(screen.queryByText('Ask the agent')).toBeNull()
  })

  it('offers the same "Ask the agent" hand-off as the side panel when the host says nothing is at stake', async () => {
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [] })
    H.memberPerpetualSet.mockRejectedValue(
      Object.assign(new Error('x'), { status: 409, body: JSON.stringify({ error: 'x', code: 'structured_monitor_not_convertible' }) }),
    )
    wrap(<CrewPerpetualSection crew="Radar" askAgent />)
    await screen.findByTestId('crew-perpetual-reason')
    fireEvent.click(switchEl())
    const notice = await screen.findByTestId('crew-perpetual-error')
    expect(notice.querySelector('button')?.textContent).toBe('Ask the agent')
  })

  it('an uncoded refusal shows the server\'s own sentence', async () => {
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [LOOP] })
    H.memberPerpetualSet.mockRejectedValue(Object.assign(new Error('trust record unreadable'), { status: 503, body: '' }))
    wrap(<CrewPerpetualSection crew="Radar" />)
    await waitFor(() => expect(switchEl().getAttribute('aria-checked')).toBe('true'))
    fireEvent.click(switchEl())
    expect((await screen.findByTestId('crew-perpetual-error')).textContent).toMatch(/trust record unreadable/)
    expect(H.memberPerpetualSet).toHaveBeenCalledWith('radar', 'Radar', false)
  })

  it('a thread never opened shows the switch disabled with the reason, and never presses', async () => {
    H.members.mockResolvedValue({ members: [{ ...ROW, slot_key: '' }] })
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await waitFor(() => expect(switchEl().getAttribute('aria-disabled')).toBe('true'))
    expect(screen.getByTestId('crew-perpetual-thread-closed').textContent).toMatch(/runs in this crewmate's chat.*Open its chat once first/)
    expect(switchEl().getAttribute('aria-describedby')).toContain('crew-perpetual-thread-closed')
    fireEvent.click(switchEl())
    await new Promise((r) => setTimeout(r, 20))
    expect(H.memberPerpetualSet).not.toHaveBeenCalled()
  })

  it('a structured monitor on the thread reads none: the roster\'s word wins over the reduced registry row', async () => {
    // `/api/autonudge` publishes a monitor as a reduced row with `active: true`
    // and no cycle accounting; the roster's `perpetual` is computed with
    // `is_structured_monitor_loop` applied, so the switch reads OFF / never
    // turned on and offers nothing the route would refuse with 409.
    H.members.mockResolvedValue({ members: [{ ...ROW, perpetual: 'none' }] })
    H.autonudgeList.mockResolvedValue({
      enabled: true,
      loops: [{ id: 'mon1', slot_key: 'member-radar', active: true, idle_secs: 300, next_due_ts: now + 200, gate: false }],
    })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await waitFor(() => expect(switchEl().getAttribute('aria-checked')).toBe('false'))
    expect(screen.getByTestId('crew-perpetual-status').getAttribute('data-state')).toBe('none')
    // Named for what it is, not "never turned on": the reader sees why the
    // switch is off. State only -- the remedy (open the chat, stop the task)
    // is the refusal's, said once when the switch is pressed, not repeated here.
    const reason = screen.getByTestId('crew-perpetual-reason').textContent ?? ''
    expect(reason).toMatch(/watch task it set up in its chat is running instead.*open its chat and stop the task there first/i)
    expect(screen.queryByTestId('crew-perpetual-next')).toBeNull()
  })

  it('the roster\'s perpetual field decides the state when it disagrees with the record', async () => {
    H.members.mockResolvedValue({ members: [{ ...ROW, perpetual: 'off' }] })
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [LOOP] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await waitFor(() => expect(switchEl().getAttribute('aria-checked')).toBe('false'))
    expect(screen.getByTestId('crew-perpetual-status').getAttribute('data-state')).toBe('off')
    expect(screen.getByTestId('crew-perpetual-reason').textContent).toMatch(/update turned Perpetual mode off.*turn it back on to resume/i)
  })

  it('roster ON with no registry record yet reads ON with the readouts withheld, never "never been turned on"', async () => {
    // The roster and the registry are two reads: right after a press (or
    // across a dropped frame) the roster can already say ON while the registry
    // still holds no record for the thread. The verdict is the roster's word;
    // only the readouts wait for the record.
    H.members.mockResolvedValue({ members: [{ ...ROW, perpetual: 'on' }] })
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    await waitFor(() => expect(switchEl().getAttribute('aria-checked')).toBe('true'))
    expect(screen.getByTestId('crew-perpetual-status').getAttribute('data-state')).toBe('on')
    expect(screen.getByTestId('crew-perpetual-verdict').textContent).toMatch(/^On\./)
    expect(screen.queryByTestId('crew-perpetual-reason')).toBeNull()
    expect(screen.queryByText(/never been turned on/)).toBeNull()
    expect(screen.queryByTestId('crew-perpetual-interval')).toBeNull()
    expect(screen.queryByTestId('crew-perpetual-next')).toBeNull()
  })

  it('withholds the switch and never says "nothing wakes it" when the registry read failed', async () => {
    H.autonudgeList.mockRejectedValue(new Error('boom'))
    wrap(<CrewPerpetualSection crew="Radar" />)
    expect(await screen.findByTestId('crew-perpetual-load-error')).toBeTruthy()
    expect(screen.queryByTestId('crew-perpetual-switch')).toBeNull()
    expect(screen.queryByTestId('crew-perpetual-reason')).toBeNull()
  })

  it('a gateway with no nudge service withholds the switch and says so, never "never been turned on"', async () => {
    H.autonudgeList.mockResolvedValue({ enabled: false, loops: [] })
    wrap(<CrewPerpetualSection crew="Radar" />)
    const reason = await screen.findByTestId('crew-perpetual-reason')
    // Plain product wording, no "gateway"; no invented settings path either.
    expect(reason.textContent).toMatch(/Perpetual mode is turned off on this install/i)
    expect(reason.textContent).not.toMatch(/gateway/i)
    expect(reason.textContent).not.toMatch(/never been turned on/i)
    expect(screen.queryByTestId('crew-perpetual-switch')).toBeNull()
    expect(H.memberPerpetualSet).not.toHaveBeenCalled()
  })

  it("a restart stop (the service's `interrupted_cycle`) reads as the restart sentence, not the raw code", async () => {
    H.members.mockResolvedValue({ members: [{ ...ROW, perpetual: 'off' }] })
    H.autonudgeList.mockResolvedValue({
      enabled: true,
      loops: [{ ...LOOP, active: false, stopped_reason: 'interrupted_cycle', next_due_ts: 0 }],
    })
    wrap(<CrewPerpetualSection crew="Radar" />)
    const reason = await screen.findByTestId('crew-perpetual-reason')
    expect(reason.textContent).toMatch(/restart/i)
    expect(reason.textContent).not.toMatch(/interrupted_cycle/)
  })

  it('a paused record with no reason is not attributed to the owner', async () => {
    H.members.mockResolvedValue({ members: [{ ...ROW, perpetual: 'off' }] })
    H.autonudgeList.mockResolvedValue({
      enabled: true,
      loops: [{ ...LOOP, active: false, stopped_reason: '', next_due_ts: 0 }],
    })
    wrap(<CrewPerpetualSection crew="Radar" />)
    const reason = await screen.findByTestId('crew-perpetual-reason')
    expect(reason.textContent).toMatch(/no reason recorded/i)
    expect(reason.textContent).not.toMatch(/turned off by you/i)
  })

  it('withholds the switch for a crew the roster does not name', async () => {
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [LOOP] })
    wrap(<CrewPerpetualSection crew="Nobody" />)
    await screen.findByTestId('crew-perpetual-status')
    expect(screen.queryByTestId('crew-perpetual-switch')).toBeNull()
    expect(screen.getByTestId('crew-perpetual-reason').textContent).toMatch(/never been turned on/)
  })

  it('only the exact roster name resolves the slot -- a slug twin does not borrow the loop', async () => {
    H.members.mockResolvedValue({ members: [ROW, { name: 'radar', slug: 'radar', slot_key: '', running: false }] })
    H.autonudgeList.mockResolvedValue({ enabled: true, loops: [LOOP] })
    wrap(<CrewPerpetualSection crew="radar" />)
    await waitFor(() => expect(switchEl().getAttribute('aria-disabled')).toBe('true'))
    expect(screen.getByTestId('crew-perpetual-status').getAttribute('data-state')).toBe('none')
  })
})
