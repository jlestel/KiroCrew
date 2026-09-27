/**
 * Screenshot harness for TurnBlock's fold toggle in the MAIN chat (#9699).
 *
 * The dashboard's ChatPage is the host whose turn toggle the issue contrasts
 * with the embed's group header. These frames show the ChatPage host rendering
 * the same pill as the app-sdk host (see capture-tool-group-affordance.mjs),
 * collapsed and then expanded, in dark and light.
 *
 * Runs the REAL built SPA (website/dist) behind the shared transcript harness
 * with every /api/** call answered from fixtures — no gateway, no token, no
 * agent — so the virtualizer, the turn grouper and the row wrappers behave
 * exactly as in production.
 *
 * Usage: node scripts/capture-turn-fold-pill.mjs [outDir]
 */
import { mkdirSync } from 'node:fs'

import { openTranscriptHarness } from './lib/transcript-harness.mjs'

const OUT = process.argv[2] || '../temp-screenshots/turn-fold-pill'
const SLOT = 'chat-turn-fold-pill'
const PROJECT = '/home/user/workspace/KiroCrew'

mkdirSync(OUT, { recursive: true })

const TOGGLE = '[data-testid="tool-group-toggle"]'
const TOGGLE_WAIT = { selector: TOGGLE, settle: 900 }

const t0 = Date.now() / 1000 - 900

const slots = [
  {
    key: SLOT,
    title: 'Which two files import the transcript row keys?',
    running: false,
    last_message: 'Two files import it.',
    messages: 4,
    agent: 'kirocrew',
    memory_mode: 'persistent',
    project: PROJECT,
    modified: Math.floor(Date.now() / 1000),
    source_links: [],
    source_links_total: 0,
  },
]

/** A settled turn: prompt, two tool calls (distinct tool_call_ids), the answer. */
const detail = {
  running: false,
  has_more: false,
  total: 4,
  queue: [],
  project: PROJECT,
  messages: [
    { role: 'user', ts: t0, content: 'Which two files import the transcript row keys?' },
    {
      role: 'tool',
      ts: t0 + 6,
      content: '🔧 grep',
      cls: '',
      meta: { tool_call_id: 'tc_grep', purpose: 'Search for rowKeys imports', input: '{"pattern":"transcript/rowKeys"}', output: '2 matches' },
    },
    {
      role: 'tool',
      ts: t0 + 12,
      content: '🔧 fs_read',
      cls: '',
      meta: { tool_call_id: 'tc_read', purpose: 'Read the TurnBlock import block', input: '{"path":"website/src/pages/chat/TurnBlock.tsx"}', output: 'ok' },
    },
    {
      role: 'assistant',
      ts: t0 + 20,
      content: 'Two files: `TurnBlock.tsx` and `ChatMessageList.tsx`. Both take `uniqueRowKeys` from `chat-core/transcript/rowKeys`.',
    },
  ],
}

async function main() {
  const { page, load, close } = await openTranscriptHarness({
    slot: SLOT,
    project: PROJECT,
    slots,
    detail,
    viewport: { width: 1280, height: 720 },
  })

  let failed = 0
  const probe = () => page.locator(TOGGLE).first().evaluate(b => ({
    className: b.className,
    text: b.textContent,
    expanded: b.getAttribute('aria-expanded'),
    label: b.getAttribute('aria-label'),
    glyphRotated: !!b.querySelector('.rotate-90'),
  }))

  async function shot(name) {
    // A first-run gate (Privacy, Customize) would sit on top of every frame while
    // the DOM probes still pass behind it, so a frame with a dialog open is a FAIL.
    const dialogs = await page.locator('[role="dialog"]').count()
    if (dialogs) { console.error(`FAIL: ${dialogs} dialog(s) open before ${name}`); failed++ }
    // The pill itself must be inside the viewport, or the frame shows a
    // transcript with no affordance in it.
    const box = await page.locator(TOGGLE).first().boundingBox()
    const vp = page.viewportSize()
    if (!box || box.y < 0 || box.y + box.height > vp.height) { console.error(`FAIL: toggle out of view before ${name}`, box); failed++ }
    // Park the cursor off-canvas so the pill is not photographed in its hover colour.
    await page.mouse.move(0, 0)
    await page.waitForTimeout(200)
    await page.screenshot({ path: `${OUT}/${name}.png` })
    console.log('wrote', `${OUT}/${name}.png`)
  }

  for (const theme of ['dark', 'light']) {
    await load(theme, TOGGLE_WAIT)
    // Pin the locale to English and turn "collapse all steps" OFF so the turn
    // folds its TOOL CALLS (the issue's subject) rather than every working
    // step; in default mode the same pill reads "Worked through 2 steps".
    // Each `load` registers an init script that CLEARS localStorage, and init
    // scripts run in registration order, so this one is registered AFTER every
    // load — once per theme — to land after that theme's clear on the reload.
    await page.addInitScript(() => {
      localStorage.setItem('mc-lang', 'en')
      localStorage.setItem('mc-chat-config', JSON.stringify({ collapseAllSteps: false }))
    })
    await page.reload({ waitUntil: 'domcontentloaded' })
    await page.waitForSelector(TOGGLE, { timeout: 20000 })
    await page.waitForTimeout(900)

    const count = await page.locator(TOGGLE).count()
    console.log(`[${theme}] toggles:`, count)
    if (count !== 1) { console.error(`FAIL [${theme}]: expected 1 toggle, saw ${count}`); failed++ }

    const collapsed = await probe()
    console.log(`[${theme}] collapsed:`, JSON.stringify(collapsed))
    if (!collapsed.text.includes('2 tool calls')) { console.error(`FAIL [${theme}]: label is not "2 tool calls"`); failed++ }
    if (collapsed.expanded !== 'false' || collapsed.glyphRotated) { console.error(`FAIL [${theme}]: not in the collapsed state`); failed++ }
    if (!/\bbg-card\b/.test(collapsed.className) || !/\bring-1\b/.test(collapsed.className)) {
      console.error(`FAIL [${theme}]: toggle is not the pill (missing bg-card / ring-1)`); failed++
    }
    await shot(`${theme}-collapsed`)

    // The transcript is virtualized and rows are absolutely positioned, so
    // dispatch the click on the node itself rather than through hit-testing.
    await page.locator(TOGGLE).first().evaluate(el => { el.scrollIntoView({ block: 'center' }); el.click() })
    await page.waitForTimeout(700)

    const expanded = await probe()
    console.log(`[${theme}] expanded:`, JSON.stringify(expanded))
    if (expanded.expanded !== 'true' || !expanded.glyphRotated) { console.error(`FAIL [${theme}]: not in the expanded state`); failed++ }
    if (expanded.label !== 'Collapse 2 tool calls') { console.error(`FAIL [${theme}]: accessible name is ${expanded.label}`); failed++ }
    // The fold must reveal its two tool rows (ToolCallLine pills are buttons).
    const rows = await page.locator('[data-testid="tool-call-line"], [data-display-index] button:not([data-testid="tool-group-toggle"])').count()
    console.log(`[${theme}] revealed rows:`, rows)
    if (rows < 2) { console.error(`FAIL [${theme}]: expanded fold shows ${rows} rows, expected >= 2`); failed++ }
    await shot(`${theme}-expanded`)
  }

  await close()
  if (failed) {
    console.error(`${failed} check(s) failed`)
    process.exit(1)
  }
  console.log('all checks passed')
}

main().catch(err => {
  console.error(err)
  process.exit(1)
})
