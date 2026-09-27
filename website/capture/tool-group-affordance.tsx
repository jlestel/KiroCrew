/**
 * Both collapsed-tool-group affordances in the app-sdk host, one above the
 * other (#9699).
 *
 * The app-sdk `ChatMessageList` is the one host that renders BOTH surfaces:
 * a settled turn folds its tool rows behind `TurnBlock`'s toggle, and a run of
 * reasoning rows outside a turn renders as a `CollapsibleToolGroup`. The frame
 * is the evidence that the two are the same pill — one glyph, one wrench, one
 * ring — photographed in one viewport so a reader compares them without
 * flipping between files.
 *
 * WHY IT MOUNTS ChatMessageList RATHER THAN THE TWO COMPONENTS. Each affordance
 * then reaches the frame through the real grouping (a turn only forms with >2
 * items and a working step; a reasoning run only groups outside one), the real
 * renderer registry (`createTranscriptRenderers`, so an expanded group shows
 * ThinkingBlock rows, not an empty well) and the real host row wrapper — the
 * geometry a member DM or embed reader actually sees. And, as in
 * transcript-row-style.tsx, this file hand-writes NO Tailwind classes:
 * `capture/` is outside the Tailwind content glob, so a class authored here is
 * never compiled and would make the frame unfalsifiable.
 *
 *   ?theme=dark|light
 */
import { createRoot } from 'react-dom/client'
import { combineReducers, configureStore } from '@reduxjs/toolkit'
import { Provider } from 'react-redux'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'

import { initI18n } from '../src/i18n'
import dashboardReducer from '../src/store/dashboardSlice'
import notificationsReducer from '../src/store/notificationsSlice'
import chatReducer from '../src/store/chatSlice'
import instancesReducer from '../src/store/instancesSlice'
import { store as realStore } from '../src/store'
import ChatMessageList from '../src/app-sdk/ChatMessageList'
import { createTranscriptRenderers } from '../src/pages/chat/transcriptRenderers'
import type { ChatMessage } from '../src/types'
import '../src/index.css'

const params = new URLSearchParams(location.search)
const theme = params.get('theme') || 'dark'

document.documentElement.setAttribute('data-theme', theme === 'light' ? 'kiro-light' : 'kiro-dark')

// MarkdownRenderer probes path-like inline code and unfurls links; neither
// endpoint exists here and a pending probe leaves a chip mid-load.
const realFetch = globalThis.fetch.bind(globalThis)
globalThis.fetch = ((input: RequestInfo | URL, init?: RequestInit) => {
  const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
  if (url.startsWith('/api/file-read')) {
    return Promise.resolve(new Response(null, { status: 200, headers: { 'X-Path-Kind': 'file' } }))
  }
  if (url.startsWith('/api/link-meta')) return Promise.resolve(Response.json({}))
  return realFetch(input as RequestInfo, init)
}) as typeof fetch

const SLOT = 'main'

const rootReducer = combineReducers({
  dashboard: dashboardReducer,
  notifications: notificationsReducer,
  chat: chatReducer,
  instances: instancesReducer,
})
const base = realStore.getState()
const store = configureStore({
  reducer: rootReducer,
  preloadedState: { ...base, chat: { ...base.chat, activeSlot: SLOT } },
})

let seq = 0
/** Distinct ts per row: ChatMessageList keys rows off it. */
const msg = (role: string, content: string, over: Partial<ChatMessage> = {}): ChatMessage => ({
  role,
  content,
  cls: '',
  ts: `2026-09-27T00:00:${String(seq++).padStart(2, '0')}.000Z`,
  ...over,
})

/**
 * Host surface 1 — a settled turn: user prompt, two tool calls, the answer.
 * Four items with working steps, so ChatMessageList wraps it as a turn and
 * TurnBlock folds the two 🔧 rows behind its toggle: "2 tool calls".
 */
const TURN_MESSAGES: ChatMessage[] = [
  msg('user', 'Which two files import the transcript row keys?'),
  msg('tool', '🔧 grep', {
    meta: { tool_call_id: 't1', purpose: 'Search for rowKeys imports', input: '{"pattern":"transcript/rowKeys"}', output: '2 matches' },
  }),
  msg('tool', '🔧 fs_read', {
    meta: { tool_call_id: 't2', purpose: 'Read the TurnBlock import block', input: '{"path":"website/src/pages/chat/TurnBlock.tsx"}', output: 'ok' },
  }),
  msg('assistant', 'Two files: `TurnBlock.tsx` and `ChatMessageList.tsx`. Both take `uniqueRowKeys` from `chat-core/transcript/rowKeys`.'),
]

/**
 * Host surface 2 — a reasoning run with no turn around it: user prompt, then
 * two thinking rows and nothing else. One grouped item never forms a turn, so
 * ChatMessageList renders the group directly as a CollapsibleToolGroup —
 * the same "2 tool calls" pill a member DM shows over a folded group.
 */
const GROUP_MESSAGES: ChatMessage[] = [
  msg('user', 'Why do the two hosts look different?'),
  msg('thinking', 'The turn toggle and the group header are two components with two styles for one concept.'),
  msg('thinking', 'Rendering both through one pill removes the split without changing what either folds.'),
]

const renderers = createTranscriptRenderers({
  slot: SLOT,
  onFileOpen: () => {},
  onFolderOpen: () => {},
  onOpenSubagentPanel: () => {},
  onToolDisclosureChange: () => {},
  toolDisclosure: {},
  appInPanel: false,
  onOpenApp: () => {},
})

const qc = new QueryClient({ defaultOptions: { queries: { retry: false } } })

initI18n('en')

createRoot(document.getElementById('root')!).render(
  <MemoryRouter>
    <QueryClientProvider client={qc}>
      <Provider store={store}>
        <div
          data-capture-root
          className="bg-bg text-text"
          style={{ width: 760, ['--mc-content-width' as string]: '700px' }}
        >
          <div className="py-4" data-capture-surface="turn">
            <ChatMessageList messages={TURN_MESSAGES} running={false} contentWidth="700px" renderers={renderers} />
          </div>
          <div className="py-4" data-capture-surface="group">
            <ChatMessageList messages={GROUP_MESSAGES} running={false} contentWidth="700px" renderers={renderers} />
          </div>
        </div>
      </Provider>
    </QueryClientProvider>
  </MemoryRouter>,
)
