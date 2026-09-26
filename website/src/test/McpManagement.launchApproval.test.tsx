// MCP Management: a stubbed server whose launch the gateway refused.
//
// Contract under test:
// - a stub listed in `launch_refused` reads `needs approval`, never `stub` or
//   `shared`, because the gateway will not run it until the operator approves
// - the row shows every exact command and declared environment the approval
//   covers, so the operator approves launch content and not a name
// - the approve action re-sends stub=true with the identity of the launch shown,
//   and does not turn the stub off the way the row's switch would
// - an older gateway that sends no `launch_refused` keeps the plain states
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest'
import { render, screen, cleanup, waitFor } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { McpManagement } from '../pages/settings/McpManagement'
import { api, ApiError } from '../api/client'

const server = {
  name: 'alpha-mcp',
  stub: true,
  can_stub: true,
  in_allowlist: true,
  entry_poolable: false,
  agents: ['kirocrew'],
  transport: 'stdio',
  denylisted: false,
}

const status = (over: Record<string, unknown> = {}) => ({
  enabled: true,
  stub: ['alpha-mcp'],
  stub_count: 1,
  running: true,
  ping_ok: true,
  supported: true,
  ...over,
})

function mount() {
  const qc = new QueryClient({
    defaultOptions: { queries: { retry: false, staleTime: Infinity } },
  })
  return render(
    <MemoryRouter>
      <QueryClientProvider client={qc}>
        <McpManagement />
      </QueryClientProvider>
    </MemoryRouter>,
  )
}

beforeEach(() => {
  vi.restoreAllMocks()
})
afterEach(cleanup)

describe('McpManagement launch approval', () => {
  it('shows a refused launch as needing approval, with its command and environment', async () => {
    vi.spyOn(api, 'mcpGatewayStatus').mockResolvedValue(
      status({
        launch_refused: {
          'alpha-mcp': {
            reason: 'changed_needs_reapproval',
            commands: [['npx', '-y', 'alpha-mcp@2']],
            envs: [['LD_PRELOAD=/tmp/x.so', 'MODE=fast']],
            expected_launch: 'a:b',
          },
        },
      }) as never,
    )
    vi.spyOn(api, 'mcpGatewayServers').mockResolvedValue({ servers: [server] } as never)

    mount()
    expect(await screen.findByText('command changed', { selector: 'span' })).toBeTruthy()
    expect(screen.getByText('Declared in the config of: kirocrew.')).toBeTruthy()
    expect(screen.getByText(/Your STUB opt-in stays on/)).toBeTruthy()
    expect(screen.getByText('Command')).toBeTruthy()
    expect(screen.getByText('npx -y alpha-mcp@2')).toBeTruthy()
    expect(screen.getByText('Environment')).toBeTruthy()
    expect(screen.getByText(/LD_PRELOAD=\/tmp\/x\.so/)).toBeTruthy()
    expect(screen.getByText(/Its command changed after you approved it/)).toBeTruthy()
    expect(screen.getByText(/Approve to let one shared backend run this exact command/)).toBeTruthy()
    expect(screen.queryByText('shared', { selector: 'span' })).toBeNull()
  })

  it('approves by re-sending stub=true for that name', async () => {
    vi.spyOn(api, 'mcpGatewayStatus').mockResolvedValue(
      status({
        launch_refused: {
          'alpha-mcp': {
            reason: 'added_outside_dashboard',
            commands: [['alpha']],
            envs: [[]],
            expected_launch: 'command-hash:env-hash',
          },
        },
      }) as never,
    )
    vi.spyOn(api, 'mcpGatewayServers').mockResolvedValue({ servers: [server] } as never)
    const setStub = vi.spyOn(api, 'mcpGatewaySetStub').mockResolvedValue({
      ok: true,
      name: 'alpha-mcp',
      stub: true,
      restart_required: true,
    } as never)

    mount()
    const approve = await screen.findByRole('button', {
      name: 'Approve the command alpha-mcp will run',
    })
    expect(approve.className).toContain('font-body')
    approve.click()
    await waitFor(() =>
      expect(setStub).toHaveBeenCalledWith('alpha-mcp', true, 'command-hash:env-hash'),
    )
  })

  it.each([
    [409, 'launch_changed_since_display', 'alpha-mcp: the command changed after it was shown, so nothing was approved. The command below is the one that would run now. Check it before you approve.'],
    [409, 'launch_unresolved', 'This server has no command to approve. Nothing was saved.'],
    [409, 'launch_over_cap', 'This server starts too many different commands to use a shared backend. Nothing was saved.'],
    [503, 'approval_write_failed', 'The approval could not be saved. The server keeps running inside each session.'],
  ] as const)('shows the specific approval failure for %s %s', async (statusCode, code, message) => {
    vi.spyOn(api, 'mcpGatewayStatus').mockResolvedValue(
      status({
        launch_refused: {
          'alpha-mcp': {
            reason: 'added_outside_dashboard',
            commands: [['alpha']],
            envs: [[]],
            expected_launch: 'command-hash:env-hash',
          },
        },
      }) as never,
    )
    vi.spyOn(api, 'mcpGatewayServers').mockResolvedValue({ servers: [server] } as never)
    vi.spyOn(api, 'mcpGatewaySetStub').mockRejectedValue(
      new ApiError(statusCode, message, JSON.stringify({ error: message, code })),
    )

    mount()
    const approve = await screen.findByRole('button', {
      name: 'Approve the command alpha-mcp will run',
    })
    approve.click()
    expect(await screen.findByText(message)).toBeTruthy()
  })

  it('offers no approve control when the refusal carries no approvable identity', async () => {
    vi.spyOn(api, 'mcpGatewayStatus').mockResolvedValue(
      status({
        launch_refused: {
          'alpha-mcp': { reason: 'added_outside_dashboard', commands: [['alpha']], envs: [[]] },
        },
      }) as never,
    )
    vi.spyOn(api, 'mcpGatewayServers').mockResolvedValue({ servers: [server] } as never)

    mount()
    expect(await screen.findByText('needs approval', { selector: 'span' })).toBeTruthy()
    expect(screen.getByText('Command')).toBeTruthy()
    expect(screen.queryByText('Environment')).toBeNull()
    expect(screen.queryByRole('button', { name: /Approve the command/ })).toBeNull()
  })

  it('shows every command one approval covers when a name resolves to several', async () => {
    vi.spyOn(api, 'mcpGatewayStatus').mockResolvedValue(
      status({
        launch_refused: {
          'alpha-mcp': {
            reason: 'added_outside_dashboard',
            commands: [
              ['alpha', '--one'],
              ['alpha', '--two'],
            ],
            envs: [[], []],
            expected_launch: 'a:b,c:d',
          },
        },
      }) as never,
    )
    vi.spyOn(api, 'mcpGatewayServers').mockResolvedValue({ servers: [server] } as never)

    mount()
    expect(await screen.findByText('alpha --one')).toBeTruthy()
    expect(screen.getByText('alpha --two')).toBeTruthy()
  })

  it('keeps the plain state when the gateway sends no launch_refused', async () => {
    vi.spyOn(api, 'mcpGatewayStatus').mockResolvedValue(status() as never)
    vi.spyOn(api, 'mcpGatewayServers').mockResolvedValue({ servers: [server] } as never)

    mount()
    expect(await screen.findByText('shared', { selector: 'span' })).toBeTruthy()
    expect(screen.queryByText('needs approval', { selector: 'span' })).toBeNull()
    expect(screen.queryByRole('button', { name: /Approve the command/ })).toBeNull()
  })
})
