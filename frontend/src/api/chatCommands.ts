import { create } from 'zustand'
import { persist } from 'zustand/middleware'

export const MODELS = [
    { alias: 'haiku', id: 'claude-haiku-4-5-20251001', description: 'Fastest and cheapest (default)' },
    { alias: 'sonnet', id: 'claude-sonnet-5-5', description: 'Balanced' },
    { alias: 'opus', id: 'claude-opus-5-5', description: 'Strongest reasoning, slower' },
    { alias: 'fable', id: 'claude-fable-5-1', description: 'Latest and most capable' },
] as const

export const DEFAULT_MODEL = MODELS[0].id

interface ChatSettings {
    /** A short reasoning pass before the roadmap and before each tool call. */
    reasoning: boolean
    model: string
    setReasoning: (on: boolean) => void
    setModel: (id: string) => void
}

export const useChatSettings = create<ChatSettings>()(persist((set) => ({
    reasoning: false,
    model: DEFAULT_MODEL,
    setReasoning: (reasoning) => set({ reasoning }),
    setModel: (model) => set({ model }),
}), { name: 'chat-settings' }))

/** Which model chat requests use right now; read at call time so a /model takes effect mid-session. */
export const activeModel = () => useChatSettings.getState().model
export const reasoningOn = () => useChatSettings.getState().reasoning

// commands

export interface CommandOption { insert: string, label: string, description: string }

interface CommandDef {
    name: string
    usage: string
    description: string
    /** Suggested arguments; each `insert` is the full text of the input once chosen. */
    options: () => CommandOption[]
}

const COMMANDS: CommandDef[] = [
    {
        name: 'reasoning', usage: '/reasoning <true|false>', description: 'Think briefly before planning and acting',
        options: () => [ true, false ].map(v => ({
            insert: `/reasoning ${v}`, label: String(v),
            description: useChatSettings.getState().reasoning === v ? 'current' : v ? 'Reason before each step' : 'Answer directly',
        })),
    },
    {
        name: 'model', usage: '/model <model>', description: 'Switch the model chat uses',
        options: () => MODELS.map(m => ({
            insert: `/model ${m.alias}`, label: m.alias,
            description: (useChatSettings.getState().model === m.id ? 'current · ' : '') + m.description,
        })),
    },
]

/** What the autofill popup offers for the text typed so far; empty when it isn't a command. */
export function commandSuggestions(input: string): CommandOption[] {
    if (!input.startsWith('/')) return []
    const [ head, ...rest ] = input.slice(1).split(/\s+/)
    if (rest.length === 0) {
        // still typing the command name
        return COMMANDS.filter(c => c.name.startsWith(head.toLowerCase()))
            .map(c => ({ insert: `/${c.name} `, label: `/${c.name}`, description: c.description }))
    }
    const command = COMMANDS.find(c => c.name === head.toLowerCase())
    if (!command || rest.length > 1) return []
    const typed = rest[0].toLowerCase()
    return command.options().filter(o => o.label.startsWith(typed))
}

export type CommandResult = { ok: true, message: string } | { ok: false, message: string }

/** Runs a `/command` line. Returns null when the line isn't a command at all. */
export function runCommand(line: string): CommandResult | null {
    const text = line.trim()
    if (!text.startsWith('/')) return null
    const [ name, ...args ] = text.slice(1).split(/\s+/)
    const settings = useChatSettings.getState()

    switch (name.toLowerCase()) {
        case 'reasoning': {
            const value = args[0]?.toLowerCase()
            if (value !== 'true' && value !== 'false') return { ok: false, message: `Usage: ${COMMANDS[0].usage}` }
            settings.setReasoning(value === 'true')
            return { ok: true, message: `Reasoning ${value === 'true' ? 'on' : 'off'}` }
        }
        case 'model': {
            const wanted = args[0]?.toLowerCase()
            const model = MODELS.find(m => m.alias === wanted || m.id === wanted)
            if (!model) return { ok: false, message: `Usage: ${COMMANDS[1].usage} — one of ${MODELS.map(m => m.alias).join(', ')}` }
            settings.setModel(model.id)
            return { ok: true, message: `Model set to ${model.id}` }
        }
        default:
            return { ok: false, message: `Unknown command /${name}. Available: ${COMMANDS.map(c => '/' + c.name).join(', ')}` }
    }
}
