import Anthropic from '@anthropic-ai/sdk';
import { anthropic, type ScoreSelection } from "./claudeTools";
import type { ToolUseBlock } from "@anthropic-ai/sdk/resources";
import type { AnyTemplateItem, ChatItem, ChordTemplateItem, Tab, ThoughtProcessEvent, ThoughtProcessTask } from '../types';
import { api } from './client';
import { applyClaudeEdit } from './applyClaudeEdit';
import { fillTemplate, scoreToTemplate } from './templates';
import { useStore } from '../store';
import { getPromptContext, getReadResults, type ReadQuery } from './readContext';

const MAX_TURNS = 10;

export async function promptTemplateAnthropic(chatHistory: ChatItem[], prompt: string, tabs: Tab[], pushCallback: (thought: ThoughtProcessEvent | ThoughtProcessTask) => void, pushChatRename: (title: string) => void) {
    // propogate memory with tab data
    const memory: Record<string, string | AnyTemplateItem[]> = {}
    tabs.forEach(t => {
        if (!t.template && !t.content) return
        memory[t.name] = (t.template ?? t.content)! // TODO: add description  + info likje instruments, bar dur,  etc..
    })
    
    pushCallback({ task: "Generating a procedure" })

    // step 1: roadmap
    const activeName = useStore.getState().activeTab
    const activeIsTemplate = tabs.some(t => t.name === activeName && t.template)
    const context = PROMPT_INTRO + "\n\n" + getHistoryContext(chatHistory) + "\n\n" + getMemoryContext(memory) + "\n\nThe user's current selection:\nTab: " + activeName
        + (activeIsTemplate ? " (a TEMPLATE; removing instruments/voices, editing clefs, tempos, key signatures or any other change to it must use edit_template, never any of the other edits)" : " (a score)") + "\nSelection: " + describeSelection(useStore.getState().selection)
    console.log(context)
    const roadmapResponse = await anthropic.messages.create({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 1024,
        tools: [ withoutScoreTools(ROADMAP_TOOL, activeIsTemplate, memory) ] as unknown as Anthropic.Messages.ToolUnion[],
        system: context,
        messages: [{ role: 'user', content: prompt }],
        tool_choice: { type: "tool", name: "roadmap_tool" }
    });

    const roadmapTool = roadmapResponse.content.find(
        (i): i is ToolUseBlock => i.type === "tool_use" && i.name === "roadmap_tool"
    );
    if (!roadmapTool) {
        console.log('Found no roadmap. Terminating...')
        return []
    }
    
    const chatName = (roadmapTool.input as any)['chat_name'] as string | null
    if (chatName !== null && chatName !== '') {
        pushChatRename(chatName)
    }

    const roadmap = "ROADMAP (complete these steps one by one):\n" + (roadmapTool.input as any).steps.map((s: { tool: string, intention: string }, idx: number) => 
        `# Step ${idx + 1} (tool=${s.tool}):\n${s.intention}`
    ).join('\n\n')

    pushCallback({ title: "Generated a procedure", input: roadmap, tokensIn: roadmapResponse.usage.input_tokens, tokensOut: roadmapResponse.usage.output_tokens, isRoadmap: true })

    const messages: Anthropic.Messages.MessageParam[] = [{ role: 'user', content: prompt }, { role: 'assistant', content: roadmap }];
    let usageIn = roadmapResponse.usage.input_tokens;
    let usageOut = roadmapResponse.usage.output_tokens;

    // Step 2: carry out roadmap
    for (let turn = 0; turn < MAX_TURNS; turn++) {
        console.log('TURN:', turn, 'MESSAGES:', messages, 'USAGE IN:', usageIn, 'USAGE OUT:', usageOut)
        pushCallback({ task: "Querying next steps" })
        const toolSelectionMessage = await anthropic.messages.create({
            model: 'claude-haiku-4-5-20251001',
            max_tokens: 1024,
            tools: withoutScoreTools(TOOL_INTROS, activeIsTemplate, memory) as unknown as Anthropic.Messages.ToolUnion[],
            system: getMemoryContext(memory),
            messages: messages,
            tool_choice: { type: "tool", name: "trigger_tool" }
        });
        usageIn += toolSelectionMessage.usage.input_tokens
        usageOut += toolSelectionMessage.usage.output_tokens

        const toolTriggerContent = (toolSelectionMessage.content.find(
            (i): i is ToolUseBlock => i.type === "tool_use" && i.name === "trigger_tool"
        )?.input as any);
        if (!toolTriggerContent.tool) {
            console.log('Found found tool. Terminating...')
            return []
        }

        const selectedTool = toolTriggerContent.tool
        const intention = toolTriggerContent.intention
        const next_steps = toolTriggerContent.next_steps
        const descriptorIndex = Math.floor(Math.random() * DESCRIPTORS.length)
        const context = getPromptContext(selectedTool, tabs)

        if (selectedTool !== 'complete') {
            pushCallback({ title: (TOOL_DESCRIPTORS[selectedTool] ?? selectedTool).replace('<>',  DESCRIPTORS[descriptorIndex]),
                loading: true, 
                input: (context ? (context + '\n\n') : '') + intention, tokensIn: toolSelectionMessage.usage.input_tokens, tokensOut: toolSelectionMessage.usage.output_tokens })
        }

        // if (selectedTool === 'edit_template') { console.log(memory); return; }
        const toolMessage = await anthropic.messages.create({
            model: 'claude-haiku-4-5-20251001',
            max_tokens: 1024,
            tools: [ TOOLS[selectedTool as keyof typeof TOOLS] ] as unknown as Anthropic.Messages.ToolUnion[],
            system: (context ? (context + '\n\n') : '') + getMemoryContext(memory),
            messages: [{ role: 'user', content: `${prompt}\n\n${roadmap}\n\nDo this now: ${intention}` }],
            tool_choice: { type: "tool", name: selectedTool }
        });
        usageIn += toolMessage.usage.input_tokens
        usageOut += toolMessage.usage.output_tokens

        const toolUses = toolMessage.content.filter((b): b is ToolUseBlock => b.type === "tool_use");
        if (selectedTool === "complete") {
            const recap = (toolUses[0].input as Record<string, unknown>)['recap']
            pushCallback({ title: '', input: recap, tokensIn: toolSelectionMessage.usage.input_tokens + toolMessage.usage.input_tokens,
                tokensOut: toolSelectionMessage.usage.output_tokens + toolMessage.usage.output_tokens, isRecap: true
            }) 
            return recap
        }
        // console.log('Found Tool Uses:', toolUses, "in message:", toolMessage, getMemoryContext(memory))

        const results: Anthropic.Messages.ToolResultBlockParam[] = []
        for (const tool of toolUses) {
            const readResult = await processCommand({ tool: tool.name, input: tool.input as Record<string, unknown> }, memory, intention)
            if (readResult) {
                results.push({
                    type: "tool_result",
                    tool_use_id: tool.id,
                    content: `${tool.name} done:\n${readResult}\nMemory holds: ${Object.keys(memory).join(', ') || 'nothing yet'}`,
                })
            }
            else {
                results.push({
                    type: "tool_result",
                    tool_use_id: tool.id,
                    content: `${tool.name} done. Memory holds: ${Object.keys(memory).join(', ') || 'nothing yet'}`,
                })
            }
        }
        messages.push({ role: 'assistant', content: toolMessage.content });
        if (results.length) messages.push({ role: 'user', content: results });
        messages.push({ role: 'assistant', content: next_steps });

        pushCallback({ title: (TOOL_DESCRIPTORS[selectedTool] ?? selectedTool).replace('ing', 'ed').replace('<>',  DESCRIPTORS_PAST[descriptorIndex]), 
            input: toolUses.length === 1 ? toolUses[0].input as Record<string, any> : toolUses.map(u => u.input as Record<string, any>), 
            tokensIn: toolSelectionMessage.usage.input_tokens,
            tokensOut: toolSelectionMessage.usage.output_tokens })
    }
    console.log('PROMPTING COMPLETE', memory, usageIn, usageOut)
}

const SCORE_ONLY_TOOLS = [ "edit_clef", "edit_time_sig", "edit_key_sig", "edit_tempo", "edit_add_instrument", "edit_delete_staff", "edit_delete_measures" ]

// Drops the score-only tools from every tool-name enum while the active tab is a template and
// memory holds no score they could act on, so they can't be picked at all.
function withoutScoreTools<T>(tools: T, activeIsTemplate: boolean, memory: Record<string, string | AnyTemplateItem[]>): T {
    if (!activeIsTemplate || Object.values(memory).some(v => typeof v === 'string')) return tools
    return JSON.parse(JSON.stringify(tools, (k, v) => k === 'enum' && Array.isArray(v) ? v.filter(x => !SCORE_ONLY_TOOLS.includes(x)) : v))
}

function describeSelection(sel?: ScoreSelection | null): string {
    if (!sel) return "The user has not selected anything on the tab.";
    const parts: string[] = [];
    if (sel.start != null && sel.end != null) {
        parts.push(sel.start === sel.end ? `measure ${sel.start}` : `measures ${sel.start} to ${sel.end} (1-based, inclusive)`);
    }
    parts.push(sel.parts && sel.parts.length > 0
        ? `staff_num ${sel.parts.join(", ")} (0-based)`
        : "all staves");
    return `The user's current selection on the tab: ${parts.join(", ")}.`;
}

function getMemoryContext(memory: Record<string, string | AnyTemplateItem[]>) {
    if (Object.keys(memory).length == 0) return "There are currently no scores or templates saved in memory." 
    return "Current scores and templates saved in memory that may be used:\n" + 
        Object.keys(memory).map(k => `${k} (${typeof memory[k] === "string" ? 'score' : 'template'})`).join(', ')
}

async function processCommand(command: { tool: string, input: Record<string, unknown> }, memory: Record<string, string | AnyTemplateItem[]>, intention: string) {
    switch (command.tool) {
        case "edit_clef":
        case "edit_time_sig":
        case "edit_key_sig":
        case "edit_tempo":
        case "edit_add_instrument":
        case "edit_delete_staff":
        case "edit_delete_measures": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            const stored = memory[in_key]
            if (Array.isArray(stored) && command.tool === "edit_add_instrument") {
                // the target is a template, not a score: add an instrument item for a new voice
                const voices = stored.flatMap(i => 'voice_idx' in i && typeof i.voice_idx === 'number' ? [i.voice_idx] : [])
                const item: AnyTemplateItem = {
                    kind: "instrument", instrument: String(command.input['new_instrument']),
                    voice_idx: voices.length ? Math.max(...voices) + 1 : 0, measure: 1, pos: 0,
                }
                saveToMemory(memory, out_key, [...stored, item])
                break
            }
            const xml = stored
            if (typeof xml !== 'string' || !xml) {
                console.warn(command, "No score stored under", in_key)
                break
            }
            const newXml = await applyClaudeEdit(command.tool, command.input, xml)
            saveToMemory(memory, out_key, newXml)
            break
        }
        case "generate_score": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            if (in_key && in_key in memory) {
                const xml = memory[in_key] as string
                const newXml = await applyClaudeEdit(command.tool, command.input, xml)
                saveToMemory(memory, out_key, newXml)
            }
            else {
                const newXml = await applyClaudeEdit(command.tool, command.input, '')
                saveToMemory(memory, out_key, newXml)
            }
            console.log("FINISHED GENERATE:",memory)
            break
        }
        case "create_template_from_scratch": {
            const { in_key: _, out_key } = getScoreInputOutputNames(command.input)
            const template = command.input['template'] as AnyTemplateItem[]
            saveToMemory(memory, out_key, template)
            break
        }
        case "create_template_from_score": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            const xml = memory[in_key] as string
            const template = await scoreToTemplate(xml)
            saveToMemory(memory, out_key, template)
            break
        }
        case "fill_template": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            const template = memory[in_key] as AnyTemplateItem[]
            if (!template) {
                console.warn(command, "No template stored under", in_key)
                break
            }
            const newXml = await fillTemplate(template)
            saveToMemory(memory, out_key, newXml)

            break
        }
        case "edit_template": {
            const { in_key } = getScoreInputOutputNames(command.input)
            const template = memory[in_key]
            // Without this the stage 2 model is handed "undefined" as its template and, having
            // nothing to edit, invents a fresh one-item template instead.
            if (!Array.isArray(template)) {
                console.warn(command, "No template stored under", in_key, "— nothing to edit. Memory holds:", Object.keys(memory))
                break
            }

            const messageStage2 = await anthropic.messages.create({
                model: 'claude-haiku-4-5-20251001',
                // The whole edited template comes back in this response, so the budget has to fit
                // every item of the input template, not just the ones being changed.
                max_tokens: 16384,
                tools: [ EDIT_TEMPLATE_STAGE2_TOOL ] as unknown as Anthropic.Messages.ToolUnion[],
                system: EDIT_TEMPLATE_STAGE2_INTRO,
                // A single self-contained request: the roadmap conversation ends on an assistant
                // turn and never carries this step's intention, so stage 2 gets its own turn.
                messages: [{
                    role: 'user',
                    content: `Template "${in_key}" — ${template.length} items:\n${JSON.stringify(template)}\n\n`
                        + `Edit this template so that: ${intention}\n\n`
                        + `Return the complete edited template: all ${template.length} items in their original order, with your edits applied.`
                }],
                tool_choice: { type: "tool", name: 'edit_template' }
            });
            console.log('EDIT TEMPLATE STAGE 2:', messageStage2)
            if (messageStage2.stop_reason === 'max_tokens') {
                console.warn('Stage 2 hit the output limit — the edited template is truncated:', messageStage2)
            }
            const toolUses = messageStage2.content.filter((b): b is ToolUseBlock => b.type === "tool_use");
            if (toolUses.length == 0) { console.warn('NO EDITS FOUND:', template, messageStage2); return }
            const input = toolUses[0].input as Record<string, unknown>
            const { out_key } = getScoreInputOutputNames(input)
            const newTemplate = input['template'] as AnyTemplateItem[]
            if (!Array.isArray(newTemplate) || newTemplate.length === 0) {
                console.warn('Stage 2 returned no template — keeping', in_key, 'as it was:', input)
                break
            }
            saveToMemory(memory, out_key || `${in_key}_edited`, keepInstruments(template, newTemplate))
            break
        }
        case "write_score": {
            // a list of writes, pasted one after another in the order given
            const writes = (command.input['writes'] ?? [ command.input ]) as Record<string, unknown>[]
            for (const write of writes) {
                const { in_key, out_key } = getScoreInputOutputNames(write)
                const source = memory[in_key] as string
                if (!source) {
                    console.warn(command, "No score stored under", in_key)
                    continue
                }
                const newXml = (await api.editWriteScore({
                    xml: (memory[out_key] as string) ?? '',
                    source_xml: source,
                    measure_start_num: write['write_start_measure'] as number,
                })).xml

                saveToMemory(memory, out_key, newXml)
            }

            const open = (command.input['open'] ?? '') as string
            console.log(command.input, open)
            if (open !== '') {
                console.log(useStore.getState().tabs)
                useStore.getState().setActiveTab(open);
            }
            break
        }
        case "read_score": {
            const reads = (command.input['queries'] ?? [ ]) as Record<string, any>[]
            const readOutput = []
            for (const read of reads) {
                const { in_key, out_key: _ } = getScoreInputOutputNames(read)
                let source = memory[in_key] as string | null
                const state = useStore.getState()
                if (source) {
                    const result = read['element'] === 'chord' ? describeChords(source as unknown as AnyTemplateItem[])
                        : getReadResults([ read['element'] as ReadQuery ], source!, in_key, state.selection)
                    readOutput.push("# Read for tab: " + in_key + "\n" + result)
                }
                else {
                    source = memory[state.activeTab] as string | null
                    const result = read['element'] === 'chord' ? describeChords(source as unknown as AnyTemplateItem[])
                        : getReadResults([ read['element'] as ReadQuery ], source!, state.activeTab, state.selection)
                    readOutput.push("# Read for tab: " + state.activeTab + "\n" + result)
                }
            }
            return readOutput.join('\n\n')
        }
        default: {
            console.warn(command, "No implementation yet")
        }
    }
}

/**
 * fill_template fails outright ("the template names no instrument to write for") when nothing
 * says what instrument to write, so an edit is never allowed to drop every instrument item.
 */
function keepInstruments(source: AnyTemplateItem[], edited: AnyTemplateItem[]): AnyTemplateItem[] {
    if (edited.some((i) => i.kind === 'instrument')) return edited
    const instruments = source.filter((i) => i.kind === 'instrument')
    if (instruments.length === 0) return edited
    console.warn('edit_template dropped every instrument item — re-adding', instruments)
    return [...instruments, ...edited]
}

// Chords aren't part of a template's XML rendering — the item list is the only place they
// live — so read_tab's "chord" element reads straight off the template's items instead of
// going through readContext's XML-based getReadResults.
function describeChords(template: AnyTemplateItem[]): string {
    const chords = (template ?? []).filter((i): i is ChordTemplateItem => i.kind === 'chord')
    if (chords.length === 0) return "No chords found."
    return chords
        .map(c => `m${c.measure ?? '?'}${c.pos != null ? `.${c.pos}` : ''}: ${c.rootNumeral ?? c.degree ?? '?'}${c.quality ?? ''}`)
        .join("\n")
}

function saveToMemory(memory: Record<string, string | AnyTemplateItem[]>, key: string, value: string | AnyTemplateItem[]) {
    memory[key] = value
    const appState = useStore.getState()

    if (!appState.tabs.some(t => t.name === key)) {
        appState.addNewTab(key)
    }
    else { console.log(key) }

    if (typeof value == 'string')
        appState.updateTab(key, { content: value })
    else appState.updateTab(key, { template: value })
}

function getScoreInputOutputNames(input: Record<string, unknown>) {
    let input_name = ""
    let output_name = ""
    if ('memory_name' in input) input_name = input['memory_name'] as string
    if ('memory_output_name' in input) output_name = input['memory_output_name'] as string
    return { in_key: input_name, out_key: output_name }
}

function getHistoryContext(history: ChatItem[]) {
    if (history.length === 0) return "" 
    return "History of Prompts:\n" + history.map(h => "User prompt: " + h.prompt + "\nOutput: " + h.output).join('\n')
}

const TOOL_DESCRIPTORS: Record<string, string> = {
    "edit_clef": "<> clefs",
    "edit_time_sig": "<> time signatures",
    "edit_key_sig": "<> key signatures",
    "edit_tempo": "<> tempos",
    "edit_add_instrument": "<> instruments",
    "edit_delete_staff": "<> instruments",
    "edit_delete_measures": "<> measures",
    "generate_score": "Generating music",
    "create_template_from_scratch": "Creating a blank template",
    "create_template_from_score": "Converting a new template",
    "edit_template": "<> a template",
    "fill_template": "Filling a template",
    "write_score": "<> the score",
    "read_tab": "Reading the score"
}

const DESCRIPTORS = [ 'Modifying', 'Editing', 'Shifting', 'Revising', 'Adjusting', 'Shaping', 'Composing', 'Arranging', 'Writing' ]
const DESCRIPTORS_PAST = [ 'Modified', 'Edited', 'Shifted', 'Revised', 'Adjusted', 'Shaped', 'Composed', 'Arranged', 'Wrote' ]

const PROMPT_INTRO = "Your job is to classify which edit types are required next to achieve the user's goal and in what order."
    + "You may use templates to achieve generation and repetitive tasks (e.g. an AABA song from scratch). A template is score abstraction where some elements are intentionally left blank to be filled in by a different model. Actions like creating a template from a score don't change anything about the score, it just modifies the type of the object so that it can be edited using other actions. When a template is filled in, it merely takes the blank/masked elements of the template and fills them in automatically."
    + "For example, if you are creating a song in AABA format, you might want to first generate 8 bars (the first A), use it to create a template, then edit the template by taking the converted score and stripping it of some of its qualities (e.g. chords, rhythm, some pitches). Then, you may command the template to be filled in and placed into the first 8 bars, then commanding it to fill in the template once more and place this in the second 8 bars (Note: when you fill in a template, you get a new object that is no longer considered a template as it is completely filled). "
    + "If nothing matches an edit type, return no tool. You also have the ability to store/retrieve templates and score segments to/from memory with a name identifier so that they can be reused (Note: operations do not happen in place automatically)."
    + "Do not worry about missing details like which measures/staves/values are meant. "
    + "If the user gives no location on where something should happen, assume the user means their current selection and tab. Consider the type of the current tab (score or template) to decide what specific commands should be used. Mention this in your roadmap."
    + "Finally, unless specified, the operation should typically be done in place."

const ROADMAP_TOOL = {
    name: "roadmap_tool",
    description: "Create a roadmap plan for how to achieve the user's task using the given tools",
    input_schema: {
        type: "object",
        properties: {
            steps: {
                type: "array",
                items: {
                    type: "object",
                    properties: {
                        tool: {
                            type: "string",
                            enum: [ "edit_clef", "edit_time_sig", "edit_key_sig", "edit_tempo", "edit_add_instrument", "edit_delete_staff", "edit_delete_measures",
                                "generate_score",
                                "create_template_from_scratch", "create_template_from_score", "edit_template",
                                "fill_template",
                                "write_score",
                                "read_tab"
                            ],
                            description: "Each distinct type of edit that can be made. Tool descriptions:"
                                + "\ngenerate_score: \"Uses a transformer model to generate a new score if no current score exists. Else, inserts into the given location. Do not include key, time_signature, or tempo when these attributes should remain the same.\""
                                + "\ncreate_template_from_scratch: \"Create a template from scratch. Only to be used when the user has very specific intentions.\""
                                + "\ncreate_template_from_score: \"Create a template from an existing score in memory. To be used in combination with generate_score when the user does not have specific intentions (e.g. melodic ideas, rhythmic ideas, chord progs, etc...)\""
                                + "\nedit_template: \"Edit a template already stored in memory (e.g. to strip it down/mask some of its musical elements). Use this to alter a template so that there is more variance when it is filled.\""
                                + "\nfill_template: \"Fills a template's masked elements, essentially converting it from a template to a score and storing it to a new name on memory.\""
                                + "\nwrite_score: \"Write a score onto another score. Essentially a copy and paste. Accepts multiple writes to perfrom at once to save tokens. Also lets you open a score if the user requires it.\""
                                + "\nread_tab: \"Use to read specific elements of a score/template, typically to locate places defined by the user (e.g. 'where the key changes to D major'). May also be used to read the user's current selection (e.g. 'where I am currently highlighting' or 'my current tab').\""
                                + "\n\nedit_clef, edit_time_sig, edit_key_sig, edit_tempo, edit_add_instrument, edit_delete_staff, and edit_delete_measures are all only usable for editing scores, not templates."
                        },
                        intention: {
                            type: "string",
                            description: "A brief description of what this step should do on a per step basis with any needed implementation details including the names of what will be read and stored to memory (e.g. generate a score with 16 measures and save to '16_bar_score')."
                        },
                        // macro_intention: {
                        //     type: "string",
                        //     description: "A brief description of what this step should achieve when looking at the big picture (e.g. generate the foundation of an A section to be converted into a template)"
                        // }
                    }
                },
                description: "The steps that comprise the plan to complete the user's request. These should be in order of implementation. Tools may be used more than once (e.g. generate score, convert score to template, edit template, etc...). If the user is asking for any edit to be made on a template, always use edit_template over any other edit like edit_delete_staff."
            },
            chat_name: {
                type: "string",
                description: "A very brief title for the name of this chat. Take into consideration previous chats. Omit to keep old name or no name at all"
            }
        }
    }
}

const TOOL_INTROS = [
    {
        name: "trigger_tool",
        description: "Select the next tool on the roadmap to execute.",
        input_schema: {
            type: "object",
            properties: {
                tool: {
                    type: "string",
                    enum: [ "edit_clef", "edit_time_sig", "edit_key_sig", "edit_tempo", "edit_add_instrument", "edit_delete_staff", "edit_delete_measures",
                        "generate_score",
                        "create_template_from_scratch", "create_template_from_score", "edit_template",
                        "fill_template",
                        "write_score",
                        "read_tab",
                        "complete"
                    ],
                    description: "The tool to execute, as provided by the current roadmap step. Use complete when the user's prompt has been completed."
                },
                intention: {
                    type: "string",
                    description: "A brief description of what this step should do on a per step basis with any needed implementation details including the names of what will be read and stored to memory (e.g. generate a score with 16 measures and save to '16_bar_score')"
                },
                next_steps: {
                    type: "string",
                    description: "A brief outline of what is complete and what you should do after this step using the roadmap at the beginning and the steps that have been completed so far."
                },
            },
            required: [ "tool" ]
        }
    },
]

const TEMPLATE_ITEM_LOCATION = {
    measure: {
        type: "integer",
        description: "The 1-based measure this item belongs to. Omit to be masked and regenerated."
    },
    pos: {
        type: "number",
        description: "The beat position (or rhythm) within the measure this item belongs to where 0 is the first beat of the measure, 1 is after a quarter note duration, and 0.5 is after an eighth note duration, etc... Omit to mask this note's rhythmic placement, to leave this item blank."
    }
}

const TEMPLATE_ITEM_SCHEMAS = [
    {
        type: "object",
        description: "Assigns a completely masked sequence of measures. This is the catalyst for completely new content. May also be used on a sub measure level by setting measure length to 1.",
        properties: {
            kind: { type: "string", enum: [ "masked" ] },
            measure_start: {
                type: "number",
                description: "The beginning of a sequence of completely masked measures (1-based)."
            },
            mask_measure_length: {
                type: "number",
                description: "The amount of measures to have completely masked.",
            },
            slots_per_measure: {
                type: "number",
                description: "The maximum number of new elements filled in per masked measure. Not all slots will be used"
            }
        },
        required: [ "kind", "measure_start", "mask_measure_length", "slots_per_measure" ]
    },
    {
        type: "object",
        description: "Assigns/blanks an instrument for a voice.",
        properties: {
            kind: { type: "string", enum: [ "instrument" ] },
            ...TEMPLATE_ITEM_LOCATION,
            instrument: {
                type: "string",
                enum: [ "other", "piano", "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin", "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano", "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir" ],
                description: "The instrument to assign. Omit to leave this item blank."
            },
            voice_idx: { type: "integer", description: "The index of the voice this item belongs to." }
        },
        required: [ "kind", "voice_idx" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a clef for a voice.",
        properties: {
            kind: { type: "string", enum: [ "clef" ] },
            ...TEMPLATE_ITEM_LOCATION,
            name: {
                type: "string",
                enum: [ "treble", "bass", "alto", "tenor", "treble8vb", "treble8va", "bass8vb", "none" ],
                description: "The clef to assign. Omit to leave this item blank."
            },
            voice_idx: { type: "integer", description: "The index of the voice this item belongs to." }
        },
        required: [ "kind", "voice_idx" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a key signature.",
        properties: {
            kind: { type: "string", enum: [ "key" ] },
            ...TEMPLATE_ITEM_LOCATION,
            key_tonic: { type: "string", description: "The tonic note of the key. Ex: C, Bb, E, F#. Omit to leave this item blank." },
            key_mode: { type: "string", description: "The mode of the key. Ex: major, minor, mixolydian. Omit to leave this item blank." }
        },
        required: [ "kind" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a time signature.",
        properties: {
            kind: { type: "string", enum: [ "time" ] },
            ...TEMPLATE_ITEM_LOCATION,
            numerator: { type: "string", description: "The time signature's numerator. Omit to leave this item blank." },
            denominator: { type: "string", description: "The time signature's denominator. Omit to leave this item blank." }
        },
        required: [ "kind" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a tempo.",
        properties: {
            kind: { type: "string", enum: [ "tempo" ] },
            ...TEMPLATE_ITEM_LOCATION,
            tempo: { type: "integer", description: "The tempo in beats per minute. Omit to leave this item blank." }
        },
        required: [ "kind" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a single note.",
        properties: {
            kind: { type: "string", enum: [ "note" ] },
            ...TEMPLATE_ITEM_LOCATION,
            duration: { type: "number", description: "The note's duration where 1 is a quarter note, 0.5 is an eighth note, etc... Omit to leave this item blank." },
            pitch: { type: "integer", description: "The note's pitch. Omit to leave this item blank." },
            voice_idx: { type: "integer", description: "The index of the voice this note belongs to. Omit to leave this item blank." }
        },
        required: [ "kind" ]
    },
    {
        type: "object",
        description: "Assigns/blanks a chord.",
        properties: {
            kind: { type: "string", enum: [ "chord" ] },
            ...TEMPLATE_ITEM_LOCATION,
            degree: { type: "string", description: "The chord's roman numeral degree. Omit to leave this item blank." },
            rootNumeral: { type: "string", description: "The numeral of the chord's root degree. Omit to leave this item blank." },
            quality: {
                type: "string",
                enum: [ "maj7", "maj", "min", "dom7", "min7", "dim7", "dim", "m7b5", "sus2", "7sus", "7aug", "aug" ],
                description: "The chord's quality. Omit to leave this item blank."
            }
        },
        required: [ "kind" ]
    }
]

const EDIT_TEMPLATE_STAGE2_INTRO = "You are editing one existing template in place. A template is an ordered list of items describing a piece of music; an element is masked by keeping its item and omitting the field being masked (e.g. drop `pitch` from a note item to mask that note's pitch), which is what lets the filler model regenerate it."
    + "\nRules:"
    + "\n- Return the COMPLETE edited template: every item of the input template, in its original order, including the items you did not touch. Any item you leave out is deleted from the template."
    + "\n- You are modifying the template you were given. Never return a brand new template, and never collapse it into a single fully-masked block, unless the request explicitly asks to replace all of its content."
    + "\n- Keep the setup items (instrument, clef, key, time, tempo) and their values. Filling fails if no item names an instrument."
    + "\n- To mask a musical element, omit the field(s) you want regenerated from its item. To remove content entirely, drop the item."

const EDIT_TEMPLATE_STAGE2_TOOL=  {
    name: "edit_template",
    description: "Return the given template with the requested edits applied (e.g. mask musical elements like rhythm, pitch, chords, etc...). The template you return replaces the one you were given, so it must contain every item you want to keep.",
    input_schema: {
        type: "object",
        properties: {
            memory_output_name: {
                type: "string",
                description: "The new name of the edited template to be stored in memory."
            },
            template: {
                type: "array",
                items: { anyOf: TEMPLATE_ITEM_SCHEMAS },
                description: "The COMPLETE edited template: every item of the template you were given, in the same order, with your edits applied — not just the items you changed. Each item's `kind` selects which fields apply. Omit a non-required field to mask it, to be filled in later. It is recommended to not edit elements like tempos, instruments, keys, and time signatures unless specified."
            },
        },
        required: [ "memory_output_name", "template" ]
    }
}

const TOOLS = {
    "edit_clef": {
        name: "edit_clef",
        description: "Edit the clef of a staff in the score by providing the measure number and new clef",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new clef into. Begins at 1."
                },
                staff_num: {
                    type: "integer",
                    description: "The index of the staff to insert the new clef into. Begins at 0."
                },
                new_clef: {
                    type: "string",
                    enum: [ "treble", "bass", "alto", "tenor", "treble8vb", "treble8va", "bass8vb", "none" ],
                    description: "The new clef to insert. None to remove clef."
                }
            }
        }
    },
    "edit_time_sig": {
        name: "edit_time_sig",
        description: "Edit the time signature of the score by providing the measure number and new time",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new time signature into. Begins at 1."
                },
                new_time: {
                    type: "string",
                    enum: [ "2/4", "3/4", "4/4", "6/8", "3/8", "12/8" ],
                    description: "The new time signature to insert"
                }
            }
        }
    },
    "edit_key_sig": {
        name: "edit_key_sig",
        description: "Edit the key signature of the score by providing the measure number and new key",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new key signature into. Begins at 1."
                },
                tonic: {
                    type: "string",
                    description: "The tonic note of the new key. Ex: C, Bb, E, F#"
                },
                mode: {
                    type: "string",
                    description: "The mode of the new key. Ex: major, minor, mixolydian"
                }
            }
        }
    },
    "edit_tempo": {
        name: "edit_tempo",
        description: "Edit the tempo of the score by providing the measure number and new tempo",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_num: {
                    type: "integer",
                    description: "The measure to insert the new tempo into. Begins at 1."
                },
                new_tempo: {
                    type: "integer",
                    description: "The new tempo in beats per minute"
                }
            }
        }
    },
    "edit_add_instrument": {
        name: "edit_add_instrument",
        description: "Add an instrument to the score by providing the new instrument",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                new_instrument: {
                    type: "string",
                    enum: [ "other", "piano",  "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin", "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano", "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir" ],
                    description: "The new instrument to add"
                }
            }
        }
    },
    "edit_delete_staff": {
        name: "edit_delete_staff",
        description: "Remove a staff on the score",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                staff_num: {
                    type: "integer",
                    description: "The index of the staff to delete. Begins at 0."
                }
            }
        }
    },
    "edit_delete_measures": {
        name: "edit_delete_measures",
        description: "Remove a range of measures on the score (inclusive)",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score stored in memory."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_start_num: {
                    type: "integer",
                    description: "The first measure of the range to be deleted"
                },
                measure_end_num: {
                    type: "integer",
                    description: "The last measure of the range to be deleted"
                }
            }
        }
    },
    "generate_score": {
        name: "generate_score",
        description: "Uses a transformer model to generate a new score if no current score exists. Else, inserts into the given location. Do not include key, time_signature, or tempo when these attributes should remain the same.",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the existing score stored in memory (if there is one)."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the score to be stored in memory. Use the same as the input name to make the operation in place."
                },
                measure_start_num: {
                    type: "integer",
                    description: "The first measure where the newly generated material will be inserted."
                },
                generation_staffs: {
                    type: "array",
                    items: {
                        type: "object",
                        properties: {
                            staff_num: {
                                type: "integer",
                                description: "The index of the staff"
                            },
                            instrument: {
                                type: "string",
                                enum: [ "other", "piano",  "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin", "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano", "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir" ],
                                description: "The name of the instrument to generate. Only include if the instruments aren't already identified by the staff_num (e.g. no current score exists)"
                            }
                        },
                        required: [ "staff_num" ]
                    },
                    description: "List the staffs to be generated. Provide an empty array if no staffs exist and the user hasn't given a list of staffs or instruments. If an instrument uses multiple staffs (like piano or organ which typically have treble and bass staffs), add that instrument multiple times."
                },
                generation_length: {
                    type: "integer",
                    description: "The number of measures to generate (multiple staffs still counts as one measure)"
                },
                overwrite_current: {
                    type: "boolean",
                    description: "True if the generation should overwrite the current measures located from measure_start_num to measure_start_num + generation_length. False means it will be inserted, creating new measures at the given location."
                },
                tonic: {
                    type: [ "string", "null" ],
                    enum: [ "Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#", null ],
                    description: "The tonic of the key the generation is in, spelled as the mode's own tonic (A minor is 'A', not 'C'). Null to carry on in whatever key is in effect before measure_start_num."
                },
                mode: {
                    type: [ "string", "null" ],
                    enum: [ "major", "minor", "dorian", "phrygian", "lydian", "mixolydian", "locrian", null ],
                    description: "The mode of that key. Give it alongside tonic; the backend derives the key signature from the pair. Null is read as major."
                },
                time_signature: {
                    type: [ "string", "null" ],
                    enum: [ "2/4", "3/4", "4/4", "6/8", "3/8", "12/8", null ],
                    description: "The time signature the generation is in. Null to carry on in whatever time signature is in effect before measure_start_num."
                },
                tempo: {
                    type: [ "integer", "null" ],
                    description: "The tempo of the generation in beats per minute. Null to carry on at whatever tempo is in effect before measure_start_num."
                }
            },
            required: [ "measure_start_num", "generation_staffs", "generation_length", "overwrite_current", "tonic", "mode", "time_signature", "tempo" ]
        }
    },
    "create_template_from_scratch": {
        name: "create_template_from_scratch",
        description: "Create a template from scratch. Only to be used when the user has very specific intentions.",
        input_schema: {
            type: "object",
            properties: {
                memory_output_name: {
                    type: "string",
                    description: "The new name of the template to be stored in memory."
                },
                template: {
                    type: "array",
                    items: { anyOf: TEMPLATE_ITEM_SCHEMAS },
                    description: "The ordered list of items making up the template. Each item's `kind` selects which fields apply. Omit a non-required field to leave that item blank, to be filled in later."
                },
            }
        }
    },
    "create_template_from_score": {
        name: "create_template_from_score",
        description: "Create a template from an existing score in memory. To be used in combination with generate_score when the user does not have specific intentions (e.g. melodic ideas, rhythmic ideas, chord progs, etc...)",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the score already stored in memory to be converted."
                },
                memory_output_name: {
                    type: "string",
                    description: "The new name of the template to be stored in memory."
                }
            }
        }
    },
    "edit_template": {
        name: "edit_template",
        description: "Edit a template already stored in memory. Note: this command is 2 stages. This is only the first stage 1. After giving the input template, you will be given the template stored in memory as json, and will then make edits to it.",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the template already stored in memory that will be edited."
                }
            }
        }
    },
    "fill_template": {
        name: "fill_template",
        description: "Fills a template, essentially converting it from a template to a score and storing it to a new name on memory.",
        input_schema: {
            type: "object",
            properties: {
                memory_name: {
                    type: "string",
                    description: "The name of the template already stored in memory that will be filled and converted."
                },
                memory_output_name: {
                    type: "string",
                    description: "The name of the filled score to be stored in memory."
                }
            }
        }
    },
    "write_score": {
        name: "write_score",
        description: "Write a score onto another score. Essentially a copy and paste. This action can perform multiple writes at once, saving tokens.",
        input_schema: {
            type: "object",
            properties: {
                writes: {
                    type: "array",
                    items: {
                        type: "object",
                        properties: {
                            memory_name: {
                                type: "string",
                                description: "The name of the score already stored in memory that will be read/copied."
                            },
                            memory_output_name: {
                                type: "string",
                                description: "The name of the score that will be written to."
                            },
                            write_start_measure: {
                                type: "number",
                                description: "The 1-based measure where the input score will be written on the output score."
                            }
                        },
                        required: [ "memory_name", "memory_output_name", "write_start_measure" ],
                        description: "A singular write command. Use multiple to make multiple edits. Edits made in the order provided."
                    },
                    description: "The write commands, carried out in the order given."
                },
                open: {
                    type: "string",
                    description: "Enter an existing tab or written-to tab to draw the user's attention by opening it. Omit to not open the tab."
                }
            },
            required: [ "writes" ]
        }
    },
    "read_tab": {
        name: "read_tab",
        description: "Reads specified elements of the given score/template",
        input_schema: {
            type: "object",
            properties: {
                queries: {
                    type: "array",
                    items: {
                        type: "object",
                        properties: {
                            element: {
                                type: "string",
                                enum: [ "clef", "key_signature", "time_signature", "tempo", "staves", "measure_count", "selection", "tab", "chord" ]
                            },
                            memory_name: {
                                type: "string",
                                description: "The name of the score/template in memory to read from. Leave blank for the user's active tab."
                            }
                        }
                    },
                    description: "Aspects to read. clef=per-staff clef changes; key_signature/time_signature/tempo=value at measure 1 and every measure it changes; staves=instrument roster with 0-based staff_num; measure_count=number of measures; selection=the measure(s) and staff/staves the user has selected on the score; chord=every chord item defined on a template, with its measure/position, roman numeral and quality."
                }
            },
            required: [ "queries" ]
        }
    },
    "complete": {
        name: "complete",
        description: "The final step: marks the user's task as complete",
        input_schema: {
            type: "object",
            properties: {
                recap: {
                    type: "string",
                    description: "Briefly summarize what was done to complete the user's prompt. A future model will use this as context for future user prompts, so ensure to include crucial details. For example, you should include what edits were made, the steps executed to complete the task, and what scores/templates in memory were modified/created and what they were for."
                }
            }
        }
    }
}