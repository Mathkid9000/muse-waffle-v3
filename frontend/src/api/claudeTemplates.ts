import Anthropic from '@anthropic-ai/sdk';
import { anthropic } from "./claudeTools";
import type { ToolUseBlock } from "@anthropic-ai/sdk/resources";
import type { AnyTemplateItem } from '../types';
import { api } from './client';
import { applyClaudeEdit } from './applyClaudeEdit';
import { fillTemplate, scoreToTemplate } from './templates';

const MAX_TURNS = 10;

export async function promptTemplateAnthropic(prompt: string, xml: string | null) {
    console.log('Prompted:', prompt, xml)

    // const memorys: Record<string, string | AnyTemplateItem[]> = {}
    // const gen_req = {
    //     tool: "generate_score",
    //     input: {
    //         "measure_start_num": 1,
    //         "generation_staffs": [
    //             {
    //                 "staff_num": 0,
    //                 "instrument": "voice_soprano"
    //             }
    //         ],
    //         "generation_length": 8,
    //         "overwrite_current": true,
    //         "tonic": "C",
    //         "mode": "major",
    //         "time_signature": "4/4",
    //         "tempo": 120,
    //         "memory_output_name": "simple_song_AABA"
    //     }
    // }
    // await processCommand(gen_req, memorys)

    // const genxml = memorys['simple_song_AABA'] as string
    // const template = await scoreToTemplate(genxml)
    // console.log(template)
    // const score = await fillTemplate(template)
    // memorys['OUTPUT'] = score

    // console.log('COMPLETE:', memorys)
    // return memorys['OUTPUT']
    // step 1: roadmap
    const roadmapResponse = await anthropic.messages.create({
        model: 'claude-haiku-4-5-20251001',
        max_tokens: 1024,
        tools: [ ROADMAP_TOOL ] as unknown as Anthropic.Messages.ToolUnion[],
        system: PROMPT_INTRO,
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
    const roadmap = "ROADMAP (complete these steps one by one):\n" + (roadmapTool.input as any).steps.map((s: { tool: string, intention: string }, idx: number) => 
        `# Step ${idx + 1} (tool=${s.tool}):\n${s.intention}`
    ).join('\n\n')

    console.log('GENERATED ROADMAP:\n', roadmap)

    const messages: Anthropic.Messages.MessageParam[] = [{ role: 'user', content: prompt }, { role: 'assistant', content: roadmap }];
    const memory: Record<string, string | AnyTemplateItem[]> = {}
    let usageIn = roadmapResponse.usage.input_tokens;
    let usageOut = roadmapResponse.usage.output_tokens;

    // Step 2: carry out roadmap
    for (let turn = 0; turn < MAX_TURNS; turn++) {
        console.log('TURN:', turn, 'MESSAGES:', messages, 'USAGE IN:', usageIn, 'USAGE OUT:', usageOut)
        const toolSelectionMessage = await anthropic.messages.create({
            model: 'claude-haiku-4-5-20251001',
            max_tokens: 1024,
            tools: TOOL_INTROS as unknown as Anthropic.Messages.ToolUnion[],
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
        if (selectedTool === 'complete') break;
        const toolMessage = await anthropic.messages.create({
            model: 'claude-haiku-4-5-20251001',
            max_tokens: 1024,
            tools: [ TOOLS[selectedTool as keyof typeof TOOLS] ] as unknown as Anthropic.Messages.ToolUnion[],
            system: getMemoryContext(memory),
            messages: [{ role: 'user', content: `${prompt}\n\n${roadmap}\n\nDo this now: ${intention}` }],
            tool_choice: { type: "tool", name: selectedTool }
        });
        usageIn += toolMessage.usage.input_tokens
        usageOut += toolMessage.usage.output_tokens

        const toolUses = toolMessage.content.filter((b): b is ToolUseBlock => b.type === "tool_use");
        // console.log('Found Tool Uses:', toolUses, "in message:", toolMessage, getMemoryContext(memory))

        const results: Anthropic.Messages.ToolResultBlockParam[] = []
        for (const tool of toolUses) {
            await processCommand({ tool: tool.name, input: tool.input as Record<string, unknown> }, memory, messages)
            results.push({
                type: "tool_result",
                tool_use_id: tool.id,
                content: `${tool.name} done. Memory holds: ${Object.keys(memory).join(', ') || 'nothing yet'}`,
            })
        }
        messages.push({ role: 'assistant', content: toolMessage.content });
        if (results.length) messages.push({ role: 'user', content: results });
        messages.push({ role: 'assistant', content: next_steps });
    }

    console.log('PROMPTING COMPLETE', memory, usageIn, usageOut)
    return memory['OUTPUT']
}

async function getCommandContext(command: string, memory: Record<string, string | AnyTemplateItem[]>) {
    switch (command) {
        case "edit_template": {
            return `The existing templates:\n${Object.keys(memory).filter(k => typeof memory[k] !== "string").map(k => k + ":\n" + JSON.stringify(memory[k])).join('\n\n')}`
        }
        default: return null
    }
}

function getMemoryContext(memory: Record<string, string | AnyTemplateItem[]>) {
    if (Object.keys(memory).length == 0) return "There are no scores or templates saved in memory." 
    return "Current scores and templates saved in memory:\n" + 
        Object.keys(memory).map(k => `${k} (${typeof memory[k] === "string" ? 'score' : 'template'})`).join(', ')
}

async function processCommand(command: { tool: string, input: Record<string, unknown> }, memory: Record<string, string | AnyTemplateItem[]>, messages: Anthropic.Messages.MessageParam[]) {
    switch (command.tool) {
        case "generate_score": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            if (in_key && in_key in memory) {
                const xml = memory[in_key] as string
                const newXml = await applyClaudeEdit(command.tool, command.input, xml)
                memory[out_key] = newXml
            }
            else {
                const newXml = await applyClaudeEdit(command.tool, command.input, '')
                memory[out_key] = newXml
            }
            console.log("FINISHED GENERATE:",memory)
            break
        }
        case "create_template_from_scratch": {
            const { in_key: _, out_key } = getScoreInputOutputNames(command.input)
            const template = command.input['template'] as AnyTemplateItem[]
            memory[out_key] = template
            break
        }
        case "create_template_from_score": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            const xml = memory[in_key] as string
            memory[out_key] = await scoreToTemplate(xml)
            break
        }
        case "fill_template": {
            const { in_key, out_key } = getScoreInputOutputNames(command.input)
            const template = memory[in_key] as AnyTemplateItem[]
            if (!template) {
                console.warn(command, "No template stored under", in_key)
                break
            }
            memory[out_key] = await fillTemplate(template)
            break
        }
        case "edit_template": {
            const { in_key, out_key: _ } = getScoreInputOutputNames(command.input)
            const template = memory[in_key] as AnyTemplateItem[]

            const messageStage2 = await anthropic.messages.create({
                model: 'claude-haiku-4-5-20251001',
                max_tokens: 1024,
                tools: [ EDIT_TEMPLATE_STAGE2_TOOL ] as unknown as Anthropic.Messages.ToolUnion[],
                system: `${in_key} Template:\n${JSON.stringify(template)}`,
                messages: messages,
                tool_choice: { type: "tool", name: 'edit_template' }
            });
            console.log('STAGE 222222222',messageStage2)
            const toolUses = messageStage2.content.filter((b): b is ToolUseBlock => b.type === "tool_use");
            if (toolUses.length == 0) { console.warn('NO EDITS FOUND:', template, messageStage2); return }
            const input = toolUses[0].input as Record<string, unknown>
            const { in_key: __, out_key } = getScoreInputOutputNames(input)
            const newTemplate = input['template'] as AnyTemplateItem[]
            memory[out_key] = newTemplate
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
                memory[out_key] = (await api.editWriteScore({
                    xml: (memory[out_key] as string) ?? '',
                    source_xml: source,
                    measure_start_num: write['write_start_measure'] as number,
                })).xml
            }
            break
        }
        default: {
            console.warn(command, "No implementation yet")
        }
    }
}

function getScoreInputOutputNames(input: Record<string, unknown>) {
    let input_name = ""
    let output_name = ""
    if ('memory_name' in input) input_name = input['memory_name'] as string
    if ('memory_output_name' in input) output_name = input['memory_output_name'] as string
    return { in_key: input_name, out_key: output_name }
}

const PROMPT_INTRO = "Your job is to classify which edit types are required next to achieve the user's goal and in what order."
    + "You may use templates to achieve generation and repetitive tasks (e.g. an AABA song from scratch). A template is score abstraction where some elements are intentionally left blank to be filled in by a different model. "
    + "For example, if you are creating a song in AABA format, you might want to first generate 8 bars (the first A), then create a template by taking this score and stripping it of some of its qualities (e.g. chords, rhythm, some pitches). Then, you may command the template to be filled in and placed into the first 8 bars, then commanding it to fill in the template once more and place this in the second 8 bars (Note: when you fill in a template, you get a new object that is no longer considered a template as it is completely filled). "
    + "If nothing matches an edit type, return no tool. You also have the ability to store/retrieve templates and score segments to/from memory with a name identifier so that they can be reused (Note: operations do not happen in place automatically)."
    + "Finally, the output, final score should be saved to the name 'OUTPUT'. Reserve this only for when you want a score to be returned. It may also be a template if the user requests a template be returned."

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
                            ],
                            description: "Each distinct type of edit that can be made. Tool descriptions:"
                                + "\ngenerate_score: \"Uses a transformer model to generate a new score if no current score exists. Else, inserts into the given location. Do not include key, time_signature, or tempo when these attributes should remain the same.\""
                                + "\ncreate_template_from_scratch: \"Create a template from scratch. Only to be used when the user has very specific intentions.\""
                                + "\ncreate_template_from_score: \"Create a template from an existing score in memory. To be used in combination with generate_score when the user does not have specific intentions (e.g. melodic ideas, rhythmic ideas, chord progs, etc...)\""
                                + "\nedit_template: \"Edit a template already stored in memory (e.g. to strip it down/mask some of its musical elements). Use this to alter a template so that there is more variance when it is filled.\""
                                + "\nfill_template: \"Fills a template's masked elements, essentially converting it from a template to a score and storing it to a new name on memory.\""
                                + "\nwrite_score: \"Write a score onto another score. Essentially a copy and paste. Accepts multiple writes to perfrom at once to save tokens.\""
                        },
                        intention: {
                            type: "string",
                            description: "A brief description of what this step should do on a per step basis with any needed implementation details including the names of what will be read and stored to memory (e.g. generate a score with 16 measures and save to '16_bar_score')"
                        },
                        // macro_intention: {
                        //     type: "string",
                        //     description: "A brief description of what this step should achieve when looking at the big picture (e.g. generate the foundation of an A section to be converted into a template)"
                        // }
                    }
                },
                description: "The steps that comprise the plan to complete the user's request. These should be in order of implementation. Tools may be used more than once (e.g. generate score, convert score to template, edit template, etc...)"
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
                        "complete"
                    ],
                    description: "The tool to execute, as provided by the current roadmap step. Use complete when the user's prompt has been completed and written to memory as 'OUTPUT'"
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
        description: "The 1-based measure this item belongs to. Omit if not yet placed."
    },
    pos: {
        type: "number",
        description: "The beat position within the measure this item belongs to. Omit if not yet placed."
    }
}

const TEMPLATE_ITEM_SCHEMAS = [
    {
        type: "object",
        description: "Assigns a completely masked sequence of measures. This is the catalyst for completely new content. May also be used on a sub measure level by setting measure length to 1.",
        properties: {
            key: { type: "string", enum: [ "masked" ] },
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
        required: [ "measure_start", "mask_measure_length", "slots_per_measure" ]
    },
    {
        type: "object",
        description: "Assigns/blanks an instrument for a voice.",
        properties: {
            kind: { type: "string", enum: [ "instrument" ] },
            ...TEMPLATE_ITEM_LOCATION,
            instrument: {
                type: "string",
                enum: [ "other", "piano",  "harpsichord", "organ", "accordion", "guitar", "bass_guitar", "violin", "viola", "cello", "double_bass", "harp", "flute", "oboe", "clarinet", "bassoon", "saxophone", "trumpet", "horn", "trombone", "tuba", "percussion", "timpani", "celesta", "voice_soprano", "voice_mezzo", "voice_alto", "voice_tenor", "voice_baritone", "voice_bass", "voice", "choir" ],
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
            duration: { type: "number", description: "The note's duration. Omit to leave this item blank." },
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

const EDIT_TEMPLATE_STAGE2_TOOL=  {
    name: "edit_template",
    description: "Edit the given template (e.g. mask musical elements like rhythm, pitch, chords, etc...)",
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
                    description: "The first measure to insert the generation into"
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
        description: "Write a score onto another score. Essentially a copy and paste. This action can perform multiple writes at once, saving tokens. Recall that you must write to 'OUTPUT' for the score to show.",
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
                }
            },
            required: [ "writes" ]
        }
    }
}