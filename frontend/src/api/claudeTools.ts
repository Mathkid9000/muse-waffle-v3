import Anthropic from '@anthropic-ai/sdk';

const anthropic = new Anthropic();

async function prompt(prompt: string) {
  const message = await anthropic.messages.create({
    model: 'claude-haiku-4-5-20251001', // Specify your desired model
    max_tokens: 1024,
    messages: [
      { role: 'user', content: prompt }
    ],
  });

  console.log(message.content);
}

const TOOLS = [
    { // Change key
        name: "edit_key",
        description: "Edit the key signature of the score at a certain measure by providing the tonic note and the musical mode.",
        input_schema: {
            type: "object",
            properties: {
                
            }
        }
    },
    {
        name: "edit_clef",
        description: "Edit the clef of a staff in the score by providing the measure number and new clef",
        input_schema: {
            type: "object",
            properties: {
                
            }
        }
    }
]