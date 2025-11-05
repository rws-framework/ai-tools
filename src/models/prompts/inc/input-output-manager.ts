import { CompoundInput, IPromptEnchantment, IRWSHistoryMessage } from './types';

export class InputOutputManager {
    private input: CompoundInput[] = [];
    private enhancedInput: IPromptEnchantment[] = [];
    private sentInput: CompoundInput[] = [];
    private originalInput: CompoundInput[] = [];
    private output: string | object = null;
    private onStream: (chunk: string) => void = () => {};

    constructor(input: CompoundInput[]) {
        this.input = input;
        this.originalInput = input;
    }

    addInput(content: CompoundInput): void {
        this.input.push(content);
    }

    addEnchantment(enchantment: IPromptEnchantment): void {
        this.enhancedInput.push(enchantment);
    }

    getEnchantedInputs(): IPromptEnchantment[] {
        return this.enhancedInput;
    }

    readSentInput(): CompoundInput[] {
        return this.sentInput;
    }

    setSentInput(input: CompoundInput[]): void {
        this.sentInput = input;
    }

    readInput(): CompoundInput[] {
        const enchantedInput: CompoundInput[] = this.enhancedInput.map(enchantment => ({ 
            role: enchantment.input.role || 'user', 
            content: enchantment.input.content
        }));

        return [...enchantedInput, ...this.input];
    }

    readBaseInput(): CompoundInput[] {
        return this.originalInput;
    }

    setBaseInput(input: CompoundInput[]): void {
        this.originalInput = input;
    }

    injestOutput(content: string): void {
        this.output = content;
    }

    readOutput(): string | object{
        return this.output;
    }

    listen(source: string | object, stream: boolean = true): void {
        if(stream){
            this.output = '';
        }        

        if (!stream) {
            this.output = source;
        } else {
            this.output += source as string;
            this.onStream(source as string);
        }
    }

    setStreamCallback(callback: (chunk: string) => void): void {
        this.onStream = callback;
    }

    addHistory(messages: IRWSHistoryMessage[], historyPrompt: string, callback?: (messages: IRWSHistoryMessage[], prompt: string) => void): void {
        const prompt = `
            ${messages.map(message => `
                ${message.creator}: ${message.content}
            `).join('\n\n')}
            ${historyPrompt}
        `;

        if (callback) {
            callback(messages, prompt);
        } else {
            this.input = [{ 
                role: 'user', 
                content: { 
                    type: 'text', 
                    text: prompt 
                } 
            }, ...this.input];
        }
    }

    async readStreamAsText(readableStream: ReadableStream, callback: (txt: string) => void): Promise<void> {
        const reader = readableStream.getReader();
        let readResult: any;

        // Continuously read from the stream
        while (!(readResult = await reader.read()).done) {
            if (readResult.value && readResult.value.response) {
                // Emit each chunk text as it's read
                callback(readResult.value.response);
            }
        }
    }

    // Getters for state access
    getInput(): CompoundInput[] {
        return this.input;
    }

    getEnhancedInput(): IPromptEnchantment[] {
        return this.enhancedInput;
    }

    getOriginalInput(): CompoundInput[] {
        return this.originalInput;
    }

    getOutput(): string | object {
        return this.output;
    }
}
