import { Readable } from 'stream';
import { PromptTemplate } from '@langchain/core/prompts';
import ConvoLoader, { IChainCallOutput } from '../convo/ConvoLoader';
import { BedrockChat } from '@langchain/community/chat_models/bedrock/web';
import { IterableReadableStream } from '@langchain/core/utils/stream';
import { ChainValues } from '@langchain/core/utils/types';

import { IContextToken } from '../../types/IContextToken';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';

interface IPromptHyperParameters {
    temperature: number,
    top_k?: number,
    top_p?: number,
    [key: string]: number
}

interface IRWSHistoryMessage { 
    content: string, creator: string 
}

interface ILLMChunk {
    content: string
    status: string
 }

interface IPromptParams {
    hyperParameters?: IPromptHyperParameters;
    input?: string;
    modelId: string;
    modelType: string;
}

interface IPromptEnchantment {
    enhancementId: string,
    enhancementName: string,
    enhancementParams: any,
    input: string
    output: string
}

type IPromptSender = (prompt: RWSPrompt) => Promise<void>;

interface IRWSPromptRequestExecutor {
    promptRequest: (prompt: RWSPrompt, contextToken?: IContextToken | null, intruderPrompt?: string | null, debugVars?: any) => Promise<RWSPrompt>
}


interface IRWSSinglePromptRequestExecutor {
    singlePromptRequest: (prompt: RWSPrompt, contextToken?: IContextToken | null, intruderPrompt?: string | null, ensureJson?: boolean, debugVars?: any) => Promise<RWSPrompt>
}


interface IRWSPromptStreamExecutor {
    promptStream: (prompt: RWSPrompt, read: (chunk: ILLMChunk) => void, end: () => void, debugVars?: any) => Promise<RWSPrompt>
}

interface IRWSPromptJSON {
    input: string;
    enhancedInput: IPromptEnchantment[];
    sentInput: string;
    originalInput: string;
    output: string;
    modelId: string;
    modelType: string;
    multiTemplate: PromptTemplate;
    convo: { id: string };
    hyperParameters: IPromptHyperParameters;
    created_at: string;
    var_storage: any;
}

type ChainStreamType = AsyncGenerator<IterableReadableStream<ChainValues>>;

class RWSPrompt {
    public _stream: ChainStreamType;
    private input: string = '';
    private enhancedInput: IPromptEnchantment[];
    private sentInput: string = '';
    private originalInput: string = '';
    private output: string = '';
    private modelId: string;
    private modelType: string;
    private multiTemplate: PromptTemplate;
    private convo: ConvoLoader<any>;
    private hyperParameters: IPromptHyperParameters;
    private created_at: Date;

    private varStorage: any = {};

    private onStream = (chunk: string) => {

    };

    constructor(params: IPromptParams){
        this.input = params.input;
        this.originalInput = params.input;
        this.hyperParameters = params.hyperParameters;
        this.modelId = params.modelId;
        this.modelType = params.modelType;

        this.created_at = new Date();
    }

    listen(source: string, stream: boolean = true): RWSPrompt
    {              
        this.output = '';

        if (!stream) {
            this.output = source;
        } else {           
            this.output += source;
            this.onStream(source);            
        }
        
        return this;
    }   

    setStreamCallback(callback: (chunk: string) => void): void
    {
        this.onStream = callback;
    }

    addEnchantment(enchantment: IPromptEnchantment): void
    {
        this.enhancedInput.push(enchantment);
        this.input = enchantment.input;        
    }

    getEnchantedInput(): string | null
    {
        return this.enhancedInput[this.enhancedInput.length - 1].output;
    }

    getModelId(): string
    {
        return this.modelId;
    }

    readSentInput(): string
    {
        return this.sentInput;
    }

    readInput(): string
    {
        return this.input;
    }

    
    readBaseInput(): string
    {
        return this.originalInput;
    }    

    setBaseInput(input: string): RWSPrompt
    {
        this.originalInput = input;
        
        return this;
    }

    injestOutput(content: string): RWSPrompt
    {
        this.output = content;

        return this;
    }

    readOutput(): string
    {
        return this.output;
    }

    getHyperParameters<T extends IPromptHyperParameters>(base: any = null): T
    {        
        if(base){
            this.hyperParameters = {...base, ...this.hyperParameters};
        }

        return this.hyperParameters as T;
    }

    getHyperParameter<T>(key: keyof IPromptHyperParameters): T
    {        
        if(!this.hyperParameters[key]){
            return null;
        }

        return this.hyperParameters[key] as T;
    }

    setHyperParameter(key: string, value: any): RWSPrompt
    {        
        this.hyperParameters[key] = value;
        
        return this;
    }
    
    setHyperParameters(value: any): RWSPrompt
    {        
        this.hyperParameters = value;
        
        return this;
    }

    setMultiTemplate(template: PromptTemplate): RWSPrompt
    {
        this.multiTemplate = template;
        return this;
    }

    getMultiTemplate(): PromptTemplate
    {
        return this.multiTemplate;
    }

    setConvo(convo: ConvoLoader<BaseChatModel>): RWSPrompt
    {
        this.convo = convo.setPrompt(this);        
        
        return this;
    }

    getConvo<T extends BaseChatModel>(): ConvoLoader<T>
    {
        return this.convo;
    }

    replacePromptVar(key: string, val: string)
    {

    }

    getModelMetadata(): [string, string]
    {
        return [this.modelType, this.modelId];
    }

    async requestWith(executor: IRWSPromptRequestExecutor, intruderPrompt: string = null, debugVars: any = {}): Promise<void>
    {
        this.sentInput = this.input;
        const returnedRWS = await executor.promptRequest(this, null, intruderPrompt, debugVars);
        this.output = returnedRWS.readOutput();        
    }

    async singleRequestWith(executor: IRWSSinglePromptRequestExecutor, intruderPrompt: string = null, ensureJson: boolean = false): Promise<void>
    {        
        await executor.singlePromptRequest(this, null, intruderPrompt, ensureJson);
        this.sentInput = this.input;
    }

    async streamWith(executor: IRWSPromptStreamExecutor, read: (chunk: ILLMChunk) => void, end: () => void = () => {}, debugVars: any = {}): Promise<RWSPrompt>
    {        
        this.sentInput = this.input;
        return executor.promptStream(this, read, end, debugVars);
    }

    setInput(content: string): RWSPrompt
    {
        this.input = content;
        return this;
    }

    getVar<T>(key: string): T
    {
        return Object.keys(this.varStorage).includes(key) ? this.varStorage[key] : null;
    }

    setVar<T>(key: string, val: T): RWSPrompt {
        this.varStorage[key] = val;

        return this;
    } 

    async _oldreadStream(stream: Readable, react: (chunk: string) => void): Promise<void>    
    {        
        let first = true;
        const chunks: string[] = []; // Replace 'any' with the actual type of your chunks
       
        for await (const event of stream) {            
            // Assuming 'event' has a specific structure. Adjust according to actual event structure.
            if ('chunk' in event && event.chunk.bytes) {
                const chunk = JSON.parse(Buffer.from(event.chunk.bytes).toString('utf-8'));
                if(first){
                    console.log('chunk', chunk);
                    first = false;
                }

                react(chunk.completion);

                chunks.push(chunk.completion || chunk.generation ); // Use the actual property of 'chunk' you're interested in
            } else if (
                'internalServerException' in event ||
                'modelStreamErrorException' in event ||
                'throttlingException' in event ||
                'validationException' in event
            ) {
                console.error(event);
                break;
            }            
        }        
    }

    private async isChainStreamType(source: any): Promise<boolean> {
        if (source && typeof source[Symbol.asyncIterator] === 'function') {
            const asyncIterator = source[Symbol.asyncIterator]();
            if (typeof asyncIterator.next === 'function' && 
                typeof asyncIterator.throw === 'function' && 
                typeof asyncIterator.return === 'function') {
                try {
                    // Optionally check if the next method yields a value of the expected type
                    const { value, done } = await asyncIterator.next();
                    return !done && value instanceof ReadableStream; // or whatever check makes sense for IterableReadableStream<ChainValues>
                } catch (error) {
                    // Handle or ignore error
                }
            }
        }
        return false;
    }

    async  readStreamAsText(readableStream: ReadableStream, callback: (txt: string) => void) {
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

    addHistory(messages: IRWSHistoryMessage[], historyPrompt: string, callback?: (messages: IRWSHistoryMessage[], prompt: string) => void){
        const prompt = `
            ${messages.map(message => `
                ${message.creator}: ${message.content}
            `).join('\n\n')}
            ${historyPrompt}
        ` ;

        if(callback){
            callback(messages, prompt);
        }else{
            this.input = prompt + this.input;
        }
    }

    toJSON(): IRWSPromptJSON
    {
        return {
            input: this.input,            
            enhancedInput: this.enhancedInput,
            sentInput: this.sentInput,
            originalInput: this.originalInput,
            output: this.output,
            modelId: this.modelId,
            modelType: this.modelType,
            multiTemplate: this.multiTemplate,            
            convo: {
                id: this.convo.getId()
            },
            hyperParameters: this.hyperParameters,
            var_storage: this.varStorage,
            created_at: this.created_at.toISOString()
        };
    }
}

export default RWSPrompt;

export { 
    IPromptSender, 
    IPromptEnchantment, 
    IPromptParams, 
    IPromptHyperParameters, 
    IRWSPromptRequestExecutor, 
    IRWSPromptStreamExecutor, 
    IRWSSinglePromptRequestExecutor, 
    IRWSPromptJSON, 
    IChainCallOutput, 
    ChainStreamType, 
    ILLMChunk 
};