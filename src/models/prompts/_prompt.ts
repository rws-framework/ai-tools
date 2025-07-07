import { Readable } from 'stream';
import { PromptTemplate } from '@langchain/core/prompts';
import { EmbedLoader, IChainCallOutput } from '../convo/EmbedLoader';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { IToolCall } from '../../types/IPrompt'
import { 
    IPromptSender, 
    IPromptEnchantment, 
    IPromptParams, 
    IPromptHyperParameters, 
    IRWSPromptRequestExecutor, 
    IRWSPromptStreamExecutor, 
    IRWSSinglePromptRequestExecutor, 
    IRWSPromptJSON, 
    ChainStreamType, 
    ILLMChunk,
    IAITool,
    IRWSHistoryMessage,
    InputType,
    CompoundInput,
    ToolHandler
} from '../../types/IPrompt';
import { IContextToken } from '../../types/IContextToken';
import { text } from 'stream/consumers';

type EntryParams = {
    modelId: string,
    body: string,
}


class RWSPrompt {
    public _stream: ChainStreamType;
    private input: CompoundInput[] = [];    
    private enhancedInput: IPromptEnchantment[] = [];
    private sentInput: CompoundInput[] = [];
    private originalInput: CompoundInput[] = [];
    private output: string = '';
    private modelId: string;
    private modelType: string;
    private multiTemplate: PromptTemplate;
    private embedLoader: EmbedLoader<any>;
    private hyperParameters: IPromptHyperParameters;
    private created_at: Date;
    private toolHandlers: Map<string, ToolHandler> = new Map();
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
    }

    getEnchantedInputs(): IPromptEnchantment[]
    {
        return this.enhancedInput;
    }

    getModelId(): string
    {
        return this.modelId;
    }

    readSentInput(): CompoundInput[]
    {
        return this.sentInput;
    }

    readInput(): CompoundInput[]
    {
        const enchantedInput: CompoundInput[] = this.enhancedInput.map(enchantment => ({ role: 'user', type:  enchantment.input.type, text: enchantment.input.text }));
        return [...enchantedInput, ...this.input];
    }

    
    readBaseInput(): CompoundInput[]
    {
        return this.originalInput;
    }    

    setBaseInput(input: CompoundInput[]): RWSPrompt
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

    setEmbedLoader(embedLoader: EmbedLoader<BaseChatModel>): RWSPrompt
    {
        this.embedLoader = embedLoader;        
        
        return this;
    }

    getEmbedLoader<T extends BaseChatModel>(): EmbedLoader<T>
    {
        return this.embedLoader;
    }

    replacePromptVar(key: string, val: string)
    {

    }

    getModelMetadata(): [string, string]
    {
        return [this.modelType, this.modelId];
    }

    async requestWith(executor: IRWSPromptRequestExecutor, intruderPrompt: string = null, debugVars: any = {}, tools?: IAITool[]): Promise<void>
    {
        this.sentInput = this.input;
        const returnedRWS = await executor.promptRequest(this, { intruderPrompt, debugVars, tools });
        this.output = returnedRWS.readOutput();        
    }

    async singleRequestWith(executor: IRWSSinglePromptRequestExecutor, intruderPrompt: string = null, ensureJson: boolean = false, tools?: IAITool[]): Promise<void>
    {        
        await executor.singlePromptRequest(this, { intruderPrompt, ensureJson, tools });
        this.sentInput = this.input;
    }

    async streamWith(executor: IRWSPromptStreamExecutor, read: (chunk: ILLMChunk) => void, end: () => void = () => {}, debugVars: any = {}, tools?: IAITool[]): Promise<RWSPrompt>
    {        
        this.sentInput = this.input;
        return executor.promptStream(this, read, end, { debugVars, tools });
    }

    addInput(content: CompoundInput): RWSPrompt
    {
        this.input.push(content);
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

    async readStreamAsText(readableStream: ReadableStream, callback: (txt: string) => void) {
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
            this.input =  [{type: 'text', text: prompt}, ...this.input];
        }
    }

    registerToolHandlers(toolHandlers: { [key: string]: ToolHandler }){
        for(const key of Object.keys(toolHandlers)){
            this.toolHandlers.set(key, toolHandlers[key]);
        }        
    }

    async callTools<T = unknown>(tools: IToolCall[]): Promise<T[]>
    {
        const results: T[] = [];
        for(const tool of tools){
            if(this.toolHandlers.has(tool.name)){
                const result = await this.callAiTool<T>(tool);
                if(result){
                    results.push(result);
                }                
            }
        }

        return results;
    }

    private async callAiTool<T>(tool: IToolCall): Promise<T>
    {
        const handler = this.toolHandlers.get(tool.name);
        return await handler(tool.arguments);
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
            embed: {
                id: this.embedLoader.getId()
            },
            hyperParameters: this.hyperParameters,
            var_storage: this.varStorage,
            created_at: this.created_at.toISOString()
        };
    }
}

export default RWSPrompt;

export { IChainCallOutput };
