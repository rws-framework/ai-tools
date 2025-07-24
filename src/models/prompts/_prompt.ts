import { Readable } from 'stream';
import { ModuleRef } from '@nestjs/core';
import { 
    IPromptParams,
    IRWSPromptRequestExecutor,
    IRWSPromptStreamExecutor,
    IRWSSinglePromptRequestExecutor,
    IRWSPromptJSON,
    ChainStreamType,
    ILLMChunk,
    IAITool,
    IRWSHistoryMessage,
    CompoundInput,
    ToolHandler,
    IToolCall,
    IPromptEnchantment,
    BaseChatModel,
    PromptTemplate,
    EmbedLoader
} from './inc/types';
import { IChainCallOutput } from './inc/types';
import { ToolManager } from './inc/tool-manager';
import { InputOutputManager } from './inc/input-output-manager';
import { ModelExecutionManager } from './inc/model-execution-manager';
import { VariableStorageManager } from './inc/variable-storage-manager';
import { ExecutionMethodsHandler, IPromptInstance } from './inc/execution-methods-handler';

class RWSPrompt implements IPromptInstance {
    public _stream: ChainStreamType;
    private created_at: Date;
    
    // Composition over inheritance - use managers for different concerns
    private toolManager: ToolManager;
    private ioManager: InputOutputManager;
    private modelManager: ModelExecutionManager;
    private varManager: VariableStorageManager;
    private executionHandler: ExecutionMethodsHandler;

    constructor(params: IPromptParams) {
        this.toolManager = new ToolManager();
        this.ioManager = new InputOutputManager(params.input);
        this.modelManager = new ModelExecutionManager(params.modelId, params.modelType, params.hyperParameters);
        this.varManager = new VariableStorageManager();
        this.executionHandler = new ExecutionMethodsHandler();
        
        this.created_at = new Date();
    }

    // Delegation methods for tool management
    setTools(tools: IAITool[]): RWSPrompt {
        this.toolManager.setTools(tools);
        return this;
    }

    getTools(): IAITool[] {
        return this.toolManager.getTools();
    }

    getTool(key: string): { definition: IAITool, handler: ToolHandler } | null {
        return this.toolManager.getTool(key);
    }

    registerToolHandlers(toolHandlers: { [key: string]: ToolHandler }): void {
        this.toolManager.registerToolHandlers(toolHandlers);
    }

    async callTools<T = unknown, O = unknown>(tools: IToolCall[], moduleRef: ModuleRef, aiToolOptions?: O): Promise<T[]> {
        return this.toolManager.callTools<T, O>(tools, moduleRef, aiToolOptions);
    }

    // Delegation methods for input/output management
    listen(source: string, stream: boolean = true): RWSPrompt {
        this.ioManager.listen(source, stream);
        return this;
    }

    setStreamCallback(callback: (chunk: string) => void): void {
        this.ioManager.setStreamCallback(callback);
    }

    addEnchantment(enchantment: IPromptEnchantment): void {
        this.ioManager.addEnchantment(enchantment);
    }

    getEnchantedInputs(): IPromptEnchantment[] {
        return this.ioManager.getEnchantedInputs();
    }

    readSentInput(): CompoundInput[] {
        return this.ioManager.readSentInput();
    }

    readInput(): CompoundInput[] {
        return this.ioManager.readInput();
    }

    readBaseInput(): CompoundInput[] {
        return this.ioManager.readBaseInput();
    }

    setBaseInput(input: CompoundInput[]): RWSPrompt {
        this.ioManager.setBaseInput(input);
        return this;
    }

    injestOutput(content: string): RWSPrompt {
        this.ioManager.injestOutput(content);
        return this;
    }

    readOutput(): string {
        return this.ioManager.readOutput();
    }

    addInput(content: CompoundInput): RWSPrompt {
        this.ioManager.addInput(content);
        return this;
    }

    addHistory(messages: IRWSHistoryMessage[], historyPrompt: string, callback?: (messages: IRWSHistoryMessage[], prompt: string) => void): void {
        this.ioManager.addHistory(messages, historyPrompt, callback);
    }

    async readStreamAsText(readableStream: ReadableStream, callback: (txt: string) => void): Promise<void> {
        return this.ioManager.readStreamAsText(readableStream, callback);
    }

    // Delegation methods for model management
    getModelId(): string {
        return this.modelManager.getModelId();
    }

    getModelMetadata(): [string, string] {
        return this.modelManager.getModelMetadata();
    }

    setMultiTemplate(template: PromptTemplate): RWSPrompt {
        this.modelManager.setMultiTemplate(template);
        return this;
    }

    getMultiTemplate(): PromptTemplate {
        return this.modelManager.getMultiTemplate();
    }

    setEmbedLoader(embedLoader: EmbedLoader<BaseChatModel>): RWSPrompt {
        this.modelManager.setEmbedLoader(embedLoader);
        return this;
    }

    getEmbedLoader<T extends BaseChatModel>(): EmbedLoader<T> {
        return this.modelManager.getEmbedLoader<T>();
    }

    getHyperParameters<T extends any>(base: any = null): T {
        return this.modelManager.getHyperParameters<T>(base);
    }

    getHyperParameter<T>(key: string): T {
        return this.modelManager.getHyperParameter<T>(key as any);
    }

    setHyperParameter(key: string, value: any): RWSPrompt {
        this.modelManager.setHyperParameter(key, value);
        return this;
    }

    setHyperParameters(value: any): RWSPrompt {
        this.modelManager.setHyperParameters(value);
        return this;
    }

    replacePromptVar(key: string, val: string): void {
        this.modelManager.replacePromptVar(key, val);
    }

    // Delegation methods for variable storage
    getVar<T>(key: string): T {
        return this.varManager.getVar<T>(key);
    }

    setVar<T>(key: string, val: T): RWSPrompt {
        this.varManager.setVar<T>(key, val);
        return this;
    }

    // Delegation methods for execution
    async requestWith(executor: IRWSPromptRequestExecutor, intruderPrompt: string = null, debugVars: any = {}, tools?: IAITool[]): Promise<void> {
        return this.executionHandler.requestWith(this, executor, intruderPrompt, debugVars, tools);
    }

    async singleRequestWith(executor: IRWSSinglePromptRequestExecutor, intruderPrompt: string = null, ensureJson: boolean = false, tools?: IAITool[]): Promise<void> {
        return this.executionHandler.singleRequestWith(this, executor, intruderPrompt, ensureJson, tools);
    }

    async streamWith(executor: IRWSPromptStreamExecutor, read: (chunk: ILLMChunk) => void, end: () => void = () => { }, debugVars: any = {}, tools?: IAITool[]): Promise<RWSPrompt> {
        return this.executionHandler.streamWith(this, executor, read, end, debugVars, tools);
    }

    // IPromptInstance interface implementation
    getInput(): CompoundInput[] {
        return this.ioManager.getInput();
    }

    setSentInput(input: CompoundInput[]): void {
        this.ioManager.setSentInput(input);
    }

    toJSON(): IRWSPromptJSON {
        return {
            input: this.ioManager.getInput(),
            enhancedInput: this.ioManager.getEnhancedInput(),
            sentInput: this.ioManager.readSentInput(),
            originalInput: this.ioManager.getOriginalInput(),
            output: this.ioManager.getOutput(),
            modelId: this.modelManager.getModelId(),
            modelType: this.modelManager.getModelType(),
            multiTemplate: this.modelManager.getMultiTemplate(),
            embed: {
                id: this.modelManager.getEmbedLoaderId()
            },
            hyperParameters: this.modelManager.getAllHyperParameters(),
            var_storage: this.varManager.getAllVars(),
            created_at: this.created_at.toISOString()
        };
    }
}

export default RWSPrompt;

export { IChainCallOutput };
