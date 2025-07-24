import { 
    PromptTemplate, 
    EmbedLoader, 
    BaseChatModel, 
    IPromptHyperParameters,
    IRWSPromptRequestExecutor,
    IRWSSinglePromptRequestExecutor,
    IRWSPromptStreamExecutor,
    ILLMChunk,
    IAITool,
    ChainStreamType
} from './types';

export class ModelExecutionManager {
    private modelId: string;
    private modelType: string;
    private multiTemplate: PromptTemplate;
    private embedLoader: EmbedLoader<any>;
    private hyperParameters: IPromptHyperParameters;
    public _stream: ChainStreamType;

    constructor(modelId: string, modelType: string, hyperParameters: IPromptHyperParameters) {
        this.modelId = modelId;
        this.modelType = modelType;
        this.hyperParameters = hyperParameters;
    }

    getModelId(): string {
        return this.modelId;
    }

    getModelMetadata(): [string, string] {
        return [this.modelType, this.modelId];
    }

    setMultiTemplate(template: PromptTemplate): void {
        this.multiTemplate = template;
    }

    getMultiTemplate(): PromptTemplate {
        return this.multiTemplate;
    }

    setEmbedLoader(embedLoader: EmbedLoader<BaseChatModel>): void {
        this.embedLoader = embedLoader;
    }

    getEmbedLoader<T extends BaseChatModel>(): EmbedLoader<T> {
        return this.embedLoader;
    }

    getHyperParameters<T = any>(base: any = null): T {
        if (base) {
            this.hyperParameters = { ...base, ...this.hyperParameters };
        }

        return this.hyperParameters as T;
    }

    getHyperParameter<T>(key: string): T {
        if (!this.hyperParameters[key]) {
            return null;
        }

        return this.hyperParameters[key] as T;
    }

    setHyperParameter(key: string, value: any): void {
        this.hyperParameters[key] = value;
    }

    setHyperParameters(value: any): void {
        this.hyperParameters = value;
    }

    replacePromptVar(key: string, val: string): void {
        // Implementation placeholder
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

    // Getters for state access
    getModelType(): string {
        return this.modelType;
    }

    getEmbedLoaderId(): string {
        return this.embedLoader?.getId() || null;
    }

    getAllHyperParameters(): IPromptHyperParameters {
        return this.hyperParameters;
    }
}
