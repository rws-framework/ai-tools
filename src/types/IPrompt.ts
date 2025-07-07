import { PromptTemplate } from '@langchain/core/prompts';
import { IterableReadableStream } from '@langchain/core/utils/stream';
import { ChainValues } from '@langchain/core/utils/types';
import { IContextToken } from './IContextToken';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';

// General tool interfaces for AI models
interface IAIToolParameterBase {
  type: string;
  description?: string;
  enum?: string[];
  required?: boolean;
}

interface IAIToolParameterObject extends IAIToolParameterBase {
  type: 'object';
  properties: Record<string, IAIToolParameter>;
}

interface IAIToolParameterArray extends IAIToolParameterBase {
  type: 'array';
  items: IAIToolParameter;
}

interface IAIToolParameterPrimitive extends IAIToolParameterBase {
  type: 'string' | 'number' | 'boolean' | 'integer';
}

type IAIToolParameter = IAIToolParameterObject | IAIToolParameterArray | IAIToolParameterPrimitive;

interface IAIToolSchema {
  type: 'object';
  properties: Record<string, IAIToolParameter>;
  required?: string[];
}

interface IToolCall {
    id: string;
    name: string;
    arguments: Record<string, any>;
  }

interface IAITool {
  name: string;
  description: string;
  input_schema?: IAIToolSchema;
}

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
    input: CompoundInput[];    
    modelId: string;
    modelType: string;
}

type InputType = 'text' | 'image';

interface IPromptEnchantment {
    enhancementId: string,
    enhancementName: string,
    enhancementParams: any,
    input: CompoundInput    
}

// Forward reference to RWSPrompt to avoid circular dependencies
type RWSPrompt = import('../models/prompts/_prompt').default;

type IPromptSender = (prompt: RWSPrompt) => Promise<void>;

interface IAIRequestOptions {
    contextToken?: IContextToken | null, 
    intruderPrompt?: string | null, 
    ensureJson?: boolean, 
    debugVars?: any,
    tools?: IAITool[]
}

interface IRWSPromptRequestExecutor {
    promptRequest: (prompt: RWSPrompt, aiOptions?: IAIRequestOptions) => Promise<RWSPrompt>
}

interface IRWSSinglePromptRequestExecutor {
    singlePromptRequest: (prompt: RWSPrompt, aiOptions?: IAIRequestOptions) => Promise<RWSPrompt>
}

interface IRWSPromptStreamExecutor {
    promptStream: (prompt: RWSPrompt, read: (chunk: ILLMChunk) => void, end: () => void, aiOptions?: IAIRequestOptions) => Promise<RWSPrompt>
}

type ToolHandler<T = any> = (...args: any) => Promise<T>;

interface IRWSPromptJSON {
    input: CompoundInput[];
    enhancedInput: IPromptEnchantment[];
    sentInput: CompoundInput[];
    originalInput: CompoundInput[];
    output: string;
    modelId: string;
    modelType: string;
    multiTemplate: PromptTemplate;
    embed?: { id: string };
    hyperParameters: IPromptHyperParameters;
    created_at: string;
    var_storage: any;
}

type ChainStreamType = AsyncGenerator<IterableReadableStream<ChainValues>>;

interface CompoundInput {
    type: InputType,
    text?: string,
    role?: string,
    source?: {
        type: string,
        media_type: string,
        data: string
    }
}

export { 
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
    IAIRequestOptions,
    IAITool,
    IAIToolSchema,
    IAIToolParameter,
    IAIToolParameterBase,
    IAIToolParameterObject,
    IAIToolParameterArray,
    IAIToolParameterPrimitive,
    IRWSHistoryMessage,
    InputType,
    CompoundInput,
    IToolCall,
    ToolHandler
};

