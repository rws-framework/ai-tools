// Re-export commonly used types from IPrompt
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
    IAITool,
    IRWSHistoryMessage,
    InputType,
    CompoundInput,
    ToolHandler,
    IToolCall
} from '../../../types/IPrompt';

export { EmbedLoader, IChainCallOutput } from '../../convo/EmbedLoader';
export { PromptTemplate } from '@langchain/core/prompts';
export { BaseChatModel } from '@langchain/core/language_models/chat_models';
export { ModuleRef } from '@nestjs/core';
export { IContextToken } from '../../../types/IContextToken';

import {
    CompoundInput,
    IPromptEnchantment,
    IPromptHyperParameters,
    ToolHandler,
    IAITool
} from '../../../types/IPrompt';
import { PromptTemplate } from '@langchain/core/prompts';
import { EmbedLoader } from '../../convo/EmbedLoader';

export type EntryParams = {
    modelId: string;
    body: string;
};

export interface IRWSPromptState {
    input: CompoundInput[];
    enhancedInput: IPromptEnchantment[];
    sentInput: CompoundInput[];
    originalInput: CompoundInput[];
    output: string;
    modelId: string;
    modelType: string;
    multiTemplate: PromptTemplate;
    embedLoader: EmbedLoader<any>;
    hyperParameters: IPromptHyperParameters;
    created_at: Date;
    toolHandlers: Map<string, ToolHandler>;
    varStorage: any;
    tools: IAITool[];
    onStream: (chunk: string) => void;
}
