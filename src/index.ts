
import RWSPrompt, { IChainCallOutput } from '@rws-framework/ai-tools/src/models/prompts/_prompt';
import { ILLMChunk, IRWSPromptRequestExecutor, IRWSSinglePromptRequestExecutor, IRWSPromptStreamExecutor, IRWSPromptJSON, ChainStreamType, IAIRequestOptions, IAITool, IAIToolSchema, IAIToolParameter, IToolCall, ToolHandler } from './types/IPrompt';
import { EmbedLoader as RWSEmbed, IConvoDebugXMLData, IEmbeddingsHandler, ISplitterParams } from './models/convo/EmbedLoader';
import RWSVectorStore from './models/convo/VectorStore';
import { LangChainEmbeddingService } from './services/LangChainEmbeddingService';
import { LangChainVectorStoreService, IVectorStoreConfig, IDocumentChunk, IVectorSearchRequest, IVectorSearchResponse, ISearchResult } from './services/LangChainVectorStoreService';
import { LangChainRAGService, ILangChainRAGConfig, IRAGIndexRequest, IRAGSearchRequest, IRAGResponse, IRAGStats } from './services/LangChainRAGService';
import { IContextToken } from './types/IContextToken';
import { IEmbeddingConfig, IChunkConfig } from './types';
import type { IAiCfg } from './types/IAiCfg';

export {    
    IAiCfg,
    RWSVectorStore,
    RWSEmbed,
    RWSPrompt,
    ILLMChunk,
    IRWSPromptRequestExecutor,
    IRWSSinglePromptRequestExecutor,
    IRWSPromptStreamExecutor,
    IChainCallOutput,
    IRWSPromptJSON,
    ChainStreamType,
    IConvoDebugXMLData,
    IEmbeddingsHandler,
    ISplitterParams,
    IContextToken,
    IAIRequestOptions,
    IAITool,
    IAIToolSchema,
    IAIToolParameter,
    IToolCall,
    ToolHandler,
    // New LangChain-based services
    LangChainEmbeddingService,
    LangChainVectorStoreService, 
    LangChainRAGService,
    // Types
    IEmbeddingConfig,
    IChunkConfig,
    IVectorStoreConfig,
    IDocumentChunk,
    IVectorSearchRequest,
    IVectorSearchResponse,
    ISearchResult,
    ILangChainRAGConfig,
    IRAGIndexRequest,
    IRAGSearchRequest,
    IRAGResponse,
    IRAGStats
};
