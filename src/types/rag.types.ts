import { ISearchResult } from './search.types';

/**
 * RAG service configuration and request/response interfaces
 */
export interface ILangChainRAGConfig {
    embedding?: import('./embedding.types').IEmbeddingConfig;
    vectorStore: import('./vectorstore.types').IVectorStoreConfig;
    chunking?: import('./embedding.types').IChunkConfig;
    persistence?: {
        enabled: boolean;
        storagePath?: string;
        autoSave?: boolean;
    };
}

export interface IRAGIndexRequest {
    content: string;
    documentId: string | number;
    metadata?: Record<string, any>;
}

export interface IRAGSearchRequest {
    query: string;
    maxResults?: number;
    threshold?: number;
    filter?: {
        knowledgeIds?: (string | number)[];
        documentIds?: (string | number)[];
        [key: string]: any;
    };
}

export interface IRAGResponse<T = any> {
    success: boolean;
    data: T | null;
    error?: string;
}

export interface IRAGStats {
    totalChunks: number;
    totalDocuments: number;
    knowledgeItems: number;
}

export interface IRateLimitConfig {
    rpm?: number;           // Requests per minute
    tpm?: number;           // Tokens per minute  
    concurrency?: number;   // Parallel requests
    maxRetries?: number;    // Maximum retry attempts
    baseBackoffMs?: number; // Base backoff delay
    safetyFactor?: number;  // Safety factor for limits
}

export interface IBatchMetadata<T = any> {
    start: number;
    batch: T[];
}