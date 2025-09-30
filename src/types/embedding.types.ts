import { IRateLimitConfig } from "./rag.types";

/**
 * Embedding service configuration interfaces
 */
export interface IEmbeddingConfig {
    provider: 'cohere' | 'openai';
    apiKey: string;
    model?: string;
    batchSize?: number;
    rateLimiting?: IRateLimitConfig
}

export interface IChunkConfig {
    chunkSize?: number;
    chunkOverlap?: number;
    separators?: string[];
}
