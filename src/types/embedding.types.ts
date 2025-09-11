/**
 * Embedding service configuration interfaces
 */
export interface IEmbeddingConfig {
    provider: 'cohere';
    apiKey: string;
    model?: string;
    batchSize?: number;
}

export interface IChunkConfig {
    chunkSize?: number;
    chunkOverlap?: number;
    separators?: string[];
}
