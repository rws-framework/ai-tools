/**
 * Search and result interfaces
 */
export interface ISearchResult {
    content: string;
    score: number;
    metadata: any;
    fileId: string | number;
    chunkId: string;
}

export interface IVectorSearchRequest {
    query: string;
    maxResults?: number;
    similarityThreshold?: number;
    filter?: {        
        documentIds?: string[];
        [key: string]: any;
    };
}

export interface IVectorSearchResponse {
    results: ISearchResult[];
    totalResults: number;
}

/**
 * Optimized search interfaces
 */
export interface IOptimizedSearchRequest {
    query: string;
    knowledgeVectors: Array<{
        fileId: string | number;
        chunks: Array<{
            content: string;
            embedding: number[];
            metadata: any;
        }>;
    }>;
    maxResults?: number;
    threshold?: number;
}

export interface IOptimizedSearchResult {
    content: string;
    score: number;
    metadata: any;
    fileId: string | number;
    chunkId: string;
}

export interface IOptimizedSearchResponse {
    results: IOptimizedSearchResult[];
    searchTime: number;
    totalCandidates: number;
}
