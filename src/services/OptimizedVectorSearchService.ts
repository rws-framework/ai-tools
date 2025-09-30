import { Injectable } from '@nestjs/common';
import { LangChainEmbeddingService } from './LangChainEmbeddingService';
import { 
    IOptimizedSearchRequest, 
    IOptimizedSearchResult, 
    IOptimizedSearchResponse,
    IVectorSearchRequest,
    IVectorSearchResponse,
    ISearchResult
} from '../types';

/**
 * Optimized vector search service for lightning-fast similarity searches
 * Uses pre-computed embeddings and direct cosine similarity calculations
 */
@Injectable()
export class OptimizedVectorSearchService {
    private queryEmbeddingCache = new Map<string, number[]>();
    private maxCacheSize = 100;

    constructor(private embeddingService: LangChainEmbeddingService) {}

    /**
     * Perform optimized similarity search across pre-computed vectors
     */
    async searchSimilar(request: IOptimizedSearchRequest): Promise<IOptimizedSearchResponse> {
        const startTime = Date.now();
        const { query, knowledgeVectors, maxResults = 5, threshold = 0.1 } = request;

        // Get or compute query embedding
        const queryEmbedding = await this.getQueryEmbedding(query);

        // Collect all candidates with parallel processing
        const allCandidates: IOptimizedSearchResult[] = [];
        let totalCandidates = 0;

        // Process all knowledge vectors in parallel
        const searchPromises = knowledgeVectors.map(async (knowledgeVector) => {
            const candidates: IOptimizedSearchResult[] = [];
            const similarities: number[] = [];  // Track all similarities for debugging
            
            for (const chunk of knowledgeVector.chunks) {
                totalCandidates++;
                
                if (!chunk.embedding || !Array.isArray(chunk.embedding)) {
                    continue;
                }

                // Compute cosine similarity
                const similarity = this.embeddingService.cosineSimilarity(queryEmbedding, chunk.embedding);
                similarities.push(similarity);
                
                if (similarity >= threshold) {
                    candidates.push({
                        content: chunk.content,
                        score: similarity,
                        metadata: chunk.metadata,
                        knowledgeId: knowledgeVector.knowledgeId,
                        chunkId: chunk.metadata?.id || `${knowledgeVector.knowledgeId}_chunk_${Date.now()}`
                    });
                }
            }
            
            // Log similarity statistics for debugging
            if (similarities.length > 0) {
                const maxSim = Math.max(...similarities);
                const avgSim = similarities.reduce((a, b) => a + b, 0) / similarities.length;
                console.log(`[VECTOR SEARCH] Knowledge ${knowledgeVector.knowledgeId}: Max similarity: ${maxSim.toFixed(4)}, Avg: ${avgSim.toFixed(4)}, Candidates above ${threshold}: ${candidates.length}`);
            }
            
            return candidates;
        });

        // Wait for all searches to complete
        const allCandidateArrays = await Promise.all(searchPromises);
        
        // Flatten results
        for (const candidates of allCandidateArrays) {
            allCandidates.push(...candidates);
        }

        // Sort by similarity score and take top results
        const results = allCandidates
            .sort((a, b) => b.score - a.score)
            .slice(0, maxResults);

        const searchTime = Date.now() - startTime;

        return {
            results,
            searchTime,
            totalCandidates
        };
    }

    /**
     * Get query embedding with caching
     */
    private async getQueryEmbedding(query: string): Promise<number[]> {
        // Check cache first
        if (this.queryEmbeddingCache.has(query)) {
            return this.queryEmbeddingCache.get(query)!;
        }

        // Generate embedding
        const embedding = await this.embeddingService.embedText(query);

        // Cache the embedding (with size limit)
        if (this.queryEmbeddingCache.size >= this.maxCacheSize) {
            // Remove oldest entry
            const firstKey = this.queryEmbeddingCache.keys().next().value;
            this.queryEmbeddingCache.delete(firstKey);
        }
        
        this.queryEmbeddingCache.set(query, embedding);
        return embedding;
    }

    /**
     * Batch search multiple queries efficiently
     */
    async batchSearch(
        queries: string[],
        knowledgeVectors: Array<{
            knowledgeId: string | number;
            chunks: Array<{
                content: string;
                embedding: number[];
                metadata: any;
            }>;
        }>,
        maxResults = 5,
        threshold = 0.1  // Updated to match other defaults
    ): Promise<Map<string, IOptimizedSearchResponse>> {
        const results = new Map<string, IOptimizedSearchResponse>();

        // Generate embeddings for all queries in batch
        const queryEmbeddings = await this.embeddingService.embedTexts(queries);

        // Process each query
        for (let i = 0; i < queries.length; i++) {
            const query = queries[i];
            const queryEmbedding = queryEmbeddings[i];

            // Cache the embedding
            this.queryEmbeddingCache.set(query, queryEmbedding);

            // Perform search with pre-computed embedding
            const response = await this.searchWithEmbedding({
                queryEmbedding,
                knowledgeVectors,
                maxResults,
                threshold
            });

            results.set(query, response);
        }

        return results;
    }

    /**
     * Search with pre-computed query embedding
     */
    private async searchWithEmbedding(request: {
        queryEmbedding: number[];
        knowledgeVectors: Array<{
            knowledgeId: string | number;
            chunks: Array<{
                content: string;
                embedding: number[];
                metadata: any;
            }>;
        }>;
        maxResults: number;
        threshold: number;
    }): Promise<IOptimizedSearchResponse> {
        const startTime = Date.now();
        const { queryEmbedding, knowledgeVectors, maxResults, threshold } = request;

        const allCandidates: IOptimizedSearchResult[] = [];
        let totalCandidates = 0;

        // Process all knowledge vectors in parallel
        const searchPromises = knowledgeVectors.map(async (knowledgeVector) => {
            const candidates: IOptimizedSearchResult[] = [];
            
            for (const chunk of knowledgeVector.chunks) {
                totalCandidates++;
                
                if (!chunk.embedding || !Array.isArray(chunk.embedding)) {
                    continue;
                }

                // Compute cosine similarity
                const similarity = this.embeddingService.cosineSimilarity(queryEmbedding, chunk.embedding);
                
                if (similarity >= threshold) {
                    candidates.push({
                        content: chunk.content,
                        score: similarity,
                        metadata: chunk.metadata,
                        knowledgeId: knowledgeVector.knowledgeId,
                        chunkId: chunk.metadata?.id || `${knowledgeVector.knowledgeId}_chunk_${Date.now()}`
                    });
                }
            }
            
            return candidates;
        });

        // Wait for all searches to complete
        const allCandidateArrays = await Promise.all(searchPromises);
        
        // Flatten results
        for (const candidates of allCandidateArrays) {
            allCandidates.push(...candidates);
        }

        // Sort by similarity score and take top results
        const results = allCandidates
            .sort((a, b) => b.score - a.score)
            .slice(0, maxResults);

        const searchTime = Date.now() - startTime;

        return {
            results,
            searchTime,
            totalCandidates
        };
    }

    /**
     * Clear query embedding cache
     */
    clearCache(): void {
        this.queryEmbeddingCache.clear();
    }

    /**
     * Get cache statistics
     */
    getCacheStats(): { size: number; maxSize: number } {
        return {
            size: this.queryEmbeddingCache.size,
            maxSize: this.maxCacheSize
        };
    }

    /**
     * Search similar documents (compatibility method from LangChainVectorStoreService)
     */
    async searchSimilarCompat(request: IVectorSearchRequest, knowledgeVectors: Array<{
        knowledgeId: string | number;
        chunks: Array<{
            content: string;
            embedding: number[];
            metadata: any;
        }>;
    }>): Promise<IVectorSearchResponse> {
        try {
            const {
                query,
                maxResults = 10,
                similarityThreshold = 0.1,  // Updated to match other defaults
                filter,
            } = request;

            // Filter knowledge vectors if needed
            let filteredVectors = knowledgeVectors;
            if (filter) {
                filteredVectors = knowledgeVectors.filter(vector => {
                    // Check knowledge IDs
                    if (filter.knowledgeIds && filter.knowledgeIds.length > 0) {
                        return filter.knowledgeIds.includes(String(vector.knowledgeId));
                    }
                    return true;
                });
            }

            // Use our optimized search
            const searchResponse = await this.searchSimilar({
                query,
                knowledgeVectors: filteredVectors,
                maxResults,
                threshold: similarityThreshold
            });

            // Convert to IVectorSearchResponse format
            const results: ISearchResult[] = searchResponse.results.map(result => ({
                content: result.content,
                score: result.score,
                metadata: result.metadata,
                chunkId: result.chunkId,
                knowledgeId: result.knowledgeId
            }));

            return {
                results,
                totalResults: results.length,
            };
        } catch (error) {
            console.error('Failed to search similar documents:', error);
            return {
                results: [],
                totalResults: 0,
            };
        }
    }

    /**
     * Get search statistics
     */
    getStats(knowledgeVectors: Array<{
        knowledgeId: string | number;
        chunks: Array<{ content: string; embedding: number[]; metadata: any; }>;
    }>): { totalChunks: number; totalKnowledge: number } {
        const totalChunks = knowledgeVectors.reduce((total, vector) => total + vector.chunks.length, 0);
        return {
            totalChunks,
            totalKnowledge: knowledgeVectors.length,
        };
    }
}
