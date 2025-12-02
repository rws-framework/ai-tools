import { Injectable } from '@nestjs/common';
import { LangChainEmbeddingService } from './LangChainEmbeddingService';
import { OptimizedVectorSearchService } from './OptimizedVectorSearchService';
import { Document } from '@langchain/core/documents';
import fs from 'fs';
import path from 'path';
import { rwsPath } from '@rws-framework/console';
import { 
    IEmbeddingConfig, 
    IChunkConfig, 
    IVectorStoreConfig,
    ISearchResult,
    IVectorSearchRequest,
    ILangChainRAGConfig,
    IRAGIndexRequest,
    IRAGSearchRequest,
    IRAGResponse,
    IRAGStats
} from '../types';

// Re-export types for convenience
export { 
    IEmbeddingConfig, 
    IChunkConfig, 
    IVectorStoreConfig,
    ISearchResult,
    IVectorSearchRequest,
    ILangChainRAGConfig,
    IRAGIndexRequest,
    IRAGSearchRequest,
    IRAGResponse,
    IRAGStats
} from '../types';

/**
 * Core LangChain-based RAG service with optimized per-knowledge vector storage
 * This service provides the main abstraction for RAG operations using LangChain
 * Uses per-knowledge vector files for lightning-fast searches
 */
@Injectable()
export class LangChainRAGService {
    private config: ILangChainRAGConfig;
    private isInitialized = false;
    private queryEmbeddingCache = new Map<string, number[]>();
    private maxCacheSize = 100;
    private logger?: any; // Optional logger interface

    constructor(
        private embeddingService: LangChainEmbeddingService,
        private vectorSearchService: OptimizedVectorSearchService
    ) {}

    /**
     * Initialize the RAG service with configuration
     */
    async initialize(config?: ILangChainRAGConfig, logger?: any): Promise<void> {
        if (this.isInitialized) {
            this.log('debug', 'RAG service already initialized, skipping...');
            return;
        }

        if (config) {
            this.config = {
                persistence: { enabled: false, autoSave: true },
                ...config
            };
        }
        
        if (logger) {
            this.logger = logger;
        }

        this.log('log', 'Starting LangChain RAG service initialization...');

        try {               
            this.isInitialized = true;
            this.log('log', 'LangChain RAG service initialized successfully');
        } catch (error) {
            this.log('error', 'Failed to initialize LangChain RAG service:', error);
            throw error;
        }
    }

    /**
     * Index knowledge content for RAG with optimized per-knowledge vector storage
     */
    async indexKnowledge(
        knowledgeId: string | number,
        content: string,
        metadata: Record<string, any> = {}
    ): Promise<IRAGResponse<{ chunkIds: string[] }>> {
        this.log('log', `[INDEXING] Starting indexKnowledge for knowledgeId: ${knowledgeId}`);
        this.log('debug', `[INDEXING] Content length: ${content.length} characters`);

        try {
            await this.ensureInitialized();

            // Chunk the content using the embedding service
            const chunks = await this.embeddingService.chunkText(content);
            this.log('debug', `[INDEXING] Split content into ${chunks.length} chunks for knowledge ${knowledgeId}`);

            // Generate embeddings for all chunks at once (batch processing for speed)
            const embeddings = await this.embeddingService.embedTexts(chunks);
            this.log('debug', `[INDEXING] Generated embeddings for ${chunks.length} chunks`);

            // Create chunk objects with embeddings
            const chunksWithEmbeddings = chunks.map((chunkContent, index) => ({
                content: chunkContent,
                embedding: embeddings[index],
                metadata: {
                    ...metadata,
                    knowledgeId,
                    chunkIndex: index,
                    id: `knowledge_${knowledgeId}_chunk_${index}`
                }
            }));

            // Save to per-knowledge vector file
            await this.saveKnowledgeVector(knowledgeId, chunksWithEmbeddings);

            const chunkIds = chunksWithEmbeddings.map(chunk => chunk.metadata.id);
            this.log('log', `[INDEXING] Successfully indexed knowledge ${knowledgeId} with ${chunkIds.length} chunks using optimized approach`);

            return {
                success: true,
                data: { chunkIds }
            };

        } catch (error: any) {
            this.log('error', `[INDEXING] Failed to index knowledge ${knowledgeId}:`, error);
            return {
                success: false,
                data: null,
                error: error.message || 'Unknown error'
            };
        }
    }

    /**
     * Search for relevant knowledge chunks using optimized vector search
     */
    async searchKnowledge(request: IRAGSearchRequest): Promise<IRAGResponse<{ results: ISearchResult[] }>> {
        this.log('log', `[SEARCH] Starting knowledge search for query: "${request.query}"`);
        this.log('debug', `[SEARCH] Search parameters: maxResults=${request.maxResults || 5}, threshold=${request.threshold || 0.3}, temporaryDocumentSearch=${request.temporaryDocumentSearch}`);

        try {
            await this.ensureInitialized();

            const knowledgeIds = request.filter?.knowledgeIds || [];
            console.log('knowledgeIds', knowledgeIds, 'temporaryDocumentSearch:', request.temporaryDocumentSearch);
            
            if (knowledgeIds.length === 0) {
                this.log('warn', '[SEARCH] No knowledge IDs provided for search, returning empty results');
                return {
                    success: true,
                    data: { results: [] }
                };
            }

            // Load all knowledge vectors in parallel (including temporary documents)
            const knowledgeVectorPromises = knowledgeIds.map(async (knowledgeId) => {
                try {
                    const vectorData = await this.loadKnowledgeVectorWithEmbeddings(knowledgeId);
                    return {
                        knowledgeId,                    
                        chunks: vectorData.chunks
                    };
                } catch (loadError) {
                    this.log('warn', `[SEARCH] Failed to load knowledge ${knowledgeId}:`, loadError);
                    return null;
                }
            });

            const knowledgeVectors = (await Promise.all(knowledgeVectorPromises)).filter(v => v !== null);
            
            if (knowledgeVectors.length === 0) {
                this.log('warn', '[SEARCH] No knowledge vectors could be loaded for search');
                return {
                    success: true,
                    data: { results: [] }
                };
            }

            // Use optimized vector search service
            const searchResponse = await this.vectorSearchService.searchSimilar({
                query: request.query,
                knowledgeVectors,
                maxResults: request.maxResults || 5,
                threshold: request.threshold || 0.1  // Use same default as PromptEnhancementService
            });
            
            // Convert results to expected format
            const results: ISearchResult[] = searchResponse.results.map(result => ({
                knowledgeId: result.metadata.knowledgeId,
                content: result.content,
                score: result.score,
                metadata: result.metadata,
                chunkId: result.chunkId,                
            }));

            this.log('log', `[SEARCH] Found ${results.length} relevant chunks for query: "${request.query}"\n`);

            return {
                success: true,
                data: { results }
            };

        } catch (error: any) {
            this.log('error', '[SEARCH] Failed to search knowledge:', error);
            return {
                success: false,
                data: null,
                error: error.message || 'Unknown error'
            };
        }
    }

    /**
     * Remove knowledge from index
     */
    async removeKnowledge(fileId: string | number): Promise<boolean> {
        this.log('log', `[REMOVE] Starting removal of knowledge: ${fileId}`);

        try {
            await this.ensureInitialized();

            // Remove the individual knowledge vector file
            const vectorFilePath = this.getKnowledgeVectorPath(fileId);
            if (fs.existsSync(vectorFilePath)) {
                fs.unlinkSync(vectorFilePath);
                this.log('log', `[REMOVE] Successfully removed vector file for knowledge ${fileId}`);
                return true;
            } else {
                this.log('warn', `[REMOVE] Vector file not found for knowledge ${fileId}`);
                return true; // Consider it successful if file doesn't exist
            }

        } catch (error: any) {
            this.log('error', `[REMOVE] Failed to remove knowledge ${fileId}:`, error);
            return false;
        }
    }

    /**
     * Get statistics about the RAG system
     */
    getStats(): IRAGStats {
        try {
            const vectorDir = path.join(rwsPath.findRootWorkspacePath(), 'files', 'vectors', 'knowledge');
            
            if (!fs.existsSync(vectorDir)) {
                return {
                    totalDocuments: 0,
                    totalChunks: 0,
                    knowledgeItems: 0
                };
            }

            const files = fs.readdirSync(vectorDir).filter(f => f.endsWith('.json'));
            let totalChunks = 0;

            for (const file of files) {
                try {
                    const filePath = path.join(vectorDir, file);
                    const data = JSON.parse(fs.readFileSync(filePath, 'utf8'));
                    totalChunks += data.chunks?.length || 0;
                } catch (error) {
                    this.log('warn', `[STATS] Failed to read vector file ${file}:`, error);
                }
            }

            this.log('debug', `[STATS] RAG system contains ${totalChunks} chunks across ${files.length} knowledge items`);

            return {
                totalChunks,
                totalDocuments: files.length,
                knowledgeItems: files.length
            };

        } catch (error: any) {
            this.log('error', '[STATS] Failed to get RAG statistics:', error);
            return {
                totalDocuments: 0,
                totalChunks: 0,
                knowledgeItems: 0
            };
        }
    }

    /**
     * Clear all indexed knowledge
     */
    async clearAll(): Promise<boolean> {
        try {
            const vectorDir = path.join(rwsPath.findRootWorkspacePath(), 'files', 'vectors', 'knowledge');
            if (fs.existsSync(vectorDir)) {
                const files = fs.readdirSync(vectorDir).filter(f => f.endsWith('.json'));
                for (const file of files) {
                    fs.unlinkSync(path.join(vectorDir, file));
                }
                this.log('log', `[CLEAR] Successfully cleared ${files.length} vector files`);
            }
            
            this.log('debug', 'Cleared all indexed knowledge');
            return true;
        } catch (error: any) {
            this.log('error', 'Failed to clear knowledge:', error);
            return false;
        }
    }

    /**
     * Get embeddings for a text query
     */
    async getQueryEmbedding(query: string): Promise<number[]> {
        await this.ensureInitialized();
        return await this.embeddingService.embedText(query);
    }

    /**
     * Save chunks to knowledge-specific vector file with embeddings
     */
    private async saveKnowledgeVector(knowledgeId: string | number, chunks: Array<{ content: string; embedding: number[]; metadata: any }>): Promise<void> {
        const vectorFilePath = this.getKnowledgeVectorPath(knowledgeId);
        const vectorDir = path.dirname(vectorFilePath);

        // Ensure directory exists
        if (!fs.existsSync(vectorDir)) {
            fs.mkdirSync(vectorDir, { recursive: true });
        }

        try {
            const vectorData = {
                knowledgeId,
                chunks,
                timestamp: new Date().toISOString()
            };

            fs.writeFileSync(vectorFilePath, JSON.stringify(vectorData, null, 2));
            this.log('debug', `[SAVE] Successfully saved ${chunks.length} chunks with embeddings for knowledge ${knowledgeId}`);

        } catch (error) {
            this.log('error', `[SAVE] Failed to save vector data for knowledge ${knowledgeId}:`, error);
            throw error;
        }
    }

    /**
     * Load vector data for a specific knowledge item with embeddings
     */
    private async loadKnowledgeVectorWithEmbeddings(knowledgeId: string | number): Promise<{ knowledgeId?: string | number, chunks: Array<{ content: string; embedding: number[]; metadata: any }> }> {
        const vectorFilePath = this.getKnowledgeVectorPath(knowledgeId);
        
        if (!fs.existsSync(vectorFilePath)) {
            this.log('debug', `[LOAD] No vector file found for knowledge ${knowledgeId}, skipping...`);
            return { chunks: [] };
        }

        try {
            this.log('debug', `[LOAD] Loading vector data with embeddings for knowledge ${knowledgeId} from ${vectorFilePath}`);
            const vectorData = JSON.parse(fs.readFileSync(vectorFilePath, 'utf8'));
            
            return {
                chunks: vectorData.chunks || [],
                knowledgeId
            };
        } catch (error) {
            this.log('error', `[LOAD] Failed to load vector data for knowledge ${knowledgeId}:`, error);
            return { chunks: [] };
        }
    }

    /**
     * Get the file path for a specific knowledge's vector data
     */
    private getKnowledgeVectorPath(knowledgeId: string | number): string {
        const vectorDir = path.join(rwsPath.findRootWorkspacePath(), 'files', 'vectors', 'knowledge');
        if (!fs.existsSync(vectorDir)) {
            fs.mkdirSync(vectorDir, { recursive: true });
        }
        return path.join(vectorDir, `knowledge_${knowledgeId}.json`);
    }

    /**
     * Ensure the service is initialized
     */
    private async ensureInitialized(): Promise<void> {
        if (!this.isInitialized) {
            this.log('debug', '[INIT] Service not initialized, triggering initialization...');
            await this.initialize();
        }
    }

    /**
     * Logging helper that uses provided logger or falls back to console
     */
    private log(level: 'debug' | 'log' | 'warn' | 'error', message: string, ...args: any[]): void {
        if (this.logger) {
            // Use provided logger (like BlackLogger)
            if (typeof this.logger[level] === 'function') {
                this.logger[level](message, ...args);
            } else if (typeof this.logger.log === 'function') {
                this.logger.log(message, ...args);
            }
        } else {
            // Fallback to console
            console[level === 'log' ? 'log' : level](message, ...args);
        }
    }
}
