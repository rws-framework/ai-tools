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
    private logger?: any; // Optional logger interface

    static SheetMimeType: string[] = [
        'text/csv',
        'text/tab-separated-values',
        'application/vnd.ms-excel',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        'application/vnd.openxmlformats-officedocument.spreadsheetml.template',
        'application/vnd.ms-excel.sheet.macroEnabled.12',
        'application/vnd.ms-excel.sheet.binary.macroEnabled.12',
        'application/vnd.oasis.opendocument.spreadsheet',
        'application/vnd.google-apps.spreadsheet',
    ];

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
        fileId: string | number,
        content: string | Record<string, any>[],
        metadata: Record<string, any> = {},
        batchCallback?: (fragments:string[], batch: number[][], percentage: number) => Promise<void>,
        ragOverride?: IChunkConfig
    ): Promise<IRAGResponse<{ chunkCount: number }>> {
        this.log('log', `[INDEXING] Starting indexKnowledge for fileId: ${fileId}`);
        this.log('debug', `[INDEXING] Content length: ${Array.isArray(content) ? content.map(r => Object.values(r).join(' ')).join('\n').length : content.length} characters`);

        try {
            await this.ensureInitialized();

            const mime = metadata.mime || null;

            let chunkTexts: string[] = undefined;
            let embeddings: number[][] = undefined;

            if(mime && LangChainRAGService.isSheetDocument(mime)) {       
                this.log('debug', `[INDEXING] SHEET extraction mode detected.`);
         
                const docs = await this.embeddingService.chunkCSV(content as Record<string, any>[], ragOverride);
                embeddings = await this.embeddingService.embedDocs(docs, batchCallback);
                chunkTexts = docs.map(d => d.pageContent);
            }else{                
                chunkTexts = await this.embeddingService.chunkText(content as string, ragOverride);
                embeddings = await this.embeddingService.embedTexts(chunkTexts, batchCallback);
            }
            
            this.log('debug', `[INDEXING] Generated embeddings for ${chunkTexts.length} chunks`);

       

            if(!batchCallback){
                // Create chunk objects with embeddings
                const chunksWithEmbeddings = chunkTexts.map((chunkContent, index) => ({
                    content: chunkContent,
                    embedding: embeddings[index],
                    metadata: {
                        ...metadata,
                        fileId,
                        chunkIndex: index,
                        id: `knowledge_${fileId}_chunk_${index}`
                    }
                }));
                await this.saveKnowledgeVector(fileId, chunksWithEmbeddings);
            }            

            this.log('log', `[INDEXING] Successfully indexed file ${fileId} with ${chunkTexts.length} chunks using optimized approach`);

            return {
                success: true,
                data: { chunkCount: chunkTexts.length }
            };

        } catch (error: any) {
            this.log('error', `[INDEXING] Failed to index file ${fileId}:`, error);
            return {
                success: false,
                data: null,
                error: error.message || 'Unknown error'
            };
        }
    }

    static isSheetDocument(mime: string): boolean {
        return LangChainRAGService.SheetMimeType.includes(mime);
    }

    /**
     * Search for relevant knowledge chunks using optimized vector search
     */
    async searchKnowledge(request: IRAGSearchRequest): Promise<IRAGResponse<{ results: ISearchResult[] }>> {
        this.log('log', `[SEARCH] Starting knowledge search for query: "${request.query}"`);
        this.log('debug', `[SEARCH] Search parameters: maxResults=${request.maxResults || 5}, threshold=${request.threshold || 0.3}, temporaryDocumentSearch=${request.temporaryDocumentSearch}`);

        try {
            await this.ensureInitialized();

            const fileIds = request.filter?.fileIds || [];
            console.log('fileIds', fileIds, 'temporaryDocumentSearch:', request.temporaryDocumentSearch);
            
            if (fileIds.length === 0) {
                this.log('warn', '[SEARCH] No file IDs provided for search, returning empty results');
                return {
                    success: true,
                    data: { results: [] }
                };
            }

            // Load all knowledge vectors in parallel (including temporary documents)
            const knowledgeVectorPromises = fileIds.map(async (fileId) => {
                try {
                    const vectorData = await this.loadKnowledgeVectorWithEmbeddings(fileId);
                    return {
                        fileId,                    
                        chunks: vectorData.chunks
                    };
                } catch (loadError) {
                    this.log('warn', `[SEARCH] Failed to load file ${fileId}:`, loadError);
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
                fileId: result.metadata?.fileId,  // Use fileId directly
                content: result.content,
                score: result.score,
                metadata: result.metadata,  // Pass metadata as-is
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
        this.log('log', `[REMOVE] Starting removal of file: ${fileId}`);

        try {
            await this.ensureInitialized();

            // Remove the individual knowledge vector file
            const vectorFilePath = this.getKnowledgeVectorPath(fileId);
            if (fs.existsSync(vectorFilePath)) {
                fs.unlinkSync(vectorFilePath);
                this.log('log', `[REMOVE] Successfully removed vector file for file ${fileId}`);
                return true;
            } else {
                this.log('warn', `[REMOVE] Vector file not found for file ${fileId}`);
                return true; // Consider it successful if file doesn't exist
            }

        } catch (error: any) {
            this.log('error', `[REMOVE] Failed to remove file ${fileId}:`, error);
            return false;
        }
    }

    embedQuery(query: string): Promise<number[]> {
        return this.vectorSearchService.getQueryEmbedding(query);
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
     * Uses streaming JSON write to handle large embedding datasets
     */
    private async saveKnowledgeVector(fileId: string | number, chunks: Array<{ content: string; embedding: number[]; metadata: any }>): Promise<void> {
        const vectorFilePath = this.getKnowledgeVectorPath(fileId);
        const vectorDir = path.dirname(vectorFilePath);

        // Ensure directory exists
        if (!fs.existsSync(vectorDir)) {
            fs.mkdirSync(vectorDir, { recursive: true });
        }

        try {
            // Stream JSON to avoid "Invalid string length" on large datasets
            const writeStream = fs.createWriteStream(vectorFilePath);
            
            await new Promise<void>((resolve, reject) => {
                writeStream.on('error', reject);
                writeStream.on('finish', resolve);
                
                writeStream.write(`{"fileId":${JSON.stringify(fileId)},"timestamp":${JSON.stringify(new Date().toISOString())},"chunks":[`);
                
                for (let i = 0; i < chunks.length; i++) {
                    if (i > 0) writeStream.write(',');
                    writeStream.write(JSON.stringify(chunks[i]));
                }
                
                writeStream.write(']}');
                writeStream.end();
            });
            
            this.log('debug', `[SAVE] Successfully saved ${chunks.length} chunks with embeddings for file ${fileId} to: "${vectorFilePath}"`);

        } catch (error) {
            this.log('error', `[SAVE] Failed to save vector data for file ${fileId}:`, error);
            throw error;
        }
    }

    /**
     * Load vector data for a specific knowledge item with embeddings
     */
    private async loadKnowledgeVectorWithEmbeddings(fileId: string | number): Promise<{ fileId?: string | number, chunks: Array<{ content: string; embedding: number[]; metadata: any }> }> {
        const vectorFilePath = this.getKnowledgeVectorPath(fileId);
        
        if (!fs.existsSync(vectorFilePath)) {
            this.log('debug', `[LOAD] No vector file found for file ${fileId}, skipping...`);
            return { chunks: [] };
        }

        try {
            this.log('debug', `[LOAD] Loading vector data with embeddings for file ${fileId} from ${vectorFilePath}`);
            const vectorData = JSON.parse(fs.readFileSync(vectorFilePath, 'utf8'));
            
            return {
                chunks: vectorData.chunks || [],
                fileId
            };
        } catch (error) {
            this.log('error', `[LOAD] Failed to load vector data for file ${fileId}:`, error);
            return { chunks: [] };
        }
    }

    /**
     * Get the file path for a specific knowledge's vector data
     */
    private getKnowledgeVectorPath(fileId: string | number): string {
        const vectorDir = path.join(rwsPath.findRootWorkspacePath(), 'files', 'vectors', 'knowledge');
        if (!fs.existsSync(vectorDir)) {
            fs.mkdirSync(vectorDir, { recursive: true });
        }
        return path.join(vectorDir, `knowledge_${fileId}.json`);
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
