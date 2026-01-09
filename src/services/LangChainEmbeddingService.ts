import { Injectable } from '@nestjs/common';
import { Embeddings } from '@langchain/core/embeddings';
import { CohereEmbeddings } from '@langchain/cohere';
import { OpenAIEmbeddings } from '@langchain/openai';
import { Document } from '@langchain/core/documents';
import { IEmbeddingConfig, IChunkConfig } from '../types';
import { TextChunker } from './TextChunker';
import RWSVectorStore, { VectorDocType, IVectorStoreConfig } from '../models/convo/VectorStore';
import { OpenAIRateLimitingService } from './OpenAIRateLimitingService';

@Injectable()
export class LangChainEmbeddingService {
    private embeddings: Embeddings;
    private config: IEmbeddingConfig;
    private chunkConfig: IChunkConfig;
    private isInitialized = false;
    private vectorStore: RWSVectorStore | null = null;
    private static embeddingsPool = new Map<string, Embeddings>(); // Connection pooling

    constructor(private rateLimitingService: OpenAIRateLimitingService) {}

    /**
     * Initialize the service with configuration
     */
    async initialize(config: IEmbeddingConfig, chunkConfig?: IChunkConfig): Promise<void> {
        if (this.isInitialized) {
            return;
        }

        this.config = config;
        this.chunkConfig = chunkConfig || {
            chunkSize: 1000,
            chunkOverlap: 200
        };
        this.initializeEmbeddings();
        this.isInitialized = true;
    }


    private initializeEmbeddings(): void {
        const poolKey = `${this.config.provider}_${this.config.model}_${this.config.apiKey.slice(-8)}`;
        
        // Check connection pool first
        if (LangChainEmbeddingService.embeddingsPool.has(poolKey)) {
            this.embeddings = LangChainEmbeddingService.embeddingsPool.get(poolKey)!;
            return;
        }
        
        switch (this.config.provider) {
            case 'cohere':
                this.embeddings = new CohereEmbeddings({
                    apiKey: this.config.apiKey,
                    model: this.config.model || 'embed-v4.0',
                    batchSize: this.config.batchSize || 96
                });
                break;

            case 'openai':
                this.embeddings = new OpenAIEmbeddings({
                    apiKey: this.config.apiKey,
                    model: this.config.model || 'text-embedding-3-large',
                    batchSize: 1 // We'll handle batching ourselves
                });                                
                        
                break;    
                
            default:
                throw new Error(`Unsupported embedding provider: ${this.config.provider}`);
        }
        
        // Store in connection pool for reuse
        LangChainEmbeddingService.embeddingsPool.set(poolKey, this.embeddings);

        if(this.config.rateLimiting){
            const rateLimitingCfg = {...OpenAIRateLimitingService.DEFAULT_CONFIG, ...this.config.rateLimiting};

            this.rateLimitingService.initialize(this.config.model || 'text-embedding-3-large', rateLimitingCfg);
        }     
    }

    private initializeTextSplitter(chunkConfig?: IChunkConfig): void {
        // Text chunking is now handled by TextChunker class
        // This method is kept for compatibility but doesn't initialize anything
    }

        /**
     * Generate embeddings for multiple texts with sophisticated rate limiting
     */
    async embedTexts(texts: string[]): Promise<number[][]> {
        this.ensureInitialized();
        
        if (this.config.rateLimiting) {
            return await this.rateLimitingService.executeWithRateLimit(
                texts,
                async (batch: string[]) => {
                    return await this.embeddings.embedDocuments(batch);
                },
                (text: string) => text // Token extractor
            );
        }
        
        // For other providers (like Cohere), use the standard approach
        return await this.embeddings.embedDocuments(texts);
    }

    /**
     * Generate embedding for a single text with rate limiting
     */
    async embedText(text: string): Promise<number[]> {
        this.ensureInitialized();
        
        if (this.config.rateLimiting) {
            
            const results = await this.rateLimitingService.executeWithRateLimit(
                [text],
                async (batch: string[]) => {
                    return await this.embeddings.embedDocuments(batch);
                },
                (text: string) => text
            );
            return results[0];
        }
        
        return await this.embeddings.embedQuery(text);
    }

    /**
     * Split text into chunks
     */
    async chunkText(text: string): Promise<string[]> {
        this.ensureInitialized();
        
        // Use our custom TextChunker instead of LangChain's splitter
        // Use safe token limits - the TextChunker handles token estimation internally
        const maxTokens = this.chunkConfig?.chunkSize || 450; // Safe token limit for embedding models
        const overlap = this.chunkConfig?.chunkOverlap || 50; // Character overlap, not token
        
        return TextChunker.chunkText(text, maxTokens, overlap);
    }

    /**
     * Split text and generate embeddings for chunks
     */
    async chunkAndEmbed(text: string): Promise<{ text: string; embedding: number[] }[]> {
        this.ensureInitialized();
        const chunks = await this.chunkText(text);
        const embeddings = await this.embedTexts(chunks);
        
        return chunks.map((chunk, index) => ({
            text: chunk,
            embedding: embeddings[index]
        }));
    }

    /**
     * Create LangChain documents from text with metadata
     */
    async createDocuments(text: string, metadata: Record<string, any> = {}): Promise<Document[]> {
        this.ensureInitialized();
        const chunks = await this.chunkText(text);
        
        return chunks.map((chunk, index) => new Document({
            pageContent: chunk,
            metadata: {
                ...metadata,
                chunkIndex: index,
                id: `${metadata.documentId || 'doc'}_chunk_${index}`
            }
        }));
    }

    /**
     * Get the underlying LangChain embeddings instance
     */
    getEmbeddingsInstance(): Embeddings {
        this.ensureInitialized();
        return this.embeddings;
    }

    /**
     * Update configuration and reinitialize
     */
    updateConfig(newConfig: Partial<IEmbeddingConfig>): void {
        this.config = { ...this.config, ...newConfig };
        this.initializeEmbeddings();
    }

    /**
     * Calculate cosine similarity between two vectors
     */
    cosineSimilarity(vecA: number[], vecB: number[]): number {
        if (vecA.length !== vecB.length) {
            throw new Error('Vectors must have the same length');
        }

        let dotProduct = 0;
        let normA = 0;
        let normB = 0;

        for (let i = 0; i < vecA.length; i++) {
            dotProduct += vecA[i] * vecB[i];
            normA += vecA[i] * vecA[i];
            normB += vecB[i] * vecB[i];
        }

        return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    /**
     * Ensure the service is initialized
     */
    private ensureInitialized(): void {
        if (!this.isInitialized) {
            throw new Error('LangChainEmbeddingService not initialized. Call initialize() first.');
        }
    }

    /**
     * Create a vector store for similarity search like the LangChain tutorial
     * This allows us to use vectorStore.similaritySearch() just like in the tutorial
     */
    async createVectorStore(documents: Document[], config?: { type?: 'memory' | 'faiss'; persistPath?: string }): Promise<RWSVectorStore> {
        this.ensureInitialized();
        
        const vectorStoreConfig = {
            type: config?.type || 'memory' as const,
            persistPath: config?.persistPath
        };
        
        const vectorStore = await new RWSVectorStore(
            documents, 
            this.embeddings, 
            vectorStoreConfig
        ).init();
        
        return vectorStore;
    }

    /**
     * Perform similarity search on a vector store (tutorial-style interface)
     * Usage: const results = await embeddingService.similaritySearch(vectorStore, query, k)
     */
    async similaritySearch(vectorStore: RWSVectorStore, query: string, k: number = 4): Promise<Document[]> {
        this.ensureInitialized();
        
        // Use RWSVectorStore's similarity search (returns documents without scores)
        const resultsWithScores = await vectorStore.similaritySearchWithScore(query, k);
        return resultsWithScores.map(([doc, _score]) => doc);
    }

    /**
     * Perform similarity search with scores (tutorial-style interface)
     * Usage: const results = await embeddingService.similaritySearchWithScore(vectorStore, query, k)
     */
    async similaritySearchWithScore(vectorStore: RWSVectorStore, query: string, k: number = 4): Promise<[Document, number][]> {
        this.ensureInitialized();
        
        return await vectorStore.similaritySearchWithScore(query, k);
    }
}
