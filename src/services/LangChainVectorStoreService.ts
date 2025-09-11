import { Injectable } from '@rws-framework/server/nest';
import { Document } from '@langchain/core/documents';
import { LangChainEmbeddingService } from './LangChainEmbeddingService';
import RWSVectorStore, { VectorDocType, IVectorStoreConfig as RWSVectorStoreConfig } from '../models/convo/VectorStore';

export interface ISearchResult {
    content: string;
    score: number;
    metadata: any;
    chunkId: string;
}

export interface IVectorStoreConfig {
    type: 'memory' | 'faiss';
    maxResults?: number;
    autoSave?: boolean;
    similarityThreshold?: number;
    persistPath?: string;
}

export interface IDocumentChunk {
    id: string;
    content: string;
    embedding?: number[];
    metadata?: {
        documentId: string;
        chunkIndex: number;
        source?: string;
        title?: string;
        knowledgeId?: string;
        [key: string]: any;
    };
}

export interface IVectorSearchRequest {
    query: string;
    maxResults?: number;
    similarityThreshold?: number;
    filter?: {
        knowledgeIds?: string[];
        documentIds?: string[];
        [key: string]: any;
    };
}

export interface IVectorSearchResponse {
    results: ISearchResult[];
    totalResults: number;
}

/**
 * LangChain-based vector store service that provides document storage and similarity search
 * Now uses RWSVectorStore for unified vector storage like the LangChain tutorial
 */
@Injectable()
export class LangChainVectorStoreService {
    private vectorStore: RWSVectorStore;
    private documents: Map<string, Document> = new Map();
    private config: IVectorStoreConfig;
    private isInitialized = false;
    private documentCount = 0;
    private embeddingService: LangChainEmbeddingService;

    constructor() {    
        // Empty constructor for NestJS dependency injection
    }

    /**
     * Initialize the service with configuration
     */
    async initialize(embeddingService: LangChainEmbeddingService, config: IVectorStoreConfig): Promise<void> {
        if (this.isInitialized) {
            return;
        }
        
        this.embeddingService = embeddingService;
        this.config = {
            type: 'memory',
            maxResults: 10,
            similarityThreshold: 0.1,  // Use lower threshold like we configured
            ...config,
        };

        const embeddings = this.embeddingService.getEmbeddingsInstance();
        
        // Initialize with empty documents first, then create RWSVectorStore
        const initialDocs: VectorDocType = [];
        this.vectorStore = await new RWSVectorStore(
            initialDocs, 
            embeddings, 
            { 
                type: this.config.type,
                persistPath: this.config.persistPath 
            }
        ).init();
        
        console.log(`Created new ${this.config.type} vector store using RWSVectorStore`);
        this.isInitialized = true;
    }

    /**
     * Add documents to the vector store
     */
    async addDocuments(documents: Document[]): Promise<string[]> {
        this.ensureInitialized();

        try {
            const ids = await this.vectorStore.addDocuments(documents);
            const docIds: string[] = [];
            
            // Store documents in our map for retrieval
            documents.forEach((doc, index) => {
                const id = `doc_${Date.now()}_${index}`;
                this.documents.set(id, doc);
                docIds.push(id);
            });

            this.documentCount += documents.length;
            return docIds;
        } catch (error) {
            console.error('Failed to add documents to vector store:', error);
            throw error;
        }
    }

    /**
     * Index a single document (split into chunks if needed)
     */
    async indexDocument(
        content: string,
        documentId: string,
        metadata: Record<string, any> = {}
    ): Promise<{ success: boolean; chunkIds: string[]; error?: string }> {
        try {
            this.ensureInitialized();

            // Remove existing chunks for this document
            await this.deleteDocument(documentId);

            // Create document chunks
            const documents = await this.embeddingService.createDocuments(content, {
                documentId,
                ...metadata,
            });

            // Add to vector store
            const chunkIds = await this.addDocuments(documents);

            return {
                success: true,
                chunkIds,
            };
        } catch (error: any) {
            console.error(`Failed to index document ${documentId}:`, error);
            return {
                success: false,
                chunkIds: [],
                error: error?.message || 'Unknown error',
            };
        }
    }

    /**
     * Search for similar documents using RWSVectorStore like the LangChain tutorial
     */
    async searchSimilar(request: IVectorSearchRequest): Promise<IVectorSearchResponse> {
        this.ensureInitialized();

        try {
            const {
                query,
                maxResults = this.config.maxResults,
                similarityThreshold = this.config.similarityThreshold,
                filter,
            } = request;

            // Use RWSVectorStore's similaritySearchWithScore method like the tutorial
            const searchResults = await this.vectorStore.similaritySearchWithScore(
                query,
                maxResults || 10
            );

            // Filter by similarity threshold and metadata filters
            let filteredResults = searchResults
                .filter(([_doc, score]: [any, any]) => score >= (similarityThreshold || 0.1));

            // Apply knowledge/document ID filters if provided
            if (filter) {
                filteredResults = filteredResults.filter(([doc, _score]: [any, any]) => {
                    // Check knowledge IDs
                    if (filter.knowledgeIds && filter.knowledgeIds.length > 0) {
                        const docKnowledgeId = doc.metadata?.knowledgeId;
                        if (!docKnowledgeId || !filter.knowledgeIds.includes(docKnowledgeId)) {
                            return false;
                        }
                    }
                    
                    // Check document IDs
                    if (filter.documentIds && filter.documentIds.length > 0) {
                        const docId = doc.metadata?.documentId;
                        if (!docId || !filter.documentIds.includes(docId)) {
                            return false;
                        }
                    }
                    
                    return true;
                });
            }

            // Format results to match our interface
            const results: ISearchResult[] = filteredResults
                .map(([doc, score]: [any, any], index: number) => ({
                    content: doc.pageContent,
                    score,
                    metadata: doc.metadata,
                    chunkId: doc.metadata?.id || `chunk_${index}`,
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
     * Delete a document and all its chunks
     */
    async deleteDocument(documentId: string): Promise<boolean> {
        try {
            await this.ensureInitialized();

            // Find all chunks for this document
            const docsToDelete: string[] = [];
            for (const [id, doc] of this.documents.entries()) {
                if (doc.metadata?.documentId === documentId) {
                    docsToDelete.push(id);
                }
            }

            if (docsToDelete.length > 0) {
                // Use RWSVectorStore's deleteDocuments method
                await this.vectorStore.deleteDocuments(docsToDelete);

                // Remove from our document map
                docsToDelete.forEach(id => this.documents.delete(id));

                if (this.config.autoSave) {
                    await this.save();
                }
            }

            return true;
        } catch (error) {
            console.error(`Failed to delete document ${documentId}:`, error);
            return false;
        }
    }

    /**
     * Get documents by filter
     */
    async getDocuments(filter?: Record<string, any>): Promise<Document[]> {
        const docs: Document[] = [];
        
        for (const doc of this.documents.values()) {
            if (!filter) {
                docs.push(doc);
                continue;
            }

            let matches = true;
            for (const [key, value] of Object.entries(filter)) {
                if (doc.metadata?.[key] !== value) {
                    matches = false;
                    break;
                }
            }

            if (matches) {
                docs.push(doc);
            }
        }

        return docs;
    }

    /**
     * Save the vector store to disk (not needed for memory store)
     */
    async save(): Promise<boolean> {
        return true; // Memory store doesn't need saving
    }

    /**
     * Get statistics about the vector store
     */
    getStats(): { totalChunks: number; totalDocuments: number } {
        const documentIds = new Set<string>();
        
        for (const doc of this.documents.values()) {
            if (doc.metadata?.documentId) {
                documentIds.add(doc.metadata.documentId);
            }
        }

        return {
            totalChunks: this.documentCount,
            totalDocuments: documentIds.size,
        };
    }

    /**
     * Clear all documents from the vector store
     */
    async clear(): Promise<boolean> {
        try {
            await this.ensureInitialized();

            // Recreate RWSVectorStore with empty documents
            const embeddings = this.embeddingService.getEmbeddingsInstance();
            const initialDocs: VectorDocType = [];
            this.vectorStore = await new RWSVectorStore(
                initialDocs, 
                embeddings, 
                { 
                    type: this.config.type,
                    persistPath: this.config.persistPath 
                }
            ).init();

            // Clear our document map
            this.documents.clear();
            this.documentCount = 0;

            return true;
        } catch (error) {
            console.error('Failed to clear vector store:', error);
            return false;
        }
    }

    /**
     * Ensure the vector store is initialized
     */
    private ensureInitialized(): void {
        if (!this.isInitialized) {
            throw new Error('LangChainVectorStoreService not initialized. Call initialize() first.');
        }
    }

    /**
     * Get the underlying RWSVectorStore instance
     */
    getVectorStore(): RWSVectorStore {
        return this.vectorStore;
    }

    /**
     * Get the underlying LangChain vector store (memory or faiss)
     */
    getLangChainVectorStore() {
        return this.vectorStore.getVectorStore();
    }

    /**
     * Update configuration
     */
    updateConfig(newConfig: Partial<IVectorStoreConfig>): void {
        this.config = { ...this.config, ...newConfig };
    }
}
