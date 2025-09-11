import { FaissStore } from '@langchain/community/vectorstores/faiss';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';
import { EmbeddingsInterface } from '@langchain/core/embeddings';
import { Document } from '@langchain/core/documents';

type VectorDocType = Document<Record<string, any>>[];

export interface IVectorStoreConfig {
    type: 'faiss' | 'memory';
    persistPath?: string;
}

export default class RWSVectorStore 
{
    private vectorStore: FaissStore | MemoryVectorStore;
    private docs: VectorDocType;
    private embeddings: EmbeddingsInterface;
    private config: IVectorStoreConfig;

    constructor(docs: VectorDocType, embeddings: EmbeddingsInterface, config: IVectorStoreConfig = { type: 'memory' }){
        this.docs = docs;
        this.embeddings = embeddings;
        this.config = config;
    }

    async init(): Promise<RWSVectorStore>
    {
        if (this.config.type === 'faiss') {
            this.vectorStore = await FaissStore.fromDocuments(this.docs, this.embeddings);
        } else {
            this.vectorStore = await MemoryVectorStore.fromDocuments(this.docs, this.embeddings);
        }

        return this;
    }

    getVectorStore(): FaissStore | MemoryVectorStore
    {
        return this.vectorStore;
    }

    getFaiss(): FaissStore
    {
        if (this.vectorStore instanceof FaissStore) {
            return this.vectorStore;
        }
        throw new Error('Vector store is not a FAISS instance');
    }

    getMemoryStore(): MemoryVectorStore
    {
        if (this.vectorStore instanceof MemoryVectorStore) {
            return this.vectorStore;
        }
        throw new Error('Vector store is not a Memory instance');
    }

    getDocs()
    {
        return this.docs;
    }

    /**
     * Add more documents to the vector store
     */
    async addDocuments(newDocs: VectorDocType): Promise<void> {
        await this.vectorStore.addDocuments(newDocs);
        this.docs.push(...newDocs);
    }

    /**
     * Search for similar documents
     */
    async similaritySearchWithScore(query: string, k: number = 4): Promise<[Document, number][]> {
        return await this.vectorStore.similaritySearchWithScore(query, k);
    }

    /**
     * Search for similar documents using vector
     */
    async similaritySearchVectorWithScore(embedding: number[], k: number = 4): Promise<[Document, number][]> {
        return await this.vectorStore.similaritySearchVectorWithScore(embedding, k);
    }

    /**
     * Delete documents (if supported)
     */
    async deleteDocuments(ids: string[]): Promise<void> {
        if ('delete' in this.vectorStore) {
            await (this.vectorStore as any).delete({ ids });
        }
    }

    /**
     * Save the vector store (FAISS only)
     */
    async save(path?: string): Promise<void> {
        if (this.vectorStore instanceof FaissStore) {
            await this.vectorStore.save(path || this.config.persistPath || './vector_store');
        }
    }

    /**
     * Load a vector store from disk (FAISS only)
     */
    static async load(path: string, embeddings: EmbeddingsInterface): Promise<RWSVectorStore> {
        const faissStore = await FaissStore.load(path, embeddings);
        const vectorStore = new RWSVectorStore([], embeddings, { type: 'faiss', persistPath: path });
        vectorStore.vectorStore = faissStore;
        return vectorStore;
    }
}

export {
    VectorDocType
};